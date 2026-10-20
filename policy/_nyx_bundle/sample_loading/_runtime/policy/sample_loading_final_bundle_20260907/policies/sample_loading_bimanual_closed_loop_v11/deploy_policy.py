"""Adapt the left-hand takeover to the measured tube pose before V10 placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v1.deploy_policy import _smooth
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _guided, _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v9.deploy_policy import _slow_initialize
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _current_action, _ik_ok


@dataclass
class Controller:
    actions: torch.Tensor | None = None
    step: int = 0
    initialized: bool = False
    ik_failures: int = 0
    release_active: bool = False
    release_steps: int = 0
    place_stage: str = "align"
    align_stable: int = 0
    handoff_rebuilt: bool = False
    adaptive_handoff: bool = False


def get_model(usr_args):
    return Controller()


def _pose(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device).reshape(1, 4, 4)


def _append_segment(base, rows, left, right, left_target, right_target, count, left_open, right_open):
    left_rows = _smooth(left, left_target, count)
    right_rows = _smooth(right, right_target, count)
    for left_value, right_value in zip(left_rows, right_rows):
        action = _current_action(base)
        action[:, :6] = left_value
        action[:, 7:13] = right_value
        action[:, 6] = 0.05 if left_open else 0.0
        action[:, 13] = 0.05 if right_open else 0.0
        rows.append(action)
    return left_target, right_target


def _rebuild_handoff(base, model, cube_pose):
    """Replace the nominal takeover tail using the currently measured held-object pose."""
    data = base.affordance_datas
    required = (
        "cube_pose", "right_arm_cube_grasp_pose", "right_arm_cube_up2_pose",
        "left_arm_takeover_pose", "left_arm_takeover_pre_pose",
        "left_arm_cube_place_pre_waypoint1_qpos",
        "left_arm_cube_place_pre_waypoint2_qpos", "left_arm_cube_place_pre_qpos",
        "right_arm_back_qpos",
    )
    missing = [key for key in required if key not in data]
    if missing:
        print(f"[BIMANUAL V11] adaptive handoff unavailable missing={missing}", flush=True)
        return

    device = base.device
    initial_cube = _pose(data["cube_pose"], device)
    right_grasp = _pose(data["right_arm_cube_grasp_pose"], device)
    right_up2 = _pose(data["right_arm_cube_up2_pose"], device)
    nominal_take = _pose(data["left_arm_takeover_pose"], device)
    nominal_pre = _pose(data["left_arm_takeover_pre_pose"], device)

    ideal_cube_up2 = right_up2 @ torch.linalg.inv(right_grasp) @ initial_cube
    desired_take = cube_pose @ torch.linalg.inv(ideal_cube_up2) @ nominal_take
    desired_pre = cube_pose @ torch.linalg.inv(ideal_cube_up2) @ nominal_pre

    left = base.robot.get_qpos(name="left_arm").clone()
    right = base.robot.get_qpos(name="right_arm").clone()
    pre_ret = base.robot.compute_ik(pose=desired_pre, joint_seed=left, name="left_arm")
    if not _ik_ok(pre_ret):
        print("[BIMANUAL V11] adaptive pre IK failed; retaining nominal tail", flush=True)
        return
    pre_q = pre_ret[1]
    take_ret = base.robot.compute_ik(pose=desired_take, joint_seed=pre_q, name="left_arm")
    if not _ik_ok(take_ret):
        print("[BIMANUAL V11] adaptive takeover IK failed; retaining nominal tail", flush=True)
        return
    take_q = take_ret[1]

    wp1 = torch.as_tensor(data["left_arm_cube_place_pre_waypoint1_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    wp2 = torch.as_tensor(data["left_arm_cube_place_pre_waypoint2_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    place_pre = torch.as_tensor(data["left_arm_cube_place_pre_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    right_back = torch.as_tensor(data["right_arm_back_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    rows = []

    left, right = _append_segment(base, rows, left, right, pre_q, right, 35, True, False)
    left, right = _append_segment(base, rows, left, right, take_q, right, 25, True, False)
    left, right = _append_segment(base, rows, left, right, left, right, 30, False, False)

    # Release the right grasp progressively so contact transfers to the left fingers.
    for opening, count in ((0.002, 8), (0.006, 8), (0.012, 6), (0.025, 4), (0.05, 4)):
        for _ in range(count):
            action = _current_action(base)
            action[:, :6] = left
            action[:, 7:13] = right
            action[:, 6] = 0.0
            action[:, 13] = opening
            rows.append(action)

    left, right = _append_segment(base, rows, left, right, wp1, right_back, 45, False, True)
    left, right = _append_segment(base, rows, left, right, wp2, right, 15, False, True)
    left, right = _append_segment(base, rows, left, right, place_pre, right, 15, False, True)
    left, right = _append_segment(base, rows, left, right, left, right, 10, False, True)

    prefix = model.actions[: model.step] if model.actions is not None else torch.empty((0, 1, 14), device=device)
    tail = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    model.actions = torch.cat([prefix.to(device), tail], dim=0)
    model.adaptive_handoff = True
    delta_pos = torch.norm(desired_take[:, :3, 3] - nominal_take[:, :3, 3], dim=-1)
    print(
        f"[BIMANUAL V11] rebuilt handoff at step={model.step} tail={len(tail)}"
        f" takeover_translation_correction={float(delta_pos.item()):.5f}", flush=True
    )


def eval(env, model, obs):
    base = env.unwrapped
    if not model.initialized:
        _slow_initialize(base, model)
        model.initialized = True

    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if model.actions is not None and model.step == 255 and not model.handoff_rebuilt:
        _rebuild_handoff(base, model, cube_pose)
        model.handoff_rebuilt = True

    cube_pos = cube_pose[:, :3, 3]
    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    opening = 0.0

    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(base.device)
        phase = "adaptive_staged" if model.handoff_rebuilt else "slow_pick"
    else:
        if (
            not model.release_active and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0
            and float(cube_pos[0, 2].item()) < 1.010
            and (model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 1.002)
        ):
            model.release_active = True
            print(
                f"[BIMANUAL V11] slot release step={model.step} slot_xy={float(slot_xy.item()):.5f}"
                f" z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f}", flush=True
            )
        if model.release_active:
            action, opening = _guided(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = "slot_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = f"place_{model.place_stage}"

    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 10 == 0 or success:
        left_eef = base.robot.compute_fk(base.robot.get_qpos(name="left_arm"), name="left_arm", to_matrix=True)
        right_eef = base.robot.compute_fk(base.robot.get_qpos(name="right_arm"), name="right_arm", to_matrix=True)
        left_dist = torch.norm(cube_pos - left_eef[:, :3, 3], dim=-1)
        right_dist = torch.norm(cube_pos - right_eef[:, :3, 3], dim=-1)
        print(
            f"[BIMANUAL V11] step={model.step} phase={phase} slot_xy={float(slot_xy.item()):.5f}"
            f" official_xy={float(official_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f}"
            f" angle={float(angle.item()):.2f} left_d={float(left_dist.item()):.4f}"
            f" right_d={float(right_dist.item()):.4f} ik_fail={model.ik_failures}"
            f" opening={opening:.3f} success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
