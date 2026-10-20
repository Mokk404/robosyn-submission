"""Use nominal handoff normally and measured-pose handoff only for outliers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v1.deploy_policy import _smooth
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _guided, _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v9.deploy_policy import _slow_initialize
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v11.deploy_policy import _append_segment, _pose
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
    handoff_trigger: int = 255
    handoff_checked: bool = False
    adaptive_handoff: bool = False
    fallback_pick: bool = False


def get_model(usr_args):
    return Controller()


def _fallback_pick(base, model):
    """Create a pick ending at the last valid right-arm lift pose."""
    data = base.affordance_datas
    required = ("right_arm_cube_pre1_qpos", "right_arm_cube_grasp_qpos", "right_arm_cube_up1_qpos")
    missing = [key for key in required if key not in data]
    if missing:
        print(f"[BIMANUAL V13] fallback pick unavailable missing={missing}", flush=True)
        return
    device = base.device
    left = base.robot.get_qpos(name="left_arm").clone()
    right = base.robot.get_qpos(name="right_arm").clone()
    pre = torch.as_tensor(data[required[0]], dtype=torch.float32, device=device).reshape(1, 6)
    grasp = torch.as_tensor(data[required[1]], dtype=torch.float32, device=device).reshape(1, 6)
    up1 = torch.as_tensor(data[required[2]], dtype=torch.float32, device=device).reshape(1, 6)
    rows = []
    left, right = _append_segment(base, rows, left, right, left, pre, 50, True, True)
    left, right = _append_segment(base, rows, left, right, left, grasp, 70, True, True)
    left, right = _append_segment(base, rows, left, right, left, right, 10, True, True)
    left, right = _append_segment(base, rows, left, right, left, right, 40, True, False)
    left, right = _append_segment(base, rows, left, right, left, up1, 50, True, False)
    model.actions = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    model.handoff_trigger = len(rows)
    model.fallback_pick = True
    print(f"[BIMANUAL V13] fallback pick frames={len(rows)}", flush=True)


def _handoff_geometry(base, cube_pose):
    data, device = base.affordance_datas, base.device
    required = ("cube_pose", "right_arm_cube_grasp_pose", "right_arm_cube_up2_pose", "left_arm_takeover_pose")
    if any(key not in data for key in required):
        return None
    initial_cube = _pose(data["cube_pose"], device)
    right_grasp = _pose(data["right_arm_cube_grasp_pose"], device)
    right_up2 = _pose(data["right_arm_cube_up2_pose"], device)
    nominal_take = _pose(data["left_arm_takeover_pose"], device)
    ideal_cube_up2 = right_up2 @ torch.linalg.inv(right_grasp) @ initial_cube
    desired_take = cube_pose @ torch.linalg.inv(ideal_cube_up2) @ nominal_take
    correction = torch.norm(desired_take[:, :3, 3] - nominal_take[:, :3, 3], dim=-1)
    return ideal_cube_up2, desired_take, correction


def _adaptive_tail(base, model, cube_pose):
    data, device = base.affordance_datas, base.device
    geometry = _handoff_geometry(base, cube_pose)
    required = (
        "left_arm_takeover_pose", "left_arm_takeover_pre_pose",
        "left_arm_cube_place_pre_waypoint1_qpos",
        "left_arm_cube_place_pre_waypoint2_qpos", "left_arm_cube_place_pre_qpos",
    )
    missing = [key for key in required if key not in data]
    if geometry is None or missing:
        print(f"[BIMANUAL V13] adaptive tail unavailable missing={missing}", flush=True)
        return False
    ideal_cube, desired_take, correction = geometry
    nominal_pre = _pose(data["left_arm_takeover_pre_pose"], device)
    desired_pre = cube_pose @ torch.linalg.inv(ideal_cube) @ nominal_pre
    left = base.robot.get_qpos(name="left_arm").clone()
    right = base.robot.get_qpos(name="right_arm").clone()
    pre_ret = base.robot.compute_ik(pose=desired_pre, joint_seed=left, name="left_arm")
    if not _ik_ok(pre_ret):
        print("[BIMANUAL V13] adaptive pre IK failed", flush=True)
        return False
    pre_q = pre_ret[1]
    take_ret = base.robot.compute_ik(pose=desired_take, joint_seed=pre_q, name="left_arm")
    if not _ik_ok(take_ret):
        print("[BIMANUAL V13] adaptive takeover IK failed", flush=True)
        return False
    take_q = take_ret[1]
    wp1 = torch.as_tensor(data["left_arm_cube_place_pre_waypoint1_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    wp2 = torch.as_tensor(data["left_arm_cube_place_pre_waypoint2_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    place_pre = torch.as_tensor(data["left_arm_cube_place_pre_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    if "right_arm_back_qpos" in data:
        right_back = torch.as_tensor(data["right_arm_back_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    else:
        right_back = torch.as_tensor(data["right_arm_init_qpos"], dtype=torch.float32, device=device).reshape(1, 6)
    rows = []
    left, right = _append_segment(base, rows, left, right, pre_q, right, 35, True, False)
    left, right = _append_segment(base, rows, left, right, take_q, right, 25, True, False)
    left, right = _append_segment(base, rows, left, right, left, right, 30, False, False)
    for opening, count in ((0.002, 8), (0.006, 8), (0.012, 6), (0.025, 4), (0.05, 4)):
        for _ in range(count):
            action = _current_action(base)
            action[:, :6], action[:, 7:13] = left, right
            action[:, 6], action[:, 13] = 0.0, opening
            rows.append(action)
    left, right = _append_segment(base, rows, left, right, wp1, right_back, 45, False, True)
    left, right = _append_segment(base, rows, left, right, wp2, right, 15, False, True)
    left, right = _append_segment(base, rows, left, right, place_pre, right, 15, False, True)
    left, right = _append_segment(base, rows, left, right, left, right, 10, False, True)
    prefix = model.actions[: model.step].to(device)
    tail = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    model.actions = torch.cat([prefix, tail], dim=0)
    model.adaptive_handoff = True
    print(
        f"[BIMANUAL V13] adaptive tail step={model.step} frames={len(tail)}"
        f" correction={float(correction.item()):.5f}", flush=True
    )
    return True


def _initialize(base, model):
    _slow_initialize(base, model)
    if model.actions is None:
        _fallback_pick(base, model)


def eval(env, model, obs):
    base = env.unwrapped
    if not model.initialized:
        _initialize(base, model)
        model.initialized = True
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))

    if model.actions is not None and model.step == model.handoff_trigger and not model.handoff_checked:
        geometry = _handoff_geometry(base, cube_pose)
        correction = float(geometry[2].item()) if geometry is not None else 999.0
        needs_adaptive = model.fallback_pick or correction > 0.05 or float(angle.item()) > 15.0
        if needs_adaptive:
            _adaptive_tail(base, model, cube_pose)
        else:
            print(f"[BIMANUAL V13] nominal tail correction={correction:.5f} angle={float(angle.item()):.2f}", flush=True)
        model.handoff_checked = True

    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    opening = 0.0
    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(base.device)
        phase = "adaptive_staged" if model.adaptive_handoff else "nominal_staged"
    else:
        if (
            not model.release_active and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0
            and float(cube_pos[0, 2].item()) < 1.010
            and (model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 1.002)
        ):
            model.release_active = True
            print(f"[BIMANUAL V13] slot release step={model.step} z={float(cube_pos[0,2].item()):.5f}", flush=True)
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
        print(
            f"[BIMANUAL V13] step={model.step} phase={phase} slot_xy={float(slot_xy.item()):.5f}"
            f" official_xy={float(official_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f}"
            f" angle={float(angle.item()):.2f} ik_fail={model.ik_failures} opening={opening:.3f}"
            f" success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
