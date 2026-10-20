"""Align over the slot at a safe height before beginning vertical insertion."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _guided, _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v9.deploy_policy import _slow_initialize
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _current_action, _ik_ok, _rotation_step


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


def get_model(usr_args):
    return Controller()


def _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle):
    robot = base.robot
    action = _current_action(base)
    qpos = robot.get_qpos(name="left_arm")
    eef = robot.compute_fk(qpos, name="left_arm", to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    target = _slot_position(rack_pose)
    if model.place_stage == "align":
        target[:, 2] = 1.045
        if float(slot_xy.item()) < 0.006 and float(angle.item()) < 2.5 and float(cube_pos[0, 2].item()) > 1.015:
            model.align_stable += 1
        else:
            model.align_stable = 0
        if model.align_stable >= 5:
            model.place_stage = "lower"
    else:
        target[:, 2] = rack_pose[:, 2, 3] + 0.052
    translation = target - cube_pos
    norm = translation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    translation *= torch.minimum(torch.ones_like(norm), 0.0025 / norm)
    desired = cube_pose.clone()
    desired[:, :3, 3] += translation
    desired[:, :3, :3] = _rotation_step(cube_pose[:, :3, :3], 0.015) @ cube_pose[:, :3, :3]
    target_eef = (desired @ torch.linalg.inv(cube_pose)) @ eef
    result = robot.compute_ik(pose=target_eef, joint_seed=qpos, name="left_arm")
    if not _ik_ok(result):
        target_eef = eef.clone()
        target_eef[:, :3, 3] += translation * 0.35
        result = robot.compute_ik(pose=target_eef, joint_seed=qpos, name="left_arm")
    if _ik_ok(result):
        action[:, :6] = qpos + (result[1] - qpos).clamp(-0.035, 0.035)
    else:
        model.ik_failures += 1
    action[:, 6] = 0.0
    action[:, 13] = 0.05
    return action


def eval(env, model, obs):
    base = env.unwrapped
    if not model.initialized:
        _slow_initialize(base, model)
        model.initialized = True
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    opening = 0.0
    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(base.device)
        phase = "slow_staged"
    else:
        if (
            not model.release_active and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0
            and float(cube_pos[0, 2].item()) < 1.010
            and (model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 1.002)
        ):
            model.release_active = True
            print(
                f"[BIMANUAL V10] slot release step={model.step} slot_xy={float(slot_xy.item()):.5f}"
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
        print(
            f"[BIMANUAL V10] step={model.step} phase={phase} slot_xy={float(slot_xy.item()):.5f}"
            f" official_xy={float(official_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f}"
            f" angle={float(angle.item()):.2f} ik_fail={model.ik_failures}"
            f" opening={opening:.3f} success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
