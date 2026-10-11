"""Use scene-specific right-arm keyposes, then servo the held tube to the rack."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class Controller:
    pick_actions: torch.Tensor | None = None
    step: int = 0
    initialized: bool = False
    initial_cube_z: float = 0.0
    ik_failures: int = 0


def get_model(usr_args):
    return Controller()


def _current_action(base) -> torch.Tensor:
    qpos = base.robot.get_qpos()
    ids = torch.as_tensor(base.active_joint_ids, dtype=torch.long, device=qpos.device)
    return qpos.index_select(-1, ids).clone()


def _interpolate(start: torch.Tensor, end: torch.Tensor, count: int) -> list[torch.Tensor]:
    rows = []
    for index in range(1, count + 1):
        alpha = index / count
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        rows.append(start * (1.0 - alpha) + end * alpha)
    return rows


def _initialize(base, model: Controller) -> None:
    # A failed full graph is still useful: node generation is exhaustive and
    # leaves the independent right-arm grasp/lift affordances in this mapping.
    full = base.create_demo_action_list()
    data = base.affordance_datas
    required = (
        "right_arm_cube_pre1_qpos",
        "right_arm_cube_grasp_qpos",
        "right_arm_cube_up1_qpos",
    )
    missing = [key for key in required if key not in data]
    if missing:
        print(f"[DIRECT RIGHT V1] missing right keyposes={missing}", flush=True)
        model.pick_actions = None
        return

    action0 = _current_action(base)
    right0 = base.robot.get_qpos(name="right_arm").clone()
    pre = torch.as_tensor(data[required[0]], dtype=torch.float32, device=base.device).reshape(1, 6)
    grasp = torch.as_tensor(data[required[1]], dtype=torch.float32, device=base.device).reshape(1, 6)
    lift = torch.as_tensor(data[required[2]], dtype=torch.float32, device=base.device).reshape(1, 6)
    rows = []

    def append_arm(values, right_open: bool):
        action = action0.clone()
        action[:, 7:13] = values
        action[:, 6] = 0.05
        action[:, 13] = 0.05 if right_open else 0.0
        rows.append(action)

    for values in _interpolate(right0, pre, 55):
        append_arm(values, True)
    for values in _interpolate(pre, grasp, 40):
        append_arm(values, True)
    for _ in range(25):
        append_arm(grasp, False)
    for values in _interpolate(grasp, lift, 45):
        append_arm(values, False)
    for _ in range(10):
        append_arm(lift, False)
    model.pick_actions = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    print(
        f"[DIRECT RIGHT V1] pick frames={len(rows)} full_graph={full is not None}",
        flush=True,
    )


def _rotation_step(cube_rotation: torch.Tensor, max_angle: float = 0.025) -> torch.Tensor:
    z_axis = cube_rotation[:, :, 2]
    world_z = torch.zeros_like(z_axis)
    world_z[:, 2] = 1.0
    cross = torch.linalg.cross(z_axis, world_z)
    sine = cross.norm(dim=-1, keepdim=True)
    cosine = (z_axis * world_z).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    angle = torch.atan2(sine, cosine).clamp(max=max_angle)
    axis = cross / sine.clamp_min(1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = torch.cos(angle[:, 0]), torch.sin(angle[:, 0])
    one = 1.0 - c
    result = torch.zeros((cube_rotation.shape[0], 3, 3), device=cube_rotation.device)
    result[:, 0, 0] = c + x * x * one
    result[:, 0, 1] = x * y * one - z * s
    result[:, 0, 2] = x * z * one + y * s
    result[:, 1, 0] = y * x * one + z * s
    result[:, 1, 1] = c + y * y * one
    result[:, 1, 2] = y * z * one - x * s
    result[:, 2, 0] = z * x * one - y * s
    result[:, 2, 1] = z * y * one + x * s
    result[:, 2, 2] = c + z * z * one
    near = (sine[:, 0] < 1e-7)
    result[near] = torch.eye(3, device=cube_rotation.device)
    return result


def _ik_ok(result) -> bool:
    if result is None:
        return False
    ok, qpos = result
    return bool(torch.as_tensor(ok).all().item()) and bool(torch.isfinite(qpos).all().item())


def _servo(base, model: Controller, cube_pose: torch.Tensor, rack_pose: torch.Tensor):
    robot = base.robot
    action = _current_action(base)
    right_qpos = robot.get_qpos(name="right_arm")
    eef_pose = robot.compute_fk(qpos=right_qpos, name="right_arm", to_matrix=True)
    cube_pos, rack_pos = cube_pose[:, :3, 3], rack_pose[:, :3, 3]
    xy = torch.norm(cube_pos[:, :2] - rack_pos[:, :2], dim=-1)

    target_pos = cube_pos.clone()
    if bool((cube_pos[:, 2] < 1.025).any().item()):
        target_pos[:, 2] = 1.035
        phase = "raise"
    elif bool((xy > 0.012).any().item()):
        target_pos[:, :2] = rack_pos[:, :2]
        target_pos[:, 2] = 1.035
        phase = "translate"
    else:
        target_pos[:, :2] = rack_pos[:, :2]
        target_pos[:, 2] = rack_pos[:, 2] + 0.055
        phase = "lower"

    translation = target_pos - cube_pos
    distance = translation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    translation *= torch.minimum(torch.ones_like(distance), 0.0035 / distance)
    rotation_step = _rotation_step(cube_pose[:, :3, :3])

    desired_cube = cube_pose.clone()
    desired_cube[:, :3, :3] = rotation_step @ cube_pose[:, :3, :3]
    desired_cube[:, :3, 3] = cube_pos + translation
    object_delta = desired_cube @ torch.linalg.inv(cube_pose)
    target_eef = object_delta @ eef_pose
    result = robot.compute_ik(pose=target_eef, joint_seed=right_qpos, name="right_arm")
    if not _ik_ok(result):
        # Rotation can be the limiting dimension near the workspace boundary.
        target_eef = eef_pose.clone()
        target_eef[:, :3, 3] += translation * 0.5
        result = robot.compute_ik(pose=target_eef, joint_seed=right_qpos, name="right_arm")
    if _ik_ok(result):
        action[:, 7:13] = right_qpos + (result[1] - right_qpos).clamp(-0.045, 0.045)
    else:
        model.ik_failures += 1
    action[:, 6] = 0.05
    action[:, 13] = 0.0
    return action, phase


def eval(env, model: Controller, obs):
    base = env.unwrapped
    if not model.initialized:
        cube = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
        model.initial_cube_z = float(cube[0, 2, 3].item())
        _initialize(base, model)
        model.initialized = True

    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if model.pick_actions is not None and model.step < len(model.pick_actions):
        action = model.pick_actions[model.step].to(device=base.device)
        phase = "pick"
    else:
        action, phase = _servo(base, model, cube_pose, rack_pose)

    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 15 == 0 or success:
        cube_pos, rack_pos = cube_pose[:, :3, 3], rack_pose[:, :3, 3]
        xy = torch.norm(cube_pos[:, :2] - rack_pos[:, :2], dim=-1)
        cube_z = cube_pose[:, :3, 2]
        angle = torch.rad2deg(torch.arccos(cube_z[:, 2].clamp(-1.0, 1.0)))
        print(
            f"[DIRECT RIGHT V1] step={model.step} phase={phase} xy={float(xy.item()):.5f}"
            f" z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f}"
            f" ik_fail={model.ik_failures} success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(
        torch.as_tensor(truncated).any().item()
    )
    return next_obs, info, done


def reset_model(model: Controller):
    model.pick_actions = None
    model.step = 0
    model.initialized = False
    model.initial_cube_z = 0.0
    model.ik_failures = 0
