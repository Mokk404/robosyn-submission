"""Execute stable bimanual keypose timing and servo the left-held tube into the rack."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import (
    _current_action,
    _ik_ok,
    _rotation_step,
)


@dataclass
class Controller:
    actions: torch.Tensor | None = None
    step: int = 0
    initialized: bool = False
    ik_failures: int = 0


def get_model(usr_args):
    return Controller()


def _smooth(start: torch.Tensor, end: torch.Tensor, count: int):
    result = []
    for index in range(1, count + 1):
        alpha = index / count
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        result.append(start * (1.0 - alpha) + end * alpha)
    return result


def _initialize(base, model: Controller):
    full = base.create_demo_action_list()
    data = base.affordance_datas
    keys = (
        "right_arm_cube_pre1_qpos", "right_arm_cube_grasp_qpos",
        "right_arm_cube_up1_qpos", "right_arm_cube_up2_qpos", "right_arm_back_qpos",
        "left_arm_takeover_pre_qpos", "left_arm_takeover_qpos",
        "left_arm_cube_place_pre_waypoint1_qpos",
        "left_arm_cube_place_pre_waypoint2_qpos", "left_arm_cube_place_pre_qpos",
    )
    missing = [key for key in keys if key not in data]
    if missing:
        print(f"[BIMANUAL V1] missing keyposes={missing}", flush=True)
        return
    device = base.device
    values = {key: torch.as_tensor(data[key], dtype=torch.float32, device=device).reshape(1, 6) for key in keys}
    left = base.robot.get_qpos(name="left_arm").clone()
    right = base.robot.get_qpos(name="right_arm").clone()
    rows = []

    def segment(left_target, right_target, count, left_open, right_open):
        nonlocal left, right
        left_rows = _smooth(left, left_target, count)
        right_rows = _smooth(right, right_target, count)
        for left_value, right_value in zip(left_rows, right_rows):
            action = _current_action(base)
            action[:, :6] = left_value
            action[:, 7:13] = right_value
            action[:, 6] = 0.05 if left_open else 0.0
            action[:, 13] = 0.05 if right_open else 0.0
            rows.append(action)
        left, right = left_target, right_target

    # Right pick with generous settle windows.
    segment(left, values["right_arm_cube_pre1_qpos"], 40, True, True)
    segment(left, values["right_arm_cube_grasp_qpos"], 30, True, True)
    segment(left, right, 22, True, False)
    segment(left, values["right_arm_cube_up1_qpos"], 35, True, False)
    segment(left, values["right_arm_cube_up2_qpos"], 35, True, False)
    # Left approaches while the right arm holds the tube fixed.
    segment(values["left_arm_takeover_pre_qpos"], right, 45, True, False)
    segment(values["left_arm_takeover_qpos"], right, 22, True, False)
    segment(left, right, 30, False, False)
    segment(left, right, 25, False, True)
    # Clear the right arm while moving the left arm toward the first waypoint.
    segment(values["left_arm_cube_place_pre_waypoint1_qpos"], values["right_arm_back_qpos"], 50, False, True)
    segment(values["left_arm_cube_place_pre_waypoint2_qpos"], right, 18, False, True)
    segment(values["left_arm_cube_place_pre_qpos"], right, 18, False, True)
    segment(left, right, 12, False, True)
    model.actions = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    print(f"[BIMANUAL V1] staged frames={len(rows)} full_graph={full is not None}", flush=True)


def _servo(base, model: Controller, cube_pose, rack_pose):
    robot = base.robot
    action = _current_action(base)
    left_qpos = robot.get_qpos(name="left_arm")
    eef_pose = robot.compute_fk(qpos=left_qpos, name="left_arm", to_matrix=True)
    cube_pos, rack_pos = cube_pose[:, :3, 3], rack_pose[:, :3, 3]
    target = rack_pos.clone()
    target[:, 2] += 0.052
    translation = target - cube_pos
    distance = translation.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    translation *= torch.minimum(torch.ones_like(distance), 0.0025 / distance)
    rotation_step = _rotation_step(cube_pose[:, :3, :3], max_angle=0.015)
    desired = cube_pose.clone()
    desired[:, :3, :3] = rotation_step @ cube_pose[:, :3, :3]
    desired[:, :3, 3] = cube_pos + translation
    target_eef = (desired @ torch.linalg.inv(cube_pose)) @ eef_pose
    result = robot.compute_ik(pose=target_eef, joint_seed=left_qpos, name="left_arm")
    if not _ik_ok(result):
        target_eef = eef_pose.clone()
        target_eef[:, :3, 3] += translation * 0.35
        result = robot.compute_ik(pose=target_eef, joint_seed=left_qpos, name="left_arm")
    if _ik_ok(result):
        action[:, :6] = left_qpos + (result[1] - left_qpos).clamp(-0.035, 0.035)
    else:
        model.ik_failures += 1
    action[:, 6] = 0.0
    action[:, 13] = 0.05
    return action


def eval(env, model: Controller, obs):
    base = env.unwrapped
    if not model.initialized:
        _initialize(base, model)
        model.initialized = True
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(device=base.device)
        phase = "staged"
    else:
        action = _servo(base, model, cube_pose, rack_pose)
        phase = "left_servo"
    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 15 == 0 or success:
        cube_pos, rack_pos = cube_pose[:, :3, 3], rack_pose[:, :3, 3]
        xy = torch.norm(cube_pos[:, :2] - rack_pos[:, :2], dim=-1)
        angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
        left_eef = base.robot.compute_fk(base.robot.get_qpos(name="left_arm"), name="left_arm", to_matrix=True)
        right_eef = base.robot.compute_fk(base.robot.get_qpos(name="right_arm"), name="right_arm", to_matrix=True)
        ld = torch.norm(cube_pos - left_eef[:, :3, 3], dim=-1)
        rd = torch.norm(cube_pos - right_eef[:, :3, 3], dim=-1)
        print(
            f"[BIMANUAL V1] step={model.step} phase={phase} xy={float(xy.item()):.5f}"
            f" z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f}"
            f" left_d={float(ld.item()):.4f} right_d={float(rd.item()):.4f}"
            f" ik_fail={model.ik_failures} success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(
        torch.as_tensor(truncated).any().item()
    )
    return next_obs, info, done


def reset_model(model: Controller):
    model.actions = None
    model.step = 0
    model.initialized = False
    model.ik_failures = 0
