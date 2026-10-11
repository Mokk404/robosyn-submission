"""Retarget verified arm trajectories by the live-to-reference tube transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v14.deploy_policy import _guided_wide
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _ik_ok
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import _dataset


@dataclass
class Controller:
    trajectory: torch.Tensor | None = None
    ref_cube: torch.Tensor | None = None
    scene_delta: torch.Tensor | None = None
    cutoff: int = 0
    left_start: int = 0
    step: int = 0
    initialized: bool = False
    nearest_episode: int = -1
    nearest_cost: float = 0.0
    retarget_ik_failures: int = 0
    ik_failures: int = 0
    release_active: bool = False
    release_steps: int = 0
    place_stage: str = "align"
    align_stable: int = 0


def get_model(usr_args):
    return Controller()


def _axis_rotation(source, target):
    source = source / np.linalg.norm(source); target = target / np.linalg.norm(target)
    cross = np.cross(source, target); sine = np.linalg.norm(cross); cosine = np.clip(np.dot(source, target), -1.0, 1.0)
    if sine < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = cross / sine
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32)
    angle = np.arctan2(sine, cosine)
    return (np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)).astype(np.float32)


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy(); axis = cube[:3, 2]
    best = None
    for episode, actions, cubes, cutoff in _dataset():
        position = np.linalg.norm(cube[:3, 3] - cubes[0, :3, 3])
        axis_angle = float(np.arccos(np.clip(np.dot(axis, cubes[0, :3, 2]), -1.0, 1.0)))
        cost = position + 0.06 * axis_angle
        if best is None or cost < best[0]:
            best = (cost, episode, actions, cubes, cutoff, position, axis_angle)
    cost, episode, actions, cubes, cutoff, position, axis_angle = best
    rotation = _axis_rotation(cubes[0, :3, 2], axis)
    delta = np.eye(4, dtype=np.float32); delta[:3, :3] = rotation
    delta[:3, 3] = cube[:3, 3] - rotation @ cubes[0, :3, 3]
    left_motion = np.linalg.norm(actions[:, :6] - actions[0:1, :6], axis=1)
    candidates = np.flatnonzero((left_motion > 0.03) & (np.arange(len(actions)) > 100))
    left_start = int(candidates[0]) if len(candidates) else cutoff
    actions = actions.copy(); actions[:, 6] *= 0.05; actions[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(actions, dtype=torch.float32, device=base.device).reshape(-1, 1, 14)
    model.ref_cube = torch.as_tensor(cubes, dtype=torch.float32, device=base.device)
    model.scene_delta = torch.as_tensor(delta, dtype=torch.float32, device=base.device).reshape(1, 4, 4)
    model.cutoff = cutoff; model.left_start = left_start; model.nearest_episode = episode; model.nearest_cost = cost
    print(f"[KNN V21] selected episode={episode} cost={cost:.6f} position={position:.6f} axis_angle={np.degrees(axis_angle):.2f} left_start={left_start} cutoff={cutoff}", flush=True)


def _retarget_arm(base, action, model, arm, action_slice):
    robot = base.robot
    reference_q = action[:, action_slice]
    reference_eef = robot.compute_fk(reference_q, name=arm, to_matrix=True)
    desired_eef = model.scene_delta @ reference_eef
    current_q = robot.get_qpos(name=arm)
    result = robot.compute_ik(pose=desired_eef, joint_seed=current_q, name=arm)
    if not _ik_ok(result):
        result = robot.compute_ik(pose=desired_eef, joint_seed=reference_q, name=arm)
    if _ik_ok(result):
        action[:, action_slice] = current_q + (result[1] - current_q).clamp(-0.060, 0.060)
    else:
        model.retarget_ik_failures += 1


def eval(env, model, obs):
    base = env.unwrapped
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if not model.initialized:
        _initialize(base, model, cube_pose); model.initialized = True
    cube_pos = cube_pose[:, :3, 3]; slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0))); opening = 0.0
    if model.step < model.cutoff:
        action = model.trajectory[model.step].clone()
        _retarget_arm(base, action, model, "right_arm", slice(7, 13))
        if model.step >= model.left_start:
            _retarget_arm(base, action, model, "left_arm", slice(0, 6))
        phase = "retarget_handoff"
    else:
        if not model.release_active and model.place_stage == "lower" and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0 and float(cube_pos[0,2].item()) < 1.018 and (model.ik_failures >= 8 or float(cube_pos[0,2].item()) < 1.002):
            model.release_active = True
        if model.release_active:
            action, opening = _guided_wide(base, model, cube_pose, rack_pose, slot_xy, angle); phase = "slot_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle); phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action); model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 20 == 0 or model.step == model.cutoff or success:
        ref_index = min(model.step - 1, len(model.ref_cube) - 1)
        desired_ref = model.scene_delta @ model.ref_cube[ref_index].reshape(1,4,4)
        ref_error = torch.norm(cube_pos[0] - desired_ref[0, :3, 3])
        print(f"[KNN V21] step={model.step} phase={phase} ref_err={float(ref_error.item()):.5f} slot_xy={float(slot_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f} retarget_ik_fail={model.retarget_ik_failures} place_ik_fail={model.ik_failures} success={success}", flush=True)
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller(); model.__dict__.clear(); model.__dict__.update(fresh.__dict__)
