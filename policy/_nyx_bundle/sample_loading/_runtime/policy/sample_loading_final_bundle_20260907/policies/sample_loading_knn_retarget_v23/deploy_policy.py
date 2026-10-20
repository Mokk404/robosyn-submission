"""Verified-200 KNN ranked by geometry and sampled retarget IK feasibility."""

from __future__ import annotations

import numpy as np
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _ik_ok
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import _dataset
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v21.deploy_policy import Controller, _retarget_arm
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v22.deploy_policy import (
    _capped_axis_rotation,
    _supported_release,
)


def get_model(usr_args):
    return Controller()


def _left_start(actions, cutoff):
    motion = np.linalg.norm(actions[:, :6] - actions[0:1, :6], axis=1)
    candidates = np.flatnonzero((motion > 0.03) & (np.arange(len(actions)) > 100))
    return int(candidates[0]) if len(candidates) else cutoff


def _scene_delta(cube, reference_cube):
    rotation = _capped_axis_rotation(reference_cube[:3, 2], cube[:3, 2], 8.0)
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = rotation
    delta[:3, 3] = cube[:3, 3] - rotation @ reference_cube[:3, 3]
    return delta


def _feasibility(base, actions, cutoff, left_start, delta):
    robot = base.robot
    delta_t = torch.as_tensor(delta, dtype=torch.float32, device=base.device).reshape(1, 4, 4)
    first = min(90, max(cutoff - 1, 0))
    frames = np.unique(np.linspace(first, max(cutoff - 1, first), 12, dtype=np.int64))
    failures = 0
    calls = 0
    for frame in frames:
        for arm, arm_slice in (("right_arm", slice(7, 13)), ("left_arm", slice(0, 6))):
            if arm == "left_arm" and int(frame) < left_start:
                continue
            q = torch.as_tensor(
                actions[int(frame), arm_slice], dtype=torch.float32, device=base.device
            ).reshape(1, 6)
            reference_eef = robot.compute_fk(q, name=arm, to_matrix=True)
            result = robot.compute_ik(
                pose=delta_t @ reference_eef, joint_seed=q, name=arm
            )
            calls += 1
            if not _ik_ok(result):
                failures += 1
    return failures / max(calls, 1), failures, calls


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy()
    candidates = []
    for episode, actions, cubes, cutoff in _dataset():
        position = float(np.linalg.norm(cube[:3, 3] - cubes[0, :3, 3]))
        axis_angle = float(
            np.arccos(np.clip(np.dot(cube[:3, 2], cubes[0, :3, 2]), -1.0, 1.0))
        )
        geometry = position + 0.12 * axis_angle
        candidates.append((geometry, episode, actions, cubes, cutoff, position, axis_angle))
    candidates.sort(key=lambda item: item[0])
    ranked = []
    for geometry, episode, actions, cubes, cutoff, position, axis_angle in candidates[:16]:
        left_start = _left_start(actions, cutoff)
        delta = _scene_delta(cube, cubes[0])
        fraction, failures, calls = _feasibility(
            base, actions, cutoff, left_start, delta
        )
        score = geometry + 0.10 * fraction
        ranked.append(
            (
                score,
                geometry,
                episode,
                actions,
                cubes,
                cutoff,
                position,
                axis_angle,
                left_start,
                delta,
                failures,
                calls,
            )
        )
    ranked.sort(key=lambda item: item[0])
    (
        score,
        geometry,
        episode,
        actions,
        cubes,
        cutoff,
        position,
        axis_angle,
        left_start,
        delta,
        failures,
        calls,
    ) = ranked[0]
    actions = actions.copy()
    actions[:, 6] *= 0.05
    actions[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(
        actions, dtype=torch.float32, device=base.device
    ).reshape(-1, 1, 14)
    model.ref_cube = torch.as_tensor(cubes, dtype=torch.float32, device=base.device)
    model.scene_delta = torch.as_tensor(
        delta, dtype=torch.float32, device=base.device
    ).reshape(1, 4, 4)
    model.cutoff = cutoff
    model.left_start = left_start
    model.nearest_episode = episode
    model.nearest_cost = score
    print(
        f"[KNN V23] selected episode={episode} score={score:.6f}"
        f" geometry={geometry:.6f} position={position:.6f}"
        f" axis_angle={np.degrees(axis_angle):.2f}"
        f" capped_angle={min(np.degrees(axis_angle), 8.0):.2f}"
        f" predicted_ik_fail={failures}/{calls}"
        f" left_start={left_start} cutoff={cutoff}",
        flush=True,
    )


def eval(env, model, obs):
    base = env.unwrapped
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if not model.initialized:
        _initialize(base, model, cube_pose)
        model.initialized = True
    cube_pos = cube_pose[:, :3, 3]
    slot_xy = torch.norm(
        cube_pos[:, :2] - _slot_position(rack_pose)[:, :2], dim=-1
    )
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    opening = 0.0
    if model.step < model.cutoff:
        action = model.trajectory[model.step].clone()
        _retarget_arm(base, action, model, "right_arm", slice(7, 13))
        if model.step >= model.left_start:
            _retarget_arm(base, action, model, "left_arm", slice(0, 6))
        phase = "feasible_capped_handoff"
    else:
        stuck_high = model.ik_failures >= 8 and float(cube_pos[0, 2].item()) < 1.032
        normally_seated = float(cube_pos[0, 2].item()) < 1.006
        if (
            not model.release_active
            and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012
            and float(angle.item()) < 5.0
            and (stuck_high or normally_seated)
        ):
            model.release_active = True
            print(
                f"[KNN V23] supported release step={model.step}"
                f" z={float(cube_pos[0,2].item()):.5f}"
                f" slot_xy={float(slot_xy.item()):.5f} ik_fail={model.ik_failures}",
                flush=True,
            )
        if model.release_active:
            action, opening = _supported_release(
                base, model, cube_pose, rack_pose, slot_xy, angle
            )
            phase = "supported_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 20 == 0 or model.step == model.cutoff or success:
        ref_index = min(model.step - 1, len(model.ref_cube) - 1)
        desired_ref = model.scene_delta @ model.ref_cube[ref_index].reshape(1, 4, 4)
        ref_error = torch.norm(cube_pos[0] - desired_ref[0, :3, 3])
        print(
            f"[KNN V23] step={model.step} phase={phase}"
            f" ref_err={float(ref_error.item()):.5f}"
            f" slot_xy={float(slot_xy.item()):.5f}"
            f" z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f}"
            f" retarget_ik_fail={model.retarget_ik_failures}"
            f" place_ik_fail={model.ik_failures} opening={opening:.3f}"
            f" success={success}",
            flush=True,
        )
    done = (
        success
        or bool(torch.as_tensor(terminated).any().item())
        or bool(torch.as_tensor(truncated).any().item())
    )
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
