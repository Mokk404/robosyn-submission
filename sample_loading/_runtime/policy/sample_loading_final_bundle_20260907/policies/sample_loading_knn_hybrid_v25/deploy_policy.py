"""V24 plus local IK reliability ranking and action-bank fallback for OOD scenes."""

from __future__ import annotations

import numpy as np
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v14.deploy_policy import (
    _guided_wide,
    eval as fallback_eval,
    get_model as fallback_get_model,
)
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import _dataset
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v21.deploy_policy import Controller, _retarget_arm
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v22.deploy_policy import _capped_axis_rotation
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v23.deploy_policy import _feasibility, _left_start


def get_model(usr_args):
    model = Controller()
    model.mode = None
    model.fallback_model = None
    return model


def _delta(cube, reference, axis_angle):
    max_angle = 8.0 if np.degrees(axis_angle) > 10.0 else 180.0
    rotation = _capped_axis_rotation(reference[:3, 2], cube[:3, 2], max_angle)
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = rotation
    delta[:3, 3] = cube[:3, 3] - rotation @ reference[:3, 3]
    return delta, min(np.degrees(axis_angle), max_angle)


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy()
    candidates = []
    for episode, actions, cubes, cutoff in _dataset():
        position = float(np.linalg.norm(cube[:3, 3] - cubes[0, :3, 3]))
        axis_angle = float(
            np.arccos(np.clip(np.dot(cube[:3, 2], cubes[0, :3, 2]), -1.0, 1.0))
        )
        geometry = position + 0.06 * axis_angle
        candidates.append((geometry, episode, actions, cubes, cutoff, position, axis_angle))
    candidates.sort(key=lambda item: item[0])
    if candidates[0][0] > 0.050:
        model.mode = "actionbank"
        model.fallback_model = fallback_get_model(None)
        print(
            f"[HYBRID V25] OOD action-bank fallback nearest_cost={candidates[0][0]:.6f}"
            f" nearest_episode={candidates[0][1]} position={candidates[0][5]:.6f}"
            f" axis_angle={np.degrees(candidates[0][6]):.2f}",
            flush=True,
        )
        return
    ranked = []
    for geometry, episode, actions, cubes, cutoff, position, axis_angle in candidates[:12]:
        left_start = _left_start(actions, cutoff)
        delta, applied_angle = _delta(cube, cubes[0], axis_angle)
        fraction, failures, calls = _feasibility(
            base, actions, cutoff, left_start, delta
        )
        score = geometry + 0.040 * fraction
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
                applied_angle,
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
        applied_angle,
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
    model.mode = "knn"
    print(
        f"[HYBRID V25] selected episode={episode} score={score:.6f}"
        f" geometry={geometry:.6f} position={position:.6f}"
        f" axis_angle={np.degrees(axis_angle):.2f} applied_angle={applied_angle:.2f}"
        f" predicted_ik_fail={failures}/{calls}"
        f" left_start={left_start} cutoff={cutoff}",
        flush=True,
    )


def eval(env, model, obs):
    base = env.unwrapped
    if model.mode is None:
        cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
        _initialize(base, model, cube_pose)
    if model.mode == "actionbank":
        return fallback_eval(env, model.fallback_model, obs)
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
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
        phase = "reliability_ranked_handoff"
    else:
        high_stall = (
            model.ik_failures >= 8 and float(cube_pos[0, 2].item()) < 1.032
        )
        normal_gate = float(cube_pos[0, 2].item()) < 1.018 and (
            model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 1.002
        )
        if (
            not model.release_active
            and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012
            and float(angle.item()) < 5.0
            and (high_stall or normal_gate)
        ):
            model.release_active = True
            print(
                f"[HYBRID V25] release step={model.step}"
                f" z={float(cube_pos[0,2].item()):.5f}"
                f" slot_xy={float(slot_xy.item()):.5f} ik_fail={model.ik_failures}",
                flush=True,
            )
        if model.release_active:
            action, opening = _guided_wide(
                base, model, cube_pose, rack_pose, slot_xy, angle
            )
            phase = "wide_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 20 == 0 or model.step == model.cutoff or success:
        print(
            f"[HYBRID V25] step={model.step} phase={phase}"
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
    fresh = get_model(None)
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
