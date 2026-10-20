"""Reliability-ranked KNN replay with an object-relative live regrasp on failed lift."""

from __future__ import annotations

import numpy as np
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v11.deploy_policy import _append_segment
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v14.deploy_policy import _guided_wide
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _ik_ok
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import _dataset
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_hybrid_v25.deploy_policy import _delta
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v21.deploy_policy import Controller, _retarget_arm
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v23.deploy_policy import _feasibility, _left_start


def get_model(usr_args):
    model = Controller()
    model.trajectory_frame = 0
    model.raw_actions = None
    model.recovery_queue = None
    model.recovery_index = 0
    model.recovery_attempted = False
    model.recovery_resume_frame = 0
    return model


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy()
    candidates = []
    for episode, actions, cubes, cutoff in _dataset():
        reference = cubes[0]
        if not (0.75 < float(reference[2, 3]) < 0.90):
            continue
        position = float(np.linalg.norm(cube[:3, 3] - reference[:3, 3]))
        axis_angle = float(
            np.arccos(np.clip(np.dot(cube[:3, 2], reference[:3, 2]), -1.0, 1.0))
        )
        geometry = position + 0.06 * axis_angle
        candidates.append((geometry, episode, actions, cubes, cutoff, position, axis_angle))
    candidates.sort(key=lambda item: item[0])
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
                score, geometry, episode, actions, cubes, cutoff, position,
                axis_angle, left_start, delta, applied_angle, failures, calls,
            )
        )
    ranked.sort(key=lambda item: item[0])
    (
        score, geometry, episode, actions, cubes, cutoff, position,
        axis_angle, left_start, delta, applied_angle, failures, calls,
    ) = ranked[0]
    model.raw_actions = actions.copy()
    scaled = actions.copy()
    scaled[:, 6] *= 0.05
    scaled[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(
        scaled, dtype=torch.float32, device=base.device
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
        f"[KNN V27] selected episode={episode} score={score:.6f}"
        f" geometry={geometry:.6f} current_xy=({cube[0,3]:.5f},{cube[1,3]:.5f})"
        f" reference_xy=({cubes[0,0,3]:.5f},{cubes[0,1,3]:.5f})"
        f" position={position:.6f} axis_angle={np.degrees(axis_angle):.2f}"
        f" applied_angle={applied_angle:.2f} predicted_ik_fail={failures}/{calls}"
        f" left_start={left_start} cutoff={cutoff}",
        flush=True,
    )


def _find_grasp_frame(actions):
    grip = actions[:, 13]
    opened = np.flatnonzero(grip > 0.8)
    start = int(opened[0] + 20) if len(opened) else 40
    run = 0
    for frame in range(start, len(grip)):
        run = run + 1 if grip[frame] < 0.10 else 0
        if run >= 8:
            return frame - 7
    return min(100, len(grip) - 1)


def _build_recovery(base, model, cube_pose):
    robot = base.robot
    actions = model.raw_actions
    grasp_frame = _find_grasp_frame(actions)
    pre_frame = max(grasp_frame - 28, 20)
    reference_cube = model.ref_cube[0].reshape(1, 4, 4)
    reference_pre_q = torch.as_tensor(
        actions[pre_frame, 7:13], dtype=torch.float32, device=base.device
    ).reshape(1, 6)
    reference_grasp_q = torch.as_tensor(
        actions[grasp_frame, 7:13], dtype=torch.float32, device=base.device
    ).reshape(1, 6)
    reference_pre = robot.compute_fk(reference_pre_q, name="right_arm", to_matrix=True)
    reference_grasp = robot.compute_fk(
        reference_grasp_q, name="right_arm", to_matrix=True
    )
    object_delta = cube_pose @ torch.linalg.inv(reference_cube)
    desired_pre = object_delta @ reference_pre
    desired_grasp = object_delta @ reference_grasp
    right_now = robot.get_qpos(name="right_arm").clone()
    left_now = robot.get_qpos(name="left_arm").clone()
    pre_ret = robot.compute_ik(
        pose=desired_pre, joint_seed=right_now, name="right_arm"
    )
    if not _ik_ok(pre_ret):
        pre_ret = robot.compute_ik(
            pose=desired_pre, joint_seed=reference_pre_q, name="right_arm"
        )
    if not _ik_ok(pre_ret):
        print("[KNN V27] recovery pregrasp IK failed", flush=True)
        return False
    pre_q = pre_ret[1]
    grasp_ret = robot.compute_ik(
        pose=desired_grasp, joint_seed=pre_q, name="right_arm"
    )
    if not _ik_ok(grasp_ret):
        print("[KNN V27] recovery grasp IK failed", flush=True)
        return False
    grasp_q = grasp_ret[1]
    lift1 = desired_grasp.clone()
    lift1[:, 2, 3] += 0.10
    lift1_ret = robot.compute_ik(pose=lift1, joint_seed=grasp_q, name="right_arm")
    if not _ik_ok(lift1_ret):
        print("[KNN V27] recovery lift1 IK failed", flush=True)
        return False
    lift2 = desired_grasp.clone()
    lift2[:, 2, 3] += 0.22
    lift2_ret = robot.compute_ik(
        pose=lift2, joint_seed=lift1_ret[1], name="right_arm"
    )
    if not _ik_ok(lift2_ret):
        print("[KNN V27] recovery lift2 IK failed", flush=True)
        return False
    rows = []
    left, right = left_now, right_now
    left, right = _append_segment(
        base, rows, left, right, left, pre_q, 30, True, True
    )
    left, right = _append_segment(
        base, rows, left, right, left, grasp_q, 25, True, True
    )
    left, right = _append_segment(
        base, rows, left, right, left, right, 15, True, False
    )
    left, right = _append_segment(
        base, rows, left, right, left, lift1_ret[1], 25, True, False
    )
    left, right = _append_segment(
        base, rows, left, right, left, lift2_ret[1], 30, True, False
    )
    model.recovery_queue = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    model.recovery_index = 0
    model.recovery_resume_frame = max(grasp_frame + 55, model.left_start - 10)
    print(
        f"[KNN V27] live recovery built grasp_frame={grasp_frame}"
        f" pre_frame={pre_frame} queue={len(rows)}"
        f" resume_frame={model.recovery_resume_frame}",
        flush=True,
    )
    return True


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
    if model.recovery_queue is not None:
        action = model.recovery_queue[model.recovery_index]
        model.recovery_index += 1
        phase = "live_recovery"
        if model.recovery_index >= len(model.recovery_queue):
            model.recovery_queue = None
            model.trajectory_frame = min(model.recovery_resume_frame, model.cutoff)
    elif model.trajectory_frame < model.cutoff:
        frame = model.trajectory_frame
        if (
            frame >= 135
            and not model.recovery_attempted
            and float(cube_pos[0, 2].item()) < 0.88
        ):
            model.recovery_attempted = True
            if _build_recovery(base, model, cube_pose):
                action = model.recovery_queue[0]
                model.recovery_index = 1
                phase = "live_recovery"
            else:
                action = model.trajectory[frame].clone()
                _retarget_arm(base, action, model, "right_arm", slice(7, 13))
                phase = "recovery_failed_handoff"
                model.trajectory_frame += 1
        else:
            action = model.trajectory[frame].clone()
            _retarget_arm(base, action, model, "right_arm", slice(7, 13))
            if frame >= model.left_start:
                _retarget_arm(base, action, model, "left_arm", slice(0, 6))
            phase = "ranked_handoff"
            model.trajectory_frame += 1
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
                f"[KNN V27] release step={model.step}"
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
    if model.step % 20 == 0 or success:
        print(
            f"[KNN V27] step={model.step} traj={model.trajectory_frame} phase={phase}"
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
