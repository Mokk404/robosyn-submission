"""V85: V84 plus fresh-RGB, multi-candidate missed-hold recovery.

The historical recovery consumed the frozen initialization pose and, when IK
failed, resumed the old handoff trajectory with an empty gripper.  V85 takes a
fresh declared RGB observation when the public qpos drop guard fires, searches
several translation/trajectory candidates, validates the closed gripper from
public qpos, and enters a bounded safe hold instead of continuing empty-handed
when no recovery is feasible.
"""
from __future__ import annotations

from typing import Any

import torch

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as metric_parent,
)
from policy.sample_loading_90plus_visual_v82_highres_rack_only_v1 import (
    deploy_policy as rack_parent,
)
from policy.sample_loading_90plus_visual_v84_initial_frame_rack_v1 import (
    deploy_policy as parent,
)


RIGHT_GRIPPER_EMPTY_Q_M = 0.003
RECOVERY_VALIDATE_INDEX = 56
SAFE_HOLD_STEPS = 500
_ORIGINAL_BUILD_RECOVERY = None


def _public_qpos(obs) -> torch.Tensor:
    qpos = torch.as_tensor(obs["robot"]["qpos"], dtype=torch.float32).reshape(-1)
    if qpos.numel() != 14:
        raise ValueError(f"Expected 14 public qpos values, got {qpos.numel()}")
    return qpos


@torch.inference_mode()
def _v85_estimate_poses(model, obs) -> None:
    rack_parent._v82_estimate_poses(model, obs)
    legacy = model.legacy_model
    drop_fired = bool(getattr(model, "v39_drop_guard_fired", False))
    if drop_fired and not model.v85_fresh_cube_captured:
        images, state = metric_parent._preprocess(model, obs)
        normalized = model.v30_dynamic_perception(images, state)[0].float()
        prediction = normalized * model.target_std + model.target_mean
        if not torch.isfinite(prediction).all():
            raise ValueError("V85 fresh RGB recovery prediction produced NaN or Inf")
        model.estimated_poses["cube"] = metric_parent._pose(
            metric_parent._axis_to_rotation(prediction[3:6]),
            prediction[0:3],
        )
        model.v85_fresh_cube_captured = True
        print(
            "[V85 FRESH RGB RECOVERY]"
            f" cube_xyz=({float(prediction[0]):.5f},"
            f"{float(prediction[1]):.5f},{float(prediction[2]):.5f})",
            flush=True,
        )

    queue = getattr(legacy, "recovery_queue", None)
    recovery_index = int(getattr(legacy, "recovery_index", 0))
    if (
        getattr(legacy, "v85_recovery_built", False)
        and not getattr(legacy, "v85_recovery_validated", False)
        and queue is not None
        and recovery_index >= RECOVERY_VALIDATE_INDEX
    ):
        right_gripper_q = float(_public_qpos(obs)[13].item())
        legacy.v85_recovery_validated = True
        if right_gripper_q < RIGHT_GRIPPER_EMPTY_Q_M:
            _install_safe_hold_from_obs(legacy, obs)
            print(
                "[V85 RECOVERY VALIDATION] failed; entering safe hold"
                f" right_gripper_q={right_gripper_q:.6f}",
                flush=True,
            )
        else:
            print(
                "[V85 RECOVERY VALIDATION] retained tube"
                f" right_gripper_q={right_gripper_q:.6f}",
                flush=True,
            )


def _install_safe_hold_from_obs(legacy, obs) -> None:
    action = _public_qpos(obs).reshape(1, 1, 14).clone()
    action[0, 0, 13] = 0.0
    legacy.recovery_queue = action.repeat(SAFE_HOLD_STEPS, 1, 1)
    legacy.recovery_index = 0
    legacy.v85_recovery_safe_hold = True


def _install_safe_hold_from_robot(base, legacy, append_segment) -> None:
    left = base.robot.get_qpos(name="left_arm").clone()
    right = base.robot.get_qpos(name="right_arm").clone()
    rows = []
    append_segment(
        base, rows, left, right, left, right, SAFE_HOLD_STEPS, True, False
    )
    legacy.recovery_queue = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    legacy.recovery_index = 0
    legacy.v85_recovery_safe_hold = True
    legacy.v85_recovery_built = False


def _ik_with_seeds(robot, pose, arm, seeds, ik_ok):
    for seed_name, seed in seeds:
        result = robot.compute_ik(pose=pose, joint_seed=seed, name=arm)
        if ik_ok(result):
            return result[1], seed_name
    return None, None


def _v85_build_recovery(base, legacy, cube_pose):
    if _ORIGINAL_BUILD_RECOVERY is None:
        raise RuntimeError("V85 recovery hook was not initialized")
    helpers = _ORIGINAL_BUILD_RECOVERY.__globals__
    find_grasp_frame = helpers["_find_grasp_frame"]
    ik_ok = helpers["_ik_ok"]
    append_segment = helpers["_append_segment"]

    robot = base.robot
    actions = legacy.raw_actions
    nominal_grasp = int(find_grasp_frame(actions))
    reference_cube = legacy.ref_cube[0].reshape(1, 4, 4)
    calibrated_cube = legacy.scene_delta @ reference_cube
    hybrid_delta = legacy.scene_delta.clone()
    hybrid_delta[:, :3, 3] += (
        cube_pose[:, :3, 3] - calibrated_cube[:, :3, 3]
    )
    right_now = robot.get_qpos(name="right_arm").clone()
    left_now = robot.get_qpos(name="left_arm").clone()

    xy_offsets = (
        (0.0, 0.0),
        (0.010, 0.0),
        (-0.010, 0.0),
        (0.0, 0.010),
        (0.0, -0.010),
    )
    frame_offsets = (0, -4, 4)
    pre_gaps = (28, 36, 20)
    chosen = None
    attempts = 0
    for frame_offset in frame_offsets:
        grasp_frame = max(1, min(len(actions) - 1, nominal_grasp + frame_offset))
        reference_grasp_q = torch.as_tensor(
            actions[grasp_frame, 7:13], dtype=torch.float32, device=base.device
        ).reshape(1, 6)
        reference_grasp = robot.compute_fk(
            reference_grasp_q, name="right_arm", to_matrix=True
        )
        for pre_gap in pre_gaps:
            pre_frame = max(grasp_frame - pre_gap, 20)
            reference_pre_q = torch.as_tensor(
                actions[pre_frame, 7:13],
                dtype=torch.float32,
                device=base.device,
            ).reshape(1, 6)
            reference_pre = robot.compute_fk(
                reference_pre_q, name="right_arm", to_matrix=True
            )
            for dx, dy in xy_offsets:
                attempts += 1
                candidate_delta = hybrid_delta.clone()
                candidate_delta[:, 0, 3] += dx
                candidate_delta[:, 1, 3] += dy
                desired_pre = candidate_delta @ reference_pre
                desired_grasp = candidate_delta @ reference_grasp
                pre_q, pre_seed = _ik_with_seeds(
                    robot,
                    desired_pre,
                    "right_arm",
                    (("current", right_now), ("reference", reference_pre_q)),
                    ik_ok,
                )
                if pre_q is None:
                    continue
                grasp_q, grasp_seed = _ik_with_seeds(
                    robot,
                    desired_grasp,
                    "right_arm",
                    (("pregrasp", pre_q), ("reference", reference_grasp_q)),
                    ik_ok,
                )
                if grasp_q is None:
                    continue
                chosen = (
                    grasp_frame,
                    pre_frame,
                    pre_q,
                    grasp_q,
                    dx,
                    dy,
                    pre_seed,
                    grasp_seed,
                )
                break
            if chosen is not None:
                break
        if chosen is not None:
            break

    if chosen is None:
        _install_safe_hold_from_robot(base, legacy, append_segment)
        print(
            "[V85 MULTI RECOVERY] no feasible IK candidate; safe hold"
            f" attempts={attempts}",
            flush=True,
        )
        return True

    (
        grasp_frame,
        pre_frame,
        pre_q,
        grasp_q,
        dx,
        dy,
        pre_seed,
        grasp_seed,
    ) = chosen
    rows = []
    left, right = left_now, right_now
    left, right = append_segment(
        base, rows, left, right, left, pre_q, 26, True, True
    )
    left, right = append_segment(
        base, rows, left, right, left, grasp_q, 22, True, True
    )
    left, right = append_segment(
        base, rows, left, right, left, right, 12, True, False
    )
    legacy.recovery_queue = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    legacy.recovery_index = 0
    legacy.recovery_resume_frame = max(
        grasp_frame + 12, int(legacy.left_start) - 20
    )
    legacy.v85_recovery_built = True
    legacy.v85_recovery_validated = False
    legacy.v85_recovery_safe_hold = False
    print(
        "[V85 MULTI RECOVERY] built"
        f" attempts={attempts} grasp_frame={grasp_frame} pre_frame={pre_frame}"
        f" offset_mm=({dx * 1000.0:.1f},{dy * 1000.0:.1f})"
        f" seeds=({pre_seed},{grasp_seed}) queue={len(rows)}"
        f" resume_frame={legacy.recovery_resume_frame}",
        flush=True,
    )
    return True


def _install(model) -> None:
    global _ORIGINAL_BUILD_RECOVERY
    metric_parent._estimate_poses = _v85_estimate_poses
    implementation = model.legacy_module.implementation
    if _ORIGINAL_BUILD_RECOVERY is None:
        _ORIGINAL_BUILD_RECOVERY = implementation._build_recovery
    implementation._build_recovery = _v85_build_recovery


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    model.v85_fresh_cube_captured = False
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    return rack_parent.parent.parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    model.v85_fresh_cube_captured = False
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]
