"""V87: public-observation-only execution path for the V86 controller.

The V85/V86 experiments already replaced cube/rack simulator poses with RGB
estimates, but retained a compatibility proxy around a historical controller
and used ``robot.get_qpos`` inside recovery.  V87 removes both ambiguities:

* cube/rack poses come directly from the declared three RGB observations;
* arm/gripper state comes directly from ``obs["robot"]["qpos"]``;
* FK/IK are used only as motion-planning calculations;
* the only robot actuation is the action passed to ``env.step``.

It preserves V86's 10 mm release cap and bounded recovery validation.
"""
from __future__ import annotations

from typing import Any

import torch

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as perception_parent,
)
from policy.sample_loading_90plus_visual_v85_fresh_rgb_multirecovery_v1 import (
    deploy_policy as recovery_parent,
)
from policy.sample_loading_90plus_visual_v86_release_cap_strict_recovery_v1 import (
    deploy_policy as parent,
)


SAFE_HOLD_STEPS = 500


def _public_qpos(obs) -> torch.Tensor:
    qpos = torch.as_tensor(obs["robot"]["qpos"], dtype=torch.float32).reshape(-1)
    if qpos.numel() != 14:
        raise ValueError(f"Expected 14 public qpos values, got {qpos.numel()}")
    return qpos


def _cached_arms(base, legacy) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = getattr(legacy, "v87_public_qpos", None)
    if qpos is None:
        raise RuntimeError("V87 recovery requires a cached public qpos observation")
    qpos = qpos.to(device=base.device, dtype=torch.float32).reshape(14)
    return qpos[:6].reshape(1, 6), qpos[7:13].reshape(1, 6)


@torch.inference_mode()
def _v87_estimate_poses(model, obs) -> None:
    model.legacy_model.v87_public_qpos = _public_qpos(obs).detach().cpu()
    parent._v86_estimate_poses(model, obs)


def _safe_hold_from_public_qpos(base, legacy, append_segment) -> None:
    left_now, right_now = _cached_arms(base, legacy)
    rows = []
    append_segment(
        base,
        rows,
        left_now,
        right_now,
        left_now,
        right_now,
        SAFE_HOLD_STEPS,
        True,
        False,
    )
    legacy.recovery_queue = torch.cat(rows, dim=0).reshape(len(rows), 1, 14)
    legacy.recovery_index = 0
    legacy.v85_recovery_safe_hold = True
    legacy.v85_recovery_built = False


def _v87_build_recovery(base, legacy, cube_pose):
    original = recovery_parent._ORIGINAL_BUILD_RECOVERY
    if original is None:
        raise RuntimeError("V87 recovery hook was not initialized")
    helpers = original.__globals__
    find_grasp_frame = helpers["_find_grasp_frame"]
    ik_ok = helpers["_ik_ok"]
    append_segment = helpers["_append_segment"]

    robot = base.robot
    actions = legacy.raw_actions
    nominal_grasp = int(find_grasp_frame(actions))
    reference_cube = legacy.ref_cube[0].reshape(1, 4, 4)
    calibrated_cube = legacy.scene_delta @ reference_cube
    hybrid_delta = legacy.scene_delta.clone()
    hybrid_delta[:, :3, 3] += cube_pose[:, :3, 3] - calibrated_cube[:, :3, 3]
    left_now, right_now = _cached_arms(base, legacy)

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
                actions[pre_frame, 7:13], dtype=torch.float32, device=base.device
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
                pre_q, pre_seed = recovery_parent._ik_with_seeds(
                    robot,
                    desired_pre,
                    "right_arm",
                    (("current", right_now), ("reference", reference_pre_q)),
                    ik_ok,
                )
                if pre_q is None:
                    continue
                grasp_q, grasp_seed = recovery_parent._ik_with_seeds(
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
        _safe_hold_from_public_qpos(base, legacy, append_segment)
        print(
            "[V87 PUBLIC RECOVERY] no feasible IK candidate; safe hold"
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
        "[V87 PUBLIC RECOVERY] built"
        f" attempts={attempts} grasp_frame={grasp_frame} pre_frame={pre_frame}"
        f" offset_mm=({dx * 1000.0:.1f},{dy * 1000.0:.1f})"
        f" seeds=({pre_seed},{grasp_seed}) queue={len(rows)}"
        f" resume_frame={legacy.recovery_resume_frame}",
        flush=True,
    )
    return True


def _controller_step(env, model, obs):
    """Execute one step without simulator-object or robot-state reads."""
    base = env.unwrapped
    legacy = model.legacy_model
    implementation = model.legacy_module.implementation
    cube_pose = model.estimated_poses["cube"]
    rack_pose = model.estimated_poses["rack"]

    if not legacy.initialized:
        implementation._initialize(base, legacy, cube_pose)
        legacy.initialized = True

    cube_pos = cube_pose[:, :3, 3]
    slot_xy = torch.norm(
        cube_pos[:, :2] - implementation._slot_position(rack_pose)[:, :2], dim=-1
    )
    angle = torch.rad2deg(
        torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0))
    )
    opening = 0.0

    if legacy.recovery_queue is not None:
        action = legacy.recovery_queue[legacy.recovery_index]
        legacy.recovery_index += 1
        phase = "public_recovery"
        if legacy.recovery_index >= len(legacy.recovery_queue):
            legacy.recovery_queue = None
            legacy.trajectory_frame = min(
                legacy.recovery_resume_frame, legacy.cutoff
            )
    elif legacy.trajectory_frame < legacy.cutoff:
        frame = legacy.trajectory_frame
        should_recover = (
            frame >= 135
            and not legacy.recovery_attempted
            and float(cube_pos[0, 2].item()) < 0.88
        )
        if should_recover:
            legacy.recovery_attempted = True
            if _v87_build_recovery(base, legacy, cube_pose):
                action = legacy.recovery_queue[0]
                legacy.recovery_index = 1
                phase = "public_recovery"
            else:
                raise RuntimeError("V87 recovery must return a queue or safe hold")
        else:
            action = legacy.trajectory[frame].clone()
            implementation._retarget_arm(
                base, action, legacy, "right_arm", slice(7, 13)
            )
            if frame >= legacy.left_start:
                implementation._retarget_arm(
                    base, action, legacy, "left_arm", slice(0, 6)
                )
            phase = "ranked_handoff"
            legacy.trajectory_frame += 1
    else:
        high_stall = legacy.ik_failures >= 8 and float(cube_pos[0, 2]) < 1.032
        normal_gate = float(cube_pos[0, 2]) < 1.018 and (
            legacy.ik_failures >= 8 or float(cube_pos[0, 2]) < 1.002
        )
        if (
            not legacy.release_active
            and legacy.place_stage == "lower"
            and float(slot_xy.item()) < 0.012
            and float(angle.item()) < 5.0
            and (high_stall or normal_gate)
        ):
            legacy.release_active = True
            print(
                "[V87 PUBLIC CONTROLLER] release"
                f" step={legacy.step} predicted_z={float(cube_pos[0,2]):.5f}"
                f" predicted_slot_xy={float(slot_xy):.5f}"
                f" ik_fail={legacy.ik_failures}",
                flush=True,
            )
        if legacy.release_active:
            action, opening = implementation._guided_wide(
                base, legacy, cube_pose, rack_pose, slot_xy, angle
            )
            phase = "wide_release"
        else:
            action = implementation._place_servo(
                base, legacy, cube_pose, rack_pose, slot_xy, angle
            )
            phase = f"place_{legacy.place_stage}"

    next_obs, _reward, terminated, truncated, info = env.step(action)
    legacy.step += 1
    if legacy.step % 20 == 0:
        print(
            "[V87 PUBLIC CONTROLLER]"
            f" step={legacy.step} traj={legacy.trajectory_frame} phase={phase}"
            f" predicted_slot_xy={float(slot_xy):.5f}"
            f" predicted_z={float(cube_pos[0,2]):.5f}"
            f" predicted_angle={float(angle):.2f}"
            f" retarget_ik_fail={legacy.retarget_ik_failures}"
            f" place_ik_fail={legacy.ik_failures} opening={opening:.3f}",
            flush=True,
        )
    done = bool(torch.as_tensor(terminated).any().item()) or bool(
        torch.as_tensor(truncated).any().item()
    )
    return next_obs, info, done


def _install(model) -> None:
    parent._install(model)
    perception_parent._estimate_poses = _v87_estimate_poses
    model.legacy_module.implementation._build_recovery = _v87_build_recovery


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    model.legacy_model.v87_public_qpos = None
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    inference_times_s: list[float] = []
    started_at = perception_parent.start_inference(model.device)
    try:
        perception_parent._estimate_poses(model, obs)
        perception_parent._apply_left_fk_anchor(model, env, obs)
        next_obs, info, done = _controller_step(env, model, obs)
    finally:
        perception_parent.finish_inference(
            started_at, inference_times_s, model.device
        )
    return next_obs, info, done, inference_times_s


def reset_model(model):
    parent.reset_model(model)
    model.legacy_model.v87_public_qpos = None
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]
