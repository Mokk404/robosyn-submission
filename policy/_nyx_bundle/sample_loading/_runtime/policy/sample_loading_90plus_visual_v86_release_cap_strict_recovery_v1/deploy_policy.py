"""V86: V85 with bounded release and physically plausible recovery checks.

V85 improved the fixed random20 result to 15/20, but four released tubes
tilted or escaped after the inherited schedule opened the left gripper from
10 mm all the way to 50 mm.  Its only true recovery episode also accepted
right-gripper qpos values of 0.47 m and 1.10 m as a retained tube.  V86 caps
the release at the already-successful 10 mm opening and validates recovery
against a bounded public-qpos interval.
"""
from __future__ import annotations

from typing import Any

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as metric_parent,
)
from policy.sample_loading_90plus_visual_v85_fresh_rgb_multirecovery_v1 import (
    deploy_policy as parent,
)


MIN_RETAINED_RIGHT_GRIPPER_Q_M = 0.003
MAX_RETAINED_RIGHT_GRIPPER_Q_M = 0.020
MAX_RELEASE_OPENING_M = 0.010


def _v86_estimate_poses(model, obs) -> None:
    legacy = model.legacy_model
    built = bool(getattr(legacy, "v85_recovery_built", False))
    validated = bool(getattr(legacy, "v85_recovery_validated", False))

    # Reuse V85 perception and drop detection, but suppress its one-sided
    # validation so physically impossible, very large qpos cannot pass.
    if built and not validated:
        legacy.v85_recovery_built = False
    try:
        parent._v85_estimate_poses(model, obs)
    finally:
        if built and not validated:
            legacy.v85_recovery_built = True

    queue = getattr(legacy, "recovery_queue", None)
    recovery_index = int(getattr(legacy, "recovery_index", 0))
    if (
        built
        and not validated
        and queue is not None
        and recovery_index >= parent.RECOVERY_VALIDATE_INDEX
    ):
        right_gripper_q = float(parent._public_qpos(obs)[13].item())
        retained = (
            MIN_RETAINED_RIGHT_GRIPPER_Q_M
            <= right_gripper_q
            <= MAX_RETAINED_RIGHT_GRIPPER_Q_M
        )
        legacy.v85_recovery_validated = True
        if retained:
            print(
                "[V86 STRICT RECOVERY VALIDATION] retained tube"
                f" right_gripper_q={right_gripper_q:.6f}",
                flush=True,
            )
        else:
            parent._install_safe_hold_from_obs(legacy, obs)
            print(
                "[V86 STRICT RECOVERY VALIDATION] rejected; safe hold"
                f" right_gripper_q={right_gripper_q:.6f}"
                f" valid=[{MIN_RETAINED_RIGHT_GRIPPER_Q_M:.3f},"
                f"{MAX_RETAINED_RIGHT_GRIPPER_Q_M:.3f}]",
                flush=True,
            )


def _v86_guided_release(base, legacy, cube_pose, rack_pose, slot_xy, angle):
    action, opening = legacy.v86_parent_guided(
        base, legacy, cube_pose, rack_pose, slot_xy, angle
    )
    applied = min(float(opening), MAX_RELEASE_OPENING_M)
    action[:, 6] = applied
    if float(opening) > applied and (
        int(getattr(legacy, "v12_phase_steps", 0)) <= 24
        or int(getattr(legacy, "v12_phase_steps", 0)) % 10 == 0
    ):
        print(
            "[V86 RELEASE CAP]"
            f" requested={float(opening):.3f} applied={applied:.3f}",
            flush=True,
        )
    return action, applied


def _install(model) -> None:
    parent._install(model)
    metric_parent._estimate_poses = _v86_estimate_poses
    legacy = model.legacy_model
    legacy.v86_parent_guided = model.v86_parent_guided
    model.legacy_module.implementation._guided_wide = _v86_guided_release


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    model.v86_parent_guided = model.legacy_module.implementation._guided_wide
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    return parent.rack_parent.parent.parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]
