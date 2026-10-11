"""V89: cautious seating followed by delayed guaranteed release."""
from __future__ import annotations

from typing import Any

from policy.sample_loading_90plus_visual_v87_public_obs_direct_controller_v1 import (
    deploy_policy as parent,
)


def _release_limit(phase_step: int) -> float:
    if phase_step < 24:
        return 0.010
    if phase_step < 40:
        return 0.012
    if phase_step < 56:
        return 0.015
    if phase_step < 72:
        return 0.020
    if phase_step < 88:
        return 0.030
    return 0.050


def _v89_guided_release(base, legacy, cube_pose, rack_pose, slot_xy, angle):
    action, requested = legacy.v86_parent_guided(
        base, legacy, cube_pose, rack_pose, slot_xy, angle
    )
    phase_step = int(getattr(legacy, "v12_phase_steps", 0))
    limit = _release_limit(phase_step)
    applied = min(float(requested), limit)
    action[:, 6] = applied
    if phase_step <= 92 or phase_step % 10 == 0:
        print(
            "[V89 DELAYED FULL RELEASE]"
            f" phase_step={phase_step} requested={float(requested):.3f}"
            f" limit={limit:.3f} applied={applied:.3f}",
            flush=True,
        )
    return action, applied


def _install(model) -> None:
    parent._install(model)
    model.legacy_module.implementation._guided_wide = _v89_guided_release


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    inference_times_s: list[float] = []
    started_at = parent.perception_parent.start_inference(model.device)
    try:
        parent.perception_parent._estimate_poses(model, obs)
        parent.perception_parent._apply_left_fk_anchor(model, env, obs)
        next_obs, info, done = parent._controller_step(env, model, obs)
    finally:
        parent.perception_parent.finish_inference(
            started_at, inference_times_s, model.device
        )
    return next_obs, info, done, inference_times_s


def reset_model(model):
    parent.reset_model(model)
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]
