"""V9 diagnostic: keep a tight grasp through the official insertion gate.

V8 proved that a 0.010 m command already lets the tube slip during descent.
V9 keeps the left gripper at 0.006 m, descends the public joint-FK object
anchor below rack_z + 0.07 m, and holds still.  The official evaluator alone
decides success.  Evaluator-returned metrics remain logging-only.
"""
from __future__ import annotations

from typing import Any

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as parent,
)


TARGET_HEIGHT_ABOVE_RACK_M = 0.064
LEGACY_PLACE_HEIGHT_ABOVE_RACK_M = 0.052
TIGHT_GRIPPER_OPENING_M = 0.006


def _v9_heldinsert(base, legacy, cube_pose, rack_pose, slot_xy, angle):
    if not hasattr(legacy, "v9_insert_phase"):
        legacy.v9_insert_phase = "descent"
        legacy.v9_phase_steps = 0

    phase = legacy.v9_insert_phase
    phase_step = int(legacy.v9_phase_steps)
    cube_z = float(cube_pose[0, 2, 3].item())
    rack_z = float(rack_pose[0, 2, 3].item())
    target_z = rack_z + TARGET_HEIGHT_ABOVE_RACK_M

    if phase == "descent":
        servo_rack = rack_pose.clone()
        servo_rack[:, 2, 3] += (
            TARGET_HEIGHT_ABOVE_RACK_M
            - LEGACY_PLACE_HEIGHT_ABOVE_RACK_M
        )
        action = legacy.v9_place_servo(
            base, legacy, cube_pose, servo_rack, slot_xy, angle
        )
        if cube_z <= target_z + 0.001:
            legacy.v9_insert_phase = "hold"
            legacy.v9_phase_steps = 0
        else:
            legacy.v9_phase_steps += 1
    else:
        action, _unused = legacy.v9_original_guided_wide(
            base, legacy, cube_pose, rack_pose, slot_xy, angle
        )
        legacy.v9_phase_steps += 1

    opening = TIGHT_GRIPPER_OPENING_M
    action[:, 6] = opening
    action[:, 13] = 0.050
    legacy.release_steps += 1
    if (
        phase_step < 30
        or phase_step % 10 == 0
        or legacy.v9_insert_phase != phase
    ):
        print(
            f"[V9 HELDINSERT] phase={phase}"
            f" phase_step={phase_step} next={legacy.v9_insert_phase}"
            f" opening={opening:.3f} z={cube_z:.5f}"
            f" target_z={target_z:.5f}"
            f" slot_xy={float(slot_xy.item()):.5f}"
            f" angle={float(angle.item()):.2f}",
            flush=True,
        )
    return action, opening


def _install_hook(model) -> None:
    implementation = model.legacy_module.implementation
    original = getattr(
        implementation, "_v9_original_guided_wide", implementation._guided_wide
    )
    place_servo = getattr(
        implementation, "_v9_place_servo", implementation._place_servo
    )
    implementation._v9_original_guided_wide = original
    implementation._v9_place_servo = place_servo
    implementation._guided_wide = _v9_heldinsert
    model.v9_original_guided_wide = original
    model.v9_place_servo = place_servo
    model.legacy_model.v9_original_guided_wide = original
    model.legacy_model.v9_place_servo = place_servo


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    _install_hook(model)
    return model


def eval(env, model, obs):
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    legacy = model.legacy_model
    legacy.v9_original_guided_wide = model.v9_original_guided_wide
    legacy.v9_place_servo = model.v9_place_servo
    for name in ("v9_insert_phase", "v9_phase_steps"):
        if hasattr(legacy, name):
            delattr(legacy, name)
    model.legacy_module.implementation._guided_wide = _v9_heldinsert


__all__ = ["eval", "get_model", "reset_model"]
