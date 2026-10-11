"""V12 diagnostic: calibrated FK insertion followed by guided release.

V11 reached the rack guide with low XY/angle error but remained suspended about
7 mm above the official z gate because the tight gripper kept bearing the tube.
V12 starts a slow release only after the calibrated FK pose reaches the observed
contact plateau.  Runtime inputs remain RGB, qpos, public FK, and the offline
training calibration.
"""
from __future__ import annotations

from typing import Any

from policy.sample_loading_90plus_visual_v28_proxy_diag_v11_calibrated_fk import (
    deploy_policy as parent,
)


RELEASE_GATE_ABOVE_RACK_M = 0.105
TARGET_HEIGHT_ABOVE_RACK_M = 0.064
LEGACY_PLACE_HEIGHT_ABOVE_RACK_M = 0.052


def _v12_guided_release(base, legacy, cube_pose, rack_pose, slot_xy, angle):
    if not hasattr(legacy, "v12_phase"):
        legacy.v12_phase = "descent"
        legacy.v12_phase_steps = 0

    phase = legacy.v12_phase
    phase_step = int(legacy.v12_phase_steps)
    cube_z = float(cube_pose[0, 2, 3].item())
    rack_z = float(rack_pose[0, 2, 3].item())
    release_gate_z = rack_z + RELEASE_GATE_ABOVE_RACK_M

    if phase == "descent":
        servo_rack = rack_pose.clone()
        servo_rack[:, 2, 3] += (
            TARGET_HEIGHT_ABOVE_RACK_M
            - LEGACY_PLACE_HEIGHT_ABOVE_RACK_M
        )
        action = legacy.v12_place_servo(
            base, legacy, cube_pose, servo_rack, slot_xy, angle
        )
        opening = 0.006
        if cube_z <= release_gate_z:
            legacy.v12_phase = "seat"
            legacy.v12_phase_steps = 0
        else:
            legacy.v12_phase_steps += 1
    elif phase == "seat":
        action, _unused = legacy.v12_original_guided_wide(
            base, legacy, cube_pose, rack_pose, slot_xy, angle
        )
        opening = 0.006
        if phase_step >= 3:
            legacy.v12_phase = "release"
            legacy.v12_phase_steps = 0
        else:
            legacy.v12_phase_steps += 1
    else:
        action, _unused = legacy.v12_original_guided_wide(
            base, legacy, cube_pose, rack_pose, slot_xy, angle
        )
        if phase_step < 6:
            opening = 0.008
        elif phase_step < 14:
            opening = 0.010
        elif phase_step < 22:
            opening = 0.012
        elif phase_step < 30:
            opening = 0.015
        elif phase_step < 38:
            opening = 0.020
        elif phase_step < 46:
            opening = 0.030
        else:
            opening = 0.050
        legacy.v12_phase_steps += 1

    action[:, 6] = opening
    action[:, 13] = 0.050
    legacy.release_steps += 1
    if (
        phase_step < 50
        or phase_step % 10 == 0
        or legacy.v12_phase != phase
    ):
        print(
            f"[V12 GUIDED RELEASE] phase={phase}"
            f" phase_step={phase_step} next={legacy.v12_phase}"
            f" opening={opening:.3f} z={cube_z:.5f}"
            f" release_gate_z={release_gate_z:.5f}"
            f" slot_xy={float(slot_xy.item()):.5f}"
            f" angle={float(angle.item()):.2f}",
            flush=True,
        )
    return action, opening


def _install_hook(model) -> None:
    implementation = model.legacy_module.implementation
    original = model.v9_original_guided_wide
    place_servo = model.v9_place_servo
    implementation._v12_original_guided_wide = original
    implementation._v12_place_servo = place_servo
    implementation._guided_wide = _v12_guided_release
    model.v12_original_guided_wide = original
    model.v12_place_servo = place_servo
    model.legacy_model.v12_original_guided_wide = original
    model.legacy_model.v12_place_servo = place_servo


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    _install_hook(model)
    return model


def eval(env, model, obs):
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    legacy = model.legacy_model
    legacy.v12_original_guided_wide = model.v12_original_guided_wide
    legacy.v12_place_servo = model.v12_place_servo
    for name in ("v12_phase", "v12_phase_steps"):
        if hasattr(legacy, name):
            delattr(legacy, name)
    model.legacy_module.implementation._guided_wide = _v12_guided_release


__all__ = ["eval", "get_model", "reset_model"]
