"""Clean entrypoint for the packaged V89 sample_loading controller."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PACKAGE_ROOT / "_runtime"
RUNTIME_POLICY = RUNTIME_ROOT / "policy"
RUNTIME_MODEL_CODE = RUNTIME_ROOT / "model_code"

import policy as _policy_package

if str(RUNTIME_POLICY) not in _policy_package.__path__:
    _policy_package.__path__.append(str(RUNTIME_POLICY))
if str(RUNTIME_MODEL_CODE) not in sys.path:
    sys.path.insert(0, str(RUNTIME_MODEL_CODE))

from policy.sample_loading_90plus_visual_v89_public_obs_delayed_full_release_v1 import (  # noqa: E402
    deploy_policy as _implementation,
)


PATH_DEFAULTS = {
    "checkpoint_path": PACKAGE_ROOT / "checkpoint" / "pose_regressor_v4.ckpt",
    "cube_highres_checkpoint_path": PACKAGE_ROOT / "checkpoint" / "cube_highres_pose_v1.ckpt",
    "rack_highres_checkpoint_path": PACKAGE_ROOT / "checkpoint" / "rack_highres_pose_v1.ckpt",
    "rgb_dynamic_checkpoint_path": PACKAGE_ROOT / "checkpoint" / "dynamic_pose_rgb_v3_epoch029.ckpt",
    "left_grasp_calibration_path": PACKAGE_ROOT / "config" / "left_grasp_calibration.json",
    "diagnostic_history_policy_path": RUNTIME_POLICY,
    "diagnostic_trainer_path": RUNTIME_MODEL_CODE,
}


def _resolved_config(usr_args: dict[str, Any] | None) -> dict[str, Any]:
    config = dict(usr_args or {})
    for key, default in PATH_DEFAULTS.items():
        value = Path(config.get(key, default)).expanduser()
        if not value.is_absolute():
            value = PACKAGE_ROOT / value
        config[key] = str(value.resolve())
    return config


def get_model(usr_args: dict[str, Any] | None):
    return _implementation.get_model(_resolved_config(usr_args))


def eval(env, model, obs):
    return _implementation.eval(env, model, obs)


def reset_model(model):
    return _implementation.reset_model(model)


__all__ = ["eval", "get_model", "reset_model"]
