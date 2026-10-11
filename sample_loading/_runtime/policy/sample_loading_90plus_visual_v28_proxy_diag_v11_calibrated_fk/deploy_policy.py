"""V11 diagnostic: offline-calibrated grasp transform after handoff.

Before the V28 handoff cutoff, cube and rack estimates come from declared RGB.
After cutoff, the tube pose is reconstructed from declared qpos, public robot
FK, and a constant left-EEF-to-tube transform fit only on training episodes.
No runtime object pose or evaluator metric is consumed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as metric_parent,
)
from policy.sample_loading_90plus_visual_v28_proxy_diag_v9_heldinsert import (
    deploy_policy as parent,
)


def _apply_calibrated_left_fk(model, env, obs) -> None:
    legacy = model.legacy_model
    if not getattr(legacy, "initialized", False):
        model.fk_tracking_steps = 0
        return
    cutoff = getattr(legacy, "cutoff", None)
    if cutoff is None:
        return
    frame = int(getattr(legacy, "trajectory_frame", 0))
    if frame < int(cutoff):
        model.v11_original_anchor(model, env, obs)
        return
    eef = metric_parent._left_eef_from_observation(env, obs)
    tracked = eef @ model.v11_left_eef_to_tube
    model.estimated_poses["cube"] = tracked
    model.fk_tracking_steps += 1
    if model.fk_tracking_steps <= 5 or model.fk_tracking_steps % 20 == 0:
        xyz = tracked[0, :3, 3]
        print(
            "[V11 CALIBRATED FK]"
            f" tracking_step={model.fk_tracking_steps}"
            f" frame={frame} cutoff={int(cutoff)}"
            f" cube_xyz=({float(xyz[0]):.5f},"
            f"{float(xyz[1]):.5f},{float(xyz[2]):.5f})",
            flush=True,
        )


def _install_calibrated_anchor(model, config: dict[str, Any]) -> None:
    calibration_path = Path(
        config["left_grasp_calibration_path"]
    ).expanduser().resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    matrix = torch.as_tensor(
        calibration["left_eef_to_tube_matrix"],
        dtype=torch.float32,
        device="cpu",
    ).reshape(1, 4, 4)
    model.v11_left_eef_to_tube = matrix
    model.v11_calibration_path = str(calibration_path)
    model.v11_original_anchor = getattr(
        metric_parent,
        "_v11_original_anchor",
        metric_parent._apply_left_fk_anchor,
    )
    metric_parent._v11_original_anchor = model.v11_original_anchor
    metric_parent._apply_left_fk_anchor = _apply_calibrated_left_fk


def get_model(usr_args: dict[str, Any]):
    config = dict(usr_args or {})
    model = parent.get_model(config)
    _install_calibrated_anchor(model, config)
    return model


def eval(env, model, obs):
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    model.fk_tracking_steps = 0
    metric_parent._apply_left_fk_anchor = _apply_calibrated_left_fk


__all__ = ["eval", "get_model", "reset_model"]
