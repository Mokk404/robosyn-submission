"""V39: V33 plus public-gripper missed-hold recovery guard.

The RGB height probe can report a lifted tube after the object has slipped.
The public right-gripper qpos cleanly separates a retained tube (about 0.007 m
in audited runs) from an empty fully closed gripper (near zero).  After the
initial lift, one guarded transition re-enables the inherited object-relative
regrasp using the observation-driven cube proxy.
"""
from __future__ import annotations

from typing import Any
import torch

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as metric_parent,
)
from policy.sample_loading_90plus_visual_v28_proxy_diag_v33_rgb_recovery_release_deadline import (
    deploy_policy as parent,
)

DROP_GUARD_START_FRAME = 145
DROP_GUARD_END_MARGIN = 20
RIGHT_GRIPPER_EMPTY_Q_M = 0.003


@torch.inference_mode()
def _v39_estimate_poses(model, obs) -> None:
    parent._v31_estimate_poses(model, obs)
    legacy = model.legacy_model
    if model.v39_drop_guard_fired or not getattr(legacy, "initialized", False):
        return
    frame = int(getattr(legacy, "trajectory_frame", 0))
    cutoff = getattr(legacy, "cutoff", None)
    if cutoff is None:
        return
    in_window = (
        frame >= DROP_GUARD_START_FRAME
        and frame < int(cutoff) - DROP_GUARD_END_MARGIN
    )
    if not in_window:
        return
    qpos = torch.as_tensor(obs["robot"]["qpos"]).reshape(-1)
    if qpos.numel() != 14:
        raise ValueError(f"Expected 14 public qpos values, got {qpos.numel()}")
    right_gripper_q = float(qpos[13].item())
    if right_gripper_q < RIGHT_GRIPPER_EMPTY_Q_M:
        legacy.recovery_attempted = False
        model.v32_recovery_enabled = True
        model.v39_drop_guard_fired = True
        print(
            "[V39 QPOS DROP GUARD]"
            f" frame={frame} right_gripper_q={right_gripper_q:.6f}"
            f" threshold={RIGHT_GRIPPER_EMPTY_Q_M:.6f}"
            " recovery_enabled=True",
            flush=True,
        )


def _install(model) -> None:
    metric_parent._estimate_poses = _v39_estimate_poses


def get_model(usr_args: dict[str, Any]):
    model = parent.get_model(dict(usr_args or {}))
    model.v39_drop_guard_fired = False
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    model.v39_drop_guard_fired = False
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]
