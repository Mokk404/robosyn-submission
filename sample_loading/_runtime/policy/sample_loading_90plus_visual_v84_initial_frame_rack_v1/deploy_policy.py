"""V84: freeze V82's rack estimate from the first pre-motion RGB frame.

V81/V82 sampled five consecutive controller frames.  Successful episodes had
stable predictions, but several failures showed centimetre-scale drift once
the moving arms occluded the rack.  This version makes the timing invariant
explicit: the high-resolution rack network is evaluated exactly once, before
the first policy action, and that estimate is frozen for the episode.
"""
from __future__ import annotations

from typing import Any

from policy.sample_loading_90plus_visual_v82_highres_rack_only_v1 import (
    deploy_policy as parent,
)


def get_model(usr_args: dict[str, Any]):
    config = dict(usr_args or {})
    config["v82_rack_sample_frames"] = 1
    model = parent.get_model(config)
    if model.v82_rack_sample_frames != 1:
        raise RuntimeError("V84 requires exactly one pre-motion rack sample")
    print(
        "[V84 INITIAL-FRAME RACK] using exactly one pre-motion RGB sample",
        flush=True,
    )
    return model


def eval(env, model, obs):
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)


__all__ = ["eval", "get_model", "reset_model"]
