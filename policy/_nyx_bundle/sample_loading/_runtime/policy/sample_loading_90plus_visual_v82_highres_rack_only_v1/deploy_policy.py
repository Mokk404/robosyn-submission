"""V82: isolate V81's high-resolution rack estimate on top of V39.

This ablation intentionally keeps V39's original visual/FK tube tracking.  It
changes only the rack XY/yaw estimate, so its result can be compared directly
with V39 and the calibrated-translation-only V83 policy.
"""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from policy.sample_loading_90plus_visual_v28_proxy_diag_v4_metriclog import (
    deploy_policy as metric_parent,
)
from policy.sample_loading_90plus_visual_v28_proxy_diag_v39_qpos_drop_guard import (
    deploy_policy as parent,
)


def _rack_image(model, obs) -> torch.Tensor:
    image = metric_parent._extract_image(obs, "cam_high", model.device)
    image = F.interpolate(
        image.unsqueeze(0),
        size=model.v82_rack_image_size,
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(
        (0.485, 0.456, 0.406), device=model.device
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        (0.229, 0.224, 0.225), device=model.device
    ).reshape(1, 3, 1, 1)
    return (image - mean) / std


def _aggregate_rack(samples: list[torch.Tensor]) -> torch.Tensor:
    stack = torch.stack(samples).to(torch.float32)
    result = stack.median(dim=0).values
    direction = F.normalize(stack[:, 2:4], dim=-1).mean(dim=0)
    result[2:4] = F.normalize(direction, dim=0)
    return result


def _rack_pose(current: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = current.detach().to(device="cpu", dtype=torch.float32).clone()
    result[0, 0, 3] = float(value[0])
    result[0, 1, 3] = float(value[1])
    direction = F.normalize(value[2:4].float(), dim=0)
    cosine = float(direction[0])
    sine = float(direction[1])
    result[0, :3, :3] = torch.tensor(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=torch.float32,
    )
    return result


@torch.inference_mode()
def _v82_estimate_poses(model, obs) -> None:
    parent._v39_estimate_poses(model, obs)
    current = model.estimated_poses.get("rack")
    if current is None:
        return
    if len(model.v82_rack_samples) < model.v82_rack_sample_frames:
        normalized = model.v82_rack_model(_rack_image(model, obs))[0].float()
        raw = normalized * model.v82_rack_target_std + model.v82_rack_target_mean
        if not torch.isfinite(raw).all():
            raise ValueError("V82 rack model produced NaN or Inf")
        model.v82_rack_samples.append(raw.detach().to(device="cpu"))
        model.v82_rack_vector = _aggregate_rack(model.v82_rack_samples)
        raw_yaw = math.degrees(math.atan2(float(raw[3]), float(raw[2])))
        agg_yaw = math.degrees(
            math.atan2(
                float(model.v82_rack_vector[3]),
                float(model.v82_rack_vector[2]),
            )
        )
        print(
            "[V82 HIGHRES RACK ONLY]"
            f" count={len(model.v82_rack_samples)}/{model.v82_rack_sample_frames}"
            f" raw_xy=({float(raw[0]):.5f},{float(raw[1]):.5f})"
            f" raw_yaw_deg={raw_yaw:.2f}"
            f" aggregate_xy=({float(model.v82_rack_vector[0]):.5f},"
            f"{float(model.v82_rack_vector[1]):.5f})"
            f" aggregate_yaw_deg={agg_yaw:.2f}",
            flush=True,
        )
    if model.v82_rack_vector is not None:
        model.estimated_poses["rack"] = _rack_pose(
            current, model.v82_rack_vector
        )


def _install_assets(model, config: dict[str, Any]) -> None:
    trainer = importlib.import_module("train_rack_highres_pose_v1")
    checkpoint_path = Path(
        config["rack_highres_checkpoint_path"]
    ).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("architecture") != "RackHighPoseNet":
        raise ValueError(
            "Expected RackHighPoseNet checkpoint, got "
            f"{checkpoint.get('architecture')!r}"
        )
    rack_model = trainer.RackHighPoseNet(
        keypoints=int(checkpoint["keypoints"])
    ).to(model.device)
    rack_model.load_state_dict(checkpoint["model_state"])
    rack_model.eval()
    model.v82_rack_model = rack_model
    model.v82_rack_target_mean = torch.as_tensor(
        checkpoint["target_mean"], dtype=torch.float32, device=model.device
    )
    model.v82_rack_target_std = torch.as_tensor(
        checkpoint["target_std"], dtype=torch.float32, device=model.device
    )
    model.v82_rack_image_size = tuple(map(int, checkpoint["image_size"]))
    model.v82_rack_checkpoint_path = str(checkpoint_path)
    model.v82_rack_sample_frames = int(config.get("v82_rack_sample_frames", 5))


def _install(model) -> None:
    metric_parent._estimate_poses = _v82_estimate_poses


def get_model(usr_args: dict[str, Any]):
    config = dict(usr_args or {})
    model = parent.get_model(config)
    _install_assets(model, config)
    model.v82_rack_samples = []
    model.v82_rack_vector = None
    _install(model)
    return model


def eval(env, model, obs):
    _install(model)
    return parent.parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    model.v82_rack_samples = []
    model.v82_rack_vector = None
    _install(model)


__all__ = ["eval", "get_model", "reset_model"]

