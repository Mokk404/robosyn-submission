"""V33 diagnostic: V32 plus an aligned guided-release deadline.

The dynamic RGB model is sampled at three predeclared mid-trajectory frames
without changing the replay target. If its median tube height shows that pickup
was missed, V27 action-level regrasp is enabled with the visual proxy pose.
Placement prevents millimetre-scale Z deadlocks from consuming the step budget.
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
from policy.sample_loading_90plus_visual_v28_proxy_diag_v12_guided_release import (
    deploy_policy as parent,
)
from policy.sample_loading_90plus_visual_v28_proxy_diag_v12_guided_release import (
    deploy_policy as release_parent,
)


CUBE_SPECIALIST_FRAMES = 1


def _v33_guided_release(base, legacy, cube_pose, rack_pose, slot_xy, angle):
    if not hasattr(legacy, "v12_phase"):
        legacy.v12_phase = "descent"
        legacy.v12_phase_steps = 0

    if legacy.v12_phase == "descent":
        phase_step = int(legacy.v12_phase_steps)
        cube_z = float(cube_pose[0, 2, 3].item())
        rack_z = float(rack_pose[0, 2, 3].item())
        release_gate_z = rack_z + release_parent.RELEASE_GATE_ABOVE_RACK_M
        slot_value = float(slot_xy.item())
        angle_value = abs(float(angle.item()))
        timed_out = phase_step >= int(legacy.v33_descent_timeout_steps)
        near_gate = (
            cube_z
            <= release_gate_z + float(legacy.v33_extra_z_tolerance_m)
        )
        aligned = (
            slot_value <= float(legacy.v33_slot_gate_m)
            and angle_value <= float(legacy.v33_angle_gate_deg)
        )
        if timed_out and near_gate and aligned:
            print(
                "[V33 RELEASE DEADLINE]"
                f" forcing seat after descent_steps={phase_step}"
                f" z_error_mm={(cube_z - release_gate_z) * 1000.0:.2f}"
                f" slot_xy={slot_value:.5f} angle={angle_value:.2f}",
                flush=True,
            )
            legacy.v12_phase = "seat"
            legacy.v12_phase_steps = 0

    return release_parent._v12_guided_release(
        base, legacy, cube_pose, rack_pose, slot_xy, angle
    )


def _rack_image(model, obs) -> torch.Tensor:
    image = metric_parent._extract_image(obs, "cam_high", model.device)
    image = F.interpolate(
        image.unsqueeze(0),
        size=model.v17_rack_image_size,
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


def _rack_pose_from_vector(value: torch.Tensor) -> torch.Tensor:
    translation = value[0:3]
    axis = F.normalize(value[3:6], dim=0)
    return metric_parent._pose(
        metric_parent._axis_to_rotation(axis), translation
    )


def _aggregate(samples: list[torch.Tensor]) -> torch.Tensor:
    stack = torch.stack(samples)
    result = stack.median(dim=0).values
    axis = F.normalize(stack[:, 3:6], dim=-1).mean(dim=0)
    result[3:6] = F.normalize(axis, dim=0)
    return result


@torch.inference_mode()
def _v17_estimate_poses(model, obs) -> None:
    model.v17_original_estimate_poses(model, obs)
    samples = model.v17_rack_samples
    if len(samples) < CUBE_SPECIALIST_FRAMES:
        normalized = model.v17_rack_perception(_rack_image(model, obs))[0].float()
        prediction = (
            normalized * model.v17_rack_target_std
            + model.v17_rack_target_mean
        )
        if not torch.isfinite(prediction).all():
            raise ValueError("V29 cube specialist produced NaN or Inf")
        samples.append(prediction.detach().to(device="cpu"))
        aggregate = _aggregate(samples)
        model.v17_rack_vector = aggregate
        model.v17_rack_pose = _rack_pose_from_vector(
            aggregate.to(device=model.device)
        )
        count = len(samples)
        if count <= CUBE_SPECIALIST_FRAMES:
            xy = torch.stack(samples)[:, :2]
            center = xy.median(dim=0).values
            p90_mm = float(
                torch.quantile(
                    torch.linalg.norm(xy - center, dim=-1), 0.90
                ) * 1000.0
            )
            axis_deg = math.degrees(
                math.acos(max(-1.0, min(1.0, abs(float(aggregate[5])))))
            )
            print(
                "[V29 HIGHRES INITIAL CUBE]"
                f" count={count}/{CUBE_SPECIALIST_FRAMES}"
                f" xy=({float(aggregate[0]):.5f},{float(aggregate[1]):.5f})"
                f" z={float(aggregate[2]):.5f}"
                f" axis_deg={axis_deg:.2f} p90_mm={p90_mm:.2f}",
                flush=True,
            )
    model.estimated_poses["cube"] = model.v17_rack_pose.detach().clone()


def _install_rack_specialist(model, config: dict[str, Any]) -> None:
    trainer = importlib.import_module("train_cube_highres_pose_v1")
    checkpoint_path = Path(
        config["cube_highres_checkpoint_path"]
    ).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    rack_model = trainer.CubeHighPoseNet(
        keypoints=int(checkpoint["keypoints"])
    ).to(model.device)
    rack_model.load_state_dict(checkpoint["model_state"])
    rack_model.eval()
    model.v17_rack_perception = rack_model
    model.v17_rack_target_mean = torch.as_tensor(
        checkpoint["target_mean"], dtype=torch.float32, device=model.device
    )
    model.v17_rack_target_std = torch.as_tensor(
        checkpoint["target_std"], dtype=torch.float32, device=model.device
    )
    model.v17_rack_image_size = tuple(map(int, checkpoint["image_size"]))
    model.v17_rack_checkpoint_path = str(checkpoint_path)
    model.v17_rack_samples = []
    model.v17_rack_vector = None
    model.v17_rack_pose = None

    original = getattr(
        metric_parent,
        "_v17_original_estimate_poses",
        metric_parent._estimate_poses,
    )
    metric_parent._v17_original_estimate_poses = original
    metric_parent._estimate_poses = _v17_estimate_poses
    model.v17_original_estimate_poses = original


@torch.inference_mode()
def _v31_estimate_poses(model, obs) -> None:
    """Freeze initialization until the handoff anchor window."""
    if model.v30_first_frame:
        model.perception = model.v30_initial_perception
        _v17_estimate_poses(model, obs)
        model.v30_first_frame = False
        model.perception = model.v30_dynamic_perception
        print(
            "[V31 GATED RGB] initialized with legacy rack and highres cube; "
            "dynamic perception is gated until the handoff anchor window",
            flush=True,
        )
        return

    legacy = model.legacy_model
    initialized = bool(getattr(legacy, "initialized", False))
    cutoff = getattr(legacy, "cutoff", None)
    frame = int(getattr(legacy, "trajectory_frame", 0))
    anchor_start = (
        max(0, int(cutoff) - int(model.fk_anchor_window))
        if cutoff is not None
        else None
    )
    probe_frames = model.v32_pickup_probe_frames
    if (
        initialized
        and frame in probe_frames
        and frame not in model.v32_pickup_probe_done
    ):
        images, state = metric_parent._preprocess(model, obs)
        normalized = model.v30_dynamic_perception(images, state)[0].float()
        prediction = normalized * model.target_std + model.target_mean
        if not torch.isfinite(prediction).all():
            raise ValueError("V32 pickup probe produced NaN or Inf")
        cube_z = float(prediction[2].item())
        model.v32_pickup_probe_done.add(frame)
        model.v32_pickup_probe_zs.append(cube_z)
        print(
            f"[V32 RGB PICKUP PROBE] frame={frame} cube_z={cube_z:.5f}"
            f" samples={len(model.v32_pickup_probe_zs)}/{len(probe_frames)}",
            flush=True,
        )
        if (
            len(model.v32_pickup_probe_zs) == len(probe_frames)
            and not model.v32_pickup_decided
        ):
            median_z = float(torch.tensor(model.v32_pickup_probe_zs).median())
            missed = median_z < model.v32_pickup_z_threshold_m
            model.v32_pickup_decided = True
            model.v32_recovery_enabled = missed
            if missed:
                legacy.recovery_attempted = False
            print(
                f"[V32 RGB PICKUP DECISION] median_z={median_z:.5f}"
                f" threshold={model.v32_pickup_z_threshold_m:.5f}"
                f" missed={missed} recovery_enabled={missed}",
                flush=True,
            )
    if not initialized or anchor_start is None or frame < anchor_start:
        model.estimated_poses["cube"] = model.v17_rack_pose.detach().clone()
        return

    model.v17_original_estimate_poses(model, obs)
    if not model.v31_dynamic_announced:
        print(
            f"[V31 GATED RGB] dynamic perception active at frame={frame} "
            f"cutoff={int(cutoff)} anchor_start={anchor_start}",
            flush=True,
        )
        model.v31_dynamic_announced = True


def _install_dynamic_perception(model, config: dict[str, Any]) -> None:
    trainer = importlib.import_module("train_dynamic_pose_rgb_only_v3")
    checkpoint_path = Path(
        config["rgb_dynamic_checkpoint_path"]
    ).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("architecture") != "MultiViewSpatialPoseNetRGBOnly":
        raise ValueError(
            "Expected MultiViewSpatialPoseNetRGBOnly checkpoint, got "
            f"{checkpoint.get('architecture')!r}"
        )
    dynamic = trainer.MultiViewSpatialPoseNet(
        keypoints=int(checkpoint["keypoints"])
    ).to(model.device)
    dynamic.load_state_dict(checkpoint["model_state"])
    dynamic.eval()

    model.v30_initial_perception = model.perception
    model.v30_dynamic_perception = dynamic
    model.v30_dynamic_checkpoint_path = str(checkpoint_path)
    model.v30_first_frame = True
    model.v31_dynamic_announced = False
    model.v32_pickup_probe_frames = tuple(
        int(value) for value in config.get(
            "pickup_probe_frames", (130, 135, 140)
        )
    )
    model.v32_pickup_z_threshold_m = float(
        config.get("pickup_probe_z_threshold_m", 0.90)
    )
    model.v32_pickup_probe_done = set()
    model.v32_pickup_probe_zs = []
    model.v32_pickup_decided = False
    model.v32_recovery_enabled = False
    model.target_mean = torch.as_tensor(
        checkpoint["target_mean"], dtype=torch.float32, device=model.device
    )
    model.target_std = torch.as_tensor(
        checkpoint["target_std"], dtype=torch.float32, device=model.device
    )
    model.state_mean = torch.as_tensor(
        checkpoint["state_mean"], dtype=torch.float32, device=model.device
    )
    model.state_std = torch.as_tensor(
        checkpoint["state_std"], dtype=torch.float32, device=model.device
    )
    model.image_size = tuple(map(int, checkpoint["image_size"]))
    metric_parent._estimate_poses = _v31_estimate_poses


def _install_release_deadline(
    model, config: dict[str, Any] | None = None
) -> None:
    if config is not None:
        model.v33_descent_timeout_steps = int(
            config.get("guided_release_descent_timeout_steps", 24)
        )
        model.v33_extra_z_tolerance_m = float(
            config.get("guided_release_extra_z_tolerance_m", 0.010)
        )
        model.v33_slot_gate_m = float(
            config.get("guided_release_slot_gate_m", 0.015)
        )
        model.v33_angle_gate_deg = float(
            config.get("guided_release_angle_gate_deg", 10.0)
        )
    legacy = model.legacy_model
    legacy.v33_descent_timeout_steps = model.v33_descent_timeout_steps
    legacy.v33_extra_z_tolerance_m = model.v33_extra_z_tolerance_m
    legacy.v33_slot_gate_m = model.v33_slot_gate_m
    legacy.v33_angle_gate_deg = model.v33_angle_gate_deg
    model.legacy_module.implementation._guided_wide = _v33_guided_release


def get_model(usr_args: dict[str, Any]):
    config = dict(usr_args or {})
    model = parent.get_model(config)
    _install_rack_specialist(model, config)
    _install_dynamic_perception(model, config)
    _install_release_deadline(model, config)
    return model


def eval(env, model, obs):
    return parent.eval(env, model, obs)


def reset_model(model):
    parent.reset_model(model)
    model.v17_rack_samples = []
    model.v17_rack_vector = None
    model.v17_rack_pose = None
    model.v30_first_frame = True
    model.v31_dynamic_announced = False
    model.perception = model.v30_initial_perception
    model.v32_pickup_probe_done = set()
    model.v32_pickup_probe_zs = []
    model.v32_pickup_decided = False
    model.v32_recovery_enabled = False
    _install_release_deadline(model)
    metric_parent._estimate_poses = _v31_estimate_poses


__all__ = ["eval", "get_model", "reset_model"]
