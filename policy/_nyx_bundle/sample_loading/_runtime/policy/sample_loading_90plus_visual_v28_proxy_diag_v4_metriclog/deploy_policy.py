"""V4 diagnostic: RGB/qpos V28 migration plus evaluator-returned metric logging.

This is intentionally not a submission package: it imports the preserved
historical controller and dataset by explicit development paths.  The proxy
prevents that controller from accessing the real cube/rack objects.  It exists
to measure the control loss when privileged poses are replaced by perception.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from policy.inference_timing import finish_inference, start_inference

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
PRIVILEGED_OBJECTS = frozenset(("cube", "rack"))


def _is_true(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


class EstimatedPoseObject:
    def __init__(self, owner: "DiagnosticModel", name: str):
        self.owner = owner
        self.name = name

    def get_local_pose(self, to_matrix: bool = True):
        if not to_matrix:
            raise RuntimeError("Diagnostic proxy exposes matrix pose only")
        pose = self.owner.estimated_poses.get(self.name)
        if pose is None:
            raise RuntimeError(f"Estimated pose unavailable for {self.name}")
        self.owner.privileged_proxy_reads[self.name] += 1
        return pose


class SimProxy:
    def __init__(self, owner: "DiagnosticModel", real_sim: Any):
        self.owner = owner
        self.real_sim = real_sim

    def get_rigid_object(self, name: str):
        if name in PRIVILEGED_OBJECTS:
            return EstimatedPoseObject(self.owner, name)
        return self.real_sim.get_rigid_object(name)

    def __getattr__(self, name: str):
        return getattr(self.real_sim, name)


class BaseProxy:
    def __init__(self, owner: "DiagnosticModel", real_base: Any):
        self.owner = owner
        self.real_base = real_base
        self._sim_proxy = SimProxy(owner, real_base.sim)

    @property
    def sim(self):
        return self._sim_proxy

    def __getattr__(self, name: str):
        return getattr(self.real_base, name)


class EnvProxy:
    def __init__(
        self,
        owner: "DiagnosticModel",
        real_env: Any,
        inference_started_at: float,
        inference_times_s: list[float],
    ):
        self.owner = owner
        self.real_env = real_env
        self._base_proxy = BaseProxy(owner, real_env.unwrapped)
        self.inference_started_at = inference_started_at
        self.inference_times_s = inference_times_s
        self.timing_finished = False

    @property
    def unwrapped(self):
        return self._base_proxy

    def finish_timing(self):
        if not self.timing_finished:
            finish_inference(
                self.inference_started_at,
                self.inference_times_s,
                self.owner.device,
            )
            self.timing_finished = True

    def step(self, action):
        self.finish_timing()
        result = self.real_env.step(action)
        self._log_official_metrics(result)
        return result

    @staticmethod
    def _scalar(value: Any):
        if value is None:
            return None
        tensor = torch.as_tensor(value).detach().reshape(-1)
        return tensor[0].item() if tensor.numel() else None

    def _log_official_metrics(self, result) -> None:
        """Log evaluator-returned metrics without feeding them to the policy."""
        if not isinstance(result, tuple) or len(result) < 5:
            return
        info = result[4]
        if not isinstance(info, dict):
            return
        metrics = info.get("metrics")
        if not isinstance(metrics, dict):
            return
        legacy = self.owner.legacy_model
        step = int(getattr(legacy, "step", 0)) + 1
        release_active = bool(getattr(legacy, "release_active", False))
        if not release_active and step % 20 != 0:
            return
        keys = (
            "cube_xy_dist", "cube_z", "rack_z", "cube_lin_vel_norm",
            "cube_vertical_angle", "bottom_z_diff",
            "cube_to_left_eef_dist", "cube_to_right_eef_dist",
            "left_gripper_q_mean", "right_gripper_q_mean",
            "placement_ok_single_frame", "place_stable_count",
        )
        fields = []
        for key in keys:
            if key in metrics:
                fields.append(f"{key}={self._scalar(metrics[key])}")
        print(f"[V4 OFFICIAL METRICS] step={step} " + " ".join(fields), flush=True)

    def __getattr__(self, name: str):
        return getattr(self.real_env, name)


@dataclass
class DiagnosticModel:
    perception: torch.nn.Module
    legacy_model: Any
    legacy_module: Any
    device: torch.device
    target_mean: torch.Tensor
    target_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor
    image_size: tuple[int, int]
    cube_alpha: float
    rack_alpha: float
    fk_anchor_window: int
    fk_visual_correction_alpha: float
    fk_visual_gate_m: float
    filtered_prediction: torch.Tensor | None = None
    estimated_poses: dict[str, torch.Tensor] | None = None
    privileged_proxy_reads: dict[str, int] | None = None
    inference_calls: int = 0
    left_object_offsets: list[torch.Tensor] | None = None
    left_object_offset: torch.Tensor | None = None
    fk_tracking_steps: int = 0

    def __post_init__(self):
        self.estimated_poses = {}
        self.privileged_proxy_reads = {"cube": 0, "rack": 0}


def _extend_development_import_paths(config: dict[str, Any]) -> None:
    history_policy = Path(config["diagnostic_history_policy_path"]).resolve()
    trainer_path = Path(config["diagnostic_trainer_path"]).resolve()
    if not history_policy.is_dir() or not trainer_path.is_dir():
        raise FileNotFoundError(
            f"Missing diagnostic dependency: {history_policy} or {trainer_path}"
        )
    import policy

    if str(history_policy) not in policy.__path__:
        policy.__path__.append(str(history_policy))
    if str(trainer_path) not in sys.path:
        sys.path.insert(0, str(trainer_path))


def get_model(usr_args: dict):
    config = dict(usr_args or {})
    _extend_development_import_paths(config)
    trainer = importlib.import_module("train_official_random_pose_regressor_v2")
    legacy_module = importlib.import_module(
        "policy.sample_loading_final_bundle_20260907.policies."
        "sample_loading_knn_ranked_v28.deploy_policy"
    )
    checkpoint_path = Path(config["checkpoint_path"]).expanduser().resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    device = torch.device(config.get("pytorch_device", "cuda"))
    perception = trainer.MultiViewSpatialPoseNet(
        keypoints=int(checkpoint["keypoints"])
    ).to(device)
    perception.load_state_dict(checkpoint["model_state"])
    perception.eval()
    return DiagnosticModel(
        perception=perception,
        legacy_model=legacy_module.get_model(None),
        legacy_module=legacy_module,
        device=device,
        target_mean=torch.as_tensor(
            checkpoint["target_mean"], dtype=torch.float32, device=device
        ),
        target_std=torch.as_tensor(
            checkpoint["target_std"], dtype=torch.float32, device=device
        ),
        state_mean=torch.as_tensor(
            checkpoint["state_mean"], dtype=torch.float32, device=device
        ),
        state_std=torch.as_tensor(
            checkpoint["state_std"], dtype=torch.float32, device=device
        ),
        image_size=tuple(map(int, checkpoint["image_size"])),
        cube_alpha=float(config.get("cube_pose_ema_alpha", 1.0)),
        rack_alpha=float(config.get("rack_pose_ema_alpha", 0.0)),
        fk_anchor_window=int(config.get("fk_anchor_window", 12)),
        fk_visual_correction_alpha=float(config.get("fk_visual_correction_alpha", 0.0)),
        fk_visual_gate_m=float(config.get("fk_visual_gate_m", 0.04)),
    )


def _extract_image(obs: dict[str, Any], camera: str, device: torch.device):
    value = obs["sensor"][camera]["color"]
    image = torch.as_tensor(value, device=device)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Unexpected {camera} shape {tuple(image.shape)}")
    if image.shape[-1] >= 3:
        image = image[..., :3].permute(2, 0, 1)
    elif image.shape[0] == 3:
        image = image
    else:
        raise ValueError(f"Unexpected {camera} channels {tuple(image.shape)}")
    image = image.to(torch.float32)
    if float(image.detach().amax().item()) > 1.5:
        image = image / 255.0
    return image


def _preprocess(model: DiagnosticModel, obs: dict[str, Any]):
    images = torch.stack(
        [_extract_image(obs, camera, model.device) for camera in CAMERAS]
    )
    images = F.interpolate(
        images,
        size=model.image_size,
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
    images = ((images - mean) / std).unsqueeze(0)
    state = torch.as_tensor(
        obs["robot"]["qpos"], dtype=torch.float32, device=model.device
    ).reshape(1, 14)
    state = (state - model.state_mean) / model.state_std
    return images, state


def _sixd_to_rotation(value: torch.Tensor) -> torch.Tensor:
    first = F.normalize(value[0:3], dim=0)
    second_raw = value[3:6]
    second = F.normalize(second_raw - first.dot(second_raw) * first, dim=0)
    third = torch.cross(first, second, dim=0)
    return torch.stack((first, second, third), dim=1)


def _axis_to_rotation(axis: torch.Tensor) -> torch.Tensor:
    third = F.normalize(axis, dim=0)
    helper = torch.tensor(
        (0.0, 0.0, 1.0) if abs(float(third[2])) < 0.9 else (1.0, 0.0, 0.0),
        dtype=third.dtype,
        device=third.device,
    )
    first = F.normalize(torch.cross(helper, third, dim=0), dim=0)
    second = torch.cross(third, first, dim=0)
    return torch.stack((first, second, third), dim=1)


def _pose(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Return the legacy controller's CPU pose representation.

    Perception executes on CUDA, while the official environment and historical
    IK controller expose robot kinematics as CPU tensors.  The proxy boundary
    performs the explicit transfer without exposing any simulator object state.
    """
    rotation_cpu = rotation.detach().to(device="cpu", dtype=torch.float32)
    translation_cpu = translation.detach().to(
        device="cpu", dtype=torch.float32
    )
    result = torch.eye(4, dtype=torch.float32).reshape(1, 4, 4)
    result[0, :3, :3] = rotation_cpu
    result[0, :3, 3] = translation_cpu
    return result


@torch.inference_mode()
def _estimate_poses(model: DiagnosticModel, obs: dict[str, Any]) -> None:
    images, state = _preprocess(model, obs)
    normalized = model.perception(images, state)[0].float()
    prediction = normalized * model.target_std + model.target_mean
    if model.filtered_prediction is None:
        filtered = prediction
    else:
        filtered = model.filtered_prediction.clone()
        filtered[0:6] = (
            model.cube_alpha * prediction[0:6]
            + (1.0 - model.cube_alpha) * filtered[0:6]
        )
        filtered[6:15] = (
            model.rack_alpha * prediction[6:15]
            + (1.0 - model.rack_alpha) * filtered[6:15]
        )
    model.filtered_prediction = filtered
    cube_translation = filtered[0:3]
    rack_translation = filtered[6:9]
    if not torch.isfinite(filtered).all():
        raise ValueError("Perception produced NaN or Inf")
    model.estimated_poses = {
        "cube": _pose(_axis_to_rotation(filtered[3:6]), cube_translation),
        "rack": _pose(_sixd_to_rotation(filtered[9:15]), rack_translation),
    }
    model.inference_calls += 1
def _left_eef_from_observation(env, obs: dict[str, Any]) -> torch.Tensor:
    """Compute the left EEF only from declared joint observations."""
    base = env.unwrapped
    qpos = torch.as_tensor(
        obs["robot"]["qpos"], dtype=torch.float32, device=base.device
    ).reshape(-1, 14)[:, :6]
    pose = base.robot.compute_fk(
        qpos, name="left_arm", to_matrix=True
    )
    return pose.detach().to(device="cpu", dtype=torch.float32)


def _robust_offset(offsets: list[torch.Tensor]) -> torch.Tensor:
    """Aggregate visual object-to-EEF transforms without privileged state."""
    stack = torch.cat(offsets, dim=0).to(dtype=torch.float32, device="cpu")
    result = offsets[-1].detach().to(
        dtype=torch.float32, device="cpu"
    ).clone()
    result[0, :3, 3] = stack[:, :3, 3].median(dim=0).values
    rotation_sum = stack[:, :3, :3].sum(dim=0)
    u, _singular, vh = torch.linalg.svd(rotation_sum)
    correction = torch.eye(3, dtype=torch.float32)
    correction[2, 2] = torch.sign(torch.det(u @ vh)).clamp(min=-1.0, max=1.0)
    if float(correction[2, 2]) == 0.0:
        correction[2, 2] = 1.0
    result[0, :3, :3] = u @ correction @ vh
    return result


def _apply_left_fk_anchor(
    model: DiagnosticModel, env, obs: dict[str, Any]
) -> None:
    """Use visual handoff anchors and observed-joint FK during placement."""
    legacy = model.legacy_model
    if not getattr(legacy, "initialized", False):
        model.left_object_offsets = []
        model.left_object_offset = None
        model.fk_tracking_steps = 0
        return
    cutoff = getattr(legacy, "cutoff", None)
    if cutoff is None:
        return
    cutoff = int(cutoff)
    frame = int(getattr(legacy, "trajectory_frame", 0))
    eef = _left_eef_from_observation(env, obs)
    visual_cube = model.estimated_poses["cube"]

    if frame < cutoff:
        model.left_object_offset = None
        model.fk_tracking_steps = 0
        raw_actions = getattr(legacy, "raw_actions", None)
        if (
            frame >= max(0, cutoff - model.fk_anchor_window)
            and raw_actions is not None
            and frame < len(raw_actions)
            and float(raw_actions[frame, 6]) <= 0.5
            and float(raw_actions[frame, 13]) > 0.5
        ):
            offset = torch.linalg.inv(eef) @ visual_cube
            if model.left_object_offsets is None:
                model.left_object_offsets = []
            model.left_object_offsets.append(offset.detach().clone())
            model.left_object_offsets = model.left_object_offsets[
                -model.fk_anchor_window:
            ]
        return

    if model.left_object_offset is None:
        offsets = model.left_object_offsets or []
        model.left_object_offset = (
            _robust_offset(offsets)
            if offsets
            else torch.linalg.inv(eef) @ visual_cube
        )
    tracked = eef @ model.left_object_offset
    residual = visual_cube[:, :3, 3] - tracked[:, :3, 3]
    residual_norm = float(torch.linalg.norm(residual).item())
    alpha = model.fk_visual_correction_alpha
    if alpha > 0.0 and residual_norm <= model.fk_visual_gate_m:
        tracked[:, :3, 3] += alpha * residual
        model.left_object_offset = torch.linalg.inv(eef) @ tracked
    model.estimated_poses["cube"] = tracked
    model.fk_tracking_steps += 1


def eval(env, model: DiagnosticModel, obs):
    inference_times_s: list[float] = []
    started_at = start_inference(model.device)
    proxy = None
    try:
        _estimate_poses(model, obs)
        _apply_left_fk_anchor(model, env, obs)
        proxy = EnvProxy(model, env, started_at, inference_times_s)
        result = model.legacy_module.eval(proxy, model.legacy_model, obs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("Legacy V28 did not return (obs, info, done)")
        final_obs, info, legacy_done = result
    finally:
        if proxy is None:
            finish_inference(started_at, inference_times_s, model.device)
        else:
            proxy.finish_timing()

    success = _is_true(env.get_wrapper_attr("is_task_success")())
    truncated = False if success else _is_true(legacy_done)
    return final_obs, info, truncated, inference_times_s


def reset_model(model: DiagnosticModel):
    model.legacy_module.reset_model(model.legacy_model)
    model.filtered_prediction = None
    model.estimated_poses = {}
    model.privileged_proxy_reads = {"cube": 0, "rack": 0}
    model.inference_calls = 0
    model.left_object_offsets = []
    model.left_object_offset = None
    model.fk_tracking_steps = 0


__all__ = [
    "BaseProxy", "DiagnosticModel", "EnvProxy", "EstimatedPoseObject",
    "SimProxy", "eval", "get_model", "reset_model",
]
