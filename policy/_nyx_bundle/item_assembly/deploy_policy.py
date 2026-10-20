"""Fixed observation-only ACT policy for RoboSynChallenge item_assembly.

The policy uses three public RGB streams plus robot qpos/qvel/qf.  A fixed,
bounded qpos-conditioned residual is added to the twelve arm joints only.
No object pose, contact signal, IK target, simulator state, or task metric is
used for action generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from .inference_timing import finish_inference, start_inference


BUNDLE_DIR = Path(__file__).resolve().parent
MODEL_SHA256 = "ff69fa981b78bb55a8ad7e82fccdd24b67bd9938aabf1b37959ca4e192edbd0c"
RESIDUAL_SHA256 = "d79b8abfb5e7556d804165d11a0e1950fe2d687ead96107c4a127f4e30e0b611"
OUTPUT_CHUNK_SIZE = 50
EXECUTION_CHUNK_SIZE = 20
RESIDUAL_CAP_RAD = 0.02
ARM_ACTION_IDS = tuple(index for index in range(14) if index not in (6, 13))
GRIPPER_ACTION_IDS = (6, 13)

OBSERVATION_PATHS = {
    "observation.state": "robot/qpos",
    "observation.qvel": "robot/qvel",
    "observation.qf": "robot/qf",
}
DEFAULT_IMAGE_KEY_MAP = {
    "observation.images.cam_high": "cam_high",
    "observation.images.cam_right_wrist": "cam_right_wrist",
    "observation.images.cam_left_wrist": "cam_left_wrist",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_path(observation, path):
    value = observation
    for key in path.split("/"):
        if key:
            value = value[key]
    return value


def _as_batch_tensor(value, device):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor


def _done(value):
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def _load_residual(path: Path):
    fingerprint = _sha256(path)
    if fingerprint != RESIDUAL_SHA256:
        raise ValueError(
            f"Residual hash mismatch: expected {RESIDUAL_SHA256}, got {fingerprint}"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    theta = np.asarray(data["theta"], dtype=np.float64)
    mean = np.asarray(data["qpos_mean"], dtype=np.float64)
    std = np.asarray(data["qpos_std"], dtype=np.float64)
    if (
        data.get("schema_version") != 1
        or data.get("base_model_sha256") != MODEL_SHA256
        or data.get("cap_rad") != RESIDUAL_CAP_RAD
    ):
        raise ValueError("Unexpected residual specification")
    if theta.shape != (12, 15) or not np.isfinite(theta).all() or (np.abs(theta) > 1).any():
        raise ValueError("Invalid residual theta")
    if (
        mean.shape != (14,)
        or std.shape != (14,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or (std <= 0).any()
    ):
        raise ValueError("Invalid qpos normalization")
    return fingerprint, theta, mean, std


def _install_execution_horizon(policy):
    if int(policy.config.chunk_size) != OUTPUT_CHUNK_SIZE:
        raise ValueError(
            f"Expected output chunk {OUTPUT_CHUNK_SIZE}, got {policy.config.chunk_size}"
        )
    if EXECUTION_CHUNK_SIZE >= OUTPUT_CHUNK_SIZE:
        raise ValueError("execution_chunk_size must be smaller than output_chunk_size")
    if policy.config.temporal_ensemble_coeff is not None:
        raise ValueError("This fixed policy requires temporal ensembling disabled")
    policy.config.n_action_steps = EXECUTION_CHUNK_SIZE
    policy.reset()
    if policy._action_queue.maxlen != EXECUTION_CHUNK_SIZE or policy._action_queue:
        raise RuntimeError("ACT did not create the expected empty action queue")


def _install_residual(policy, residual_path: Path):
    fingerprint, theta, mean, std = _load_residual(residual_path)
    device = policy.act_device
    weights = torch.as_tensor(theta, dtype=torch.float32, device=device)
    mean_tensor = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_tensor = torch.as_tensor(std, dtype=torch.float32, device=device)
    original_predict = policy.predict_action_chunk

    def predict_with_residual(batch):
        chunk = original_predict(batch)
        qpos = batch["observation.state"]
        if qpos.shape != (1, 14) or chunk.shape != (1, OUTPUT_CHUNK_SIZE, 14):
            raise ValueError(
                "Fixed policy requires qpos [1,14] and ACT chunk [1,50,14]"
            )
        features = torch.cat(
            [
                ((qpos - mean_tensor) / std_tensor).clamp(-3, 3) / 3,
                torch.ones_like(qpos[:, :1]),
            ],
            dim=-1,
        )
        delta = RESIDUAL_CAP_RAD * torch.tanh(features @ weights.T)
        if not bool(torch.isfinite(delta).all()):
            raise ValueError("Residual produced non-finite values")
        result = chunk.clone()
        result[:, :, ARM_ACTION_IDS] += delta.unsqueeze(1)
        maximum = float((result - chunk).abs().max())
        if maximum > RESIDUAL_CAP_RAD + 1e-6:
            raise RuntimeError("Residual exceeded its fixed bound")
        if not torch.equal(
            result[:, :, GRIPPER_ACTION_IDS], chunk[:, :, GRIPPER_ACTION_IDS]
        ):
            raise RuntimeError("Residual modified a gripper action")
        return result

    policy.predict_action_chunk = predict_with_residual
    policy.item_assembly_residual_sha256 = fingerprint


def get_model(usr_args):
    task_name = usr_args.get("task_name", "item_assembly")
    if task_name != "item_assembly":
        raise ValueError("item_assembly_1 only supports the item_assembly task")

    supplied_checkpoint = usr_args.get("checkpoint_path")
    checkpoint_path = Path(supplied_checkpoint) if supplied_checkpoint else BUNDLE_DIR / "checkpoint"
    checkpoint_path = checkpoint_path.expanduser().resolve()
    model_file = checkpoint_path / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing bundled ACT model: {model_file}")
    model_hash = _sha256(model_file)
    if model_hash != MODEL_SHA256:
        raise ValueError(f"Checkpoint hash mismatch: expected {MODEL_SHA256}, got {model_hash}")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    try:
        policy = ACTPolicy.from_pretrained(checkpoint_path, cli_overrides=cli_overrides)
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path, cli_overrides=cli_overrides
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 10))
    if policy.act_step <= 0:
        raise ValueError("act_step must be positive")
    policy.act_image_keys = list(policy.config.image_features)
    policy.image_key_map = dict(DEFAULT_IMAGE_KEY_MAP)
    policy.image_key_map.update(usr_args.get("image_key_map") or {})
    policy.state_observation_paths = dict(OBSERVATION_PATHS)
    policy.state_observation_paths.update(usr_args.get("state_observation_paths") or {})
    configured_inputs = set(policy.config.input_features)
    unsupported = configured_inputs - set(policy.act_image_keys) - set(policy.state_observation_paths)
    if unsupported:
        raise ValueError(
            "Checkpoint requests unsupported/non-public inputs: "
            + ", ".join(sorted(unsupported))
        )
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.scale_gripper_01_to_env = bool(
        usr_args.get("scale_gripper_01_to_env", True)
    )
    policy.gripper_action_ids = GRIPPER_ACTION_IDS

    _install_execution_horizon(policy)
    _install_residual(policy, BUNDLE_DIR / "residual.json")
    print(
        "[item_assembly_1] "
        f"checkpoint_sha256={model_hash} residual_sha256={policy.item_assembly_residual_sha256} "
        f"output_chunk={OUTPUT_CHUNK_SIZE} execution_chunk={EXECUTION_CHUNK_SIZE} "
        f"residual_cap_rad={RESIDUAL_CAP_RAD}",
        flush=True,
    )
    return policy


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        runs_model_inference = (
            model.config.temporal_ensemble_coeff is not None
            or len(model._action_queue) == 0
        )
        started_at = start_inference(model.act_device) if runs_model_inference else None

        batch = {}
        for feature_name in model.config.input_features:
            if feature_name in model.act_image_keys:
                camera_name = model.image_key_map.get(
                    feature_name,
                    feature_name.removeprefix("observation.images."),
                )
                image = final_obs["sensor"][camera_name]["color"]
                image_tensor = _as_batch_tensor(image, model.act_device)
                image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
                if torch.max(image_tensor) > 1.5:
                    image_tensor = image_tensor / 255.0
                batch[feature_name] = image_tensor
            else:
                batch[feature_name] = _as_batch_tensor(
                    _read_path(final_obs, model.state_observation_paths[feature_name]),
                    model.act_device,
                )

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action [B,D], got {tuple(action.shape)}")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action dim {policy_action_dim}; environment expects {env_action_dim}"
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        if model.scale_gripper_01_to_env:
            action = action.clone()
            action_low = torch.as_tensor(
                env.unwrapped.single_action_space.low,
                device=action.device,
                dtype=action.dtype,
            )
            action_high = torch.as_tensor(
                env.unwrapped.single_action_space.high,
                device=action.device,
                dtype=action.dtype,
            )
            gripper_ids = list(model.gripper_action_ids)
            action[:, gripper_ids] = (
                action_low[gripper_ids]
                + action[:, gripper_ids].clamp(0.0, 1.0)
                * (action_high[gripper_ids] - action_low[gripper_ids])
            )

        action_tensor = action.detach().to(
            device=env.unwrapped.device, dtype=torch.float32
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        final_obs, _, terminated, truncated, info = env.step(action_tensor)

        if _done(env.get_wrapper_attr("is_task_success")()):
            break
        if _done(terminated) or _done(truncated):
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()


__all__ = ["get_model", "eval", "reset_model"]
