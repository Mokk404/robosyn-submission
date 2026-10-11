"""Fixed observation-only ACT policy for RoboSynChallenge water_pouring.

This is the packaged form of spatial-translation checkpoint 000200. Action
generation uses only three public RGB observations and robot qpos/qvel/qf.
No simulator object pose, contact state, task metric, or success label is used
to produce actions.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from .inference_timing import finish_inference, start_inference


BUNDLE_DIR = Path(__file__).resolve().parent
MODEL_SHA256 = "acfa8d21e852b9006849a7edccdf8c7d692fe0117ab1213c50b01fb105eec406"
OUTPUT_CHUNK_SIZE = 50
EXECUTION_CHUNK_SIZE = 50
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


def get_model(usr_args):
    if usr_args.get("task_name", "water_pouring") != "water_pouring":
        raise ValueError("This package only supports water_pouring.")

    supplied = usr_args.get("checkpoint_path")
    checkpoint_path = Path(supplied) if supplied else BUNDLE_DIR / "checkpoint"
    checkpoint_path = checkpoint_path.expanduser().resolve()
    model_file = checkpoint_path / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"Missing bundled ACT model: {model_file}")
    model_hash = _sha256(model_file)
    if model_hash != MODEL_SHA256:
        raise ValueError(
            f"Checkpoint hash mismatch: expected {MODEL_SHA256}, got {model_hash}"
        )

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

    if int(policy.config.chunk_size) != OUTPUT_CHUNK_SIZE:
        raise ValueError(
            f"Expected chunk_size={OUTPUT_CHUNK_SIZE}, got {policy.config.chunk_size}"
        )
    if policy.config.temporal_ensemble_coeff is not None:
        raise ValueError("This fixed policy requires temporal ensembling disabled.")
    policy.config.n_action_steps = EXECUTION_CHUNK_SIZE
    policy.reset()
    if policy._action_queue.maxlen != EXECUTION_CHUNK_SIZE or policy._action_queue:
        raise RuntimeError("ACT did not create the expected empty action queue.")

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 10))
    if policy.act_step <= 0:
        raise ValueError("act_step must be positive.")
    policy.act_image_keys = list(policy.config.image_features)
    policy.image_key_map = dict(DEFAULT_IMAGE_KEY_MAP)
    policy.image_key_map.update(usr_args.get("image_key_map") or {})
    policy.state_observation_paths = dict(OBSERVATION_PATHS)
    policy.state_observation_paths.update(
        usr_args.get("state_observation_paths") or {}
    )
    configured_inputs = set(policy.config.input_features)
    unsupported = configured_inputs - set(policy.act_image_keys) - set(
        policy.state_observation_paths
    )
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
    print(
        "[water_pouring] "
        f"checkpoint_sha256={model_hash} output_chunk={OUTPUT_CHUNK_SIZE} "
        f"execution_chunk={EXECUTION_CHUNK_SIZE}",
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
                    feature_name, feature_name.removeprefix("observation.images.")
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
            message = (
                f"Policy action dim {policy_action_dim}; environment expects {env_action_dim}"
            )
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
