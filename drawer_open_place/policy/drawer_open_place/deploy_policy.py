"""Offline qpos-and-RGB ACT adapter for drawer_open_place.

The model reads joint/gripper position and three RGB observations. Legacy
qvel/qf checkpoint features are replaced by their frozen normalization means
without reading those observation fields. No object pose, contact, success, or
other simulator-only state is used to produce actions.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy

OBSERVATION_PATHS = {"observation.state": "robot/qpos"}


def start_inference(device=None):
    """Start CUDA-synchronized end-to-end policy timing."""
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def finish_inference(started_at, samples, device=None):
    """Finish CUDA-synchronized end-to-end policy timing."""
    if device is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    samples.append(time.perf_counter() - started_at)


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
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")
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

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    state_paths = dict(OBSERVATION_PATHS)
    state_paths.update(usr_args.get("state_observation_paths") or {})
    configured_inputs = set(policy.config.input_features)
    image_keys = list(policy.config.image_features)
    unsupported = configured_inputs - set(image_keys) - set(state_paths)

    neutralize = tuple(
        str(name) for name in usr_args.get("neutralize_state_features", [])
    )
    invalid = set(neutralize) - configured_inputs
    invalid.update(set(neutralize).intersection(image_keys))
    unsupported -= set(neutralize)
    if unsupported:
        raise ValueError(
            "Checkpoint requests unsupported observation inputs: "
            + ", ".join(sorted(unsupported))
        )
    if invalid:
        raise ValueError(
            "Invalid neutralized checkpoint inputs: " + ", ".join(sorted(invalid))
        )

    neutral_values = {}
    for feature_name in neutralize:
        buffer_name = "buffer_" + feature_name.replace(".", "_")
        buffer = getattr(policy.normalize_inputs, buffer_name)
        mean = buffer["mean"].detach().to(device=next(policy.parameters()).device)
        if not torch.isfinite(mean).all():
            raise ValueError(f"Non-finite checkpoint normalization mean: {feature_name}")
        neutral_values[feature_name] = mean

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 10))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map
    policy.state_observation_paths = state_paths
    policy.neutral_state_values = neutral_values
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.scale_gripper_01_to_env = bool(
        usr_args.get("scale_gripper_01_to_env", True)
    )
    policy.gripper_action_ids = tuple(
        int(index) for index in usr_args.get("gripper_action_ids", [6, 13])
    )
    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")
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
            elif feature_name in model.neutral_state_values:
                batch[feature_name] = model.neutral_state_values[feature_name].unsqueeze(0)
            else:
                obs_path = model.state_observation_paths[feature_name]
                batch[feature_name] = _as_batch_tensor(
                    _read_path(final_obs, obs_path), model.act_device
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
                f"Policy action has dim {policy_action_dim}, "
                f"but env expects {env_action_dim}."
            )
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        if model.scale_gripper_01_to_env:
            action = action.clone()
            low = torch.as_tensor(
                env.unwrapped.single_action_space.low,
                device=action.device,
                dtype=action.dtype,
            )
            high = torch.as_tensor(
                env.unwrapped.single_action_space.high,
                device=action.device,
                dtype=action.dtype,
            )
            ids = list(model.gripper_action_ids)
            action[:, ids] = low[ids] + action[:, ids].clamp(0.0, 1.0) * (
                high[ids] - low[ids]
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
