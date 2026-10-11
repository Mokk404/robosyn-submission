"""RGB and robot qpos ACT baseline; complete action selection timing."""

from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
import torch.nn.functional as F
from .pose_translation_model import TranslationPoseNet
from . import rbf_impl

def _read_path(observation, path):
    value = observation
    for key in str(path).split("/"):
        if key:
            value = value[key]
    return value

def _as_batch_tensor(value, device):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor

def _done(value):
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)

def _clear_action_queue(model):
    queue = getattr(model, "_action_queue", None)
    if queue is not None:
        queue.clear()

def _map_action_to_environment(env, model, action):
    if action.ndim == 1:
        action = action.unsqueeze(0)
    if action.ndim != 2:
        raise ValueError(f"Expected [B,D] action, got {tuple(action.shape)}")
    action_space = env.unwrapped.single_action_space
    env_dim = int(np.prod(action_space.shape))
    policy_dim = int(action.shape[-1])
    if policy_dim != env_dim:
        message = f"Policy action dim {policy_dim}; environment action dim {env_dim}"
        if model.strict_action_dim or policy_dim < env_dim:
            raise ValueError(message)
        action = action[:, :env_dim]
    if model.scale_gripper_01_to_env:
        action = action.clone()
        low = torch.as_tensor(
            action_space.low, device=action.device, dtype=action.dtype
        )
        high = torch.as_tensor(
            action_space.high, device=action.device, dtype=action.dtype
        )
        ids = list(model.gripper_action_ids)
        action[:, ids] = low[ids] + action[:, ids].clamp(0.0, 1.0) * (
            high[ids] - low[ids]
        )
    return action.detach().to(
        device=env.unwrapped.device, dtype=torch.float32
    )

def _public_array(observation, path):
    value = _read_path(observation, path)
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value).tolist()

def reset_model(model):
    model.reset()
    model.aux_model.reset()
    model.public_action_step = 0
    model.gripper_action_queue.clear()
    model.translation_plan = None
    model.predicted_translation = None
    model.episode_index += 1
    if model.public_trace_handle is not None:
        model.public_trace_handle.close()
        model.public_trace_handle = None
    if model.public_trace_root:
        path = model.public_trace_root / f"episode_{model.episode_index:03d}.jsonl"
        model.public_trace_handle = path.open("x", encoding="utf-8")


CAMERAS = ('cam_high', 'cam_right_wrist', 'cam_left_wrist')
FEATURES = {'observation.state'} | {'observation.images.' + x for x in CAMERAS}

def get_model(usr_args):
    device = usr_args.get('pytorch_device', 'cuda')
    config = PreTrainedConfig.from_pretrained(usr_args['checkpoint_path'], cli_overrides=[f'--device={device}'])
    config.temporal_ensemble_coeff = float(usr_args.get('temporal_ensemble_coeff', 0.01))
    # Legacy checkpoint has unused normalizer entries. No corresponding observation is read.
    if set(config.input_features) - FEATURES - {'observation.qvel', 'observation.qf'}:
        raise ValueError('Checkpoint requests inputs outside the RGB/qpos whitelist')
    if not FEATURES <= set(config.input_features):
        raise ValueError('Three RGB streams and robot qpos are required')
    model = ACTPolicy.from_pretrained(usr_args['checkpoint_path'], config=config)
    model.eval()
    auxiliary_path = usr_args.get('auxiliary_checkpoint_path')
    if not auxiliary_path:
        raise ValueError('auxiliary_checkpoint_path is required')
    auxiliary_config = PreTrainedConfig.from_pretrained(
        auxiliary_path, cli_overrides=[f'--device={device}']
    )
    auxiliary_config.temporal_ensemble_coeff = None
    if set(auxiliary_config.input_features) - FEATURES - {'observation.qvel', 'observation.qf'}:
        raise ValueError('Auxiliary checkpoint requests inputs outside the RGB/qpos whitelist')
    model.aux_model = ACTPolicy.from_pretrained(auxiliary_path, config=auxiliary_config)
    model.aux_model.eval()
    model.ensemble_alpha = float(usr_args.get('ensemble_alpha', 0.5))
    model.act_device = next(model.parameters()).device
    model.act_step = int(usr_args.get('act_step', 10))
    model.force_replan = bool(usr_args.get('force_replan', False))
    model.strict_action_dim = True
    model.scale_gripper_01_to_env = True
    model.gripper_action_ids = (6, 13)
    model.public_trace_root = Path(usr_args['public_trace_root']) if usr_args.get('public_trace_root') else None
    if model.public_trace_root:
        model.public_trace_root.mkdir(parents=True, exist_ok=False)
    model.public_trace_handle = None
    model.episode_index = -1
    model.public_action_step = 0
    model.gripper_action_queue = __import__('collections').deque()
    schedule_path = Path(__file__).resolve().parent / 'assets/gripper_schedule.npy'
    model.gripper_schedule = torch.as_tensor(
        np.load(schedule_path, allow_pickle=False),
        device=model.act_device,
        dtype=torch.float32,
    )
    assets = Path(__file__).resolve().parent / 'assets'
    model.translation_estimator = TranslationPoseNet().to(model.act_device)
    model.translation_estimator.load_state_dict(torch.load(
        assets / 'translation_pose.pt', map_location=model.act_device, weights_only=True
    ))
    model.translation_estimator.eval()
    normal = np.load(assets / 'translation_normalizers.npz', allow_pickle=False)
    model.translation_stats = {
        key: torch.as_tensor(normal[key], device=model.act_device, dtype=torch.float32)
        for key in normal.files
    }
    rbf_args = dict(usr_args)
    rbf_args['artifact_path'] = str(assets / 'translation_robust_rbf_v1.npz')
    rbf_args['public_trace_root'] = None
    model.translation_rbf = rbf_impl.get_model(rbf_args)
    model.translation_plan = None
    model.predicted_translation = None
    model.translation_blend = float(usr_args.get('translation_blend', 0.12))
    model.translation_blend_steps = int(usr_args.get('translation_blend_steps', 150))
    return model

def _encode_observation(model, observation):
    batch = {'observation.state': _as_batch_tensor(observation['robot']['qpos'], model.act_device)}
    for camera in CAMERAS:
        raw = observation['sensor'][camera]['color']
        tensor = _as_batch_tensor(raw, model.act_device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor[..., :3].permute(0, 3, 1, 2).contiguous()
        if torch.max(tensor) > 1.5:
            tensor = tensor / 255.0
        batch['observation.images.' + camera] = tensor
    return batch

def _sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)

def eval(env, model, obs):
    current, info, truncated = obs, None, False
    inference_times = []
    for local_step in range(model.act_step):
        _sync(model.act_device)
        started = time.perf_counter()
        if local_step == 0 and model.force_replan:
            _clear_action_queue(model)
        did_forward = model.config.temporal_ensemble_coeff is not None or len(model._action_queue) == 0
        batch = _encode_observation(model, current)
        _sync(model.act_device)
        model_started = time.perf_counter()
        # Smooth arm joints with temporal ensembling. For grippers, preserve
        # ACT's native ten-step action queue, because averaging open/close
        # transitions across chunks shifts contact timing.
        primary_chunk = model.predict_action_chunk(batch)
        auxiliary_chunk = model.aux_model.predict_action_chunk(batch)
        alpha = model.ensemble_alpha
        action_chunk = primary_chunk.lerp(auxiliary_chunk, alpha)
        action = model.temporal_ensembler.update(action_chunk).clone()
        if model.translation_plan is None:
            images = torch.stack([
                F.interpolate(
                    batch['observation.images.' + camera], size=(240, 320),
                    mode='bilinear', align_corners=False, antialias=True,
                ) for camera in CAMERAS
            ], dim=1)
            stats = model.translation_stats
            qnorm = (batch['observation.state'] - stats['qpos_mean']) / stats['qpos_std']
            raw = model.translation_estimator(images, qnorm)
            translation = raw * stats['target_std'] + stats['target_mean']
            matrices = []
            for offset in (0, 3):
                pose = torch.eye(4, device=model.act_device).unsqueeze(0).repeat(translation.shape[0], 1, 1)
                pose[:, :3, 3] = translation[:, offset:offset+3]
                matrices.append(pose)
            proxy = {'robot': {'qpos': current['robot']['qpos']},
                     'pen_pose': matrices[0], 'holder_pose': matrices[1]}
            model.translation_plan = rbf_impl._predict_plan(model.translation_rbf, proxy)
            model.predicted_translation = translation.detach().cpu().tolist()
        if model.public_action_step < len(model.translation_plan):
            prior = torch.as_tensor(
                model.translation_plan[model.public_action_step],
                device=action.device, dtype=action.dtype,
            ).unsqueeze(0)
            arm_ids = [index for index in range(14) if index not in (6, 13)]
            if model.public_action_step < 150:
                beta = 0.08
            elif model.public_action_step < 260:
                beta = 0.08
            else:
                beta = 0.18
            action[:, arm_ids] = (1.0 - beta) * action[:, arm_ids] + beta * prior[:, arm_ids]
        schedule_step = min(model.public_action_step, model.gripper_schedule.shape[0] - 1)
        action[:, [6, 13]] = model.gripper_schedule[schedule_step]
        _sync(model.act_device)
        forward_seconds = time.perf_counter() - model_started if did_forward else 0.0
        env_action = _map_action_to_environment(env, model, action)
        if not torch.isfinite(env_action).all().item():
            raise ValueError('Nonfinite executable action')
        _sync(env_action.device)
        elapsed = time.perf_counter() - started
        inference_times.append(elapsed)
        next_obs, _, terminated, truncated, info = env.step(env_action)
        model.public_action_step += 1
        # Official ACT adapter permits this query exclusively for post-step stopping.
        success = _done(env.get_wrapper_attr('is_task_success')())
        if model.public_trace_handle:
            row = dict(episode_index=model.episode_index, action_step=model.public_action_step,
                       qpos=_public_array(current, 'robot/qpos'),
                       policy_action=action.detach().cpu().tolist(),
                       environment_action=env_action.detach().cpu().tolist(),
                       full_policy_seconds=elapsed, model_forward_seconds=forward_seconds,
                       did_forward=did_forward, predicted_translation=model.predicted_translation)
            model.public_trace_handle.write(json.dumps(row, separators=(',', ':')) + '\n')
            model.public_trace_handle.flush()
        current = next_obs
        if success or _done(terminated) or _done(truncated):
            break
    return current, info, truncated, inference_times

