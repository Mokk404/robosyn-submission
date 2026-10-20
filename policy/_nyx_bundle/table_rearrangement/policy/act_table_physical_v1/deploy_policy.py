from pathlib import Path
"""Table ACT adapter. select_action returns physical actions; never rescale them."""
import time
import numpy as np
import torch


def encode_action(action, env):
    """Preserve all arm targets; clip physical gripper coordinates 6 and 13."""
    if action.ndim == 1:
        action = action.unsqueeze(0)
    space = env.unwrapped.single_action_space
    if action.ndim != 2 or action.shape[-1] != 14 or tuple(space.shape) != (14,):
        raise ValueError("Table requires actions shaped [B, 14] and a 14D action space")
    if not torch.isfinite(action).all():
        raise ValueError("Non-finite ACT action")
    action = action.clone()
    ids = [6, 13]
    low = torch.as_tensor(space.low, device=action.device, dtype=action.dtype)[ids]
    high = torch.as_tensor(space.high, device=action.device, dtype=action.dtype)[ids]
    if not (torch.isfinite(low).all() and torch.isfinite(high).all() and (high > low).all()):
        raise ValueError("Invalid gripper action limits")
    action[..., ids] = torch.maximum(low, torch.minimum(high, action[..., ids]))
    return action.detach().to(device=env.unwrapped.device, dtype=torch.float32)


def _read(obs, path):
    for part in path.split('/'):
        obs = obs[part]
    return obs


def encode_obs(obs, model):
    batch = {}
    for name in model.config.input_features:
        if name.startswith('observation.images.'):
            camera = model.table_images[name]
            value = torch.as_tensor(_read(obs, f'sensor/{camera}/color'), device=model.table_device)
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4:
                raise ValueError(f'Invalid camera shape: {name} {value.shape}')
            if value.shape[-1] in (3, 4):
                value = value[..., :3].permute(0, 3, 1, 2)
            if value.shape[1] != 3:
                raise ValueError(f'Expected RGB image: {name}')
            if value.dtype == torch.uint8:
                value = value.float() / 255.0
            else:
                value = value.float()
                if value.min() < 0 or value.max() > 1:
                    raise ValueError('Floating RGB images must be in [0,1]')
            batch[name] = value.contiguous()
        else:
            value = torch.as_tensor(_read(obs, model.table_states[name]), device=model.table_device, dtype=torch.float32)
            batch[name] = value.unsqueeze(0) if value.ndim == 1 else value
    return batch


def get_model(usr_args):
    from lerobot.policies.act.modeling_act import ACTPolicy
    task = usr_args.get('task_name')
    if task != 'table_rearrangement':
        raise ValueError('This adapter is exclusively for table_rearrangement')
    if usr_args.get('scale_gripper_01_to_env', False) not in (False, 'False', 'false'):
        raise ValueError('Physical Table checkpoint must not use gripper rescaling')
    checkpoint = usr_args.get('checkpoint_path') or str(Path(__file__).resolve().parent / 'checkpoint')
    if not checkpoint:
        raise ValueError('checkpoint_path is required')
    device = usr_args.get('pytorch_device', 'cuda')
    try:
        model = ACTPolicy.from_pretrained(checkpoint, cli_overrides=[f'--device={device}'])
    except TypeError as exc:
        if 'cli_overrides' not in str(exc):
            raise
        from lerobot.configs.policies import PreTrainedConfig
        config = PreTrainedConfig.from_pretrained(checkpoint, cli_overrides=[f'--device={device}'])
        model = ACTPolicy.from_pretrained(checkpoint, config=config)
    model.to(device).eval()
    model.table_device = next(model.parameters()).device
    model.table_act_step = int(usr_args.get('act_step', 10))
    if model.table_act_step < 1:
        raise ValueError('act_step must be positive')
    if model.config.chunk_size != 50 or model.config.n_action_steps != 50:
        raise ValueError('Validated Table checkpoint requires K=N=50')
    model.table_states = {'observation.state': 'robot/qpos', 'observation.qvel': 'robot/qvel', 'observation.qf': 'robot/qf'}
    model.table_images = {f'observation.images.{name}': name for name in ('cam_high', 'cam_right_wrist', 'cam_left_wrist')}
    model.table_images.update(usr_args.get('image_key_map', {}))
    for name in model.config.input_features:
        if name not in model.table_states and name not in model.table_images:
            raise ValueError(f'Unsupported observation feature: {name}')
    # Verified 2026-09-20 official snapshot consumes four values; legacy uses three.
    model.table_timing = usr_args.get('return_inference_timing', True) in (True, 'True', 'true')
    print('Table gripper mode: physical; clip indices 6,13 to environment bounds; no rescaling', flush=True)
    model.reset()
    return model


def _done(value):
    return bool(torch.as_tensor(value).any())


@torch.inference_mode()
def eval(env, model, obs):
    times = []
    info, truncated = None, False
    executed_steps = 0
    for _ in range(model.table_act_step):
        inference = model.config.temporal_ensemble_coeff is not None or len(model._action_queue) == 0
        if inference and model.table_device.type == 'cuda':
            torch.cuda.synchronize(model.table_device)
        started = time.perf_counter()
        action = encode_action(model.select_action(encode_obs(obs, model)), env)
        if inference:
            if model.table_device.type == 'cuda':
                torch.cuda.synchronize(model.table_device)
            times.append(time.perf_counter() - started)
        obs, _, terminated, truncated, info = env.step(action)
        executed_steps += 1
        if _done(terminated) or _done(truncated):
            break
        if _done(env.get_wrapper_attr('is_task_success')()):
            break
    info = dict(info or {})
    info['_policy_action_steps'] = executed_steps
    result = (obs, info, truncated)
    return (*result, times) if model.table_timing else result


def reset_model(model):
    model.reset()
