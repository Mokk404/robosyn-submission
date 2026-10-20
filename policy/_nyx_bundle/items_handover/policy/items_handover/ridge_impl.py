"""Public-observation trajectory regression policy for ``items_handover``.

The fitted artifact maps only observations explicitly registered by the
official random gym config (robot qpos, pen_pose, and holder_pose) to the
official 14-D joint-position action trajectory.  It never reads simulator
objects, randomization parameters, contacts, metrics, or task state.  The one
``is_task_success`` call is solely the evaluator-template-compatible early
stop after an action has already been applied.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _read_path(observation, path):
    value = observation
    for key in str(path).split("/"):
        if key:
            value = value[key]
    return value


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single(value, expected_shape, name):
    array = _numpy(value)
    if array.ndim == len(expected_shape) + 1 and array.shape[0] == 1:
        array = array[0]
    if tuple(array.shape) != tuple(expected_shape):
        raise ValueError(f"Expected {name} shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite public observation in {name}")
    return array.astype(np.float64, copy=False)


def _done(value):
    if isinstance(value, torch.Tensor):
        return bool(value.any().item())
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


@dataclass
class PublicRidgePolicy:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_keep: np.ndarray
    weights: np.ndarray
    trajectory_length: int
    action_dim: int
    artifact_path: Path
    act_step: int
    qpos_path: str
    pen_pose_path: str
    holder_pose_path: str
    gripper_action_ids: tuple[int, ...]
    scale_gripper_01_to_env: bool
    binary_gripper: bool
    public_trace_root: Path | None
    plan: np.ndarray | None = None
    plan_step: int = 0
    episode_index: int = -1
    public_action_step: int = 0
    inference_times: list[float] | None = None
    trace_handle: object | None = None


def get_model(usr_args):
    artifact = Path(usr_args.get("artifact_path") or "").expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(f"Public ridge artifact not found: {artifact}")
    data = np.load(artifact, allow_pickle=False)
    required = {
        "feature_mean",
        "feature_std",
        "feature_keep",
        "weights",
        "trajectory_length",
        "action_dim",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Ridge artifact missing keys: {sorted(missing)}")
    mean = np.asarray(data["feature_mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["feature_std"], dtype=np.float64).reshape(-1)
    keep = np.asarray(data["feature_keep"], dtype=bool).reshape(-1)
    weights = np.asarray(data["weights"], dtype=np.float64)
    length = int(np.asarray(data["trajectory_length"]).item())
    action_dim = int(np.asarray(data["action_dim"]).item())
    kept_dim = int(keep.sum())
    if mean.shape != std.shape or mean.shape != keep.shape:
        raise ValueError("Ridge feature statistics have inconsistent shapes")
    if weights.shape != (kept_dim + 1, length * action_dim):
        raise ValueError(
            f"Expected weights {(kept_dim + 1, length * action_dim)}, got {weights.shape}"
        )
    if mean.size != 38 or action_dim != 14:
        raise ValueError(f"Expected public input 38 and action 14, got {mean.size}/{action_dim}")
    trace_root_value = usr_args.get("public_trace_root")
    trace_root = Path(trace_root_value).resolve() if trace_root_value else None
    if trace_root:
        trace_root.mkdir(parents=True, exist_ok=True)
    return PublicRidgePolicy(
        feature_mean=mean,
        feature_std=std,
        feature_keep=keep,
        weights=weights,
        trajectory_length=length,
        action_dim=action_dim,
        artifact_path=artifact,
        act_step=int(usr_args.get("act_step", 10)),
        qpos_path=str(usr_args.get("qpos_path", "robot/qpos")),
        pen_pose_path=str(usr_args.get("pen_observation_path", "pen_pose")),
        holder_pose_path=str(usr_args.get("holder_observation_path", "holder_pose")),
        gripper_action_ids=tuple(int(i) for i in usr_args.get("gripper_action_ids", [6, 13])),
        scale_gripper_01_to_env=bool(usr_args.get("scale_gripper_01_to_env", True)),
        binary_gripper=bool(usr_args.get("binary_gripper", True)),
        public_trace_root=trace_root,
        inference_times=[],
    )


def _public_feature(model, observation):
    qpos = _single(_read_path(observation, model.qpos_path), (14,), model.qpos_path)
    pen = _single(_read_path(observation, model.pen_pose_path), (4, 4), model.pen_pose_path)
    holder = _single(
        _read_path(observation, model.holder_pose_path), (4, 4), model.holder_pose_path
    )
    # Exact schema recorded in the artifact provenance: qpos followed by the
    # row-major 3x3 rotation and XYZ translation of each registered pose.
    return np.concatenate(
        [
            qpos,
            pen[:3, :3].reshape(-1),
            pen[:3, 3],
            holder[:3, :3].reshape(-1),
            holder[:3, 3],
        ]
    )


def _predict_plan(model, observation):
    raw = _public_feature(model, observation)
    safe_std = np.where(model.feature_std > 1e-12, model.feature_std, 1.0)
    normalized = (raw - model.feature_mean) / safe_std
    design = np.concatenate(([1.0], normalized[model.feature_keep]))
    plan = (design @ model.weights).reshape(model.trajectory_length, model.action_dim)
    if not np.isfinite(plan).all():
        raise ValueError("Ridge inference produced a non-finite action")
    if model.binary_gripper:
        plan[:, list(model.gripper_action_ids)] = (
            plan[:, list(model.gripper_action_ids)] >= 0.5
        ).astype(plan.dtype)
    return plan.astype(np.float32)


def _environment_action(env, model, action):
    space = env.unwrapped.single_action_space
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    if low.size != model.action_dim or high.size != model.action_dim:
        raise ValueError(f"Official action space is {low.size}D; expected {model.action_dim}D")
    result = np.asarray(action, dtype=np.float32).copy()
    if model.scale_gripper_01_to_env:
        ids = list(model.gripper_action_ids)
        normalized = np.clip(result[ids], 0.0, 1.0)
        result[ids] = low[ids] + normalized * (high[ids] - low[ids])
    # Enforce the public action interface bounds for every joint.  This only
    # constrains the policy output; it does not alter robot or simulator state.
    result = np.clip(result, low, high)
    return torch.as_tensor(result[None], dtype=torch.float32, device=env.unwrapped.device)


def _write_trace(model, observation, raw_action, env_action, elapsed, success):
    if model.trace_handle is None:
        return
    record = {
        "episode_index": model.episode_index,
        "public_action_step": model.public_action_step,
        "qpos": _numpy(_read_path(observation, model.qpos_path)).tolist(),
        "pen_pose": _numpy(_read_path(observation, model.pen_pose_path)).tolist(),
        "holder_pose": _numpy(_read_path(observation, model.holder_pose_path)).tolist(),
        "policy_action": np.asarray(raw_action).tolist(),
        "environment_action": env_action.detach().cpu().numpy().tolist(),
        "inference_wall_s": elapsed,
        "success_after_action": bool(success),
    }
    model.trace_handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    model.trace_handle.flush()


def eval(env, model, obs):
    current = obs
    info = None
    truncated = False
    actions_this_call = 0
    for _ in range(model.act_step):
        if model.plan is None:
            started = time.perf_counter()
            model.plan = _predict_plan(model, current)
            elapsed = time.perf_counter() - started
            model.inference_times.append(elapsed)
        else:
            elapsed = 0.0
        index = min(model.plan_step, model.trajectory_length - 1)
        raw_action = model.plan[index]
        env_action = _environment_action(env, model, raw_action)
        next_obs, _, terminated, truncated, info = env.step(env_action)
        model.plan_step += 1
        model.public_action_step += 1
        actions_this_call += 1
        success = _done(env.get_wrapper_attr("is_task_success")())
        _write_trace(model, current, raw_action, env_action, elapsed, success)
        current = next_obs
        if success:
            print(f"POLICY_TASK_SUCCESS public_action_step={model.public_action_step}")
            break
        if _done(terminated) or _done(truncated):
            break
    if info is None:
        info = {}
    else:
        info = dict(info)
    info["_policy_action_steps"] = actions_this_call
    return current, info, truncated


def reset_model(model):
    model.plan = None
    model.plan_step = 0
    model.public_action_step = 0
    model.inference_times = []
    model.episode_index += 1
    if model.trace_handle is not None:
        model.trace_handle.close()
        model.trace_handle = None
    if model.public_trace_root:
        trace_path = model.public_trace_root / f"episode_{model.episode_index:03d}.jsonl"
        model.trace_handle = trace_path.open("x", encoding="utf-8")


__all__ = ["get_model", "eval", "reset_model"]
