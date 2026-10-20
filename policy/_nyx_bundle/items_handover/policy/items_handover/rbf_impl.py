"""Nonlinear trajectory policy using only officially registered observations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ridge_impl as common


@dataclass
class PublicRbfPolicy:
    feature_mean: np.ndarray
    feature_std: np.ndarray
    feature_keep: np.ndarray
    train_features: np.ndarray
    dual: np.ndarray
    trajectory_mean: np.ndarray
    trajectory_components: np.ndarray
    gamma: float
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
        raise FileNotFoundError(f"Public RBF artifact not found: {artifact}")
    data = np.load(artifact, allow_pickle=False)
    required = {
        "feature_mean",
        "feature_std",
        "feature_keep",
        "train_features",
        "dual",
        "trajectory_mean",
        "trajectory_components",
        "gamma",
        "trajectory_length",
        "action_dim",
    }
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"RBF artifact missing keys: {sorted(missing)}")
    mean = np.asarray(data["feature_mean"], dtype=np.float64).reshape(-1)
    std = np.asarray(data["feature_std"], dtype=np.float64).reshape(-1)
    keep = np.asarray(data["feature_keep"], dtype=bool).reshape(-1)
    train = np.asarray(data["train_features"], dtype=np.float64)
    dual = np.asarray(data["dual"], dtype=np.float64)
    trajectory_mean = np.asarray(data["trajectory_mean"], dtype=np.float64).reshape(-1)
    components = np.asarray(data["trajectory_components"], dtype=np.float64)
    length = int(np.asarray(data["trajectory_length"]).item())
    action_dim = int(np.asarray(data["action_dim"]).item())
    if mean.size != 38 or mean.shape != std.shape or mean.shape != keep.shape:
        raise ValueError("RBF public feature schema mismatch")
    if train.shape != (dual.shape[0], int(keep.sum())):
        raise ValueError("RBF train-feature/dual shape mismatch")
    if dual.shape[1] != components.shape[0]:
        raise ValueError("RBF latent shape mismatch")
    if components.shape[1] != length * action_dim or trajectory_mean.size != length * action_dim:
        raise ValueError("RBF trajectory shape mismatch")
    if action_dim != 14:
        raise ValueError(f"Expected official 14-D action, got {action_dim}")
    trace_value = usr_args.get("public_trace_root")
    trace_root = Path(trace_value).resolve() if trace_value else None
    if trace_root:
        trace_root.mkdir(parents=True, exist_ok=True)
    return PublicRbfPolicy(
        feature_mean=mean,
        feature_std=std,
        feature_keep=keep,
        train_features=train,
        dual=dual,
        trajectory_mean=trajectory_mean,
        trajectory_components=components,
        gamma=float(np.asarray(data["gamma"]).item()),
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


def _predict_plan(model, observation):
    raw = common._public_feature(model, observation)
    safe_std = np.where(model.feature_std > 1e-12, model.feature_std, 1.0)
    feature = ((raw - model.feature_mean) / safe_std)[model.feature_keep]
    delta = model.train_features - feature[None]
    kernel = np.exp(-model.gamma * np.sum(delta * delta, axis=1))
    latent = kernel @ model.dual
    plan = (latent @ model.trajectory_components + model.trajectory_mean).reshape(
        model.trajectory_length, model.action_dim
    )
    if not np.isfinite(plan).all():
        raise ValueError("RBF inference produced a non-finite action")
    if model.binary_gripper:
        ids = list(model.gripper_action_ids)
        plan[:, ids] = (plan[:, ids] >= 0.5).astype(plan.dtype)
    return plan.astype(np.float32)


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
            print(f"PUBLIC_RBF_INFERENCE wall_s={elapsed:.9f}")
        else:
            elapsed = 0.0
        index = min(model.plan_step, model.trajectory_length - 1)
        raw_action = model.plan[index]
        env_action = common._environment_action(env, model, raw_action)
        next_obs, _, terminated, truncated, info = env.step(env_action)
        model.plan_step += 1
        model.public_action_step += 1
        actions_this_call += 1
        success = common._done(env.get_wrapper_attr("is_task_success")())
        common._write_trace(model, current, raw_action, env_action, elapsed, success)
        current = next_obs
        if success:
            print(f"POLICY_TASK_SUCCESS public_action_step={model.public_action_step}")
            break
        if common._done(terminated) or common._done(truncated):
            break
    info = {} if info is None else dict(info)
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
        path = model.public_trace_root / f"episode_{model.episode_index:03d}.jsonl"
        model.trace_handle = path.open("x", encoding="utf-8")


__all__ = ["get_model", "eval", "reset_model"]
