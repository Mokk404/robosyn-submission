"""Replay only a verified pickup/handoff, then place relative to the live rack."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v14.deploy_policy import _guided_wide
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position


PACKAGE_ROOT = Path(__file__).resolve().parents[4]
DATA = str(
    PACKAGE_ROOT
    / "model_assets"
    / "trajectory_libraries"
    / "verified200"
    / "file-000.parquet"
)


@lru_cache(maxsize=1)
def _dataset():
    table = pq.read_table(DATA, columns=["episode_index", "frame_index", "action", "cube_pose"])
    episodes = np.asarray(table["episode_index"].to_numpy())
    frames = np.asarray(table["frame_index"].to_numpy())
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    cubes = np.asarray(table["cube_pose"].to_pylist(), dtype=np.float32)
    result = []
    for episode in np.unique(episodes):
        ids = np.flatnonzero(episodes == episode)
        ids = ids[np.argsort(frames[ids])]
        trajectory, cube_track = actions[ids], cubes[ids]
        mask = (trajectory[:, 6] < 0.1) & (trajectory[:, 13] > 0.9) & (np.arange(len(trajectory)) > 100)
        cutoff = None
        run = 0
        for index, value in enumerate(mask):
            run = run + 1 if value else 0
            if run >= 8:
                cutoff = index + 1
                break
        if cutoff is not None:
            result.append((int(episode), trajectory, cube_track, cutoff))
    return result


@dataclass
class Controller:
    trajectory: torch.Tensor | None = None
    ref_cube: torch.Tensor | None = None
    cutoff: int = 0
    step: int = 0
    initialized: bool = False
    nearest_episode: int = -1
    nearest_cost: float = 0.0
    ik_failures: int = 0
    release_active: bool = False
    release_steps: int = 0
    place_stage: str = "align"
    align_stable: int = 0


def get_model(usr_args):
    return Controller()


def _rotation_distance(a, b):
    rel = a[:3, :3] @ b[:3, :3].T
    return float(np.arccos(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)))


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy()
    best = None
    for episode, actions, cubes, cutoff in _dataset():
        position = np.linalg.norm(cube[:3, 3] - cubes[0, :3, 3])
        rotation = _rotation_distance(cube, cubes[0])
        cost = position + 0.06 * rotation
        if best is None or cost < best[0]:
            best = (cost, episode, actions, cubes, cutoff)
    cost, episode, actions, cubes, cutoff = best
    actions = actions.copy()
    actions[:, 6] *= 0.05
    actions[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(actions, dtype=torch.float32, device=base.device).reshape(-1, 1, 14)
    model.ref_cube = torch.as_tensor(cubes, dtype=torch.float32, device=base.device)
    model.cutoff = cutoff
    model.nearest_episode = episode
    model.nearest_cost = cost
    print(f"[KNN V18] selected episode={episode} cube_cost={cost:.6f} cutoff={cutoff}", flush=True)


def eval(env, model, obs):
    base = env.unwrapped
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if not model.initialized:
        _initialize(base, model, cube_pose); model.initialized = True
    cube_pos = cube_pose[:, :3, 3]
    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    opening = 0.0
    if model.step < model.cutoff:
        action = model.trajectory[model.step]; phase = "knn_handoff"
    else:
        if not model.release_active and model.place_stage == "lower" and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0 and float(cube_pos[0,2].item()) < 1.018 and (model.ik_failures >= 8 or float(cube_pos[0,2].item()) < 1.002):
            model.release_active = True
            print(f"[KNN V18] slot release step={model.step} z={float(cube_pos[0,2].item()):.5f}", flush=True)
        if model.release_active:
            action, opening = _guided_wide(base, model, cube_pose, rack_pose, slot_xy, angle); phase = "slot_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle); phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action); model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 20 == 0 or model.step == model.cutoff or success:
        ref_index = min(model.step - 1, len(model.ref_cube) - 1)
        ref_error = torch.norm(cube_pos[0] - model.ref_cube[ref_index, :3, 3])
        print(f"[KNN V18] step={model.step} phase={phase} ref_err={float(ref_error.item()):.5f} slot_xy={float(slot_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f} ik_fail={model.ik_failures} success={success}", flush=True)
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller(); model.__dict__.clear(); model.__dict__.update(fresh.__dict__)
