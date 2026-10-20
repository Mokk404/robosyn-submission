"""V19 controller using all 365 official-gated verified demonstrations."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import glob

import numpy as np
import pyarrow.parquet as pq
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v14.deploy_policy import _guided_wide
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import Controller


PACKAGE_ROOT = Path(__file__).resolve().parents[4]
DATA_GLOB = str(
    PACKAGE_ROOT
    / "model_assets"
    / "trajectory_libraries"
    / "union365"
    / "chunk-*"
    / "episode_*.parquet"
)


@lru_cache(maxsize=1)
def _dataset():
    result = []
    for path in sorted(glob.glob(DATA_GLOB)):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "action", "cube_pose"])
        frames = np.asarray(table["frame_index"].to_numpy())
        order = np.argsort(frames)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[order]
        cubes = np.asarray(table["cube_pose"].to_pylist(), dtype=np.float32)[order]
        episode = int(table["episode_index"][0].as_py())
        mask = (actions[:, 6] < 0.1) & (actions[:, 13] > 0.9) & (np.arange(len(actions)) > 100)
        run = 0; cutoff = None
        for index, value in enumerate(mask):
            run = run + 1 if value else 0
            if run >= 8:
                cutoff = index + 1; break
        if cutoff is not None:
            result.append((episode, actions, cubes, cutoff))
    print(f"[KNN V20] loaded verified episodes={len(result)}", flush=True)
    return result


def get_model(usr_args):
    return Controller()


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy(); axis = cube[:3, 2]
    best = None
    for episode, actions, cubes, cutoff in _dataset():
        position = np.linalg.norm(cube[:3, 3] - cubes[0, :3, 3])
        axis_angle = float(np.arccos(np.clip(np.dot(axis, cubes[0, :3, 2]), -1.0, 1.0)))
        cost = position + 0.06 * axis_angle
        if best is None or cost < best[0]:
            best = (cost, episode, actions, cubes, cutoff, position, axis_angle)
    cost, episode, actions, cubes, cutoff, position, axis_angle = best
    actions = actions.copy(); actions[:, 6] *= 0.05; actions[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(actions, dtype=torch.float32, device=base.device).reshape(-1, 1, 14)
    model.ref_cube = torch.as_tensor(cubes, dtype=torch.float32, device=base.device)
    model.cutoff = cutoff; model.nearest_episode = episode; model.nearest_cost = cost
    print(f"[KNN V20] selected episode={episode} axis_cost={cost:.6f} position={position:.6f} axis_angle={np.degrees(axis_angle):.2f} cutoff={cutoff}", flush=True)


def eval(env, model, obs):
    base = env.unwrapped
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    if not model.initialized:
        _initialize(base, model, cube_pose); model.initialized = True
    cube_pos = cube_pose[:, :3, 3]; slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0))); opening = 0.0
    if model.step < model.cutoff:
        action = model.trajectory[model.step]; phase = "union_knn_handoff"
    else:
        if not model.release_active and model.place_stage == "lower" and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0 and float(cube_pos[0,2].item()) < 1.018 and (model.ik_failures >= 8 or float(cube_pos[0,2].item()) < 1.002):
            model.release_active = True
        if model.release_active:
            action, opening = _guided_wide(base, model, cube_pose, rack_pose, slot_xy, angle); phase = "slot_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle); phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action); model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 20 == 0 or model.step == model.cutoff or success:
        ref_index = min(model.step - 1, len(model.ref_cube) - 1)
        ref_error = torch.norm(cube_pos[0] - model.ref_cube[ref_index, :3, 3])
        print(f"[KNN V20] step={model.step} phase={phase} ref_err={float(ref_error.item()):.5f} slot_xy={float(slot_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f} ik_fail={model.ik_failures} success={success}", flush=True)
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller(); model.__dict__.clear(); model.__dict__.update(fresh.__dict__)
