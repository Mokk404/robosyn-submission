"""Stretch the contact-sensitive pick stages while retaining V7 placement."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v1.deploy_policy import _initialize
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _guided, _servo, _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _current_action


@dataclass
class Controller:
    actions: torch.Tensor | None = None
    step: int = 0
    initialized: bool = False
    ik_failures: int = 0
    release_active: bool = False
    release_steps: int = 0


def get_model(usr_args):
    return Controller()


def _resample(segment: torch.Tensor, count: int):
    if len(segment) == count:
        return segment.clone()
    positions = torch.linspace(0, len(segment) - 1, count)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    alpha = (positions - lower).reshape(-1, 1, 1)
    return segment[lower] * (1.0 - alpha) + segment[upper] * alpha


def _slow_initialize(base, model):
    _initialize(base, model)
    if model.actions is None:
        return
    original = model.actions
    pre = _resample(original[0:40], 50)
    approach = _resample(original[40:70], 70)
    settle = original[69:70].repeat(10, 1, 1)
    close = original[91:92].repeat(40, 1, 1)
    lift = _resample(original[92:127], 50)
    model.actions = torch.cat([pre, approach, settle, close, lift, original[127:]], dim=0)
    print(f"[BIMANUAL V9] slow staged frames={len(model.actions)}", flush=True)


def eval(env, model, obs):
    base = env.unwrapped
    if not model.initialized:
        _slow_initialize(base, model)
        model.initialized = True
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    opening = 0.0
    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(base.device)
        phase = "slow_staged"
    else:
        if (
            not model.release_active and float(slot_xy.item()) < 0.015
            and float(angle.item()) < 5.0 and float(cube_pos[0, 2].item()) < 1.030
            and (model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 0.985)
        ):
            model.release_active = True
            print(
                f"[BIMANUAL V9] slot release step={model.step} slot_xy={float(slot_xy.item()):.5f}"
                f" z={float(cube_pos[0,2].item()):.5f} angle={float(angle.item()):.2f}", flush=True
            )
        if model.release_active:
            action, opening = _guided(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = "slot_release"
        else:
            action = _servo(base, model, cube_pose, rack_pose)
            phase = "slot_servo"
    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 10 == 0 or success:
        print(
            f"[BIMANUAL V9] step={model.step} phase={phase} slot_xy={float(slot_xy.item()):.5f}"
            f" official_xy={float(official_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f}"
            f" angle={float(angle.item()):.2f} ik_fail={model.ik_failures}"
            f" opening={opening:.3f} success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
