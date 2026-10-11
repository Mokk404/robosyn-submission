"""V13 handoff and alignment with a release that cannot remain friction-locked."""

from __future__ import annotations

import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v7.deploy_policy import _slot_position
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v10.deploy_policy import _place_servo
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_bimanual_closed_loop_v13.deploy_policy import (
    Controller,
    _adaptive_tail,
    _handoff_geometry,
    _initialize,
)
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_direct_right_v1.deploy_policy import _current_action, _ik_ok, _rotation_step


def get_model(usr_args):
    return Controller()


def _guided_wide(base, model, cube_pose, rack_pose, slot_xy, angle):
    robot = base.robot
    action = _current_action(base)
    qpos = robot.get_qpos(name="left_arm")
    eef = robot.compute_fk(qpos, name="left_arm", to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    slot = _slot_position(rack_pose)
    desired = cube_pose.clone()
    correction = slot[:, :2] - cube_pos[:, :2]
    norm = correction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    correction *= torch.minimum(torch.ones_like(norm), 0.0010 / norm)
    desired[:, :2, 3] += correction
    desired[:, :3, :3] = _rotation_step(cube_pose[:, :3, :3], 0.010) @ cube_pose[:, :3, :3]
    target_eef = (desired @ torch.linalg.inv(cube_pose)) @ eef
    result = robot.compute_ik(pose=target_eef, joint_seed=qpos, name="left_arm")
    if _ik_ok(result):
        action[:, :6] = qpos + (result[1] - qpos).clamp(-0.020, 0.020)

    severe = float(slot_xy.item()) > 0.020 or float(angle.item()) > 10.0
    seated_height = float(rack_pose[0, 2, 3].item()) + 0.066
    if float(cube_pos[0, 2].item()) <= seated_height:
        opening = 0.030
    elif severe:
        opening = 0.025
    elif model.release_steps < 6:
        opening = 0.006
    elif model.release_steps < 12:
        opening = 0.009
    elif model.release_steps < 20:
        opening = 0.015
    else:
        opening = 0.025
    action[:, 6] = opening
    action[:, 13] = 0.05
    model.release_steps += 1
    return action, opening


def eval(env, model, obs):
    base = env.unwrapped
    if not model.initialized:
        _initialize(base, model)
        model.initialized = True
    cube_pose = base.sim.get_rigid_object("cube").get_local_pose(to_matrix=True)
    rack_pose = base.sim.get_rigid_object("rack").get_local_pose(to_matrix=True)
    cube_pos = cube_pose[:, :3, 3]
    angle = torch.rad2deg(torch.arccos(cube_pose[:, 2, 2].clamp(-1.0, 1.0)))
    if model.actions is not None and model.step == model.handoff_trigger and not model.handoff_checked:
        geometry = _handoff_geometry(base, cube_pose)
        correction = float(geometry[2].item()) if geometry is not None else 999.0
        needs_adaptive = model.fallback_pick or correction > 0.05 or float(angle.item()) > 15.0
        if needs_adaptive:
            _adaptive_tail(base, model, cube_pose)
        else:
            print(f"[BIMANUAL V14] nominal tail correction={correction:.5f} angle={float(angle.item()):.2f}", flush=True)
        model.handoff_checked = True
    slot = _slot_position(rack_pose)
    slot_xy = torch.norm(cube_pos[:, :2] - slot[:, :2], dim=-1)
    official_xy = torch.norm(cube_pos[:, :2] - rack_pose[:, :2, 3], dim=-1)
    opening = 0.0
    if model.actions is not None and model.step < len(model.actions):
        action = model.actions[model.step].to(base.device)
        phase = "adaptive_staged" if model.adaptive_handoff else "nominal_staged"
    else:
        if (
            not model.release_active and model.place_stage == "lower"
            and float(slot_xy.item()) < 0.012 and float(angle.item()) < 5.0
            and float(cube_pos[0, 2].item()) < 1.010
            and (model.ik_failures >= 8 or float(cube_pos[0, 2].item()) < 1.002)
        ):
            model.release_active = True
            print(f"[BIMANUAL V14] slot release step={model.step} z={float(cube_pos[0,2].item()):.5f}", flush=True)
        if model.release_active:
            action, opening = _guided_wide(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = "slot_release"
        else:
            action = _place_servo(base, model, cube_pose, rack_pose, slot_xy, angle)
            phase = f"place_{model.place_stage}"
    next_obs, _reward, terminated, truncated, info = env.step(action)
    model.step += 1
    success = bool(torch.as_tensor(base.is_task_success()).any().item())
    if model.step % 10 == 0 or success:
        print(
            f"[BIMANUAL V14] step={model.step} phase={phase} slot_xy={float(slot_xy.item()):.5f}"
            f" official_xy={float(official_xy.item()):.5f} z={float(cube_pos[0,2].item()):.5f}"
            f" angle={float(angle.item()):.2f} ik_fail={model.ik_failures} opening={opening:.3f}"
            f" success={success}", flush=True
        )
    done = success or bool(torch.as_tensor(terminated).any().item()) or bool(torch.as_tensor(truncated).any().item())
    return next_obs, info, done


def reset_model(model):
    fresh = Controller()
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)
