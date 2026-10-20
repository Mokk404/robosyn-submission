"""Keep V27 control but strongly reject trajectories with predicted IK failures."""

import numpy as np
import torch

from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_handoff_v18.deploy_policy import _dataset
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_hybrid_v25.deploy_policy import _delta
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_recovery_v27 import deploy_policy as implementation
from policy.sample_loading_final_bundle_20260907.policies.sample_loading_knn_retarget_v23.deploy_policy import _feasibility, _left_start


def get_model(usr_args):
    model = implementation.get_model(usr_args)
    model.recovery_attempted = True
    return model


def _initialize(base, model, cube_pose):
    cube = cube_pose[0].detach().cpu().numpy()
    candidates = []
    for episode, actions, cubes, cutoff in _dataset():
        reference = cubes[0]
        if not (0.75 < float(reference[2, 3]) < 0.90):
            continue
        position = float(np.linalg.norm(cube[:3, 3] - reference[:3, 3]))
        axis_angle = float(
            np.arccos(np.clip(np.dot(cube[:3, 2], reference[:3, 2]), -1.0, 1.0))
        )
        geometry = position + 0.06 * axis_angle
        candidates.append((geometry, episode, actions, cubes, cutoff, position, axis_angle))
    candidates.sort(key=lambda item: item[0])
    ranked = []
    for geometry, episode, actions, cubes, cutoff, position, axis_angle in candidates[:16]:
        left_start = _left_start(actions, cutoff)
        delta, applied_angle = _delta(cube, cubes[0], axis_angle)
        fraction, failures, calls = _feasibility(
            base, actions, cutoff, left_start, delta
        )
        score = geometry + 0.100 * fraction
        ranked.append(
            (
                score, geometry, episode, actions, cubes, cutoff, position,
                axis_angle, left_start, delta, applied_angle, failures, calls,
            )
        )
    ranked.sort(key=lambda item: item[0])
    (
        score, geometry, episode, actions, cubes, cutoff, position,
        axis_angle, left_start, delta, applied_angle, failures, calls,
    ) = ranked[0]
    model.raw_actions = actions.copy()
    scaled = actions.copy()
    scaled[:, 6] *= 0.05
    scaled[:, 13] *= 0.05
    model.trajectory = torch.as_tensor(
        scaled, dtype=torch.float32, device=base.device
    ).reshape(-1, 1, 14)
    model.ref_cube = torch.as_tensor(cubes, dtype=torch.float32, device=base.device)
    model.scene_delta = torch.as_tensor(
        delta, dtype=torch.float32, device=base.device
    ).reshape(1, 4, 4)
    model.cutoff = cutoff
    model.left_start = left_start
    model.nearest_episode = episode
    model.nearest_cost = score
    model.recovery_attempted = True
    print(
        f"[KNN V28] selected episode={episode} score={score:.6f}"
        f" geometry={geometry:.6f} current_xy=({cube[0,3]:.5f},{cube[1,3]:.5f})"
        f" reference_xy=({cubes[0,0,3]:.5f},{cubes[0,1,3]:.5f})"
        f" position={position:.6f} axis_angle={np.degrees(axis_angle):.2f}"
        f" applied_angle={applied_angle:.2f} predicted_ik_fail={failures}/{calls}"
        f" left_start={left_start} cutoff={cutoff}",
        flush=True,
    )


implementation._initialize = _initialize


def eval(env, model, obs):
    return implementation.eval(env, model, obs)


def reset_model(model):
    fresh = get_model(None)
    model.__dict__.clear()
    model.__dict__.update(fresh.__dict__)


__all__ = ["get_model", "eval", "reset_model"]

