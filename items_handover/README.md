# items_handover RGB-only v60

This directory is the self-contained policy payload for `items_handover_rgb_fullplan_translation_dual_v60_20260916`.

## Runtime inputs

- `observation['robot']['qpos']` (14D)
- RGB color from `cam_high`, `cam_right_wrist`, and `cam_left_wrist`

The policy does not consume simulator object poses, depth, intrinsics, point clouds,
episode seeds, expert plans, contacts, rewards, or hidden task state. The post-step
`is_task_success` query is used only to stop an already successful episode.

## Method

Two RGB/qpos ACT checkpoints are averaged for arm motion. A three-camera ResNet-18
regresses the pen and holder translations from RGB/qpos once at episode start; those
predictions select a translation-only RBF trajectory prior. A fixed gripper schedule
is used. All perception, planning, action conversion, validation, and CUDA work before
`env.step` is included in the returned inference timing.

## Run

From an official RoboSynChallenge checkout with EmbodiChain installed:

```bash
PYTHON_BIN=/path/to/python EMBODICHAIN_ROOT=/path/to/EmbodiChain bash policy/items_handover/eval.sh 0 formal_v60_seed0_run1_random100 100 0
```

Arguments are `seed`, `run_name`, `max_episodes`, `GPU_ID`, optional result root,
and optional trace root. Python 3.11.15 and the pinned package versions in
`requirements-runtime.txt` were used for acceptance testing.

The two ACT runtime directories contain only `config.json` and
`model.safetensors`; historical training configs are retained in the external
provenance record because they are not runtime dependencies.
