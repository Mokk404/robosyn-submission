# click_bell — final ACT submission

## Selected policy and checkpoint

- Task: `click_bell`; submitted inference: LeRobot ACT without an online teacher, expert action bank, or recovery intervention.
- Selected checkpoint: `080000` from `act_click_bell_006_007_merged_bs8_v0`. This is the checkpoint explicitly loaded by the existing left/right evaluation logs and named by the saved full-task random evaluation. Other training checkpoints were not established as better by a comparable full-task rollout; this selection does **not** claim a measured superiority over every unevaluated checkpoint.
- Original source checkpoint: `/home/user/Robotsynchallenge/RoboSynChallenge/outputs/act_click_bell_006_007_merged_bs8_v0/checkpoints/080000/pretrained_model` (provenance only).
- Submission checkpoint: `checkpoint/080000/pretrained_model/`. Its original `config.json`, `model.safetensors`, and `train_config.json` are copied intact.

| Checkpoint file | SHA256 |
|---|---|
| `config.json` | `b882672f8029f1887651ce1719ac9047ee1ddb1565085d6bb5e1e2868f6cdd2c` |
| `model.safetensors` | `b8adc47d50b0308a09859688c97a467e5087ee178935f32189fdfab2b451ff85` |

`train_config.json` retains absolute dataset/output paths from the original training machine. They are historical metadata, not inputs to the submitted inference entry. Normalization buffers are loaded from the original checkpoint weights; do not regenerate or externally reapply normalization statistics.

## Existing evaluation evidence

The 080000 checkpoint was explicitly loaded in five 10-episode `train_left/random` runs (47/50 success) and five `train_right/random` runs (47/50 success). These are **arm-specific settings**, not 94/100 for the complete `random` task. The saved full-task `random` run named `click_bell_080000_random100_20260907_162848` was interrupted: 35 completed videos contain 6 successes and 29 failures, plus one temporary video. It is **not** a completed random100 result. The videos do not embed a checkpoint SHA, so the full-task attribution also relies on the saved experiment naming and provenance. No new evaluation was run while assembling this package.

## Inference contract

- Recommended runtime: Ubuntu 22.04, Python 3.11.x, LeRobot 0.3.3, and a compatible CUDA-enabled PyTorch environment. The source `policy/act/.venv` is used by the launcher but is not bundled.
- ACT: `chunk_size=50`, `n_action_steps=10`, 10 outer environment steps per evaluator call. These values preserve the original checkpoint and `policy/act/deploy_policy.yml` execution settings; **do not substitute** the manipulate_pipette 50-action execution configuration.
- Observation: 14D `robot/qpos` mapped to `observation.state`, 14D `robot/qvel` and `robot/qf`, plus RGB from `cam_high`, `cam_right_wrist`, and `cam_left_wrist`. The adapter takes three color channels, changes HWC to CHW, and converts uint8-like image values to `[0,1]` once.
- Action: 14D. `[0:6]` left-arm absolute joint targets, `[6]` left-gripper policy command, `[7:13]` right-arm absolute joint targets, `[13]` right-gripper policy command. Gripper policy commands are `[0,1]`; the copied adapter performs the existing single mapping to each environment action-space physical range before `env.step()`.
- Final inference uses ordinary robot/camera observations only. It does not rely on simulator object/button ground truth or an online teacher. The byte-identical adapter includes a dormant `debug_button_press` branch that reads simulator state; both the submitted config and launcher force that branch off.

## Package entry and dependencies

`policy/submitted_act/deploy_policy.py` remains a byte-identical copy of the generic `policy/act` inference adapter. The package initializer and `official_contract.py` add the official evaluator's fourth return value: actual ACT chunk-inference times. The click_bell-specific files under `policy/act/click_bell/` contain official-source snapshots and teacher/debug utilities, not imports required by the ACT inference adapter; they are intentionally excluded. `config/deploy_policy.yml` relocates `policy_name` and sets the local-only `eval_freeze_interval_events` option to `false`; the official evaluator has no interval-to-reset rewrite. The launcher selects the inspected three- or four-result evaluator contract and fails closed on an unknown one. See `docs/OFFICIAL_RUNTIME_COMPATIBILITY.md`.

This package **requires the RoboSynChallenge runtime**: `scripts/eval_policy.py`, task environment/configs, the `robosynchallenge` package, EmbodiChain/DexSim, LeRobot and their dependencies. It does not bundle the whole project. The relocated submission entry was statically checked but has **not** itself been rollout-tested; the evaluation numbers above belong to existing source-project runs.

Portable Linux invocation (not executed during packaging):

```bash
SUBMISSION_ROOT=/path/to/click_bell
export ROBOSYNCHALLENGE_ROOT=/path/to/RoboSynChallenge
cd "$SUBMISSION_ROOT"
bash deploy/eval.sh 0 100 0
```

The arguments are GPU ID, episode count, and evaluator seed. `MANIFEST.txt` lists every package file; `SHA256SUMS.txt` checks all content files except itself and the manifest. No dataset, video, optimizer state, training log, or unselected checkpoint is part of the submission. Existing success counts came from earlier local conditions; an official-runtime submission-entry smoke and full-task evaluation remain unperformed.
