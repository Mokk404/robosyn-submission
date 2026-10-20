# manipulate_pipette — final submission

## Final artifact

- Task: `manipulate_pipette`, `random` setting.
- Framework: LeRobot ACT, **Pure ACT** inference. No teacher, recovery controller, press guard, action-queue refresh, or T300 diagnostic intervention is enabled.
- Selected checkpoint: **current1500**. Its folder was named `000500` because it contains 500 further weight updates from an already trained 1000-step continuation model. The optimizer was newly initialized for those 500 steps; this was **not** a strict optimizer-state resume.
- Observed Pure ACT random100: **57 / 100**. The existing video directory contains 100 files, 57 named success and 43 named fail. This is an observed run result, not a precisely causal +12 percentage-point claim versus historical baseline. The videos themselves do not embed the checkpoint SHA; the run-to-checkpoint attribution also relies on the recorded experiment provenance.

Original Linux checkpoint:

`/home/user/Robotsynchallenge/RoboSynChallenge/experiments/manipulate_pipette/outputs/continuation_training_low_interference_from1k_to1p5k_train/checkpoints/000500/pretrained_model`

Submission checkpoint: `checkpoint/current1500/pretrained_model/`. The entire original directory is copied, including `config.json`, `model.safetensors`, and `train_config.json`; the latter is retained for provenance, not required for inference.

`checkpoint/current1500/pretrained_model/train_config.json` contains historical absolute paths from the original training environment, including its dataset root and output directory. Those paths are provenance metadata only; the submitted Pure ACT inference entry does not use them.

| File | SHA256 |
|---|---|
| `checkpoint/current1500/pretrained_model/config.json` | `025e02433a3e8c844093550fff0d6db5e7494b412f6e74dce7674910b36874e9` |
| `checkpoint/current1500/pretrained_model/model.safetensors` | `5318fbb1d4c371290931d083c2fe76e257a190e803965386648f6518a23f9fec` |

Both copied files were hashed against the source and the expected values. `SHA256SUMS.txt` lists every packaged file other than itself and `MANIFEST.txt`.

## Inference contract

- Python: CPython 3.11.x. The inspected source venv uses LeRobot **0.3.3**, PyTorch **2.7.1**, torchvision 0.22.1, Gymnasium 0.29.1. Ubuntu 22.04 with a compatible NVIDIA driver/CUDA-enabled PyTorch installation is recommended; the exact driver/toolkit version was not established from the venv metadata. Check `torch.cuda.is_available()` in the target environment.
- `chunk_size=50`, `n_action_steps=50`, `action_dim=14`. The outer evaluator calls the wrapper in blocks of 10 environment steps; ACT's queue ordinarily generates a fresh 50-action chunk only when depleted.
- Observation: `robot/qpos` → `observation.state` `[14]`; `robot/qvel` and `robot/qf` `[14]`; three RGB cameras `cam_high`, `cam_left_wrist`, `cam_right_wrist`. The adapter takes the first three channels, converts HWC to CHW, and scales uint8-like image values to `[0,1]` once. Qpos includes the observable gripper state.
- Action: `[0:6]` left-arm absolute joint targets, `[6]` left-gripper policy command, `[7:13]` right-arm absolute joint targets, `[13]` right-gripper policy command. Both gripper policy commands are `[0,1]`.
- LeRobot ACT loads normalization buffers directly from `model.safetensors`. The current1500 training run records normalization fingerprint `e5c0de98a7d349cf346de23b57b11a741723ec3ae4ea5dcf96e94de9b41d2310`, unchanged from the original ACT contract. Do not recompute statistics or normalize actions externally. The adapter maps each gripper command **once** to its environment action-space low/high physical range before `env.step()`.
- Final inference uses only ordinary RGB and robot observation. It does not require simulator object/button/slide ground truth, teacher graph, expert affordances or a runtime teacher. The byte-identical source adapter contains a dormant `debug_button_press` diagnostic branch that can read simulator state; the submitted configuration and launcher force this flag to `False`, so it is not part of final policy operation.

## Code and deployment

`policy/submitted_act/deploy_policy.py` is a byte-identical copy of the generic `policy/act` ACT adapter. The package initializer and `official_contract.py` add only the official evaluator's fourth return value: measured times for genuine ACT chunk inference. The formal 57/100 videos were stored by the evaluator under `.../act/random/...`, whereas the experimental `experiments/manipulate_pipette/act_policy/` package identifies itself as `experiments.manipulate_pipette.act_policy` and includes optional refresh/T300 diagnostic code. That experimental package is intentionally **not** submitted as the final Pure ACT adapter. The video path supports the generic adapter identity; no immutable run manifest was found inside the video directory to independently prove the model hash loaded for that historical evaluation.

`config/deploy_policy.yml` retains ACT inference settings while relocating `policy_name` to `submitted_act` and setting the local-only `eval_freeze_interval_events` option to `false`. The prior `true` setting converted interval randomization events into reset events in the modified local evaluator; the inspected official evaluator does not perform that conversion. The launcher detects the evaluator's three- or four-result interface and fails closed on an unknown interface. No model output or action conversion is changed. This package has passed static/mock checks but **has not been used for a new rollout**; 57/100 belongs to the earlier local source-project conditions and must not be presented as official-condition performance. See `docs/OFFICIAL_RUNTIME_COMPATIBILITY.md`.

This submission includes model, adapter and launcher, but **requires the RoboSynChallenge runtime** (task environment, simulator, `scripts/eval_policy.py`, `robosynchallenge` package, EmbodiChain/DexSim and dependencies). It does not bundle the entire main repository. It does not read training datasets during inference.

Example from Linux, without running it as part of packaging:

```bash
SUBMISSION_ROOT=/path/to/manipulate_pipette
export ROBOSYNCHALLENGE_ROOT=/path/to/RoboSynChallenge
cd "$SUBMISSION_ROOT"
bash deploy/eval.sh 0 100 0
```

Arguments are GPU ID, episode count, and evaluator seed. The launcher verifies checkpoint SHA256 and unsets intervention/debug environment variables before calling the unmodified project evaluator. Set the two path variables to the locations on the target machine; no path inside the submitted package needs editing.

## Model selection and exclusions

The later failure-matched 250-step checkpoint was excluded: recovery first20 improved less than 1% versus clean current1500 while normal control degraded; offline gate failed. The matched three-stream 100-step checkpoint was excluded: it triggered hard early-stop, with matched normal t300 first20 error rising 2.94×. Neither failed checkpoint, optimizer state, dataset, video, log, intermediate audit, or training artifact is present here.

Use `MANIFEST.txt` for every packaged file and its purpose, and `SHA256SUMS.txt` to check integrity. No source-project file was moved, deleted, or modified to create this package.
