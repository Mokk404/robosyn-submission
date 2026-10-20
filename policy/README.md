# Self-contained ten-task policy tree

Packaging-only correction for eval_50aaeafd932748, dated 2026-10-20.
Baseline: GitHub commit 4adbc13a7aaa62533bc5a2f06a01e4e9f039c0ce.

Copy/merge the complete `policy/` tree into the official RoboSynChallenge
repository. No file from the participant repository root is required at runtime,
for setup, or for verification. Retain all nested source directories.

```text
policy/
  nyx_<task>/                 # 10 standard adapters, each with four entry files
  _submission.py             # imports implementations from the private bundle
  run_task.py                # unchanged official evaluator invocation
  eval.sh                    # optional generic wrapper
  prepare_assets.py          # use organizer's existing local downloads
  download_assets.py         # optional pinned HF download
  verify_bundle.py           # source/provenance and optional asset checks
  _nyx_bundle/
    TASKS.json
    ASSETS.json
    requirements-common.txt
    requirements-sample-loading.txt
    <task>/                  # all 10 complete original implementation trees
  tests/
  SOURCE_SHA256SUMS
  PACKAGING_PROVENANCE.json
```

## Runtime dependencies

Use the organizer's official Linux simulator and compatible CUDA/PyTorch/Python
runtime. Reference ACT versions are unchanged from the previous submission:
Python 3.11, torch 2.7.1, torchvision 0.22.1, LeRobot 0.3.3, numpy 1.26.4.
See `_nyx_bundle/ENVIRONMENT.md`, requirements files and the original per-task
`SUBMISSION.md` records for task-specific dependencies. Those original records
are historical; the paths and setup commands in this README supersede their
old participant-root commands. The sample-loading runtime/model source is
included under `_nyx_bundle/sample_loading/_runtime/`.

Do not reinstall or modify official simulator, collector or scoring code.
Only add the participant policy tree and its required Python dependencies.
Set `PYTHON_BIN` to the actual official environment interpreter.

```bash
cd /absolute/path/to/official/RoboSynChallenge
export PYTHON_BIN=/absolute/path/to/official/environment/bin/python
# Only if the official installation requires this:
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
```

## Reuse the evaluator's existing downloads (preferred)

The original checkpoint repository, revision and internal file paths are
unchanged. If the evaluator already downloaded a complete HF snapshot with the
ten task directories, prepare local links and verify the pinned SHA-256 hashes:

```bash
"$PYTHON_BIN" policy/prepare_assets.py --asset-root /absolute/path/to/existing/hf/snapshot
```

No network is used by this command. It links individual unchanged asset files
into `_nyx_bundle/<task>/` so frozen implementations can retain their original
relative auxiliary paths. It never rewrites checkpoint contents. Use `--copy`
if symlinks are unavailable; copied bytes are the same pinned files. Existing
incorrect files cause an error instead of being silently replaced.

If each task is downloaded to a different directory, write a JSON map (location
of this JSON itself is arbitrary). Each task value must be the **contents of
that task's HF subdirectory**, not its parent snapshot:

```json
{
  "table_rearrangement": "/existing/download/table_task",
  "click_bell": "/existing/download/bell_task",
  "items_handover": "/existing/download/handover_task",
  "sample_loading": "/existing/download/loading_task"
}
```

Add entries for the other six tasks if their assets are also outside the bundle.
Exact-file mappings are also supported, for example:

```json
{
  "table_rearrangement/policy/act_table_physical_v1/checkpoint/model.safetensors": "/existing/cache/model.safetensors"
}
```

Task-directory and exact-file entries can be combined in one map. Relative map
values are resolved from the map file's directory.

```bash
"$PYTHON_BIN" policy/prepare_assets.py --asset-map /absolute/path/to/download-map.json
"$PYTHON_BIN" policy/verify_bundle.py --assets
```

For deferred preparation through the submitted wrapper, `--asset-root` or
`--asset-map` may be supplied to any task `eval.sh`. Alternatively set
`NYX_ASSET_ROOT` or `NYX_ASSET_MAP`. For direct official evaluator calls, the same
paths can be configured through environment variables or `asset_root` /
`asset_map` config keys. Explicit absolute checkpoint-path overrides remain
supported by the adapter. Hybrid task auxiliary files still require preparation.

## Optional fresh asset download

Only use this if local downloads are unavailable. The download helper is inside
the retained policy tree and fetches only manifest-listed runtime files; it does
not overwrite policy source or metadata with the HF repository README.

```bash
"$PYTHON_BIN" policy/download_assets.py --repo-id Lastforever16/robosyn-ten-task-models --revision da23c29b897034d0ca44f703d09985f21d629def
"$PYTHON_BIN" policy/download_assets.py --verify-only
```

All 396 asset hashes, configs and checkpoint contents remain pinned to this
original revision. Keep complete hybrid auxiliaries, including normalizers,
calibration, residuals, RGB model weights and trajectory libraries.

## All ten entrypoints

These commands use the same defaults as the previous submission (public random,
100 episodes, seed 0, GPU 0). The organizer can supply its own evaluation
parameters. No seed, action limit, preprocessing, chunk/replan setting, gripper
conversion or inference-timing logic is changed by this correction.

```bash
bash policy/nyx_click_bell/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_drawer_open_place/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_handle_basket/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_item_assembly/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_items_handover/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_manipulate_pipette/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_mixer_operating/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_sample_loading/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_table_rearrangement/eval.sh --gpu 0 --episodes 100 --seed 0
bash policy/nyx_water_pouring/eval.sh --gpu 0 --episodes 100 --seed 0
```

Use `--episodes 1` for a smoke test or append `--dry-run` for a path-only check.
Extra official evaluator options can be passed after `--`. The wrapper uses
`ROBOSYNCHALLENGE_ROOT` if set; otherwise it assumes the official repository
is the parent of the retained `policy/` directory.

### Direct official evaluator commands

After local asset preparation, the official evaluator can import all adapters
directly. No participant launcher, TASKS.json or setup utility is required from
the official repository root.

```bash
export CUDA_VISIBLE_DEVICES=0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_click_bell/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_drawer_open_place/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_handle_basket/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_item_assembly/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_items_handover/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_manipulate_pipette/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_mixer_operating/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_sample_loading/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_table_rearrangement/deploy_policy.yml --overrides --max_episodes 100 --seed 0
"$PYTHON_BIN" scripts/eval_policy.py --config policy/nyx_water_pouring/deploy_policy.yml --overrides --max_episodes 100 --seed 0
```

### Fixed checkpoint mapping

| Task | Config | Original fixed checkpoint/asset URL |
| --- | --- | --- |
| click_bell | `policy/nyx_click_bell/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/click_bell |
| drawer_open_place | `policy/nyx_drawer_open_place/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/drawer_open_place |
| handle_basket | `policy/nyx_handle_basket/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/handle_basket |
| item_assembly | `policy/nyx_item_assembly/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/item_assembly |
| items_handover | `policy/nyx_items_handover/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/items_handover |
| manipulate_pipette | `policy/nyx_manipulate_pipette/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/manipulate_pipette |
| mixer_operating | `policy/nyx_mixer_operating/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/mixer_operating |
| sample_loading | `policy/nyx_sample_loading/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/sample_loading |
| table_rearrangement | `policy/nyx_table_rearrangement/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/table_rearrangement |
| water_pouring | `policy/nyx_water_pouring/deploy_policy.yml` | https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/water_pouring |

## Checks and scope

```bash
"$PYTHON_BIN" policy/verify_bundle.py
"$PYTHON_BIN" policy/verify_bundle.py --assets
"$PYTHON_BIN" policy/tests/test_retained_tree.py
# Full isolated packaging test using an existing unchanged snapshot:
"$PYTHON_BIN" policy/tests/test_retained_tree.py --asset-root /existing/hf/snapshot
```

`PACKAGING_PROVENANCE.json` records unchanged baseline task files. The isolated
retention test copies only policy/ into a temporary official-layout fixture:
no participant-root launcher, metadata, setup tool or task directories exist.
It exercises asset mapping, all ten launchers and official-name adapter imports.
Its minimal evaluator fixture is not a simulator; model/interface tests use
model test doubles. These checks do not certify a GPU rollout, performance or
organizer approval.

Previously disclosed observation-field and FK/IK compatibility questions remain
unchanged. See `_nyx_bundle/PENDING.md` and data provenance records. Model weights
and the underlying task-control implementations are preserved.
