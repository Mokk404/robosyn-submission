# RoboSynChallenge: complete ten-task policy submission

Updated 2026-10-14 for preparation receipt **eval_50aaeafd932748**.
This code repository includes the full submitted policy source and ten standard
`policy/<policy_name>/` entrypoints. Model weights and auxiliary runtime assets
are hosted separately on Hugging Face.

## Standard policy layout

Every task has `__init__.py`, `deploy_policy.py`, `deploy_policy.yml`, and `eval.sh`
in the policy directory listed below. The three exported functions are
`get_model(usr_args)`, `eval(env, model, obs)`, and `reset_model(model)`.
`eval` preserves the existing four-result interface including inference timings.

The adapters delegate to the complete frozen source packages kept under each
root task directory. Keep the **entire repository**; do not copy only the small
adapter files. `policy/_submission.py` resolves assets relative to this clone and
loads the original implementations with distinct names. This keeps task-local
imports, relative auxiliary paths, and existing control settings intact.

| Task | Policy name | Configuration | Frozen source | Fixed checkpoint URL |
| --- | --- | --- | --- | --- |
| `click_bell` | `nyx_click_bell` | [config](policy/nyx_click_bell/deploy_policy.yml) | [implementation](click_bell/policy/submitted_act/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/click_bell) |
| `drawer_open_place` | `nyx_drawer_open_place` | [config](policy/nyx_drawer_open_place/deploy_policy.yml) | [implementation](drawer_open_place/policy/drawer_open_place/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/drawer_open_place) |
| `handle_basket` | `nyx_handle_basket` | [config](policy/nyx_handle_basket/deploy_policy.yml) | [implementation](handle_basket/policy/act_handle_basket_submission_v1/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/handle_basket) |
| `item_assembly` | `nyx_item_assembly` | [config](policy/nyx_item_assembly/deploy_policy.yml) | [implementation](item_assembly/./deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/item_assembly) |
| `items_handover` | `nyx_items_handover` | [config](policy/nyx_items_handover/deploy_policy.yml) | [implementation](items_handover/policy/items_handover/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/items_handover) |
| `manipulate_pipette` | `nyx_manipulate_pipette` | [config](policy/nyx_manipulate_pipette/deploy_policy.yml) | [implementation](manipulate_pipette/policy/submitted_act/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/manipulate_pipette) |
| `mixer_operating` | `nyx_mixer_operating` | [config](policy/nyx_mixer_operating/deploy_policy.yml) | [implementation](mixer_operating/policy/act_mixer_submission_v1/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/mixer_operating) |
| `sample_loading` | `nyx_sample_loading` | [config](policy/nyx_sample_loading/deploy_policy.yml) | [implementation](sample_loading/./deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/sample_loading) |
| `table_rearrangement` | `nyx_table_rearrangement` | [config](policy/nyx_table_rearrangement/deploy_policy.yml) | [implementation](table_rearrangement/policy/act_table_physical_v1/deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/table_rearrangement) |
| `water_pouring` | `nyx_water_pouring` | [config](policy/nyx_water_pouring/deploy_policy.yml) | [implementation](water_pouring/./deploy_policy.py) | [checkpoint assets](https://huggingface.co/Lastforever16/robosyn-ten-task-models/tree/da23c29b897034d0ca44f703d09985f21d629def/water_pouring) |

## Setup

Use Linux, Python 3.11, a compatible NVIDIA/CUDA stack, and the official
RoboSynChallenge/EmbodiChain runtime. Follow [ENVIRONMENT.md](ENVIRONMENT.md)
and task-specific `SUBMISSION.md` files for dependencies. Reference versions
are recorded in `requirements-common.txt`; sample_loading has additional
requirements. Install simulator-compatible dependencies before running.

From the submission clone root, restore and verify all 396 runtime assets:

```bash
python -m pip install huggingface_hub
python download_assets.py --repo-id Lastforever16/robosyn-ten-task-models --revision da23c29b897034d0ca44f703d09985f21d629def
python download_assets.py --verify-only
export ROBOSYNCHALLENGE_ROOT=/absolute/path/to/official/RoboSynChallenge
export PYTHON_BIN=/absolute/path/to/python
# If required by the simulator installation:
export EMBODICHAIN_ROOT=/absolute/path/to/EmbodiChain
```

Preserve the downloaded directory structure, including all hybrid model,
calibration, normalization, and trajectory assets. [ASSETS.json](ASSETS.json)
records their hashes; [TASKS.json](TASKS.json) records frozen implementation paths.

## All ten task entrypoints

Run these from this repository root. Each script invokes the unchanged official
evaluator with its corresponding standard policy adapter and config:

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

For a path-only check append `--dry-run`; for a simulator smoke test use
`--episodes 1`. The organizer may change seeds, episode counts, and `--setting`.
The root launcher remains available as `bash eval.sh --task TASK ...`.
See [ENTRYPOINTS.md](ENTRYPOINTS.md) for direct official-evaluator installation
and commands for all ten policies.

## Validation and scope

```bash
python -B tests/test_runner.py
python -B tests/test_policy_tree.py
python -B verify_submission.py --integrity
```

The 2026-10-14 correction checks adapter discovery, task coverage, configuration
and asset path resolution, delegation of the public interface, and all 396 asset
hashes. They do not perform model inference or certify a GPU simulator rollout.
The model weights and original task control implementations are unchanged.
Table's physical gripper handling remains in its original submitted policy.

## Provenance and outstanding compatibility questions

See [DATA_SOURCES.md](DATA_SOURCES.md), [DATA_PROVENANCE.json](DATA_PROVENANCE.json),
and [PENDING.md](PENDING.md). Available provenance covers simulation demonstrations
and task-specific visual/calibration assets; exact counts and revisions are not
fully established for every component. Some ACT policies require qvel/qf.
`sample_loading` uses robot FK/IK access whose organizer approval remains
outstanding. The preparation-layout correction does not resolve those separate
questions or claim organizer approval.

Historical status and portal templates dated 2026-10-11 are background records;
the entrypoint mapping and fixed checkpoint URLs above describe this correction.
