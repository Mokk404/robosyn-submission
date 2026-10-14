# Official evaluator entrypoints for all ten tasks

The preferred commands are the ten `policy/nyx_TASK/eval.sh` commands in
[README.md](README.md). They register the on-disk adapter in the official policy
namespace and execute the unmodified official evaluator. No simulator source
patch is required.

## Direct official import mode (Linux)

If the evaluator is invoked directly instead of through the supplied launcher,
link the adapters into its policy search directory. Keep the submission clone
and all restored assets in place. These commands do not overwrite existing
paths: stop and inspect any `File exists` error.

```bash
export SUBMISSION_ROOT=/absolute/path/to/robosyn-submission
export ROBOSYNCHALLENGE_ROOT=/absolute/path/to/official/RoboSynChallenge
export PYTHON_BIN=/absolute/path/to/python
export CUDA_VISIBLE_DEVICES=0
# After restoring assets using README.md:
ln -s "$SUBMISSION_ROOT/policy/_submission.py" "$ROBOSYNCHALLENGE_ROOT/policy/_submission.py"
for adapter in "$SUBMISSION_ROOT"/policy/nyx_*; do
  ln -s "$adapter" "$ROBOSYNCHALLENGE_ROOT/policy/$(basename "$adapter")" || exit 1
done
cd "$ROBOSYNCHALLENGE_ROOT"
```

Each configuration selects its task and its corresponding adapter. Paths in the
configuration are resolved against the original submission clone, regardless of
current working directory. An explicitly supplied absolute checkpoint override
is preserved.

```bash
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_click_bell/deploy_policy.yml" --overrides --task_name click_bell --policy_name nyx_click_bell --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_drawer_open_place/deploy_policy.yml" --overrides --task_name drawer_open_place --policy_name nyx_drawer_open_place --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_handle_basket/deploy_policy.yml" --overrides --task_name handle_basket --policy_name nyx_handle_basket --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_item_assembly/deploy_policy.yml" --overrides --task_name item_assembly --policy_name nyx_item_assembly --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_items_handover/deploy_policy.yml" --overrides --task_name items_handover --policy_name nyx_items_handover --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_manipulate_pipette/deploy_policy.yml" --overrides --task_name manipulate_pipette --policy_name nyx_manipulate_pipette --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_mixer_operating/deploy_policy.yml" --overrides --task_name mixer_operating --policy_name nyx_mixer_operating --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_sample_loading/deploy_policy.yml" --overrides --task_name sample_loading --policy_name nyx_sample_loading --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_table_rearrangement/deploy_policy.yml" --overrides --task_name table_rearrangement --policy_name nyx_table_rearrangement --setting random --max_episodes 100 --seed 0 --headless True
"$PYTHON_BIN" scripts/eval_policy.py --config "policy/nyx_water_pouring/deploy_policy.yml" --overrides --task_name water_pouring --policy_name nyx_water_pouring --setting random --max_episodes 100 --seed 0 --headless True
```

`deploy_policy.py` loads the frozen task package only when called. Actual model
loading additionally requires each task's dependencies and restored assets.
The official evaluator interface is documented at
https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
and implemented at
https://github.com/EDEM-AI/RoboSynChallenge/blob/main/scripts/eval_policy.py.
