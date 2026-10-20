# All ten task entrypoints

Use policy/README.md for setup and the full checkpoint mapping.

From the official repository root:

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

Direct official evaluator mode after preparing assets:

```bash
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
