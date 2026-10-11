# drawer_open_place submission package

This package contains the frozen 002500 ACT policy, its complete checkpoint,
the task adapter, and a portable launcher. It requires an existing, unmodified
RoboSynChallenge/EmbodiChain simulator runtime; it does not bundle the simulator.

## Contents and provenance

- Policy: `policy/drawer_open_place/deploy_policy.py` (four-result evaluator interface).
- Checkpoint: `checkpoint/pretrained_model/` copied without modifying its three files
  from `RoboSynChallenge_official_clean_20260920/policy/drawer_open_place_placement_weighted_002500_frozen/checkpoint/pretrained_model/`.
- Checkpoint SHA256: `config.json` `c5e76c7a78ee99414f212b4e7c89db169a6185a4bc47098fec2177a3a6338509`;
  `model.safetensors` `0e62f070480f10dc768f49de3307e713809ec3c68da9eaaf5b50057480803efe`.
- The historical training paths inside `train_config.json` are provenance only.

## Runtime

The recorded source environment was Python 3.11.15, LeRobot 0.3.3,
PyTorch 2.7.1+cu126, NumPy 1.26.4 and DexSim 0.4.3. See
`requirements-runtime.txt`; install the official simulator/runtime first.

On Linux, with the package mounted or copied to a local path:

```bash
export ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge
export PYTHON_BIN="$ROBOSYNCHALLENGE_ROOT/policy/act/.venv/bin/python"
cd /path/to/drawer_open_place
bash policy/drawer_open_place/eval.sh 0 1 0
```

The positional arguments are optional checkpoint path, GPU ID, episode count,
and seed. The packaged checkpoint is used when no checkpoint path is supplied.
The launcher validates the evaluator's four-result interface but does not
force a historical Git commit, task-config hash, or hidden-test overlay.
An unsupported evaluator contract fails closed rather than changing evaluator
code. It does not modify the official task, randomizers, or success check.

## Validation boundary

The frozen source has a historical local final-gate report under its source
directory. The *E-drive packaged launcher* has only been inspected statically;
it has not been run in the official Linux simulator from this package. Run the
one-episode smoke above, inspect model loading, action shape, return contract,
and simulator exit, then conduct the organizer's full evaluation. Do not report
the source gate result as a fresh packaged-entry result.

The code repository URL, checkpoint hosting URL, data declaration, and
organizer submission form are external submission steps, not fabricated here.
