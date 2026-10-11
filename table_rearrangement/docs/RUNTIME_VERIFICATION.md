# table_rearrangement runtime verification

The package records dependency versions observed in the existing ACT Linux
virtual environment: Python 3.11.15, LeRobot 0.3.3, NumPy 1.26.4,
PyTorch 2.7.1 (CUDA 12.6 build), torchvision 0.22.1, safetensors 0.8.0,
av 18.1.0, OpenCV 4.11.0.86 and DexSim 0.4.3. The official
RoboSynChallenge/EmbodiChain simulator must be provisioned separately.

The E-drive package has passed prior static/package checks, but **its
checkpoint load and simulator rollout have not been re-run after packaging**.
The source project's historical result is not evidence of the packaged entry.
Do not claim a newly verified official success rate.

Linux smoke command after placing the package on a path visible to Linux:

```bash
export ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge
export PYTHON_BIN="$ROBOSYNCHALLENGE_ROOT/policy/act/.venv/bin/python"
cd /path/to/table_rearrangement
bash policy/act_table_physical_v1/eval.sh 0 --max_episodes 1 --seed 51001
```

Pass criteria: official evaluator imports this package, loads its exact
checkpoint, completes one episode, records four policy return values and a
valid action, and exits without exception. A full official evaluation remains
separate. No official task/config/overlay modification is requested.
