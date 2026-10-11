# handle_basket runtime verification

The package records dependency versions observed in the existing ACT Linux
virtual environment: Python 3.11.15, LeRobot 0.3.3, NumPy 1.26.4,
PyTorch 2.7.1 (CUDA 12.6 build), torchvision 0.22.1, safetensors 0.8.0,
av 18.1.0, OpenCV 4.11.0.86 and DexSim 0.4.3. The official
RoboSynChallenge/EmbodiChain simulator must be provisioned separately.

The package has a historical 90/100 local random100 result, but **the E-drive
packaged entry has not been re-run in the official Linux simulator**.
Do not turn the historical score into a new package-validation claim.

Linux smoke command after placing the package on a path visible to Linux:

```bash
export ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge
export PYTHON_BIN="$ROBOSYNCHALLENGE_ROOT/policy/act/.venv/bin/python"
cd /path/to/handle_basket
bash policy/act_handle_basket_submission_v1/eval.sh 0 --max_episodes 1 --seed 0
```

Pass criteria: unchanged official evaluator imports this package, loads its
exact checkpoint, completes one episode, records the four-result policy
interface and a valid action, and exits without exception. A fresh full-task
evaluation must be recorded separately.
