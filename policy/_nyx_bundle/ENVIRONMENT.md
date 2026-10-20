# Runtime and installation

Target: Linux, Python 3.11, a compatible NVIDIA driver and the organizer's RoboSynChallenge / EmbodiChain / DexSim runtime. Windows is used here for packaging only. The simulator is not bundled.

1. Install the official runtime: https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html
2. Use its CUDA-compatible PyTorch environment. Existing ACT packages record torch 2.7.1, torchvision 0.22.1, LeRobot 0.3.3 and NumPy 1.26.4. Some historical runs used CUDA 12.6, EmbodiChain 0.2.4 and DexSim 0.4.3; these are evidence, not a requirement to replace the organizer's simulator.
3. Review `requirements-common.txt` and each task's `requirements-runtime.txt` in an isolated environment. A reference installation is `python -m pip install -r requirements-common.txt`. For sample_loading additionally install `requirements-sample-loading.txt` and its task requirements. No newly tested universal dependency lock is claimed.
4. Ensure ffmpeg is available for video recording. Set `ROBOSYNCHALLENGE_ROOT`, `PYTHON_BIN` and, if needed, `EMBODICHAIN_ROOT` to actual absolute Linux paths.
5. Run a dry run and one simulator episode per task as in README.md. Collect package hash, official commit, environment versions, console log and evaluation_metrics.json. Only then record a successful runtime validation.

When starting from the GitHub code-only release, restore the Hugging Face assets first. The model repository must mirror this submission's relative paths:

```bash
python -m pip install huggingface_hub
python download_assets.py --repo-id ORG/REPO --revision PINNED_REVISION
python -B download_assets.py --verify-only
```

Record the real environment and per-task results in `OFFICIAL_RUNTIME_VALIDATION.json`; do not replace `not_run` with `passed` without a model-loaded simulator run.

All relative model paths are resolved inside this submission. Training paths in train_config.json are provenance, not runtime dataset dependencies. sample_loading also requires the bundled `_runtime` trajectory/model code and calibration assets. items_handover requires both ACT checkpoints, the RGB regressor, normalizers, RBF prior and gripper schedule; do not upload only its primary checkpoint.

The unified runner registers the package as `policy.nyx_<task>` in memory and executes the unmodified official evaluator. It does not patch task configuration, success conditions, randomizer code or evaluator functions. Existing per-task launchers are retained for historical reproducibility; use the unified runner for this submission.
