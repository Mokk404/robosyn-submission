#!/usr/bin/env bash
# Pure ACT evaluator. Requires the unmodified RoboSynChallenge runtime.
# Usage: ROBOSYNCHALLENGE_ROOT=/path/to/RoboSynChallenge bash eval.sh [gpu_id] [episodes] [seed]
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="${ROBOSYNCHALLENGE_ROOT:?Set ROBOSYNCHALLENGE_ROOT to the RoboSynChallenge project root}"
REPO_ROOT="$(cd "$REPO_ROOT" && pwd)"
CHECKPOINT="$PACKAGE_ROOT/checkpoint/current1500/pretrained_model"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
GPU_ID="${1:-0}"
EPISODES="${2:-20}"
SEED="${3:-0}"

test -f "$CHECKPOINT/config.json"
test -f "$CHECKPOINT/model.safetensors"
test -f "$REPO_ROOT/scripts/eval_policy.py"
test -x "$PYTHON_BIN"
if grep -Fq 'obs, info, truncated, new_times = policy_pkg.eval' "$REPO_ROOT/scripts/eval_policy.py"; then
  export SUBMITTED_ACT_EVAL_CONTRACT=official4
elif grep -Fq 'obs, info, truncated = policy_pkg.eval' "$REPO_ROOT/scripts/eval_policy.py"; then
  export SUBMITTED_ACT_EVAL_CONTRACT=local3
else
  echo 'Unsupported evaluator return contract; refusing to run.' >&2
  exit 1
fi
echo "[SUBMISSION EVAL CONTRACT] $SUBMITTED_ACT_EVAL_CONTRACT"
echo "025e02433a3e8c844093550fff0d6db5e7494b412f6e74dce7674910b36874e9  $CHECKPOINT/config.json" | sha256sum --check --status
echo "5318fbb1d4c371290931d083c2fe76e257a190e803965386648f6518a23f9fec  $CHECKPOINT/model.safetensors" | sha256sum --check --status

unset ACT_PRESS_REFRESH_STEP ACT_T300_RECOVERY_DIAGNOSTIC ACT_T300_RECOVERY_DELTA_PATH
unset ACT_EXEC_TRACE_PATH ACT_REPRO_AUDIT_PATH ACT_REPRO_SEED
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PACKAGE_ROOT/policy:$REPO_ROOT:$REPO_ROOT/policy${PYTHONPATH:+:$PYTHONPATH}"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$(dirname "$REPO_ROOT")/EmbodiChain}"
if [[ -d "$EMBODICHAIN_ROOT" ]]; then
  export PYTHONPATH="$EMBODICHAIN_ROOT:$PYTHONPATH"
fi

cd "$REPO_ROOT"
"$PYTHON_BIN" "$PACKAGE_ROOT/deploy/run_eval.py" \
  --config "$PACKAGE_ROOT/config/deploy_policy.yml" \
  --overrides \
  --task_name manipulate_pipette \
  --setting random \
  --checkpoint_path "$CHECKPOINT" \
  --model_name pretrained_model \
  --max_episodes "$EPISODES" \
  --seed "$SEED" \
  --headless True \
  --n_action_steps 50 \
  --action_chunk_offset 0 \
  --debug_button_press False \
  --debug_trajectory False
