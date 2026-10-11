#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="${ROBOSYNCHALLENGE_ROOT:?Set ROBOSYNCHALLENGE_ROOT to the official RoboSynChallenge root}"
REPO_ROOT="$(cd -- "$REPO_ROOT" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"

CHECKPOINT_PATH="$PACKAGE_ROOT/checkpoint/pretrained_model"
if [[ $# -gt 0 && ! "$1" =~ ^[0-9]+$ && "$1" != --* ]]; then
  CHECKPOINT_PATH="$1"
  shift
fi
GPU_ID="${1:-0}"
MAX_EPISODES="${2:-1}"
SEED="${3:-0}"

test -f "$REPO_ROOT/scripts/eval_policy.py"
test -f "$CHECKPOINT_PATH/config.json"
test -f "$CHECKPOINT_PATH/model.safetensors"
test -x "$PYTHON_BIN"
if ! grep -Fq 'obs, info, truncated, new_times = policy_pkg.eval' "$REPO_ROOT/scripts/eval_policy.py"; then
  echo 'This package requires the official four-result evaluator interface.' >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONDONTWRITEBYTECODE=1
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/entrypoint.py" "$REPO_ROOT" \
  --config "$SCRIPT_DIR/deploy_policy.yml" \
  --overrides \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --task_name drawer_open_place \
  --setting random \
  --model_name drawer_placement_weighted_002500 \
  --max_episodes "$MAX_EPISODES" \
  --seed "$SEED" \
  --headless True
