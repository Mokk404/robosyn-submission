#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROBOSYNCHALLENGE_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "$ROBOSYNCHALLENGE_ROOT" && pwd)"
elif [[ -f "$SCRIPT_DIR/../../scripts/eval_policy.py" ]]; then
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
elif [[ -f "$SCRIPT_DIR/../../RoboSynChallenge/scripts/eval_policy.py" ]]; then
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../RoboSynChallenge" && pwd)"
else
  echo "Cannot locate RoboSynChallenge; set ROBOSYNCHALLENGE_ROOT." >&2
  exit 2
fi

TASK_NAME="${1:-water_pouring}"
SETTING="${2:-random}"
CHECKPOINT_PATH="${3:-$SCRIPT_DIR/checkpoint}"
GPU_ID="${4:-0}"
shift "$(( $# < 4 ? $# : 4 ))"
if [[ "$TASK_NAME" != "water_pouring" ]]; then
  echo "This package only supports water_pouring." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)/EmbodiChain}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR/compat_imports_v1:$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u "$SCRIPT_DIR/manual_eval.py" \
  --config "$SCRIPT_DIR/deploy_policy.yml" \
  --overrides \
  --task_name "$TASK_NAME" \
  --setting "$SETTING" \
  --policy_name water_pouring \
  --model_name water_pouring_spatial_000200 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  "$@"
