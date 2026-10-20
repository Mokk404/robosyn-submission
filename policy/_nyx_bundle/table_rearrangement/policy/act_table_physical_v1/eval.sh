#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ROBOSYNCHALLENGE_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="$SCRIPT_DIR/checkpoint"
# Preserve the existing CHECKPOINT [GPU] calling convention.
if [[ $# -gt 0 && ! "$1" =~ ^[0-9]+$ && "$1" != --* ]]; then
  CHECKPOINT="$1"; shift
fi
GPU=0
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then GPU="$1"; shift; fi
test -f "$REPO_ROOT/scripts/eval_policy.py"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u -X faulthandler "$SCRIPT_DIR/entrypoint.py" "$REPO_ROOT" \
  --config "$SCRIPT_DIR/deploy_policy.yml" --overrides \
  --checkpoint_path "$CHECKPOINT" "$@"
