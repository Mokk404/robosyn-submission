#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
SEED="${1:-0}"
RUN_NAME="${2:-formal_v60_seed${SEED}_random100}"
MAX_EPISODES="${3:-100}"
GPU_ID="${4:-0}"
RESULT_ROOT="${5:-$REPO_ROOT/eval_result}"
TRACE_ROOT="${6:-$RESULT_ROOT/${RUN_NAME}_traces}"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONDONTWRITEBYTECODE=1
export EMBODICHAIN_ROOT
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PRIMARY="$SCRIPT_DIR/assets/primary_act"
AUXILIARY="$SCRIPT_DIR/assets/auxiliary_act"
test -f "$PRIMARY/model.safetensors"
test -f "$AUXILIARY/model.safetensors"
mkdir -p "$RESULT_ROOT"
cd "$REPO_ROOT"
exec "$PYTHON_BIN" scripts/eval_policy.py \
  --config "policy/items_handover/deploy_policy.yml" \
  --overrides \
  --checkpoint_path "$PRIMARY" \
  --auxiliary_checkpoint_path "$AUXILIARY" \
  --train_config_name "$RUN_NAME" \
  --model_name rgb_dual_v60_frozen \
  --seed "$SEED" \
  --max_episodes "$MAX_EPISODES" \
  --headless True \
  --eval_video_log True \
  --max_steps None \
  --eval_result_dir "$RESULT_ROOT" \
  --public_trace_root "$TRACE_ROOT"
