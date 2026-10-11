#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${ROBOSYNCHALLENGE_ROOT:-}" ]]; then
  REPO_ROOT="$(cd -- "$ROBOSYNCHALLENGE_ROOT" && pwd)"
elif [[ -f "$SCRIPT_DIR/../../RoboSynChallenge_official_clean_20260919_v6_alternate/scripts/eval_policy.py" ]]; then
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../RoboSynChallenge_official_clean_20260919_v6_alternate" && pwd)"
elif [[ -f "$SCRIPT_DIR/../../RoboSynChallenge/scripts/eval_policy.py" ]]; then
  REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../RoboSynChallenge" && pwd)"
else
  echo "Cannot locate RoboSynChallenge; set ROBOSYNCHALLENGE_ROOT." >&2
  exit 2
fi

TASK_NAME="${1:-sample_loading}"
SETTING="${2:-random}"
GPU_ID="${3:-0}"
shift "$(( $# < 3 ? $# : 3 ))"

if [[ "$TASK_NAME" != "sample_loading" ]]; then
  echo "This package only supports sample_loading." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/policy/act/.venv/bin/python}"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$(cd -- "$REPO_ROOT/.." && pwd)/EmbodiChain}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCRIPT_DIR/_runtime:$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -u scripts/eval_policy.py \
  --config "$SCRIPT_DIR/deploy_policy.yml" \
  --overrides \
  --task_name "$TASK_NAME" \
  --setting "$SETTING" \
  --policy_name sample_loading_90plus_visual_v89_public_obs_delayed_full_release_v1 \
  --model_name sample_loading_v89_public_obs \
  --checkpoint_path "$SCRIPT_DIR/checkpoint/pose_regressor_v4.ckpt" \
  --cube_highres_checkpoint_path "$SCRIPT_DIR/checkpoint/cube_highres_pose_v1.ckpt" \
  --rack_highres_checkpoint_path "$SCRIPT_DIR/checkpoint/rack_highres_pose_v1.ckpt" \
  --rgb_dynamic_checkpoint_path "$SCRIPT_DIR/checkpoint/dynamic_pose_rgb_v3_epoch029.ckpt" \
  --left_grasp_calibration_path "$SCRIPT_DIR/config/left_grasp_calibration.json" \
  --diagnostic_history_policy_path "$SCRIPT_DIR/_runtime/policy" \
  --diagnostic_trainer_path "$SCRIPT_DIR/_runtime/model_code" \
  "$@"
