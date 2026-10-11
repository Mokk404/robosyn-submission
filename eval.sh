#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python3}" -B -u "$HERE/run_task.py" "$@"
