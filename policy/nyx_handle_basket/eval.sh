#!/usr/bin/env bash
set -euo pipefail
HERE="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
exec "${PYTHON_BIN:-python3}" -B -u "$HERE/../run_task.py" --task handle_basket "$@"
