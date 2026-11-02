#!/usr/bin/env bash
# Dependencies only: setup never starts training or enables services.
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin="${DRIVING_PYTHON:-python3}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3, 11), "Python 3.11+ required"'
if [[ ! -x .venv/bin/python ]]; then
    "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
torch_args=()
if [[ -n "${DRIVING_TORCH_INDEX:-}" ]]; then
    torch_args+=(--index-url "$DRIVING_TORCH_INDEX")
fi
.venv/bin/python -m pip install 'torch==2.13.0' "${torch_args[@]}"
.venv/bin/python -m pip install 'numpy>=2.0'
.venv/bin/python -m pip freeze > requirements-local.lock.txt
printf 'Ready. No services installed or started. Start manually with ./scripts/run.sh.\n'
