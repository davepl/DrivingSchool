#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv_bin="${DRIVING_UV:-uv}"
python_bin="${DRIVING_PYTHON:-python3}"
"$uv_bin" venv --python "$python_bin" .venv
"$uv_bin" pip install --python .venv/bin/python torch==2.13.0 numpy==2.5.2 --index https://download.pytorch.org/whl/cu130
"$uv_bin" pip freeze --python .venv/bin/python > requirements.lock.txt
printf 'Ready. Start with ./scripts/run.sh or install the user service.\n'
