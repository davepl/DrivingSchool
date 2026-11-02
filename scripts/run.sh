#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Honor an existing CUDA selection; default to the first GPU on a new host.
export CUDA_VISIBLE_DEVICES="${DRIVING_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export PYTHON_GIL=1
exec .venv/bin/python -m drivingschool "$@"
