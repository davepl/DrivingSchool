#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# CUDA_VISIBLE_DEVICES uses PCI enumeration on this Dell: 1 is the GB300.
export CUDA_VISIBLE_DEVICES="${DRIVING_GPU:-1}"
export PYTHON_GIL=1
exec .venv/bin/python -m drivingschool "$@"
