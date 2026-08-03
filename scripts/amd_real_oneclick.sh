#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SV_ROOT="${SV_ROOT:-/workspace/persistent/silent-vision}"
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-$SV_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$SV_ROOT/cache/torch}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export COMMAND_BACKEND="${COMMAND_BACKEND:-prototype}" # default: COMMAND_BACKEND=prototype
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"

bash scripts/setup_amd_real.sh
bash scripts/start_real_rocm.sh
