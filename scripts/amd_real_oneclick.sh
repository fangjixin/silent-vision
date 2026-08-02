#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SV_ROOT="${SV_ROOT:-/workspace/persistent/silent-vision}"
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-$SV_ROOT}"
export MPC_REPO="${MPC_REPO:-$SV_ROOT/repos/Visual_Speech_Recognition_for_Multiple_Languages}"
export HF_HOME="${HF_HOME:-$SV_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$SV_ROOT/cache/torch}"
export MPC001_REPO_DIR="${MPC001_REPO_DIR:-$MPC_REPO}"
export MPC001_PYTHON="${MPC001_PYTHON:-/opt/venv/bin/python}"
export PYTHON_BIN="${PYTHON_BIN:-$MPC001_PYTHON}"
export MODEL_BACKEND="${MODEL_BACKEND:-real}"
export RECOGNITION_MODE="${RECOGNITION_MODE:-command}"
export COMMAND_BACKEND="${COMMAND_BACKEND:-fake}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"
export MPC001_GPU_IDX="${MPC001_GPU_IDX:-0}"
export MPC001_TIMEOUT_SECONDS="${MPC001_TIMEOUT_SECONDS:-180}"

bash scripts/setup_amd_real.sh
bash scripts/start_real_rocm.sh
