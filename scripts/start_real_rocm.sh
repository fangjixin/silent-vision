#!/usr/bin/env bash
set -euo pipefail

export MODEL_BACKEND="${MODEL_BACKEND:-real}"
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-/workspace/persistent/silent-vision}"
export MPC001_REPO_DIR="${MPC001_REPO_DIR:-$PERSISTENCE_ROOT/repos/Visual_Speech_Recognition_for_Multiple_Languages}"
export HF_HOME="${HF_HOME:-$PERSISTENCE_ROOT/cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSISTENCE_ROOT/cache/torch}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"
export MPC001_GPU_IDX="${MPC001_GPU_IDX:-0}"
export MPC001_TIMEOUT_SECONDS="${MPC001_TIMEOUT_SECONDS:-180}"
export MPC001_PYTHON="${MPC001_PYTHON:-/opt/venv/bin/python}"

"$MPC001_PYTHON" - <<'PY'
import sys
import torch
from importlib import metadata

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is not available; refusing to start real mode")

hub_version = metadata.version("huggingface-hub")
transformers_version = metadata.version("transformers")
print("transformers:", transformers_version)
print("huggingface-hub:", hub_version)
if int(hub_version.split(".", 1)[0]) >= 1:
    raise SystemExit(
        "huggingface-hub must be <1.0 for transformers 4.51.0. "
        "Run: /opt/venv/bin/python -m pip install --force-reinstall --no-deps "
        "'huggingface-hub>=0.30.0,<1.0.0'"
    )
PY

"$MPC001_PYTHON" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
