#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export COMMAND_BACKEND=prototype
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-/workspace/persistent/silent-vision}"
export TORCH_HOME="${TORCH_HOME:-$PERSISTENCE_ROOT/cache/torch}"
export ALLOWED_ORIGINS="${ALLOWED_ORIGINS:-*}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"

"$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is not available; refusing to start real mode")

root = Path(os.environ["PERSISTENCE_ROOT"])
print("command backend: prototype")
print("persistence root:", root)
profile_root = root / "profiles"
profile_root.mkdir(parents=True, exist_ok=True)
print("prototype profiles:", profile_root)
PY

"$PYTHON_BIN" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level "${LOG_LEVEL:-info}"
