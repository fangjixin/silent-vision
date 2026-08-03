#!/usr/bin/env bash
set -euo pipefail

export COMMAND_BACKEND="${COMMAND_BACKEND:-prototype}"
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-/workspace/persistent/silent-vision}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"

"$PYTHON_BIN" - <<'PY'
import os
from pathlib import Path
import torch

print("torch:", torch.__version__)
print("torch hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is not available")

backend = os.environ.get("COMMAND_BACKEND", "prototype")
root = Path(os.environ["PERSISTENCE_ROOT"])
if backend == "prototype":
    (root / "profiles" / "global").mkdir(parents=True, exist_ok=True)
elif backend == "torch":
    checkpoint = os.environ.get("COMMAND_CLASSIFIER_CHECKPOINT")
    if not checkpoint or not Path(checkpoint).exists():
        raise SystemExit("COMMAND_CLASSIFIER_CHECKPOINT is required and must exist when COMMAND_BACKEND=torch")
else:
    raise SystemExit(f"unsupported COMMAND_BACKEND: {backend}")
print("command backend:", backend)
PY

"$PYTHON_BIN" -m pytest \
  tests/test_deployment_files.py \
  tests/test_prototype_recognition.py \
  tests/test_command_classifier.py \
  -q
