#!/usr/bin/env bash
set -euo pipefail

export COMMAND_BACKEND=prototype
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

root = Path(os.environ["PERSISTENCE_ROOT"])
(root / "profiles" / "global").mkdir(parents=True, exist_ok=True)
print("command backend: prototype")
PY

"$PYTHON_BIN" -m pytest \
  tests/test_deployment_files.py \
  tests/test_prototype_recognition.py \
  tests/test_command_classifier.py \
  -q
