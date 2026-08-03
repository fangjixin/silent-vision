#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export SV_ROOT="${SV_ROOT:-/workspace/persistent/silent-vision}"
export PERSISTENCE_ROOT="${PERSISTENCE_ROOT:-$SV_ROOT}"
export TORCH_HOME="${TORCH_HOME:-$SV_ROOT/cache/torch}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/venv/bin/python}"
export COMMAND_BACKEND="${COMMAND_BACKEND:-prototype}"

mkdir -p \
  "$SV_ROOT/logs" \
  "$SV_ROOT/logs/command-runs" \
  "$SV_ROOT/profiles/global" \
  "$SV_ROOT/models" \
  "$TORCH_HOME"

check_no_removed_dependencies() {
  if grep -E '^(transformers|huggingface-hub|accelerate|safetensors|librosa|soundfile|minicpmo-utils)([<=> ]|$)' requirements.txt; then
    echo "removed open-vocabulary model dependency found in requirements.txt" >&2
    exit 1
  fi
}

check_rocm_python() {
  "$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch hip:", torch.version.hip)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is not available in this Python environment")
PY
}

check_app_dependencies() {
  "$PYTHON_BIN" - <<'PY'
from importlib import metadata

for package in ["fastapi", "uvicorn", "pydantic", "numpy", "cv2", "mediapipe", "av", "PIL"]:
    try:
        module_name = "opencv-python-headless" if package == "cv2" else "pillow" if package == "PIL" else package
        print(f"{package}:", metadata.version(module_name))
    except Exception as exc:
        raise SystemExit(f"missing required runtime dependency {package}: {exc}") from exc

import mediapipe as mp
solutions = getattr(mp, "solutions", None)
if solutions is not None:
    face_mesh = solutions.face_mesh
else:
    from mediapipe.python.solutions import face_mesh
print("FaceMesh:", face_mesh.FaceMesh)
PY
}

check_command_backend() {
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import os

backend = os.environ.get("COMMAND_BACKEND", "prototype")
root = Path(os.environ["PERSISTENCE_ROOT"])
print("command backend:", backend)
print("persistence root:", root)

if backend == "prototype":
    (root / "profiles" / "global").mkdir(parents=True, exist_ok=True)
    print("prototype profiles:", root / "profiles")
elif backend == "torch":
    checkpoint = os.environ.get("COMMAND_CLASSIFIER_CHECKPOINT")
    if not checkpoint:
        raise SystemExit("COMMAND_CLASSIFIER_CHECKPOINT is required when COMMAND_BACKEND=torch")
    path = Path(checkpoint)
    if not path.exists():
        raise SystemExit(f"COMMAND_CLASSIFIER_CHECKPOINT does not exist: {path}")
    print("command checkpoint:", path)
else:
    raise SystemExit(f"unsupported COMMAND_BACKEND: {backend}")
PY
}

check_no_removed_dependencies
check_rocm_python
"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --upgrade -r requirements.txt
check_rocm_python
check_app_dependencies
check_command_backend

echo "setup complete"
