#!/usr/bin/env bash
set -euo pipefail

export SV_ROOT="${SV_ROOT:-/workspace/persistent/silent-vision}"
MPC_REPO=${MPC_REPO:-$SV_ROOT/repos/Visual_Speech_Recognition_for_Multiple_Languages}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}

export HF_HOME=${HF_HOME:-$SV_ROOT/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$SV_ROOT/cache/torch}

mkdir -p "$SV_ROOT/downloads" "$SV_ROOT/extract" "$SV_ROOT/repos" "$SV_ROOT/logs"
mkdir -p "$SV_ROOT/models/minicpm-o-4_5" "$HF_HOME" "$TORCH_HOME"

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

check_transformers_stack() {
  "$PYTHON_BIN" - <<'PY'
from importlib import metadata

def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"{name} is not installed")

hub_version = version("huggingface-hub")
transformers_version = version("transformers")
tokenizers_version = version("tokenizers")

print("transformers:", transformers_version)
print("tokenizers:", tokenizers_version)
print("huggingface-hub:", hub_version)

major = int(hub_version.split(".", 1)[0])
if major >= 1:
    raise SystemExit(
        "huggingface-hub must be <1.0 for transformers 4.51.0; "
        f"found {hub_version}"
    )

from transformers import AutoModel, AutoTokenizer

print("transformers imports:", AutoModel.__name__, AutoTokenizer.__name__)
PY
}

check_rocm_python

apt-get update || true
DEBIAN_FRONTEND=noninteractive apt-get install -y ca-certificates git aria2
update-ca-certificates || true
command -v aria2c

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install --upgrade -r requirements.txt
"$PYTHON_BIN" -m pip install --upgrade --force-reinstall --no-deps "huggingface-hub>=0.30.0,<1.0.0"
check_rocm_python
check_transformers_stack

"$PYTHON_BIN" - <<'PY'
import mediapipe as mp

solutions = getattr(mp, "solutions", None)
if solutions is not None:
    face_mesh = solutions.face_mesh
else:
    from mediapipe.python.solutions import face_mesh

print("mediapipe:", getattr(mp, "__version__", None))
print("FaceMesh:", face_mesh.FaceMesh)
PY

cd "$SV_ROOT/repos"
if [ ! -d "$MPC_REPO/.git" ]; then
  GIT_SSL_NO_VERIFY=true git clone https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages.git "$MPC_REPO"
fi

test -f "$MPC_REPO/configs/LRS3_V_WER19.1.ini"
test -f "$MPC_REPO/configs/CMLR_V_WER8.0.ini"

cd "$SV_ROOT/downloads"
aria2c -c -x 16 -s 16 -k 1M --check-certificate=false \
  -o LRS3_V_WER19.1.zip \
  'https://github.com/fangjixin/silent-vision/releases/download/models-v1/LRS3_V_WER19.1.zip'
aria2c -c -x 16 -s 16 -k 1M --check-certificate=false \
  -o CMLR_V_WER8.0.zip \
  'https://github.com/fangjixin/silent-vision/releases/download/models-v1/CMLR_V_WER8.0.zip'
aria2c -c -x 16 -s 16 -k 1M --check-certificate=false \
  -o lm_en_subword.zip \
  'https://github.com/fangjixin/silent-vision/releases/download/models-v1/lm_en_subword.zip'
aria2c -c -x 16 -s 16 -k 1M --check-certificate=false \
  -o lm_zh.zip \
  'https://github.com/fangjixin/silent-vision/releases/download/models-v1/lm_zh.zip'

"$PYTHON_BIN" - <<'PY'
from pathlib import Path
import configparser
import os
import shutil
import zipfile

root = Path(os.environ["SV_ROOT"])
repo = root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages"

def config_paths(config_file: Path) -> dict[str, Path]:
    cfg = configparser.ConfigParser()
    cfg.read(config_file)
    result = {}
    for key in ["model_path", "model_conf", "rnnlm", "rnnlm_conf"]:
        value = cfg.get("model", key, fallback="")
        if value:
            result[key] = repo / value
    return result

lrs3_paths = config_paths(repo / "configs" / "LRS3_V_WER19.1.ini")
cmlr_paths = config_paths(repo / "configs" / "CMLR_V_WER8.0.ini")

required = [
    lrs3_paths["model_path"],
    lrs3_paths["model_conf"],
    lrs3_paths["rnnlm"],
    lrs3_paths["rnnlm_conf"],
    cmlr_paths["model_path"],
    cmlr_paths["model_conf"],
    cmlr_paths["rnnlm"],
    cmlr_paths["rnnlm_conf"],
]

if all(path.exists() for path in required):
    print("mpc001 assets already prepared; skipping extract/copy")
    raise SystemExit(0)

archives = {
    "LRS3_V_WER19.1.zip": "lrs3_model",
    "CMLR_V_WER8.0.zip": "cmlr_model",
    "lm_en_subword.zip": "lrs3_lm",
    "lm_zh.zip": "cmlr_lm",
}

for zip_name, extract_name in archives.items():
    src = root / "downloads" / zip_name
    dst = root / "extract" / extract_name
    if not src.exists():
        raise SystemExit(f"missing zip file: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(src) as zf:
        zf.extractall(dst)

def find_asset_dir(base: Path) -> Path:
    matches = [p.parent for p in base.rglob("model.pth") if (p.parent / "model.json").exists()]
    if not matches:
        raise SystemExit(f"no model.pth + model.json found under {base}")
    return matches[0]

jobs = [
    (root / "extract" / "lrs3_model", lrs3_paths["model_path"].parent),
    (root / "extract" / "cmlr_model", cmlr_paths["model_path"].parent),
    (root / "extract" / "lrs3_lm", lrs3_paths["rnnlm"].parent),
    (root / "extract" / "cmlr_lm", cmlr_paths["rnnlm"].parent),
]

for src_base, target_dir in jobs:
    src_dir = find_asset_dir(src_base)
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ["model.pth", "model.json"]:
        shutil.copy2(src_dir / name, target_dir / name)
        print("copied", target_dir / name)
PY

"$PYTHON_BIN" -m pip install --upgrade --force-reinstall --no-deps "huggingface-hub>=0.30.0,<1.0.0"
check_transformers_stack
"$PYTHON_BIN" - <<'PY'
from huggingface_hub import snapshot_download
import os
from pathlib import Path

snapshot_download(
    repo_id="openbmb/MiniCPM-o-4_5",
    local_dir=str(Path(os.environ["SV_ROOT"]) / "models" / "minicpm-o-4_5"),
)
PY
check_rocm_python

echo "setup complete"
