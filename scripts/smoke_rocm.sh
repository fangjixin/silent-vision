#!/usr/bin/env bash
set -euo pipefail
export MODEL_BACKEND=real
export PERSISTENCE_ROOT=/workspace/persistence/silent-vision
python - <<'PY'
from pathlib import Path

root = Path("/workspace/persistence/silent-vision")
required = [
    Path("scripts/mpc001_mouth_infer.py"),
    root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages" / "infer.py",
    root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages" / "pipelines" / "model.py",
    root
    / "repos"
    / "Visual_Speech_Recognition_for_Multiple_Languages"
    / "pipelines"
    / "data"
    / "transforms.py",
    root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages" / "configs" / "LRS3_V_WER19.1.ini",
    root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages" / "configs" / "CMLR_V_WER8.0.ini",
    root
    / "repos"
    / "Visual_Speech_Recognition_for_Multiple_Languages"
    / "benchmarks"
    / "LRS3"
    / "models"
    / "LRS3_V_WER19.1"
    / "model.pth",
    root
    / "repos"
    / "Visual_Speech_Recognition_for_Multiple_Languages"
    / "benchmarks"
    / "CMLR"
    / "models"
    / "CMLR_V_WER8.0"
    / "model.pth",
    root / "models" / "minicpm-o-4_5",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required model paths: " + ", ".join(missing))
PY
pytest tests/test_rocm_models.py -m "rocm and model_integration" -v
docker compose -f docker/docker-compose.yml up --build
