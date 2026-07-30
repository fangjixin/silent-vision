#!/usr/bin/env bash
set -euo pipefail
export MODEL_BACKEND=real
export PERSISTENCE_ROOT=/workspace/persistence/silent-vision
python - <<'PY'
from pathlib import Path

root = Path("/workspace/persistence/silent-vision")
required = [
    root / "models" / "avhubert" / "model.pt",
    root / "models" / "cmlr" / "model.pth",
    root / "models" / "minicpm-o-4_5",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required model paths: " + ", ".join(missing))
PY
pytest tests/test_rocm_models.py -m "rocm and model_integration" -v
docker compose -f docker/docker-compose.yml up --build
