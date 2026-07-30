# Silent Vision

Realtime bilingual lipreading prototype for one active anonymous browser session.

## Fake mode

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## ROCm development access

```bash
ssh -L 8000:127.0.0.1:8000 user@rocm-server
```

Open `http://localhost:8000` from the machine with the camera.

## Persistence layout

```text
/workspace/persistence/silent-vision/
├── models/
│   ├── avhubert/model.pt
│   ├── cmlr/model.pth
│   ├── cmlr/language-model.pth
│   └── minicpm-o-4_5/
├── cache/
├── reports/
└── logs/
```

## ROCm container

```bash
mkdir -p /workspace/persistence/silent-vision/models/{avhubert,cmlr,minicpm-o-4_5}
mkdir -p /workspace/persistence/silent-vision/cache/{huggingface,torch}
mkdir -p /workspace/persistence/silent-vision/reports/{benchmarks,diagnostics}
mkdir -p /workspace/persistence/silent-vision/logs
docker compose -f docker/docker-compose.yml up --build
```

## Verification

Default fake mode:

```bash
./scripts/smoke_fake.sh
```

ROCm/model mode on the Radeon 7900 server:

```bash
./scripts/smoke_rocm.sh
```

For the browser path, forward the service:

```bash
ssh -L 8000:127.0.0.1:8000 user@rocm-server
```

Then open `http://localhost:8000` on the local machine that has the camera.
