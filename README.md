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
│   └── minicpm-o-4_5/
├── repos/
│   └── Visual_Speech_Recognition_for_Multiple_Languages/
│       ├── configs/LRS3_V_WER19.1.ini
│       ├── configs/CMLR_V_WER8.0.ini
│       └── benchmarks/
├── cache/
├── reports/
└── logs/
```

## ROCm container

```bash
mkdir -p /workspace/persistence/silent-vision/models/minicpm-o-4_5
mkdir -p /workspace/persistence/silent-vision/repos
mkdir -p /workspace/persistence/silent-vision/cache/{huggingface,torch}
mkdir -p /workspace/persistence/silent-vision/reports/{benchmarks,diagnostics}
mkdir -p /workspace/persistence/silent-vision/logs
docker compose -f docker/docker-compose.yml up --build
```

## Real model setup

MiniCPM-o 4.5 is stored as a Hugging Face snapshot under:

```text
/workspace/persistence/silent-vision/models/minicpm-o-4_5
```

The real lip readers use `mpc001/Visual_Speech_Recognition_for_Multiple_Languages` instead of the old TorchScript placeholder files. Silent Vision does not call the upstream `infer.py` for live WebSocket windows, because that script runs its own face tracking. Instead, `scripts/mpc001_mouth_infer.py` feeds the already-cropped 75-frame mouth window directly into the mpc001 model stack. Clone the repo under:

```text
/workspace/persistence/silent-vision/repos/Visual_Speech_Recognition_for_Multiple_Languages
```

Then download and extract the model zoo packages referenced by:

```text
configs/LRS3_V_WER19.1.ini
configs/CMLR_V_WER8.0.ini
```

The expected benchmark files include:

```text
benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth
benchmarks/CMLR/models/CMLR_V_WER8.0/model.pth
```

Install the mpc001 repository dependencies in the same Python environment used to start FastAPI before running `MODEL_BACKEND=real`.

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
