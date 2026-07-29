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
