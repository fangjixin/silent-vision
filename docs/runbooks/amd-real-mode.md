# AMD ROCm command-mode runbook

Silent Vision on the AMD machine must run with the ROCm Python environment:

```bash
/opt/venv/bin/python
```

Do not start it with `/usr/bin/python`, `python`, or a global `uvicorn`, because
those can load a CUDA/NVIDIA PyTorch wheel.

## Setup

From the Silent Vision repository on the AMD machine:

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/setup_amd_real.sh
```

The script prepares:

- `/workspace/persistent/silent-vision`
- command prototype/profile directories
- Python dependencies in `/opt/venv`
- PyAV / MediaPipe FaceMesh validation
- ROCm PyTorch validation
- optional torch classifier checkpoint validation when `COMMAND_BACKEND=torch`

It does not download open-vocabulary transcription models.

## Start

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/start_real_rocm.sh
```

For prototype mode:

```bash
export COMMAND_BACKEND=prototype
bash scripts/start_real_rocm.sh
```

For trained torch classifier mode:

```bash
export COMMAND_BACKEND=torch
export COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt
bash scripts/start_real_rocm.sh
```

## One-command setup and start

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/amd_real_oneclick.sh
```

This script does not expose the public tunnel. Keep the tunnel command in a
separate terminal.

Expected startup signs:

```text
torch hip: 7.x
cuda available: True
setup complete
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

## Tunnel

In a second AMD terminal:

```bash
$HOME/.local/bin/rc-tunnel expose --port 8000
```

Open the emitted `https://rc-....radeon.firstdg.ai` URL from the local browser.

## Runtime behavior

- Browser `localhost:8000` is not the AMD server.
- `Start` records one 2-5 second video clip and submits it once.
- `Cancel` cancels the current recording and releases the server session.
- Recognition runs after the full clip arrives; there is no 75-frame streaming
  window in the runtime flow.
- When `DEBUG_DUMP_WINDOWS=true`, debug artifacts are saved under
  `/workspace/persistent/silent-vision/logs/command-runs/`.
