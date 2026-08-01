# AMD ROCm real-mode runbook

Silent Vision real mode must run with the ROCm Python environment:

```bash
/opt/venv/bin/python
```

Do not start real mode with `/usr/bin/python`, `python`, or the global `uvicorn`, because those can load a CUDA/NVIDIA PyTorch wheel.

## Setup

From the Silent Vision repository on the AMD machine:

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/setup_amd_real.sh
```

The script prepares:

- `/workspace/persistent/silent-vision`
- mpc001 LRS3 and CMLR visual-only model assets
- MiniCPM-o 4.5 under `/workspace/persistent/silent-vision/models/minicpm-o-4_5`
- Python dependencies in `/opt/venv`
- MediaPipe FaceMesh validation
- ROCm PyTorch validation

## Start

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/start_real_rocm.sh
```

## One-command setup and start

To run setup and then start the server in one terminal:

```bash
cd /workspace/template-repos/template-907/repo
bash scripts/amd_real_oneclick.sh
```

This script does not expose the public tunnel. Keep the tunnel command in a separate terminal.

Expected startup signs:

```text
torch hip: 7.x
cuda available: True
Loading checkpoint shards: 100%
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

## Tunnel

In a second AMD terminal:

```bash
$HOME/.local/bin/rc-tunnel expose --port 8000
```

Open the emitted `https://rc-....radeon.firstdg.ai` URL from the local browser.

## Notes

- Browser `localhost:8000` is not the AMD server.
- Start can be clicked again; the latest browser connection takes over the single active server slot.
- Stop closes the camera and WebSocket so the active server slot is released.
- Real inference runs in a background task so frame capture can continue while the current 75-frame window is being analyzed.
