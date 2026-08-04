# Silent Vision

Silent Vision reads a short, silent camera clip and classifies it as one command
from a closed set. The current repository can return both a `command.result` and
a structured `agent.result`. It does not transcribe arbitrary speech, control a
light or door, or create a browser recording or still-image artifact.

The intended hackathon demonstration uses CPU preprocessing followed by a
PyTorch temporal classifier on AMD Radeon through ROCm. The final Radeon
checkpoint, validation report, run log, and video are pending. No benchmark or
accuracy figure is claimed here.

Repository: <https://github.com/fangjixin/silent-vision>

## What Silent Vision Does

The browser records one 2-5 second `video/webm` clip with audio disabled. The
backend decodes the clip at 25 FPS, finds one face, extracts a stable 96 x 96
grayscale mouth region, and sends that sequence to one of three command
backends:

- `fake` gives deterministic local test behavior.
- `prototype` compares the clip with saved examples in the global profile.
- `torch` loads a temporal classifier checkpoint.

The current labels are `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, and
`UNKNOWN`. Confidence and top-1 margin checks reject uncertain results as
`UNKNOWN`. Accepted smart-space labels reach the agent boundary as structured
`action="execute"` results, but no integration in this repository acts on them.
`CHAT_OTHER` produces `action="ignore"`, and rejected commands produce
`action="reject"`.

This bounded behavior is useful when a microphone is unwanted or unreliable:
an accessibility desk, a noisy worksite, or a creator setup where a person needs
a small command vocabulary. It is not a general lip-reading system.

## Creator Workflow

The intended downstream use is hands-free creator control: an accepted visual
command could start a browser recording, stop it and expose a WebM download, or
capture a PNG still. That Creator Mode and its creator intent labels are planned
demo behavior, not current behavior in this checkout.

Today the browser offers one-shot recognition and prototype calibration. Click
`Start`, wait through the countdown, mouth one complete command, and allow the
five-second capture to finish. The page shows the command candidates, acceptance
decision, and structured agent result. `Cancel` stops the active capture and
connection. The browser does not produce a downloadable creator artifact.

Do not connect the current `execute` result to a physical device without adding
an explicit allowlist, confirmation policy, device adapter, and failure handling.

## Architecture

```text
Browser camera (audio: false)
  -> 2-5 second WebM over WebSocket
  -> PyAV decode and 25 FPS resampling on CPU
  -> MediaPipe single-face landmarks on CPU
  -> smoothed 96 x 96 grayscale mouth ROI on CPU
  -> fake, NumPy prototype, or PyTorch temporal classifier
  -> confidence + top-1 margin rejection
  -> command.result
  -> agent.result (execute, ignore, or reject)
```

FastAPI creates a short-lived session at `POST /api/sessions` and accepts the
clip at `/ws/{sessionId}`. Only configured WebSocket origins are accepted. The
Torch model uses four Conformer-style blocks with feed-forward layers,
self-attention, depthwise temporal convolution, attentive pooling, and a linear
classifier. PyTorch calls a ROCm device `cuda:0`; this name does not imply an
NVIDIA CUDA runtime when `torch.version.hip` is set.

Health endpoints are available at `/health/live` and `/health/ready`.

## AMD Radeon and ROCm

The intended hackathon path keeps video decoding, face detection, cropping, and
NumPy feature preparation on CPU. The temporal PyTorch classifier runs on AMD
Radeon through ROCm. This split keeps ordinary video work off the accelerator
and uses Radeon for the learned temporal classifier.

Current source has two separate safeguards and one limitation:

- `scripts/setup_amd_real.sh` and `scripts/start_real_rocm.sh` stop if
  `torch.version.hip` is absent or `torch.cuda.is_available()` is false.
- With `COMMAND_BACKEND=torch`, those scripts also require an existing
  `COMMAND_CLASSIFIER_CHECKPOINT`.
- `TorchCommandClassifierBackend` itself still chooses `cuda:0` when PyTorch
  reports an accelerator and otherwise falls back to CPU. Direct `uvicorn`
  startup therefore does not provide an application-level GPU-only guarantee.

For the final evidence run, capture the Python path, PyTorch version,
`torch.version.hip`, device availability, selected backend, checkpoint path, and
real command results. The run and its evidence are pending.

## Requirements and Dependencies

Use Python 3.11 for the documented development environment. Runtime packages in
`requirements.txt` are exactly:

```text
fastapi>=0.141.1,<1.0.0
uvicorn[standard]>=0.52.0,<1.0.0
pydantic>=2.13.4,<3.0.0
pydantic-settings>=2.14.2,<3.0.0
numpy>=1.26.4,<2.0.0
opencv-python-headless>=4.10.0,<4.11.0
mediapipe==0.10.14
python-multipart>=0.0.32,<1.0.0
orjson>=3.11.9,<4.0.0
Pillow==10.4.0
av>=16.0.0,<17.0.0
```

`requirements-dev.txt` adds `pytest`, `pytest-asyncio`, `httpx`, `httpx2`,
`websockets`, `ruff`, and `playwright` in the version ranges recorded in that
file. `package.json` adds `@playwright/test` for browser tests.

PyTorch is deliberately not pinned in `requirements.txt`. The Radeon environment
must provide a ROCm-compatible PyTorch build. Check compatibility against the
installed ROCm image before training or inference.

The browser needs camera permission, `getUserMedia`, `MediaRecorder`, WebSocket,
and WebM recording support. Use a current Chromium-based browser. Camera access
requires a secure context; `http://localhost` is allowed for local development,
while a remote browser should use HTTPS. The current recorder prefers VP9 and
falls back to the browser's default WebM encoder.

Local installation:

```bash
cd /path/to/silent-vision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
npm install
```

## Prototype Calibration

Prototype mode is for calibration and development, not Radeon classifier
evidence. Start it explicitly:

```bash
export COMMAND_BACKEND=prototype
export PERSISTENCE_ROOT=/workspace/persistent/silent-vision
export ALLOWED_ORIGINS=http://localhost:8000
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. In the Calibration panel, choose the correct
intent and language, enter the phrase, and click `Save Sample`. Record 5-10
correctly labeled samples per intent, with small natural changes in pace and
head position. Include `UNKNOWN` and `CHAT_OTHER` examples instead of treating
every mouth movement as an executable command. Useful sample phrases include
`你好，请帮我打开灯` and `hello, please turn on the light`.

The browser writes to `profileId=global`. Inspect sample counts with:

```bash
/opt/venv/bin/python scripts/inspect_prototypes.py \
  --root /workspace/persistent/silent-vision
```

## Train the Command Classifier

The current manifest helper creates a starter JSONL template:

```bash
/opt/venv/bin/python scripts/record_command_manifest.py \
  --output /workspace/persistent/silent-vision/commands/manifest.jsonl
```

Its rows contain blank video paths, so they are not trainable as written. Add a
real `mouth_roi_npy` path and correct intent to each training row, for example:

```json
{"intent":"LIGHT_ON","mouth_roi_npy":"/workspace/persistent/silent-vision/profiles/global/LIGHT_ON/<sample-id>/mouth_roi.npy"}
```

The planned submission workflow names a scanner
`scripts/build_command_manifest.py`. That file is not present in this checkout;
do not present it as a working command until it is implemented. The current
working helper is `scripts/record_command_manifest.py`.

On the Radeon environment, verify ROCm first, then train:

```bash
/opt/venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("accelerator available:", torch.cuda.is_available())
if torch.version.hip is None or not torch.cuda.is_available():
    raise SystemExit("ROCm PyTorch is required for the Radeon training run")
PY

/opt/venv/bin/python scripts/train_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --output /workspace/persistent/silent-vision/models/command_classifier.pt \
  --epochs 20
```

The training script currently uses `cuda:0` when PyTorch reports an accelerator
and CPU otherwise. The preflight above is therefore required for an explicit
Radeon run. Training uses every manifest row with a valid `mouth_roi_npy`; the
current script does not create a train/validation split.

Validate with a separate, manually held-out manifest:

```bash
/opt/venv/bin/python scripts/validate_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/validation.jsonl \
  --checkpoint /workspace/persistent/silent-vision/models/command_classifier.pt \
  --threshold 0.85 \
  --margin 0.20
```

Save the terminal output for the final submission. No validation result has yet
been recorded in this repository.

## GPU-Only Startup

This heading describes the explicit Radeon demo configuration. It is not a
claim that every application entry point rejects CPU fallback.

On an AMD Radeon machine with a ROCm PyTorch environment at
`/opt/venv/bin/python`, start from the root of an ordinary clone:

```bash
cd /path/to/silent-vision
```

For the event-hosted Radeon workspace example, the repository root is:

```bash
cd /workspace/template-repos/template-907/repo
```

Then configure and start the explicit Radeon path:

```bash
export PERSISTENCE_ROOT=/workspace/persistent/silent-vision
export COMMAND_BACKEND=torch
export COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt
export ALLOWED_ORIGINS='*'
bash scripts/amd_real_oneclick.sh
```

Although the real scripts currently default to `prototype`, the explicit
`COMMAND_BACKEND=torch` export above takes precedence. Setup upgrades the Python
requirements, checks ROCm, checks the checkpoint, and then starts Uvicorn on
`127.0.0.1:8000`. Do not use the default prototype backend as Radeon classifier
evidence.

In a second Radeon terminal, expose the local server if the event environment
provides `rc-tunnel`:

```bash
$HOME/.local/bin/rc-tunnel expose --port 8000
```

Open the emitted HTTPS URL in the browser that has the camera. The real scripts
set permissive origins by default for the tunnel; restrict `ALLOWED_ORIGINS` to
the final public origin when possible.

## Persistent Storage

The default root is `/workspace/persistent/silent-vision`:

```text
/workspace/persistent/silent-vision/
├── profiles/global/<INTENT>/<sample-id>/
│   ├── original.webm
│   ├── mouth_roi.npy
│   ├── embedding.npy
│   ├── metadata.json
│   ├── aligned_face_video.mp4    # only when DEBUG_DUMP_WINDOWS=true
│   └── mouth_roi_video.mp4       # only when DEBUG_DUMP_WINDOWS=true
├── models/
│   └── command_classifier.pt
├── commands/
│   ├── manifest.jsonl
│   └── validation.jsonl
├── cache/torch/
└── logs/command-runs/            # debug artifacts only when enabled
```

Calibration intentionally persists the original WebM, derived arrays, and
metadata. Model checkpoints and real recordings should stay out of Git.

## Local Fake Mode

Fake mode tests the HTTP, WebSocket, vision, decision, and agent-result flow
without a real face detector or learned checkpoint:

```bash
export COMMAND_BACKEND=fake
export ALLOWED_ORIGINS=http://localhost:8000
.venv/bin/uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000`. Motion in the fake mouth frames can return
`LIGHT_ON`; still input is rejected as `UNKNOWN`. Fake mode is not evidence of
recognition quality or Radeon execution.

## Docker Notes

`docker/Dockerfile` starts from `rocm/pytorch:latest`, installs
`requirements.txt`, and runs Uvicorn. `docker/docker-compose.yml` passes
`/dev/kfd` and `/dev/dri`, mounts the persistent root, and defaults to the
prototype backend.

Treat these files as deployment scaffolding, not a pinned release image. The
base image uses the mutable `latest` tag, the compose backend is not Torch, and
the container command currently binds Uvicorn to `127.0.0.1`. Confirm host
reachability, ROCm visibility, origin policy, and checkpoint selection in the
target environment before recording evidence.

## Verification

The documentation/submission pass uses focused checks for the changed generator,
bundle builder, and regression tests:

```bash
.venv/bin/ruff check scripts/generate_submission_assets.py scripts/build_contest_bundle.py tests/test_submission_docs.py tests/test_contest_bundle.py
.venv/bin/pytest tests/test_submission_docs.py tests/test_contest_bundle.py --noconftest -q
```

These focused checks do not imply that the whole repository is lint-clean. A
full `.venv/bin/ruff check .` has 24 known pre-existing findings. Runtime and
full-suite verification were deferred and are not part of this documentation
pass.

For a separate runtime verification pass, the helper below runs the fast
fake-mode tests and then starts the local server:

```bash
./scripts/smoke_fake.sh
```

On the Radeon host, with the explicit Torch environment already exported, run:

```bash
COMMAND_BACKEND=torch \
COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt \
./scripts/smoke_rocm.sh
```

The ROCm smoke script checks the HIP runtime and accelerator visibility before
running deployment, prototype, and classifier tests. A passing smoke script does
not replace held-out classifier validation or a recorded end-to-end demo.

## Privacy and Limitations

- Browser capture requests video only; `audio: false` prevents microphone
  capture by this application.
- Ordinary command clips are processed from memory and a temporary decode file.
  They persist only when `DEBUG_DUMP_WINDOWS=true`.
- Calibration always stores the original video, mouth arrays, embedding, and
  metadata under the global profile. Delete these files according to the
  operator's retention policy; no consent or retention UI is included.
- Debug metadata can include local artifact paths and timing data. Do not enable
  debug dumps on a public service without access controls.
- The repository has no authentication, device authorization, or TLS server.
  The tunnel provides transport access but is not an application security layer.
- Recognition is closed-set and sensitive to camera angle, lighting, occlusion,
  speaking style, sample quality, and the configured thresholds.
- The current Torch path has a CPU fallback outside the ROCm startup scripts.
- The current agent returns structured decisions only. No light, door, recording,
  or capture action is implemented.
- No measured accuracy, latency, throughput, or memory result is published.

## Submission Materials

The editable and generated submission materials are indexed in
[`submission/README.md`](submission/README.md). Source copy lives under
`docs/submission/` as reviewed reference copy; the generator remains the
canonical generated-artifact copy/layout source. The generated project profile
PDF and poster PDF/PNG are complete. The Radeon evidence, final checkpoint,
held-out validation, Creator Mode actions, demo video, and video URL remain
pending.

Required pull request title: `Track 1, Jixin Fang, Silent Vision`.
