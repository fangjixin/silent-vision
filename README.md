# Silent Vision

Silent Vision now supports a closed-set visual command path for short
utterance clips. The browser records a 2-5 second `video/webm` clip, uploads
it as binary WebSocket data, and the backend classifies business intents such
as `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, and `UNKNOWN`.

Rejected commands do not call MiniCPM and do not execute actions.

AMD ROCm command-mode startup:

```bash
cd /workspace/template-repos/template-907/repo
export RECOGNITION_MODE=command
export COMMAND_BACKEND=prototype
export DEBUG_DUMP_WINDOWS=true
bash scripts/amd_real_oneclick.sh
```

Prototype mode is the recommended first real workflow. It does not try to
transcribe arbitrary Chinese. It compares the current mouth ROI clip against
saved command examples.

Use the browser Calibration panel first:

1. Pick an intent such as `LIGHT_ON`.
2. Type the phrase you are recording, for example `你好，请帮我打开灯`.
3. Click `Save Sample`.
4. Record 5-10 samples per command.
5. Press `Start` to test recognition.

Samples are saved under anonymous browser profiles:

```text
/workspace/persistent/silent-vision/profiles/<profileId>/<INTENT>/<sampleId>/
  original.webm
  mouth_roi.npy
  embedding.npy
  metadata.json
```

`Personal Profile` means the current browser's anonymous `profileId`. It is
used first when matching commands. `Global Profile` means server-provided
defaults shared by all browsers:

```text
/workspace/persistent/silent-vision/profiles/global/
```

To promote your personal samples into the global defaults:

```bash
/opt/venv/bin/python scripts/inspect_prototypes.py --root /workspace/persistent/silent-vision
/opt/venv/bin/python scripts/build_global_prototypes.py \
  --root /workspace/persistent/silent-vision \
  --from-profile <profileId>
```

Classifier training is still available after enough data exists:

```bash
/opt/venv/bin/python scripts/record_command_manifest.py --output /workspace/persistent/silent-vision/commands/manifest.jsonl
/opt/venv/bin/python scripts/train_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --output /workspace/persistent/silent-vision/models/command_classifier.pt
```

Then run the server with:

```bash
export COMMAND_BACKEND=torch
export COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt
bash scripts/amd_real_oneclick.sh
```

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
/workspace/persistent/silent-vision/
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
mkdir -p /workspace/persistent/silent-vision/models/minicpm-o-4_5
mkdir -p /workspace/persistent/silent-vision/repos
mkdir -p /workspace/persistent/silent-vision/cache/{huggingface,torch}
mkdir -p /workspace/persistent/silent-vision/reports/{benchmarks,diagnostics}
mkdir -p /workspace/persistent/silent-vision/logs
docker compose -f docker/docker-compose.yml up --build
```

## Real model setup

MiniCPM-o 4.5 is stored as a Hugging Face snapshot under:

```text
/workspace/persistent/silent-vision/models/minicpm-o-4_5
```

The real lip readers use `mpc001/Visual_Speech_Recognition_for_Multiple_Languages` instead of the old TorchScript placeholder files. Silent Vision does not call the upstream `infer.py` for live WebSocket windows, because that script runs its own face tracking. Instead, `scripts/mpc001_mouth_infer.py` feeds the already-cropped 75-frame mouth window directly into the mpc001 model stack. Clone the repo under:

```text
/workspace/persistent/silent-vision/repos/Visual_Speech_Recognition_for_Multiple_Languages
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
