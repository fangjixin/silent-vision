# Silent Vision

Silent Vision is now a closed-set visual command recognition system. The
browser records one 2-5 second `video/webm` utterance clip, uploads it as
binary WebSocket data, and the backend classifies business intents such as
`LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, and `UNKNOWN`.

It no longer performs open-vocabulary bilingual transcription in the runtime
path. Rejected commands are not executed by the agent.

## AMD ROCm startup

```bash
cd /workspace/template-repos/template-907/repo
export COMMAND_BACKEND=prototype
export DEBUG_DUMP_WINDOWS=true
bash scripts/amd_real_oneclick.sh
```

The tunnel is still handled in a second AMD terminal:

```bash
$HOME/.local/bin/rc-tunnel expose --port 8000
```

Open the emitted `https://rc-....radeon.firstdg.ai` URL from the local browser
that has the camera.

## Browser workflow

Normal command recognition:

1. Click `Start`.
2. The page previews the camera and counts down.
3. Speak one complete 2-5 second command.
4. The browser auto-stops recording and uploads the clip.
5. The backend decodes/resamples to 25 FPS, extracts a stable mouth ROI clip,
   classifies the command, and returns `command.result` plus `agent.result`.

`Cancel` only cancels the current recording/connection. You do not need to wait
for the old frame-buffer counter, and you should not manually press stop after
speaking.

## Prototype calibration

Prototype mode is the recommended first real workflow. It compares the current
mouth ROI clip against saved command examples instead of trying to transcribe
arbitrary Chinese or English.

Use the browser Calibration panel first:

1. Pick an intent such as `LIGHT_ON`.
2. Type the phrase you are recording, for example `你好，请帮我打开灯`.
3. Click `Save Sample`.
4. Record 5-10 samples per command.
5. Press `Start` to test recognition.

Samples are saved in the shared global prototype profile:

```text
/workspace/persistent/silent-vision/profiles/global/<INTENT>/<sampleId>/
  original.webm
  mouth_roi.npy
  embedding.npy
  metadata.json
```

The browser always uses `profileId=global`. This avoids losing access to
samples when the public tunnel domain changes, the browser changes, or the
server restarts.

To inspect saved samples:

```bash
/opt/venv/bin/python scripts/inspect_prototypes.py --root /workspace/persistent/silent-vision
```

## Torch classifier training

After enough command data exists, train a classifier:

```bash
/opt/venv/bin/python scripts/record_command_manifest.py \
  --output /workspace/persistent/silent-vision/commands/manifest.jsonl

/opt/venv/bin/python scripts/train_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --output /workspace/persistent/silent-vision/models/command_classifier.pt
```

Then run:

```bash
export COMMAND_BACKEND=torch
export COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt
bash scripts/amd_real_oneclick.sh
```

## Persistence layout

```text
/workspace/persistent/silent-vision/
├── profiles/
│   └── global/
├── models/
│   └── command_classifier.pt      # optional torch backend
├── cache/
│   └── torch/
├── commands/
│   └── manifest.jsonl
└── logs/
    └── command-runs/
```

## Local fake mode

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Verification

Default fake mode:

```bash
./scripts/smoke_fake.sh
```

ROCm command mode on the Radeon server:

```bash
./scripts/smoke_rocm.sh
```
