# Silent Vision Demo Video Script

Status: pending recording on the AMD Radeon environment. Target length: 3-5
minutes. Do not publish this as a completed demo until the real checkpoint,
Creator Mode, downloadable artifact, and rejection gate have been verified in
one recording.

## Recording checklist

- Use one continuous screen recording that shows the terminal and browser.
- Use the real Radeon environment and final classifier checkpoint.
- Do not paste simulated GPU output or edit in benchmark values.
- Verify that Creator Mode and the three creator intents exist in the checked-out
  source before recording. They are not present in the current checkout.
- Prepare one accepted creator command and one unknown or low-confidence command.
- Keep the browser's result JSON and downloadable WebM or PNG visible.
- Add the final video URL to `submission/README.md` and
  `submission/pull-request-description.md` after upload.

## 0:00-0:35 — Problem and current boundary

**On screen:** Repository README, then the browser.

**Say:**

“Silent Vision is a closed-set visual command interface for situations where
audio is unavailable, unreliable, or unwanted. The browser records a short
camera clip with audio disabled. The server reads the mouth-region sequence and
returns a bounded intent with confidence, margin, and an explicit acceptance
decision. This is not open-ended transcription. The current repository also
keeps external effects separate from recognition, so a structured result is not
presented as proof that a light or door changed.”

## 0:35-1:20 — Terminal proof and server startup

**On screen:** Run the commands; do not use a prepared screenshot.

```bash
/opt/venv/bin/python - <<'PY'
import torch
print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY

export COMMAND_BACKEND=torch
export COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt
export PERSISTENCE_ROOT=/workspace/persistent/silent-vision
bash scripts/start_real_rocm.sh
```

**Say:**

“This is the real Python and PyTorch environment. The HIP value and Radeon
device prove that this PyTorch process is using ROCm. I am explicitly selecting
the Torch backend and the recorded checkpoint. The startup guard stops if ROCm
is missing or the checkpoint does not exist. Video decode, face landmarks, and
mouth cropping run on CPU. The temporal PyTorch classifier is the Radeon stage.”

## 1:20-2:40 — Creator Mode and a real artifact

**Recording gate:** Perform this section only after Creator Mode is implemented
and tested. The current checkout cannot produce this artifact.

**On screen:** Open the HTTPS tunnel URL, enter Creator Mode, and show the live
preview. Recognize `START_RECORDING`, record several seconds, then recognize
`STOP_RECORDING`. Open or download the resulting WebM. If the final demo uses
`CAPTURE_FRAME` instead, download and open the PNG.

**Say:**

“Creator Mode keeps one camera stream active. I will mouth ‘start recording.’
The short command clip goes to the backend, and only an accepted creator intent
reaches the browser controller. The creator recording is now active. I will
mouth ‘stop recording.’ The accepted result stops the recorder and exposes this
WebM download. Here is the real file created in the browser. The command clip
and the creator recording are separate media operations over the same preview
stream.”

**Show:** The `command.result`, the `agent.result`, and the downloaded artifact.
Point out the backend, confidence, margin, selected device metadata if available,
and recorded inference timing without calling it a benchmark.

## 2:40-3:30 — Rejection safety

**On screen:** Start another recognition attempt. Use an out-of-set phrase or a
deliberately ambiguous sample. Keep Creator Mode status and the artifact area in
view.

**Say:**

“Now I will give a command that should not pass the confidence and margin gates.
The backend returns `UNKNOWN` or a rejected result. The agent action is `reject`,
and the browser controller receives no creator command. No new recording starts,
no still is captured, and the previous artifact is unchanged. This rejection
path is part of the product behavior, not an error hidden from the user.”

## 3:30-4:10 — Close

**On screen:** Return to the README architecture and repository link.

**Say:**

“Silent Vision turns a deliberate silent clip into a small, inspectable command
decision. CPU preprocessing feeds a temporal classifier running through PyTorch
on AMD Radeon and ROCm. Uncertain commands stop at the safety gate. The code,
setup instructions, training commands, submission sources, and limitations are
available at github.com/fangjixin/silent-vision.”

Stop the recording only after the repository URL is legible.
