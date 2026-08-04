# Silent Vision Track 1 Submission Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a truthful Track 1 submission package in which visual commands trigger a real browser creator workflow, production classification runs only on AMD Radeon/ROCm, and every required English artifact is reproducible from the repository.

**Architecture:** Keep decoding and mouth-ROI preprocessing on CPU, then run the temporal classifier with PyTorch on a required ROCm device. Reuse one browser camera stream for short command clips and creator recording/capture actions. Build the Torch dataset from saved global calibration samples, and generate the PDF, poster, submission index, and contest bundle from reviewed English sources.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, NumPy, PyTorch/ROCm, MediaPipe, PyAV, browser MediaRecorder/Canvas, Playwright, pytest, ReportLab, pypdf, pdfplumber, Poppler.

## Global Constraints

- Public submission materials, project descriptions, and pull-request text must be English.
- Pull-request title must be exactly `Track 1, Jixin Fang, Silent Vision`.
- Production and demo runtime must use `COMMAND_BACKEND=torch` on AMD Radeon/ROCm and must not fall back to CPU.
- The NumPy prototype matcher remains a calibration and development path only.
- `START_RECORDING`, `STOP_RECORDING`, and `CAPTURE_FRAME` must produce observable browser behavior; rejected commands must produce no creator artifact.
- Do not publish accuracy, latency, throughput, or memory figures without a saved Radeon run that produced them.
- Public copy must use plain, concrete English and avoid generic AI-marketing language.
- The 3-5 minute video is recorded later on the Radeon environment; this plan produces its final script and evidence checklist now.

---

## File Map

- `command/labels.py`: closed-set intent vocabulary and executable-intent policy.
- `backend/schemas.py`: API schema literals matching the intent vocabulary.
- `command/inference.py`: fake, prototype, and GPU-only Torch classifiers.
- `command/dataset.py`: deterministic calibration-sample discovery and train/validation manifest creation.
- `scripts/build_command_manifest.py`: CLI for the dataset builder.
- `scripts/train_command_classifier.py`: train-split-only Radeon training and run-summary output.
- `scripts/validate_command_classifier.py`: validation-split evaluation and JSON report output.
- `frontend/camera.js`: reusable command-clip recorder owning one camera stream.
- `frontend/creator.js`: creator recording and still-capture state machine.
- `frontend/websocket.js`: session lifecycle and accepted-intent-to-creator-action wiring.
- `frontend/index.html`, `frontend/styles.css`: Creator Mode controls and artifact display.
- `README.md`: complete English setup, training, startup, usage, and dependency guide.
- `docs/submission/project-profile-source.md`: reviewed source copy for the six-page profile.
- `docs/submission/poster-copy.md`: reviewed source copy for the poster.
- `submission/*`: final contest-facing artifacts and index.
- `scripts/generate_submission_assets.py`: deterministic profile and poster generator.
- `scripts/build_contest_bundle.py`: allowlisted, English-facing contest bundle builder.

---

### Task 0: Bootstrap the local verification environment

**Files:**
- Create: `package-lock.json`
- Modify other files only when the baseline reveals a pre-existing environment declaration defect.

**Interfaces:**
- Local Python commands use `.venv/bin/python`, `.venv/bin/pytest`, and `.venv/bin/ruff`.
- Browser commands use the repository's npm lockfile and local Playwright binary.
- PDF visual verification requires `pdftoppm` and `pdfinfo` from Poppler.

- [ ] **Step 1: Create the ignored Python 3.11 environment and install current dependencies**

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

- [ ] **Step 2: Install browser-test dependencies**

```bash
npm install
npx playwright install chromium
```

- [ ] **Step 3: Install Poppler when the commands are absent**

On macOS, first check `command -v pdftoppm` and `command -v pdfinfo`. If either is absent, run:

```bash
brew install poppler
```

On the Radeon Linux environment, use the image's package manager only if Poppler is absent; this dependency is for document rendering and does not affect the application runtime.

- [ ] **Step 4: Capture the baseline before implementation**

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Record any baseline failure before changing application code. Do not treat a pre-existing failure as caused by later tasks.

- [ ] **Step 5: Commit the npm dependency lock**

```bash
git add package-lock.json
git commit -m "build: lock browser test dependencies"
```

---

### Task 1: Add creator intents and enforce a GPU-only Torch backend

**Files:**
- Modify: `command/labels.py`
- Modify: `backend/schemas.py`
- Modify: `command/inference.py`
- Modify: `.env.example`
- Modify: `scripts/setup_amd_real.sh`
- Modify: `scripts/start_real_rocm.sh`
- Modify: `scripts/amd_real_oneclick.sh`
- Modify: `scripts/smoke_rocm.sh`
- Test: `tests/test_command_classifier.py`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Produces: `CommandIntent.START_RECORDING`, `CommandIntent.STOP_RECORDING`, and `CommandIntent.CAPTURE_FRAME`.
- Produces: `require_rocm_device(torch_module: object) -> str`, returning only `"cuda:0"` or raising `RuntimeError`.
- Preserves: `build_command_classifier(settings: Settings) -> CommandClassifierBackend`.
- Changes: only the three creator intents are executable; legacy smart-space labels remain recognizable but produce `action="ignore"` because this repository has no lighting or access-control integration.

- [ ] **Step 1: Write failing intent and GPU-contract tests**

Add to `tests/test_command_classifier.py`:

```python
from types import SimpleNamespace
import pytest

from command.inference import require_rocm_device
from command.labels import EXECUTABLE_INTENTS


def test_creator_intents_are_executable():
    assert {
        CommandIntent.START_RECORDING,
        CommandIntent.STOP_RECORDING,
        CommandIntent.CAPTURE_FRAME,
    } <= EXECUTABLE_INTENTS


def test_unimplemented_smart_space_intents_are_not_executable():
    assert {
        CommandIntent.LIGHT_ON,
        CommandIntent.LIGHT_OFF,
        CommandIntent.OPEN_DOOR,
    }.isdisjoint(EXECUTABLE_INTENTS)


def test_rocm_device_contract_rejects_cpu_only_torch():
    torch_module = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(RuntimeError, match="ROCm"):
        require_rocm_device(torch_module)


def test_rocm_device_contract_accepts_hip_device():
    torch_module = SimpleNamespace(
        version=SimpleNamespace(hip="6.2"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert require_rocm_device(torch_module) == "cuda:0"
```

Update `tests/test_deployment_files.py` to require `COMMAND_BACKEND="${COMMAND_BACKEND:-torch}"` in all real-mode scripts and to reject the old prototype default. Assert `.env.example` uses current `COMMAND_*` keys and contains none of the removed MPC001 settings.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```bash
.venv/bin/pytest tests/test_command_classifier.py tests/test_deployment_files.py -q
```

Expected: FAIL because the creator enum members and `require_rocm_device` do not exist and the scripts still default to `prototype`.

- [ ] **Step 3: Extend the shared label and schema vocabulary**

Add the three creator intents to `CommandIntent`, append them to `COMMAND_LABELS`, make them the only values in `EXECUTABLE_INTENTS`, and add matching values to `CommandIntentName` in `backend/schemas.py`. Update the existing agent-policy test to use `START_RECORDING` for its execute case. Make the fake backend return `CAPTURE_FRAME` for its accepted high-motion path so local fake mode exercises a real creator action. Add plain starter phrases for calibration:

```python
CommandIntent.START_RECORDING: ["start recording", "开始录制"],
CommandIntent.STOP_RECORDING: ["stop recording", "停止录制"],
CommandIntent.CAPTURE_FRAME: ["capture frame", "拍一张照片"],
```

- [ ] **Step 4: Make Torch construction fail without ROCm**

Add to `command/inference.py`:

```python
def require_rocm_device(torch_module: object) -> str:
    hip = getattr(getattr(torch_module, "version", None), "hip", None)
    cuda = getattr(torch_module, "cuda", None)
    available = bool(cuda and cuda.is_available())
    if hip is None or not available:
        raise RuntimeError("AMD ROCm GPU is required for the Torch command backend")
    return "cuda:0"
```

Replace the CPU fallback in `TorchCommandClassifierBackend.__init__` with:

```python
self.device = torch.device(require_rocm_device(torch))
```

Include `backend="torch"`, `device=str(self.device)`, and `rocmHip=str(torch.version.hip)` in result metadata.

- [ ] **Step 5: Switch every real-mode script to the Torch backend**

Use this default in `setup_amd_real.sh`, `start_real_rocm.sh`, `amd_real_oneclick.sh`, and `smoke_rocm.sh`:

```bash
export COMMAND_BACKEND="${COMMAND_BACKEND:-torch}"
export COMMAND_CLASSIFIER_CHECKPOINT="${COMMAND_CLASSIFIER_CHECKPOINT:-$PERSISTENCE_ROOT/models/command_classifier.pt}"
```

Remove real-mode branches that create a prototype profile. Keep prototype startup available only through an explicit local command documented later.

Replace stale `.env.example` settings with the current `Settings` names. Use `COMMAND_BACKEND=fake` as the safe local default, include a commented checkpoint example, retain capture, threshold, privacy, host, and port settings that the application still reads, and remove `MPC001_*`, `WINDOW_FRAMES`, and `INFERENCE_STRIDE`.

- [ ] **Step 6: Run focused and schema tests**

Run:

```bash
.venv/bin/pytest tests/test_command_classifier.py tests/test_config.py tests/test_deployment_files.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add command/labels.py backend/schemas.py command/inference.py .env.example scripts/setup_amd_real.sh scripts/start_real_rocm.sh scripts/amd_real_oneclick.sh scripts/smoke_rocm.sh tests/test_command_classifier.py tests/test_deployment_files.py
git commit -m "feat: require ROCm for creator command inference"
```

---

### Task 2: Build a reproducible calibration-to-checkpoint dataset path

**Files:**
- Create: `command/dataset.py`
- Create: `scripts/build_command_manifest.py`
- Modify: `scripts/train_command_classifier.py`
- Modify: `scripts/validate_command_classifier.py`
- Delete: `scripts/record_command_manifest.py`
- Test: `tests/test_command_dataset.py`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Produces: `ManifestSummary(total: int, train: int, validation: int, per_intent: dict[str, dict[str, int]])`.
- Produces: `build_manifest(root: Path, output: Path, validation_fraction: float = 0.2, seed: str = "silent-vision-v1") -> ManifestSummary`.
- Manifest rows contain `sample_id`, `intent`, `phrase`, `language`, `mouth_roi_npy`, `metadata_path`, and `split`.
- `load_samples(manifest: Path, extractor, split: str) -> list[tuple[np.ndarray, int]]` consumes only the requested split.

- [ ] **Step 1: Write failing dataset discovery and split tests**

Create `tests/test_command_dataset.py` with a helper that writes two saved samples for every command label, then assert exact behavior:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from command.dataset import DatasetError, build_manifest
from command.labels import COMMAND_LABELS


def write_sample(root: Path, intent: str, sample_id: str) -> None:
    sample = root / "profiles" / "global" / intent / sample_id
    sample.mkdir(parents=True)
    np.save(sample / "mouth_roi.npy", np.zeros((25, 96, 96), dtype=np.uint8))
    (sample / "metadata.json").write_text(json.dumps({
        "intent": intent,
        "phrase": f"{intent} phrase",
        "language": "en",
    }))


def test_build_manifest_uses_real_samples_and_splits_each_intent(tmp_path):
    for intent in COMMAND_LABELS:
        write_sample(tmp_path, intent.value, "sample-a")
        write_sample(tmp_path, intent.value, "sample-b")
    output = tmp_path / "manifest.jsonl"

    summary = build_manifest(tmp_path, output, validation_fraction=0.5)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary.total == len(COMMAND_LABELS) * 2
    assert {row["split"] for row in rows} == {"train", "validation"}
    assert all(Path(row["mouth_roi_npy"]).exists() for row in rows)
    assert all(summary.per_intent[intent.value] == {"train": 1, "validation": 1} for intent in COMMAND_LABELS)


def test_build_manifest_rejects_an_intent_with_one_sample(tmp_path):
    for intent in COMMAND_LABELS:
        write_sample(tmp_path, intent.value, "sample-a")
        if intent.value != "CAPTURE_FRAME":
            write_sample(tmp_path, intent.value, "sample-b")
    with pytest.raises(DatasetError, match="CAPTURE_FRAME"):
        build_manifest(tmp_path, tmp_path / "manifest.jsonl")
```

- [ ] **Step 2: Run the dataset tests and confirm failure**

Run:

```bash
.venv/bin/pytest tests/test_command_dataset.py -q
```

Expected: FAIL because `command.dataset` does not exist.

- [ ] **Step 3: Implement deterministic manifest construction**

In `command/dataset.py`, scan only `profiles/global/<intent>/<sample>/mouth_roi.npy`. Validate the array is readable, load adjacent metadata, group by every value in `COMMAND_LABELS`, require at least two usable samples per label, order each group by `sha256(f"{seed}:{sample_id}")`, allocate at least one validation row and one training row, and write JSONL atomically through `output.with_suffix(output.suffix + ".tmp")` followed by `replace()`.

Move `load_samples` out of `scripts/train_command_classifier.py` into `command/dataset.py`, filter rows by the exact requested split, reject unknown split values, and import it from both tests and the training CLI.

In `scripts/build_command_manifest.py`, expose:

```bash
/opt/venv/bin/python scripts/build_command_manifest.py \
  --root /workspace/persistent/silent-vision \
  --output /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --validation-fraction 0.2
```

Print the serialized `ManifestSummary` as JSON.

- [ ] **Step 4: Make training and validation consume explicit splits**

Add `--split` and `--summary` arguments to training, defaulting to `train` and the output checkpoint name plus `.training.json`. Call `require_rocm_device(torch)` before model construction so training cannot run on CPU or non-ROCm CUDA. Record device, HIP version, sample count, epochs, feature dimension, final loss, checkpoint path, and label order.

Make validation default to `validation`, serialize enum values with `decision.intent.value`, and add `--output` for a JSON report containing `total`, `acceptedCount`, `correctAccepted`, `overallAccuracy` (`correctAccepted / total`), `acceptedPrecision` (`correctAccepted / acceptedCount`), rejection rate, confusion, backend, device, HIP version, and checkpoint path. Use `null` rather than a fabricated number when `acceptedCount` is zero.

Delete `scripts/record_command_manifest.py` and update deployment-file tests to require `scripts/build_command_manifest.py` instead.

- [ ] **Step 5: Add split-loader tests**

Add a small manifest with one train and one validation row to `tests/test_command_dataset.py`. Use `StatisticalVisualFeatureExtractor(256)` and assert `load_samples(..., split="train")` returns one item and `load_samples(..., split="validation")` returns the other.

- [ ] **Step 6: Run dataset and classifier tests**

Run:

```bash
.venv/bin/pytest tests/test_command_dataset.py tests/test_command_classifier.py tests/test_deployment_files.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add command/dataset.py scripts/build_command_manifest.py scripts/train_command_classifier.py scripts/validate_command_classifier.py scripts/record_command_manifest.py tests/test_command_dataset.py tests/test_deployment_files.py
git commit -m "feat: train commands from calibration samples"
```

---

### Task 3: Implement a reusable browser creator controller

**Files:**
- Modify: `frontend/camera.js`
- Create: `frontend/creator.js`
- Create: `tests/e2e/creator.spec.js`

**Interfaces:**
- `ClipRecorder.getStream() -> MediaStream | null` exposes but does not transfer ownership of the preview stream.
- `CreatorController.startRecording() -> void` throws when already recording or when no stream exists.
- `CreatorController.stopRecording() -> Promise<Blob>` resolves to a non-empty WebM.
- `CreatorController.captureFrame() -> Promise<Blob>` resolves to a non-empty PNG.
- `CreatorController.dispose() -> void` stops an active creator recorder without stopping the shared camera stream.

- [ ] **Step 1: Write browser-level state-machine tests**

Create `tests/e2e/creator.spec.js`. Import `/static/creator.js` into the page and inject a fake `MediaRecorder` whose `stop()` emits one WebM chunk. Assert:

```javascript
controller.startRecording();
expect(() => controller.startRecording()).toThrow(/already recording/);
const clip = await controller.stopRecording();
expect(clip.type).toContain("video/webm");
expect(clip.size).toBeGreaterThan(0);
await expect(controller.stopRecording()).rejects.toThrow(/not recording/);
```

Inject `captureFrameImpl: async () => new Blob(["png"], { type: "image/png" })` and assert `captureFrame()` returns that blob. Inject `streamProvider: () => null`; assert `startRecording()` throws and `captureFrame()` rejects with `camera stream is unavailable`. Assert `dispose()` does not call `stop()` on any track in the shared stream.

- [ ] **Step 2: Run the Playwright test and confirm failure**

Start fake mode in one terminal:

```bash
COMMAND_BACKEND=fake uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Run in another:

```bash
npx playwright test tests/e2e/creator.spec.js
```

Expected: FAIL because `/static/creator.js` does not exist.

- [ ] **Step 3: Expose the shared camera stream**

Add to `ClipRecorder`:

```javascript
getStream() {
  return this.stream;
}
```

Keep `recordClip()` independent: it may create a short command `MediaRecorder` while another recorder uses the same stream, and it must never stop tracks itself.

- [ ] **Step 4: Implement `CreatorController`**

Constructor dependencies:

```javascript
constructor({
  video,
  streamProvider,
  mediaRecorderFactory = (stream, options) => new MediaRecorder(stream, options),
  captureFrameImpl = null,
  mimeType = "video/webm;codecs=vp9",
})
```

Use a single active recorder, collect non-empty chunks, retain a pending stop promise, and reject invalid transitions. The default capture implementation sizes a temporary canvas from `video.videoWidth` and `video.videoHeight`, draws the video, and resolves `canvas.toBlob(..., "image/png")`; reject zero dimensions and a null blob.

- [ ] **Step 5: Run the browser tests**

```bash
npx playwright test tests/e2e/creator.spec.js tests/e2e/camera.spec.js
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add frontend/camera.js frontend/creator.js tests/e2e/creator.spec.js
git commit -m "feat: add browser creator recorder"
```

---

### Task 4: Wire accepted visual commands to real creator artifacts

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/styles.css`
- Modify: `frontend/websocket.js`
- Modify: `tests/e2e/camera.spec.js`
- Modify: `tests/test_deployment_files.py`

**Interfaces:**
- Creator Mode owns one `ClipRecorder`, one `CreatorController`, and one WebSocket session until the user exits.
- `executeCreatorIntent(intent: string) -> Promise<void>` accepts only the three creator intents.
- `executeAcceptedCreatorResult(event: object, executeIntent: function) -> Promise<void>` calls the injected executor only when `event.action === "execute"`.
- `publishArtifact(blob: Blob, filename: string) -> void` revokes the previous object URL before showing the new download.

- [ ] **Step 1: Write failing UI contract tests**

Extend `tests/e2e/camera.spec.js` to require visible controls and status:

```javascript
await expect(page.locator("#creatorModeButton")).toHaveText("Enter Creator Mode");
await expect(page.locator("#startButton")).toHaveText("Recognize Command");
await expect(page.locator("#creatorStatus")).toContainText("inactive");
await expect(page.locator("#creatorArtifact")).toBeHidden();
```

Extend `tests/test_deployment_files.py` to assert `frontend/websocket.js` imports `CreatorController`, maps all three creator intents, and checks `event.action === "execute"` before calling `executeCreatorIntent`.

- [ ] **Step 2: Run focused tests and confirm failure**

```bash
.venv/bin/pytest tests/test_deployment_files.py -q
npx playwright test tests/e2e/camera.spec.js
```

Expected: FAIL because Creator Mode elements and wiring do not exist.

- [ ] **Step 3: Add Creator Mode UI**

Add an English-only creator section with:

```html
<button id="creatorModeButton" type="button">Enter Creator Mode</button>
<p id="creatorStatus">inactive</p>
<a id="creatorArtifact" hidden download>Download latest artifact</a>
```

Rename the existing start button to `Recognize Command`. Add compact status styling and a visible recording state without adding decorative AI imagery.

- [ ] **Step 4: Refactor the session lifecycle for persistent Creator Mode**

In `frontend/websocket.js`:

- Entering Creator Mode starts preview, creates `CreatorController`, and connects once.
- Recognize Command records a short clip on the existing camera stream and sends `clip.start` plus binary data without closing the camera.
- `agent.result` awaits creator action before returning the UI to ready state.
- Exiting Creator Mode cancels command capture, disposes the creator recorder, revokes any object URL, closes the socket, and stops camera tracks.
- Calibration explicitly exits Creator Mode first and keeps its existing one-shot behavior.
- Because creator execution is asynchronous, make `handleEvent` asynchronous and have `socket.onmessage` call it with an explicit rejection handler that reports the error and restores a usable UI state.

Export and use this gate so it can be tested without a production-only test hook:

```javascript
export async function executeAcceptedCreatorResult(event, executeIntent) {
  if (event.action === "execute") {
    await executeIntent(event.arguments.intent);
  }
}
```

The `agent.result` handler awaits `executeAcceptedCreatorResult(event, executeCreatorIntent)` and then returns the UI to its ready state. Map `START_RECORDING` to `creator.startRecording()`, `STOP_RECORDING` to `await creator.stopRecording()` and a dated `.webm` download, and `CAPTURE_FRAME` to `await creator.captureFrame()` and a dated `.png` download. Legacy smart-space intents arrive as `action="ignore"`, remain visible decisions, and do not pretend to control a device.

- [ ] **Step 5: Add an integration test for rejected-command safety**

In `tests/e2e/camera.spec.js`, import `executeAcceptedCreatorResult` from `/static/websocket.js`. Inject a counter function, pass `{action: "reject", arguments: {intent: "CAPTURE_FRAME"}}`, and assert the counter stays at zero; pass an execute result and assert it increments once. Add an ignore case for `LIGHT_ON` and assert the counter remains unchanged.

- [ ] **Step 6: Run frontend and backend lifecycle tests**

```bash
.venv/bin/pytest tests/test_deployment_files.py tests/test_websocket_flow.py -q
npx playwright test tests/e2e
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add frontend/index.html frontend/styles.css frontend/websocket.js tests/e2e/camera.spec.js tests/test_deployment_files.py
git commit -m "feat: execute visual creator commands"
```

---

### Task 5: Write the English README and submission source copy

**Files:**
- Modify: `README.md`
- Create: `docs/submission/project-profile-source.md`
- Create: `docs/submission/poster-copy.md`
- Create: `submission/README.md`
- Create: `submission/demo-video-script.md`
- Create: `submission/pull-request-description.md`
- Create: `tests/test_submission_docs.py`

**Interfaces:**
- `README.md` is the canonical reproduction guide.
- `docs/submission/*.md` is the canonical editable copy for generated visual artifacts.
- `submission/README.md` maps official requirements to stable relative paths.

- [ ] **Step 1: Write failing documentation-contract tests**

Create `tests/test_submission_docs.py`:

```python
from pathlib import Path

REQUIRED_README_HEADINGS = [
    "## What Silent Vision Does",
    "## Creator Workflow",
    "## Architecture",
    "## AMD Radeon and ROCm",
    "## Requirements and Dependencies",
    "## Train the Command Classifier",
    "## GPU-Only Startup",
    "## Verification",
    "## Privacy and Limitations",
    "## Submission Materials",
]


def test_readme_is_a_complete_reproduction_guide():
    text = Path("README.md").read_text()
    assert all(heading in text for heading in REQUIRED_README_HEADINGS)
    assert "COMMAND_BACKEND=torch" in text
    assert "build_command_manifest.py" in text
    assert "prototype mode is for calibration" in text.lower()


def test_submission_copy_avoids_unverified_marketing_claims():
    paths = [Path("README.md"), *Path("submission").glob("*.md"), *Path("docs/submission").glob("*.md")]
    banned = {"revolutionary", "game-changing", "cutting-edge", "next-generation", "harness the power of ai"}
    combined = "\n".join(path.read_text().lower() for path in paths if path.exists())
    assert not any(phrase in combined for phrase in banned)
```

Also assert the submission index names the profile PDF, poster PDF/PNG, demo script, source repository, and exact PR title.

- [ ] **Step 2: Run the doc tests and confirm failure**

```bash
.venv/bin/pytest tests/test_submission_docs.py -q
```

Expected: FAIL because the required files and headings do not exist.

- [ ] **Step 3: Rewrite the root README**

Use the headings listed in the test. Include exact dependencies from `requirements.txt`, browser requirements, the persistent storage layout, calibration guidance of 5-10 correctly labeled samples per intent, manifest creation, train and validation commands, GPU-only startup, public tunnel command, fake-mode tests, Docker notes, privacy behavior, and known limitations.

State directly that preprocessing runs on CPU and temporal classification runs on Radeon/ROCm. Do not include measured performance values.

- [ ] **Step 4: Write the profile and poster source copy**

`project-profile-source.md` must contain the six required sections with concrete details from the code. `poster-copy.md` must contain only text that appears on the poster, including the headline `Silent control when audio is not an option.` and the repository URL.

- [ ] **Step 5: Write the submission index, demo script, and PR description**

The demo script must fit 3-5 minutes at a normal speaking pace and include terminal proof, Creator Mode artifact generation, rejection safety, and closing repository link. The PR description must contain an unchecked video line phrased as `Demo video: added after the recorded Radeon run` until the real URL exists; this status is a truthful workflow note, not a fabricated deliverable.

- [ ] **Step 6: Run the documentation tests**

```bash
.venv/bin/pytest tests/test_submission_docs.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add README.md docs/submission submission tests/test_submission_docs.py
git commit -m "docs: write Track 1 submission copy"
```

---

### Task 6: Generate and visually verify the profile PDF and poster

**Files:**
- Modify: `requirements-dev.txt`
- Create: `scripts/generate_submission_assets.py`
- Create: `submission/Silent-Vision-Project-Profile.pdf`
- Create: `submission/Silent-Vision-Poster.pdf`
- Create: `submission/Silent-Vision-Poster.png`
- Create: `output/pdf/Silent-Vision-Project-Profile.pdf`
- Create: `output/pdf/Silent-Vision-Poster.pdf`
- Modify: `tests/test_submission_docs.py`

**Interfaces:**
- `build_profile_pdf(output: Path) -> None` creates exactly six A4 pages.
- `build_poster_pdf(output: Path) -> None` creates one A3 portrait page.
- `build_poster_png(pdf: Path, output: Path) -> None` creates a legible raster preview.
- The script reads reviewed copy from `docs/submission/` and writes stable artifacts to both `submission/` and `output/pdf/`.

- [ ] **Step 1: Add failing PDF structure tests**

Extend `tests/test_submission_docs.py`:

```python
from pypdf import PdfReader


def test_profile_and_poster_pdf_structure():
    profile = PdfReader("submission/Silent-Vision-Project-Profile.pdf")
    poster = PdfReader("submission/Silent-Vision-Poster.pdf")
    assert len(profile.pages) == 6
    assert len(poster.pages) == 1
    profile_text = "\n".join(page.extract_text() or "" for page in profile.pages)
    for phrase in ["Target users", "System architecture", "Model and algorithm", "AMD Radeon and ROCm"]:
        assert phrase in profile_text
    assert Path("submission/Silent-Vision-Poster.png").exists()
```

- [ ] **Step 2: Run the PDF test and confirm failure**

```bash
.venv/bin/pytest tests/test_submission_docs.py::test_profile_and_poster_pdf_structure -q
```

Expected: FAIL because the assets do not exist.

- [ ] **Step 3: Add document-generation dependencies**

Append these bounded development dependencies to `requirements-dev.txt`; do not add them to runtime `requirements.txt`:

```text
reportlab>=4.4.4,<5.0.0
pypdf>=6.1.1,<7.0.0
pdfplumber>=0.11.7,<1.0.0
qrcode[pil]>=8.2,<9.0.0
```

Install the updated development set:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

- [ ] **Step 4: Implement deterministic asset generation**

Use ReportLab with built-in Helvetica fonts, A4 pages for the profile, and A3 portrait for the poster. Implement reusable helpers for page headers, footers, wrapped paragraphs, labeled cards, arrows, and architecture boxes. Use charcoal `#101418`, off-white `#F4F1EA`, and Radeon red `#ED1C24`. Generate a QR code for `https://github.com/fangjixin/silent-vision`.

Copy completed PDFs to both stable destinations. Render the poster PDF to PNG at 150 DPI through `pdftoppm`; fail with a clear message when Poppler is unavailable.

- [ ] **Step 5: Generate and run machine checks**

```bash
.venv/bin/python scripts/generate_submission_assets.py
pdfinfo submission/Silent-Vision-Project-Profile.pdf
pdfinfo submission/Silent-Vision-Poster.pdf
.venv/bin/pytest tests/test_submission_docs.py -q
```

Expected: six-page profile, one-page A3 poster, and passing tests.

- [ ] **Step 6: Render every PDF page for visual QA**

```bash
mkdir -p tmp/pdfs/profile tmp/pdfs/poster
pdftoppm -png -r 150 submission/Silent-Vision-Project-Profile.pdf tmp/pdfs/profile/page
pdftoppm -png -r 150 submission/Silent-Vision-Poster.pdf tmp/pdfs/poster/page
```

Inspect all seven PNGs with the image viewer. Fix any clipping, overlap, dense copy, weak contrast, broken QR code, or inconsistent page numbering, then regenerate and re-render until no defect remains.

- [ ] **Step 7: Commit Task 6**

```bash
git add requirements-dev.txt scripts/generate_submission_assets.py submission/Silent-Vision-Project-Profile.pdf submission/Silent-Vision-Poster.pdf submission/Silent-Vision-Poster.png output/pdf/Silent-Vision-Project-Profile.pdf output/pdf/Silent-Vision-Poster.pdf tests/test_submission_docs.py
git commit -m "docs: generate Track 1 profile and poster"
```

---

### Task 7: Build an allowlisted contest bundle

**Files:**
- Modify: `.gitignore`
- Create: `scripts/build_contest_bundle.py`
- Create: `tests/test_contest_bundle.py`

**Interfaces:**
- `build_bundle(source: Path, destination: Path) -> list[Path]` copies only approved paths.
- Default destination: `dist/contest/submissions/track1-silent-vision`.
- The bundle contains runtime source, tests, deployment files, dependencies, root README, and `submission/`; it excludes `.git`, environments, datasets, checkpoints, logs, caches, `docs/superpowers`, and non-English historical plans.

- [ ] **Step 1: Write failing allowlist and safety tests**

Create `tests/test_contest_bundle.py`:

```python
from pathlib import Path

from scripts.build_contest_bundle import build_bundle


def test_bundle_contains_complete_runtime_and_public_materials(tmp_path):
    target = tmp_path / "submissions" / "track1-silent-vision"
    copied = build_bundle(Path.cwd(), target)
    assert target.joinpath("backend/main.py").exists()
    assert target.joinpath("frontend/websocket.js").exists()
    assert target.joinpath("README.md").exists()
    assert target.joinpath("submission/Silent-Vision-Project-Profile.pdf").exists()
    assert copied


def test_bundle_excludes_internal_and_sensitive_paths(tmp_path):
    target = tmp_path / "bundle"
    build_bundle(Path.cwd(), target)
    assert not target.joinpath("docs/superpowers").exists()
    assert not target.joinpath(".env").exists()
    assert not target.joinpath("models").exists()
    assert not target.joinpath(".git").exists()
    assert not any(path.is_symlink() for path in target.rglob("*"))
```

- [ ] **Step 2: Run bundle tests and confirm failure**

```bash
.venv/bin/pytest tests/test_contest_bundle.py -q
```

Expected: FAIL because the builder does not exist.

- [ ] **Step 3: Implement explicit directory and root-file allowlists**

Allow runtime directories `agent`, `api`, `backend`, `command`, `docker`, `frontend`, `scripts`, `session`, `tests`, `video`, and `vision`; allow only `docs/submission` under `docs`; allow root files `.env.example`, `.gitignore`, `README.md`, `package.json`, `package-lock.json`, `pytest.ini`, `requirements.txt`, and `requirements-dev.txt`; allow the complete `submission` directory. Skip `__pycache__`, generated reports, model files, recordings, and any file larger than 50 MiB. Reject symlinks rather than following them.

Add `dist/` and `tmp/pdfs/` to `.gitignore`.

- [ ] **Step 4: Add a public-text audit**

After copying, scan contest-facing Markdown outside `tests/` for banned generic marketing phrases. Confirm every Markdown file decodes as UTF-8. Do not reject Chinese product sample phrases in Python, JavaScript, or test fixtures.

- [ ] **Step 5: Run bundle tests and inspect output**

```bash
.venv/bin/pytest tests/test_contest_bundle.py -q
.venv/bin/python scripts/build_contest_bundle.py
find dist/contest/submissions/track1-silent-vision -type f | sort
du -sh dist/contest/submissions/track1-silent-vision
```

Expected: tests pass; no internal design docs, secrets, data, or checkpoints appear.

- [ ] **Step 6: Commit Task 7**

```bash
git add .gitignore scripts/build_contest_bundle.py tests/test_contest_bundle.py
git commit -m "build: create contest submission bundle"
```

---

### Task 8: Run complete local verification and prepare the Radeon handoff

**Files:**
- Modify only if verification exposes a defect in files from Tasks 1-7.

**Interfaces:**
- Produces: a clean local test result, rendered visual QA set, contest bundle, and exact Radeon commands for data collection, training, validation, startup, and video recording.

- [ ] **Step 1: Run formatting and the complete Python suite**

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
```

Expected: zero Ruff errors and all tests pass.

- [ ] **Step 2: Run the complete browser suite**

Start fake mode:

```bash
COMMAND_BACKEND=fake uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then run:

```bash
npx playwright test tests/e2e
```

Expected: all browser tests pass, including creator state transitions and rejected-command safety.

- [ ] **Step 3: Rebuild and inspect submission artifacts**

```bash
.venv/bin/python scripts/generate_submission_assets.py
.venv/bin/python scripts/build_contest_bundle.py
git diff --check
git status --short
```

Expected: deterministic artifacts, no whitespace errors, and only intended changes.

- [ ] **Step 4: Run the local fake smoke test**

```bash
./scripts/smoke_fake.sh
```

Expected: PASS. Do not run `smoke_rocm.sh` on a machine without Radeon/ROCm.

- [ ] **Step 5: Record the exact Radeon handoff commands**

Use these commands on the Radeon server after recording at least 5-10 correctly labeled samples for every intent:

```bash
cd /workspace/template-repos/template-907/repo
/opt/venv/bin/python scripts/build_command_manifest.py \
  --root /workspace/persistent/silent-vision \
  --output /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --validation-fraction 0.2

/opt/venv/bin/python scripts/train_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --output /workspace/persistent/silent-vision/models/command_classifier.pt \
  --summary /workspace/persistent/silent-vision/models/training-run.json

/opt/venv/bin/python scripts/validate_command_classifier.py \
  --manifest /workspace/persistent/silent-vision/commands/manifest.jsonl \
  --checkpoint /workspace/persistent/silent-vision/models/command_classifier.pt \
  --output /workspace/persistent/silent-vision/models/validation-report.json

COMMAND_BACKEND=torch \
COMMAND_CLASSIFIER_CHECKPOINT=/workspace/persistent/silent-vision/models/command_classifier.pt \
bash scripts/amd_real_oneclick.sh
```

In a second terminal:

```bash
$HOME/.local/bin/rc-tunnel expose --port 8000
```

- [ ] **Step 6: Stop at the Radeon evidence gate**

Do not claim the submission is complete and do not open the final contest PR until the user supplies or records:

- a successful training-run JSON from Radeon,
- a held-out validation report,
- the 3-5 minute real workflow video,
- and the final video URL added to `submission/README.md` and `submission/pull-request-description.md`.

- [ ] **Step 7: Commit verification-only fixes, if any**

If verification exposes a defect, return to the task that owns the affected file, add a regression test there, repeat that task's focused and full verification, and use that task's explicit file list for the corrective commit. If no files changed, do not create an empty commit.

---

## Final PR Procedure After Radeon Evidence Exists

1. Regenerate submission PDFs if measured results are added; render and inspect every page again.
2. Rebuild `dist/contest/submissions/track1-silent-vision`.
3. Fork `AMD-DEV-CONTEST/Radeon-hackathon-2026-07` using an authenticated GitHub session.
4. Copy the built bundle into the fork at `submissions/track1-silent-vision/`.
5. Commit and push the fork branch without adding credentials, recordings, datasets, or model checkpoints.
6. Open the English pull request titled `Track 1, Jixin Fang, Silent Vision` using the reviewed description.
7. Open every PDF, poster, source, and video link from the PR diff before submission.
