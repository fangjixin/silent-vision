# Global and Personal Prototype Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add few-shot prototype command recognition so the system can use server-provided global templates and optional anonymous per-browser personal templates before falling back to classifier/rejection.

**Architecture:** Browser records utterance-level video clips as already implemented. The backend converts each clip to a stable mouth ROI, extracts deterministic embeddings, stores global/personal labeled prototypes, and predicts by cosine similarity with confidence and margin rejection. Personal prototypes are selected by anonymous `profileId`; if personal data exists it is used first, then global prototypes are used as fallback.

**Tech Stack:** FastAPI WebSocket, PyAV, MediaPipe, NumPy, PyTorch optional, browser `localStorage`, existing `CommandDecision` schema.

## Global Constraints

- No user login; use anonymous browser-generated `profileId`.
- Labels are command intents: `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, `UNKNOWN`.
- Chinese and English spoken variants map to the same business intent.
- Rejected commands must not call LLM or execute tools.
- Save original video, mouth ROI, embedding, logits/similarity metadata for debug/sample requests.
- Keep existing `COMMAND_BACKEND=fake|torch` path; add prototype backend without deleting classifier training scripts.
- Use `/workspace/persistent/silent-vision` on AMD through `PERSISTENCE_ROOT`.

---

## File Structure

- `command/prototype.py`: prototype embedding, sample persistence, manifest format, cosine matching, global/personal store loading.
- `command/inference.py`: add `PrototypeCommandClassifierBackend` and backend selection.
- `backend/config.py`: add prototype settings and thresholds.
- `backend/schemas.py`: add sample-save and prototype metadata schemas if needed.
- `api/websocket.py`: handle calibration/sample upload messages and include `profileId`.
- `frontend/index.html`: add calibration controls.
- `frontend/websocket.js`: create/read anonymous `profileId`, send calibration metadata, handle sample-save results.
- `frontend/camera.js`: reuse clip recording for both inference and calibration.
- `scripts/build_global_prototypes.py`: build global profile from saved samples.
- `scripts/inspect_prototypes.py`: print prototype counts and thresholds.
- `tests/test_prototype_recognition.py`: unit tests for embedding, persistence, matching, threshold rejection.
- `tests/test_deployment_files.py`: static checks for no-login profile and rejected no-LLM path.
- `README.md`: update AMD workflow.

---

### Task 1: Prototype Core Library

**Files:**
- Create: `command/prototype.py`
- Test: `tests/test_prototype_recognition.py`

**Interfaces:**
- Produces:
  - `extract_roi_embedding(mouth_frames: np.ndarray, feature_dim: int = 128) -> np.ndarray`
  - `save_prototype_sample(root: Path, profile_id: str, intent: str, mouth_frames: np.ndarray, metadata: dict[str, Any]) -> Path`
  - `load_profile_prototypes(root: Path, profile_id: str) -> list[PrototypeSample]`
  - `match_prototypes(embedding: np.ndarray, samples: Sequence[PrototypeSample], confidence_threshold: float, margin_threshold: float) -> PrototypeMatch`

- [ ] **Step 1: Write failing tests for deterministic embeddings**

```python
import numpy as np

from command.prototype import extract_roi_embedding


def test_extract_roi_embedding_is_unit_normalized_and_deterministic():
    frames = np.zeros((10, 96, 96), dtype=np.uint8)
    frames[3:7, 32:64, 32:64] = 255

    first = extract_roi_embedding(frames, feature_dim=128)
    second = extract_roi_embedding(frames, feature_dim=128)

    assert first.shape == (128,)
    assert np.allclose(first, second)
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5
```

- [ ] **Step 2: Write failing tests for save/load**

```python
import numpy as np

from command.labels import CommandIntent
from command.prototype import load_profile_prototypes, save_prototype_sample


def test_save_and_load_profile_prototype(tmp_path):
    frames = np.ones((8, 96, 96), dtype=np.uint8) * 120

    path = save_prototype_sample(
        tmp_path,
        profile_id="global",
        intent=CommandIntent.LIGHT_ON.value,
        mouth_frames=frames,
        metadata={"language": "zh", "phrase": "请开灯"},
    )

    samples = load_profile_prototypes(tmp_path, "global")

    assert path.exists()
    assert len(samples) == 1
    assert samples[0].profile_id == "global"
    assert samples[0].intent == CommandIntent.LIGHT_ON.value
    assert samples[0].metadata["phrase"] == "请开灯"
```

- [ ] **Step 3: Write failing tests for threshold and margin rejection**

```python
import numpy as np

from command.prototype import PrototypeSample, match_prototypes


def sample(intent: str, vector: np.ndarray) -> PrototypeSample:
    return PrototypeSample(
        profile_id="global",
        intent=intent,
        embedding=vector.astype(np.float32),
        sample_path=None,
        metadata={},
    )


def test_match_prototypes_accepts_clear_top1():
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    samples = [
        sample("LIGHT_ON", np.array([1.0, 0.0, 0.0])),
        sample("LIGHT_OFF", np.array([0.0, 1.0, 0.0])),
    ]

    match = match_prototypes(query, samples, confidence_threshold=0.8, margin_threshold=0.2)

    assert match.accepted is True
    assert match.intent == "LIGHT_ON"
    assert match.reason == "accepted"


def test_match_prototypes_rejects_close_margin():
    query = np.array([1.0, 0.0], dtype=np.float32)
    samples = [
        sample("LIGHT_ON", np.array([1.0, 0.0])),
        sample("LIGHT_OFF", np.array([0.98, 0.2])),
    ]

    match = match_prototypes(query, samples, confidence_threshold=0.5, margin_threshold=0.2)

    assert match.accepted is False
    assert match.intent == "UNKNOWN"
    assert match.reason == "margin_below_threshold"
```

- [ ] **Step 4: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_prototype_recognition.py -q
```

Expected: import errors for `command.prototype`.

- [ ] **Step 5: Implement `command/prototype.py`**

Implementation requirements:

- Store samples under:

```text
{PERSISTENCE_ROOT}/profiles/{profile_id}/{intent}/{sample_id}/
  mouth_roi.npy
  embedding.npy
  metadata.json
```

- `extract_roi_embedding` must be deterministic and CPU-safe. Use mouth ROI statistics, temporal difference statistics, row/column projections, and FFT/DCT-style compact features via NumPy; normalize to unit length.
- `match_prototypes` must average scores by intent, report topK, confidence, and margin.

- [ ] **Step 6: Run tests and confirm pass**

Run:

```bash
python -m pytest tests/test_prototype_recognition.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add command/prototype.py tests/test_prototype_recognition.py
git commit -m "feat: add prototype command matching core"
```

---

### Task 2: Prototype Backend Integration

**Files:**
- Modify: `backend/config.py`
- Modify: `command/inference.py`
- Test: `tests/test_command_classifier.py`

**Interfaces:**
- Consumes:
  - `extract_roi_embedding`
  - `load_profile_prototypes`
  - `match_prototypes`
- Produces:
  - `PrototypeCommandClassifierBackend.predict(...) -> CommandDecision`
  - New backend setting: `COMMAND_BACKEND=prototype`

- [ ] **Step 1: Write failing tests for prototype backend selection and rejection**

```python
import numpy as np

from backend.config import Settings
from command.inference import build_command_classifier


def test_builds_prototype_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE_ROOT", str(tmp_path))
    monkeypatch.setenv("COMMAND_BACKEND", "prototype")

    backend = build_command_classifier(Settings())

    assert backend.__class__.__name__ == "PrototypeCommandClassifierBackend"


def test_prototype_backend_rejects_without_samples(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE_ROOT", str(tmp_path))
    monkeypatch.setenv("COMMAND_BACKEND", "prototype")

    backend = build_command_classifier(Settings())
    frames = np.zeros((25, 96, 96), dtype=np.uint8)
    decision = backend.predict(frames, metadata={"profileId": "abc"})

    assert decision.intent == "UNKNOWN"
    assert decision.accepted is False
    assert decision.reason == "no_prototypes"
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
python -m pytest tests/test_command_classifier.py::test_builds_prototype_backend tests/test_command_classifier.py::test_prototype_backend_rejects_without_samples -q
```

Expected: `COMMAND_BACKEND=prototype` validation error or unknown backend.

- [ ] **Step 3: Update config**

Add settings:

```python
command_backend: Literal["fake", "torch", "prototype"] = "fake"
prototype_feature_dim: int = 128
prototype_confidence_threshold: float = 0.82
prototype_top1_margin: float = 0.12
prototype_prefer_personal: bool = True
```

- [ ] **Step 4: Implement `PrototypeCommandClassifierBackend`**

Behavior:

1. Extract embedding from `mouth_frames`.
2. Read `profileId` from metadata.
3. If `profileId` exists and personal samples exist, match personal samples first.
4. If no accepted personal match, match `global`.
5. Return `CommandDecision` with:
   - `intent`
   - `accepted`
   - `executable`
   - `confidence`
   - `margin`
   - `topK`
   - `logits={intent: score}`
   - `reason`
   - `metadata={"backend": "prototype", "profileId": ..., "profileScope": "personal|global|none"}`

- [ ] **Step 5: Run tests and confirm pass**

Run:

```bash
python -m pytest tests/test_command_classifier.py tests/test_prototype_recognition.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add backend/config.py command/inference.py tests/test_command_classifier.py
git commit -m "feat: add prototype command backend"
```

---

### Task 3: Calibration Upload Over WebSocket

**Files:**
- Modify: `api/websocket.py`
- Modify: `backend/schemas.py`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes:
  - Existing command clip decode and ROI extraction path.
  - `save_prototype_sample(...)`.
- Produces:
  - WebSocket text command `calibration.start`
  - Binary clip following `calibration.start`
  - WebSocket result `calibration.saved`

- [ ] **Step 1: Add static failing test for calibration path**

```python
from pathlib import Path


def test_websocket_has_calibration_upload_path():
    source = Path("api/websocket.py").read_text()

    assert "calibration.start" in source
    assert "calibration.saved" in source
    assert "save_prototype_sample" in source
    assert "profileId" in source
```

- [ ] **Step 2: Run static test and confirm failure**

Run:

```bash
python -m pytest tests/test_deployment_files.py::test_websocket_has_calibration_upload_path -q
```

Expected: assertion failure.

- [ ] **Step 3: Implement calibration state**

Add per-WebSocket pending calibration metadata:

```python
pending_calibration: dict[str, Any] | None = None
```

Handle text message:

```json
{
  "type": "calibration.start",
  "profileId": "anonymous-id",
  "intent": "LIGHT_ON",
  "language": "zh",
  "phrase": "你好，请帮我打开灯",
  "scope": "personal"
}
```

Validate:

- `intent` must be one of command labels.
- `scope` must be `personal` or `global`.
- `global` scope is allowed only when `ALLOW_GLOBAL_PROFILE_WRITE=true`.
- If validation fails, send `calibration.error`.

- [ ] **Step 4: Save calibration sample when the next binary clip arrives**

Use the same pipeline as inference:

1. Save original video when debug is enabled.
2. Decode video to 25 FPS.
3. Extract mouth ROI.
4. Save prototype sample under `profiles/{profileId}` or `profiles/global`.
5. Send:

```json
{
  "type": "calibration.saved",
  "profileId": "anonymous-id",
  "scope": "personal",
  "intent": "LIGHT_ON",
  "samplePath": "...",
  "frames": 75,
  "detectedFrames": 75
}
```

- [ ] **Step 5: Ensure normal inference binary path remains unchanged**

If no `pending_calibration` exists and `RECOGNITION_MODE=command`, binary clips continue to run command inference.

- [ ] **Step 6: Run tests**

Run:

```bash
python -m pytest tests/test_deployment_files.py tests/test_command_classifier.py tests/test_prototype_recognition.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add api/websocket.py backend/schemas.py tests/test_deployment_files.py
git commit -m "feat: save calibration clips as prototypes"
```

---

### Task 4: Browser Calibration UI

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/websocket.js`
- Modify: `frontend/styles.css`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes:
  - `ClipRecorder.recordClip(...)`
  - WebSocket `calibration.start`
- Produces:
  - Anonymous `profileId` in `localStorage`
  - UI controls for intent/language/phrase/scope
  - `calibration.saved` display

- [ ] **Step 1: Add static failing tests for UI markers**

```python
from pathlib import Path


def test_frontend_has_prototype_calibration_ui():
    html = Path("frontend/index.html").read_text()
    js = Path("frontend/websocket.js").read_text()

    assert "calibration-intent" in html
    assert "calibration-phrase" in html
    assert "Save Sample" in html
    assert "silentVisionProfileId" in js
    assert "calibration.start" in js
```

- [ ] **Step 2: Run static test and confirm failure**

Run:

```bash
python -m pytest tests/test_deployment_files.py::test_frontend_has_prototype_calibration_ui -q
```

Expected: assertion failure.

- [ ] **Step 3: Add UI controls**

Add a compact calibration block:

```html
<section class="panel">
  <h2>Calibration</h2>
  <select id="calibration-intent">...</select>
  <select id="calibration-language">...</select>
  <input id="calibration-phrase" placeholder="e.g. 你好，请帮我打开灯" />
  <button id="save-sample">Save Sample</button>
  <p id="profile-id"></p>
  <pre id="calibration-result"></pre>
</section>
```

- [ ] **Step 4: Generate anonymous profile ID**

In `frontend/websocket.js`:

```javascript
const PROFILE_STORAGE_KEY = "silentVisionProfileId";

function getOrCreateProfileId() {
  const existing = localStorage.getItem(PROFILE_STORAGE_KEY);
  if (existing) return existing;
  const created = crypto.randomUUID();
  localStorage.setItem(PROFILE_STORAGE_KEY, created);
  return created;
}
```

- [ ] **Step 5: Implement Save Sample**

When clicked:

1. Ensure WebSocket is connected.
2. Send `calibration.start` with `profileId`, intent, language, phrase, scope `personal`.
3. Record clip using existing `ClipRecorder`.
4. Send clip as binary.
5. Render `calibration.saved` or `calibration.error`.

- [ ] **Step 6: Include `profileId` in normal inference metadata**

When sending inference clips, include `profileId` in the latest command metadata so backend prototype matching can prefer personal samples.

- [ ] **Step 7: Run JS checks and tests**

Run:

```bash
node --check frontend/websocket.js
node --check frontend/camera.js
python -m pytest tests/test_deployment_files.py -q
```

Expected: pass.

- [ ] **Step 8: Commit**

```bash
git add frontend/index.html frontend/websocket.js frontend/styles.css tests/test_deployment_files.py
git commit -m "feat: add browser prototype calibration UI"
```

---

### Task 5: Scripts and AMD Startup Defaults

**Files:**
- Create: `scripts/inspect_prototypes.py`
- Create: `scripts/build_global_prototypes.py`
- Modify: `scripts/amd_real_oneclick.sh`
- Modify: `scripts/start_real_rocm.sh`
- Modify: `README.md`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes:
  - Prototype directory layout from Task 1.
- Produces:
  - CLI inspection of profile sample counts.
  - Global prototype creation from existing personal samples.

- [ ] **Step 1: Add static failing test for scripts**

```python
from pathlib import Path


def test_prototype_scripts_and_startup_defaults_exist():
    assert Path("scripts/inspect_prototypes.py").exists()
    assert Path("scripts/build_global_prototypes.py").exists()

    oneclick = Path("scripts/amd_real_oneclick.sh").read_text()
    readme = Path("README.md").read_text()

    assert "COMMAND_BACKEND=prototype" in oneclick
    assert "profiles/global" in readme
    assert "Personal Profile" in readme
```

- [ ] **Step 2: Run static test and confirm failure**

Run:

```bash
python -m pytest tests/test_deployment_files.py::test_prototype_scripts_and_startup_defaults_exist -q
```

Expected: assertion failure.

- [ ] **Step 3: Implement `inspect_prototypes.py`**

CLI behavior:

```bash
python scripts/inspect_prototypes.py --root /workspace/persistent/silent-vision
```

Output:

```text
Profile: global
  LIGHT_ON: 8 samples
  LIGHT_OFF: 7 samples
Profile: <anonymous-id>
  LIGHT_ON: 5 samples
```

- [ ] **Step 4: Implement `build_global_prototypes.py`**

CLI behavior:

```bash
python scripts/build_global_prototypes.py \
  --root /workspace/persistent/silent-vision \
  --from-profile <profileId>
```

Behavior:

- Copy selected samples from `profiles/<profileId>/...` into `profiles/global/...`.
- Do not overwrite existing sample directories.
- Print copied sample count.

- [ ] **Step 5: Update startup scripts**

Set default command backend to prototype:

```bash
export RECOGNITION_MODE="${RECOGNITION_MODE:-command}"
export COMMAND_BACKEND="${COMMAND_BACKEND:-prototype}"
```

Keep classifier backend available by explicit override:

```bash
COMMAND_BACKEND=torch bash scripts/amd_real_oneclick.sh
```

- [ ] **Step 6: Update README AMD workflow**

Document:

1. Start with prototype backend.
2. Open browser.
3. Save 5–10 personal samples per command.
4. Test recognition.
5. Optionally promote personal samples to global:

```bash
python scripts/build_global_prototypes.py \
  --root /workspace/persistent/silent-vision \
  --from-profile <profileId>
```

- [ ] **Step 7: Run validation**

Run:

```bash
python -m compileall command api backend scripts -q
python scripts/inspect_prototypes.py --root /tmp/nonexistent-silent-vision
python -m pytest tests/test_deployment_files.py tests/test_prototype_recognition.py tests/test_command_classifier.py -q
```

Expected: compile passes; inspect script prints no profiles; tests pass.

- [ ] **Step 8: Commit**

```bash
git add scripts/inspect_prototypes.py scripts/build_global_prototypes.py scripts/amd_real_oneclick.sh scripts/start_real_rocm.sh README.md tests/test_deployment_files.py
git commit -m "feat: add prototype profile operations"
```

---

## Self-Review

- Spec coverage: Global profile, personal profile, anonymous ID, sample save, prototype matching, threshold/margin rejection, no-LLM rejection path, AMD startup, and debug persistence are covered.
- Placeholder scan: No TBD/TODO placeholders remain.
- Type consistency: `profileId`, `CommandDecision`, prototype threshold names, and backend name `prototype` are consistent across tasks.
- Scope check: The plan intentionally avoids training a universal multi-person classifier. It implements few-shot personal/global prototype recognition only.
