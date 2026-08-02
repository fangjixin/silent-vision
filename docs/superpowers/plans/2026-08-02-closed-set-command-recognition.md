# Closed-Set Command Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Silent Vision from open-vocabulary visual transcription into utterance-level closed-set visual command recognition with reject-safe execution.

**Architecture:** The browser records one 2-5 second video utterance at 25 FPS and uploads the binary clip over WebSocket. The backend decodes/resamples the clip, extracts stable mouth ROI frames, runs a command classifier that returns business intents, and only executes actions for accepted intents.

**Tech Stack:** FastAPI WebSocket, MediaRecorder/WebCodecs-compatible browser recording, PyAV/FFmpeg, MediaPipe, PyTorch ROCm, frozen visual encoder adapter, 4-layer Conformer backend, attentive pooling, classifier head.

## Global Constraints

- The main runtime path must classify business intents such as `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, and `UNKNOWN`.
- The LLM must not be called when the command classifier rejects an utterance.
- Tools/actions must not execute when confidence thresholding or top-1/top-2 margin rejection fails.
- Every debug request must save original video, mouth ROI video, logits, and inference metadata.
- Training, validation, and inference scripts must work with `/opt/venv/bin/python` and AMD ROCm PyTorch.
- Open-vocabulary Auto-AVSR/CMLR remains fallback/debug only, not authoritative command execution.

---

### Task 1: Utterance-Level Browser Capture

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/camera.js`
- Modify: `frontend/websocket.js`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Produces: `ClipRecorder.startPreview()`, `ClipRecorder.recordClip({durationMs}) -> Promise<Blob>`.
- Produces: WebSocket text command `clip.start`, binary video blob, and `clip.cancel`.

- [ ] Write failing static tests that require MediaRecorder/WebCodecs-style clip capture and no 10 FPS Base64 JPEG streaming.
- [ ] Implement `ClipRecorder` using `navigator.mediaDevices.getUserMedia` and `MediaRecorder`.
- [ ] Update the Start button flow to connect WebSocket, preview, countdown, record one clip, then send binary video data.
- [ ] Update UI copy from `Lip/MiniCPM` to `Command/Decision`.
- [ ] Run `node --check frontend/camera.js frontend/websocket.js`.

### Task 2: Video Clip Decode and Stable Mouth ROI

**Files:**
- Create: `video/__init__.py`
- Create: `video/clip.py`
- Create: `video/mouth_roi.py`
- Modify: `api/websocket.py`
- Modify: `requirements.txt`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Produces: `decode_video_clip(data: bytes, target_fps: int) -> DecodedClip`.
- Produces: `extract_mouth_roi_clip(frames: Sequence[np.ndarray], face_detector, mouth_size: int) -> MouthRoiClip`.

- [ ] Write failing tests/static checks for PyAV dependency, exact 25 FPS resampling, and debug video artifact names.
- [ ] Implement lazy PyAV import so local syntax checks work without PyAV installed.
- [ ] Decode uploaded clips to RGB frames resampled to `settings.command_clip_fps`.
- [ ] Smooth normalized mouth boxes over time using an exponential moving average.
- [ ] Save original upload and mouth ROI debug video when `DEBUG_DUMP_WINDOWS=true`.

### Task 3: Closed-Set Command Classifier

**Files:**
- Create: `command/__init__.py`
- Create: `command/labels.py`
- Create: `command/model.py`
- Create: `command/inference.py`
- Modify: `backend/schemas.py`
- Modify: `backend/config.py`
- Modify: `backend/main.py`
- Test: `tests/test_command_classifier.py`

**Interfaces:**
- Produces: `CommandIntent` labels.
- Produces: `CommandDecision(intent, accepted, confidence, margin, logits, reason)`.
- Produces: `CommandClassifierBackend.predict(mouth_frames: np.ndarray, metadata: dict) -> CommandDecision`.

- [ ] Write failing tests for threshold acceptance, margin rejection, and `UNKNOWN` handling.
- [ ] Implement label map with `LIGHT_ON`, `LIGHT_OFF`, `OPEN_DOOR`, `CHAT_OTHER`, `UNKNOWN`.
- [ ] Implement Conformer block, 4-layer temporal backend, attentive pooling, and classifier head.
- [ ] Implement fake backend for local tests and checkpoint backend for ROCm.
- [ ] Save logits and metadata JSON for debug requests.

### Task 4: Safe Routing Without LLM

**Files:**
- Modify: `api/websocket.py`
- Modify: `agent/agent.py`
- Modify: `frontend/websocket.js`
- Test: `tests/test_websocket_flow.py`

**Interfaces:**
- Consumes: `CommandDecision`.
- Produces: WebSocket events `command.result` and `agent.result`.

- [ ] Write failing tests verifying rejected clips do not call semantic interpreter or action policy.
- [ ] Route accepted executable intents directly to `AgentPolicy.decide_command`.
- [ ] Route `CHAT_OTHER` and rejected commands to non-executing responses.
- [ ] Keep open-vocabulary lipreading as optional fallback/debug only.

### Task 5: Dataset and ROCm Training Scripts

**Files:**
- Create: `scripts/record_command_manifest.py`
- Create: `scripts/train_command_classifier.py`
- Create: `scripts/validate_command_classifier.py`
- Create: `scripts/infer_command_clip.py`
- Modify: `scripts/amd_real_oneclick.sh`
- Modify: `README.md`
- Test: `tests/test_deployment_files.py`

**Interfaces:**
- Produces: dataset manifest JSONL with `intent`, `variant`, `original_video`, `mouth_roi_video`, and `metadata`.
- Produces: classifier checkpoint compatible with `CommandClassifierBackend`.

- [ ] Write failing static tests requiring training/validation/inference scripts.
- [ ] Add manifest generator for the starter utterances in Chinese and English.
- [ ] Add ROCm training script that freezes the visual encoder by default and trains the temporal backend/head.
- [ ] Add validation script reporting accuracy, rejection rate, confusion matrix, and threshold recommendations.
- [ ] Add inference script for one clip with logits and metadata output.
