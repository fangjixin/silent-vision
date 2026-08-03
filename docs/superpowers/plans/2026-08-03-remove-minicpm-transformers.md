# Remove MiniCPM and Transformers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Silent Vision runtime into a command-recognition-only app with no MiniCPM, Hugging Face Transformers, or MiniCPM-o model dependency.

**Architecture:** The app keeps browser clip capture, PyAV/MediaPipe mouth ROI extraction, prototype matching, and optional PyTorch ROCm command classifier loading. It removes open-vocabulary transcription and LLM semantic interpretation from the active runtime.

**Tech Stack:** FastAPI, Uvicorn, Pydantic, NumPy, OpenCV, MediaPipe, PyAV, optional PyTorch ROCm supplied by the AMD base image.

## Global Constraints

- Do not require `/workspace/persistent/silent-vision/models/minicpm-o-4_5`.
- Do not install or import `transformers`, `huggingface-hub`, `accelerate`, `safetensors`, `librosa`, `soundfile`, or `minicpmo-utils`.
- Keep `COMMAND_BACKEND=prototype` as the AMD script default.
- Keep `COMMAND_BACKEND=torch` able to load a local PyTorch checkpoint via `torch.load`.
- Rejected commands must not call an LLM or execute tools.

---

### Task 1: Remove Runtime MiniCPM Wiring

**Files:**
- Modify: `backend/config.py`
- Modify: `backend/main.py`
- Modify: `api/websocket.py`
- Delete: `llm/minicpm.py`

**Interfaces:**
- Consumes: `build_command_classifier(settings)`
- Produces: command-only FastAPI app state with `command_classifier`, `agent_policy`, and `face_detector`.

- [ ] Remove `recognition_mode`, `command_fallback_transcription`, and `minicpm_model_path`.
- [ ] Remove `build_minicpm_interpreter` import and `semantic_interpreter` initialization.
- [ ] Remove the legacy JPEG transcription branch that calls MiniCPM.
- [ ] Delete `llm/minicpm.py`.

### Task 2: Remove Dependency and Script Requirements

**Files:**
- Modify: `requirements.txt`
- Modify: `scripts/setup_amd_real.sh`
- Modify: `scripts/start_real_rocm.sh`
- Modify: `scripts/amd_real_oneclick.sh`
- Modify: `scripts/smoke_rocm.sh`
- Modify: `docker/Dockerfile`

**Interfaces:**
- Produces: setup/start scripts that validate ROCm Python, app dependencies, prototype storage, and optional torch checkpoint only.

- [ ] Remove Hugging Face and MiniCPM dependency installation/checks.
- [ ] Remove MiniCPM model directory creation/download.
- [ ] Keep zip reuse checks for mpc001 assets only if old files still reference mpc001.
- [ ] Start app with command/prototype defaults.

### Task 3: Update Tests and Docs

**Files:**
- Delete: `tests/test_minicpm_agent.py`
- Modify: `tests/test_rocm_models.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_deployment_files.py`
- Modify: `README.md`
- Modify: `docs/runbooks/amd-real-mode.md`

**Interfaces:**
- Produces: tests and documentation that assert MiniCPM is absent from active runtime.

- [ ] Remove MiniCPM-specific tests.
- [ ] Update dependency assertions.
- [ ] Document prototype and torch checkpoint loading.
- [ ] Document that open-vocabulary transcription is no longer the supported path.

### Task 4: Verify

**Files:**
- No production edits.

- [ ] Run `python3 -m compileall agent api backend command lip session video vision scripts -q`.
- [ ] Run `node --check frontend/camera.js`.
- [ ] Run `node --check frontend/websocket.js`.
- [ ] Run targeted pytest suites covering config, deployment, command classifier, and prototype recognition.
- [ ] Run `git diff --check`.
