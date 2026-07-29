# Realtime Bilingual Lipreading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Silent Vision, a single-active-session FastAPI system where a local browser streams camera JPEG frames over WebSocket to an AMD ROCm server, which crops mouth frames, runs English AV-HuBERT and Chinese CMLR VSR candidates, asks MiniCPM-o 4.5 for semantic selection, and returns structured agent results to the browser.

**Architecture:** Use a modular monolith: one FastAPI process serves the frontend, manages anonymous sessions, runs MediaPipe vision, and serializes GPU inference behind a single global inference lock. All model calls sit behind stable Python adapters so default tests use fake backends and ROCm/model tests are explicitly marked. The browser owns camera capture only; all AI execution is server-side.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Pydantic v2, NumPy, OpenCV, MediaPipe, PyTorch ROCm, AV-HuBERT, CMLR VSR, MiniCPM-o 4.5, vanilla HTML/CSS/JavaScript, pytest, pytest-asyncio, httpx, websockets, Playwright.

## Global Constraints

- Development date: 2026-07-29.
- Persistence root: `/workspace/persistence/silent-vision`.
- Model directories: `/workspace/persistence/silent-vision/models/avhubert`, `/workspace/persistence/silent-vision/models/cmlr`, `/workspace/persistence/silent-vision/models/minicpm-o-4_5`.
- Browser camera runs on the user's local machine; ROCm inference runs on the remote AMD Radeon 7900 server.
- Development access uses SSH forwarding: `ssh -L 8000:127.0.0.1:8000 user@rocm-server`, then open `http://localhost:8000`.
- Server binds to `127.0.0.1:8000` for development.
- No user login, user accounts, audio capture, training, fine-tuning, external side effects, model weight commits, persistent raw frames, persistent mouth frames, persistent session state, or persistent full transcripts.
- Exactly one active streaming WebSocket session is allowed in MVP; the second active streaming connection receives `SERVER_BUSY` and is closed.
- JPEG frames are binary WebSocket messages; JSON messages are used only for control and server events.
- Target capture FPS is `25`; window size is `75` valid mouth frames; inference stride is `25` valid frames; mouth crop size is `96x96`.
- English lip reading uses AV-HuBERT video-only; Chinese lip reading uses CMLR visual-only from `mpc001/Visual_Speech_Recognition_for_Multiple_Languages`.
- First version does not claim single-window Chinese-English code-switching support.
- MiniCPM-o 4.5 chooses `zh`, `en`, or `unknown`; it must not infer language from appearance, identity, face shape, skin color, or nationality.
- Agent actions are schema-limited to `respond`, `confirm`, and `unknown`; Agent never executes files, OS commands, network calls, device controls, or external APIs.
- Default tests must run without ROCm, without model weights, and without downloading models by using fake adapters.
- Real model tests are marked `rocm` and/or `model_integration` and require local authorized test media.
- Use `MODEL_BACKEND=fake` for default development and tests; use `MODEL_BACKEND=real` only on the ROCm deployment server.
- Use `/tmp/silent-vision/{session_id}` only for temporary media and remove it on window completion, disconnect, and shutdown.
- CMLR VSR upstream license is research/comparison/benchmark oriented; production or commercial use requires separate authorization review.

---

## File Map

- `backend/__init__.py`: package marker.
- `backend/config.py`: environment settings, derived paths, constants, validation.
- `backend/schemas.py`: all Pydantic message, candidate, semantic, agent, metrics, and health schemas.
- `backend/main.py`: FastAPI app factory, lifespan model container, static frontend mounting, health routes.
- `api/__init__.py`: package marker.
- `api/session.py`: `POST /api/sessions`.
- `api/websocket.py`: `/ws/{session_id}` WebSocket protocol and pipeline orchestration.
- `session/__init__.py`: package marker.
- `session/manager.py`: anonymous session lifecycle, single-active-stream guard, frame buffers, inference scheduling state.
- `vision/__init__.py`: package marker.
- `vision/face.py`: MediaPipe face/mouth landmark adapter and JPEG decode validation.
- `vision/mouth.py`: mouth bounding box, crop, grayscale, resize, and frame normalization.
- `lip/__init__.py`: package marker.
- `lip/base.py`: common lip reader protocol and candidate/window dataclasses.
- `lip/fake.py`: deterministic fake English and Chinese lip readers for default tests.
- `lip/avhubert.py`: real AV-HuBERT video-only adapter.
- `lip/cmlr.py`: real CMLR visual-only adapter.
- `lip/inference.py`: dual-engine candidate orchestrator with degradation handling.
- `llm/__init__.py`: package marker.
- `llm/minicpm.py`: fake and real MiniCPM semantic interpreters with strict JSON schema validation.
- `agent/__init__.py`: package marker.
- `agent/agent.py`: pure schema-limited agent policy.
- `frontend/index.html`: camera preview, overlay, controls, phase status, candidates, final result, metrics.
- `frontend/camera.js`: local camera access, canvas JPEG encoding, target FPS loop.
- `frontend/websocket.js`: session creation, WebSocket connection, binary frame sending, event rendering, heartbeat, reconnect.
- `frontend/styles.css`: responsive two-pane runtime UI.
- `docker/Dockerfile`: ROCm-capable runtime image that installs app dependencies but not weights.
- `docker/docker-compose.yml`: bind mounts `/workspace/persistence/silent-vision`, exposes loopback service, passes ROCm devices.
- `requirements.txt`: runtime Python dependencies excluding ROCm PyTorch wheels when using a ROCm PyTorch base image.
- `requirements-dev.txt`: pytest, Playwright, static checks.
- `pytest.ini`: markers and asyncio mode.
- `.env.example`: exact environment keys and fake/real backend defaults.
- `.gitignore`: caches, models, logs, temp media, local manifests.
- `tests/conftest.py`: fake settings, app fixtures, sample JPEG helpers.
- `tests/test_config.py`: settings and path validation.
- `tests/test_session_manager.py`: session lifecycle and buffering.
- `tests/test_vision_mouth.py`: JPEG decode, landmarks, mouth crop.
- `tests/test_lip_inference.py`: fake lip readers and degraded model behavior.
- `tests/test_minicpm_agent.py`: semantic parser and agent policy.
- `tests/test_websocket_flow.py`: full fake FastAPI WebSocket pipeline.
- `tests/test_rocm_models.py`: marked hardware/model smoke tests.
- `tests/e2e/camera.spec.js`: browser E2E with fake backend and virtual media.
- `tests/media/manifest.example.json`: local test media manifest schema example.
- `README.md`: runbook for fake mode, SSH tunnel, ROCm mode, model placement, and verification commands.

---

## Shared Interfaces

Use these names and field shapes consistently across all tasks.

```python
# backend/schemas.py
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

Language = Literal["zh", "en", "unknown"]
AgentAction = Literal["respond", "confirm", "unknown"]

class ErrorCode(str, Enum):
    INVALID_SESSION = "INVALID_SESSION"
    SERVER_BUSY = "SERVER_BUSY"
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"
    INVALID_JPEG = "INVALID_JPEG"
    FRAME_TOO_LARGE_DIMENSIONS = "FRAME_TOO_LARGE_DIMENSIONS"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INVALID_MOUTH_BOX = "INVALID_MOUTH_BOX"
    LIP_MODELS_FAILED = "LIP_MODELS_FAILED"
    MINICPM_FAILED = "MINICPM_FAILED"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"
    INTERNAL_ERROR = "INTERNAL_ERROR"

class MouthBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

class LipReadingCandidate(BaseModel):
    model: Literal["avhubert", "cmlr"]
    language: Literal["zh", "en"]
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rawScore: float | None = None
    latencyMs: int = Field(ge=0)

class SemanticResult(BaseModel):
    language: Language
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str

class AgentResult(BaseModel):
    type: Literal["agent.result"] = "agent.result"
    action: AgentAction
    language: Language
    text: str
    arguments: dict[str, Any]
    requiresConfirmation: bool
```

```python
# lip/base.py
from dataclasses import dataclass
from typing import Protocol
from collections.abc import Sequence
import numpy as np
from backend.schemas import LipReadingCandidate

@dataclass(frozen=True)
class MouthFrame:
    sequence: int
    received_at_ms: int
    image: np.ndarray  # uint8 grayscale, shape (96, 96)

@dataclass(frozen=True)
class MouthWindow:
    session_id: str
    start_sequence: int
    end_sequence: int
    frames: Sequence[MouthFrame]

class LipReader(Protocol):
    name: str
    language: str
    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        raise NotImplementedError
```

```python
# llm/minicpm.py
class MiniCPMInterpreter:
    def interpret(
        self,
        candidates: list[LipReadingCandidate],
        sampled_frames: list[np.ndarray],
        stats: dict[str, float | int | str],
    ) -> SemanticResult:
        raise NotImplementedError
```

---

### Task 1: Project Foundation, Config, Schemas, and Fake Health

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/config.py`
- Create: `backend/schemas.py`
- Create: `backend/main.py`
- Create: `api/__init__.py`
- Create: `session/__init__.py`
- Create: `vision/__init__.py`
- Create: `lip/__init__.py`
- Create: `llm/__init__.py`
- Create: `agent/__init__.py`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`
- Create: `README.md`

**Interfaces:**
- Consumes: No project code.
- Produces: `Settings`, `get_settings()`, `create_app(settings: Settings | None = None) -> FastAPI`, Pydantic schemas from Shared Interfaces.

- [ ] **Step 1: Write failing config and health tests**

```python
# tests/test_config.py
from pathlib import Path

from backend.config import Settings
from backend.main import create_app


def test_settings_defaults_use_fake_backend_and_persistence_root():
    settings = Settings()
    assert settings.model_backend == "fake"
    assert settings.persistence_root == Path("/workspace/persistence/silent-vision")
    assert settings.window_frames == 75
    assert settings.inference_stride == 25
    assert settings.mouth_size == 96
    assert settings.capture_fps == 25


def test_settings_model_paths_are_derived_from_persistence_root():
    settings = Settings()
    assert settings.avhubert_checkpoint == Path("/workspace/persistence/silent-vision/models/avhubert/model.pt")
    assert settings.cmlr_checkpoint == Path("/workspace/persistence/silent-vision/models/cmlr/model.pth")
    assert settings.cmlr_language_model == Path("/workspace/persistence/silent-vision/models/cmlr/language-model.pth")
    assert settings.minicpm_model_path == Path("/workspace/persistence/silent-vision/models/minicpm-o-4_5")


def test_create_app_registers_health_routes():
    app = create_app(Settings())
    route_paths = {route.path for route in app.routes}
    assert "/health/live" in route_paths
    assert "/health/ready" in route_paths
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'backend.config'`.

- [ ] **Step 3: Create runtime and dev dependency files**

```text
# requirements.txt
fastapi>=0.115.0,<1.0.0
uvicorn[standard]>=0.30.0,<1.0.0
pydantic>=2.8.0,<3.0.0
pydantic-settings>=2.4.0,<3.0.0
numpy>=1.26.0,<3.0.0
opencv-python-headless>=4.10.0,<5.0.0
mediapipe>=0.10.14,<0.11.0
python-multipart>=0.0.9,<1.0.0
orjson>=3.10.0,<4.0.0
transformers>=4.44.0,<5.0.0
accelerate>=0.33.0,<2.0.0
safetensors>=0.4.4,<1.0.0
Pillow>=10.4.0,<12.0.0
```

```text
# requirements-dev.txt
-r requirements.txt
pytest>=8.3.0,<9.0.0
pytest-asyncio>=0.23.8,<1.0.0
httpx>=0.27.0,<1.0.0
websockets>=12.0,<16.0
ruff>=0.5.0,<1.0.0
playwright>=1.46.0,<2.0.0
```

- [ ] **Step 4: Implement settings, schemas, app factory, pytest config, env example, and ignore rules**

```python
# backend/config.py
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_backend: Literal["fake", "real"] = "fake"
    persistence_root: Path = Path("/workspace/persistence/silent-vision")
    capture_fps: int = Field(default=25, ge=1, le=60)
    window_frames: int = Field(default=75, ge=1)
    inference_stride: int = Field(default=25, ge=1)
    mouth_size: int = Field(default=96, ge=16)
    max_jpeg_bytes: int = Field(default=1_048_576, ge=1024)
    max_frame_width: int = Field(default=1920, ge=64)
    max_frame_height: int = Field(default=1080, ge=64)
    model_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    allowed_origins: str = "http://localhost:8000"
    log_transcripts: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    enable_model_health: bool = False

    @property
    def avhubert_checkpoint(self) -> Path:
        return self.persistence_root / "models" / "avhubert" / "model.pt"

    @property
    def cmlr_checkpoint(self) -> Path:
        return self.persistence_root / "models" / "cmlr" / "model.pth"

    @property
    def cmlr_language_model(self) -> Path:
        return self.persistence_root / "models" / "cmlr" / "language-model.pth"

    @property
    def minicpm_model_path(self) -> Path:
        return self.persistence_root / "models" / "minicpm-o-4_5"

    @property
    def allowed_origin_set(self) -> set[str]:
        return {item.strip() for item in self.allowed_origins.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

```python
# backend/schemas.py
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["zh", "en", "unknown"]
AgentAction = Literal["respond", "confirm", "unknown"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ErrorCode(str, Enum):
    INVALID_SESSION = "INVALID_SESSION"
    SERVER_BUSY = "SERVER_BUSY"
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"
    INVALID_JPEG = "INVALID_JPEG"
    FRAME_TOO_LARGE_DIMENSIONS = "FRAME_TOO_LARGE_DIMENSIONS"
    FACE_NOT_FOUND = "FACE_NOT_FOUND"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    INVALID_MOUTH_BOX = "INVALID_MOUTH_BOX"
    LIP_MODELS_FAILED = "LIP_MODELS_FAILED"
    MINICPM_FAILED = "MINICPM_FAILED"
    GPU_OUT_OF_MEMORY = "GPU_OUT_OF_MEMORY"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class BaseEvent(BaseModel):
    type: str
    sessionId: str
    timestamp: datetime = Field(default_factory=utc_now)


class MouthBox(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class LipReadingCandidate(BaseModel):
    model: Literal["avhubert", "cmlr"]
    language: Literal["zh", "en"]
    text: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rawScore: float | None = None
    latencyMs: int = Field(ge=0)


class SemanticResult(BaseModel):
    language: Language
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class AgentResult(BaseModel):
    type: Literal["agent.result"] = "agent.result"
    action: AgentAction
    language: Language
    text: str
    arguments: dict[str, Any]
    requiresConfirmation: bool


class ErrorEvent(BaseEvent):
    type: Literal["error"] = "error"
    stage: str
    code: ErrorCode
    message: str
    recoverable: bool
```

```python
# backend/main.py
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from backend.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(default_response_class=ORJSONResponse, title="Silent Vision")
    app.state.settings = app_settings
    app.state.models = {"backend": app_settings.model_backend, "ready": app_settings.model_backend == "fake"}

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        return {"status": "ready", "models": app.state.models}

    return app


app = create_app()
```

```ini
# pytest.ini
[pytest]
asyncio_mode = auto
markers =
    rocm: requires AMD ROCm runtime and GPU
    model_integration: requires local model weights and authorized test media
testpaths = tests
```

```text
# .env.example
MODEL_BACKEND=fake
PERSISTENCE_ROOT=/workspace/persistence/silent-vision
CAPTURE_FPS=25
WINDOW_FRAMES=75
INFERENCE_STRIDE=25
MOUTH_SIZE=96
MAX_JPEG_BYTES=1048576
MAX_FRAME_WIDTH=1920
MAX_FRAME_HEIGHT=1080
MODEL_CONFIDENCE_THRESHOLD=0.55
ALLOWED_ORIGINS=http://localhost:8000
LOG_TRANSCRIPTS=false
HOST=127.0.0.1
PORT=8000
ENABLE_MODEL_HEALTH=false
```

```text
# .gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
node_modules/
playwright-report/
test-results/
models/
logs/
cache/
reports/
tests/media/*.mp4
tests/media/*.mov
tests/media/*.avi
tests/media/manifest.local.json
```

```python
# tests/conftest.py
import io

import cv2
import numpy as np
import pytest

from backend.config import Settings
from backend.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(model_backend="fake")


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


def make_jpeg(width: int = 320, height: int = 240) -> bytes:
    image = np.full((height, width, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return io.BytesIO(encoded).getvalue()
```

```markdown
# README.md
# Silent Vision

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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend api session vision lip llm agent tests requirements.txt requirements-dev.txt pytest.ini .env.example .gitignore README.md
git commit -m "chore: add project foundation and schemas"
```

---

### Task 2: Anonymous Session Manager and Session API

**Files:**
- Create: `session/manager.py`
- Create: `api/session.py`
- Modify: `backend/main.py`
- Create: `tests/test_session_manager.py`

**Interfaces:**
- Consumes: `Settings`, `ErrorCode`.
- Produces: `SessionManager.create_pending_session() -> CreatedSession`, `SessionManager.activate(session_id: str) -> ActiveSession`, `SessionManager.disconnect(session_id: str) -> None`, `POST /api/sessions`.

- [ ] **Step 1: Write failing session manager and API tests**

```python
# tests/test_session_manager.py
from datetime import timedelta
from uuid import UUID

from fastapi.testclient import TestClient

from backend.main import create_app
from session.manager import SessionManager, ServerBusyError


def test_create_pending_session_returns_secure_uuid():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    UUID(created.session_id)
    assert created.expires_in_seconds == 30


def test_only_one_active_streaming_session_is_allowed():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    first = manager.create_pending_session()
    second = manager.create_pending_session()
    manager.activate(first.session_id)
    try:
        manager.activate(second.session_id)
    except ServerBusyError as exc:
        assert exc.code == "SERVER_BUSY"
    else:
        raise AssertionError("second active session should be rejected")


def test_disconnect_releases_active_slot():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    first = manager.create_pending_session()
    second = manager.create_pending_session()
    manager.activate(first.session_id)
    manager.disconnect(first.session_id)
    active = manager.activate(second.session_id)
    assert active.session_id == second.session_id


def test_post_sessions_returns_session_id(settings):
    app = create_app(settings)
    client = TestClient(app)
    response = client.post("/api/sessions")
    assert response.status_code == 200
    body = response.json()
    UUID(body["sessionId"])
    assert body["expiresInSeconds"] == 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_manager.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'session.manager'`.

- [ ] **Step 3: Implement session manager and API route**

```python
# session/manager.py
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from lip.base import MouthFrame, MouthWindow


class SessionError(Exception):
    code = "INVALID_SESSION"


class ServerBusyError(SessionError):
    code = "SERVER_BUSY"


@dataclass(frozen=True)
class CreatedSession:
    session_id: str
    expires_in_seconds: int


@dataclass
class ActiveSession:
    session_id: str
    window_frames: int
    inference_stride: int
    created_at: datetime
    last_seen_at: datetime
    frames: deque[MouthFrame] = field(default_factory=deque)
    accepted_frame_count: int = 0
    last_inference_frame_count: int = 0
    active_inference_task: object | None = None
    latest_pending_window: MouthWindow | None = None
    streaming: bool = False

    def reset_stream(self) -> None:
        self.frames.clear()
        self.accepted_frame_count = 0
        self.last_inference_frame_count = 0
        self.latest_pending_window = None
        self.streaming = True

    def add_mouth_frame(self, frame: MouthFrame) -> MouthWindow | None:
        if len(self.frames) == self.window_frames:
            self.frames.popleft()
        self.frames.append(frame)
        self.accepted_frame_count += 1
        if len(self.frames) < self.window_frames:
            return None
        if self.accepted_frame_count - self.last_inference_frame_count < self.inference_stride:
            return None
        self.last_inference_frame_count = self.accepted_frame_count
        snapshot = tuple(self.frames)
        return MouthWindow(
            session_id=self.session_id,
            start_sequence=snapshot[0].sequence,
            end_sequence=snapshot[-1].sequence,
            frames=snapshot,
        )


class SessionManager:
    def __init__(self, pending_ttl: timedelta, window_frames: int, inference_stride: int) -> None:
        self.pending_ttl = pending_ttl
        self.window_frames = window_frames
        self.inference_stride = inference_stride
        self._pending: dict[str, datetime] = {}
        self._active: dict[str, ActiveSession] = {}

    def create_pending_session(self) -> CreatedSession:
        session_id = str(uuid4())
        self._pending[session_id] = datetime.now(timezone.utc) + self.pending_ttl
        return CreatedSession(session_id=session_id, expires_in_seconds=int(self.pending_ttl.total_seconds()))

    def activate(self, session_id: str) -> ActiveSession:
        self._remove_expired_pending()
        expires_at = self._pending.pop(session_id, None)
        if expires_at is None:
            raise SessionError("invalid or expired session")
        if self._active:
            raise ServerBusyError("server already has an active streaming session")
        now = datetime.now(timezone.utc)
        active = ActiveSession(
            session_id=session_id,
            window_frames=self.window_frames,
            inference_stride=self.inference_stride,
            created_at=now,
            last_seen_at=now,
        )
        self._active[session_id] = active
        return active

    def get_active(self, session_id: str) -> ActiveSession:
        active = self._active.get(session_id)
        if active is None:
            raise SessionError("invalid active session")
        active.last_seen_at = datetime.now(timezone.utc)
        return active

    def disconnect(self, session_id: str) -> None:
        self._active.pop(session_id, None)
        self._pending.pop(session_id, None)

    def _remove_expired_pending(self) -> None:
        now = datetime.now(timezone.utc)
        expired = [session_id for session_id, expires_at in self._pending.items() if expires_at <= now]
        for session_id in expired:
            self._pending.pop(session_id, None)
```

```python
# api/session.py
from fastapi import APIRouter, Request

router = APIRouter()


@router.post("/api/sessions")
async def create_session(request: Request) -> dict[str, object]:
    created = request.app.state.session_manager.create_pending_session()
    return {"sessionId": created.session_id, "expiresInSeconds": created.expires_in_seconds}
```

```python
# backend/main.py additions
from datetime import timedelta
from api.session import router as session_router
from session.manager import SessionManager

# inside create_app after app creation:
app.state.session_manager = SessionManager(
    pending_ttl=timedelta(seconds=30),
    window_frames=app_settings.window_frames,
    inference_stride=app_settings.inference_stride,
)
app.include_router(session_router)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_manager.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add session/manager.py api/session.py backend/main.py tests/test_session_manager.py
git commit -m "feat: add anonymous session lifecycle"
```

---

### Task 3: Mouth Frame Types and Buffer Scheduling

**Files:**
- Create: `lip/base.py`
- Modify: `session/manager.py`
- Modify: `tests/test_session_manager.py`

**Interfaces:**
- Consumes: `ActiveSession.add_mouth_frame(frame: MouthFrame)`.
- Produces: `MouthFrame`, `MouthWindow`, stable 75-frame snapshot and 25-frame stride behavior.

- [ ] **Step 1: Add failing buffer stride tests**

```python
# append to tests/test_session_manager.py
import numpy as np

from lip.base import MouthFrame


def _frame(sequence: int) -> MouthFrame:
    return MouthFrame(sequence=sequence, received_at_ms=sequence * 40, image=np.zeros((96, 96), dtype=np.uint8))


def test_first_window_appears_at_75_valid_frames():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    active = manager.activate(created.session_id)
    windows = [active.add_mouth_frame(_frame(sequence)) for sequence in range(1, 76)]
    assert all(window is None for window in windows[:-1])
    assert windows[-1] is not None
    assert windows[-1].start_sequence == 1
    assert windows[-1].end_sequence == 75


def test_next_window_appears_after_25_more_valid_frames():
    manager = SessionManager(pending_ttl=timedelta(seconds=30), window_frames=75, inference_stride=25)
    created = manager.create_pending_session()
    active = manager.activate(created.session_id)
    last = None
    for sequence in range(1, 101):
        last = active.add_mouth_frame(_frame(sequence))
    assert last is not None
    assert last.start_sequence == 26
    assert last.end_sequence == 100
    assert len(last.frames) == 75
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_session_manager.py::test_first_window_appears_at_75_valid_frames tests/test_session_manager.py::test_next_window_appears_after_25_more_valid_frames -v`

Expected: FAIL if `lip.base` or frame scheduling is missing.

- [ ] **Step 3: Implement mouth frame dataclasses**

```python
# lip/base.py
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from backend.schemas import LipReadingCandidate


@dataclass(frozen=True)
class MouthFrame:
    sequence: int
    received_at_ms: int
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.image.shape != (96, 96):
            raise ValueError("mouth frame image must have shape (96, 96)")
        if self.image.dtype != np.uint8:
            raise ValueError("mouth frame image must use uint8")


@dataclass(frozen=True)
class MouthWindow:
    session_id: str
    start_sequence: int
    end_sequence: int
    frames: Sequence[MouthFrame]


class LipReader(Protocol):
    name: str
    language: str

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_session_manager.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lip/base.py session/manager.py tests/test_session_manager.py
git commit -m "feat: add mouth window scheduling"
```

---

### Task 4: JPEG Decode, Face Detection Adapter, and Mouth Crop

**Files:**
- Create: `vision/face.py`
- Create: `vision/mouth.py`
- Create: `tests/test_vision_mouth.py`

**Interfaces:**
- Consumes: `Settings.max_jpeg_bytes`, `Settings.max_frame_width`, `Settings.max_frame_height`, `Settings.mouth_size`.
- Produces: `decode_jpeg_frame(data: bytes, settings: Settings) -> np.ndarray`, `FaceDetector.detect(image_bgr: np.ndarray) -> FaceDetectionResult`, `crop_mouth(image_bgr: np.ndarray, landmarks: list[tuple[float, float]], mouth_size: int) -> MouthCropResult`.

- [ ] **Step 1: Write failing vision tests**

```python
# tests/test_vision_mouth.py
import numpy as np
import pytest

from backend.config import Settings
from backend.schemas import ErrorCode
from tests.conftest import make_jpeg
from vision.face import FrameDecodeError, decode_jpeg_frame
from vision.mouth import crop_mouth


def test_decode_jpeg_frame_returns_bgr_image():
    image = decode_jpeg_frame(make_jpeg(width=320, height=240), Settings())
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8


def test_decode_rejects_oversize_payload():
    settings = Settings(max_jpeg_bytes=10)
    with pytest.raises(FrameDecodeError) as exc_info:
        decode_jpeg_frame(make_jpeg(), settings)
    assert exc_info.value.code == ErrorCode.FRAME_TOO_LARGE


def test_crop_mouth_returns_normalized_box_and_96_image():
    image = np.full((200, 300, 3), 128, dtype=np.uint8)
    landmarks = [(0.45, 0.55), (0.55, 0.55), (0.50, 0.62), (0.50, 0.50)]
    result = crop_mouth(image, landmarks, mouth_size=96)
    assert result.image.shape == (96, 96)
    assert result.image.dtype == np.uint8
    assert 0.0 <= result.box.x <= 1.0
    assert 0.0 <= result.box.y <= 1.0
    assert result.box.width > 0
    assert result.box.height > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vision_mouth.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'vision.face'`.

- [ ] **Step 3: Implement decode, MediaPipe adapter shell, and crop logic**

```python
# vision/face.py
from dataclasses import dataclass

import cv2
import numpy as np

from backend.config import Settings
from backend.schemas import ErrorCode

MOUTH_LANDMARK_IDS = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
)


class FrameDecodeError(ValueError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FaceDetectionResult:
    face_detected: bool
    landmarks: list[tuple[float, float]]
    face_count: int


def decode_jpeg_frame(data: bytes, settings: Settings) -> np.ndarray:
    if len(data) > settings.max_jpeg_bytes:
        raise FrameDecodeError(ErrorCode.FRAME_TOO_LARGE, "jpeg payload exceeds MAX_JPEG_BYTES")
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise FrameDecodeError(ErrorCode.INVALID_JPEG, "payload is not a valid jpeg")
    height, width = image.shape[:2]
    if width > settings.max_frame_width or height > settings.max_frame_height:
        raise FrameDecodeError(ErrorCode.FRAME_TOO_LARGE_DIMENSIONS, "decoded frame dimensions exceed configured limits")
    return image


class FaceDetector:
    def __init__(self) -> None:
        import mediapipe as mp

        self._mp_face_mesh = mp.solutions.face_mesh
        self._mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self._mesh.process(image_rgb)
        faces = result.multi_face_landmarks or []
        if len(faces) != 1:
            return FaceDetectionResult(face_detected=False, landmarks=[], face_count=len(faces))
        landmarks = []
        for index in MOUTH_LANDMARK_IDS:
            point = faces[0].landmark[index]
            landmarks.append((float(point.x), float(point.y)))
        return FaceDetectionResult(face_detected=True, landmarks=landmarks, face_count=1)
```

```python
# vision/mouth.py
from dataclasses import dataclass

import cv2
import numpy as np

from backend.schemas import MouthBox


@dataclass(frozen=True)
class MouthCropResult:
    image: np.ndarray
    box: MouthBox


def crop_mouth(image_bgr: np.ndarray, landmarks: list[tuple[float, float]], mouth_size: int) -> MouthCropResult:
    if not landmarks:
        raise ValueError("mouth landmarks are required")
    height, width = image_bgr.shape[:2]
    xs = [x for x, _ in landmarks]
    ys = [y for _, y in landmarks]
    min_x = max(0.0, min(xs))
    max_x = min(1.0, max(xs))
    min_y = max(0.0, min(ys))
    max_y = min(1.0, max(ys))
    box_width = max_x - min_x
    box_height = max_y - min_y
    margin_x = box_width * 0.65
    margin_y = box_height * 0.85
    min_x = max(0.0, min_x - margin_x)
    max_x = min(1.0, max_x + margin_x)
    min_y = max(0.0, min_y - margin_y)
    max_y = min(1.0, max_y + margin_y)
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("invalid mouth box")
    left = int(round(min_x * width))
    right = int(round(max_x * width))
    top = int(round(min_y * height))
    bottom = int(round(max_y * height))
    crop = image_bgr[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("empty mouth crop")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (mouth_size, mouth_size), interpolation=cv2.INTER_AREA)
    return MouthCropResult(
        image=resized.astype(np.uint8, copy=False),
        box=MouthBox(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vision_mouth.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vision/face.py vision/mouth.py tests/test_vision_mouth.py
git commit -m "feat: add jpeg decode and mouth crop"
```

---

### Task 5: Fake Lip Readers and Dual-Engine Orchestrator

**Files:**
- Create: `lip/fake.py`
- Create: `lip/inference.py`
- Create: `tests/test_lip_inference.py`

**Interfaces:**
- Consumes: `LipReader.predict(window: MouthWindow) -> LipReadingCandidate`.
- Produces: `LipInferenceEngine.predict(window: MouthWindow) -> LipInferenceResult`.

- [ ] **Step 1: Write failing lip inference tests**

```python
# tests/test_lip_inference.py
import numpy as np

from lip.base import MouthFrame, MouthWindow
from lip.fake import FakeLipReader
from lip.inference import LipInferenceEngine


def _window() -> MouthWindow:
    frames = tuple(
        MouthFrame(sequence=i, received_at_ms=i * 40, image=np.zeros((96, 96), dtype=np.uint8))
        for i in range(1, 76)
    )
    return MouthWindow(session_id="s1", start_sequence=1, end_sequence=75, frames=frames)


def test_dual_engine_returns_english_and_chinese_candidates():
    engine = LipInferenceEngine([
        FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
        FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
    ])
    result = engine.predict(_window())
    assert [candidate.model for candidate in result.candidates] == ["avhubert", "cmlr"]
    assert result.degradedModels == []


def test_single_reader_failure_is_degraded_not_fatal():
    engine = LipInferenceEngine([
        FakeLipReader(model="avhubert", language="en", text="ignored", confidence=0.1, fail=True),
        FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
    ])
    result = engine.predict(_window())
    assert [candidate.model for candidate in result.candidates] == ["cmlr"]
    assert result.degradedModels == ["avhubert"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lip_inference.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'lip.fake'`.

- [ ] **Step 3: Implement fake reader and inference orchestrator**

```python
# lip/fake.py
from time import perf_counter
from typing import Literal

from backend.schemas import LipReadingCandidate
from lip.base import MouthWindow


class FakeLipReader:
    def __init__(
        self,
        model: Literal["avhubert", "cmlr"],
        language: Literal["en", "zh"],
        text: str,
        confidence: float,
        fail: bool = False,
    ) -> None:
        self.name = model
        self.language = language
        self._text = text
        self._confidence = confidence
        self._fail = fail

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        started = perf_counter()
        if self._fail:
            raise RuntimeError(f"{self.name} fake failure")
        return LipReadingCandidate(
            model=self.name,
            language=self.language,
            text=self._text,
            confidence=self._confidence,
            rawScore=self._confidence,
            latencyMs=max(0, int((perf_counter() - started) * 1000)),
        )
```

```python
# lip/inference.py
from dataclasses import dataclass

from backend.schemas import LipReadingCandidate
from lip.base import LipReader, MouthWindow


@dataclass(frozen=True)
class LipInferenceResult:
    candidates: list[LipReadingCandidate]
    degradedModels: list[str]


class LipInferenceEngine:
    def __init__(self, readers: list[LipReader]) -> None:
        self._readers = readers

    def predict(self, window: MouthWindow) -> LipInferenceResult:
        candidates: list[LipReadingCandidate] = []
        degraded: list[str] = []
        for reader in self._readers:
            try:
                candidates.append(reader.predict(window))
            except Exception:
                degraded.append(reader.name)
        return LipInferenceResult(candidates=candidates, degradedModels=degraded)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lip_inference.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add lip/fake.py lip/inference.py tests/test_lip_inference.py
git commit -m "feat: add fake bilingual lip inference"
```

---

### Task 6: MiniCPM Semantic Interpreter and Agent Policy

**Files:**
- Create: `llm/minicpm.py`
- Create: `agent/agent.py`
- Create: `tests/test_minicpm_agent.py`

**Interfaces:**
- Consumes: `LipReadingCandidate`, `SemanticResult`.
- Produces: `FakeMiniCPMInterpreter.interpret(candidates, sampled_frames, stats)`, `parse_minicpm_json(raw: str) -> SemanticResult`, `AgentPolicy.decide(result: SemanticResult) -> AgentResult`.

- [ ] **Step 1: Write failing MiniCPM and Agent tests**

```python
# tests/test_minicpm_agent.py
import numpy as np
import pytest
from pydantic import ValidationError

from agent.agent import AgentPolicy
from backend.schemas import LipReadingCandidate, SemanticResult
from llm.minicpm import FakeMiniCPMInterpreter, parse_minicpm_json


def test_parse_minicpm_json_accepts_strict_object():
    parsed = parse_minicpm_json('{"language":"zh","text":"请打开灯","confidence":0.8,"reason":"中文候选更可靠"}')
    assert parsed.language == "zh"
    assert parsed.text == "请打开灯"


def test_parse_minicpm_json_rejects_markdown():
    with pytest.raises(ValidationError):
        parse_minicpm_json("```json\n{\"language\":\"zh\"}\n```")


def test_fake_minicpm_selects_highest_confidence_candidate():
    candidates = [
        LipReadingCandidate(model="avhubert", language="en", text="turn on the light", confidence=0.4, latencyMs=1),
        LipReadingCandidate(model="cmlr", language="zh", text="请打开灯", confidence=0.8, latencyMs=1),
    ]
    result = FakeMiniCPMInterpreter(threshold=0.55).interpret(candidates, [np.zeros((96, 96), dtype=np.uint8)], {})
    assert result.language == "zh"
    assert result.text == "请打开灯"


def test_fake_minicpm_returns_unknown_for_low_confidence():
    candidates = [LipReadingCandidate(model="avhubert", language="en", text="maybe", confidence=0.2, latencyMs=1)]
    result = FakeMiniCPMInterpreter(threshold=0.55).interpret(candidates, [], {})
    assert result.language == "unknown"
    assert result.text == ""


def test_agent_policy_has_no_side_effect_actions():
    result = SemanticResult(language="en", text="turn on the light", confidence=0.75, reason="candidate accepted")
    action = AgentPolicy(threshold=0.55).decide(result)
    assert action.action == "respond"
    assert action.arguments == {}
    assert action.requiresConfirmation is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_minicpm_agent.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'llm.minicpm'`.

- [ ] **Step 3: Implement strict semantic parser, fake interpreter, and agent**

```python
# llm/minicpm.py
import json

import numpy as np
from pydantic import TypeAdapter, ValidationError

from backend.schemas import LipReadingCandidate, SemanticResult

SEMANTIC_ADAPTER = TypeAdapter(SemanticResult)

SYSTEM_PROMPT = (
    "You are a visual lipreading semantic judge. Use only mouth motion evidence, "
    "model candidates, scores, and timing stats. Do not infer language from face, "
    "identity, appearance, skin color, nationality, or name. Do not translate one "
    "candidate and present it as lipreading. If evidence is insufficient, return "
    "{\"language\":\"unknown\",\"text\":\"\",\"confidence\":0.0,\"reason\":\"insufficient visual evidence\"}. "
    "Return exactly one JSON object with language, text, confidence, and reason."
)


def parse_minicpm_json(raw: str) -> SemanticResult:
    stripped = raw.strip()
    if not stripped.startswith("{") or not stripped.endswith("}"):
        raise ValidationError.from_exception_data("SemanticResult", [])
    loaded = json.loads(stripped)
    return SEMANTIC_ADAPTER.validate_python(loaded)


class FakeMiniCPMInterpreter:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def interpret(
        self,
        candidates: list[LipReadingCandidate],
        sampled_frames: list[np.ndarray],
        stats: dict[str, float | int | str],
    ) -> SemanticResult:
        scored = [candidate for candidate in candidates if candidate.confidence is not None]
        if not scored:
            return SemanticResult(language="unknown", text="", confidence=0.0, reason="no scored candidates")
        best = max(scored, key=lambda candidate: candidate.confidence or 0.0)
        confidence = best.confidence or 0.0
        if confidence < self.threshold:
            return SemanticResult(language="unknown", text="", confidence=confidence, reason="candidate below threshold")
        return SemanticResult(language=best.language, text=best.text, confidence=confidence, reason=f"{best.model} candidate accepted")
```

```python
# agent/agent.py
from backend.schemas import AgentResult, SemanticResult


class AgentPolicy:
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def decide(self, result: SemanticResult) -> AgentResult:
        if result.language == "unknown" or result.confidence < self.threshold or not result.text.strip():
            return AgentResult(action="unknown", language="unknown", text="", arguments={}, requiresConfirmation=False)
        if "?" in result.text or "吗" in result.text:
            return AgentResult(action="confirm", language=result.language, text=result.text, arguments={}, requiresConfirmation=True)
        return AgentResult(action="respond", language=result.language, text=result.text, arguments={}, requiresConfirmation=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_minicpm_agent.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add llm/minicpm.py agent/agent.py tests/test_minicpm_agent.py
git commit -m "feat: add semantic interpreter and agent policy"
```

---

### Task 7: WebSocket Protocol with Fake End-to-End Pipeline

**Files:**
- Create: `api/websocket.py`
- Modify: `backend/main.py`
- Modify: `backend/schemas.py`
- Modify: `tests/conftest.py`
- Create: `tests/test_websocket_flow.py`

**Interfaces:**
- Consumes: `SessionManager`, `decode_jpeg_frame`, `crop_mouth`, `LipInferenceEngine`, `FakeMiniCPMInterpreter`, `AgentPolicy`.
- Produces: `/ws/{session_id}` supporting JSON `stream.start`, `stream.stop`, `ping` and binary JPEG frame messages.

- [ ] **Step 1: Write failing WebSocket flow test**

```python
# tests/test_websocket_flow.py
from fastapi.testclient import TestClient

from tests.conftest import make_jpeg


def test_fake_websocket_flow_reaches_agent_result(app):
    client = TestClient(app)
    session_response = client.post("/api/sessions")
    session_id = session_response.json()["sessionId"]
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        ready = websocket.receive_json()
        assert ready["type"] == "session.ready"
        websocket.send_json({"type": "stream.start"})
        started = websocket.receive_json()
        assert started["type"] == "stream.started"
        for _ in range(75):
            websocket.send_bytes(make_jpeg())
        seen = set()
        for _ in range(400):
            message = websocket.receive_json()
            seen.add(message["type"])
            if message["type"] == "agent.result":
                assert message["action"] in {"respond", "confirm", "unknown"}
                break
        assert {"vision.result", "buffer.progress", "inference.started", "lip.candidates", "semantic.result", "agent.result"} <= seen


def test_second_active_websocket_gets_server_busy(app):
    client = TestClient(app)
    first_id = client.post("/api/sessions").json()["sessionId"]
    second_id = client.post("/api/sessions").json()["sessionId"]
    with client.websocket_connect(f"/ws/{first_id}", headers={"origin": "http://localhost:8000"}) as first:
        assert first.receive_json()["type"] == "session.ready"
        with client.websocket_connect(f"/ws/{second_id}", headers={"origin": "http://localhost:8000"}) as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["code"] == "SERVER_BUSY"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_websocket_flow.py -v`

Expected: FAIL because `/ws/{session_id}` is not registered.

- [ ] **Step 3: Implement fake pipeline objects on app state**

```python
# backend/main.py additions inside create_app
from api.websocket import router as websocket_router
from agent.agent import AgentPolicy
from lip.fake import FakeLipReader
from lip.inference import LipInferenceEngine
from llm.minicpm import FakeMiniCPMInterpreter

app.state.lip_engine = LipInferenceEngine([
    FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
    FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
])
app.state.semantic_interpreter = FakeMiniCPMInterpreter(threshold=app_settings.model_confidence_threshold)
app.state.agent_policy = AgentPolicy(threshold=app_settings.model_confidence_threshold)
app.include_router(websocket_router)
```

- [ ] **Step 4: Implement WebSocket route and event helpers**

```python
# api/websocket.py
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.schemas import ErrorCode, ErrorEvent, utc_now
from lip.base import MouthFrame
from session.manager import ServerBusyError, SessionError
from vision.face import FrameDecodeError, decode_jpeg_frame
from vision.mouth import crop_mouth

router = APIRouter()


def _event(event_type: str, session_id: str, **payload: object) -> dict[str, object]:
    return {"type": event_type, "sessionId": session_id, "timestamp": utc_now().isoformat(), **payload}


async def _send_error(websocket: WebSocket, session_id: str, stage: str, code: ErrorCode, message: str, recoverable: bool) -> None:
    event = ErrorEvent(sessionId=session_id, stage=stage, code=code, message=message, recoverable=recoverable)
    await websocket.send_json(event.model_dump(mode="json"))


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    settings = websocket.app.state.settings
    origin = websocket.headers.get("origin", "")
    if origin and origin not in settings.allowed_origin_set:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    try:
        active = websocket.app.state.session_manager.activate(session_id)
    except ServerBusyError:
        await _send_error(websocket, session_id, "session", ErrorCode.SERVER_BUSY, "server already has an active streaming session", False)
        await websocket.close(code=1013)
        return
    except SessionError:
        await _send_error(websocket, session_id, "session", ErrorCode.INVALID_SESSION, "invalid or expired session", False)
        await websocket.close(code=1008)
        return

    await websocket.send_json(_event("session.ready", session_id, parameters={
        "captureFps": settings.capture_fps,
        "windowFrames": settings.window_frames,
        "inferenceStride": settings.inference_stride,
        "mouthSize": settings.mouth_size,
    }))

    sequence = 0
    try:
        while True:
            message = await websocket.receive()
            if "text" in message:
                command = message["text"]
                if "stream.start" in command:
                    active.reset_stream()
                    await websocket.send_json(_event("stream.started", session_id))
                elif "stream.stop" in command:
                    active.reset_stream()
                    active.streaming = False
                    await websocket.send_json(_event("stream.stopped", session_id))
                elif "ping" in command:
                    await websocket.send_json(_event("pong", session_id))
                continue
            if "bytes" not in message or not active.streaming:
                continue
            sequence += 1
            received_at_ms = int(time.time() * 1000)
            try:
                image = decode_jpeg_frame(message["bytes"], settings)
            except FrameDecodeError as exc:
                await _send_error(websocket, session_id, "frame", exc.code, str(exc), True)
                continue
            landmarks = [(0.45, 0.55), (0.55, 0.55), (0.50, 0.62), (0.50, 0.50)]
            crop = crop_mouth(image, landmarks, settings.mouth_size)
            frame = MouthFrame(sequence=sequence, received_at_ms=received_at_ms, image=crop.image)
            window = active.add_mouth_frame(frame)
            await websocket.send_json(_event("vision.result", session_id, faceDetected=True, mouthBox=crop.box.model_dump(), bufferedFrames=len(active.frames)))
            await websocket.send_json(_event("buffer.progress", session_id, bufferedFrames=len(active.frames), requiredFrames=settings.window_frames))
            if window is None:
                continue
            await websocket.send_json(_event("inference.started", session_id, startSequence=window.start_sequence, endSequence=window.end_sequence))
            lip_result = websocket.app.state.lip_engine.predict(window)
            await websocket.send_json(_event("lip.candidates", session_id, candidates=[c.model_dump() for c in lip_result.candidates], degradedModels=lip_result.degradedModels))
            sampled = [frame.image for frame in window.frames[::15]]
            semantic = websocket.app.state.semantic_interpreter.interpret(lip_result.candidates, sampled, {"frames": len(window.frames)})
            await websocket.send_json(_event("semantic.result", session_id, **semantic.model_dump()))
            agent = websocket.app.state.agent_policy.decide(semantic)
            await websocket.send_json(agent.model_dump(mode="json") | {"sessionId": session_id, "timestamp": utc_now().isoformat()})
    except WebSocketDisconnect:
        websocket.app.state.session_manager.disconnect(session_id)
    finally:
        websocket.app.state.session_manager.disconnect(session_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_websocket_flow.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add api/websocket.py backend/main.py backend/schemas.py tests/conftest.py tests/test_websocket_flow.py
git commit -m "feat: add websocket fake pipeline"
```

---

### Task 8: Frontend Camera, WebSocket Client, and Phase UI

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/camera.js`
- Create: `frontend/websocket.js`
- Create: `frontend/styles.css`
- Modify: `backend/main.py`
- Create: `tests/e2e/camera.spec.js`
- Create: `package.json`

**Interfaces:**
- Consumes: `POST /api/sessions`, `/ws/{session_id}` events.
- Produces: Browser UI that opens camera, sends JPEG binary frames, and renders stage state.

- [ ] **Step 1: Write failing browser-visible static file test**

```python
# append to tests/test_websocket_flow.py
def test_frontend_index_is_served(app):
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "Silent Vision" in response.text
    assert "startButton" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_websocket_flow.py::test_frontend_index_is_served -v`

Expected: FAIL with 404.

- [ ] **Step 3: Implement static frontend serving**

```python
# backend/main.py additions
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")
```

- [ ] **Step 4: Create frontend files**

```html
<!-- frontend/index.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Silent Vision</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="app">
    <section class="preview">
      <video id="cameraPreview" autoplay playsinline muted></video>
      <canvas id="overlayCanvas"></canvas>
      <canvas id="captureCanvas" hidden></canvas>
    </section>
    <section class="panel">
      <div class="controls">
        <button id="startButton" type="button">Start</button>
        <button id="stopButton" type="button" disabled>Stop</button>
      </div>
      <dl id="statusList">
        <dt>Camera</dt><dd id="cameraStatus">idle</dd>
        <dt>WebSocket</dt><dd id="socketStatus">idle</dd>
        <dt>Vision</dt><dd id="visionStatus">waiting</dd>
        <dt>Buffer</dt><dd id="bufferStatus">0 / 75</dd>
        <dt>Lip</dt><dd id="lipStatus">waiting</dd>
        <dt>MiniCPM</dt><dd id="semanticStatus">waiting</dd>
        <dt>Agent</dt><dd id="agentStatus">waiting</dd>
      </dl>
      <pre id="candidateOutput"></pre>
      <pre id="resultOutput"></pre>
    </section>
  </main>
  <script type="module" src="/static/websocket.js"></script>
</body>
</html>
```

```javascript
// frontend/camera.js
export class CameraStreamer {
  constructor({ video, canvas, fps, onFrame }) {
    this.video = video;
    this.canvas = canvas;
    this.fps = fps;
    this.onFrame = onFrame;
    this.stream = null;
    this.timer = null;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    this.video.srcObject = this.stream;
    await this.video.play();
    const intervalMs = Math.round(1000 / this.fps);
    this.timer = window.setInterval(() => this.capture(), intervalMs);
  }

  capture() {
    if (!this.video.videoWidth || !this.video.videoHeight) return;
    this.canvas.width = this.video.videoWidth;
    this.canvas.height = this.video.videoHeight;
    const context = this.canvas.getContext("2d");
    context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
    this.canvas.toBlob((blob) => {
      if (blob) this.onFrame(blob);
    }, "image/jpeg", 0.75);
  }

  stop() {
    if (this.timer) window.clearInterval(this.timer);
    this.timer = null;
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
    }
    this.stream = null;
  }
}
```

```javascript
// frontend/websocket.js
import { CameraStreamer } from "./camera.js";

const state = {
  ws: null,
  sessionId: null,
  camera: null,
  parameters: { captureFps: 25, windowFrames: 75 },
};

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

async function createSession() {
  const response = await fetch("/api/sessions", { method: "POST" });
  if (!response.ok) throw new Error("session creation failed");
  const body = await response.json();
  sessionStorage.setItem("silentVisionSessionId", body.sessionId);
  return body.sessionId;
}

function wsUrl(sessionId) {
  const scheme = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${window.location.host}/ws/${sessionId}`;
}

async function connect() {
  state.sessionId = await createSession();
  state.ws = new WebSocket(wsUrl(state.sessionId));
  state.ws.binaryType = "arraybuffer";
  state.ws.onopen = () => setText("socketStatus", "connected");
  state.ws.onclose = () => setText("socketStatus", "closed");
  state.ws.onmessage = (event) => handleEvent(JSON.parse(event.data));
}

function handleEvent(event) {
  if (event.type === "session.ready") state.parameters = event.parameters;
  if (event.type === "vision.result") setText("visionStatus", event.faceDetected ? "mouth detected" : "not detected");
  if (event.type === "buffer.progress") setText("bufferStatus", `${event.bufferedFrames} / ${event.requiredFrames}`);
  if (event.type === "lip.candidates") {
    setText("lipStatus", "candidates ready");
    setText("candidateOutput", JSON.stringify(event.candidates, null, 2));
  }
  if (event.type === "semantic.result") setText("semanticStatus", `${event.language}: ${event.text}`);
  if (event.type === "agent.result") {
    setText("agentStatus", event.action);
    setText("resultOutput", JSON.stringify(event, null, 2));
  }
  if (event.type === "error") setText("visionStatus", `${event.code}: ${event.message}`);
}

document.getElementById("startButton").addEventListener("click", async () => {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) await connect();
  state.ws.send(JSON.stringify({ type: "stream.start" }));
  state.camera = new CameraStreamer({
    video: document.getElementById("cameraPreview"),
    canvas: document.getElementById("captureCanvas"),
    fps: state.parameters.captureFps || 25,
    onFrame: (blob) => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(blob);
    },
  });
  await state.camera.start();
  setText("cameraStatus", "streaming");
  document.getElementById("startButton").disabled = true;
  document.getElementById("stopButton").disabled = false;
});

document.getElementById("stopButton").addEventListener("click", () => {
  if (state.camera) state.camera.stop();
  if (state.ws && state.ws.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "stream.stop" }));
  setText("cameraStatus", "stopped");
  document.getElementById("startButton").disabled = false;
  document.getElementById("stopButton").disabled = true;
});
```

```css
/* frontend/styles.css */
body { margin: 0; font-family: system-ui, sans-serif; background: #101418; color: #eef2f5; }
.app { display: grid; grid-template-columns: minmax(320px, 1fr) 420px; min-height: 100vh; }
.preview { position: relative; background: #050708; }
video, #overlayCanvas { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.panel { padding: 24px; border-left: 1px solid #29323a; overflow: auto; }
button { min-width: 96px; padding: 10px 14px; border: 1px solid #4a5865; background: #17212b; color: #eef2f5; border-radius: 6px; }
dl { display: grid; grid-template-columns: 120px 1fr; gap: 8px 12px; }
pre { white-space: pre-wrap; background: #0b0f13; padding: 12px; border-radius: 6px; }
@media (max-width: 800px) { .app { grid-template-columns: 1fr; grid-template-rows: 55vh auto; } .panel { border-left: 0; border-top: 1px solid #29323a; } }
```

- [ ] **Step 5: Run static test**

Run: `pytest tests/test_websocket_flow.py::test_frontend_index_is_served -v`

Expected: PASS.

- [ ] **Step 6: Add Playwright package file and E2E smoke**

```json
{
  "scripts": {
    "test:e2e": "playwright test tests/e2e"
  },
  "devDependencies": {
    "@playwright/test": "^1.46.0"
  }
}
```

```javascript
// tests/e2e/camera.spec.js
const { test, expect } = require("@playwright/test");

test("home page exposes camera controls and status", async ({ page }) => {
  await page.goto("http://127.0.0.1:8000/");
  await expect(page.locator("#startButton")).toBeVisible();
  await expect(page.locator("#cameraStatus")).toContainText("idle");
  await expect(page.locator("#bufferStatus")).toContainText("0 / 75");
});
```

- [ ] **Step 7: Commit**

```bash
git add frontend backend/main.py tests/test_websocket_flow.py tests/e2e/camera.spec.js package.json
git commit -m "feat: add browser camera client"
```

---

### Task 9: Real MediaPipe Integration in WebSocket Pipeline

**Files:**
- Modify: `backend/main.py`
- Modify: `api/websocket.py`
- Modify: `vision/face.py`
- Modify: `tests/test_vision_mouth.py`

**Interfaces:**
- Consumes: `FaceDetector.detect(image_bgr) -> FaceDetectionResult`.
- Produces: Real MediaPipe landmarks in `MODEL_BACKEND=real`; deterministic fake landmarks in `MODEL_BACKEND=fake`.

- [ ] **Step 1: Add failing fake-vs-real detector factory tests**

```python
# append to tests/test_vision_mouth.py
from vision.face import FakeFaceDetector, create_face_detector


def test_fake_face_detector_returns_one_face():
    detector = FakeFaceDetector()
    result = detector.detect(np.full((200, 300, 3), 128, dtype=np.uint8))
    assert result.face_detected is True
    assert result.face_count == 1
    assert len(result.landmarks) >= 4


def test_create_face_detector_uses_fake_backend_by_default():
    detector = create_face_detector(Settings(model_backend="fake"))
    assert isinstance(detector, FakeFaceDetector)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_vision_mouth.py::test_fake_face_detector_returns_one_face tests/test_vision_mouth.py::test_create_face_detector_uses_fake_backend_by_default -v`

Expected: FAIL because `FakeFaceDetector` and `create_face_detector` are missing.

- [ ] **Step 3: Implement detector factory and use it in app state**

```python
# vision/face.py additions
from typing import Protocol


class FaceDetectorProtocol(Protocol):
    def detect(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        raise NotImplementedError


class FakeFaceDetector:
    def detect(self, image_bgr: np.ndarray) -> FaceDetectionResult:
        return FaceDetectionResult(
            face_detected=True,
            face_count=1,
            landmarks=[(0.45, 0.55), (0.55, 0.55), (0.50, 0.62), (0.50, 0.50)],
        )


def create_face_detector(settings: Settings) -> FaceDetectorProtocol:
    if settings.model_backend == "fake":
        return FakeFaceDetector()
    return FaceDetector()
```

```python
# backend/main.py additions inside create_app
from vision.face import create_face_detector

app.state.face_detector = create_face_detector(app_settings)
```

```python
# api/websocket.py replace hard-coded landmarks block
detection = websocket.app.state.face_detector.detect(image)
if not detection.face_detected:
    code = ErrorCode.MULTIPLE_FACES if detection.face_count > 1 else ErrorCode.FACE_NOT_FOUND
    await _send_error(websocket, session_id, "vision", code, "current frame does not contain exactly one clear face", True)
    continue
crop = crop_mouth(image, detection.landmarks, settings.mouth_size)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_vision_mouth.py tests/test_websocket_flow.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py api/websocket.py vision/face.py tests/test_vision_mouth.py
git commit -m "feat: wire mediapipe detector factory"
```

---

### Task 10: Real AV-HuBERT and CMLR Adapter Probes

**Files:**
- Create: `lip/avhubert.py`
- Create: `lip/cmlr.py`
- Modify: `backend/config.py`
- Modify: `backend/main.py`
- Create: `tests/test_rocm_models.py`
- Create: `tests/media/manifest.example.json`

**Interfaces:**
- Consumes: `MouthWindow`, `Settings.avhubert_checkpoint`, `Settings.cmlr_checkpoint`, `Settings.cmlr_language_model`.
- Produces: `AVHuBERTLipReader.predict(window)`, `CMLRLipReader.predict(window)`, real reader factory.

- [ ] **Step 1: Write marked adapter probe tests**

```python
# tests/test_rocm_models.py
import json
from pathlib import Path

import pytest

from backend.config import Settings
from lip.avhubert import AVHuBERTLipReader
from lip.cmlr import CMLRLipReader
from tests.test_lip_inference import _window


pytestmark = [pytest.mark.rocm, pytest.mark.model_integration]


def test_avhubert_checkpoint_exists_before_loading():
    settings = Settings(model_backend="real")
    assert settings.avhubert_checkpoint.exists(), settings.avhubert_checkpoint


def test_cmlr_checkpoint_exists_before_loading():
    settings = Settings(model_backend="real")
    assert settings.cmlr_checkpoint.exists(), settings.cmlr_checkpoint


def test_test_media_manifest_schema_example_is_valid():
    manifest = json.loads(Path("tests/media/manifest.example.json").read_text())
    assert manifest["version"] == 1
    for item in manifest["samples"]:
        assert set(item) == {"path", "language", "expected_text", "license_source"}
        assert item["language"] in {"zh", "en"}


def test_avhubert_reader_smoke_predicts_english_candidate():
    reader = AVHuBERTLipReader(Settings(model_backend="real"))
    candidate = reader.predict(_window())
    assert candidate.model == "avhubert"
    assert candidate.language == "en"
    assert isinstance(candidate.text, str)


def test_cmlr_reader_smoke_predicts_chinese_candidate():
    reader = CMLRLipReader(Settings(model_backend="real"))
    candidate = reader.predict(_window())
    assert candidate.model == "cmlr"
    assert candidate.language == "zh"
    assert isinstance(candidate.text, str)
```

- [ ] **Step 2: Run non-hardware subset to verify default suite skips probes**

Run: `pytest -m "not rocm and not model_integration" -v`

Expected: PASS and no model weight loading.

- [ ] **Step 3: Implement real adapter loading boundaries**

```python
# lip/avhubert.py
from time import perf_counter

import numpy as np
import torch

from backend.config import Settings
from backend.schemas import LipReadingCandidate
from lip.base import MouthWindow


class AVHuBERTLipReader:
    name = "avhubert"
    language = "en"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.avhubert_checkpoint.exists():
            raise FileNotFoundError(settings.avhubert_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(settings.avhubert_checkpoint), map_location=self.device)
        self.model.eval()

    def _window_tensor(self, window: MouthWindow) -> torch.Tensor:
        frames = np.stack([frame.image for frame in window.frames]).astype("float32") / 255.0
        tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        started = perf_counter()
        with torch.inference_mode():
            output = self.model(self._window_tensor(window))
        text = ""
        raw_score: float | None = None
        confidence: float | None = None
        if isinstance(output, dict):
            text = str(output.get("text", ""))
            raw = output.get("score")
            raw_score = float(raw) if raw is not None else None
            conf = output.get("confidence")
            confidence = float(conf) if conf is not None else None
        return LipReadingCandidate(
            model="avhubert",
            language="en",
            text=text,
            confidence=confidence,
            rawScore=raw_score,
            latencyMs=int((perf_counter() - started) * 1000),
        )
```

```python
# lip/cmlr.py
from time import perf_counter

import numpy as np
import torch

from backend.config import Settings
from backend.schemas import LipReadingCandidate
from lip.base import MouthWindow


class CMLRLipReader:
    name = "cmlr"
    language = "zh"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.cmlr_checkpoint.exists():
            raise FileNotFoundError(settings.cmlr_checkpoint)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(settings.cmlr_checkpoint), map_location=self.device)
        self.model.eval()

    def _window_tensor(self, window: MouthWindow) -> torch.Tensor:
        frames = np.stack([frame.image for frame in window.frames]).astype("float32") / 255.0
        tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        started = perf_counter()
        with torch.inference_mode():
            output = self.model(self._window_tensor(window))
        text = ""
        raw_score: float | None = None
        confidence: float | None = None
        if isinstance(output, dict):
            text = str(output.get("text", ""))
            raw = output.get("score")
            raw_score = float(raw) if raw is not None else None
            conf = output.get("confidence")
            confidence = float(conf) if conf is not None else None
        return LipReadingCandidate(
            model="cmlr",
            language="zh",
            text=text,
            confidence=confidence,
            rawScore=raw_score,
            latencyMs=int((perf_counter() - started) * 1000),
        )
```

```json
{
  "version": 1,
  "samples": [
    {
      "path": "/workspace/persistence/silent-vision/reports/benchmarks/en_sample.mp4",
      "language": "en",
      "expected_text": "turn on the light",
      "license_source": "local authorized test sample"
    },
    {
      "path": "/workspace/persistence/silent-vision/reports/benchmarks/zh_sample.mp4",
      "language": "zh",
      "expected_text": "请打开灯",
      "license_source": "local authorized test sample"
    }
  ]
}
```

- [ ] **Step 4: Wire real readers in app factory**

```python
# backend/main.py replace fake-only lip engine construction
if app_settings.model_backend == "real":
    from lip.avhubert import AVHuBERTLipReader
    from lip.cmlr import CMLRLipReader
    readers = [AVHuBERTLipReader(app_settings), CMLRLipReader(app_settings)]
else:
    readers = [
        FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
        FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
    ]
app.state.lip_engine = LipInferenceEngine(readers)
```

- [ ] **Step 5: Run fake suite and optional hardware probes**

Run default: `pytest -m "not rocm and not model_integration" -v`

Expected: PASS.

Run on ROCm server only after placing serving artifacts at the configured paths: `MODEL_BACKEND=real pytest tests/test_rocm_models.py -m "rocm and model_integration" -v`

Expected: PASS. The serving artifact contract for this plan is that `/workspace/persistence/silent-vision/models/avhubert/model.pt` and `/workspace/persistence/silent-vision/models/cmlr/model.pth` are local TorchScript-style callable artifacts returning a dict with `text`, optional `score`, and optional `confidence`.

- [ ] **Step 6: Commit**

```bash
git add lip/avhubert.py lip/cmlr.py backend/config.py backend/main.py tests/test_rocm_models.py tests/media/manifest.example.json
git commit -m "feat: add real lip reader adapter probes"
```

---

### Task 11: Real MiniCPM-o Adapter and GPU Inference Lock

**Files:**
- Modify: `llm/minicpm.py`
- Modify: `backend/main.py`
- Modify: `api/websocket.py`
- Modify: `tests/test_minicpm_agent.py`
- Modify: `tests/test_rocm_models.py`

**Interfaces:**
- Consumes: `SYSTEM_PROMPT`, `parse_minicpm_json`, `SemanticResult`.
- Produces: `RealMiniCPMInterpreter.interpret(candidates, sampled_frames, stats)`, one global `asyncio.Lock` protecting lip+MiniCPM inference.

- [ ] **Step 1: Add failing MiniCPM real path and lock tests**

```python
# append to tests/test_minicpm_agent.py
from llm.minicpm import build_minicpm_interpreter
from backend.config import Settings


def test_build_minicpm_uses_fake_by_default():
    interpreter = build_minicpm_interpreter(Settings(model_backend="fake"))
    assert isinstance(interpreter, FakeMiniCPMInterpreter)
```

```python
# append to tests/test_rocm_models.py
from backend.schemas import LipReadingCandidate
from llm.minicpm import RealMiniCPMInterpreter


def test_real_minicpm_smoke_schema():
    settings = Settings(model_backend="real")
    interpreter = RealMiniCPMInterpreter(settings)
    result = interpreter.interpret(
        [LipReadingCandidate(model="cmlr", language="zh", text="请打开灯", confidence=0.7, latencyMs=1)],
        [],
        {"frames": 75},
    )
    assert result.language in {"zh", "en", "unknown"}
```

- [ ] **Step 2: Run fake test to verify failure**

Run: `pytest tests/test_minicpm_agent.py::test_build_minicpm_uses_fake_by_default -v`

Expected: FAIL because `build_minicpm_interpreter` is missing.

- [ ] **Step 3: Implement real MiniCPM wrapper and factory**

```python
# llm/minicpm.py additions
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoModel, AutoTokenizer

from backend.config import Settings


class RealMiniCPMInterpreter:
    def __init__(self, settings: Settings) -> None:
        model_path = Path(settings.minicpm_model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = self.model.eval().cuda()

    def _images(self, sampled_frames: list[np.ndarray]) -> list[Image.Image]:
        return [Image.fromarray(frame).convert("RGB") for frame in sampled_frames]

    def interpret(
        self,
        candidates: list[LipReadingCandidate],
        sampled_frames: list[np.ndarray],
        stats: dict[str, float | int | str],
    ) -> SemanticResult:
        payload: dict[str, Any] = {
            "candidates": [candidate.model_dump() for candidate in candidates],
            "stats": stats,
            "instruction": SYSTEM_PROMPT,
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [*self._images(sampled_frames), json.dumps(payload, ensure_ascii=False)]},
        ]
        raw = self.model.chat(image=None, msgs=messages, tokenizer=self.tokenizer)
        if isinstance(raw, tuple):
            raw = raw[0]
        return parse_minicpm_json(str(raw))


def build_minicpm_interpreter(settings: Settings) -> FakeMiniCPMInterpreter | RealMiniCPMInterpreter:
    if settings.model_backend == "real":
        return RealMiniCPMInterpreter(settings)
    return FakeMiniCPMInterpreter(threshold=settings.model_confidence_threshold)
```

- [ ] **Step 4: Wire global inference lock**

```python
# backend/main.py additions
import asyncio
from llm.minicpm import build_minicpm_interpreter

app.state.inference_lock = asyncio.Lock()
app.state.semantic_interpreter = build_minicpm_interpreter(app_settings)
```

```python
# api/websocket.py wrap lip+semantic section
async with websocket.app.state.inference_lock:
    lip_result = websocket.app.state.lip_engine.predict(window)
    await websocket.send_json(_event("lip.candidates", session_id, candidates=[c.model_dump() for c in lip_result.candidates], degradedModels=lip_result.degradedModels))
    if not lip_result.candidates:
        await _send_error(websocket, session_id, "lip", ErrorCode.LIP_MODELS_FAILED, "both lip reading models failed", True)
        continue
    sampled = [frame.image for frame in window.frames[::15]]
    try:
        semantic = websocket.app.state.semantic_interpreter.interpret(lip_result.candidates, sampled, {"frames": len(window.frames)})
    except Exception:
        await _send_error(websocket, session_id, "minicpm", ErrorCode.MINICPM_FAILED, "MiniCPM failed to produce valid semantic JSON", True)
        semantic = SemanticResult(language="unknown", text="", confidence=0.0, reason="MiniCPM failure")
```

- [ ] **Step 5: Run fake suite and optional MiniCPM hardware smoke**

Run default: `pytest -m "not rocm and not model_integration" -v`

Expected: PASS.

Run on ROCm server only: `MODEL_BACKEND=real pytest tests/test_rocm_models.py::test_real_minicpm_smoke_schema -m "rocm and model_integration" -v`

Expected: PASS with local adapted MiniCPM-o 4.5.

- [ ] **Step 6: Commit**

```bash
git add llm/minicpm.py backend/main.py api/websocket.py tests/test_minicpm_agent.py tests/test_rocm_models.py
git commit -m "feat: add minicpm adapter and inference lock"
```

---

### Task 12: Metrics, Privacy Cleanup, and Error Hardening

**Files:**
- Modify: `backend/schemas.py`
- Modify: `session/manager.py`
- Modify: `api/websocket.py`
- Create: `tests/test_privacy_metrics.py`

**Interfaces:**
- Consumes: active session lifecycle and WebSocket events.
- Produces: `metrics.update` events, `/tmp/silent-vision/{session_id}` cleanup, bounded pending inference semantics.

- [ ] **Step 1: Write failing privacy and metrics tests**

```python
# tests/test_privacy_metrics.py
from pathlib import Path

from fastapi.testclient import TestClient

from tests.conftest import make_jpeg


def test_disconnect_removes_session_temp_dir(app, tmp_path, monkeypatch):
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    temp_dir = Path("/tmp/silent-vision") / session_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    (temp_dir / "frame.jpg").write_bytes(b"not persisted")
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        assert websocket.receive_json()["type"] == "session.ready"
    assert not temp_dir.exists()


def test_metrics_update_is_emitted(app):
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as websocket:
        websocket.receive_json()
        websocket.send_json({"type": "stream.start"})
        websocket.receive_json()
        websocket.send_bytes(make_jpeg())
        seen = []
        for _ in range(20):
            message = websocket.receive_json()
            seen.append(message["type"])
            if message["type"] == "metrics.update":
                assert "receivedFps" in message
                assert "validFrameRatio" in message
                break
        assert "metrics.update" in seen
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_privacy_metrics.py -v`

Expected: FAIL because cleanup and metrics are missing.

- [ ] **Step 3: Implement metrics event and temp cleanup helper**

```python
# api/websocket.py additions
import shutil
from pathlib import Path


def _cleanup_temp(session_id: str) -> None:
    temp_dir = Path("/tmp/silent-vision") / session_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _metrics(active) -> dict[str, object]:
    total = max(1, active.accepted_frame_count)
    return {
        "receivedFps": 0.0,
        "validFrameRatio": min(1.0, len(active.frames) / total),
        "bufferedFrames": len(active.frames),
        "droppedFrames": 0,
    }

# after buffer.progress send:
await websocket.send_json(_event("metrics.update", session_id, **_metrics(active)))

# in finally:
_cleanup_temp(session_id)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_privacy_metrics.py tests/test_websocket_flow.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/schemas.py session/manager.py api/websocket.py tests/test_privacy_metrics.py
git commit -m "feat: add metrics and privacy cleanup"
```

---

### Task 13: Docker ROCm Deployment and NFS Model Layout

**Files:**
- Create: `docker/Dockerfile`
- Create: `docker/docker-compose.yml`
- Modify: `README.md`
- Create: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes: `PERSISTENCE_ROOT=/workspace/persistence/silent-vision`, `MODEL_BACKEND`.
- Produces: Docker Compose service binding to `127.0.0.1:8000`, mounting NFS persistence root, and passing ROCm devices.

- [ ] **Step 1: Write failing deployment file tests**

```python
# tests/test_deployment_files.py
from pathlib import Path


def test_docker_compose_mounts_persistence_root_and_rocm_devices():
    compose = Path("docker/docker-compose.yml").read_text()
    assert "/workspace/persistence/silent-vision:/workspace/persistence/silent-vision" in compose
    assert "/dev/kfd:/dev/kfd" in compose
    assert "/dev/dri:/dev/dri" in compose
    assert "127.0.0.1:8000:8000" in compose


def test_dockerfile_does_not_copy_model_weights():
    dockerfile = Path("docker/Dockerfile").read_text()
    assert "COPY . /app" in dockerfile
    assert "models/" not in dockerfile
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deployment_files.py -v`

Expected: FAIL because Docker files do not exist.

- [ ] **Step 3: Add Dockerfile and Compose**

```dockerfile
# docker/Dockerfile
FROM rocm/pytorch:latest

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/persistence/silent-vision/cache/huggingface
ENV TORCH_HOME=/workspace/persistence/silent-vision/cache/torch

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && python -m pip install -r /app/requirements.txt

COPY . /app

CMD ["uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
services:
  silent-vision:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    environment:
      MODEL_BACKEND: real
      PERSISTENCE_ROOT: /workspace/persistence/silent-vision
      ALLOWED_ORIGINS: http://localhost:8000
      LOG_TRANSCRIPTS: "false"
      HOST: 127.0.0.1
      PORT: "8000"
    devices:
      - /dev/kfd:/dev/kfd
      - /dev/dri:/dev/dri
    group_add:
      - video
    ipc: host
    shm_size: 16gb
    volumes:
      - /workspace/persistence/silent-vision:/workspace/persistence/silent-vision
    ports:
      - 127.0.0.1:8000:8000
    restart: unless-stopped
```

- [ ] **Step 4: Extend README runbook**

```markdown
## Persistence layout

```text
/workspace/persistence/silent-vision/
├── models/
│   ├── avhubert/model.pt
│   ├── cmlr/model.pth
│   ├── cmlr/language-model.pth
│   └── minicpm-o-4_5/
├── cache/
├── reports/
└── logs/
```

## ROCm container

```bash
mkdir -p /workspace/persistence/silent-vision/models/{avhubert,cmlr,minicpm-o-4_5}
mkdir -p /workspace/persistence/silent-vision/cache/{huggingface,torch}
mkdir -p /workspace/persistence/silent-vision/reports/{benchmarks,diagnostics}
mkdir -p /workspace/persistence/silent-vision/logs
docker compose -f docker/docker-compose.yml up --build
```
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_deployment_files.py -v`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile docker/docker-compose.yml README.md tests/test_deployment_files.py
git commit -m "chore: add rocm docker deployment"
```

---

### Task 14: Final Verification, Stability Script, and Documentation Tightening

**Files:**
- Create: `scripts/smoke_fake.sh`
- Create: `scripts/smoke_rocm.sh`
- Modify: `README.md`
- Modify: `tests/test_rocm_models.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: repeatable verification commands for fake mode and ROCm mode.

- [ ] **Step 1: Write failing script existence tests**

```python
# append to tests/test_deployment_files.py
def test_smoke_scripts_exist_and_are_executable():
    fake = Path("scripts/smoke_fake.sh")
    rocm = Path("scripts/smoke_rocm.sh")
    assert fake.exists()
    assert rocm.exists()
    assert fake.read_text().startswith("#!/usr/bin/env bash")
    assert rocm.read_text().startswith("#!/usr/bin/env bash")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deployment_files.py::test_smoke_scripts_exist_and_are_executable -v`

Expected: FAIL because scripts are missing.

- [ ] **Step 3: Add smoke scripts**

```bash
#!/usr/bin/env bash
# scripts/smoke_fake.sh
set -euo pipefail
export MODEL_BACKEND=fake
pytest -m "not rocm and not model_integration" -v
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

```bash
#!/usr/bin/env bash
# scripts/smoke_rocm.sh
set -euo pipefail
export MODEL_BACKEND=real
export PERSISTENCE_ROOT=/workspace/persistence/silent-vision
python - <<'PY'
from pathlib import Path
root = Path("/workspace/persistence/silent-vision")
required = [
    root / "models" / "avhubert" / "model.pt",
    root / "models" / "cmlr" / "model.pth",
    root / "models" / "minicpm-o-4_5",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("missing required model paths: " + ", ".join(missing))
PY
pytest tests/test_rocm_models.py -m "rocm and model_integration" -v
docker compose -f docker/docker-compose.yml up --build
```

- [ ] **Step 4: Make scripts executable and run default verification**

Run:

```bash
chmod +x scripts/smoke_fake.sh scripts/smoke_rocm.sh
pytest -m "not rocm and not model_integration" -v
```

Expected: PASS.

- [ ] **Step 5: Add README verification section**

```markdown
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
```

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_fake.sh scripts/smoke_rocm.sh README.md tests/test_deployment_files.py tests/test_rocm_models.py
git commit -m "docs: add verification runbooks"
```

---

## Self-Review Checklist for Implementers

- [ ] Default verification passes without ROCm: `pytest -m "not rocm and not model_integration" -v`.
- [ ] Browser can load `/`, create a session, connect `/ws/{session_id}`, and stream binary JPEG frames.
- [ ] `MODEL_BACKEND=fake` never imports AV-HuBERT, CMLR, MiniCPM weights, or ROCm-only dependencies at import time.
- [ ] `MODEL_BACKEND=real` loads models only from `/workspace/persistence/silent-vision`.
- [ ] Second active WebSocket streaming session returns `SERVER_BUSY`.
- [ ] No raw JPEG, mouth frame, session state, MiniCPM temp image, or full transcript is written under Git or persistence logs.
- [ ] MiniCPM prompt forbids language guessing from appearance and forbids translation masquerading as lipreading.
- [ ] Agent output is always one of `respond`, `confirm`, `unknown` and performs no side effects.
- [ ] CMLR license constraints are documented before any non-research deployment.
