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
