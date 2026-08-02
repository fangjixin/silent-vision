from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

Language = Literal["zh", "en", "unknown"]
AgentAction = Literal["respond", "confirm", "unknown", "execute", "reject", "ignore"]
CommandIntentName = Literal["LIGHT_ON", "LIGHT_OFF", "OPEN_DOOR", "CHAT_OTHER", "UNKNOWN"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ErrorCode(str, Enum):
    INVALID_SESSION = "INVALID_SESSION"
    SESSION_REPLACED = "SESSION_REPLACED"
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


class CommandDecision(BaseModel):
    intent: CommandIntentName
    accepted: bool
    executable: bool
    confidence: float = Field(ge=0.0, le=1.0)
    margin: float = Field(ge=0.0, le=1.0)
    topK: list[dict[str, Any]]
    logits: list[float] | dict[str, float]
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


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
