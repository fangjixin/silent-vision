from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.schemas import LipReadingCandidate


class LipInferenceCancelled(Exception):
    """Raised when a lip inference job is cancelled by stream lifecycle changes."""


@dataclass(frozen=True)
class MouthFrame:
    sequence: int
    received_at_ms: int
    image: np.ndarray
    debug_image: np.ndarray | None = None
    debug_mouth_box: object | None = None
    debug_landmarks: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.image.shape != (96, 96):
            raise ValueError("mouth frame image must have shape (96, 96)")
        if self.image.dtype != np.uint8:
            raise ValueError("mouth frame image must use uint8")
        if self.debug_image is not None:
            if self.debug_image.ndim != 3 or self.debug_image.shape[2] != 3:
                raise ValueError("debug image must have RGB shape (height, width, 3)")
            if self.debug_image.dtype != np.uint8:
                raise ValueError("debug image must use uint8")


@dataclass(frozen=True)
class MouthWindow:
    session_id: str
    start_sequence: int
    end_sequence: int
    frames: Sequence[MouthFrame]


class LipReader(Protocol):
    name: str
    language: str

    def predict(self, window: MouthWindow, cancel_event: object | None = None) -> LipReadingCandidate:
        raise NotImplementedError
