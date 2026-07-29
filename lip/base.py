from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from backend.schemas import LipReadingCandidate


@dataclass(frozen=True)
class MouthFrame:
    sequence: int
    received_at_ms: int
    image: np.ndarray


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
