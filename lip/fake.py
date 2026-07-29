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
