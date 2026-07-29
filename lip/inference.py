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
