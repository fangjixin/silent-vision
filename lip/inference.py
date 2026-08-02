from dataclasses import dataclass
import logging
from time import perf_counter

from backend.schemas import LipReadingCandidate
from lip.base import LipInferenceCancelled, LipReader, MouthWindow

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LipInferenceResult:
    candidates: list[LipReadingCandidate]
    degradedModels: list[str]


class LipInferenceEngine:
    def __init__(self, readers: list[LipReader]) -> None:
        self._readers = readers

    def predict(self, window: MouthWindow, cancel_event: object | None = None) -> LipInferenceResult:
        candidates: list[LipReadingCandidate] = []
        degraded: list[str] = []
        for reader in self._readers:
            if _cancelled(cancel_event):
                raise LipInferenceCancelled("lip inference cancelled before reader started")
            started = perf_counter()
            logger.info(
                "lip reader started reader=%s language=%s session_id=%s window=%s-%s frames=%s",
                reader.name,
                reader.language,
                window.session_id,
                window.start_sequence,
                window.end_sequence,
                len(window.frames),
            )
            try:
                candidate = reader.predict(window, cancel_event=cancel_event)
                candidates.append(candidate)
                logger.info(
                    "lip reader completed reader=%s language=%s latency_ms=%s text_len=%s",
                    reader.name,
                    reader.language,
                    int((perf_counter() - started) * 1000),
                    len(candidate.text),
                )
            except LipInferenceCancelled:
                logger.info(
                    "lip reader cancelled reader=%s language=%s session_id=%s window=%s-%s",
                    reader.name,
                    reader.language,
                    window.session_id,
                    window.start_sequence,
                    window.end_sequence,
                )
                raise
            except Exception:
                logger.exception(
                    "lip reader failed reader=%s language=%s session_id=%s window=%s-%s",
                    reader.name,
                    reader.language,
                    window.session_id,
                    window.start_sequence,
                    window.end_sequence,
                )
                degraded.append(reader.name)
        return LipInferenceResult(candidates=candidates, degradedModels=degraded)


def _cancelled(cancel_event: object | None) -> bool:
    return bool(cancel_event is not None and getattr(cancel_event, "is_set")())
