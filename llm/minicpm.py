import numpy as np
from pydantic import TypeAdapter

from backend.schemas import LipReadingCandidate, SemanticResult

SEMANTIC_ADAPTER = TypeAdapter(SemanticResult)

SYSTEM_PROMPT = (
    "You are a visual lipreading semantic judge. Use only mouth motion evidence, "
    "model candidates, scores, and timing stats. Do not infer language from face, "
    "identity, appearance, skin color, nationality, or name. Do not translate one "
    "candidate and present it as lipreading. If evidence is insufficient, return "
    '{"language":"unknown","text":"","confidence":0.0,"reason":"insufficient visual evidence"}. '
    "Return exactly one JSON object with language, text, confidence, and reason."
)


def parse_minicpm_json(raw: str) -> SemanticResult:
    return SEMANTIC_ADAPTER.validate_json(raw.strip())


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
        return SemanticResult(
            language=best.language,
            text=best.text,
            confidence=confidence,
            reason=f"{best.model} candidate accepted",
        )
