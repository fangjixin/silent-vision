import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from backend.config import Settings
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


class RealMiniCPMInterpreter:
    def __init__(self, settings: Settings) -> None:
        model_path = Path(settings.minicpm_model_path)
        if not model_path.exists():
            raise FileNotFoundError(model_path)
        self.model_path = model_path
        auto_model, auto_tokenizer = _import_transformers()
        self.tokenizer = auto_tokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = auto_model.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = self.model.eval()
        if hasattr(self.model, "cuda"):
            self.model = self.model.cuda()

    def _images(self, sampled_frames: list[np.ndarray]) -> list[Any]:
        image = _import_pil_image()
        return [image.fromarray(frame).convert("RGB") for frame in sampled_frames]

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


def _import_transformers() -> tuple[Any, Any]:
    from transformers import AutoModel, AutoTokenizer

    return AutoModel, AutoTokenizer


def _import_pil_image() -> Any:
    from PIL import Image

    return Image
