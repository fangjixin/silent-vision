import json
import logging
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from pydantic import TypeAdapter

from backend.config import Settings
from backend.schemas import LipReadingCandidate, SemanticResult

SEMANTIC_ADAPTER = TypeAdapter(SemanticResult)
logger = logging.getLogger(__name__)

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
        self.settings = settings
        self.model_path = model_path
        auto_model, auto_tokenizer = _import_transformers()
        logger.info("MiniCPM loading model_path=%s", model_path)
        with _quiet_minicpm_model_load():
            self.tokenizer = auto_tokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            self.model = auto_model.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = self.model.eval()
        if hasattr(self.model, "cuda"):
            self.model = self.model.cuda()
        logger.info("MiniCPM loaded model_path=%s", model_path)

    def _images(self, sampled_frames: list[np.ndarray]) -> list[Any]:
        image = _import_pil_image()
        return [image.fromarray(frame).convert("RGB") for frame in sampled_frames]

    def interpret(
        self,
        candidates: list[LipReadingCandidate],
        sampled_frames: list[np.ndarray],
        stats: dict[str, float | int | str],
    ) -> SemanticResult:
        started = perf_counter()
        payload: dict[str, Any] = {
            "candidates": [candidate.model_dump() for candidate in candidates],
            "stats": stats,
            "instruction": SYSTEM_PROMPT,
        }
        logger.info(
            "MiniCPM interpret started candidates=%s sampled_frames=%s stats=%s candidate_summary=%s",
            len(candidates),
            len(sampled_frames),
            stats,
            _candidate_summary(candidates, include_text=self.settings.log_transcripts),
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [*self._images(sampled_frames), json.dumps(payload, ensure_ascii=False)]},
        ]
        raw = self.model.chat(image=None, msgs=messages, tokenizer=self.tokenizer)
        if isinstance(raw, tuple):
            raw = raw[0]
        raw_text = str(raw)
        try:
            result = parse_minicpm_json(raw_text)
        except Exception:
            logger.exception(
                "MiniCPM JSON parse failed raw=%s",
                raw_text[-1200:] if self.settings.log_transcripts else "<redacted>",
            )
            raise
        logger.info(
            "MiniCPM interpret completed latency_ms=%s language=%s confidence=%.3f text_len=%s reason=%s raw=%s",
            int((perf_counter() - started) * 1000),
            result.language,
            result.confidence,
            len(result.text),
            result.reason,
            raw_text[-1200:] if self.settings.log_transcripts else "<redacted>",
        )
        return result


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


def _candidate_summary(candidates: list[LipReadingCandidate], *, include_text: bool) -> list[dict[str, Any]]:
    summary = []
    for candidate in candidates:
        item: dict[str, Any] = {
            "model": candidate.model,
            "language": candidate.language,
            "confidence": candidate.confidence,
            "latencyMs": candidate.latencyMs,
            "textLen": len(candidate.text),
        }
        if include_text:
            item["text"] = candidate.text
        summary.append(item)
    return summary


class _quiet_minicpm_model_load:
    def __enter__(self) -> None:
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r".*image_processor_class argument is deprecated.*",
            category=FutureWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Using a slow image processor as `use_fast` is unset.*",
        )
        try:
            from transformers.utils import logging as transformers_logging
        except Exception:
            self._transformers_logging = None
            self._previous_verbosity = None
        else:
            self._transformers_logging = transformers_logging
            self._previous_verbosity = transformers_logging.get_verbosity()
            transformers_logging.set_verbosity_error()

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self._transformers_logging is not None and self._previous_verbosity is not None:
            self._transformers_logging.set_verbosity(self._previous_verbosity)
        self._warnings.__exit__(exc_type, exc_value, traceback)
