from time import perf_counter
from typing import Any

import numpy as np

from backend.config import Settings
from backend.schemas import LipReadingCandidate
from lip.base import MouthWindow


class AVHuBERTLipReader:
    name = "avhubert"
    language = "en"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if not settings.avhubert_checkpoint.exists():
            raise FileNotFoundError(settings.avhubert_checkpoint)
        torch = _import_torch()
        self._torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(settings.avhubert_checkpoint), map_location=self.device)
        self.model.eval()

    def _window_tensor(self, window: MouthWindow) -> Any:
        frames = np.stack([frame.image for frame in window.frames]).astype("float32") / 255.0
        tensor = self._torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        started = perf_counter()
        with self._torch.inference_mode():
            output = self.model(self._window_tensor(window))
        text = ""
        raw_score: float | None = None
        confidence: float | None = None
        if isinstance(output, dict):
            text = str(output.get("text", ""))
            raw = output.get("score")
            raw_score = float(raw) if raw is not None else None
            conf = output.get("confidence")
            confidence = float(conf) if conf is not None else None
        return LipReadingCandidate(
            model="avhubert",
            language="en",
            text=text,
            confidence=confidence,
            rawScore=raw_score,
            latencyMs=int((perf_counter() - started) * 1000),
        )


def _import_torch() -> Any:
    import torch

    return torch
