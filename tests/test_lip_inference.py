import numpy as np

from lip.base import MouthFrame, MouthWindow
from lip.fake import FakeLipReader
from lip.inference import LipInferenceEngine


def _window() -> MouthWindow:
    frames = tuple(
        MouthFrame(sequence=i, received_at_ms=i * 40, image=np.zeros((96, 96), dtype=np.uint8))
        for i in range(1, 76)
    )
    return MouthWindow(session_id="s1", start_sequence=1, end_sequence=75, frames=frames)


def test_dual_engine_returns_english_and_chinese_candidates():
    engine = LipInferenceEngine(
        [
            FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
            FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
        ]
    )
    result = engine.predict(_window())
    assert [candidate.model for candidate in result.candidates] == ["avhubert", "cmlr"]
    assert result.degradedModels == []


def test_single_reader_failure_is_degraded_not_fatal():
    engine = LipInferenceEngine(
        [
            FakeLipReader(model="avhubert", language="en", text="ignored", confidence=0.1, fail=True),
            FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
        ]
    )
    result = engine.predict(_window())
    assert [candidate.model for candidate in result.candidates] == ["cmlr"]
    assert result.degradedModels == ["avhubert"]
