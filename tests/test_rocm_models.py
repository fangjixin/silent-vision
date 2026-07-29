import json
from pathlib import Path

import pytest

from backend.config import Settings
from lip.avhubert import AVHuBERTLipReader
from lip.cmlr import CMLRLipReader
from tests.test_lip_inference import _window

pytestmark = [pytest.mark.rocm, pytest.mark.model_integration]


def test_avhubert_checkpoint_exists_before_loading():
    settings = Settings(model_backend="real")
    assert settings.avhubert_checkpoint.exists(), settings.avhubert_checkpoint


def test_cmlr_checkpoint_exists_before_loading():
    settings = Settings(model_backend="real")
    assert settings.cmlr_checkpoint.exists(), settings.cmlr_checkpoint


def test_test_media_manifest_schema_example_is_valid():
    manifest = json.loads(Path("tests/media/manifest.example.json").read_text())
    assert manifest["version"] == 1
    for item in manifest["samples"]:
        assert set(item) == {"path", "language", "expected_text", "license_source"}
        assert item["language"] in {"zh", "en"}


def test_avhubert_reader_smoke_predicts_english_candidate():
    reader = AVHuBERTLipReader(Settings(model_backend="real"))
    candidate = reader.predict(_window())
    assert candidate.model == "avhubert"
    assert candidate.language == "en"
    assert isinstance(candidate.text, str)


def test_cmlr_reader_smoke_predicts_chinese_candidate():
    reader = CMLRLipReader(Settings(model_backend="real"))
    candidate = reader.predict(_window())
    assert candidate.model == "cmlr"
    assert candidate.language == "zh"
    assert isinstance(candidate.text, str)
