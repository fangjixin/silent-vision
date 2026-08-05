from __future__ import annotations

import math
from hashlib import sha256

import numpy as np
import pytest

from command.checkpoint import validate_phrase_checkpoint_schema


def metadata_payload() -> dict:
    return {
        "schemaVersion": "silent-vision.fixed-phrase.v1",
        "modelState": {},
        "phraseIds": ["zh_light_on_hello", "zh_chat_meal"],
        "phraseCatalog": [
            {
                "phraseId": "zh_light_on_hello",
                "text": "你好，请帮我打开灯",
                "language": "zh",
                "intent": "LIGHT_ON",
                "enabled": True,
            },
            {
                "phraseId": "zh_chat_meal",
                "text": "你吃饭了吗？",
                "language": "zh",
                "intent": "CHAT_OTHER",
                "enabled": True,
            },
        ],
        "featureConfig": {"fps": 25, "height": 96, "width": 96, "downsample": 16},
        "modelConfig": {"embeddingDim": 64, "parameterCap": 150000},
        "decisionThresholds": {
            "minProbability": 0.80,
            "maxCosineDistance": {"zh_light_on_hello": 0.20, "zh_chat_meal": 0.20},
        },
        "classCentroids": np.zeros((2, 64), dtype=np.float32),
        "trainingSummary": {"seed": 17, "evidentiary": False},
    }


def test_checkpoint_schema_validation_runs_without_torch():
    validated = validate_phrase_checkpoint_schema(metadata_payload())
    assert validated.phrase_ids == ("zh_light_on_hello", "zh_chat_meal")
    assert tuple(validated.centroids.shape) == (2, 64)


def test_checkpoint_phrase_ids_must_match_enabled_catalog_order():
    payload = metadata_payload()
    payload["phraseIds"].reverse()
    with pytest.raises(ValueError, match="enabled catalog order"):
        validate_phrase_checkpoint_schema(payload)


def test_checkpoint_thresholds_must_be_finite():
    payload = metadata_payload()
    payload["decisionThresholds"]["minProbability"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        validate_phrase_checkpoint_schema(payload)


def test_checkpoint_thresholds_must_be_numeric():
    payload = metadata_payload()
    payload["decisionThresholds"]["minProbability"] = "0.80"
    with pytest.raises(ValueError, match="numeric"):
        validate_phrase_checkpoint_schema(payload)


def test_checkpoint_requires_fixed_parameter_cap():
    payload = metadata_payload()
    payload["modelConfig"]["parameterCap"] = 1_000_000
    with pytest.raises(ValueError, match="150000"):
        validate_phrase_checkpoint_schema(payload)


def valid_payload(model):
    torch = pytest.importorskip("torch")
    payload = metadata_payload()
    payload["modelState"] = model.state_dict()
    payload["classCentroids"] = torch.nn.functional.normalize(torch.rand(2, 64), dim=1)
    return payload


def test_checkpoint_builds_dynamic_two_phrase_head(tmp_path):
    pytest.importorskip("torch")
    from command.checkpoint import load_phrase_checkpoint, save_phrase_checkpoint
    from command.model import build_fixed_phrase_model

    model = build_fixed_phrase_model(2)
    path = tmp_path / "phrase.pt"
    payload = valid_payload(model)
    digest = save_phrase_checkpoint(path, payload)
    loaded = load_phrase_checkpoint(path, "cpu")
    assert digest == sha256(path.read_bytes()).hexdigest()
    assert loaded.phrase_ids == ("zh_light_on_hello", "zh_chat_meal")
    assert loaded.centroids.shape == (2, 64)
    assert loaded.model.classifier.out_features == 2


def test_legacy_intent_checkpoint_has_migration_error(tmp_path):
    torch = pytest.importorskip("torch")
    from command.checkpoint import load_phrase_checkpoint

    path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "labels": ["LIGHT_ON", "LIGHT_OFF"]}, path)
    with pytest.raises(ValueError, match="legacy intent-only checkpoint"):
        load_phrase_checkpoint(path, "cpu")


def test_head_shape_mismatch_is_rejected(tmp_path):
    torch = pytest.importorskip("torch")
    from command.checkpoint import load_phrase_checkpoint
    from command.model import build_fixed_phrase_model

    payload = valid_payload(build_fixed_phrase_model(2))
    payload["phraseIds"].append("third")
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="classifier head"):
        load_phrase_checkpoint(tmp_path / "bad.pt", "cpu")


def test_save_rejects_incomplete_state_without_replacing_existing_file(tmp_path):
    pytest.importorskip("torch")
    from command.checkpoint import save_phrase_checkpoint
    from command.model import build_fixed_phrase_model

    path = tmp_path / "phrase.pt"
    path.write_bytes(b"existing checkpoint")
    payload = valid_payload(build_fixed_phrase_model(2))
    payload["modelState"].pop("projection.weight")
    with pytest.raises(ValueError, match="modelState"):
        save_phrase_checkpoint(path, payload)
    assert path.read_bytes() == b"existing checkpoint"
