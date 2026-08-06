from __future__ import annotations

import inspect
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from command.evaluation import (
    EvaluationRecord,
    _evaluate_dataset,
    _validate_partition_records,
    build_evaluation_report,
    evaluate_checkpoint,
)


def test_evaluation_manifest_records_require_a_supported_language():
    # Catches a regression that evaluates a clip without the language needed to
    # select its eligible phrase classes.
    _validate_partition_records(
        [{"phrase_id": "zh_light_on", "language": "zh"}], known=True
    )
    _validate_partition_records(
        [{"phrase_id": None, "language": "en"}], known=False
    )

    with pytest.raises(ValueError, match="language"):
        _validate_partition_records([{"phrase_id": "zh_light_on"}], known=True)
    with pytest.raises(ValueError, match="language"):
        _validate_partition_records(
            [{"phrase_id": None, "language": "unknown"}], known=False
        )


def test_evaluation_passes_manifest_language_to_the_classifier():
    # Catches a regression that records language in the manifest but omits it
    # from the classifier call that produces the decision.
    class Frames:
        def numpy(self):
            return "frames"

    class Dataset:
        def __init__(self):
            self.records = [
                {
                    "sample_id": "sample-1",
                    "phrase_id": "zh_light_on",
                    "language": "zh",
                }
            ]

        def __getitem__(self, index):
            return Frames(), 0

    received = []

    class Backend:
        def predict(self, frames, language, metadata):
            received.append((frames, language, metadata))
            return SimpleNamespace(
                accepted=True,
                confidence=0.99,
                metadata={"predictedPhraseId": "zh_light_on", "openSetDistance": 0.01},
            )

    records = _evaluate_dataset(Backend(), Dataset(), Path("known.jsonl"))

    assert records[0].predicted_phrase_id == "zh_light_on"
    assert received == [
        (
            "frames",
            "zh",
            {"manifest": "known.jsonl", "sampleId": "sample-1"},
        )
    ]


def test_evaluation_metrics_keep_acceptance_and_accuracy_separate():
    known = [
        EvaluationRecord("a", "a", True, 0.9, 0.1),
        EvaluationRecord("a", "b", True, 0.9, 0.1),
        EvaluationRecord("b", "b", False, 0.4, 0.3),
    ]
    unknown = [
        EvaluationRecord(None, "a", False, 0.5, 0.4),
        EvaluationRecord(None, "b", True, 0.9, 0.1),
    ]

    report = build_evaluation_report(
        known,
        unknown,
        {"a": "LIGHT_ON", "b": "CHAT_OTHER"},
        {"thresholdSource": "checkpoint"},
    )

    assert report["phraseAccuracy"] == pytest.approx(2 / 3)
    assert report["mappedIntentAccuracy"] == pytest.approx(2 / 3)
    assert report["knownAcceptanceRate"] == pytest.approx(2 / 3)
    assert report["acceptedPhraseAccuracy"] == pytest.approx(1 / 2)
    assert report["acceptedPrecision"] == pytest.approx(1 / 2)
    assert report["unknownFalseAcceptRate"] == pytest.approx(1 / 2)
    assert report["unknownRejectionRate"] == pytest.approx(1 / 2)
    assert report["confusionMatrix"]["a"] == {"a": 1, "b": 1}
    assert report["perPhrase"]["b"]["knownTotal"] == 1


def test_evaluation_report_exposes_every_rate_denominator_and_rejected_confusion():
    report = build_evaluation_report(
        [EvaluationRecord("a", "a", False, 0.7, 0.3)],
        [EvaluationRecord(None, "a", False, 0.6, 0.4)],
        {"a": "LIGHT_ON"},
        {"thresholdSource": "checkpoint"},
    )

    assert report["acceptedPhraseAccuracy"] is None
    assert report["acceptedPrecision"] is None
    assert report["rawCounts"] == {
        "phraseAccuracy": {"numerator": 1, "denominator": 1},
        "mappedIntentAccuracy": {"numerator": 1, "denominator": 1},
        "knownAcceptanceRate": {"numerator": 0, "denominator": 1},
        "acceptedPhraseAccuracy": {"numerator": 0, "denominator": 0},
        "acceptedPrecision": {"numerator": 0, "denominator": 0},
        "unknownFalseAcceptRate": {"numerator": 0, "denominator": 1},
        "unknownRejectionRate": {"numerator": 1, "denominator": 1},
    }
    assert report["confusionMatrix"] == {
        "a": {"UNKNOWN": 1},
        "UNKNOWN": {"UNKNOWN": 1},
    }


def test_evaluate_checkpoint_signature_has_no_calibration_inputs():
    parameters = inspect.signature(evaluate_checkpoint).parameters

    assert list(parameters) == [
        "checkpoint_path",
        "known_manifest",
        "unknown_manifest",
        "output_path",
        "probability_override",
        "distance_override",
    ]
    assert not any("calibration" in name for name in parameters)


def test_evaluate_checkpoint_rejects_non_final_partition_names_before_gpu_work(
    tmp_path,
):
    with pytest.raises(ValueError, match="evaluation-known.jsonl"):
        evaluate_checkpoint(
            checkpoint_path=tmp_path / "missing.pt",
            known_manifest=tmp_path / "calibration-known.jsonl",
            unknown_manifest=tmp_path / "evaluation-unknown.jsonl",
            output_path=tmp_path / "report.json",
            probability_override=None,
            distance_override=None,
        )


@pytest.mark.parametrize(
    ("script_name", "required_options"),
    [
        (
            "validate_command_classifier.py",
            ("--checkpoint", "--known-manifest", "--unknown-manifest", "--output"),
        ),
        ("infer_command_clip.py", ("--checkpoint", "--mouth-roi", "--language")),
    ],
)
def test_phrase_cli_help_runs_without_torch(script_name, required_options):
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / script_name), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert all(option in result.stdout for option in required_options)
    assert "calibration" not in result.stdout.lower()
    assert "--manifest " not in result.stdout
    assert "--threshold" not in result.stdout
    assert "--margin" not in result.stdout


@pytest.fixture(scope="module")
def rocm_evaluation_artifacts(tmp_path_factory):
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        pytest.skip("requires a Radeon ROCm/HIP device")

    from command.checkpoint import save_phrase_checkpoint
    from command.model import build_fixed_phrase_model

    root = tmp_path_factory.mktemp("phrase-evaluation")
    clip = np.zeros((12, 96, 96), dtype=np.uint8)
    clip[:, 32:64, 24:72] = np.arange(12, dtype=np.uint8)[:, None, None] * 10
    mouth_roi = root / "mouth-roi.npy"
    np.save(mouth_roi, clip)

    model = build_fixed_phrase_model(2).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([10.0, -10.0]))
        _, embedding = model(torch.from_numpy(clip).unsqueeze(0))
    checkpoint = root / "fixed-phrase.pt"
    save_phrase_checkpoint(
        checkpoint,
        {
            "schemaVersion": "silent-vision.fixed-phrase.v2",
            "modelState": model.state_dict(),
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
            "featureConfig": {
                "fps": 25,
                "height": 96,
                "width": 96,
                "downsample": 16,
            },
            "modelConfig": {"embeddingDim": 64, "parameterCap": 150000},
            "decisionPolicy": {
                "languageSelectionRequired": True,
                "probabilityNormalization": "selected-language-softmax",
            },
            "decisionThresholds": {
                "minProbability": 0.99,
                "maxCosineDistance": {
                    "zh_light_on_hello": 0.05,
                    "zh_chat_meal": 0.05,
                },
            },
            "classCentroids": torch.cat([embedding, -embedding], dim=0),
            "trainingSummary": {"seed": 17, "evidentiary": False},
        },
    )
    record_digest = sha256(mouth_roi.read_bytes()).hexdigest()
    known_manifest = root / "evaluation-known.jsonl"
    unknown_manifest = root / "evaluation-unknown.jsonl"
    known_manifest.write_text(
        json.dumps(
            {
                "phrase_id": "zh_light_on_hello",
                "language": "zh",
                "mouth_roi_npy": str(mouth_roi),
                "sha256": record_digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    unknown_manifest.write_text(
        json.dumps(
            {
                "phrase_id": None,
                "language": "zh",
                "mouth_roi_npy": str(mouth_roi),
                "sha256": record_digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "known_manifest": known_manifest,
        "unknown_manifest": unknown_manifest,
        "mouth_roi": mouth_roi,
        "root": root,
    }


def test_checkpoint_evaluation_uses_resolved_checkpoint_thresholds_and_hashes(
    rocm_evaluation_artifacts,
):
    paths = rocm_evaluation_artifacts
    output = paths["root"] / "evaluation.json"

    report = evaluate_checkpoint(
        checkpoint_path=paths["checkpoint"],
        known_manifest=paths["known_manifest"],
        unknown_manifest=paths["unknown_manifest"],
        output_path=output,
        probability_override=None,
        distance_override=None,
    )

    assert report["thresholdSource"] == "checkpoint"
    assert report["effectiveThresholds"] == {
        "minProbability": 0.99,
        "maxCosineDistance": {
            "zh_light_on_hello": 0.05,
            "zh_chat_meal": 0.05,
        },
    }
    assert (
        report["checkpointSha256"]
        == sha256(paths["checkpoint"].read_bytes()).hexdigest()
    )
    assert report["manifestSha256"] == {
        "evaluation-known.jsonl": sha256(
            paths["known_manifest"].read_bytes()
        ).hexdigest(),
        "evaluation-unknown.jsonl": sha256(
            paths["unknown_manifest"].read_bytes()
        ).hexdigest(),
    }
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_single_clip_inference_prints_phrase_runtime_contract(
    rocm_evaluation_artifacts,
):
    paths = rocm_evaluation_artifacts
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/infer_command_clip.py"),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--mouth-roi",
            str(paths["mouth_roi"]),
            "--language",
            "zh",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout)
    assert output["intent"] == "LIGHT_ON"
    assert output["accepted"] is True
    assert output["confidence"] > 0.99
    assert output["margin"] > 0.99
    assert output["phraseId"] == "zh_light_on_hello"
    assert output["matchedPhrase"] == "你好，请帮我打开灯"
    assert output["backend"] == "torch"
    assert output["device"] == "cuda:0"
    assert output["thresholdSource"] == "checkpoint"
    assert output["openSetDistance"] == pytest.approx(0.0, abs=1e-5)
