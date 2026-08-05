from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from command.training import (
    CalibrationRecord,
    ManifestDataset,
    _validate_manifest_hashes,
    calibrate_thresholds,
    compute_class_centroids,
    require_rocm,
    train_phrase_classifier,
)


def _write_manifest_file(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def test_manifest_hashes_are_bound_to_semantic_roles(tmp_path):
    train = tmp_path / "train.jsonl"
    known = tmp_path / "calibration-known.jsonl"
    unknown = tmp_path / "calibration-unknown.jsonl"
    inventory = {
        "manifestSha256": {
            "train.jsonl": _write_manifest_file(train, "train\n"),
            "calibration-known.jsonl": _write_manifest_file(known, "known\n"),
            "calibration-unknown.jsonl": _write_manifest_file(unknown, "unknown\n"),
        }
    }

    with pytest.raises(ValueError, match="train.jsonl"):
        _validate_manifest_hashes(inventory, (known, train, unknown))


def test_evaluation_manifest_cannot_be_used_for_calibration(tmp_path):
    train = tmp_path / "train.jsonl"
    known = tmp_path / "calibration-known.jsonl"
    unknown = tmp_path / "calibration-unknown.jsonl"
    evaluation = tmp_path / "evaluation-known.jsonl"
    inventory = {
        "manifestSha256": {
            "train.jsonl": _write_manifest_file(train, "train\n"),
            "calibration-known.jsonl": _write_manifest_file(known, "same bytes\n"),
            "calibration-unknown.jsonl": _write_manifest_file(unknown, "unknown\n"),
            "evaluation-known.jsonl": _write_manifest_file(evaluation, "same bytes\n"),
        }
    }

    with pytest.raises(ValueError, match="calibration-known.jsonl"):
        _validate_manifest_hashes(inventory, (train, evaluation, unknown))


def test_manifest_dataset_rejects_npy_modified_after_manifest_creation(tmp_path):
    clip = tmp_path / "clip.npy"
    np.save(clip, np.full((2, 96, 96), 1, dtype=np.uint8))
    original_digest = sha256(clip.read_bytes()).hexdigest()
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {"phrase_id": "a", "mouth_roi_npy": str(clip), "sha256": original_digest}
        )
        + "\n",
        encoding="utf-8",
    )
    np.save(clip, np.full((2, 96, 96), 2, dtype=np.uint8))

    with pytest.raises(ValueError, match="mouth ROI SHA-256"):
        ManifestDataset(manifest, {"a": 0})


def test_manifest_dataset_requires_per_sample_sha256(tmp_path):
    clip = tmp_path / "clip.npy"
    np.save(clip, np.full((2, 96, 96), 1, dtype=np.uint8))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps({"phrase_id": "a", "mouth_roi_npy": str(clip)}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires sha256"):
        ManifestDataset(manifest, {"a": 0})


def test_calibration_uses_known_radius_and_limits_unknown_false_accepts():
    known = [
        CalibrationRecord("a", "a", 0.92, 0.08),
        CalibrationRecord("a", "a", 0.88, 0.10),
        CalibrationRecord("b", "b", 0.91, 0.07),
        CalibrationRecord("b", "b", 0.86, 0.11),
    ]
    unknown = [
        CalibrationRecord(None, "a", 0.81, 0.09),
        CalibrationRecord(None, "b", 0.55, 0.30),
    ]

    result = calibrate_thresholds(known, unknown, ("a", "b"))

    assert result["maxCosineDistance"] == {"a": 0.10, "b": 0.11}
    assert result["minProbability"] >= 0.82
    assert result["calibrationUnknownFalseAcceptRate"] == 0.0
    assert result["calibrationTargetMet"] is True


def test_calibration_uses_highest_probability_threshold_when_known_acceptance_ties():
    known = [CalibrationRecord("a", "a", 0.90, 0.10)]
    unknown = [CalibrationRecord(None, "a", 0.49, 0.05)]

    result = calibrate_thresholds(known, unknown, ("a",))

    assert result["minProbability"] == 0.90
    assert result["calibrationKnownCorrectAcceptedCount"] == 1


def test_calibration_without_unknown_clips_uses_non_evidentiary_default():
    known = [CalibrationRecord("a", "a", 0.92, 0.10)]

    result = calibrate_thresholds(known, [], ("a",))

    assert result["minProbability"] == 0.85
    assert result["calibrationTargetMet"] is False
    assert result["evidentiary"] is False


def test_official_calibration_rejects_a_class_without_correct_known_record():
    known = [CalibrationRecord("a", "b", 0.92, 0.10)]

    with pytest.raises(ValueError, match="no correctly classified known-calibration record"):
        calibrate_thresholds(known, [], ("a", "b"))


def test_smoke_calibration_uses_training_distance_fallback_and_records_it():
    training = [
        CalibrationRecord("a", "a", 1.0, 0.10),
        CalibrationRecord("a", "a", 1.0, 0.20),
    ]

    result = calibrate_thresholds(
        [],
        [CalibrationRecord(None, "a", 0.60, 0.15)],
        ("a",),
        evidentiary=False,
        training_records=training,
    )

    assert result["maxCosineDistance"]["a"] == pytest.approx(0.195)
    assert result["radiusFallbackPhraseIds"] == ["a"]
    assert result["evidentiary"] is False


def test_rocm_guard_does_not_accept_cuda_without_hip():
    fake = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        require_rocm(fake)


def test_rocm_guard_requires_an_available_device():
    fake = SimpleNamespace(
        version=SimpleNamespace(hip="6.3"),
        cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0),
    )
    with pytest.raises(RuntimeError, match="cuda:0"):
        require_rocm(fake)


def test_rocm_guard_accepts_hip_cuda_zero():
    fake = SimpleNamespace(
        version=SimpleNamespace(hip="6.3"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    assert require_rocm(fake) == "cuda:0"


def test_training_checks_rocm_before_reading_catalog_or_samples(monkeypatch, tmp_path):
    fake = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)

    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        train_phrase_classifier(
            catalog_path=tmp_path / "missing-catalog.json",
            inventory_path=tmp_path / "missing-inventory.json",
            train_manifest=tmp_path / "missing-train.jsonl",
            calibration_known_manifest=tmp_path / "missing-known.jsonl",
            calibration_unknown_manifest=tmp_path / "missing-unknown.jsonl",
            output_path=tmp_path / "model.pt",
            run_summary_path=tmp_path / "run.json",
            epochs=80,
            seed=17,
        )


def test_training_cli_help_runs_without_torch():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["python3", str(project_root / "scripts/train_command_classifier.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--calibration-known" in result.stdout
    assert "--calibration-unknown" in result.stdout
    assert "--run-summary" in result.stdout


def test_centroids_are_class_means_normalized():
    torch = pytest.importorskip("torch")
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])

    centroids = compute_class_centroids(embeddings, torch.tensor([0, 0, 1]), 2)

    assert centroids.shape == (2, 2)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(2))


def test_centroids_reject_a_class_without_training_embeddings():
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="class 1"):
        compute_class_centroids(torch.tensor([[1.0, 0.0]]), torch.tensor([0]), 2)


def test_manifest_dataset_loads_only_listed_npy_and_repeats_final_frame(tmp_path):
    torch = pytest.importorskip("torch")
    from command.training import pad_clip_batch

    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"
    unlisted = tmp_path / "unlisted.npy"
    np.save(first, np.full((2, 96, 96), 1, dtype=np.uint8))
    np.save(second, np.full((3, 96, 96), 2, dtype=np.uint8))
    np.save(unlisted, np.full((4, 96, 96), 9, dtype=np.uint8))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phrase_id": "a",
                        "mouth_roi_npy": str(first),
                        "sha256": sha256(first.read_bytes()).hexdigest(),
                    }
                ),
                json.dumps(
                    {
                        "phrase_id": "b",
                        "mouth_roi_npy": str(second),
                        "sha256": sha256(second.read_bytes()).hexdigest(),
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = ManifestDataset(manifest, {"a": 0, "b": 1})
    frames, labels = pad_clip_batch([dataset[0], dataset[1]])

    assert frames.shape == (2, 3, 96, 96)
    assert labels.tolist() == [0, 1]
    assert torch.equal(frames[0, -1], frames[0, -2])


def test_seeded_training_augmentation_is_reproducible():
    torch = pytest.importorskip("torch")
    from command.training import augment_clip

    frames = torch.arange(5 * 96 * 96, dtype=torch.float32).reshape(5, 96, 96)
    first = augment_clip(frames, torch.Generator().manual_seed(17))
    second = augment_clip(frames, torch.Generator().manual_seed(17))

    assert torch.equal(first, second)
    assert first.shape[0] in {4, 5, 6}


def test_rocm_training_writes_checkpoint_and_external_run_summary(tmp_path):
    torch = pytest.importorskip("torch")
    if not getattr(torch.version, "hip", None) or not torch.cuda.is_available():
        pytest.skip("requires a Radeon ROCm/HIP device")
    from command.catalog import catalog_sha256, load_phrase_catalog

    project_root = Path(__file__).resolve().parents[1]
    catalog_path = project_root / "command/phrase_catalog.json"
    catalog = load_phrase_catalog(catalog_path)
    manifests = {}
    for manifest_name in (
        "train.jsonl",
        "calibration-known.jsonl",
        "calibration-unknown.jsonl",
    ):
        manifests[manifest_name] = tmp_path / manifest_name

    train_records = []
    calibration_records = []
    for index, entry in enumerate(catalog.entries):
        train_clip = tmp_path / f"train-{index}.npy"
        calibration_clip = tmp_path / f"calibration-{index}.npy"
        np.save(train_clip, np.full((3, 96, 96), 20 + index * 100, dtype=np.uint8))
        np.save(calibration_clip, np.full((3, 96, 96), 30 + index * 100, dtype=np.uint8))
        train_records.append(
            {
                "phrase_id": entry.phrase_id,
                "mouth_roi_npy": str(train_clip),
                "sha256": sha256(train_clip.read_bytes()).hexdigest(),
            }
        )
        calibration_records.append(
            {
                "phrase_id": entry.phrase_id,
                "mouth_roi_npy": str(calibration_clip),
                "sha256": sha256(calibration_clip.read_bytes()).hexdigest(),
            }
        )
    manifests["train.jsonl"].write_text(
        "".join(json.dumps(record) + "\n" for record in train_records), encoding="utf-8"
    )
    manifests["calibration-known.jsonl"].write_text(
        "".join(json.dumps(record) + "\n" for record in calibration_records), encoding="utf-8"
    )
    manifests["calibration-unknown.jsonl"].write_text("", encoding="utf-8")
    manifest_hashes = {
        name: sha256(path.read_bytes()).hexdigest() for name, path in manifests.items()
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "evidentiary": False,
                "seed": 17,
                "catalogSha256": catalog_sha256(catalog),
                "manifestSha256": manifest_hashes,
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "fixed-phrase.pt"
    summary_path = tmp_path / "fixed-phrase-run.json"

    result = train_phrase_classifier(
        catalog_path=catalog_path,
        inventory_path=inventory_path,
        train_manifest=manifests["train.jsonl"],
        calibration_known_manifest=manifests["calibration-known.jsonl"],
        calibration_unknown_manifest=manifests["calibration-unknown.jsonl"],
        output_path=checkpoint_path,
        run_summary_path=summary_path,
        epochs=1,
        seed=17,
    )

    assert result["checkpointSha256"] == sha256(checkpoint_path.read_bytes()).hexdigest()
    assert result["device"] == "cuda:0"
    assert result["epochs"] == 1
    assert result["evidentiary"] is False
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["decisionThresholds"]["minProbability"] == 0.85
    assert tuple(checkpoint["classCentroids"].shape) == (len(catalog.entries), 64)
