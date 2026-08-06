import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from command.dataset import build_dataset_manifests


def save_sample(root: Path, sample_id: str, phrase: str, intent: str, value: int):
    sample = root / "global" / intent / sample_id
    sample.mkdir(parents=True)
    metadata = {
        "sampleId": sample_id,
        "phrase": phrase,
        "intent": intent,
        "language": "zh",
    }
    (sample / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
    )
    np.save(sample / "mouth_roi.npy", np.full((8, 96, 96), value, dtype=np.uint8))
    return sample / "metadata.json"


def test_catalog_corrects_source_intent_without_mutating_metadata(tmp_path):
    source = save_sample(tmp_path, "meal-1", "你吃饭了吗？", "LIGHT_ON", 7)
    original = source.read_bytes()
    inventory = build_dataset_manifests(
        tmp_path,
        Path("command/phrase_catalog.json"),
        tmp_path / "out",
        allow_small_dataset=True,
        seed=17,
    )
    records = [
        json.loads(line)
        for line in (tmp_path / "out/train.jsonl").read_text().splitlines()
    ]
    records += [
        json.loads(line)
        for line in (tmp_path / "out/calibration-known.jsonl").read_text().splitlines()
    ]
    records += [
        json.loads(line)
        for line in (tmp_path / "out/evaluation-known.jsonl").read_text().splitlines()
    ]
    meal = next(record for record in records if record["sample_id"] == "meal-1")
    assert meal["phrase_id"] == "zh_chat_meal"
    assert meal["intent"] == "CHAT_OTHER"
    assert meal["source_intent"] == "LIGHT_ON"
    assert source.read_bytes() == original
    assert inventory["intentMismatches"] == 1


def test_duplicate_mouth_arrays_are_never_split_across_partitions(tmp_path):
    save_sample(tmp_path, "a", "你好，请帮我打开灯", "LIGHT_ON", 1)
    save_sample(tmp_path, "b", "你好，请帮我打开灯", "LIGHT_ON", 1)
    inventory = build_dataset_manifests(
        tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17
    )
    assert inventory["duplicates"] == 1


def test_later_sample_id_duplicate_is_rejected_even_if_first_record_is_excluded(
    tmp_path,
):
    save_sample(tmp_path, "duplicate", "未注册短语", "LIGHT_ON", 1)
    sample = tmp_path / "global" / "second-source" / "second"
    sample.mkdir(parents=True)
    (sample / "metadata.json").write_text(
        json.dumps(
            {
                "sampleId": "duplicate",
                "phrase": "你好，请帮我打开灯",
                "intent": "LIGHT_ON",
                "language": "zh",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    np.save(sample / "mouth_roi.npy", np.full((8, 96, 96), 2, dtype=np.uint8))

    inventory = build_dataset_manifests(
        tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17
    )

    known_records = "".join(
        (tmp_path / "out" / name).read_text()
        for name in ("train.jsonl", "calibration-known.jsonl", "evaluation-known.jsonl")
    )
    assert '"sample_id": "duplicate"' not in known_records
    assert inventory["duplicates"] == 1


def test_official_gate_rejects_too_few_samples(tmp_path):
    save_sample(tmp_path, "a", "你好，请帮我打开灯", "LIGHT_ON", 1)
    with pytest.raises(ValueError, match="15 independent takes"):
        build_dataset_manifests(
            tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", False, 17
        )


def test_explicit_unknown_is_not_a_training_class(tmp_path):
    save_sample(tmp_path, "u", "今天天气不错", "UNKNOWN", 3)
    build_dataset_manifests(
        tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17
    )
    assert '"sample_id": "u"' not in (tmp_path / "out/train.jsonl").read_text()
    unknown_text = (tmp_path / "out/evaluation-unknown.jsonl").read_text()
    unknown_text += (tmp_path / "out/calibration-unknown.jsonl").read_text()
    assert '"sample_id": "u"' in unknown_text


def test_inventory_includes_per_phrase_counts_and_split_membership(tmp_path):
    for index in range(3):
        save_sample(tmp_path, f"light-{index}", "你好，请帮我打开灯", "LIGHT_ON", index)

    inventory = build_dataset_manifests(
        tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17
    )

    assert inventory["counts"]["byPhrase"] == {
        "zh_light_on_hello": 3,
        "zh_chat_meal": 0,
        "en_light_on_hello": 0,
        "en_chat_meal": 0,
    }
    membership = inventory["splitMembership"]
    recorded_ids = set().union(*map(set, membership.values()))
    assert recorded_ids == {"light-0", "light-1", "light-2"}


def test_manifest_cli_builds_small_dataset_artifacts(tmp_path):
    save_sample(tmp_path, "light", "你好，请帮我打开灯", "LIGHT_ON", 1)
    save_sample(tmp_path, "unknown", "今天天气不错", "UNKNOWN", 2)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "artifacts"
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_command_manifest.py"),
            "--profile-root",
            str(tmp_path),
            "--catalog",
            str(project_root / "command" / "phrase_catalog.json"),
            "--output-dir",
            str(output_dir),
            "--seed",
            "29",
            "--allow-small-dataset",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    for filename in (
        "train.jsonl",
        "calibration-known.jsonl",
        "evaluation-known.jsonl",
        "calibration-unknown.jsonl",
        "evaluation-unknown.jsonl",
        "inventory.json",
    ):
        assert (output_dir / filename).is_file()
    inventory = json.loads((output_dir / "inventory.json").read_text(encoding="utf-8"))
    assert inventory["evidentiary"] is False
    assert inventory["seed"] == 29
    assert json.loads(result.stdout)["evidentiary"] is False
