from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from command.catalog import (
    PhraseCatalog,
    PhraseEntry,
    catalog_sha256,
    load_phrase_catalog,
    normalize_phrase,
)
from command.language import validate_recognition_language


@dataclass(frozen=True)
class ManifestSample:
    sample_id: str
    phrase_id: str | None
    text: str
    language: str
    intent: str
    source_intent: str
    mouth_roi_npy: str
    source_metadata: str
    sha256: str


def classify_source(metadata: dict, catalog: PhraseCatalog):
    source_intent = str(metadata.get("intent", "UNKNOWN"))
    if source_intent == "UNKNOWN":
        return "unknown", None
    normalized = normalize_phrase(str(metadata.get("phrase", "")))
    by_text = {normalize_phrase(entry.text): entry for entry in catalog.entries}
    entry = by_text.get(normalized)
    if entry is None:
        return "excluded", "phrase text is not registered"
    return "known", entry


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sort_samples(samples: list[ManifestSample], seed: int) -> list[ManifestSample]:
    return sorted(
        samples,
        key=lambda sample: hashlib.sha256(
            f"{seed}:{sample.sample_id}".encode()
        ).hexdigest(),
    )


def split_known(samples: list[ManifestSample], allow_small: bool):
    if not allow_small and len(samples) < 15:
        raise ValueError("official evidence requires 15 independent takes per phrase")
    if allow_small:
        if len(samples) == 1:
            return samples, [], []
        if len(samples) == 2:
            return samples[:1], [], samples[1:]
        return samples[2:], samples[1:2], samples[:1]
    return samples[5:], samples[3:5], samples[:3]


def split_unknown(samples: list[ManifestSample], allow_small: bool):
    if not allow_small and len(samples) < 15:
        raise ValueError("official evidence requires 15 unrelated clips")
    if allow_small:
        if len(samples) < 2:
            return [], samples
        return samples[:1], samples[1:]
    return samples[:5], samples[5:]


def _unknown_counts_by_language(samples: list[ManifestSample]) -> dict[str, int]:
    return {
        language: sum(sample.language == language for sample in samples)
        for language in ("zh", "en")
    }


def _manifest_sample(
    sample_dir: Path,
    metadata_path: Path,
    mouth_roi_path: Path,
    metadata: dict,
    entry: PhraseEntry | None,
    sha256: str,
) -> ManifestSample:
    source_intent = str(metadata.get("intent", "UNKNOWN"))
    return ManifestSample(
        sample_id=str(metadata.get("sampleId") or sample_dir.name),
        phrase_id=entry.phrase_id if entry else None,
        text=entry.text if entry else str(metadata.get("phrase", "")),
        language=(
            entry.language
            if entry
            else validate_recognition_language(metadata.get("language"))
        ),
        intent=entry.intent.value if entry else "UNKNOWN",
        source_intent=source_intent,
        mouth_roi_npy=str(mouth_roi_path.resolve()),
        source_metadata=str(metadata_path.resolve()),
        sha256=sha256,
    )


def _write_jsonl(path: Path, samples: list[ManifestSample]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(
                json.dumps(asdict(sample), ensure_ascii=False, sort_keys=True) + "\n"
            )
    return _sha256_file(path)


def build_dataset_manifests(
    profile_root: Path,
    catalog_path: Path,
    output_dir: Path,
    allow_small_dataset: bool,
    seed: int,
) -> dict:
    catalog = load_phrase_catalog(catalog_path)
    root = profile_root.resolve()
    known_by_phrase: dict[str, list[ManifestSample]] = {
        entry.phrase_id: [] for entry in catalog.entries
    }
    unknown: list[ManifestSample] = []
    exclusions: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    duplicates = 0
    intent_mismatches = 0

    recordings_root = root / "global"
    if recordings_root.exists():
        sample_dirs = sorted(
            path
            for source_dir in recordings_root.iterdir()
            if source_dir.is_dir()
            for path in source_dir.iterdir()
            if path.is_dir()
        )
    else:
        sample_dirs = []

    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        mouth_roi_path = sample_dir / "mouth_roi.npy"
        if not metadata_path.is_file() or not mouth_roi_path.is_file():
            raise ValueError(
                f"recording requires metadata.json and mouth_roi.npy: {sample_dir.resolve()}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(  # noqa: TRY004
                f"recording metadata must be an object: {metadata_path.resolve()}"
            )
        sample_id = str(metadata.get("sampleId") or sample_dir.name)
        mouth_roi_sha256 = _sha256_file(mouth_roi_path)
        if sample_id in seen_ids or mouth_roi_sha256 in seen_hashes:
            duplicates += 1
            continue

        seen_ids.add(sample_id)
        seen_hashes.add(mouth_roi_sha256)

        kind, entry_or_reason = classify_source(metadata, catalog)
        if kind == "excluded":
            exclusions.append(
                {
                    "sourceMetadata": str(metadata_path.resolve()),
                    "reason": str(entry_or_reason),
                }
            )
            continue

        if kind == "known":
            entry = entry_or_reason
            assert isinstance(entry, PhraseEntry)
            sample = _manifest_sample(
                sample_dir,
                metadata_path,
                mouth_roi_path,
                metadata,
                entry,
                mouth_roi_sha256,
            )
            known_by_phrase[entry.phrase_id].append(sample)
            if sample.source_intent != sample.intent:
                intent_mismatches += 1
        else:
            unknown.append(
                _manifest_sample(
                    sample_dir,
                    metadata_path,
                    mouth_roi_path,
                    metadata,
                    None,
                    mouth_roi_sha256,
                )
            )

    train: list[ManifestSample] = []
    calibration_known: list[ManifestSample] = []
    evaluation_known: list[ManifestSample] = []
    for phrase_samples in known_by_phrase.values():
        phrase_train, phrase_calibration, phrase_evaluation = split_known(
            _sort_samples(phrase_samples, seed), allow_small_dataset
        )
        train.extend(phrase_train)
        calibration_known.extend(phrase_calibration)
        evaluation_known.extend(phrase_evaluation)

    unknown_by_language = _unknown_counts_by_language(unknown)
    if not allow_small_dataset and not all(unknown_by_language.values()):
        raise ValueError("official evidence requires 15 unrelated clips in both zh and en")

    calibration_unknown, evaluation_unknown = split_unknown(
        _sort_samples(unknown, seed), allow_small_dataset
    )
    manifests = {
        "train.jsonl": train,
        "calibration-known.jsonl": calibration_known,
        "evaluation-known.jsonl": evaluation_known,
        "calibration-unknown.jsonl": calibration_unknown,
        "evaluation-unknown.jsonl": evaluation_unknown,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_sha256 = {
        name: _write_jsonl(output_dir / name, samples)
        for name, samples in manifests.items()
    }
    inventory = {
        "evidentiary": not allow_small_dataset,
        "catalogSha256": catalog_sha256(catalog),
        "seed": seed,
        "counts": {
            "known": sum(len(samples) for samples in known_by_phrase.values()),
            "unknown": len(unknown),
            "unknownByLanguage": unknown_by_language,
            "excluded": len(exclusions),
            "byPhrase": {
                phrase_id: len(samples)
                for phrase_id, samples in known_by_phrase.items()
            },
            **{
                name.removesuffix(".jsonl"): len(samples)
                for name, samples in manifests.items()
            },
        },
        "exclusions": exclusions,
        "duplicates": duplicates,
        "intentMismatches": intent_mismatches,
        "manifestSha256": manifest_sha256,
        "splitMembership": {
            name: [sample.sample_id for sample in samples]
            for name, samples in manifests.items()
        },
    }
    (output_dir / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory
