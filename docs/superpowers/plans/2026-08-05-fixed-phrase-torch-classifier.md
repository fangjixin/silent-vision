# Fixed-Phrase Torch Classifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a personalized fixed-phrase visual classifier that returns exact catalog text, trains and runs on AMD ROCm, and rejects unrelated clips with calibrated probability and embedding-distance gates.

**Architecture:** A versioned phrase catalog owns label-to-text-to-intent mapping. Immutable recordings are inventoried into disjoint training, threshold-calibration, and final-evaluation manifests; a sub-150k-parameter Torch temporal-convolution model learns phrase classes and centroids; one shared decision function is used by runtime and evaluation. The contest path fails closed when ROCm or a valid phrase checkpoint is unavailable.

**Tech Stack:** Python 3.11, NumPy, PyTorch 2.9 ROCm/HIP, FastAPI WebSocket API, Pydantic settings, pytest, Bash deployment scripts.

## Global Constraints

- Phrase classes are stable `phraseId` values; `UNKNOWN` is a rejection result and is never a trained class.
- Runtime display text is copied exactly from the checkpoint catalog and is never generated from logits.
- The initial catalog contains `你好，请帮我打开灯` → `LIGHT_ON` and `你吃饭了吗？` → `CHAT_OTHER`.
- Source `metadata.json` and `mouth_roi.npy` recordings remain immutable.
- The catalog, not the source intent field, owns the corrected label; the original value is retained as `sourceIntent`.
- Scan the global profile only and deduplicate by both sample identifier and mouth-array SHA-256.
- Official evidence requires at least 15 independent takes per phrase: 10 training, 2 calibration, and 3 final evaluation.
- Official evidence requires at least 15 unrelated clips: 5 calibration and 10 final evaluation.
- `--allow-small-dataset` artifacts are marked non-evidentiary and cannot support accuracy or rejection claims.
- Model input is 25 FPS, 96 × 96 grayscale mouth ROI; training and inference use the same Torch feature path.
- The model uses a 16 × 16 appearance map, adjacent-frame difference map, 64-dimensional projection, two depthwise-separable temporal blocks, attentive pooling, normalized embedding, and a phrase head.
- The complete trainable model must contain fewer than 150,000 parameters.
- Training and production Torch inference require non-empty `torch.version.hip`, `torch.cuda.is_available() == True`, and a visible `cuda:0` device; there is no CPU classifier fallback.
- CPU video decoding, face detection, and mouth cropping remain CPU preprocessing and must be described as such.
- Acceptance requires both the checkpoint probability threshold and the predicted phrase's maximum cosine-distance threshold.
- Top-1/top-2 margin is diagnostic only and is not an acceptance gate.
- Calibration data selects thresholds; final-evaluation data is untouched until thresholds are frozen.
- Runtime and validation call the same rejection function and report whether thresholds came from the checkpoint or explicit overrides.
- CMLR, LRS3, AV-HuBERT, Auto-AVSR, MISP2021, ChatCLR, and CAS-VSR-W1k are not dependencies, fallbacks, or claimed components.
- Legacy intent-only checkpoints fail with an explicit migration error.

---

## File Map

### New files

- `command/phrase_catalog.json` — versioned initial phrase definitions.
- `command/catalog.py` — catalog types, normalization, validation, and SHA-256.
- `command/dataset.py` — immutable recording scan, deduplication, deterministic splits, and inventory output.
- `command/checkpoint.py` — versioned checkpoint validation, loading, and saving.
- `command/training.py` — ROCm-only training, centroid construction, and threshold calibration.
- `command/evaluation.py` — shared metric aggregation and auditable report generation.
- `scripts/build_command_manifest.py` — dataset inventory CLI.
- `tests/test_phrase_catalog.py` — catalog contract tests.
- `tests/test_command_dataset.py` — immutable mapping, deduplication, and split-gate tests.
- `tests/test_phrase_model.py` — input/output, parameter-cap, and determinism tests.
- `tests/test_phrase_checkpoint.py` — checkpoint schema and migration tests.
- `tests/test_phrase_training.py` — centroid and calibration tests.
- `tests/test_phrase_runtime.py` — dynamic head, exact text, rejection, and ROCm guard tests.
- `tests/test_phrase_evaluation.py` — final metric and threshold-provenance tests.

### Modified files

- `command/model.py` — replace the four-layer Conformer path with the low-capacity fixed-phrase model.
- `command/inference.py` — load the catalog checkpoint dynamically and perform shared probability-plus-distance rejection.
- `command/labels.py` — retain business intents and executable-intent safety policy; stop using intent labels as Torch logits.
- `backend/config.py` — add optional Torch probability and distance overrides while preserving prototype settings.
- `scripts/train_command_classifier.py` — become the ROCm phrase-training CLI.
- `scripts/validate_command_classifier.py` — evaluate frozen thresholds on disjoint known and unrelated manifests.
- `scripts/infer_command_clip.py` — emit phrase-oriented output from the production Torch backend.
- `scripts/start_real_rocm.sh` — default the official command path to Torch and fail without checkpoint/ROCm.
- `scripts/setup_amd_real.sh` — prepare persistent paths and validate a fixed-phrase checkpoint under Torch-first defaults.
- `scripts/amd_real_oneclick.sh` — use Torch by default and document the explicit recording-mode override.
- `scripts/smoke_rocm.sh` — verify a real ROCm forward pass and one checkpoint-backed prediction.
- `tests/test_command_classifier.py` — remove old Torch intent/margin assumptions and retain prototype safety coverage.
- `tests/test_deployment_files.py` — assert the new model/deployment contract and absence of obsolete claims.
- `tests/test_websocket_flow.py` — verify accepted phrase metadata and rejected-action behavior through WebSocket.
- `README.md` — document catalog, recording counts, training, startup, and honest product boundary.
- `docs/runbooks/amd-real-mode.md` — separate official Torch startup from explicit prototype recording mode.
- `docs/submission/project-profile-source.md` — replace old feature/Conformer claims and describe the measured ROCm path.
- `docs/submission/poster-copy.md` — align poster reference copy with probability-plus-distance rejection.
- `submission/README.md` — update the submission index and evidence status.
- `submission/pull-request-description.md` — align submission language with the implemented fixed-phrase system.
- `submission/demo-video-script.md` — use a checkpoint-backed ROCm demonstration and honest rejection wording.
- `scripts/generate_submission_assets.py` — generate corrected PDF/poster copy.
- `submission/Silent-Vision-Project-Profile.pdf` — regenerated project profile.
- `submission/Silent-Vision-Poster.pdf` — regenerated poster PDF.
- `submission/Silent-Vision-Poster.png` — regenerated poster image.

### Removed files

- `command/encoder.py` — obsolete statistical and unused Auto-AVSR feature paths.
- `scripts/record_command_manifest.py` — superseded by the immutable inventory/split builder.

---

### Task 1: Versioned Phrase Catalog

**Files:**
- Create: `command/phrase_catalog.json`
- Create: `command/catalog.py`
- Create: `tests/test_phrase_catalog.py`
- Modify: `command/labels.py`

**Interfaces:**
- Produces: `normalize_phrase(text: str) -> str`
- Produces: `PhraseEntry(phrase_id, text, language, intent, enabled)`
- Produces: `PhraseCatalog.from_records(records) -> PhraseCatalog`
- Produces: `load_phrase_catalog(path: Path) -> PhraseCatalog`
- Produces: `catalog_sha256(catalog: PhraseCatalog) -> str`
- Consumes: `CommandIntent` from `command.labels`

- [ ] **Step 1: Write catalog contract tests**

```python
from pathlib import Path

import pytest

from command.catalog import PhraseCatalog, load_phrase_catalog, normalize_phrase


def test_initial_catalog_has_exact_registered_text_and_intents():
    catalog = load_phrase_catalog(Path("command/phrase_catalog.json"))
    assert [(e.phrase_id, e.text, e.intent.value) for e in catalog.entries] == [
        ("zh_light_on_hello", "你好，请帮我打开灯", "LIGHT_ON"),
        ("zh_chat_meal", "你吃饭了吗？", "CHAT_OTHER"),
    ]


def test_normalization_preserves_punctuation():
    assert normalize_phrase("  Ｈello   世界？ ") == "hello 世界?"


@pytest.mark.parametrize(
    "records, message",
    [
        ([{"phraseId": "x", "text": "a", "language": "zh", "intent": "LIGHT_ON", "enabled": True},
          {"phraseId": "x", "text": "b", "language": "zh", "intent": "LIGHT_OFF", "enabled": True}], "duplicate phraseId"),
        ([{"phraseId": "x", "text": "a", "language": "zh", "intent": "NOT_REAL", "enabled": True}], "unknown intent"),
    ],
)
def test_invalid_catalog_is_rejected(records, message):
    with pytest.raises(ValueError, match=message):
        PhraseCatalog.from_records(records)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `pytest -q tests/test_phrase_catalog.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'command.catalog'`.

- [ ] **Step 3: Add the initial JSON catalog**

```json
{
  "schemaVersion": "silent-vision.phrase-catalog.v1",
  "phrases": [
    {
      "phraseId": "zh_light_on_hello",
      "text": "你好，请帮我打开灯",
      "language": "zh",
      "intent": "LIGHT_ON",
      "enabled": true
    },
    {
      "phraseId": "zh_chat_meal",
      "text": "你吃饭了吗？",
      "language": "zh",
      "intent": "CHAT_OTHER",
      "enabled": true
    }
  ]
}
```

- [ ] **Step 4: Implement normalization and strict catalog validation**

```python
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

from command.labels import CommandIntent


@dataclass(frozen=True)
class PhraseEntry:
    phrase_id: str
    text: str
    language: str
    intent: CommandIntent
    enabled: bool = True


@dataclass(frozen=True)
class PhraseCatalog:
    entries: tuple[PhraseEntry, ...]

    @classmethod
    def from_records(cls, records: list[dict]) -> "PhraseCatalog":
        entries: list[PhraseEntry] = []
        ids: set[str] = set()
        texts: set[str] = set()
        for record in records:
            phrase_id = str(record.get("phraseId", "")).strip()
            text = str(record.get("text", "")).strip()
            language = str(record.get("language", "")).strip()
            if not record.get("enabled", True):
                continue
            if not phrase_id or not text:
                raise ValueError("blank phraseId or text")
            if language not in {"zh", "en"}:
                raise ValueError(f"unsupported language: {language}")
            try:
                intent = CommandIntent(str(record["intent"]))
            except (KeyError, ValueError) as exc:
                raise ValueError(f"unknown intent: {record.get('intent')}") from exc
            normalized = normalize_phrase(text)
            if phrase_id in ids:
                raise ValueError(f"duplicate phraseId: {phrase_id}")
            if normalized in texts:
                raise ValueError(f"duplicate normalized phrase text: {text}")
            ids.add(phrase_id)
            texts.add(normalized)
            entries.append(PhraseEntry(phrase_id, text, language, intent, True))
        if not entries:
            raise ValueError("catalog has no enabled phrases")
        return cls(tuple(entries))


def normalize_phrase(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip()
    return re.sub(r"\s+", " ", normalized).lower()


def load_phrase_catalog(path: Path) -> PhraseCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "silent-vision.phrase-catalog.v1":
        raise ValueError("unsupported phrase catalog schema")
    return PhraseCatalog.from_records(payload["phrases"])


def catalog_sha256(catalog: PhraseCatalog) -> str:
    records = [{**asdict(entry), "intent": entry.intent.value} for entry in catalog.entries]
    encoded = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
```

- [ ] **Step 5: Run catalog tests**

Run: `pytest -q tests/test_phrase_catalog.py`

Expected: PASS.

- [ ] **Step 6: Commit the catalog boundary**

```bash
git add command/phrase_catalog.json command/catalog.py command/labels.py tests/test_phrase_catalog.py
git commit -m "feat: add fixed phrase catalog"
```

---

### Task 2: Immutable Dataset Inventory and Deterministic Splits

**Files:**
- Create: `command/dataset.py`
- Create: `scripts/build_command_manifest.py`
- Create: `tests/test_command_dataset.py`
- Remove: `scripts/record_command_manifest.py`

**Interfaces:**
- Consumes: `PhraseCatalog`, `load_phrase_catalog()`, and `normalize_phrase()` from Task 1
- Produces: `ManifestSample` with `sample_id`, `phrase_id`, `text`, `language`, `intent`, `source_intent`, `mouth_roi_npy`, `source_metadata`, and `sha256`
- Produces: `build_dataset_manifests(profile_root: Path, catalog_path: Path, output_dir: Path, allow_small_dataset: bool, seed: int) -> dict`
- Produces: six JSON artifacts: five partition manifests plus `inventory.json`

- [ ] **Step 1: Write tests using temporary source recordings**

```python
import json
from pathlib import Path

import numpy as np
import pytest

from command.dataset import build_dataset_manifests


def save_sample(root: Path, sample_id: str, phrase: str, intent: str, value: int):
    sample = root / "global" / intent / sample_id
    sample.mkdir(parents=True)
    metadata = {"sampleId": sample_id, "phrase": phrase, "intent": intent, "language": "zh"}
    (sample / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
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
    records = [json.loads(line) for line in (tmp_path / "out/train.jsonl").read_text().splitlines()]
    records += [json.loads(line) for line in (tmp_path / "out/calibration-known.jsonl").read_text().splitlines()]
    records += [json.loads(line) for line in (tmp_path / "out/evaluation-known.jsonl").read_text().splitlines()]
    meal = next(record for record in records if record["sample_id"] == "meal-1")
    assert meal["phrase_id"] == "zh_chat_meal"
    assert meal["intent"] == "CHAT_OTHER"
    assert meal["source_intent"] == "LIGHT_ON"
    assert source.read_bytes() == original
    assert inventory["intentMismatches"] == 1


def test_duplicate_mouth_arrays_are_never_split_across_partitions(tmp_path):
    save_sample(tmp_path, "a", "你好，请帮我打开灯", "LIGHT_ON", 1)
    save_sample(tmp_path, "b", "你好，请帮我打开灯", "LIGHT_ON", 1)
    inventory = build_dataset_manifests(tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17)
    assert inventory["duplicates"] == 1


def test_official_gate_rejects_too_few_samples(tmp_path):
    save_sample(tmp_path, "a", "你好，请帮我打开灯", "LIGHT_ON", 1)
    with pytest.raises(ValueError, match="15 independent takes"):
        build_dataset_manifests(tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", False, 17)


def test_explicit_unknown_is_not_a_training_class(tmp_path):
    save_sample(tmp_path, "u", "今天天气不错", "UNKNOWN", 3)
    build_dataset_manifests(tmp_path, Path("command/phrase_catalog.json"), tmp_path / "out", True, 17)
    assert "\"sample_id\": \"u\"" not in (tmp_path / "out/train.jsonl").read_text()
    unknown_text = (tmp_path / "out/evaluation-unknown.jsonl").read_text()
    unknown_text += (tmp_path / "out/calibration-unknown.jsonl").read_text()
    assert "\"sample_id\": \"u\"" in unknown_text
```

- [ ] **Step 2: Run the dataset tests and verify failure**

Run: `pytest -q tests/test_command_dataset.py`

Expected: FAIL during collection because `command.dataset` does not exist.

- [ ] **Step 3: Implement immutable scanning and deduplication**

Use these exact classification rules in `command/dataset.py`:

```python
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
```

For every recording directory under `<profile_root>/global/*/*`, require `metadata.json` and `mouth_roi.npy`. Resolve paths, read bytes without writing them, use `metadata["sampleId"]` when present and otherwise the recording directory name, hash the NPY file in 1 MiB chunks, and reject a later record when either its resolved sample ID or its NPY SHA-256 has already appeared. Store source paths as absolute strings in generated manifests.

- [ ] **Step 4: Implement deterministic partitioning and evidence gates**

Sort each group by `sha256(f"{seed}:{sample_id}")` and apply these rules:

```python
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
```

Write JSONL with `sort_keys=True` and UTF-8, and write `inventory.json` with `evidentiary: not allow_small_dataset`, catalog hash, seed, counts, exclusions, duplicates, mismatches, and a SHA-256 for each manifest. Empty partitions are valid only when `allow_small_dataset` is true.

- [ ] **Step 5: Add the manifest CLI and remove the obsolete starter-manifest script**

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("command/phrase_catalog.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    inventory = build_dataset_manifests(
        args.profile_root, args.catalog, args.output_dir, args.allow_small_dataset, args.seed
    )
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 6: Run dataset tests and a local inventory smoke test**

Run: `pytest -q tests/test_command_dataset.py`

Expected: PASS.

Run: `python3 scripts/build_command_manifest.py --help`

Expected: exit 0 with arguments for the profile root, catalog, output directory, seed, and small-dataset override.

- [ ] **Step 7: Commit dataset inventory support**

```bash
git add command/dataset.py scripts/build_command_manifest.py tests/test_command_dataset.py
git rm scripts/record_command_manifest.py
git commit -m "feat: build immutable phrase manifests"
```

---

### Task 3: Low-Capacity Phrase Model and Versioned Checkpoint

**Files:**
- Modify: `command/model.py`
- Create: `command/checkpoint.py`
- Create: `tests/test_phrase_model.py`
- Create: `tests/test_phrase_checkpoint.py`
- Remove: `command/encoder.py`

**Interfaces:**
- Produces: `build_fixed_phrase_model(num_classes: int, embedding_dim: int = 64) -> torch.nn.Module`
- Model forward: `model(frames: Tensor[B,T,96,96]) -> tuple[Tensor[B,C], Tensor[B,64]]`
- Produces: `count_trainable_parameters(model) -> int`
- Produces: `LoadedPhraseCheckpoint` with `model`, `catalog`, `phrase_ids`, `centroids`, `decision_thresholds`, and `training_summary`
- Produces: `save_phrase_checkpoint(path: Path, payload: dict) -> str`
- Produces: `load_phrase_checkpoint(path: Path, device: str) -> LoadedPhraseCheckpoint`
- Consumes: catalog types from Task 1

- [ ] **Step 1: Write shape, normalization, parameter-cap, and temporal-sensitivity tests**

```python
import pytest

torch = pytest.importorskip("torch")

from command.model import build_fixed_phrase_model, count_trainable_parameters


def test_phrase_model_shape_and_parameter_cap():
    model = build_fixed_phrase_model(num_classes=2)
    logits, embedding = model(torch.rand(3, 12, 96, 96))
    assert logits.shape == (3, 2)
    assert embedding.shape == (3, 64)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(3), atol=1e-5)
    assert count_trainable_parameters(model) < 150_000


def test_phrase_model_uses_temporal_order():
    torch.manual_seed(17)
    model = build_fixed_phrase_model(num_classes=2).eval()
    frames = torch.rand(1, 10, 96, 96)
    forward_logits, _ = model(frames)
    reverse_logits, _ = model(frames.flip(1))
    assert not torch.allclose(forward_logits, reverse_logits)
```

- [ ] **Step 2: Run the model tests and verify failure**

Run: `pytest -q tests/test_phrase_model.py`

Expected: FAIL because `build_fixed_phrase_model` is not defined.

- [ ] **Step 3: Replace the Conformer with the fixed phrase model**

Implement a module with these exact operations:

```python
class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.GELU(),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x):
        return x + self.net(x)


class FixedPhraseModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(512, 64)
        self.temporal = nn.Sequential(TemporalBlock(64, 1), TemporalBlock(64, 2))
        self.attention = nn.Linear(64, 1)
        self.embedding = nn.Linear(64, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, frames):
        x = frames.float()
        if x.max().item() > 1.0:
            x = x / 255.0
        small = F.interpolate(
            x.reshape(-1, 1, 96, 96), size=(16, 16), mode="bilinear", align_corners=False
        ).reshape(x.shape[0], x.shape[1], 256)
        motion = torch.zeros_like(small)
        motion[:, 1:] = small[:, 1:] - small[:, :-1]
        sequence = self.projection(torch.cat([small, motion], dim=-1))
        sequence = self.temporal(sequence.transpose(1, 2)).transpose(1, 2)
        weights = torch.softmax(self.attention(sequence), dim=1)
        pooled = (weights * sequence).sum(dim=1)
        embedding = F.normalize(self.embedding(pooled), dim=-1)
        return self.classifier(embedding), embedding
```

Validate rank and 96 × 96 dimensions before reshaping. Keep normalization and feature extraction inside the model so training and inference cannot diverge. Delete the obsolete `command/encoder.py` after removing its imports.

- [ ] **Step 4: Run model tests**

Run: `pytest -q tests/test_phrase_model.py`

Expected: PASS on a Torch environment; SKIP with a clear reason on the local non-Torch environment.

- [ ] **Step 5: Write checkpoint schema tests**

```python
import pytest

torch = pytest.importorskip("torch")

from command.checkpoint import load_phrase_checkpoint, save_phrase_checkpoint
from command.model import build_fixed_phrase_model


def valid_payload(model):
    return {
        "schemaVersion": "silent-vision.fixed-phrase.v1",
        "modelState": model.state_dict(),
        "phraseIds": ["zh_light_on_hello", "zh_chat_meal"],
        "phraseCatalog": [
            {"phraseId": "zh_light_on_hello", "text": "你好，请帮我打开灯", "language": "zh", "intent": "LIGHT_ON", "enabled": True},
            {"phraseId": "zh_chat_meal", "text": "你吃饭了吗？", "language": "zh", "intent": "CHAT_OTHER", "enabled": True},
        ],
        "featureConfig": {"fps": 25, "height": 96, "width": 96, "downsample": 16},
        "modelConfig": {"embeddingDim": 64, "parameterCap": 150000},
        "decisionThresholds": {
            "minProbability": 0.80,
            "maxCosineDistance": {"zh_light_on_hello": 0.20, "zh_chat_meal": 0.20},
        },
        "classCentroids": torch.nn.functional.normalize(torch.rand(2, 64), dim=1),
        "trainingSummary": {"seed": 17, "evidentiary": False},
    }


def test_checkpoint_builds_dynamic_two_phrase_head(tmp_path):
    model = build_fixed_phrase_model(2)
    path = tmp_path / "phrase.pt"
    payload = valid_payload(model)
    save_phrase_checkpoint(path, payload)
    loaded = load_phrase_checkpoint(path, "cpu")
    assert loaded.phrase_ids == ("zh_light_on_hello", "zh_chat_meal")
    assert loaded.centroids.shape == (2, 64)
    assert loaded.model.classifier.out_features == 2


def test_legacy_intent_checkpoint_has_migration_error(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"model": {}, "labels": ["LIGHT_ON", "LIGHT_OFF"]}, path)
    with pytest.raises(ValueError, match="legacy intent-only checkpoint"):
        load_phrase_checkpoint(path, "cpu")


def test_head_shape_mismatch_is_rejected(tmp_path):
    payload = valid_payload(build_fixed_phrase_model(2))
    payload["phraseIds"].append("third")
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="classifier head"):
        load_phrase_checkpoint(tmp_path / "bad.pt", "cpu")
```

- [ ] **Step 6: Implement checkpoint validation and atomic saving**

The schema must be exactly `silent-vision.fixed-phrase.v1`. Validate all required keys, reconstruct the catalog from embedded records, require ordered phrase IDs to match enabled catalog entries, require centroid shape `[class_count, embedding_dim]`, require finite thresholds, build the model from `modelConfig`, enforce the parameter cap, and load `modelState` strictly.

Save through a sibling temporary file and `Path.replace()`. Return the completed file's SHA-256 from `save_phrase_checkpoint`; keep that hash in the external run summary, not inside the checkpoint.

- [ ] **Step 7: Run checkpoint tests**

Run: `pytest -q tests/test_phrase_checkpoint.py`

Expected: PASS on a Torch environment; SKIP locally if Torch is unavailable.

- [ ] **Step 8: Commit the model/checkpoint boundary**

```bash
git add command/model.py command/checkpoint.py tests/test_phrase_model.py tests/test_phrase_checkpoint.py
git rm command/encoder.py
git commit -m "feat: add compact phrase model checkpoint"
```

---

### Task 4: ROCm Training, Centroids, and Calibration

**Files:**
- Create: `command/training.py`
- Modify: `scripts/train_command_classifier.py`
- Create: `tests/test_phrase_training.py`

**Interfaces:**
- Consumes: Task 2 manifests and inventory; Task 3 model/checkpoint APIs
- Produces: `require_rocm(torch_module) -> str`, returning `cuda:0` or raising `RuntimeError`
- Produces: `compute_class_centroids(embeddings, labels, class_count) -> Tensor[class_count,64]`
- Produces: `CalibrationRecord(expected_phrase_id, predicted_phrase_id, confidence, distance)`
- Produces: `calibrate_thresholds(known_records, unknown_records, phrase_ids) -> dict`
- Produces: `train_phrase_classifier(catalog_path: Path, inventory_path: Path, train_manifest: Path, calibration_known_manifest: Path, calibration_unknown_manifest: Path, output_path: Path, run_summary_path: Path, epochs: int, seed: int) -> dict`, returning the external run summary

- [ ] **Step 1: Write pure calibration and ROCm-guard tests**

```python
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from command.training import CalibrationRecord, calibrate_thresholds, compute_class_centroids, require_rocm


def test_centroids_are_class_means_normalized():
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    centroids = compute_class_centroids(embeddings, torch.tensor([0, 0, 1]), 2)
    assert centroids.shape == (2, 2)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(2))


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


def test_rocm_guard_does_not_accept_cuda_without_hip():
    fake = SimpleNamespace(
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
    )
    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        require_rocm(fake)
```

- [ ] **Step 2: Run training tests and verify failure**

Run: `pytest -q tests/test_phrase_training.py`

Expected: FAIL because `command.training` does not exist.

- [ ] **Step 3: Implement ROCm guard and centroid calculation**

```python
def require_rocm(torch_module) -> str:
    if not getattr(torch_module.version, "hip", None):
        raise RuntimeError("ROCm/HIP PyTorch is required")
    if not torch_module.cuda.is_available() or torch_module.cuda.device_count() < 1:
        raise RuntimeError("ROCm device cuda:0 is not available")
    return "cuda:0"


def compute_class_centroids(embeddings, labels, class_count: int):
    centroids = []
    for class_index in range(class_count):
        selected = embeddings[labels == class_index]
        if selected.shape[0] == 0:
            raise ValueError(f"class {class_index} has no training embedding")
        centroids.append(torch.nn.functional.normalize(selected.mean(dim=0), dim=0))
    return torch.stack(centroids)
```

- [ ] **Step 4: Implement deterministic threshold calibration**

Use only correctly classified known-calibration records to set each class radius to the maximum observed cosine distance. If a class has no correct known-calibration record, fail an official run; a non-evidentiary smoke run may use the 95th percentile of that class's training distances and must record the fallback.

With radii frozen, test probability candidates `0.50, 0.51, …, 0.99`. A record is accepted only when confidence is at least the candidate and distance is no greater than the predicted class radius. Choose candidates with unknown false-accept rate at most `0.10`; among them maximize correctly accepted known count, then choose the highest threshold on ties. If none meets the false-accept target, choose the candidate with the lowest false-accept rate, then the highest correctly accepted known count, then the highest threshold, and set `calibrationTargetMet` to false. With no unrelated calibration clips, use `0.85`, set the target false, and mark the run non-evidentiary.

- [ ] **Step 5: Implement the manifest dataset and seeded augmentation**

Load only the NPY paths named by each manifest. Convert clips to `[T,96,96]` tensors and pad a batch by repeating its final frame. Training augmentation uses a `torch.Generator` seeded from the run seed and applies brightness scale `[0.9,1.1]`, integer x/y shifts `[-2,2]`, and removal or duplication of at most one interior frame. Calibration has no augmentation.

- [ ] **Step 6: Rewrite the training CLI**

The command contract is:

```bash
/opt/venv/bin/python scripts/train_command_classifier.py \
  --catalog command/phrase_catalog.json \
  --inventory artifacts/phrase-data/inventory.json \
  --train-manifest artifacts/phrase-data/train.jsonl \
  --calibration-known artifacts/phrase-data/calibration-known.jsonl \
  --calibration-unknown artifacts/phrase-data/calibration-unknown.jsonl \
  --output artifacts/checkpoints/fixed-phrase.pt \
  --run-summary artifacts/checkpoints/fixed-phrase-run.json \
  --epochs 80 \
  --seed 17
```

Before loading samples, call `require_rocm(torch)`. Train with cross entropy, AdamW at `3e-4`, weight decay `1e-4`, batch size `4`, fixed 80 epochs, and deterministic seeds. Do not select epochs or hyperparameters using evaluation manifests. Compute centroids from unaugmented training embeddings after the final epoch, calibrate thresholds, save the checkpoint, then write the returned checkpoint SHA-256 plus Torch version, HIP version, `cuda:0` device name, seed, catalog hash, manifest hashes, epoch count, parameter count, calibration metrics, and `evidentiary` state to the external run summary.

- [ ] **Step 7: Run training tests and CLI help**

Run: `pytest -q tests/test_phrase_training.py`

Expected: PASS on a Torch environment; SKIP locally when Torch is unavailable.

Run: `python3 scripts/train_command_classifier.py --help`

Expected: exit 0 without importing or initializing a GPU.

- [ ] **Step 8: Commit training and calibration**

```bash
git add command/training.py scripts/train_command_classifier.py tests/test_phrase_training.py
git commit -m "feat: train fixed phrase model on ROCm"
```

---

### Task 5: Shared Runtime Rejection and Exact Phrase Output

**Files:**
- Modify: `command/inference.py`
- Modify: `backend/config.py`
- Create: `tests/test_phrase_runtime.py`
- Modify: `tests/test_command_classifier.py`

**Interfaces:**
- Consumes: `load_phrase_checkpoint()` from Task 3 and `require_rocm()` from Task 4
- Produces: `ThresholdResolution(min_probability, max_cosine_distance, source)`
- Produces: `resolve_thresholds(checkpoint_thresholds, probability_override, distance_override) -> ThresholdResolution`
- Produces: `evaluate_phrase_rejection(probability, distance, predicted_phrase_id, thresholds) -> tuple[bool, str | None]`
- Preserves: `CommandClassifier.predict(mouth_rois, metadata) -> CommandDecision`

- [ ] **Step 1: Write rejection and exact-output tests**

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.config import Settings
from command.checkpoint import save_phrase_checkpoint
from command.inference import (
    ThresholdResolution,
    TorchCommandClassifierBackend,
    evaluate_phrase_rejection,
    resolve_thresholds,
)
from command.model import build_fixed_phrase_model


@pytest.fixture
def mouth_clip():
    clip = np.zeros((12, 96, 96), dtype=np.uint8)
    clip[:, 32:64, 24:72] = np.arange(12, dtype=np.uint8)[:, None, None] * 10
    return clip


@pytest.fixture
def loaded_backend(tmp_path, monkeypatch, mouth_clip):
    model = build_fixed_phrase_model(2).eval()
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.tensor([10.0, -10.0]))
        _, embedding = model(torch.from_numpy(mouth_clip).unsqueeze(0))
    checkpoint = tmp_path / "phrase.pt"
    save_phrase_checkpoint(checkpoint, {
        "schemaVersion": "silent-vision.fixed-phrase.v1",
        "modelState": model.state_dict(),
        "phraseIds": ["zh_light_on_hello", "zh_chat_meal"],
        "phraseCatalog": [
            {"phraseId": "zh_light_on_hello", "text": "你好，请帮我打开灯", "language": "zh", "intent": "LIGHT_ON", "enabled": True},
            {"phraseId": "zh_chat_meal", "text": "你吃饭了吗？", "language": "zh", "intent": "CHAT_OTHER", "enabled": True},
        ],
        "featureConfig": {"fps": 25, "height": 96, "width": 96, "downsample": 16},
        "modelConfig": {"embeddingDim": 64, "parameterCap": 150000},
        "decisionThresholds": {
            "minProbability": 0.80,
            "maxCosineDistance": {"zh_light_on_hello": 0.05, "zh_chat_meal": 0.05},
        },
        "classCentroids": torch.cat([embedding, -embedding], dim=0),
        "trainingSummary": {"seed": 17, "evidentiary": False},
    })
    monkeypatch.setattr("command.inference.require_rocm", lambda torch_module: "cpu")
    return TorchCommandClassifierBackend(Settings(
        command_backend="torch",
        command_classifier_checkpoint=checkpoint,
    ))


def test_both_probability_and_distance_must_pass():
    thresholds = ThresholdResolution(0.80, {"a": 0.20}, "checkpoint")
    assert evaluate_phrase_rejection(0.90, 0.10, "a", thresholds) == (True, None)
    assert evaluate_phrase_rejection(0.70, 0.10, "a", thresholds) == (False, "low_probability")
    assert evaluate_phrase_rejection(0.90, 0.30, "a", thresholds) == (False, "embedding_distance")


def test_override_source_is_auditable():
    resolved = resolve_thresholds(
        {"minProbability": 0.80, "maxCosineDistance": {"a": 0.20}},
        probability_override=0.90,
        distance_override=None,
    )
    assert resolved.min_probability == 0.90
    assert resolved.max_cosine_distance == {"a": 0.20}
    assert resolved.source == "override:probability"


def test_accepted_decision_uses_exact_catalog_text(loaded_backend, mouth_clip):
    decision = loaded_backend.predict(mouth_clip, {})
    assert decision.accepted is True
    assert decision.intent.value == "LIGHT_ON"
    assert decision.metadata["matchedPhrase"] == "你好，请帮我打开灯"
    assert decision.metadata["displayText"] == "你好，请帮我打开灯"
    assert decision.metadata["thresholdSource"] == "checkpoint"
```

Use a deterministic two-class checkpoint fixture whose classifier bias selects the first phrase and whose centroid equals the fixture clip embedding. Patch only `require_rocm` in the fixture so this unit test can run on CPU; keep a separate test proving the unpatched production builder calls the ROCm guard.

- [ ] **Step 2: Run runtime tests and verify failure**

Run: `pytest -q tests/test_phrase_runtime.py`

Expected: FAIL because the new threshold interfaces are not defined.

- [ ] **Step 3: Implement the shared rejection functions**

```python
def evaluate_phrase_rejection(probability, distance, predicted_phrase_id, thresholds):
    if probability < thresholds.min_probability:
        return False, "low_probability"
    if distance > thresholds.max_cosine_distance[predicted_phrase_id]:
        return False, "embedding_distance"
    return True, None
```

Validate overrides in `[0,1]`. A distance override replaces every class radius. The source value is `checkpoint`, `override:probability`, `override:distance`, or `override:probability,distance`.

- [ ] **Step 4: Replace Torch intent inference with checkpoint-driven phrase inference**

At construction, require ROCm, load the checkpoint on `cuda:0`, and resolve thresholds. At prediction, create a `[1,T,96,96]` tensor directly from the mouth ROI array on `cuda:0`, call the model under `torch.inference_mode()`, calculate softmax and the top two phrase scores, calculate cosine distance to the predicted phrase centroid, and call `evaluate_phrase_rejection`.

Accepted metadata must contain `phraseId`, exact `matchedPhrase`, exact `displayText`, `language`, `openSetDistance`, `thresholdSource`, `backend: "torch"`, and phrase-oriented `topK`. Rejected decisions use intent `UNKNOWN`, omit phrase/display text, include the rejection reason and diagnostics, and remain `accepted: false`. Margin is calculated for diagnostics only.

- [ ] **Step 5: Add optional explicit Torch overrides to settings**

```python
command_phrase_probability_override: float | None = None
command_phrase_distance_override: float | None = None
```

Leave existing prototype calibration thresholds in place for recording mode, but do not pass the prototype margin setting into Torch rejection.

- [ ] **Step 6: Run runtime and legacy safety tests**

Run: `pytest -q tests/test_phrase_runtime.py tests/test_command_classifier.py`

Expected: PASS. The prototype backend remains available only when explicitly selected; Torch never falls back to CPU or prototype.

- [ ] **Step 7: Commit runtime phrase inference**

```bash
git add command/inference.py backend/config.py tests/test_phrase_runtime.py tests/test_command_classifier.py
git commit -m "feat: infer exact phrases with calibrated rejection"
```

---

### Task 6: Frozen Final Evaluation and CLI Output

**Files:**
- Create: `command/evaluation.py`
- Modify: `scripts/validate_command_classifier.py`
- Modify: `scripts/infer_command_clip.py`
- Create: `tests/test_phrase_evaluation.py`

**Interfaces:**
- Consumes: the exact Task 5 threshold resolver and rejection function
- Produces: `EvaluationRecord(expected_phrase_id, predicted_phrase_id, accepted, confidence, distance)`
- Produces: `build_evaluation_report(known_records, unknown_records, phrase_intents, provenance) -> dict`
- Produces: `evaluate_checkpoint(checkpoint_path: Path, known_manifest: Path, unknown_manifest: Path, output_path: Path, probability_override: float | None, distance_override: float | None) -> dict`
- Produces report fields: `phraseAccuracy`, `mappedIntentAccuracy`, `acceptedPrecision`, `knownAcceptanceRate`, `acceptedPhraseAccuracy`, `unknownFalseAcceptRate`, `unknownRejectionRate`, per-phrase counts, confusion matrix, raw counts, checkpoint hash, manifest hashes, and threshold provenance

- [ ] **Step 1: Write metric tests with explicit denominators**

```python
import pytest

from command.evaluation import EvaluationRecord, build_evaluation_report


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
```

- [ ] **Step 2: Run the evaluation test and verify failure**

Run: `pytest -q tests/test_phrase_evaluation.py`

Expected: FAIL because `command.evaluation` does not exist.

- [ ] **Step 3: Implement report aggregation**

Define phrase accuracy as correct top-1 phrase predictions divided by all known clips, including rejected clips. Define mapped-intent accuracy by mapping expected and predicted phrase IDs through `phrase_intents`. Define known acceptance as accepted known clips divided by all known clips. Define accepted-phrase accuracy and accepted precision as correctly predicted accepted known clips divided by accepted known clips, returning `null` when none are accepted. Define unrelated false acceptance as accepted unrelated clips divided by all unrelated clips, and rejection as its complement. Include raw integer numerators and denominators beside every rate, a phrase-by-phrase count object, and a confusion matrix that includes rejected predictions under `UNKNOWN`.

- [ ] **Step 4: Rewrite validation to use frozen checkpoint thresholds**

The command contract is:

```bash
/opt/venv/bin/python scripts/validate_command_classifier.py \
  --checkpoint artifacts/checkpoints/fixed-phrase.pt \
  --known-manifest artifacts/phrase-data/evaluation-known.jsonl \
  --unknown-manifest artifacts/phrase-data/evaluation-unknown.jsonl \
  --output artifacts/reports/fixed-phrase-evaluation.json
```

Require ROCm, load thresholds from the checkpoint unless explicit CLI overrides are provided, run each clip once, and atomically write the report with checkpoint SHA-256 and both manifest SHA-256 values. Never read calibration manifests in this command.

Add a unit test that calls `evaluate_checkpoint()` with temporary known/evaluation manifests and asserts its function signature has no calibration-manifest parameter and its report uses the already-resolved checkpoint thresholds.

- [ ] **Step 5: Update single-clip inference output**

Require `--checkpoint` and `--mouth-roi`. Print one JSON object containing public intent, accepted state, confidence, margin diagnostics, exact phrase metadata when accepted, rejection reason when rejected, backend, device, and threshold source.

- [ ] **Step 6: Run evaluation tests and CLI help**

Run: `pytest -q tests/test_phrase_evaluation.py`

Expected: PASS.

Run: `python3 scripts/validate_command_classifier.py --help && python3 scripts/infer_command_clip.py --help`

Expected: both commands exit 0 without initializing a GPU.

- [ ] **Step 7: Commit frozen evaluation support**

```bash
git add command/evaluation.py scripts/validate_command_classifier.py scripts/infer_command_clip.py tests/test_phrase_evaluation.py
git commit -m "feat: evaluate frozen phrase thresholds"
```

---

### Task 7: WebSocket Safety and ROCm Deployment Defaults

**Files:**
- Modify: `tests/test_websocket_flow.py`
- Modify: `tests/test_deployment_files.py`
- Modify: `scripts/start_real_rocm.sh`
- Modify: `scripts/setup_amd_real.sh`
- Modify: `scripts/amd_real_oneclick.sh`
- Modify: `scripts/smoke_rocm.sh`

**Interfaces:**
- Consumes: existing `CommandDecision` and `AgentPolicy` behavior
- Preserves: accepted executable phrases may execute; rejected clips and accepted `CHAT_OTHER` phrases never execute
- Produces: official startup that defaults `COMMAND_BACKEND` to `torch`

- [ ] **Step 1: Add WebSocket end-to-end safety tests**

Add this stub and collection helper to `tests/test_websocket_flow.py`, reusing the existing `patch_clip_pipeline()`:

```python
from backend.schemas import CommandDecision


class StubClassifier:
    def __init__(self, decision: CommandDecision):
        self.decision = decision

    def predict(self, mouth_frames, metadata):
        return self.decision.model_copy(update={"metadata": self.decision.metadata | metadata})


def send_stubbed_clip(app, monkeypatch, decision):
    patch_clip_pipeline(monkeypatch)
    app.state.command_classifier = StubClassifier(decision)
    client = TestClient(app)
    session_id = client.post("/api/sessions").json()["sessionId"]
    messages = {}
    with client.websocket_connect(f"/ws/{session_id}", headers={"origin": "http://localhost:8000"}) as ws:
        assert ws.receive_json()["type"] == "session.ready"
        ws.send_json({"type": "clip.start", "profileId": "global"})
        assert ws.receive_json()["type"] == "clip.started"
        ws.send_bytes(b"fake-webm")
        for _ in range(20):
            message = ws.receive_json()
            messages[message["type"]] = message
            if message["type"] == "agent.result":
                break
    return messages
```

Create three `CommandDecision` values with `topK=[]`, `logits=[]`, and explicit reasons. Assert the accepted light decision returns exact `displayText` and agent action `execute`; an unaccepted `UNKNOWN` returns `reject`; an accepted, non-executable `CHAT_OTHER` returns `ignore` and agent text `你吃饭了吗？`.

- [ ] **Step 2: Replace deployment-file assertions**

Assert the scripts mention `COMMAND_BACKEND:-torch`, `torch.version.hip`, checkpoint presence, and a real classifier smoke prediction. Assert production files do not contain `Conformer`, `FrozenAutoAVSRFeatureExtractor`, `StatisticalVisualFeatureExtractor`, or a Torch CPU fallback.

- [ ] **Step 3: Run deployment tests and verify current defaults fail**

Run: `pytest -q tests/test_websocket_flow.py tests/test_deployment_files.py`

Expected: FAIL because the deployment scripts still default to prototype and old architecture assertions remain.

- [ ] **Step 4: Make official startup Torch-first and fail closed**

In `start_real_rocm.sh`, `setup_amd_real.sh`, and `amd_real_oneclick.sh`, use:

```bash
export COMMAND_BACKEND="${COMMAND_BACKEND:-torch}"
export COMMAND_CLASSIFIER_CHECKPOINT="${COMMAND_CLASSIFIER_CHECKPOINT:-$PERSISTENCE_ROOT/models/fixed-phrase.pt}"
if [[ "$COMMAND_BACKEND" == "torch" && ! -s "${COMMAND_CLASSIFIER_CHECKPOINT}" ]]; then
  echo "Torch phrase checkpoint not found: ${COMMAND_CLASSIFIER_CHECKPOINT}" >&2
  exit 1
fi
```

Run a Python preflight that requires HIP, device availability, and at least one device before starting the server. The recording/calibration command must explicitly set `COMMAND_BACKEND=prototype`; it is labeled recording mode and is not the official classifier demo.

- [ ] **Step 5: Strengthen the remote smoke script**

After dependency and HIP checks, run `tests/test_phrase_model.py`, `tests/test_phrase_checkpoint.py`, `tests/test_phrase_runtime.py`, then run `scripts/infer_command_clip.py` against a supplied checkpoint and sample NPY. Fail if output does not report `backend: torch`, `device: cuda:0`, and a threshold source.

- [ ] **Step 6: Run integration and deployment tests**

Run: `pytest -q tests/test_websocket_flow.py tests/test_deployment_files.py tests/test_command_classifier.py`

Expected: PASS.

Run: `bash -n scripts/start_real_rocm.sh scripts/setup_amd_real.sh scripts/amd_real_oneclick.sh scripts/smoke_rocm.sh`

Expected: exit 0.

- [ ] **Step 7: Commit deployment and safety integration**

```bash
git add tests/test_websocket_flow.py tests/test_deployment_files.py scripts/start_real_rocm.sh scripts/setup_amd_real.sh scripts/amd_real_oneclick.sh scripts/smoke_rocm.sh
git commit -m "feat: make ROCm phrase classifier the demo path"
```

---

### Task 8: Submission Copy and Generated Assets

**Files:**
- Modify: `README.md`
- Modify: `docs/runbooks/amd-real-mode.md`
- Modify: `docs/submission/project-profile-source.md`
- Modify: `docs/submission/poster-copy.md`
- Modify: `submission/README.md`
- Modify: `submission/pull-request-description.md`
- Modify: `submission/demo-video-script.md`
- Modify: `scripts/generate_submission_assets.py`
- Modify: `submission/Silent-Vision-Project-Profile.pdf`
- Modify: `submission/Silent-Vision-Poster.pdf`
- Modify: `submission/Silent-Vision-Poster.png`
- Modify: `tests/test_deployment_files.py`

**Interfaces:**
- Consumes: actual command names, checkpoint schema, device behavior, and report field names from Tasks 1–7
- Produces: English contest submission materials with no unmeasured performance claim

- [ ] **Step 1: Add static claim-regression tests**

Read the README, Radeon runbook, submission index, profile source, poster copy, PR description, demo script, and asset generator as text. Assert the relevant architecture sections name the fixed-phrase classifier, ROCm classifier execution, CPU preprocessing, phrase catalog, and heuristic rejection. Assert none claims the submitted classifier uses CMLR, LRS3, AV-HuBERT, Auto-AVSR, a four-layer Conformer, or mean/std/motion repetition.

- [ ] **Step 2: Run the claim tests and verify failure**

Run: `pytest -q tests/test_deployment_files.py`

Expected: FAIL on obsolete architecture statements.

- [ ] **Step 3: Rewrite the English documentation from measured facts only**

Document these boundaries verbatim in substance:

- Silent Vision recognizes a personalized catalog of fixed phrases; it is not open-vocabulary lipreading.
- The Torch phrase model trains and runs on AMD Radeon through ROCm.
- Decode, face detection, alignment, and mouth cropping run on CPU.
- A prediction is accepted only when probability and centroid-distance gates pass.
- Exact displayed text and intent come from the registered catalog.
- The current small-data smoke run proves pipeline execution only; accuracy figures appear only after the official evaluation report exists.

Include copy-paste commands for manifest building, training, final validation, official startup, and explicit prototype recording mode.

- [ ] **Step 4: Regenerate PDF and poster**

Run: `python3 scripts/generate_submission_assets.py`

Expected: exit 0 and updated `submission/Silent-Vision-Project-Profile.pdf`, `submission/Silent-Vision-Poster.pdf`, and `submission/Silent-Vision-Poster.png`.

- [ ] **Step 5: Render and inspect generated assets**

Render every PDF page to PNG using the repository's PDF verification workflow. Verify no clipped text, missing Chinese glyph boxes, page overflow, stale architecture wording, or performance claims absent from an evaluation report. Inspect the poster at original resolution for the same issues.

- [ ] **Step 6: Run documentation and bundle tests**

Run: `pytest -q tests/test_deployment_files.py tests/test_contest_bundle.py`

Expected: PASS.

Run: `python3 scripts/build_contest_bundle.py`

Expected: exit 0; the bundle contains source, phrase catalog, README, profile PDF, poster, and demo script, and excludes recordings/checkpoints.

- [ ] **Step 7: Commit corrected submission materials**

```bash
git add README.md docs/runbooks/amd-real-mode.md docs/submission/project-profile-source.md docs/submission/poster-copy.md submission/README.md submission/Silent-Vision-Project-Profile.pdf submission/Silent-Vision-Poster.pdf submission/Silent-Vision-Poster.png submission/pull-request-description.md submission/demo-video-script.md scripts/generate_submission_assets.py tests/test_deployment_files.py
git commit -m "docs: align submission with fixed phrase classifier"
```

---

### Task 9: Local and Radeon Non-Evidentiary Pipeline Verification

**Files:**
- Modify only if a verified defect is found in Tasks 1–8.
- Generate outside Git: `artifacts/phrase-data/*`, `artifacts/checkpoints/*`, `artifacts/reports/*`

**Interfaces:**
- Consumes: all implementation and CLI contracts from Tasks 1–8
- Produces: reproducible local test evidence and a Radeon smoke-run summary marked `evidentiary: false`

- [ ] **Step 1: Run the complete local suite**

Run: `pytest -q`

Expected: all non-Torch tests PASS; Torch-only tests PASS when local Torch is installed or SKIP with the declared missing-Torch reason.

- [ ] **Step 2: Verify the worktree before remote transfer**

Run: `git status --short && git log -8 --oneline`

Expected: no unintended files and the task commits listed in order.

- [ ] **Step 3: Transfer a committed snapshot to a new Radeon directory**

Run from the local repository:

```bash
git archive --format=tar HEAD | ssh -p 31394 root@36.150.116.206 \
  'mkdir -p /workspace/silent-vision-fixed-phrase && tar -xf - -C /workspace/silent-vision-fixed-phrase'
```

Expected: exit 0. Do not overwrite the existing remote checkout or persistent recordings.

- [ ] **Step 4: Build small-data manifests from the existing global recordings**

Run on Radeon:

```bash
cd /workspace/silent-vision-fixed-phrase
/opt/venv/bin/python scripts/build_command_manifest.py \
  --profile-root /workspace/persistent/silent-vision/profiles \
  --catalog command/phrase_catalog.json \
  --output-dir artifacts/phrase-data \
  --seed 17 \
  --allow-small-dataset
```

Expected: four light-on and five meal samples are catalog-mapped before deduplication; the meal samples retain `sourceIntent: LIGHT_ON` but use `intent: CHAT_OTHER`; output states `evidentiary: false`. If `PERSISTENCE_ROOT` points elsewhere, locate its `profiles/global` directory read-only and rerun with that explicit `profiles` path.

- [ ] **Step 5: Train a ROCm smoke checkpoint**

Run the Task 4 training command with the generated manifests and `--epochs 5`.

Expected: exit 0; run summary reports non-empty HIP version, `cuda:0`, fewer than 150,000 parameters, checkpoint SHA-256, and `evidentiary: false`.

- [ ] **Step 6: Run model/runtime smoke tests and one inference**

Run:

```bash
/opt/venv/bin/python -m pytest -q \
  tests/test_phrase_model.py \
  tests/test_phrase_checkpoint.py \
  tests/test_phrase_training.py \
  tests/test_phrase_runtime.py
/opt/venv/bin/python scripts/infer_command_clip.py \
  --checkpoint artifacts/checkpoints/fixed-phrase.pt \
  --mouth-roi "$(/opt/venv/bin/python -c 'import json; print(json.loads(open("artifacts/phrase-data/train.jsonl").readline())["mouth_roi_npy"])')"
```

Expected: all tests PASS on ROCm; inference emits valid JSON with `backend: torch`, `device: cuda:0`, phrase-oriented top-K, distance, and threshold source. The prediction is a functional smoke result, not validation evidence.

- [ ] **Step 7: Record verification outputs without committing generated data**

Keep checkpoint, manifests, and run summaries under ignored `artifacts/`. Record command output and artifact hashes in the implementation handoff. Do not copy recordings or model checkpoints into the contest source bundle.

---

### Task 10: Record the Missing Data and Produce Official Evidence

**Files:**
- Generate outside Git: persistent calibration samples and `artifacts/*`
- Modify after measurement: `docs/submission/project-profile-source.md`
- Modify after measurement: `submission/pull-request-description.md`
- Modify after measurement: `submission/demo-video-script.md`
- Modify after measurement: `scripts/generate_submission_assets.py`
- Regenerate after measurement: `submission/Silent-Vision-Project-Profile.pdf`
- Regenerate after measurement: `submission/Silent-Vision-Poster.pdf`
- Regenerate after measurement: `submission/Silent-Vision-Poster.png`

**Interfaces:**
- Consumes: official no-override data gate and all ROCm commands from prior tasks
- Produces: evidentiary checkpoint/run summary, untouched final-evaluation report, and documentation that cites only those measured values

- [ ] **Step 1: Record the exact missing known-phrase counts**

Using independent takes in the existing calibration UI, record at least 11 additional `你好，请帮我打开灯` samples and at least 10 additional `你吃饭了吗？` samples. Select `LIGHT_ON` for the first phrase and `CHAT_OTHER` for the second. Vary natural timing and head position while keeping the same target deployment camera and speaker.

- [ ] **Step 2: Record unrelated phrases for rejection evaluation**

Record at least 15 independent clips whose source intent is explicitly `UNKNOWN`. Use natural daily phrases not present in the catalog. Do not train on these clips and do not relabel them as either registered phrase.

- [ ] **Step 3: Build official manifests without the override**

Run the Task 2 manifest command without `--allow-small-dataset`.

Expected: exit 0; inventory reports at least 15 unique samples per phrase, at least 15 unique unrelated clips, all required partitions non-empty, and `evidentiary: true`.

- [ ] **Step 4: Train the official ROCm checkpoint**

Run the Task 4 command with 80 epochs and seed 17.

Expected: exit 0; checkpoint and external run summary contain matching catalog/manifest provenance, a real HIP version, `cuda:0`, fewer than 150,000 parameters, frozen thresholds, and `evidentiary: true`.

- [ ] **Step 5: Run final evaluation exactly once after calibration is frozen**

Run the Task 6 validation command against `evaluation-known.jsonl` and `evaluation-unknown.jsonl`.

Expected: exit 0 and an atomic report with known acceptance, accepted-phrase accuracy, unrelated false acceptance, unrelated rejection, raw counts, hashes, and checkpoint threshold provenance. Do not retrain or retune thresholds in response to this final report without creating a new split and a new run identifier.

- [ ] **Step 6: Verify the live official path**

Start with `COMMAND_BACKEND=torch` and the evidentiary checkpoint. Demonstrate both registered phrases and at least one unrelated phrase. Confirm that only the light phrase can reach `execute`, the meal phrase returns exact text with `ignore`, and the unrelated phrase returns `UNKNOWN` with `reject`.

- [ ] **Step 7: Insert only measured values into submission materials**

Copy the four final report rates with integer denominators, checkpoint SHA-256, GPU name, Torch version, and HIP version into the English project profile and PR description. Keep the scope statement personalized and fixed-phrase; do not convert measured same-speaker results into cross-speaker or open-vocabulary claims.

- [ ] **Step 8: Regenerate and verify the final assets**

Run: `python3 scripts/generate_submission_assets.py && pytest -q`

Expected: asset generation exits 0 and the full test suite passes. Render every PDF page and inspect the poster at original resolution before submission.

- [ ] **Step 9: Commit official measured documentation**

```bash
git add docs/submission/project-profile-source.md submission/Silent-Vision-Project-Profile.pdf submission/Silent-Vision-Poster.pdf submission/Silent-Vision-Poster.png submission/pull-request-description.md submission/demo-video-script.md scripts/generate_submission_assets.py
git commit -m "docs: add measured fixed phrase results"
```

- [ ] **Step 10: Build the final source bundle and inspect status**

Run: `python3 scripts/build_contest_bundle.py && git status --short`

Expected: the bundle is generated successfully, contains no recordings or checkpoints, and the worktree contains no unintended changes.
