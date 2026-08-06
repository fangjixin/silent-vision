# Bilingual Fixed-Phrase Language Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the approved Chinese-and-English scope by training one four-phrase Torch head and requiring an explicit language that identically constrains runtime, calibration, and evaluation decisions.

**Architecture:** The catalog remains the authority for phrase text, language, and intent. A focused `command/language.py` helper validates `zh`/`en` and computes a softmax over only the selected language's head indices while preserving original checkpoint indices. WebSocket recognition and calibration use typed requests; calibration sends a phrase ID and the server derives canonical metadata.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, PyTorch/ROCm, NumPy, browser ES modules, pytest, Playwright.

## Global Constraints

- The catalog has exactly four initial phrases: two Chinese and two English.
- Training loss remains one four-class cross-entropy loss; language masking is decision-time behavior only.
- Missing or unsupported recognition language fails before video decoding or model execution.
- Runtime, threshold calibration, one-clip inference, and final evaluation use the same language-scoring helper.
- `UNKNOWN` is a rejection result and evaluation pool, never a learned class.
- Registered calibration metadata is derived from the catalog; unrelated calibration requires a non-empty free-form phrase and an explicit `zh` or `en`.
- The checkpoint schema is bumped to `silent-vision.fixed-phrase.v2`; v1 checkpoints are rejected.
- Torch training and production inference remain ROCm-only; local CPU tests are non-evidentiary.
- Exact output text is copied from catalog metadata only after acceptance.

---

## File Structure

- Create `command/language.py`: language validation and selected-language score construction.
- Modify `command/phrase_catalog.json`: add the two approved English phrase classes.
- Modify `command/catalog.py`: catalog lookup/serialization helpers and calibration target resolution.
- Modify `command/checkpoint.py`: require v2 schema and the selected-language decision policy marker.
- Modify `command/training.py`: use shared selected-language scoring for calibration records.
- Modify `command/evaluation.py`: validate and pass each manifest row's language.
- Modify `command/inference.py`: require language as a first-class argument and apply shared scores.
- Modify `backend/schemas.py`: typed clip-start and catalog-driven calibration requests.
- Modify `api/websocket.py`: validate requests, expose catalog, and fail before clip processing.
- Modify `frontend/index.html`, `frontend/websocket.js`, `frontend/styles.css`: runtime language selector and catalog-driven recording controls.
- Modify `scripts/infer_command_clip.py`: require `--language zh|en`.
- Modify focused tests under `tests/` and submission documentation after behavior is green.

---

### Task 1: Four-Phrase Catalog and Versioned Decision Contract

**Files:**
- Modify: `command/phrase_catalog.json`
- Modify: `command/catalog.py`
- Modify: `command/checkpoint.py`
- Test: `tests/test_phrase_catalog.py`
- Test: `tests/test_phrase_checkpoint.py`
- Test fixtures: `tests/test_phrase_runtime.py`, `tests/test_phrase_evaluation.py`, `tests/test_websocket_flow.py`

**Interfaces:**
- Produces: `catalog_records(catalog: PhraseCatalog) -> list[dict[str, object]]`
- Produces: `PhraseCatalog.by_id(phrase_id: str) -> PhraseEntry`
- Produces: checkpoint schema `silent-vision.fixed-phrase.v2` with `decisionPolicy.probabilityNormalization == "selected-language-softmax"` and `decisionPolicy.languageSelectionRequired is True`.

- [ ] **Step 1: Write failing catalog tests**

Assert the repository catalog contains these ordered records:

```python
[
    ("zh_light_on_hello", "你好，请帮我打开灯", "zh", "LIGHT_ON"),
    ("zh_chat_meal", "你吃饭了吗？", "zh", "CHAT_OTHER"),
    ("en_light_on_hello", "Hello, please turn on the light.", "en", "LIGHT_ON"),
    ("en_chat_meal", "Have you eaten?", "en", "CHAT_OTHER"),
]
```

Also assert `by_id()` returns the canonical record and raises `ValueError("unknown phraseId")` for an unknown ID.

- [ ] **Step 2: Run catalog tests and verify RED**

Run: `pytest -q tests/test_phrase_catalog.py`

Expected: FAIL because the English entries and `by_id()` do not exist.

- [ ] **Step 3: Add English entries and catalog helpers**

Add the two English records in the approved order. Implement lookup without a second source of truth, and move the existing private catalog serialization shape into public `catalog_records()`.

- [ ] **Step 4: Run catalog tests and verify GREEN**

Run: `pytest -q tests/test_phrase_catalog.py`

Expected: PASS.

- [ ] **Step 5: Write failing v2 checkpoint tests**

Update the valid fixture to four classes and add:

```python
assert payload["schemaVersion"] == "silent-vision.fixed-phrase.v2"
assert payload["decisionPolicy"] == {
    "languageSelectionRequired": True,
    "probabilityNormalization": "selected-language-softmax",
}
```

Assert v1 and a missing or different policy are rejected before model loading.

- [ ] **Step 6: Run checkpoint tests and verify RED**

Run: `pytest -q tests/test_phrase_checkpoint.py`

Expected: FAIL because the current schema is v1 and has no decision policy.

- [ ] **Step 7: Implement the v2 checkpoint boundary**

Set `SCHEMA_VERSION` to v2, add `decisionPolicy` to required keys, validate the exact two policy values, and save them from training. Do not add a v1 migration path. Mechanically update every synthetic checkpoint fixture returned by `rg 'silent-vision\.fixed-phrase\.v1' tests` so the schema boundary does not leave unrelated tests broken.

- [ ] **Step 8: Run checkpoint tests and verify GREEN**

Run: `pytest -q tests/test_phrase_checkpoint.py tests/test_phrase_runtime.py tests/test_phrase_evaluation.py tests/test_websocket_flow.py`

Expected: PASS.

- [ ] **Step 9: Commit Task 1**

```bash
git add command/phrase_catalog.json command/catalog.py command/checkpoint.py tests/test_phrase_catalog.py tests/test_phrase_checkpoint.py tests/test_phrase_runtime.py tests/test_phrase_evaluation.py tests/test_websocket_flow.py
git commit -m "feat: define bilingual phrase checkpoint contract"
```

---

### Task 2: Shared Selected-Language Scores in Runtime, Calibration, and Evaluation

**Files:**
- Create: `command/language.py`
- Modify: `command/inference.py`
- Modify: `command/training.py`
- Modify: `command/evaluation.py`
- Modify: `scripts/infer_command_clip.py`
- Test: `tests/test_phrase_runtime.py`
- Test: `tests/test_phrase_training.py`
- Test: `tests/test_phrase_evaluation.py`

**Interfaces:**
- Produces: `validate_recognition_language(value: object) -> Literal["zh", "en"]`
- Produces: `score_language_candidates(logits, phrase_languages: Sequence[str], language: object) -> LanguageCandidateScores`
- Produces: `CommandClassifierBackend.predict(mouth_frames: np.ndarray, language: str, metadata: dict[str, object]) -> CommandDecision`

- [ ] **Step 1: Write failing score-helper tests**

Use logits `[100.0, 1.0, 2.0, 0.0]`, languages `["en", "zh", "zh", "en"]`, and selected language `zh`. Assert only original indices `(2, 1)` are ranked, their two probabilities sum to 1, index 2 is best, and index 0 never appears despite its larger logit. Assert missing, `unknown`, and unsupported values raise `ValueError`.

- [ ] **Step 2: Run focused runtime tests and verify RED**

Run: `pytest -q tests/test_phrase_runtime.py -k language`

Expected: FAIL because `command.language` and required language scoring do not exist.

- [ ] **Step 3: Implement `command/language.py`**

Create an immutable result carrying selected language, eligible original indices, ranked original indices, and probabilities keyed by original index. Require one-dimensional logits, matching catalog-language length, at least one eligible class, and finite softmax results.

- [ ] **Step 4: Make runtime language a required argument**

Change every backend signature to `predict(mouth_frames, language, metadata)`. Torch uses `score_language_candidates()` for best/second confidence and top-K, stores `selectedLanguage` and `eligiblePhraseIds` on accepted and rejected results, and never exposes matched text on rejection. Prototype filters loaded samples to the selected language before matching. Fake validates language before its deterministic result.

- [ ] **Step 5: Run runtime tests and verify GREEN**

Run: `pytest -q tests/test_phrase_runtime.py tests/test_command_classifier.py`

Expected: PASS.

- [ ] **Step 6: Write failing calibration and evaluation language tests**

Add a calibration case where the excluded language has the largest raw logit and assert the recorded prediction uses the eligible language. Add known and UNKNOWN manifest rows with `language`; assert missing/invalid language is rejected. Assert `_evaluate_dataset()` passes the manifest language into the backend.

- [ ] **Step 7: Run training/evaluation tests and verify RED**

Run: `pytest -q tests/test_phrase_training.py tests/test_phrase_evaluation.py`

Expected: FAIL because calibration currently softmaxes all classes and evaluation omits language.

- [ ] **Step 8: Share scoring with calibration and evaluation**

Keep four-way cross-entropy unchanged. During `_calibration_records()`, obtain the row language from the manifest and call `score_language_candidates()` on each row. During final evaluation, validate the manifest language and call `backend.predict(frames, language, metadata)`.

- [ ] **Step 9: Require language in one-clip inference**

Add required `--language` choices `zh` and `en`; pass it as the classifier argument rather than embedding it in debug metadata.

- [ ] **Step 10: Run Task 2 tests and verify GREEN**

Run: `pytest -q tests/test_phrase_runtime.py tests/test_command_classifier.py tests/test_phrase_training.py tests/test_phrase_evaluation.py`

Expected: PASS.

- [ ] **Step 11: Commit Task 2**

```bash
git add command/language.py command/inference.py command/training.py command/evaluation.py scripts/infer_command_clip.py tests/test_phrase_runtime.py tests/test_command_classifier.py tests/test_phrase_training.py tests/test_phrase_evaluation.py
git commit -m "feat: route phrase decisions by selected language"
```

---

### Task 3: Typed WebSocket Recognition and Catalog-Authoritative Recording

**Files:**
- Modify: `backend/schemas.py`
- Modify: `api/websocket.py`
- Modify: `backend/main.py`
- Modify: `command/dataset.py`
- Test: `tests/test_websocket_flow.py`
- Test: `tests/test_command_dataset.py`

**Interfaces:**
- Produces: `ClipStartRequest(type="clip.start", profileId: str, language: Literal["zh", "en"])`
- Produces: `CalibrationRequest(..., language: Literal["zh", "en"], phraseId: str, phrase: str = "")`
- Produces: `ErrorCode.INVALID_REQUEST` for recoverable typed-request failures.
- Session-ready `parameters.phraseCatalog` uses the canonical catalog records.

- [ ] **Step 1: Write failing WebSocket request tests**

Assert valid `clip.start` echoes `language` in `clip.started` and reaches a stub classifier with that explicit argument. For missing/invalid language, assert the generic `error` event has stage `clip`, code `INVALID_REQUEST`, and `recoverable: true`; no bytes are accepted for classification, and patched decode/classifier sentinels are not called.

- [ ] **Step 2: Write failing catalog-authority tests**

Assert a registered calibration request sends `phraseId` and the persisted metadata uses catalog text, language, and intent even if no intent/text is supplied. Assert a phrase-language mismatch fails. Assert `phraseId: "UNKNOWN"` requires a non-empty free-form `phrase`, stores `intent: UNKNOWN`, and retains selected language.

- [ ] **Step 3: Run WebSocket tests and verify RED**

Run: `pytest -q tests/test_websocket_flow.py tests/test_command_dataset.py`

Expected: FAIL because clip language is untyped and calibration trusts independent fields.

- [ ] **Step 4: Implement typed requests and server resolution**

Parse `ClipStartRequest` before resetting stream state. On validation failure clear pending state and send a recoverable `ErrorEvent` with stage `clip` and code `INVALID_REQUEST`. Resolve registered calibration via `PhraseCatalog.by_id()` and derive canonical fields. Treat `UNKNOWN` as the only path that accepts free-form phrase text.

- [ ] **Step 5: Expose the canonical catalog in `session.ready`**

Every backend exposes its active catalog; Torch uses its checkpoint catalog and prototype/fake use the repository catalog. Include canonical records in session parameters so the browser has no duplicate phrase definitions.

- [ ] **Step 6: Validate manifest language**

Known samples use the catalog language. UNKNOWN source metadata must contain `zh` or `en`; exclude or fail with an explicit validation error rather than producing a blank language row.

- [ ] **Step 7: Run Task 3 tests and verify GREEN**

Run: `pytest -q tests/test_websocket_flow.py tests/test_command_dataset.py`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add backend/schemas.py api/websocket.py backend/main.py command/dataset.py tests/test_websocket_flow.py tests/test_command_dataset.py
git commit -m "feat: enforce bilingual websocket contracts"
```

---

### Task 4: Browser Language Selection and Catalog-Driven Recording

**Files:**
- Modify: `frontend/index.html`
- Modify: `frontend/websocket.js`
- Modify: `frontend/styles.css`
- Modify: `tests/e2e/camera.spec.js`
- Test: `tests/test_websocket_flow.py`

**Interfaces:**
- Runtime select: `#recognition-language` with `zh` and `en`.
- Recording select: `#calibration-phrase-id`, populated from `session.ready.parameters.phraseCatalog` and filtered by `#calibration-language`.
- UNKNOWN text input: `#calibration-unknown-phrase`, enabled only for the UNKNOWN option.

- [ ] **Step 1: Write failing served-page and browser tests**

Assert the three controls exist, the old manual intent control is absent, changing calibration language filters the phrase options, choosing a known phrase disables free text, choosing UNKNOWN enables and requires it, and runtime `clip.start` contains the recognition language.

- [ ] **Step 2: Run browser-facing tests and verify RED**

Run: `pytest -q tests/test_websocket_flow.py -k frontend && npm run test:e2e`

Expected: FAIL because the new controls and payloads do not exist.

- [ ] **Step 3: Implement the controls without duplicating catalog data**

Store `phraseCatalog` from `session.ready`, render phrase options for the selected calibration language, append `UNKNOWN`, and send `{type, profileId, language, phraseId, phrase, scope}`. Send `{type: "clip.start", profileId, language}` for recognition. Keep all visible copy in English for the submission UI while option labels may show `中文` and `English`.

- [ ] **Step 4: Run browser-facing tests and verify GREEN**

Run: `pytest -q tests/test_websocket_flow.py -k frontend && npm run test:e2e`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add frontend/index.html frontend/websocket.js frontend/styles.css tests/e2e/camera.spec.js tests/test_websocket_flow.py
git commit -m "feat: add bilingual recording and recognition controls"
```

---

### Task 5: Submission Copy, Full Verification, and ROCm Handoff

**Files:**
- Modify: `README.md`
- Modify: `submission/README.md`
- Modify: `submission/demo-video-script.md`
- Modify: `submission/pull-request-description.md`
- Modify: `tests/test_submission_docs.py`
- Modify: `tests/test_contest_bundle.py`

**Interfaces:**
- Documents must describe one four-phrase bilingual catalog, explicit language selection, selected-language softmax, exact phrase output, heuristic UNKNOWN rejection, and ROCm-only Torch execution.

- [ ] **Step 1: Write failing documentation assertions**

Assert public English materials name both English phrases, state that language is selected by the user rather than inferred, and do not describe the current catalog as two Chinese phrases.

- [ ] **Step 2: Run documentation tests and verify RED**

Run: `pytest -q tests/test_submission_docs.py tests/test_contest_bundle.py`

Expected: FAIL on stale two-phrase copy.

- [ ] **Step 3: Update English documentation and recording counts**

State that official evidence requires 15 independent takes for each of four phrases plus at least 15 unrelated clips spanning both selected languages. Do not claim bilingual accuracy until the final report exists.

- [ ] **Step 4: Run documentation tests and verify GREEN**

Run: `pytest -q tests/test_submission_docs.py tests/test_contest_bundle.py`

Expected: PASS.

- [ ] **Step 5: Run local quality gates**

Run:

```bash
ruff check .
pytest -q
npm run test:e2e
python scripts/build_contest_bundle.py --help
```

Expected: all available local tests pass; ROCm-only tests skip locally with explicit reasons.

- [ ] **Step 6: Run Radeon verification**

Synchronize the committed branch to Radeon, then run the existing smoke script and focused real Torch WebSocket test. Rebuild manifests and checkpoint after English recordings exist; do not reuse the v1/two-class checkpoint.

- [ ] **Step 7: Commit Task 5**

```bash
git add README.md submission tests/test_submission_docs.py tests/test_contest_bundle.py
git commit -m "docs: describe bilingual fixed phrase workflow"
```

- [ ] **Step 8: Request final code review**

Review all commits after `c70c18c` against the approved design and this plan. Fix Critical and Important findings before recording official evidence.
