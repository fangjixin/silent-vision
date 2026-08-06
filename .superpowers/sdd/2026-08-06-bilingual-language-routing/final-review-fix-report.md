# Final review fix report

Date: 2026-08-06

Implementation commit: `4208941c4a3aec5a534204840649a13c8b30b2c1`

## Implemented fixes

- Checkpoint schema validation rejects non-finite, zero-length, and materially
  non-unit class centroids. Runtime rejection fails closed on non-finite
  probability, similarity, or distance and reports a finite sanitized distance.
- One shared strict dataset validator authenticates the inventory, catalog, seed,
  all five role hashes and record schemas, inventory counts, deterministic split
  membership, official per-role minimums, bilingual UNKNOWN coverage, and global
  sample-ID/ROI-hash disjointness.
- Training requires all five manifest roles, uses evaluation roles only for
  validation, and binds the inventory SHA-256, catalog SHA-256, seed, all five
  role hashes, and explicit evidentiary status into the checkpoint.
- Final evaluation requires the catalog, inventory, and all five manifests. It
  rejects renamed, mixed, overlapping, non-evidentiary, or lineage-mismatched
  inputs before ROCm inference and emits an explicit evidence status.
- `.env.example`, CLI examples, the runbook, design documentation, synthetic
  checkpoints, and regression tests were updated for the strict contract.

## TDD evidence

Focused RED runs captured before the corresponding source changes:

- `.venv/bin/python -m pytest -q tests/test_phrase_checkpoint.py -k 'corrupt_class_centroids or schema_validation'`
  - Result: 5 failed, 1 passed, 13 deselected. Each NaN/inf/zero/non-unit centroid
    case failed because validation did not raise.
- `.venv/bin/python -m pytest -q tests/test_phrase_checkpoint.py -k 'corrupt_class_centroids or schema_validation' tests/test_phrase_runtime.py -k 'non_finite_runtime_scores'`
  - Result: 4 failed, 41 deselected. NaN/inf scores either passed or returned the
    wrong rejection reason.
- `.venv/bin/python -m pytest -q tests/test_command_dataset.py -k evidence_validation`
  - Result: collection error because `MANIFEST_ROLES` and
    `validate_dataset_bundle` did not exist.
- `.venv/bin/python -m pytest -q tests/test_phrase_checkpoint.py -k complete_explicit_evidence_lineage`
  - Result: 3 failed, 19 deselected because incomplete/invalid lineage was
    accepted.
- `.venv/bin/python -m pytest -q tests/test_phrase_evaluation.py -k 'signature or non_evidentiary_or_mismatched or cli_help'`
  - Result: collection error because `_validate_checkpoint_lineage` did not
    exist.
- `.venv/bin/python -m pytest -q tests/test_command_dataset.py -k mixed_roles`
  - Result: 1 failed, 17 deselected because swapped calibration/evaluation roles
    with edited inventory metadata were accepted.

Focused GREEN runs:

- `.venv/bin/python -m pytest -q tests/test_phrase_checkpoint.py -k 'corrupt_class_centroids or schema_validation'`
  - Result: 6 passed, 13 deselected.
- `.venv/bin/python -m pytest -q tests/test_phrase_runtime.py -k 'non_finite_runtime_scores'`
  - Result: 4 passed, 22 deselected.
- `.venv/bin/python -m pytest -q tests/test_command_dataset.py`
  - Result after the final adversarial addition: 18 passed.
- `.venv/bin/python -m pytest -q tests/test_phrase_evaluation.py -k 'signature or non_evidentiary_or_mismatched or rejects_non_final or cli_help'`
  - Result: 5 passed, 6 deselected.

## Final verification

- `.venv/bin/ruff check .`
  - Result: `All checks passed!`
- `.venv/bin/python -m pytest -q`
  - Result: `195 passed, 22 skipped in 27.73s`.
  - The skips are existing environment-gated Torch/ROCm tests. No ROCm or final
    evaluation evidence is claimed by this local run.
- `.venv/bin/python -m pytest -q tests/test_phrase_checkpoint.py tests/test_phrase_runtime.py tests/test_phrase_training.py tests/test_command_dataset.py tests/test_phrase_evaluation.py tests/test_deployment_files.py tests/test_submission_docs.py tests/test_contest_bundle.py`
  - Result: `130 passed, 15 skipped in 22.91s`.
- Temporary local backend:
  `COMMAND_BACKEND=fake ALLOWED_ORIGINS=http://127.0.0.1:8000 .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`
  followed by `npm run test:e2e`.
  - Result: `3 passed (10.2s)`; the backend was then stopped cleanly.
- `.venv/bin/python scripts/train_command_classifier.py --help` and
  `.venv/bin/python scripts/validate_command_classifier.py --help`
  - Result: both exited 0 and exposed the new complete-lineage arguments.
- `bash -n scripts/*.sh`
  - Result: exit 0.
- `.venv/bin/python scripts/build_contest_bundle.py`
  - Result: `Built 86 files in .../dist/contest/submissions/track1-silent-vision`.
- `git diff --check`
  - Result: exit 0. This checks textual patch whitespace; it does not inspect
    binary contents. No generated PDF or other binary file was modified in this
    fix.

ROCm training, Radeon inference, and an official final evaluation were not run
and remain pending on the trusted Radeon host with real evidentiary recordings.
