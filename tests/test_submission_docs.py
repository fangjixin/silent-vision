from pathlib import Path

from pypdf import PdfReader

REQUIRED_README_HEADINGS = [
    "## What Silent Vision Does",
    "## Creator Workflow",
    "## Architecture",
    "## AMD Radeon and ROCm",
    "## Requirements and Dependencies",
    "## Train the Command Classifier",
    "## GPU-Only Startup",
    "## Verification",
    "## Privacy and Limitations",
    "## Submission Materials",
]

CONTEST_PREFIX = "submissions/track1-silent-vision/"


def _pdf_text(path: str) -> str:
    reader = PdfReader(path)
    return " ".join(
        " ".join((page.extract_text() or "").split()) for page in reader.pages
    )


def test_readme_is_a_complete_reproduction_guide():
    text = Path("README.md").read_text()
    assert all(heading in text for heading in REQUIRED_README_HEADINGS)
    assert "COMMAND_BACKEND=torch" in text
    assert "build_command_manifest.py" in text
    assert "prototype mode is for calibration" in text.lower()


def test_readme_truthfully_scopes_submission_verification_and_artifacts():
    text = Path("README.md").read_text()
    normalized_text = " ".join(text.split())
    assert "cd /path/to/silent-vision" in text
    assert "event-hosted Radeon workspace example" in text
    assert "\n.venv/bin/ruff check .\n" not in text
    assert (
        ".venv/bin/ruff check scripts/generate_submission_assets.py "
        "scripts/build_contest_bundle.py tests/test_submission_docs.py "
        "tests/test_contest_bundle.py"
    ) in text
    assert (
        "The generated project profile PDF and poster PDF/PNG are complete."
        in normalized_text
    )


def test_submission_copy_avoids_unverified_marketing_claims():
    paths = [
        Path("README.md"),
        *Path("submission").glob("*.md"),
        *Path("docs/submission").glob("*.md"),
    ]
    banned = {
        "revolutionary",
        "game-changing",
        "cutting-edge",
        "next-generation",
        "harness the power of ai",
    }
    combined = "\n".join(path.read_text().lower() for path in paths if path.exists())
    assert not any(phrase in combined for phrase in banned)


def test_submission_index_names_required_materials():
    text = Path("submission/README.md").read_text()
    assert "Silent-Vision-Project-Profile.pdf" in text
    assert "Silent-Vision-Poster.pdf" in text
    assert "Silent-Vision-Poster.png" in text
    assert "demo-video-script.md" in text
    assert "https://github.com/fangjixin/silent-vision" in text
    assert "Track 1, Jixin Fang, Silent Vision" in text


def test_submission_index_marks_generated_artifacts_complete():
    text = Path("submission/README.md").read_text()
    expected_rows = [
        "| Project profile PDF | [`Silent-Vision-Project-Profile.pdf`](Silent-Vision-Project-Profile.pdf) | Complete |",
        "| Poster PDF | [`Silent-Vision-Poster.pdf`](Silent-Vision-Poster.pdf) | Complete |",
        "| Poster PNG | [`Silent-Vision-Poster.png`](Silent-Vision-Poster.png) | Complete |",
    ]
    assert all(row in text for row in expected_rows)


def test_pr_description_uses_final_contest_paths_and_completed_artifacts():
    text = Path("submission/pull-request-description.md").read_text()
    expected_paths = [
        "README.md",
        "docs/submission/project-profile-source.md",
        "submission/Silent-Vision-Project-Profile.pdf",
        "docs/submission/poster-copy.md",
        "submission/Silent-Vision-Poster.pdf",
        "submission/Silent-Vision-Poster.png",
        "submission/demo-video-script.md",
    ]
    assert all(f"`{CONTEST_PREFIX}{path}`" in text for path in expected_paths)
    assert "- [x] Generated project profile PDF:" in text
    assert "- [x] Generated poster PDF and PNG:" in text
    assert "- [ ] Demo video:" in text
    assert "](../" not in text
    assert "](Silent-Vision" not in text


def test_reviewed_reference_copy_matches_generated_high_risk_status():
    profile_anchor = (
        "Pending: final Radeon checkpoint, held-out validation report, selected "
        "environment record, Creator Mode actions, and the 3-5 minute end-to-end "
        "video. No accuracy or latency number is published."
    )
    poster_anchor = (
        "Final Radeon run, trained checkpoint, validation evidence, Creator Mode "
        "actions, and recorded demo are pending."
    )
    profile_markdown = " ".join(
        Path("docs/submission/project-profile-source.md").read_text().split()
    )
    poster_markdown = " ".join(Path("docs/submission/poster-copy.md").read_text().split())
    assert profile_anchor in profile_markdown
    assert profile_anchor in _pdf_text("submission/Silent-Vision-Project-Profile.pdf")
    assert poster_anchor in poster_markdown
    assert poster_anchor in _pdf_text("submission/Silent-Vision-Poster.pdf")


def test_profile_and_poster_pdf_structure():
    profile = PdfReader("submission/Silent-Vision-Project-Profile.pdf")
    poster = PdfReader("submission/Silent-Vision-Poster.pdf")
    assert len(profile.pages) == 6
    assert len(poster.pages) == 1
    profile_text = "\n".join(page.extract_text() or "" for page in profile.pages)
    for phrase in [
        "Target users",
        "System architecture",
        "Model and algorithm",
        "AMD Radeon and ROCm",
    ]:
        assert phrase in profile_text
    assert Path("submission/Silent-Vision-Poster.png").exists()
