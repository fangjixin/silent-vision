from pathlib import Path

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


def test_readme_is_a_complete_reproduction_guide():
    text = Path("README.md").read_text()
    assert all(heading in text for heading in REQUIRED_README_HEADINGS)
    assert "COMMAND_BACKEND=torch" in text
    assert "build_command_manifest.py" in text
    assert "prototype mode is for calibration" in text.lower()


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
