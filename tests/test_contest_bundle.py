import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_contest_bundle import build_bundle


def test_bundle_contains_complete_runtime_and_public_materials(tmp_path):
    target = tmp_path / "submissions" / "track1-silent-vision"

    copied = build_bundle(Path.cwd(), target)

    assert target.joinpath("backend/main.py").exists()
    assert target.joinpath("frontend/websocket.js").exists()
    assert target.joinpath("README.md").exists()
    assert target.joinpath("docs/submission/project-profile-source.md").exists()
    assert target.joinpath("scripts/generate_submission_assets.py").exists()
    assert target.joinpath("submission/Silent-Vision-Project-Profile.pdf").exists()
    assert copied


def test_bundle_excludes_internal_and_sensitive_paths(tmp_path):
    target = tmp_path / "bundle"

    build_bundle(Path.cwd(), target)

    assert not target.joinpath("docs/superpowers").exists()
    assert not target.joinpath(".env").exists()
    assert not target.joinpath("models").exists()
    assert not target.joinpath(".git").exists()
    assert not any(path.is_symlink() for path in target.rglob("*"))


def test_bundle_rejects_an_allowlisted_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    (source / "agent").mkdir()
    (source / "agent" / "linked.py").symlink_to(source / "README.md")

    with pytest.raises(ValueError, match="symlink"):
        build_bundle(source, tmp_path / "bundle")


def test_bundle_excludes_secrets_inside_an_allowlisted_directory(tmp_path):
    source = tmp_path / "source"
    secret = source / "backend" / "secrets" / "token.txt"
    secret.parent.mkdir(parents=True)
    secret.write_text("do not publish", encoding="utf-8")

    build_bundle(source, tmp_path / "bundle")

    assert not (tmp_path / "bundle" / "backend" / "secrets" / "token.txt").exists()


def test_bundle_rejects_a_symlink_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("# Safe source\n", encoding="utf-8")
    destination_target = tmp_path / "destination-target"
    destination_target.mkdir()
    destination = tmp_path / "bundle"
    destination.symlink_to(destination_target, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink destination"):
        build_bundle(source, destination)
