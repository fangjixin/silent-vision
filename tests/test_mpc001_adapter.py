import sys
from pathlib import Path

import pytest

from backend.config import Settings
from lip.base import MouthWindow
from lip.mpc001 import MPC001LipReader
from tests.test_lip_inference import _window


def _fake_mpc001_repo(tmp_path: Path, text: str = "hello there") -> Path:
    repo = tmp_path / "Visual_Speech_Recognition_for_Multiple_Languages"
    configs = repo / "configs"
    model_dir = repo / "benchmarks" / "LRS3" / "models" / "LRS3_V_WER19.1"
    lm_dir = repo / "benchmarks" / "LRS3" / "language_models" / "lm_en_subword"
    pipelines = repo / "pipelines" / "data"
    configs.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    lm_dir.mkdir(parents=True)
    pipelines.mkdir(parents=True)
    (model_dir / "model.pth").write_bytes(b"model")
    (model_dir / "model.json").write_text("{}")
    (lm_dir / "model.pth").write_bytes(b"lm")
    (lm_dir / "model.json").write_text("{}")
    (repo / "pipelines" / "model.py").write_text("# fake model module\n")
    (pipelines / "transforms.py").write_text("# fake transforms module\n")
    (configs / "LRS3_V_WER19.1.ini").write_text(
        "\n".join(
            [
                "[input]",
                "modality=video",
                "v_fps=25",
                "[model]",
                "model_path=benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth",
                "model_conf=benchmarks/LRS3/models/LRS3_V_WER19.1/model.json",
                "rnnlm=benchmarks/LRS3/language_models/lm_en_subword/model.pth",
                "rnnlm_conf=benchmarks/LRS3/language_models/lm_en_subword/model.json",
                "[decode]",
                "beam_size=40",
            ]
        )
    )
    return repo


def _fake_runner(tmp_path: Path, text: str = "hello there") -> Path:
    runner = tmp_path / "fake_mpc001_mouth_infer.py"
    runner.write_text(
        "import sys\n"
        "assert '--repo-dir' in sys.argv\n"
        "assert '--config' in sys.argv\n"
        "assert '--frames-npy' in sys.argv\n"
        f"print('hyp: {text}')\n"
    )
    return runner


def test_mpc001_reader_invokes_mouth_runner_and_parses_hypothesis(tmp_path: Path):
    repo = _fake_mpc001_repo(tmp_path, text="hello there")
    runner = _fake_runner(tmp_path, text="hello there")
    settings = Settings(
        model_backend="real",
        mpc001_repo_dir=repo,
        mpc001_runner_path=runner,
        mpc001_python=sys.executable,
        mpc001_timeout_seconds=10,
    )
    reader = MPC001LipReader(
        settings,
        model_name="avhubert",
        language="en",
        config_path=settings.mpc001_english_config_path,
    )

    candidate = reader.predict(_window())

    assert candidate.model == "avhubert"
    assert candidate.language == "en"
    assert candidate.text == "hello there"
    assert candidate.confidence is None


def test_mpc001_reader_reports_missing_repo_with_actionable_message(tmp_path: Path):
    settings = Settings(model_backend="real", persistence_root=tmp_path / "sv")

    with pytest.raises(FileNotFoundError, match="clone mpc001"):
        MPC001LipReader(
            settings,
            model_name="cmlr",
            language="zh",
            config_path=settings.mpc001_chinese_config_path,
        )


def test_mpc001_reader_rejects_empty_windows(tmp_path: Path):
    repo = _fake_mpc001_repo(tmp_path)
    runner = _fake_runner(tmp_path)
    settings = Settings(
        model_backend="real",
        mpc001_repo_dir=repo,
        mpc001_runner_path=runner,
        mpc001_python=sys.executable,
    )
    reader = MPC001LipReader(
        settings,
        model_name="avhubert",
        language="en",
        config_path=settings.mpc001_english_config_path,
    )

    with pytest.raises(ValueError, match="empty mouth window"):
        reader.predict(MouthWindow(session_id="test", frames=[], start_sequence=1, end_sequence=0))
