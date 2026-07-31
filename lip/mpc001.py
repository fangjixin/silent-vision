import configparser
import os
import re
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from backend.config import Settings
from backend.schemas import LipReadingCandidate
from lip.base import MouthWindow

HYPOTHESIS_RE = re.compile(r"^hyp:\s*(?P<text>.*)\s*$", re.MULTILINE)


class MPC001LipReader:
    name: str
    language: str

    def __init__(
        self,
        settings: Settings,
        *,
        model_name: Literal["avhubert", "cmlr"],
        language: Literal["en", "zh"],
        config_path: Path,
    ) -> None:
        self.settings = settings
        self.name = model_name
        self.language = language
        self.repo_dir = settings.mpc001_repo_path
        self.config_path = config_path
        self.runner_path = settings.mpc001_mouth_runner_path
        self._validate_assets()

    def predict(self, window: MouthWindow) -> LipReadingCandidate:
        if not window.frames:
            raise ValueError("empty mouth window cannot be sent to mpc001 inference")
        started = perf_counter()
        with tempfile.TemporaryDirectory(prefix=f"silent-vision-{window.session_id}-") as temp_dir:
            frames_path = Path(temp_dir) / f"{self.name}-{window.start_sequence}-{window.end_sequence}.npy"
            self._write_window_frames(window, frames_path)
            completed = self._run_inference(frames_path)
        text = self._parse_hypothesis(completed.stdout)
        return LipReadingCandidate(
            model=self.name,
            language=self.language,
            text=text,
            confidence=None,
            rawScore=None,
            latencyMs=int((perf_counter() - started) * 1000),
        )

    def _validate_assets(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"missing mpc001 repo at {self.repo_dir}; clone mpc001 "
                "Visual_Speech_Recognition_for_Multiple_Languages into this directory"
            )
        if not self.runner_path.exists():
            raise FileNotFoundError(f"missing Silent Vision mpc001 mouth runner at {self.runner_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"missing mpc001 config file at {self.config_path}")
        for required_repo_file in ("pipelines/model.py", "pipelines/data/transforms.py"):
            path = self.repo_dir / required_repo_file
            if not path.exists():
                raise FileNotFoundError(f"missing mpc001 repo file at {path}")
        missing = [path for path in self._required_files_from_config() if not path.exists()]
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"missing mpc001 benchmark assets for {self.config_path.name}: {formatted}. "
                "Download and extract the corresponding model zoo package under the mpc001 repo benchmarks directory."
            )

    def _required_files_from_config(self) -> list[Path]:
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        required: list[Path] = []
        for key in ("model_path", "model_conf", "rnnlm", "rnnlm_conf"):
            value = parser.get("model", key, fallback="")
            if value:
                required.append((self.repo_dir / value).resolve())
        return required

    def _write_window_frames(self, window: MouthWindow, frames_path: Path) -> None:
        frames = np.stack([frame.image for frame in window.frames]).astype("uint8", copy=False)
        np.save(frames_path, frames)

    def _run_inference(self, frames_path: Path) -> subprocess.CompletedProcess[str]:
        command = [
            self.settings.mpc001_python,
            str(self.runner_path),
            "--repo-dir",
            str(self.repo_dir),
            "--config",
            str(self.config_path),
            "--frames-npy",
            str(frames_path),
            "--gpu-idx",
            str(self.settings.mpc001_gpu_idx),
        ]
        env = os.environ.copy()
        env.setdefault("HYDRA_FULL_ERROR", "1")
        return subprocess.run(
            command,
            cwd=self.repo_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.settings.mpc001_timeout_seconds,
            check=True,
        )

    def _parse_hypothesis(self, stdout: str) -> str:
        match = HYPOTHESIS_RE.search(stdout)
        if not match:
            raise RuntimeError(f"mpc001 output did not contain a 'hyp:' line: {stdout[-1000:]}")
        return match.group("text").strip()
