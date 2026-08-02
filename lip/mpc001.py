import configparser
import logging
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from PIL import Image

from backend.config import Settings
from backend.schemas import LipReadingCandidate
from lip.base import LipInferenceCancelled, MouthWindow

HYPOTHESIS_RE = re.compile(r"^hyp:\s*(?P<text>.*)\s*$", re.MULTILINE)
logger = logging.getLogger(__name__)


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
        decode = self._decode_settings()
        logger.info(
            "mpc001 reader initialized reader=%s language=%s repo=%s config=%s lm_weight=%s beam_size=%s ctc_weight=%s",
            self.name,
            self.language,
            self.repo_dir,
            self.config_path,
            decode.get("lm_weight"),
            decode.get("beam_size"),
            decode.get("ctc_weight"),
        )

    def predict(self, window: MouthWindow, cancel_event: object | None = None) -> LipReadingCandidate:
        if not window.frames:
            raise ValueError("empty mouth window cannot be sent to mpc001 inference")
        if _cancelled(cancel_event):
            raise LipInferenceCancelled("mpc001 predict cancelled before writing frames")
        started = perf_counter()
        duration_ms = window.frames[-1].received_at_ms - window.frames[0].received_at_ms
        logger.info(
            "mpc001 predict started reader=%s language=%s session_id=%s window=%s-%s frames=%s duration_ms=%s",
            self.name,
            self.language,
            window.session_id,
            window.start_sequence,
            window.end_sequence,
            len(window.frames),
            duration_ms,
        )
        with tempfile.TemporaryDirectory(prefix=f"silent-vision-{window.session_id}-") as temp_dir:
            frames_path = Path(temp_dir) / f"{self.name}-{window.start_sequence}-{window.end_sequence}.npy"
            self._write_window_frames(window, frames_path)
            completed = self._run_inference(frames_path, cancel_event=cancel_event)
        text = self._parse_hypothesis(completed.stdout)
        logger.info(
            "mpc001 predict completed reader=%s language=%s latency_ms=%s text_len=%s text=%s",
            self.name,
            self.language,
            int((perf_counter() - started) * 1000),
            len(text),
            text if self.settings.log_transcripts else "<redacted>",
        )
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

    def _decode_settings(self) -> dict[str, str]:
        parser = configparser.ConfigParser()
        parser.read(self.config_path)
        if not parser.has_section("decode"):
            return {}
        return {key: value for key, value in parser.items("decode")}

    def _write_window_frames(self, window: MouthWindow, frames_path: Path) -> None:
        frames = np.stack([frame.image for frame in window.frames]).astype("uint8", copy=False)
        np.save(frames_path, frames)
        if self.settings.debug_dump_windows:
            self._dump_debug_window(window, frames)
        logger.debug(
            "mpc001 frames written reader=%s path=%s shape=%s dtype=%s min=%s max=%s mean=%.2f",
            self.name,
            frames_path,
            frames.shape,
            frames.dtype,
            int(frames.min()),
            int(frames.max()),
            float(frames.mean()),
        )

    def _dump_debug_window(self, window: MouthWindow, frames: np.ndarray) -> None:
        target_dir = self.settings.debug_window_dir or (self.settings.persistence_root / "logs" / "mouth-windows")
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{window.session_id}-{self.name}-{window.start_sequence}-{window.end_sequence}"
        npy_path = target_dir / f"{stem}.npy"
        png_path = target_dir / f"{stem}.png"
        np.save(npy_path, frames)
        _write_contact_sheet(frames, png_path)
        duration_ms = window.frames[-1].received_at_ms - window.frames[0].received_at_ms
        logger.info(
            "debug mouth window dumped reader=%s npy=%s png=%s frames=%s duration_ms=%s",
            self.name,
            npy_path,
            png_path,
            len(frames),
            duration_ms,
        )

    def _run_inference(self, frames_path: Path, cancel_event: object | None = None) -> subprocess.CompletedProcess[str]:
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
        logger.debug("mpc001 subprocess command reader=%s command=%s", self.name, command)
        process = subprocess.Popen(
            command,
            cwd=self.repo_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        started = perf_counter()
        while True:
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if _cancelled(cancel_event):
                    logger.info("mpc001 subprocess cancelled reader=%s pid=%s", self.name, process.pid)
                    _terminate_process(process)
                    raise LipInferenceCancelled(f"mpc001 subprocess cancelled for {self.name}")
                elapsed = perf_counter() - started
                if elapsed > self.settings.mpc001_timeout_seconds:
                    logger.error(
                        "mpc001 subprocess timed out reader=%s timeout_seconds=%s pid=%s",
                        self.name,
                        self.settings.mpc001_timeout_seconds,
                        process.pid,
                    )
                    _terminate_process(process)
                    raise subprocess.TimeoutExpired(command, self.settings.mpc001_timeout_seconds)
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if completed.returncode != 0:
            logger.error(
                "mpc001 subprocess failed reader=%s returncode=%s stdout_tail=%r stderr_tail=%r",
                self.name,
                completed.returncode,
                _tail(completed.stdout, enabled=self.settings.log_transcripts),
                _tail(completed.stderr, enabled=True),
            )
            raise subprocess.CalledProcessError(
                completed.returncode,
                command,
                output=completed.stdout,
                stderr=completed.stderr,
            )
        if completed.stderr.strip():
            logger.debug("mpc001 subprocess stderr reader=%s stderr_tail=%r", self.name, _tail(completed.stderr, True))
        return completed

    def _parse_hypothesis(self, stdout: str) -> str:
        match = HYPOTHESIS_RE.search(stdout)
        if not match:
            raise RuntimeError(f"mpc001 output did not contain a 'hyp:' line: {stdout[-1000:]}")
        return match.group("text").strip()


def _cancelled(cancel_event: object | None) -> bool:
    if cancel_event is None:
        return False
    is_set = getattr(cancel_event, "is_set", None)
    return bool(callable(is_set) and is_set())


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    _signal_process_tree(process, signal.SIGTERM)
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _signal_process_tree(process, signal.SIGKILL)
        process.communicate()


def _signal_process_tree(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        return
    except Exception:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _tail(value: str | None, enabled: bool, limit: int = 1200) -> str:
    if not enabled:
        return "<redacted>"
    if not value:
        return ""
    return value[-limit:]


def _write_contact_sheet(frames: np.ndarray, path: Path, columns: int = 15) -> None:
    if frames.ndim != 3:
        raise ValueError(f"expected [T,H,W] frames, got {frames.shape}")
    rows = int(np.ceil(frames.shape[0] / columns))
    height = frames.shape[1]
    width = frames.shape[2]
    sheet = np.zeros((rows * height, columns * width), dtype=np.uint8)
    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns
        sheet[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    Image.fromarray(sheet, mode="L").save(path)
