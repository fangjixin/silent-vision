from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_backend: Literal["fake", "real"] = "fake"
    recognition_mode: Literal["command", "transcription"] = "command"
    command_backend: Literal["fake", "torch", "prototype"] = "fake"
    persistence_root: Path = Path("/workspace/persistent/silent-vision")
    capture_fps: int = Field(default=25, ge=1, le=60)
    capture_countdown_seconds: int = Field(default=3, ge=0, le=10)
    command_clip_fps: int = Field(default=25, ge=1, le=60)
    command_clip_min_seconds: float = Field(default=2.0, ge=0.2)
    command_clip_max_seconds: float = Field(default=5.0, ge=0.2)
    command_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    command_top1_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    command_classifier_checkpoint: Path | None = None
    command_fallback_transcription: bool = False
    command_feature_dim: int = Field(default=256, ge=1)
    prototype_feature_dim: int = Field(default=128, ge=1)
    prototype_confidence_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    prototype_top1_margin: float = Field(default=0.12, ge=0.0, le=1.0)
    prototype_prefer_personal: bool = True
    allow_global_profile_write: bool = False
    window_frames: int = Field(default=75, ge=1)
    inference_stride: int = Field(default=25, ge=1)
    mouth_size: int = Field(default=96, ge=16)
    max_jpeg_bytes: int = Field(default=1_048_576, ge=1024)
    max_frame_width: int = Field(default=1920, ge=64)
    max_frame_height: int = Field(default=1080, ge=64)
    model_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    allowed_origins: str = "http://localhost:8000"
    log_transcripts: bool = False
    log_level: str = "INFO"
    debug_dump_windows: bool = False
    debug_window_dir: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    enable_model_health: bool = False
    mpc001_repo_dir: Path | None = None
    mpc001_python: str = "python"
    mpc001_gpu_idx: int = 0
    mpc001_timeout_seconds: int = Field(default=120, ge=1)
    mpc001_english_config: Path | None = None
    mpc001_chinese_config: Path | None = None
    mpc001_runner_path: Path | None = None

    @property
    def avhubert_checkpoint(self) -> Path:
        return self.persistence_root / "models" / "avhubert" / "model.pt"

    @property
    def cmlr_checkpoint(self) -> Path:
        return self.persistence_root / "models" / "cmlr" / "model.pth"

    @property
    def cmlr_language_model(self) -> Path:
        return self.persistence_root / "models" / "cmlr" / "language-model.pth"

    @property
    def minicpm_model_path(self) -> Path:
        return self.persistence_root / "models" / "minicpm-o-4_5"

    @property
    def mpc001_repo_path(self) -> Path:
        return self.mpc001_repo_dir or (
            self.persistence_root / "repos" / "Visual_Speech_Recognition_for_Multiple_Languages"
        )

    @property
    def mpc001_english_config_path(self) -> Path:
        return self.mpc001_english_config or (self.mpc001_repo_path / "configs" / "LRS3_V_WER19.1.ini")

    @property
    def mpc001_chinese_config_path(self) -> Path:
        return self.mpc001_chinese_config or (self.mpc001_repo_path / "configs" / "CMLR_V_WER8.0.ini")

    @property
    def mpc001_mouth_runner_path(self) -> Path:
        return self.mpc001_runner_path or (Path(__file__).resolve().parent.parent / "scripts" / "mpc001_mouth_infer.py")

    @property
    def allowed_origin_set(self) -> set[str]:
        return {item.strip() for item in self.allowed_origins.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
