from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_backend: Literal["fake", "real"] = "fake"
    persistence_root: Path = Path("/workspace/persistence/silent-vision")
    capture_fps: int = Field(default=25, ge=1, le=60)
    window_frames: int = Field(default=75, ge=1)
    inference_stride: int = Field(default=25, ge=1)
    mouth_size: int = Field(default=96, ge=16)
    max_jpeg_bytes: int = Field(default=1_048_576, ge=1024)
    max_frame_width: int = Field(default=1920, ge=64)
    max_frame_height: int = Field(default=1080, ge=64)
    model_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    allowed_origins: str = "http://localhost:8000"
    log_transcripts: bool = False
    host: str = "127.0.0.1"
    port: int = 8000
    enable_model_health: bool = False

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
    def allowed_origin_set(self) -> set[str]:
        return {item.strip() for item in self.allowed_origins.split(",") if item.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()
