import io

import pytest

from backend.config import Settings
from backend.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(model_backend="fake")


@pytest.fixture
def app(settings: Settings):
    return create_app(settings)


def make_jpeg(width: int = 320, height: int = 240) -> bytes:
    import numpy as np
    from PIL import Image

    image = np.full((height, width, 3), 127, dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buffer, format="JPEG")
    return buffer.getvalue()
