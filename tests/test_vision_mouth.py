import numpy as np
import pytest

from backend.config import Settings
from backend.schemas import ErrorCode
from tests.conftest import make_jpeg
from vision.face import FakeFaceDetector, FrameDecodeError, create_face_detector, decode_jpeg_frame
from vision.mouth import crop_mouth


def test_decode_jpeg_frame_returns_rgb_image():
    image = decode_jpeg_frame(make_jpeg(width=320, height=240), Settings())
    assert image.shape == (240, 320, 3)
    assert image.dtype == np.uint8


def test_decode_rejects_oversize_payload():
    settings = Settings(max_jpeg_bytes=1024)
    with pytest.raises(FrameDecodeError) as exc_info:
        decode_jpeg_frame(make_jpeg(width=1024, height=1024), settings)
    assert exc_info.value.code == ErrorCode.FRAME_TOO_LARGE


def test_crop_mouth_returns_normalized_box_and_96_image():
    image = np.full((200, 300, 3), 128, dtype=np.uint8)
    landmarks = [(0.45, 0.55), (0.55, 0.55), (0.50, 0.62), (0.50, 0.50)]
    result = crop_mouth(image, landmarks, mouth_size=96)
    assert result.image.shape == (96, 96)
    assert result.image.dtype == np.uint8
    assert 0.0 <= result.box.x <= 1.0
    assert 0.0 <= result.box.y <= 1.0
    assert result.box.width > 0
    assert result.box.height > 0


def test_fake_face_detector_returns_one_face():
    detector = FakeFaceDetector()
    result = detector.detect(np.full((200, 300, 3), 128, dtype=np.uint8))
    assert result.face_detected is True
    assert result.face_count == 1
    assert len(result.landmarks) >= 4


def test_create_face_detector_uses_fake_backend_by_default():
    detector = create_face_detector(Settings(model_backend="fake"))
    assert isinstance(detector, FakeFaceDetector)
