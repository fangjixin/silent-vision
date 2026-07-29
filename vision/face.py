from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.config import Settings
from backend.schemas import ErrorCode

MOUTH_LANDMARK_IDS = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
)


class FrameDecodeError(ValueError):
    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class FaceDetectionResult:
    face_detected: bool
    landmarks: list[tuple[float, float]]
    face_count: int


def decode_jpeg_frame(data: bytes, settings: Settings) -> np.ndarray:
    if len(data) > settings.max_jpeg_bytes:
        raise FrameDecodeError(ErrorCode.FRAME_TOO_LARGE, "jpeg payload exceeds MAX_JPEG_BYTES")
    try:
        with Image.open(BytesIO(data)) as image:
            image_rgb = image.convert("RGB")
            width, height = image_rgb.size
            if width > settings.max_frame_width or height > settings.max_frame_height:
                raise FrameDecodeError(
                    ErrorCode.FRAME_TOO_LARGE_DIMENSIONS,
                    "decoded frame dimensions exceed configured limits",
                )
            return np.asarray(image_rgb, dtype=np.uint8)
    except FrameDecodeError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise FrameDecodeError(ErrorCode.INVALID_JPEG, "payload is not a valid jpeg") from exc


class FaceDetector:
    def __init__(self) -> None:
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        result = self._mesh.process(image_rgb)
        faces = result.multi_face_landmarks or []
        if len(faces) != 1:
            return FaceDetectionResult(face_detected=False, landmarks=[], face_count=len(faces))
        landmarks = []
        for index in MOUTH_LANDMARK_IDS:
            point = faces[0].landmark[index]
            landmarks.append((float(point.x), float(point.y)))
        return FaceDetectionResult(face_detected=True, landmarks=landmarks, face_count=1)
