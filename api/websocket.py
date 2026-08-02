import asyncio
import json
import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image, ImageDraw

from backend.schemas import ErrorCode, ErrorEvent, SemanticResult, utc_now
from lip.base import MouthFrame
from session.manager import SessionError, SessionReplacedError
from vision.face import FrameDecodeError, decode_jpeg_frame
from vision.mouth import crop_mouth

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_REUSED_MOUTH_FRAMES = 10


def _event(event_type: str, session_id: str, **payload: object) -> dict[str, object]:
    return {"type": event_type, "sessionId": session_id, "timestamp": utc_now().isoformat(), **payload}


async def _send_error(
    websocket: WebSocket,
    session_id: str,
    stage: str,
    code: ErrorCode,
    message: str,
    recoverable: bool,
) -> None:
    event = ErrorEvent(sessionId=session_id, stage=stage, code=code, message=message, recoverable=recoverable)
    await websocket.send_json(event.model_dump(mode="json"))


def _parse_command(raw: str) -> str | None:
    try:
        loaded: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    command_type = loaded.get("type")
    return command_type if isinstance(command_type, str) else None


def _cleanup_temp(session_id: str) -> None:
    temp_dir = Path("/tmp/silent-vision") / session_id
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def _metrics(active: Any) -> dict[str, object]:
    total = max(1, active.accepted_frame_count)
    return {
        "receivedFps": 0.0,
        "validFrameRatio": min(1.0, len(active.frames) / total),
        "bufferedFrames": len(active.frames),
        "droppedFrames": 0,
    }


def _origin_allowed(origin: str, allowed_origins: set[str]) -> bool:
    return not origin or "*" in allowed_origins or origin in allowed_origins


def _landmark_bounds(landmarks: list[tuple[float, float]]) -> dict[str, float] | None:
    if not landmarks:
        return None
    xs = [x for x, _ in landmarks]
    ys = [y for _, y in landmarks]
    return {
        "minX": min(xs),
        "maxX": max(xs),
        "minY": min(ys),
        "maxY": max(ys),
    }


def _debug_box_value(box: object, key: str) -> float | None:
    if isinstance(box, dict):
        value = box.get(key)
        return float(value) if isinstance(value, int | float) else None
    value = getattr(box, key, None)
    return float(value) if isinstance(value, int | float) else None


def _draw_debug_frame(frame: MouthFrame, size: tuple[int, int]) -> Image.Image:
    if frame.debug_image is None:
        return Image.new("RGB", size, (32, 32, 32))

    image = Image.fromarray(frame.debug_image).resize(size, Image.Resampling.BOX)
    draw = ImageDraw.Draw(image)
    scale_x = size[0]
    scale_y = size[1]

    box = frame.debug_mouth_box
    if box is not None:
        x = _debug_box_value(box, "x")
        y = _debug_box_value(box, "y")
        width = _debug_box_value(box, "width")
        height = _debug_box_value(box, "height")
        if x is not None and y is not None and width is not None and height is not None:
            draw.rectangle(
                [
                    int(x * scale_x),
                    int(y * scale_y),
                    int((x + width) * scale_x),
                    int((y + height) * scale_y),
                ],
                outline=(255, 48, 48),
                width=2,
            )

    for landmark_x, landmark_y in frame.debug_landmarks:
        center_x = int(landmark_x * scale_x)
        center_y = int(landmark_y * scale_y)
        draw.ellipse([center_x - 2, center_y - 2, center_x + 2, center_y + 2], fill=(48, 255, 48))

    draw.text((4, 4), str(frame.sequence), fill=(255, 255, 0))
    return image


def _dump_raw_debug_window(settings: Any, window: Any) -> None:
    if not settings.debug_dump_windows:
        return
    output_dir = settings.debug_window_dir or settings.persistence_root / "logs" / "mouth-windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_size = (160, 120)
    columns = 5
    rows = (len(window.frames) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_size[0] * columns, cell_size[1] * rows), (16, 16, 16))
    for index, frame in enumerate(window.frames):
        tile = _draw_debug_frame(frame, cell_size)
        x = (index % columns) * cell_size[0]
        y = (index // columns) * cell_size[1]
        sheet.paste(tile, (x, y))

    raw_png = output_dir / f"{window.session_id}-raw-{window.start_sequence}-{window.end_sequence}.png"
    sheet.save(raw_png)
    logger.info(
        "debug raw window dumped raw_png=%s frames=%s window=%s-%s",
        raw_png,
        len(window.frames),
        window.start_sequence,
        window.end_sequence,
    )


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    settings = websocket.app.state.settings
    origin = websocket.headers.get("origin", "")
    allowed_origins = settings.allowed_origin_set
    if not _origin_allowed(origin, allowed_origins):
        logger.warning(
            "websocket origin rejected session_id=%s origin=%s allowed_origins=%s",
            session_id,
            origin,
            sorted(allowed_origins),
        )
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("websocket accepted session_id=%s origin=%s", session_id, origin)
    send_lock = asyncio.Lock()

    async def send_json(payload: dict[str, object]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    async def send_error(stage: str, code: ErrorCode, message: str, recoverable: bool) -> None:
        event = ErrorEvent(sessionId=session_id, stage=stage, code=code, message=message, recoverable=recoverable)
        await send_json(event.model_dump(mode="json"))

    try:
        active = websocket.app.state.session_manager.activate(session_id)
    except SessionError:
        await send_error("session", ErrorCode.INVALID_SESSION, "invalid or expired session", False)
        await websocket.close(code=1008)
        return

    async def ensure_current_or_close() -> bool:
        try:
            websocket.app.state.session_manager.ensure_current(active)
        except SessionReplacedError:
            await send_error(
                "session",
                ErrorCode.SESSION_REPLACED,
                "session was replaced by a newer connection",
                False,
            )
            await websocket.close(code=1000)
            return False
        return True

    def is_live_stream(generation: int) -> bool:
        return (
            websocket.app.state.session_manager.is_current(session_id)
            and active.streaming
            and active.stream_generation == generation
        )

    async def cancel_active_inference(reason: str) -> None:
        task = active.active_inference_task
        active.latest_pending_window = None
        if active.inference_cancel_event is not None:
            active.inference_cancel_event.set()
        if task is not None and not task.done():
            logger.info("inference task cancelled session_id=%s reason=%s", session_id, reason)
            task.cancel()
            await asyncio.sleep(0)

    async def run_inference_loop(initial_window: Any, generation: int) -> None:
        current = initial_window
        try:
            while current is not None:
                if not is_live_stream(generation):
                    return
                logger.info(
                    "inference window started session_id=%s window=%s-%s frames=%s",
                    session_id,
                    current.start_sequence,
                    current.end_sequence,
                    len(current.frames),
                )
                await send_json(
                    _event(
                        "inference.started",
                        session_id,
                        startSequence=current.start_sequence,
                        endSequence=current.end_sequence,
                    )
                )
                async with websocket.app.state.inference_lock:
                    if not is_live_stream(generation):
                        return
                    cancel_event = active.inference_cancel_event
                    lip_result = await asyncio.to_thread(
                        websocket.app.state.lip_engine.predict,
                        current,
                        cancel_event,
                    )
                    if not is_live_stream(generation):
                        return
                    logger.info(
                        "lip candidates ready session_id=%s candidates=%s degraded=%s summary=%s",
                        session_id,
                        len(lip_result.candidates),
                        lip_result.degradedModels,
                        [
                            {
                                "model": candidate.model,
                                "language": candidate.language,
                                "textLen": len(candidate.text),
                                "latencyMs": candidate.latencyMs,
                            }
                            for candidate in lip_result.candidates
                        ],
                    )
                    await send_json(
                        _event(
                            "lip.candidates",
                            session_id,
                            candidates=[candidate.model_dump() for candidate in lip_result.candidates],
                            degradedModels=lip_result.degradedModels,
                        )
                    )
                    if not lip_result.candidates:
                        await send_error(
                            "lip",
                            ErrorCode.LIP_MODELS_FAILED,
                            "both lip reading models failed",
                            True,
                        )
                    else:
                        sampled = [frame.image for frame in current.frames[::15]]
                        logger.info(
                            "MiniCPM dispatch session_id=%s sampled_frames=%s frame_stride=%s",
                            session_id,
                            len(sampled),
                            15,
                        )
                        try:
                            semantic = await asyncio.to_thread(
                                websocket.app.state.semantic_interpreter.interpret,
                                lip_result.candidates,
                                sampled,
                                {"frames": len(current.frames)},
                            )
                        except Exception:
                            logger.exception("MiniCPM inference failed session_id=%s", session_id)
                            await send_error(
                                "minicpm",
                                ErrorCode.MINICPM_FAILED,
                                "MiniCPM failed to produce valid semantic JSON",
                                True,
                            )
                            semantic = SemanticResult(
                                language="unknown",
                                text="",
                                confidence=0.0,
                                reason="MiniCPM failure",
                            )
                        if not is_live_stream(generation):
                            return
                        logger.info(
                            "semantic result session_id=%s language=%s confidence=%.3f text_len=%s reason=%s",
                            session_id,
                            semantic.language,
                            semantic.confidence,
                            len(semantic.text),
                            semantic.reason,
                        )
                        await send_json(_event("semantic.result", session_id, **semantic.model_dump()))
                        agent = websocket.app.state.agent_policy.decide(semantic)
                        logger.info(
                            "agent result session_id=%s action=%s language=%s requires_confirmation=%s",
                            session_id,
                            agent.action,
                            agent.language,
                            agent.requiresConfirmation,
                        )
                        await send_json(
                            agent.model_dump(mode="json")
                            | {"sessionId": session_id, "timestamp": utc_now().isoformat()}
                        )
                    if not is_live_stream(generation):
                        return
                current = active.latest_pending_window
                active.latest_pending_window = None
                if current is not None:
                    logger.info(
                        "inference continuing with pending latest window session_id=%s window=%s-%s",
                        session_id,
                        current.start_sequence,
                        current.end_sequence,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("inference task failed session_id=%s", session_id)
            if websocket.app.state.session_manager.is_current(session_id):
                await send_error("inference", ErrorCode.INTERNAL_ERROR, "inference task failed", True)
        finally:
            if active.active_inference_task is asyncio.current_task():
                active.active_inference_task = None
                active.inference_cancel_event = None

    def enqueue_inference(window: Any) -> None:
        task = active.active_inference_task
        if task is not None and not task.done():
            active.latest_pending_window = window
            logger.info(
                "inference busy; queued latest pending window session_id=%s window=%s-%s",
                session_id,
                window.start_sequence,
                window.end_sequence,
            )
            return
        active.latest_pending_window = None
        active.inference_cancel_event = threading.Event()
        logger.info(
            "inference task created session_id=%s window=%s-%s",
            session_id,
            window.start_sequence,
            window.end_sequence,
        )
        active.active_inference_task = asyncio.create_task(run_inference_loop(window, active.stream_generation))

    await send_json(
        _event(
            "session.ready",
            session_id,
            parameters={
                "captureFps": settings.capture_fps,
                "captureCountdownSeconds": settings.capture_countdown_seconds,
                "windowFrames": settings.window_frames,
                "inferenceStride": settings.inference_stride,
                "mouthSize": settings.mouth_size,
            },
        )
    )
    logger.info(
        "session ready session_id=%s capture_fps=%s window_frames=%s inference_stride=%s mouth_size=%s",
        session_id,
        settings.capture_fps,
        settings.window_frames,
        settings.inference_stride,
        settings.mouth_size,
    )

    sequence = 0
    reused_mouth_frames = 0
    last_mouth_image = None

    async def publish_buffered_frame(
        frame: MouthFrame,
        *,
        face_detected: bool,
        mouth_box: object | None,
        reused_last_mouth_crop: bool,
    ) -> None:
        window = active.add_mouth_frame(frame)
        await send_json(
            _event(
                "vision.result",
                session_id,
                faceDetected=face_detected,
                mouthBox=mouth_box,
                reusedLastMouthCrop=reused_last_mouth_crop,
                bufferedFrames=len(active.frames),
            )
        )
        await send_json(
            _event(
                "buffer.progress",
                session_id,
                bufferedFrames=len(active.frames),
                requiredFrames=settings.window_frames,
            )
        )
        await send_json(_event("metrics.update", session_id, **_metrics(active)))
        if window is not None:
            active.commit_stream()
            await send_json(_event("stream.committed", session_id))
            try:
                _dump_raw_debug_window(settings, window)
            except Exception:
                logger.exception("debug raw window dump failed session_id=%s", session_id)
            logger.info(
                "mouth window ready session_id=%s window=%s-%s buffered=%s accepted=%s reused=%s",
                session_id,
                window.start_sequence,
                window.end_sequence,
                len(active.frames),
                active.accepted_frame_count,
                reused_last_mouth_crop,
            )
            enqueue_inference(window)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if not await ensure_current_or_close():
                break
            text = message.get("text")
            if text is not None:
                command_type = _parse_command(text)
                if command_type == "stream.start":
                    await cancel_active_inference("stream restarted")
                    active.reset_stream()
                    sequence = 0
                    reused_mouth_frames = 0
                    last_mouth_image = None
                    logger.info("stream started session_id=%s", session_id)
                    await send_json(_event("stream.started", session_id))
                elif command_type == "stream.stop":
                    await cancel_active_inference("stream stopped")
                    active.stop_stream()
                    sequence = 0
                    reused_mouth_frames = 0
                    last_mouth_image = None
                    logger.info("stream stopped session_id=%s", session_id)
                    await send_json(_event("stream.stopped", session_id))
                elif command_type == "stream.commit":
                    active.commit_stream()
                    logger.info("stream committed session_id=%s", session_id)
                    await send_json(_event("stream.committed", session_id))
                elif command_type == "ping":
                    await send_json(_event("pong", session_id))
                continue

            data = message.get("bytes")
            if data is None or not active.accepting_frames:
                continue

            sequence += 1
            received_at_ms = int(time.time() * 1000)
            try:
                image = decode_jpeg_frame(data, settings)
            except FrameDecodeError as exc:
                await send_error("frame", exc.code, str(exc), True)
                continue

            detection = websocket.app.state.face_detector.detect(image)
            if not detection.face_detected:
                logger.debug(
                    "face detection miss session_id=%s sequence=%s face_count=%s reused_available=%s reused_count=%s",
                    session_id,
                    sequence,
                    detection.face_count,
                    last_mouth_image is not None,
                    reused_mouth_frames,
                )
                code = ErrorCode.MULTIPLE_FACES if detection.face_count > 1 else ErrorCode.FACE_NOT_FOUND
                await send_error("vision", code, "current frame does not contain exactly one clear face", True)
                if last_mouth_image is not None and reused_mouth_frames < MAX_REUSED_MOUTH_FRAMES:
                    reused_mouth_frames += 1
                    await publish_buffered_frame(
                        MouthFrame(sequence=sequence, received_at_ms=received_at_ms, image=last_mouth_image.copy()),
                        face_detected=False,
                        mouth_box=None,
                        reused_last_mouth_crop=True,
                    )
                continue
            try:
                crop = crop_mouth(image, detection.landmarks, settings.mouth_size)
            except ValueError as exc:
                logger.warning(
                    "mouth crop rejected session_id=%s sequence=%s reason=%s landmark_bounds=%s",
                    session_id,
                    sequence,
                    exc,
                    _landmark_bounds(detection.landmarks),
                )
                await send_error("vision", ErrorCode.INVALID_MOUTH_BOX, str(exc), True)
                if last_mouth_image is not None and reused_mouth_frames < MAX_REUSED_MOUTH_FRAMES:
                    reused_mouth_frames += 1
                    await publish_buffered_frame(
                        MouthFrame(sequence=sequence, received_at_ms=received_at_ms, image=last_mouth_image.copy()),
                        face_detected=True,
                        mouth_box=None,
                        reused_last_mouth_crop=True,
                    )
                continue
            last_mouth_image = crop.image.copy()
            reused_mouth_frames = 0
            frame = MouthFrame(
                sequence=sequence,
                received_at_ms=received_at_ms,
                image=crop.image,
                debug_image=image.copy(),
                debug_mouth_box=crop.box.model_dump(),
                debug_landmarks=tuple(detection.landmarks),
            )
            await publish_buffered_frame(
                frame,
                face_detected=True,
                mouth_box=crop.box.model_dump(),
                reused_last_mouth_crop=False,
            )
    except WebSocketDisconnect:
        logger.info("websocket disconnected session_id=%s", session_id)
        websocket.app.state.session_manager.disconnect(session_id)
    finally:
        logger.info("websocket cleanup session_id=%s active_inference_task=%s", session_id, active.active_inference_task is not None)
        await cancel_active_inference("websocket cleanup")
        active.stop_stream()
        websocket.app.state.session_manager.disconnect(session_id)
        _cleanup_temp(session_id)
