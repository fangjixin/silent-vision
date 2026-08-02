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

from backend.schemas import CalibrationRequest, CalibrationSaved, ErrorCode, ErrorEvent, SemanticResult, utc_now
from command.inference import save_command_debug
from command.labels import CommandIntent
from command.prototype import save_prototype_sample, sanitize_profile_id
from lip.base import MouthFrame
from session.manager import SessionError, SessionReplacedError
from video.clip import decode_video_clip, save_original_video
from video.mouth_roi import extract_mouth_roi_clip, write_aligned_face_video, write_mouth_roi_video
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
    loaded = _parse_json_object(raw)
    if loaded is None:
        return None
    command_type = loaded.get("type")
    return command_type if isinstance(command_type, str) else None


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    try:
        loaded: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    return loaded


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

    async def process_command_clip(data: bytes, extra_metadata: dict[str, object] | None = None) -> None:
        started = time.perf_counter()
        debug_dir = settings.debug_window_dir or settings.persistence_root / "logs" / "command-runs"
        original_video = None
        aligned_face_video = None
        mouth_roi_video = None
        mouth_roi_npy = None
        if settings.debug_dump_windows:
            original_video = save_original_video(data, debug_dir, session_id)
        await send_json(_event("clip.received", session_id, bytes=len(data)))
        try:
            decoded = await asyncio.to_thread(decode_video_clip, data, settings.command_clip_fps)
            roi = await asyncio.to_thread(
                extract_mouth_roi_clip,
                frames=decoded.frames,
                face_detector=websocket.app.state.face_detector,
                settings=settings,
            )
            if settings.debug_dump_windows:
                import numpy as np

                mouth_roi_npy = debug_dir / f"{session_id}-mouth_roi.npy"
                np.save(mouth_roi_npy, roi.mouth_frames)
                aligned_face_video = await asyncio.to_thread(
                    write_aligned_face_video,
                    roi.aligned_face_frames,
                    debug_dir / f"{session_id}-aligned_face_video.mp4",
                    settings.command_clip_fps,
                )
                mouth_roi_video = await asyncio.to_thread(
                    write_mouth_roi_video,
                    roi.mouth_frames,
                    debug_dir / f"{session_id}-mouth_roi_video.mp4",
                    settings.command_clip_fps,
                )
            command_decision = await asyncio.to_thread(
                websocket.app.state.command_classifier.predict,
                roi.mouth_frames,
                {
                    "frames": len(roi.mouth_frames),
                    "sourceFrames": len(decoded.frames),
                    "fps": decoded.fps,
                    "durationMs": decoded.duration_ms,
                    "detectedFrames": roi.detected_frames,
                    "reusedFrames": roi.reused_frames,
                    "original_video": str(original_video) if original_video else None,
                    "aligned_face_video": str(aligned_face_video) if aligned_face_video else None,
                    "mouth_roi_video": str(mouth_roi_video) if mouth_roi_video else None,
                    "mouth_roi_npy": str(mouth_roi_npy) if mouth_roi_npy else None,
                }
                | (extra_metadata or {}),
            )
        except Exception:
            logger.exception("command clip processing failed session_id=%s bytes=%s", session_id, len(data))
            await send_error("command", ErrorCode.INTERNAL_ERROR, "command clip processing failed", True)
            return

        metadata_path = None
        if settings.debug_dump_windows:
            metadata_path = save_command_debug(settings, session_id, command_decision)
        logger.info(
            "command result session_id=%s intent=%s accepted=%s executable=%s confidence=%.3f margin=%.3f reason=%s latency_ms=%s",
            session_id,
            command_decision.intent,
            command_decision.accepted,
            command_decision.executable,
            command_decision.confidence,
            command_decision.margin,
            command_decision.reason,
            int((time.perf_counter() - started) * 1000),
        )
        await send_json(
            _event(
                "command.result",
                session_id,
                **command_decision.model_dump(),
                debugMetadataPath=str(metadata_path) if metadata_path else None,
            )
        )
        agent = websocket.app.state.agent_policy.decide_command(command_decision)
        await send_json(agent.model_dump(mode="json") | {"sessionId": session_id, "timestamp": utc_now().isoformat()})

    async def process_calibration_clip(data: bytes, calibration: dict[str, Any]) -> None:
        debug_dir = settings.debug_window_dir or settings.persistence_root / "logs" / "command-runs"
        profile_id = str(calibration["profileId"])
        scope = str(calibration["scope"])
        target_profile = "global" if scope == "global" else sanitize_profile_id(profile_id)
        original_video = None
        aligned_face_video = None
        mouth_roi_video = None
        if settings.debug_dump_windows:
            original_video = save_original_video(data, debug_dir, session_id, suffix="-calibration.webm")
        await send_json(_event("clip.received", session_id, bytes=len(data), calibration=True))
        try:
            decoded = await asyncio.to_thread(decode_video_clip, data, settings.command_clip_fps)
            roi = await asyncio.to_thread(
                extract_mouth_roi_clip,
                frames=decoded.frames,
                face_detector=websocket.app.state.face_detector,
                settings=settings,
            )
            sample_path = await asyncio.to_thread(
                save_prototype_sample,
                settings.persistence_root,
                target_profile,
                str(calibration["intent"]),
                roi.mouth_frames,
                {
                    "scope": scope,
                    "profileId": profile_id,
                    "language": calibration.get("language", "unknown"),
                    "phrase": calibration.get("phrase", ""),
                    "sourceFrames": len(decoded.frames),
                    "fps": decoded.fps,
                    "durationMs": decoded.duration_ms,
                    "detectedFrames": roi.detected_frames,
                    "reusedFrames": roi.reused_frames,
                },
            )
            (sample_path / "original.webm").write_bytes(data)
            if settings.debug_dump_windows:
                aligned_face_video = await asyncio.to_thread(
                    write_aligned_face_video,
                    roi.aligned_face_frames,
                    sample_path / "aligned_face_video.mp4",
                    settings.command_clip_fps,
                )
                mouth_roi_video = await asyncio.to_thread(
                    write_mouth_roi_video,
                    roi.mouth_frames,
                    sample_path / "mouth_roi_video.mp4",
                    settings.command_clip_fps,
                )
        except Exception:
            logger.exception("calibration clip processing failed session_id=%s bytes=%s", session_id, len(data))
            await send_json(
                _event(
                    "calibration.error",
                    session_id,
                    profileId=profile_id,
                    scope=scope,
                    intent=calibration.get("intent"),
                    message="calibration clip processing failed",
                )
            )
            return

        logger.info(
            "calibration sample saved session_id=%s profile_id=%s scope=%s intent=%s sample_path=%s frames=%s detected=%s",
            session_id,
            profile_id,
            scope,
            calibration["intent"],
            sample_path,
            len(roi.mouth_frames),
            roi.detected_frames,
        )
        event = CalibrationSaved(
            sessionId=session_id,
            profileId=profile_id,
            scope=scope,  # type: ignore[arg-type]
            intent=calibration["intent"],
            samplePath=str(sample_path),
            frames=len(roi.mouth_frames),
            detectedFrames=roi.detected_frames,
        ).model_dump(mode="json")
        event["type"] = "calibration.saved"
        event["alignedFaceVideo"] = str(aligned_face_video) if aligned_face_video else None
        event["mouthRoiVideo"] = str(mouth_roi_video) if mouth_roi_video else None
        event["originalVideo"] = str(original_video) if original_video else str(sample_path / "original.webm")
        await send_json(event)

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
                "recognitionMode": settings.recognition_mode,
                "commandClipFps": settings.command_clip_fps,
                "commandClipMinSeconds": settings.command_clip_min_seconds,
                "commandClipMaxSeconds": settings.command_clip_max_seconds,
                "commandConfidenceThreshold": settings.command_confidence_threshold,
                "commandTop1Margin": settings.command_top1_margin,
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
    pending_calibration: dict[str, Any] | None = None
    pending_clip_metadata: dict[str, object] = {}

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
                command_payload = _parse_json_object(text) or {}
                command_type = command_payload.get("type") if isinstance(command_payload.get("type"), str) else None
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
                elif command_type == "clip.start":
                    await cancel_active_inference("clip started")
                    active.reset_stream()
                    pending_calibration = None
                    pending_clip_metadata = {
                        key: value
                        for key, value in command_payload.items()
                        if key in {"profileId"} and isinstance(value, str)
                    }
                    logger.info("clip started session_id=%s", session_id)
                    await send_json(_event("clip.started", session_id))
                elif command_type == "clip.cancel":
                    await cancel_active_inference("clip cancelled")
                    active.stop_stream()
                    pending_calibration = None
                    pending_clip_metadata = {}
                    logger.info("clip cancelled session_id=%s", session_id)
                    await send_json(_event("stream.stopped", session_id))
                elif command_type == "calibration.start":
                    try:
                        request = CalibrationRequest.model_validate(command_payload)
                        if request.scope == "global" and not settings.allow_global_profile_write:
                            raise ValueError("global profile writes are disabled")
                        if request.intent not in {intent.value for intent in CommandIntent}:
                            raise ValueError("invalid calibration intent")
                        pending_calibration = request.model_dump()
                        await cancel_active_inference("calibration started")
                        active.reset_stream()
                        logger.info(
                            "calibration started session_id=%s profile_id=%s scope=%s intent=%s",
                            session_id,
                            request.profileId,
                            request.scope,
                            request.intent,
                        )
                        await send_json(
                            _event(
                                "calibration.started",
                                session_id,
                                profileId=request.profileId,
                                scope=request.scope,
                                intent=request.intent,
                            )
                        )
                    except Exception as exc:
                        pending_calibration = None
                        logger.warning("calibration request rejected session_id=%s reason=%s", session_id, exc)
                        await send_json(
                            _event(
                                "calibration.error",
                                session_id,
                                profileId=command_payload.get("profileId"),
                                scope=command_payload.get("scope"),
                                intent=command_payload.get("intent"),
                                message=str(exc),
                            )
                        )
                continue

            data = message.get("bytes")
            if data is None or not active.accepting_frames:
                continue

            if pending_calibration is not None:
                calibration = pending_calibration
                pending_calibration = None
                active.commit_stream()
                await process_calibration_clip(data, calibration)
                continue

            if settings.recognition_mode == "command":
                active.commit_stream()
                data_metadata = pending_clip_metadata.copy()
                pending_clip_metadata = {}
                await process_command_clip(data, data_metadata)
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
