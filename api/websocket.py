import json
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.schemas import ErrorCode, ErrorEvent, SemanticResult, utc_now
from lip.base import MouthFrame
from session.manager import ServerBusyError, SessionError
from vision.face import FrameDecodeError, decode_jpeg_frame
from vision.mouth import crop_mouth

router = APIRouter()


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


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str) -> None:
    settings = websocket.app.state.settings
    origin = websocket.headers.get("origin", "")
    if origin and origin not in settings.allowed_origin_set:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    try:
        active = websocket.app.state.session_manager.activate(session_id)
    except ServerBusyError:
        await _send_error(
            websocket,
            session_id,
            "session",
            ErrorCode.SERVER_BUSY,
            "server already has an active streaming session",
            False,
        )
        await websocket.close(code=1013)
        return
    except SessionError:
        await _send_error(
            websocket,
            session_id,
            "session",
            ErrorCode.INVALID_SESSION,
            "invalid or expired session",
            False,
        )
        await websocket.close(code=1008)
        return

    await websocket.send_json(
        _event(
            "session.ready",
            session_id,
            parameters={
                "captureFps": settings.capture_fps,
                "windowFrames": settings.window_frames,
                "inferenceStride": settings.inference_stride,
                "mouthSize": settings.mouth_size,
            },
        )
    )

    sequence = 0
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            text = message.get("text")
            if text is not None:
                command_type = _parse_command(text)
                if command_type == "stream.start":
                    active.reset_stream()
                    await websocket.send_json(_event("stream.started", session_id))
                elif command_type == "stream.stop":
                    active.reset_stream()
                    active.streaming = False
                    await websocket.send_json(_event("stream.stopped", session_id))
                elif command_type == "ping":
                    await websocket.send_json(_event("pong", session_id))
                continue

            data = message.get("bytes")
            if data is None or not active.streaming:
                continue

            sequence += 1
            received_at_ms = int(time.time() * 1000)
            try:
                image = decode_jpeg_frame(data, settings)
            except FrameDecodeError as exc:
                await _send_error(websocket, session_id, "frame", exc.code, str(exc), True)
                continue

            detection = websocket.app.state.face_detector.detect(image)
            if not detection.face_detected:
                code = ErrorCode.MULTIPLE_FACES if detection.face_count > 1 else ErrorCode.FACE_NOT_FOUND
                await _send_error(
                    websocket,
                    session_id,
                    "vision",
                    code,
                    "current frame does not contain exactly one clear face",
                    True,
                )
                continue
            crop = crop_mouth(image, detection.landmarks, settings.mouth_size)
            frame = MouthFrame(sequence=sequence, received_at_ms=received_at_ms, image=crop.image)
            window = active.add_mouth_frame(frame)
            await websocket.send_json(
                _event(
                    "vision.result",
                    session_id,
                    faceDetected=True,
                    mouthBox=crop.box.model_dump(),
                    bufferedFrames=len(active.frames),
                )
            )
            await websocket.send_json(
                _event(
                    "buffer.progress",
                    session_id,
                    bufferedFrames=len(active.frames),
                    requiredFrames=settings.window_frames,
                )
            )
            if window is None:
                continue

            await websocket.send_json(
                _event(
                    "inference.started",
                    session_id,
                    startSequence=window.start_sequence,
                    endSequence=window.end_sequence,
                )
            )
            async with websocket.app.state.inference_lock:
                lip_result = websocket.app.state.lip_engine.predict(window)
                await websocket.send_json(
                    _event(
                        "lip.candidates",
                        session_id,
                        candidates=[candidate.model_dump() for candidate in lip_result.candidates],
                        degradedModels=lip_result.degradedModels,
                    )
                )
                if not lip_result.candidates:
                    await _send_error(
                        websocket,
                        session_id,
                        "lip",
                        ErrorCode.LIP_MODELS_FAILED,
                        "both lip reading models failed",
                        True,
                    )
                    continue
                sampled = [frame.image for frame in window.frames[::15]]
                try:
                    semantic = websocket.app.state.semantic_interpreter.interpret(
                        lip_result.candidates,
                        sampled,
                        {"frames": len(window.frames)},
                    )
                except Exception:
                    await _send_error(
                        websocket,
                        session_id,
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
            await websocket.send_json(_event("semantic.result", session_id, **semantic.model_dump()))
            agent = websocket.app.state.agent_policy.decide(semantic)
            await websocket.send_json(
                agent.model_dump(mode="json") | {"sessionId": session_id, "timestamp": utc_now().isoformat()}
            )
    except WebSocketDisconnect:
        websocket.app.state.session_manager.disconnect(session_id)
    finally:
        websocket.app.state.session_manager.disconnect(session_id)
