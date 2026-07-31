import asyncio
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.session import router as session_router
from api.websocket import router as websocket_router
from agent.agent import AgentPolicy
from backend.config import Settings, get_settings
from lip.fake import FakeLipReader
from lip.inference import LipInferenceEngine
from llm.minicpm import build_minicpm_interpreter
from session.manager import SessionManager
from vision.face import create_face_detector


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(title="Silent Vision")
    app.state.settings = app_settings
    app.state.models = {"backend": app_settings.model_backend, "ready": False}
    app.state.session_manager = SessionManager(
        pending_ttl=timedelta(seconds=30),
        window_frames=app_settings.window_frames,
        inference_stride=app_settings.inference_stride,
    )
    if app_settings.model_backend == "real":
        from lip.avhubert import AVHuBERTLipReader
        from lip.cmlr import CMLRLipReader

        readers = [AVHuBERTLipReader(app_settings), CMLRLipReader(app_settings)]
    else:
        readers = [
            FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
            FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
        ]
    app.state.lip_engine = LipInferenceEngine(readers)
    app.state.inference_lock = asyncio.Lock()
    app.state.semantic_interpreter = build_minicpm_interpreter(app_settings)
    app.state.agent_policy = AgentPolicy(threshold=app_settings.model_confidence_threshold)
    app.state.face_detector = create_face_detector(app_settings)
    app.state.models["ready"] = True
    app.include_router(session_router)
    app.include_router(websocket_router)
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        return {"status": "ready", "models": app.state.models}

    return app


app = create_app()
