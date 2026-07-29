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
from llm.minicpm import FakeMiniCPMInterpreter
from session.manager import SessionManager


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(title="Silent Vision")
    app.state.settings = app_settings
    app.state.models = {"backend": app_settings.model_backend, "ready": app_settings.model_backend == "fake"}
    app.state.session_manager = SessionManager(
        pending_ttl=timedelta(seconds=30),
        window_frames=app_settings.window_frames,
        inference_stride=app_settings.inference_stride,
    )
    app.state.lip_engine = LipInferenceEngine(
        [
            FakeLipReader(model="avhubert", language="en", text="turn on the light", confidence=0.72),
            FakeLipReader(model="cmlr", language="zh", text="请打开灯", confidence=0.76),
        ]
    )
    app.state.semantic_interpreter = FakeMiniCPMInterpreter(threshold=app_settings.model_confidence_threshold)
    app.state.agent_policy = AgentPolicy(threshold=app_settings.model_confidence_threshold)
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
