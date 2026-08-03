import asyncio
import logging
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.session import router as session_router
from api.websocket import router as websocket_router
from agent.agent import AgentPolicy
from backend.config import Settings, get_settings
from command.inference import build_command_classifier
from session.manager import SessionManager
from vision.face import create_face_detector


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings)
    app = FastAPI(title="Silent Vision")
    app.state.settings = app_settings
    app.state.models = {"backend": app_settings.command_backend, "ready": False}
    app.state.session_manager = SessionManager(pending_ttl=timedelta(seconds=30))
    app.state.command_classifier = build_command_classifier(app_settings)
    app.state.inference_lock = asyncio.Lock()
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


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    logging.getLogger().setLevel(level)
    for logger_name in ("api", "backend", "vision", "agent", "session", "command", "video"):
        logging.getLogger(logger_name).setLevel(level)


app = create_app()
