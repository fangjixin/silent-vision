from datetime import timedelta

from fastapi import FastAPI

from api.session import router as session_router
from backend.config import Settings, get_settings
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
    app.include_router(session_router)

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        return {"status": "ready", "models": app.state.models}

    return app


app = create_app()
