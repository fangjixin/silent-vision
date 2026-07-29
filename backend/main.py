from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from backend.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(default_response_class=ORJSONResponse, title="Silent Vision")
    app.state.settings = app_settings
    app.state.models = {"backend": app_settings.model_backend, "ready": app_settings.model_backend == "fake"}

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready() -> dict[str, object]:
        return {"status": "ready", "models": app.state.models}

    return app


app = create_app()
