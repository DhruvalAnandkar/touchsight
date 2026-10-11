"""TouchSight API."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import Settings, get_settings
from app.jobs import JobStore, build_sam
from app.routes import router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="TouchSight", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.origin_list(),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.settings = settings
    app.state.jobs = JobStore(settings, sam=build_sam(settings))
    app.include_router(router)
    return app


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


_configure_logging()
app = create_app()
