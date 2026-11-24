"""FastAPI application factory and lifecycle management."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_assistant import __version__
from rag_assistant.api.routes.health import router as health_router
from rag_assistant.config import Settings, get_settings
from rag_assistant.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and configure a FastAPI application instance.

    Accepting an optional `settings` object makes the app trivial to configure
    with isolated temporary directories and test doubles during pytest runs.
    """
    active_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(active_settings.log_level)
        active_settings.ensure_directories()
        logger.info(
            "Starting %s v%s (env=%s, embedding_model=%s, llm_model=%s, llm_configured=%s)",
            active_settings.app_name,
            __version__,
            active_settings.app_env,
            active_settings.embedding_model,
            active_settings.gemini_model,
            active_settings.has_gemini_api_key,
        )
        yield
        logger.info("Shutting down %s", active_settings.app_name)

    app = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        description="Bilingual (Arabic/English) RAG Document Assistant API",
        lifespan=lifespan,
    )
    app.state.settings = active_settings
    app.include_router(health_router, prefix="/api/v1")

    return app


app = create_app()
