"""Health check endpoint reporting service readiness and active model configuration."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rag_assistant import __version__
from rag_assistant.config import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response schema for `/api/v1/health`."""

    status: str
    app_name: str
    version: str
    environment: str
    llm_model: str
    llm_configured: bool
    embedding_model: str
    chroma_persist_dir: str


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health_check(request: Request) -> HealthResponse:
    """Return service status and non-sensitive runtime configuration."""
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=__version__,
        environment=settings.app_env,
        llm_model=settings.gemini_model,
        llm_configured=settings.has_gemini_api_key,
        embedding_model=settings.embedding_model,
        chroma_persist_dir=str(settings.chroma_persist_dir),
    )
