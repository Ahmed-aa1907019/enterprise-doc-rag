"""Unit tests for the FastAPI health endpoint and app lifecycle."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rag_assistant import __version__
from rag_assistant.config import Settings
from rag_assistant.main import create_app


def test_health_endpoint_returns_expected_metadata(tmp_path: Path) -> None:
    """GET /api/v1/health should return 200 and non-sensitive runtime details."""
    settings = Settings(
        _env_file=None,
        app_env="test",
        gemini_api_key="super-secret-value",
        upload_dir=tmp_path / "uploads",
        chroma_persist_dir=tmp_path / "chroma",
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "app_name": settings.app_name,
        "version": __version__,
        "environment": "test",
        "llm_model": "gemini-2.5-flash",
        "llm_configured": True,
        "embedding_model": "intfloat/multilingual-e5-small",
        "chroma_persist_dir": str(tmp_path / "chroma"),
    }
    assert "super-secret-value" not in response.text
    assert (tmp_path / "uploads").is_dir()
    assert (tmp_path / "chroma").is_dir()


def test_health_reports_unconfigured_llm_when_key_missing(tmp_path: Path) -> None:
    """llm_configured should be False when GEMINI_API_KEY is absent."""
    settings = Settings(
        _env_file=None,
        app_env="test",
        gemini_api_key=None,
        upload_dir=tmp_path / "uploads",
        chroma_persist_dir=tmp_path / "chroma",
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is False
