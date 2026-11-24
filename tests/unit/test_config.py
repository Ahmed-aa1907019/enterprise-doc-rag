"""Unit tests for typed configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_assistant.config import Settings


def test_default_settings_values() -> None:
    """Settings should initialize cleanly with safe defaults even without `.env`."""
    settings = Settings(_env_file=None)

    assert settings.gemini_model == "gemini-2.5-flash"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 150
    assert settings.max_upload_bytes == 25 * 1024 * 1024
    assert settings.has_gemini_api_key is False


def test_chunk_overlap_must_be_smaller_than_chunk_size() -> None:
    """Overlap equal to or larger than chunk_size should raise a validation error."""
    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings(_env_file=None, chunk_size=500, chunk_overlap=500)


def test_has_gemini_api_key_strips_whitespace() -> None:
    """Whitespace-only API keys should count as unconfigured."""
    assert Settings(_env_file=None, gemini_api_key="   ").has_gemini_api_key is False
    assert Settings(_env_file=None, gemini_api_key="test-key").has_gemini_api_key is True


def test_ensure_directories_creates_storage_paths(tmp_path: Path) -> None:
    """ensure_directories should create both upload and chroma directories."""
    upload_dir = tmp_path / "uploads"
    chroma_dir = tmp_path / "chroma"
    settings = Settings(
        _env_file=None,
        upload_dir=upload_dir,
        chroma_persist_dir=chroma_dir,
    )

    assert not upload_dir.exists()
    assert not chroma_dir.exists()

    settings.ensure_directories()

    assert upload_dir.is_dir()
    assert chroma_dir.is_dir()
