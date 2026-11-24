"""Typed application configuration loaded from environment variables and `.env`."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="Bilingual RAG Document Assistant")
    app_env: Literal["development", "staging", "production", "test"] = Field(default="development")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO")

    # LLM Provider
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Embeddings
    embedding_model: str = Field(default="intfloat/multilingual-e5-small")
    embedding_batch_size: int = Field(default=32, ge=1, le=256)

    # Storage
    upload_dir: Path = Field(default=Path("data/uploads"))
    chroma_persist_dir: Path = Field(default=Path("data/chroma"))
    chroma_collection_name: str = Field(default="enterprise_docs", min_length=1)

    # PDF Ingestion & Chunking
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    min_chars_per_page: int = Field(default=20, ge=0, le=500)
    chunk_size: int = Field(default=800, ge=100, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)

    # Retrieval & Grounding
    default_top_k: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.72, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_chunking_bounds(self) -> Settings:
        """Ensure chunk overlap is strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})."
            )
        return self

    @property
    def max_upload_bytes(self) -> int:
        """Maximum accepted PDF upload size in bytes."""
        return self.max_upload_mb * 1024 * 1024

    @property
    def has_gemini_api_key(self) -> bool:
        """Whether a non-empty Gemini API key is configured."""
        return bool(self.gemini_api_key and self.gemini_api_key.strip())

    def ensure_directories(self) -> None:
        """Create local runtime directories if they do not yet exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance for production runtime."""
    return Settings()
