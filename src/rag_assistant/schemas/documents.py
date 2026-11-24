"""Document, page, and chunk schemas shared across ingestion, storage, and API layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

LanguageCode = Literal["ar", "en", "mixed", "unknown"]


class PageContent(BaseModel):
    """Extracted and normalized content for a single PDF page."""

    page_number: int = Field(..., ge=1, description="1-based page number in the source PDF.")
    raw_text: str = Field(..., description="Unmodified text extracted directly from the PDF.")
    display_text: str = Field(
        ...,
        description="Clean, human-readable text used for UI citations and LLM context.",
    )
    search_text: str = Field(
        ...,
        description="Normalized text (diacritics/alef/digits normalized) used for embeddings.",
    )
    language: LanguageCode = Field(default="unknown")
    char_count: int = Field(..., ge=0)
    used_ocr: bool = Field(default=False)
    is_low_text: bool = Field(
        default=False,
        description="True when extracted text falls below min_chars_per_page (likely scanned).",
    )


class ParsedDocument(BaseModel):
    """Structured output produced by a document parser for an entire PDF."""

    filename: str
    page_count: int = Field(..., ge=0)
    pages: list[PageContent] = Field(default_factory=list)
    low_text_pages: list[int] = Field(default_factory=list)
    languages: list[LanguageCode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TextChunk(BaseModel):
    """A single indexable text chunk preserving document and page provenance."""

    chunk_id: str = Field(..., description="Deterministic unique chunk identifier.")
    doc_id: str = Field(..., description="Parent document identifier.")
    filename: str
    page_number: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    display_text: str
    search_text: str
    language: LanguageCode = Field(default="unknown")
    char_count: int = Field(..., ge=0)


class DocumentRecord(BaseModel):
    """Metadata record describing an indexed document in the vector store."""

    doc_id: str
    filename: str
    sha256: str
    page_count: int = Field(..., ge=0)
    chunk_count: int = Field(..., ge=0)
    languages: list[LanguageCode] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    uploaded_at: datetime


class UploadResponse(BaseModel):
    """API response returned after uploading and indexing a PDF."""

    document: DocumentRecord
    is_duplicate: bool = False
