"""Protocols and exceptions for document parsing and optional OCR adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_assistant.schemas.documents import ParsedDocument


class DocumentParseError(ValueError):
    """Raised when an uploaded document cannot be opened or parsed."""


@runtime_checkable
class BaseOCRAdapter(Protocol):
    """Extension hook for extracting text from scanned/image-only PDF pages."""

    def extract_text_from_page_image(self, image_bytes: bytes, page_number: int) -> str:
        """Run OCR on a rendered PNG page image and return extracted text."""
        ...


@runtime_checkable
class BaseDocumentParser(Protocol):
    """Contract implemented by document parsers (e.g., PyMuPDFParser)."""

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Extract page-level text, language metadata, and warnings from raw bytes."""
        ...
