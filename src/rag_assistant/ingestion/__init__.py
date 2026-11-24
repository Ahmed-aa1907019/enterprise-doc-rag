"""Bilingual (Arabic/English) PDF parsing, text normalization, and chunking pipeline."""

from rag_assistant.ingestion.base import BaseDocumentParser, BaseOCRAdapter, DocumentParseError
from rag_assistant.ingestion.chunker import BilingualTextChunker
from rag_assistant.ingestion.normalizer import (
    detect_language,
    normalize_display_text,
    normalize_search_text,
)
from rag_assistant.ingestion.pdf_parser import PyMuPDFParser

__all__ = [
    "BaseDocumentParser",
    "BaseOCRAdapter",
    "BilingualTextChunker",
    "DocumentParseError",
    "PyMuPDFParser",
    "detect_language",
    "normalize_display_text",
    "normalize_search_text",
]
