"""PyMuPDF-based PDF parser with bilingual normalization and an optional OCR hook."""

from __future__ import annotations

import logging

import pymupdf

from rag_assistant.ingestion.base import BaseOCRAdapter, DocumentParseError
from rag_assistant.ingestion.normalizer import (
    detect_language,
    normalize_display_text,
    normalize_search_text,
)
from rag_assistant.schemas.documents import LanguageCode, PageContent, ParsedDocument

logger = logging.getLogger(__name__)

_PDF_MAGIC_HEADER = b"%PDF-"


class PyMuPDFParser:
    """Extract page-by-page text from digital PDFs using PyMuPDF.

    If a page yields fewer characters than `min_chars_per_page`, the parser
    either delegates that page to an optional `BaseOCRAdapter` or flags it in
    `low_text_pages` and `warnings` so users know it may be a scanned image.
    """

    def __init__(
        self,
        min_chars_per_page: int = 20,
        ocr_adapter: BaseOCRAdapter | None = None,
        ocr_dpi: int = 200,
    ) -> None:
        self.min_chars_per_page = min_chars_per_page
        self.ocr_adapter = ocr_adapter
        self.ocr_dpi = ocr_dpi

    def parse(self, file_bytes: bytes, filename: str) -> ParsedDocument:
        """Parse raw PDF bytes into a `ParsedDocument` with page-level metadata."""
        if not file_bytes or not file_bytes.lstrip().startswith(_PDF_MAGIC_HEADER):
            raise DocumentParseError(
                f"File '{filename}' is not a valid PDF (missing %PDF- header)."
            )

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            raise DocumentParseError(f"Failed to open PDF '{filename}': {exc}") from exc

        try:
            if doc.needs_pass:
                raise DocumentParseError(
                    f"PDF '{filename}' is password-protected and cannot be indexed."
                )

            pages: list[PageContent] = []
            low_text_pages: list[int] = []
            detected_languages: set[LanguageCode] = set()

            for page_idx, page in enumerate(doc, start=1):
                # Keep sort=False: PyMuPDF's sort=True sorts words left-to-right by X
                # coordinate, which reverses word order on Arabic (RTL) lines.
                raw_text = page.get_text("text", sort=False) or ""
                display_text = normalize_display_text(raw_text)
                used_ocr = False

                if len(display_text) < self.min_chars_per_page and self.ocr_adapter is not None:
                    logger.info(
                        "Page %d of '%s' has %d chars (< %d); invoking OCR adapter.",
                        page_idx,
                        filename,
                        len(display_text),
                        self.min_chars_per_page,
                    )
                    pixmap = page.get_pixmap(dpi=self.ocr_dpi)
                    ocr_text = self.ocr_adapter.extract_text_from_page_image(
                        pixmap.tobytes("png"),
                        page_number=page_idx,
                    )
                    if ocr_text and ocr_text.strip():
                        raw_text = ocr_text
                        display_text = normalize_display_text(ocr_text)
                        used_ocr = True

                is_low_text = len(display_text) < self.min_chars_per_page
                if is_low_text:
                    low_text_pages.append(page_idx)

                search_text = normalize_search_text(display_text)
                language = detect_language(display_text)
                if language != "unknown":
                    detected_languages.add(language)

                pages.append(
                    PageContent(
                        page_number=page_idx,
                        raw_text=raw_text,
                        display_text=display_text,
                        search_text=search_text,
                        language=language,
                        char_count=len(display_text),
                        used_ocr=used_ocr,
                        is_low_text=is_low_text,
                    )
                )

            warnings: list[str] = []
            if low_text_pages:
                pages_str = ", ".join(str(p) for p in low_text_pages)
                warnings.append(
                    f"Pages [{pages_str}] contain little or no selectable text and may be "
                    "scanned images. Configure an OCR adapter to extract scanned pages."
                )

            ordered_languages: list[LanguageCode] = (
                sorted(detected_languages) if detected_languages else ["unknown"]
            )

            return ParsedDocument(
                filename=filename,
                page_count=len(pages),
                pages=pages,
                low_text_pages=low_text_pages,
                languages=ordered_languages,
                warnings=warnings,
            )
        finally:
            doc.close()
