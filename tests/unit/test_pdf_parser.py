"""Unit tests for PyMuPDFParser, including bilingual extraction and OCR hook."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from rag_assistant.ingestion.base import DocumentParseError
from rag_assistant.ingestion.pdf_parser import PyMuPDFParser


class FakeOCRAdapter:
    """Deterministic OCR test double used to verify the parser's OCR extension hook."""

    def __init__(self, text_by_page: dict[int, str]) -> None:
        self.text_by_page = text_by_page
        self.called_pages: list[int] = []

    def extract_text_from_page_image(self, image_bytes: bytes, page_number: int) -> str:
        assert image_bytes.startswith(b"\x89PNG")
        self.called_pages.append(page_number)
        return self.text_by_page.get(page_number, "")


def test_parser_extracts_bilingual_multi_page_pdf(
    pdf_factory: Callable[[list[str]], bytes],
) -> None:
    """Parser should extract English and Arabic pages with accurate page numbers and languages."""
    pdf_bytes = pdf_factory(
        [
            "Enterprise Security Policy: All employees must enable multi-factor authentication.",
            "سياسة العمل عن بعد: يسمح للموظفين بالعمل عن بعد يومين في الأسبوع بموافقة المدير.",
        ]
    )
    parser = PyMuPDFParser(min_chars_per_page=20)
    parsed = parser.parse(pdf_bytes, filename="bilingual_handbook.pdf")

    assert parsed.filename == "bilingual_handbook.pdf"
    assert parsed.page_count == 2
    assert parsed.low_text_pages == []
    assert parsed.warnings == []
    assert set(parsed.languages) == {"ar", "en"}

    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].language == "en"
    assert "multi-factor authentication" in parsed.pages[0].display_text

    assert parsed.pages[1].page_number == 2
    assert parsed.pages[1].language == "ar"
    assert "سياسة" in parsed.pages[1].display_text


def test_parser_flags_low_text_pages_without_ocr(
    pdf_factory: Callable[[list[str]], bytes],
) -> None:
    """Blank or scanned pages without an OCR adapter should be reported in warnings."""
    pdf_bytes = pdf_factory(
        [
            "Page one has plenty of selectable English text for normal indexing.",
            "",
        ]
    )
    parser = PyMuPDFParser(min_chars_per_page=20)
    parsed = parser.parse(pdf_bytes, filename="partial_scan.pdf")

    assert parsed.page_count == 2
    assert parsed.low_text_pages == [2]
    assert parsed.pages[1].is_low_text is True
    assert parsed.pages[1].used_ocr is False
    assert len(parsed.warnings) == 1
    assert "2" in parsed.warnings[0]


def test_parser_invokes_ocr_hook_on_low_text_pages(
    pdf_factory: Callable[[list[str]], bytes],
) -> None:
    """When an OCR adapter is configured, low-text pages should delegate to it."""
    pdf_bytes = pdf_factory(
        [
            "Digital page with selectable text that does not need OCR.",
            "",
        ]
    )
    fake_ocr = FakeOCRAdapter(
        {2: "Recovered OCR text from scanned page two for enterprise policy compliance."}
    )
    parser = PyMuPDFParser(min_chars_per_page=20, ocr_adapter=fake_ocr)
    parsed = parser.parse(pdf_bytes, filename="scanned_page.pdf")

    assert fake_ocr.called_pages == [2]
    assert parsed.low_text_pages == []
    assert parsed.warnings == []
    assert parsed.pages[1].used_ocr is True
    assert parsed.pages[1].is_low_text is False
    assert "Recovered OCR text" in parsed.pages[1].display_text


def test_parser_rejects_non_pdf_bytes() -> None:
    """Invalid or non-PDF payloads should raise DocumentParseError cleanly."""
    parser = PyMuPDFParser()

    with pytest.raises(DocumentParseError, match="not a valid PDF"):
        parser.parse(b"Hello, this is plain text, not a PDF.", filename="fake.pdf")

    with pytest.raises(DocumentParseError, match="not a valid PDF"):
        parser.parse(b"", filename="empty.pdf")
