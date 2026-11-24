"""Shared pytest fixtures, including programmatic Arabic & English PDF builders."""

from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path

import pymupdf
import pytest

from rag_assistant.ingestion.normalizer import detect_language


def _find_unicode_font() -> Path | None:
    """Locate a system TrueType font that supports both Arabic and Latin glyphs."""
    candidates = [
        Path(r"C:\Windows\Fonts\tahoma.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def build_pdf_bytes(pages: list[str]) -> bytes:
    """Generate a real multi-page PDF in memory containing the supplied page texts.

    Uses PyMuPDF's Story/HTML layout (`insert_htmlbox`) with `dir="rtl"` for Arabic
    pages and a Unicode TrueType font so Arabic shaping and logical reading order
    match real-world bilingual PDFs.
    """
    doc = pymupdf.open()
    unicode_font = _find_unicode_font()
    archive = pymupdf.Archive(str(unicode_font.parent)) if unicode_font is not None else None
    css = (
        f"@font-face {{ font-family: TestUni; src: url({unicode_font.name}); }} "
        "body { font-family: TestUni; font-size: 11pt; }"
        if unicode_font is not None
        else "body { font-size: 11pt; }"
    )

    try:
        for text in pages:
            page = doc.new_page(width=595, height=842)  # A4
            if not text:
                continue

            rect = pymupdf.Rect(50, 50, 545, 792)
            direction = "rtl" if detect_language(text) == "ar" else "ltr"
            html_body = "<br/>".join(escape(line) for line in text.splitlines())
            html = f'<div dir="{direction}">{html_body}</div>'
            page.insert_htmlbox(rect, html, css=css, archive=archive)

        return doc.tobytes()
    finally:
        doc.close()


@pytest.fixture
def pdf_factory() -> Callable[[list[str]], bytes]:
    """Return a helper that builds multi-page PDF bytes from a list of page strings."""
    return build_pdf_bytes
