"""Unit tests for the bilingual recursive text chunker."""

from __future__ import annotations

import pytest

from rag_assistant.ingestion.chunker import BilingualTextChunker
from rag_assistant.schemas.documents import PageContent


def _make_page(page_number: int, text: str, language: str = "en") -> PageContent:
    return PageContent(
        page_number=page_number,
        raw_text=text,
        display_text=text,
        search_text=text,
        language=language,  # type: ignore[arg-type]
        char_count=len(text),
    )


def test_chunker_rejects_invalid_overlap() -> None:
    """Overlap equal to or larger than chunk_size should raise ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        BilingualTextChunker(chunk_size=200, chunk_overlap=200)


def test_chunker_splits_english_text_with_overlap_and_preserves_page_numbers() -> None:
    """Chunks should respect chunk_size, carry overlap, and keep exact page provenance."""
    page1_text = " ".join(
        f"Sentence number {i} explains the enterprise security policy clearly."
        for i in range(1, 12)
    )
    page2_text = "Page two covers incident response procedures and escalation timelines."

    chunker = BilingualTextChunker(chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_pages(
        pages=[_make_page(1, page1_text, "en"), _make_page(2, page2_text, "en")],
        doc_id="doc123",
        filename="security.pdf",
    )

    assert len(chunks) >= 3
    page1_chunks = [c for c in chunks if c.page_number == 1]
    page2_chunks = [c for c in chunks if c.page_number == 2]

    assert len(page1_chunks) >= 2
    assert len(page2_chunks) == 1
    assert all(len(c.display_text) <= 180 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert chunks[0].chunk_id == "doc123_p1_c0"

    # Verify consecutive chunks on the same page share overlapping trailing words.
    tail_word = page1_chunks[0].display_text.split()[-1]
    assert tail_word in page1_chunks[1].display_text


def test_chunker_handles_arabic_punctuation_and_search_normalization() -> None:
    """Arabic sentences separated by '؟' and '؛' should split cleanly and normalize search_text."""
    arabic_text = (
        "ما هي شروط الإجازة السنوية للموظفين؟ "
        "يستحق الموظف إجازة سنوية مدفوعة الأجر لمدة ٣٠ يوماً؛ "
        "ويجب تقديم الطلب قبل موعد الإجازة بأسبوعين على الأقل، "
        "مع موافقة المدير المباشر قبل السفر."
    )
    chunker = BilingualTextChunker(chunk_size=110, chunk_overlap=25)
    chunks = chunker.chunk_pages(
        pages=[_make_page(1, arabic_text, "ar")],
        doc_id="ar_doc",
        filename="hr_ar.pdf",
    )

    assert len(chunks) >= 2
    assert all(c.language == "ar" for c in chunks)
    assert all(len(c.display_text) <= 110 for c in chunks)
    # Search text across chunks should normalize Arabic-Indic "٣٠" -> "30".
    assert any("30" in c.search_text for c in chunks)


def test_chunker_skips_empty_pages() -> None:
    """Blank pages should not produce empty chunks."""
    chunker = BilingualTextChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_pages(
        pages=[
            _make_page(1, "   ", "unknown"),
            _make_page(2, "Valid content on page two.", "en"),
        ],
        doc_id="doc_empty",
        filename="sparse.pdf",
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
