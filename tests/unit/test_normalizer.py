"""Unit tests for Arabic/English text normalization and language detection."""

from __future__ import annotations

from rag_assistant.ingestion.normalizer import (
    detect_language,
    normalize_display_text,
    normalize_search_text,
)


def test_normalize_display_text_cleans_presentation_forms_and_whitespace() -> None:
    """NFKC should convert Arabic presentation forms and tidy irregular PDF spacing."""
    # "\uFE8D\uFEDF\uFECC\uFEAE\uFE91\uFEF4\uFE94" is Arabic Presentation Forms for "العربية"
    raw = (
        "  \u200f\uFE8D\uFEDF\uFECC\uFEAE\uFE91\uFEF4\uFE94   و   English"
        "\r\n\r\n\r\nParagraph 2  "
    )
    cleaned = normalize_display_text(raw)

    assert cleaned == "العربية و English\n\nParagraph 2"
    assert "\u200f" not in cleaned
    assert normalize_display_text("multi\u00adfactor\nwrap\u00ad\nped") == "multi-factor\nwrapped"


def test_normalize_search_text_strips_tashkeel_tatweel_and_normalizes_alef_and_digits() -> None:
    """Search normalization should make diacritized Arabic and Arabic-Indic numerals uniform."""
    raw = "المَادَّةُ ١٥: إِجَازَةُ الـــمُوَظَّفِ أَوْ آلِيَّةُ العَمَلِ عَلَى المَشْرُوعِ"
    normalized = normalize_search_text(raw)

    assert "15" in normalized
    assert "١٥" not in normalized
    assert "ـ" not in normalized
    assert normalized == "المادة 15: اجازة الموظف او الية العمل علي المشروع"


def test_display_text_preserves_diacritics_while_search_text_strips_them() -> None:
    """Display text stays readable while search text normalizes orthographic variants."""
    raw = "الإِجَازَةُ السَّنَوِيَّةُ"
    assert normalize_display_text(raw) == raw
    assert normalize_search_text(raw) == "الاجازة السنوية"


def test_detect_language_handles_ar_en_mixed_and_unknown() -> None:
    """Language detector should classify Arabic, English, mixed, and numeric-only strings."""
    assert detect_language("ما هي سياسة العمل عن بعد للموظفين؟") == "ar"
    assert detect_language("What is the annual leave policy for employees?") == "en"
    assert detect_language("سياسة SLA لخدمات Cloud API للمؤسسات") == "mixed"
    assert detect_language("12345 — 2026") == "unknown"
    assert detect_language("   ") == "unknown"
