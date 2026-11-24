"""Arabic and English text normalization and lightweight script-based language detection."""

from __future__ import annotations

import re
import unicodedata

from rag_assistant.schemas.documents import LanguageCode

# Invisible Unicode formatting/directional control characters frequently embedded in Arabic PDFs.
_ZERO_WIDTH_AND_BIDI_RE = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]")

# Non-printable ASCII/Unicode control characters except tab (\t) and newline (\n).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Arabic diacritics (tashkeel / harakat) and Quranic annotation marks.
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

# Tatweel / Kashida elongation character (e.g., "الـــعربية" -> "العربية").
_TATWEEL = "\u0640"

# Unify Alef variants and Arabic-Indic digits for consistent search and embedding.
_SEARCH_CHAR_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)

_HORIZONTAL_WS_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u064A\u066E-\u06D3\u0750-\u077F\u08A0-\u08FF]")
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")


def normalize_display_text(text: str) -> str:
    """Clean raw PDF text for human-readable display and LLM context.

    Applies Unicode NFKC normalization (which converts Arabic Presentation Forms
    and PDF ligatures back into standard characters), removes stray directional
    control markers, and tidies irregular PDF whitespace while preserving
    paragraph breaks and natural Arabic orthography.
    """
    if not text:
        return ""

    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin line-wrapped soft hyphens and convert inline soft hyphens to standard '-'.
    cleaned = cleaned.replace("\u00ad\n", "").replace("\u00ad", "-")
    cleaned = _ZERO_WIDTH_AND_BIDI_RE.sub("", cleaned)
    cleaned = _CONTROL_CHARS_RE.sub("", cleaned)

    lines = [_HORIZONTAL_WS_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_search_text(text: str) -> str:
    """Normalize text further for semantic embedding and retrieval consistency.

    In addition to `normalize_display_text`, removes Arabic diacritics (tashkeel),
    tatweel elongation characters, unifies Alef/Alef-Maksura forms, and converts
    Arabic-Indic numerals (٠-٩) to Western digits (0-9) so cross-lingual queries
    like 'Article 15' match 'المادة ١٥'.
    """
    cleaned = normalize_display_text(text)
    if not cleaned:
        return ""

    cleaned = _ARABIC_DIACRITICS_RE.sub("", cleaned)
    cleaned = cleaned.replace(_TATWEEL, "")
    cleaned = cleaned.translate(_SEARCH_CHAR_MAP)
    cleaned = _HORIZONTAL_WS_RE.sub(" ", cleaned)
    return cleaned.strip()


def detect_language(text: str) -> LanguageCode:
    """Classify text language as `'ar'`, `'en'`, `'mixed'`, or `'unknown'`.

    Uses Unicode script ratios (Arabic vs. Latin letters), making it instant,
    offline, deterministic, and reliable even on very short queries.
    """
    normalized = normalize_display_text(text)
    if not normalized:
        return "unknown"

    arabic_count = len(_ARABIC_LETTER_RE.findall(normalized))
    latin_count = len(_LATIN_LETTER_RE.findall(normalized))
    total = arabic_count + latin_count

    if total == 0:
        return "unknown"

    arabic_ratio = arabic_count / total
    latin_ratio = latin_count / total

    if arabic_ratio >= 0.70:
        return "ar"
    if latin_ratio >= 0.70:
        return "en"
    return "mixed"
