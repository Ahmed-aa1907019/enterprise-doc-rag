"""Recursive, punctuation-aware text chunker for Arabic and English documents."""

from __future__ import annotations

from rag_assistant.ingestion.normalizer import detect_language, normalize_search_text
from rag_assistant.schemas.documents import PageContent, TextChunk

# Ordered from coarsest semantic boundary (paragraphs) down to sentence, clause, word, and char.
DEFAULT_BILINGUAL_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "؟ ",
    ". ",
    "! ",
    "? ",
    "؛ ",
    "، ",
    ", ",
    " ",
    "",
)


class BilingualTextChunker:
    """Split parsed PDF pages into overlapping chunks while preserving page numbers.

    Chunks are created per page so every chunk maps unambiguously to a single
    `page_number` for accurate inline citations (e.g., `[1] policy.pdf, p. 4`).
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: tuple[str, ...] = DEFAULT_BILINGUAL_SEPARATORS,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def chunk_pages(
        self,
        pages: list[PageContent],
        doc_id: str,
        filename: str,
    ) -> list[TextChunk]:
        """Chunk all non-empty pages of a document into `TextChunk` records."""
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            text = page.display_text.strip()
            if not text:
                continue

            page_chunks = self.split_text(text)
            for display_chunk in page_chunks:
                display_chunk = display_chunk.strip()
                if not display_chunk:
                    continue

                search_chunk = normalize_search_text(display_chunk)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}_p{page.page_number}_c{chunk_index}",
                        doc_id=doc_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        display_text=display_chunk,
                        search_text=search_chunk,
                        language=detect_language(display_chunk),
                        char_count=len(display_chunk),
                    )
                )
                chunk_index += 1

        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split a single block of text into chunks respecting size and overlap."""
        cleaned = text.strip()
        if not cleaned:
            return []
        if len(cleaned) <= self.chunk_size:
            return [cleaned]

        segments = self._recursive_split(cleaned, self.separators)
        return self._merge_segments_with_overlap(segments)

    def _recursive_split(self, text: str, separators: tuple[str, ...]) -> list[str]:
        """Recursively break text down until every piece fits within `chunk_size`."""
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep == "":
            return [
                text[i : i + self.chunk_size]
                for i in range(0, len(text), self.chunk_size)
            ]

        if sep not in text:
            return self._recursive_split(text, remaining_seps)

        raw_parts = text.split(sep)
        pieces: list[str] = []
        keep_punctuation = sep.strip() != ""

        for idx, part in enumerate(raw_parts):
            is_last = idx == len(raw_parts) - 1
            # Keep punctuation marks (e.g., '.', '؟', '؛') attached to the end of the sentence.
            piece = f"{part}{sep.strip()}" if keep_punctuation and not is_last else part

            piece = piece.strip()
            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                pieces.append(piece)
            else:
                pieces.extend(self._recursive_split(piece, remaining_seps))

        return pieces

    def _merge_segments_with_overlap(self, segments: list[str]) -> list[str]:
        """Greedily pack segments into `chunk_size` windows with clean overlap."""
        merged: list[str] = []
        current = ""

        for segment in segments:
            candidate = segment if not current else f"{current} {segment}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                merged.append(current)

            overlap_prefix = self._extract_overlap(current)
            if overlap_prefix and len(f"{overlap_prefix} {segment}") <= self.chunk_size:
                current = f"{overlap_prefix} {segment}"
            else:
                current = segment

        if current:
            merged.append(current)

        return merged

    def _extract_overlap(self, text: str) -> str:
        """Take up to `chunk_overlap` trailing characters trimmed to a word boundary."""
        if self.chunk_overlap <= 0 or not text:
            return ""
        if len(text) <= self.chunk_overlap:
            return text

        tail = text[-self.chunk_overlap :]
        # Avoid starting overlap mid-word when a whitespace boundary exists.
        first_space = tail.find(" ")
        if 0 < first_space < len(tail) - 1:
            return tail[first_space + 1 :].strip()
        return tail.strip()
