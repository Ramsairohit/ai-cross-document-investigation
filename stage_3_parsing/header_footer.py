"""
Stage 3: Structural Parsing - Header/Footer Detection

Deterministic detection of headers and footers using rule-based patterns.

Rules applied:
1. Repeated text across multiple pages
2. Page number patterns (Page X of Y, — X —, etc.)
3. Document title boilerplate text

IMPORTANT: Text is TAGGED, never deleted or modified.
"""

import re
from dataclasses import dataclass
from typing import Any

# Page number regex patterns - compiled for performance
PAGE_NUMBER_PATTERNS: list[re.Pattern[str]] = [
    # "Page 1 of 5", "Page 1/5"
    re.compile(r"^\s*page\s+\d+\s*(?:of|/)\s*\d+\s*$", re.IGNORECASE),
    # "— 1 —", "- 1 -", "-- 1 --"
    re.compile(r"^\s*[-—]+\s*\d+\s*[-—]+\s*$"),
    # Just a number (common page format)
    re.compile(r"^\s*\d+\s*$"),
    # "[Page 1]", "(Page 1)"
    re.compile(r"^\s*[\[(]?\s*page\s*\d+\s*[\])]?\s*$", re.IGNORECASE),
    # "1 | Document Name" or "Document Name | 1"
    re.compile(r"^\s*\d+\s*\|\s*.+$|^.+\s*\|\s*\d+\s*$"),
]


@dataclass
class HeaderFooterResult:
    """Result of header/footer detection for a single block."""

    is_header: bool
    is_footer: bool


def _normalize_text_for_comparison(text: str) -> str:
    """
    Normalize text for repetition comparison.

    Strips whitespace and lowercases to catch slight variations
    while maintaining deterministic behavior.
    """
    return " ".join(text.lower().split())


def _is_page_number(text: str) -> bool:
    """
    Check if text matches any page number pattern.

    Args:
        text: Text to check

    Returns:
        True if text matches a page number pattern
    """
    stripped = text.strip()
    if not stripped:
        return False

    for pattern in PAGE_NUMBER_PATTERNS:
        if pattern.match(stripped):
            return True
    return False


def _is_short_repeated_text(text: str, max_length: int = 100) -> bool:
    """
    Check if text is short enough to be a potential header/footer.

    Long blocks are unlikely to be headers/footers.
    """
    return len(text.strip()) <= max_length


def detect_headers_footers(
    blocks: list[dict[str, Any]],
    min_page_repetition: int = 2,
) -> dict[str, HeaderFooterResult]:
    """
    Detect headers and footers across all blocks.

    Uses deterministic rules:
    1. Text repeated on multiple pages (likely header/footer)
    2. Page number patterns
    3. Position heuristics (first/last blocks on pages)

    Args:
        blocks: List of content blocks with 'block_id', 'text', and 'page' keys
        min_page_repetition: Minimum pages text must appear on to be header/footer

    Returns:
        Dictionary mapping block_id to HeaderFooterResult
    """
    results: dict[str, HeaderFooterResult] = {}

    if not blocks:
        return results

    # Group blocks by page
    blocks_by_page: dict[int, list[dict[str, Any]]] = {}
    for block in blocks:
        page = block.get("page", 1)
        if page not in blocks_by_page:
            blocks_by_page[page] = []
        blocks_by_page[page].append(block)

    # Track text occurrences across pages for repetition detection
    # Key: normalized text, Value: set of pages where it appears
    text_page_occurrences: dict[str, set[int]] = {}

    for block in blocks:
        text = block.get("text", "")
        page = block.get("page", 1)

        if _is_short_repeated_text(text):
            normalized = _normalize_text_for_comparison(text)
            if normalized:
                if normalized not in text_page_occurrences:
                    text_page_occurrences[normalized] = set()
                text_page_occurrences[normalized].add(page)

    # Identify repeated texts (appear on multiple pages)
    repeated_texts: set[str] = {
        text for text, pages in text_page_occurrences.items() if len(pages) >= min_page_repetition
    }

    # Process each block
    for _page_num, page_blocks in blocks_by_page.items():
        for idx, block in enumerate(page_blocks):
            block_id = block.get("block_id", "")
            text = block.get("text", "")
            normalized = _normalize_text_for_comparison(text)

            is_header = False
            is_footer = False

            # Rule 1: Page numbers are footers
            if _is_page_number(text):
                is_footer = True

            # Rule 2: Repeated text across pages
            elif normalized in repeated_texts:
                # First block on page is likely header
                if idx == 0:
                    is_header = True
                # Last block on page is likely footer
                elif idx == len(page_blocks) - 1:
                    is_footer = True
                # If repeated but middle of page, mark as header (conservative)
                else:
                    is_header = True

            results[block_id] = HeaderFooterResult(
                is_header=is_header,
                is_footer=is_footer,
            )

    return results
