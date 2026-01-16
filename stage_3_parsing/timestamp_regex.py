"""
Stage 3: Structural Parsing - Raw Timestamp Extraction

Regex-based extraction of timestamp mentions from text.

Extracts timestamps EXACTLY as they appear in text:
- Time patterns: 8:15 PM, 10:30 hours, 2345 hours
- Date patterns: March 15, 15/03/2024, 2024-03-15
- Combined: March 15, 2024 at 8:15 PM

IMPORTANT:
- NO normalization
- NO interpretation
- NO date parsing
- Returns raw strings exactly as found
"""

import re
from typing import Optional

# Time patterns
TIME_PATTERNS: list[re.Pattern[str]] = [
    # 12-hour format: 8:15 PM, 8:15PM, 8:15 am, 8:15:30 PM
    re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm])\b"),
    # 24-hour format: 14:30, 23:15:00
    re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b(?!\s*[AaPp][Mm])"),
    # Military/police format: 0815 hours, 2345 hours, 1430 hrs
    re.compile(r"\b(\d{4}\s*(?:hours?|hrs?))\b", re.IGNORECASE),
    # Written time: "at 8 o'clock", "at 9 o'clock PM"
    re.compile(r"\b(\d{1,2}\s+o'clock(?:\s*[AaPp][Mm])?)\b", re.IGNORECASE),
    # Approximate times: "around 8 PM", "approximately 10:30 AM"
    re.compile(
        r"\b((?:around|approximately|about|approx\.?)\s+\d{1,2}(?::\d{2})?\s*[AaPp][Mm])\b",
        re.IGNORECASE,
    ),
]

# Date patterns
DATE_PATTERNS: list[re.Pattern[str]] = [
    # Full month name: March 15, 2024 / March 15th, 2024 / 15 March 2024
    re.compile(
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"(?:\s*,?\s*\d{4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b",
        re.IGNORECASE,
    ),
    # Abbreviated month: Mar 15, 2024 / 15 Mar 2024 / Mar. 15
    re.compile(
        r"\b(\d{1,2}(?:st|nd|rd|th)?\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
        r"(?:\s*,?\s*\d{4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?)\b",
        re.IGNORECASE,
    ),
    # ISO format: 2024-03-15
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
    # US format: 03/15/2024, 3/15/24
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"),
    # European format: 15/03/2024, 15-03-2024
    re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
    # Dot format: 15.03.2024
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b"),
    # Month and year only: March 2024
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{4})\b",
        re.IGNORECASE,
    ),
]

# Combined date-time patterns
DATETIME_PATTERNS: list[re.Pattern[str]] = [
    # "March 15, 2024 at 8:15 PM"
    re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*\d{4})?"
        r"\s+(?:at\s+)?\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm])\b",
        re.IGNORECASE,
    ),
    # "15/03/2024 10:30"
    re.compile(r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s+\d{1,2}:\d{2}(?::\d{2})?)\b"),
    # "2024-03-15T10:30:00" (ISO 8601)
    re.compile(r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:?\d{2}|Z)?)\b"),
]

# Relative time expressions (commonly found in statements)
RELATIVE_PATTERNS: list[re.Pattern[str]] = [
    # "last Monday", "last week", "yesterday"
    re.compile(
        r"\b((?:last|this|next)\s+"
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
        r"week|month|year|night|morning|evening|afternoon))\b",
        re.IGNORECASE,
    ),
    # "yesterday", "today", "tonight"
    re.compile(r"\b(yesterday|today|tonight|tomorrow)\b", re.IGNORECASE),
    # "the night of", "the morning of"
    re.compile(r"\b(the\s+(?:night|morning|afternoon|evening)\s+of)\b", re.IGNORECASE),
]


def extract_timestamps(text: str) -> list[str]:
    """
    Extract all timestamp mentions from text.

    Returns timestamps EXACTLY as they appear in the text.
    No normalization or interpretation is performed.

    Args:
        text: Text to extract timestamps from

    Returns:
        List of raw timestamp strings in order of appearance
    """
    if not text:
        return []

    timestamps: list[tuple[int, str]] = []  # (position, text) for ordering

    # Combine all patterns
    all_patterns = DATETIME_PATTERNS + DATE_PATTERNS + TIME_PATTERNS + RELATIVE_PATTERNS

    # Track matched spans to avoid duplicates
    matched_spans: set[tuple[int, int]] = set()

    for pattern in all_patterns:
        for match in pattern.finditer(text):
            start, end = match.span(1) if match.lastindex else match.span()
            matched_text = match.group(1) if match.lastindex else match.group()

            # Check for overlap with existing matches
            is_overlapping = False
            for existing_start, existing_end in matched_spans:
                if not (end <= existing_start or start >= existing_end):
                    # Overlapping - prefer longer match
                    if (end - start) > (existing_end - existing_start):
                        matched_spans.discard((existing_start, existing_end))
                        # Remove the shorter match from timestamps
                        timestamps = [
                            (pos, txt) for pos, txt in timestamps if not (pos == existing_start)
                        ]
                    else:
                        is_overlapping = True
                    break

            if not is_overlapping:
                matched_spans.add((start, end))
                timestamps.append((start, matched_text.strip()))

    # Sort by position and return just the text
    timestamps.sort(key=lambda x: x[0])
    return [ts[1] for ts in timestamps]


def extract_timestamps_with_positions(text: str) -> list[tuple[str, int, int]]:
    """
    Extract timestamps with their positions in the text.

    Args:
        text: Text to extract timestamps from

    Returns:
        List of (timestamp_text, start_pos, end_pos) tuples
    """
    if not text:
        return []

    results: list[tuple[str, int, int]] = []
    matched_spans: set[tuple[int, int]] = set()

    all_patterns = DATETIME_PATTERNS + DATE_PATTERNS + TIME_PATTERNS + RELATIVE_PATTERNS

    for pattern in all_patterns:
        for match in pattern.finditer(text):
            start, end = match.span(1) if match.lastindex else match.span()
            matched_text = match.group(1) if match.lastindex else match.group()

            # Check overlaps
            is_overlapping = any(not (end <= es or start >= ee) for es, ee in matched_spans)

            if not is_overlapping:
                matched_spans.add((start, end))
                results.append((matched_text.strip(), start, end))

    results.sort(key=lambda x: x[1])
    return results
