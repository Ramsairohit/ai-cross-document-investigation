"""
Stage 6: NER - Rule-Based Entity Extraction

Regex and keyword-based extraction for specialized entity types:
- Phone numbers
- Addresses
- Weapons
- Evidence

IMPORTANT:
- Rules ADD entities, never ALTER text
- Deterministic: same input → same output
- No inference or interpretation
"""

import re
from dataclasses import dataclass

from .models import EVIDENCE_KEYWORDS, WEAPON_KEYWORDS, EntityType


@dataclass
class RuleMatch:
    """A single rule-based entity match."""

    text: str
    start_char: int
    end_char: int
    entity_type: EntityType
    confidence: float


# Phone number patterns
PHONE_PATTERNS: list[re.Pattern[str]] = [
    # US formats: (555) 123-4567, 555-123-4567, 555.123.4567
    re.compile(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
    # International: +1-555-123-4567, +44 20 7123 4567
    re.compile(r"\+\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}"),
    # Simple: 5551234567 (10 digits)
    re.compile(r"\b\d{10}\b"),
]

# Address patterns
ADDRESS_PATTERNS: list[re.Pattern[str]] = [
    # Street address: 420 Harrow Lane, 123 Main Street
    re.compile(
        r"\b\d{1,5}\s+[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+"
        r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|"
        r"Place|Pl|Way|Circle|Cir|Terrace|Ter|Highway|Hwy)\b",
        re.IGNORECASE,
    ),
    # PO Box
    re.compile(r"\bP\.?O\.?\s*Box\s+\d+\b", re.IGNORECASE),
]


def extract_phone_numbers(text: str) -> list[RuleMatch]:
    """
    Extract phone numbers from text using regex patterns.

    Args:
        text: Input text to search.

    Returns:
        List of RuleMatch objects for phone numbers found.
    """
    matches: list[RuleMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern in PHONE_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Avoid overlapping matches
            if any(not (end <= s or start >= e) for s, e in seen_spans):
                continue

            matched_text = match.group().strip()
            if len(matched_text) >= 7:  # Minimum phone length
                matches.append(
                    RuleMatch(
                        text=matched_text,
                        start_char=start,
                        end_char=end,
                        entity_type=EntityType.PHONE,
                        confidence=0.85,
                    )
                )
                seen_spans.add((start, end))

    return matches


def extract_addresses(text: str) -> list[RuleMatch]:
    """
    Extract street addresses from text using regex patterns.

    Args:
        text: Input text to search.

    Returns:
        List of RuleMatch objects for addresses found.
    """
    matches: list[RuleMatch] = []
    seen_spans: set[tuple[int, int]] = set()

    for pattern in ADDRESS_PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            # Avoid overlapping matches
            if any(not (end <= s or start >= e) for s, e in seen_spans):
                continue

            matched_text = match.group().strip()
            matches.append(
                RuleMatch(
                    text=matched_text,
                    start_char=start,
                    end_char=end,
                    entity_type=EntityType.ADDRESS,
                    confidence=0.80,
                )
            )
            seen_spans.add((start, end))

    return matches


def extract_weapons(text: str) -> list[RuleMatch]:
    """
    Extract weapon mentions from text using keyword matching.

    Args:
        text: Input text to search.

    Returns:
        List of RuleMatch objects for weapons found.
    """
    matches: list[RuleMatch] = []
    text_lower = text.lower()

    for keyword in WEAPON_KEYWORDS:
        # Find all occurrences of the keyword
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break

            # Check word boundaries
            before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
            after_pos = pos + len(keyword)
            after_ok = after_pos >= len(text_lower) or not text_lower[after_pos].isalnum()

            if before_ok and after_ok:
                # Get original case text
                original_text = text[pos : pos + len(keyword)]
                matches.append(
                    RuleMatch(
                        text=original_text,
                        start_char=pos,
                        end_char=pos + len(keyword),
                        entity_type=EntityType.WEAPON,
                        confidence=0.90,
                    )
                )

            start = pos + 1

    # Remove duplicates by span
    seen_spans: set[tuple[int, int]] = set()
    unique_matches: list[RuleMatch] = []
    for m in matches:
        span = (m.start_char, m.end_char)
        if span not in seen_spans:
            seen_spans.add(span)
            unique_matches.append(m)

    return unique_matches


def extract_evidence(text: str) -> list[RuleMatch]:
    """
    Extract evidence mentions from text using keyword matching.

    Args:
        text: Input text to search.

    Returns:
        List of RuleMatch objects for evidence found.
    """
    matches: list[RuleMatch] = []
    text_lower = text.lower()

    for keyword in EVIDENCE_KEYWORDS:
        start = 0
        while True:
            pos = text_lower.find(keyword, start)
            if pos == -1:
                break

            # Check word boundaries
            before_ok = pos == 0 or not text_lower[pos - 1].isalnum()
            after_pos = pos + len(keyword)
            after_ok = after_pos >= len(text_lower) or not text_lower[after_pos].isalnum()

            if before_ok and after_ok:
                original_text = text[pos : pos + len(keyword)]
                matches.append(
                    RuleMatch(
                        text=original_text,
                        start_char=pos,
                        end_char=pos + len(keyword),
                        entity_type=EntityType.EVIDENCE,
                        confidence=0.85,
                    )
                )

            start = pos + 1

    # Remove duplicates
    seen_spans: set[tuple[int, int]] = set()
    unique_matches: list[RuleMatch] = []
    for m in matches:
        span = (m.start_char, m.end_char)
        if span not in seen_spans:
            seen_spans.add(span)
            unique_matches.append(m)

    return unique_matches


def extract_all_rule_based(text: str) -> list[RuleMatch]:
    """
    Extract all rule-based entities from text.

    Combines phone, address, weapon, and evidence extraction.

    Args:
        text: Input text to search.

    Returns:
        List of all RuleMatch objects found.
    """
    all_matches: list[RuleMatch] = []

    all_matches.extend(extract_phone_numbers(text))
    all_matches.extend(extract_addresses(text))
    all_matches.extend(extract_weapons(text))
    all_matches.extend(extract_evidence(text))

    # Sort by start position
    all_matches.sort(key=lambda m: m.start_char)

    return all_matches
