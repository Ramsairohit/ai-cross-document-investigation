"""
Stage 3: Structural Parsing - Section Boundary Detection

Rule-based detection of document section boundaries.

Section headers are identified by:
1. All-caps lines (e.g., STATEMENT, INTERVIEW, OBSERVATIONS)
2. Short lines that appear to be headings
3. Common legal/police document section patterns

IMPORTANT: Uses pattern matching only. No NLP or inference.
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

# Common section header patterns in police/legal documents
SECTION_KEYWORDS: set[str] = {
    # Interview/statement sections
    "STATEMENT",
    "WITNESS STATEMENT",
    "VICTIM STATEMENT",
    "INTERVIEW",
    "INTERROGATION",
    "DEPOSITION",
    "TESTIMONY",
    "DECLARATION",
    # Report sections
    "SUMMARY",
    "EXECUTIVE SUMMARY",
    "SYNOPSIS",
    "OVERVIEW",
    "INTRODUCTION",
    "BACKGROUND",
    "OBSERVATIONS",
    "FINDINGS",
    "ANALYSIS",
    "CONCLUSION",
    "CONCLUSIONS",
    "RECOMMENDATIONS",
    # Evidence sections
    "EVIDENCE",
    "EXHIBITS",
    "PHYSICAL EVIDENCE",
    "DOCUMENTARY EVIDENCE",
    "ATTACHMENTS",
    "APPENDIX",
    "APPENDICES",
    # Narrative sections
    "NARRATIVE",
    "INCIDENT NARRATIVE",
    "FACTS",
    "STATEMENT OF FACTS",
    "CHRONOLOGY",
    "TIMELINE",
    "SEQUENCE OF EVENTS",
    # Administrative sections
    "CASE INFORMATION",
    "CASE DETAILS",
    "INCIDENT DETAILS",
    "RESPONDING OFFICERS",
    "OFFICERS PRESENT",
    "PARTIES INVOLVED",
    "WITNESSES",
    "SUSPECTS",
    "VICTIMS",
    # Closing sections
    "DISPOSITION",
    "REFERRALS",
    "FOLLOW-UP",
    "NEXT STEPS",
    "CERTIFICATION",
    "SIGNATURES",
}

# Patterns for section headers
SECTION_PATTERNS: list[re.Pattern[str]] = [
    # Numbered sections: "1. INTRODUCTION", "I. BACKGROUND"
    re.compile(r"^(?:\d+\.|\(?[IVXLC]+[.)])\s*([A-Z][A-Z\s]+)$"),
    # Lettered sections: "A. STATEMENT", "(a) FINDINGS"
    re.compile(r"^(?:[A-Z][.)]\s*|\([a-z]\)\s*)([A-Z][A-Z\s]+)$"),
    # Section with colon: "SECTION: INTERVIEW"
    re.compile(r"^SECTION\s*:\s*([A-Z][A-Z\s]+)$", re.IGNORECASE),
    # Underlined effect (repeated dashes/equals after text)
    re.compile(r"^([A-Z][A-Z\s]+)\s*[-=]{3,}$"),
]


@dataclass
class SectionDetectionResult:
    """Result of section detection for a single block."""

    is_section_header: bool
    section_name: Optional[str]


class SectionTracker:
    """
    Tracks current section state across blocks.

    Maintains the current section name as blocks are processed sequentially.
    """

    def __init__(self) -> None:
        self._current_section: Optional[str] = None

    @property
    def current_section(self) -> Optional[str]:
        """Get the current section name."""
        return self._current_section

    def update_section(self, section_name: str) -> None:
        """Update the current section."""
        self._current_section = section_name

    def reset(self) -> None:
        """Reset the section tracker."""
        self._current_section = None


def _is_all_caps(text: str) -> bool:
    """Check if text is primarily uppercase letters."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return False
    uppercase_count = sum(1 for c in alpha_chars if c.isupper())
    return uppercase_count / len(alpha_chars) >= 0.9


def _normalize_section_name(text: str) -> str:
    """Normalize section name for consistency."""
    # Remove leading/trailing whitespace and punctuation
    normalized = text.strip().strip(".:;-–—")
    # Normalize spacing
    normalized = " ".join(normalized.split())
    # Uppercase for consistency
    return normalized.upper()


def detect_section(
    text: str,
    max_length: int = 50,
) -> SectionDetectionResult:
    """
    Detect if a text block is a section header.

    Args:
        text: Text content to analyze
        max_length: Maximum character length for section headers

    Returns:
        SectionDetectionResult indicating if this is a section header
    """
    if not text or not text.strip():
        return SectionDetectionResult(is_section_header=False, section_name=None)

    stripped = text.strip()

    # Section headers are typically short
    if len(stripped) > max_length:
        return SectionDetectionResult(is_section_header=False, section_name=None)

    # Check if it matches known section keywords
    normalized = _normalize_section_name(stripped)
    if normalized in SECTION_KEYWORDS:
        return SectionDetectionResult(
            is_section_header=True,
            section_name=normalized,
        )

    # Check pattern matches
    for pattern in SECTION_PATTERNS:
        match = pattern.match(stripped)
        if match:
            section_name = match.group(1) if match.lastindex else stripped
            return SectionDetectionResult(
                is_section_header=True,
                section_name=_normalize_section_name(section_name),
            )

    # Check for all-caps short lines that look like headers
    # Must be all caps, relatively short, and contain only letters/spaces/punctuation
    if (
        len(stripped) <= 30
        and _is_all_caps(stripped)
        and re.match(r"^[A-Z][A-Z\s\-–—:.]+$", stripped)
    ):
        return SectionDetectionResult(
            is_section_header=True,
            section_name=_normalize_section_name(stripped),
        )

    return SectionDetectionResult(is_section_header=False, section_name=None)


def assign_sections(
    blocks: list[dict[str, Any]],
    max_section_header_length: int = 50,
) -> dict[str, Optional[str]]:
    """
    Assign section names to all blocks.

    Processes blocks sequentially, tracking the current section
    and assigning it to subsequent blocks until a new section is detected.

    Args:
        blocks: List of content blocks with 'block_id' and 'text' keys
        max_section_header_length: Maximum length for section headers

    Returns:
        Dictionary mapping block_id to section name (or None)
    """
    results: dict[str, Optional[str]] = {}
    tracker = SectionTracker()

    for block in blocks:
        block_id = block.get("block_id", "")
        text = block.get("text", "")

        detection = detect_section(text, max_length=max_section_header_length)

        if detection.is_section_header and detection.section_name:
            tracker.update_section(detection.section_name)

        results[block_id] = tracker.current_section

    return results
