"""
Stage 3: Structural Parsing - Speaker Label Detection

Regex-based detection and extraction of speaker labels from text blocks.

Supported formats:
- DET. SMITH:
- OFFICER JOHN DOE:
- WITNESS:
- Q: / A:
- THE COURT:
- MR./MRS./MS./DR. NAME:

IMPORTANT: Uses regex only. No NLP, no inference.
"""

import re
from dataclasses import dataclass
from typing import Optional

# Speaker label patterns - order matters (more specific first)
# Each tuple: (compiled pattern, group index for speaker name)
SPEAKER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    # Title + Name patterns: "DET. SMITH:", "OFFICER JOHN DOE:", "SGT. MILLER:"
    (
        re.compile(
            r"^((?:DET\.?|DETECTIVE|OFFICER|OFC\.?|SGT\.?|SERGEANT|LT\.?|LIEUTENANT|"
            r"CPT\.?|CAPTAIN|CHIEF|DEPUTY|AGENT|INSPECTOR|INVESTIGATOR)\s+"
            r"[A-Z][A-Z\s\.\-']+)\s*:\s*",
            re.IGNORECASE,
        ),
        1,
    ),
    # Honorific patterns: "MR. SMITH:", "DR. JONES:", "MS. DOE:"
    (
        re.compile(
            r"^((?:MR\.?|MRS\.?|MS\.?|MISS|DR\.?|PROF\.?|JUDGE|HON\.?|THE\s+HONORABLE)\s+"
            r"[A-Z][A-Z\s\.\-']+)\s*:\s*",
            re.IGNORECASE,
        ),
        1,
    ),
    # Role-based patterns: "WITNESS:", "DEFENDANT:", "THE COURT:"
    (
        re.compile(
            r"^(THE\s+(?:COURT|WITNESS|DEFENDANT|PLAINTIFF|PROSECUTOR|"
            r"DEFENSE|STATE|GOVERNMENT|ACCUSED))\s*:\s*",
            re.IGNORECASE,
        ),
        1,
    ),
    # Simple role patterns: "WITNESS:", "VICTIM:", "SUSPECT:"
    (
        re.compile(
            r"^(WITNESS|VICTIM|SUSPECT|COMPLAINANT|DEFENDANT|PLAINTIFF|"
            r"PROSECUTOR|ATTORNEY|COUNSEL|CLERK|BAILIFF|REPORTER)\s*:\s*",
            re.IGNORECASE,
        ),
        1,
    ),
    # Numbered witness: "WITNESS 1:", "WITNESS #2:"
    (
        re.compile(r"^(WITNESS\s*#?\d+)\s*:\s*", re.IGNORECASE),
        1,
    ),
    # Q&A format: "Q:", "A:", "Q.", "A."
    (
        re.compile(r"^([QA])\s*[.:]\s*", re.IGNORECASE),
        1,
    ),
    # Generic NAME: pattern (ALL CAPS name followed by colon)
    # This is last as it's most permissive
    (
        re.compile(r"^([A-Z][A-Z\s\.\-']{1,30})\s*:\s*"),
        1,
    ),
]


@dataclass
class SpeakerDetectionResult:
    """Result of speaker detection for a single block."""

    speaker: Optional[str]
    cleaned_text: str


def detect_speaker(text: str) -> SpeakerDetectionResult:
    """
    Detect and extract speaker label from text.

    The speaker label is removed from the text and returned separately.
    Speaker labels must appear at the START of the text block.

    Args:
        text: Original text content

    Returns:
        SpeakerDetectionResult with speaker name (or None) and cleaned text
    """
    if not text or not text.strip():
        return SpeakerDetectionResult(speaker=None, cleaned_text=text)

    stripped_text = text.strip()

    for pattern, group_idx in SPEAKER_PATTERNS:
        match = pattern.match(stripped_text)
        if match:
            # Extract speaker name and normalize
            speaker = match.group(group_idx).strip()
            # Normalize spacing and case
            speaker = " ".join(speaker.split()).upper()

            # Get remaining text after speaker label
            remaining_text = stripped_text[match.end() :].strip()

            return SpeakerDetectionResult(
                speaker=speaker,
                cleaned_text=remaining_text,
            )

    # No speaker detected
    return SpeakerDetectionResult(speaker=None, cleaned_text=text)


def normalize_speaker_name(speaker: str) -> str:
    """
    Normalize speaker name for consistency.

    Expands common abbreviations and standardizes format.

    Args:
        speaker: Raw speaker name

    Returns:
        Normalized speaker name
    """
    if not speaker:
        return speaker

    # Already uppercase from detect_speaker
    normalized = speaker.strip().upper()

    # Normalize common abbreviations
    abbreviation_map = {
        "DET ": "DETECTIVE ",
        "OFC ": "OFFICER ",
        "SGT ": "SERGEANT ",
        "LT ": "LIEUTENANT ",
        "CPT ": "CAPTAIN ",
        "DR ": "DR. ",
        "MR ": "MR. ",
        "MRS ": "MRS. ",
        "MS ": "MS. ",
        "HON ": "HONORABLE ",
    }

    for abbrev, full in abbreviation_map.items():
        if normalized.startswith(abbrev):
            normalized = full + normalized[len(abbrev) :]
            break

    return normalized
