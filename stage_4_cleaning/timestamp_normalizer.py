"""
Stage 4: Semantic Cleaning - Timestamp Normalization

Parse raw timestamps to ISO-8601 format with confidence scores.

IMPORTANT:
- Uses deterministic date parsing (dateutil, dateparser)
- If parsing is ambiguous → iso is null
- NO temporal inference
- NO timeline ordering
- NO date guessing
- Deterministic: same input → same output
"""

import re
from datetime import date, datetime, time
from typing import Optional

import dateparser
from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

from .models import NormalizedTimestamp

# Regex patterns for time-only detection
TIME_ONLY_PATTERNS = [
    re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?\s*[AaPp][Mm]$"),  # 8:15 PM
    re.compile(r"^\d{1,2}:\d{2}(?::\d{2})?$"),  # 14:30
    re.compile(r"^\d{4}\s*(?:hours?|hrs?)$", re.IGNORECASE),  # 0815 hours
    re.compile(r"^\d{1,2}\s+o'clock(?:\s*[AaPp][Mm])?$", re.IGNORECASE),  # 8 o'clock
]

# Patterns that indicate relative/ambiguous time references
AMBIGUOUS_PATTERNS = [
    re.compile(r"^(?:yesterday|today|tonight|tomorrow)$", re.IGNORECASE),
    re.compile(r"^(?:last|this|next)\s+", re.IGNORECASE),
    re.compile(r"^the\s+(?:night|morning|afternoon|evening)\s+of$", re.IGNORECASE),
    re.compile(r"^(?:around|approximately|about|approx\.?)\s+", re.IGNORECASE),
]

# Date formats that indicate high confidence parsing
UNAMBIGUOUS_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),  # ISO format: 2024-03-15
    re.compile(  # Full month name with year
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}",
        re.IGNORECASE,
    ),
]


def is_time_only(raw_timestamp: str) -> bool:
    """
    Check if a timestamp string contains only time (no date).

    Args:
        raw_timestamp: The raw timestamp string

    Returns:
        True if the string is time-only, False otherwise
    """
    stripped = raw_timestamp.strip()
    return any(pattern.match(stripped) for pattern in TIME_ONLY_PATTERNS)


def is_ambiguous_reference(raw_timestamp: str) -> bool:
    """
    Check if a timestamp is a relative/ambiguous reference.

    These references like "yesterday", "last Monday" cannot be
    deterministically parsed without knowing the reference date.

    Args:
        raw_timestamp: The raw timestamp string

    Returns:
        True if the timestamp is ambiguous, False otherwise
    """
    stripped = raw_timestamp.strip()
    return any(pattern.match(stripped) for pattern in AMBIGUOUS_PATTERNS)


def has_unambiguous_date(raw_timestamp: str) -> bool:
    """
    Check if a timestamp has an unambiguous date format.

    Args:
        raw_timestamp: The raw timestamp string

    Returns:
        True if the date format is unambiguous, False otherwise
    """
    return any(pattern.search(raw_timestamp) for pattern in UNAMBIGUOUS_DATE_PATTERNS)


def parse_military_time(raw_timestamp: str) -> Optional[time]:
    """
    Parse military/police time format (e.g., "0815 hours").

    Args:
        raw_timestamp: The raw timestamp string

    Returns:
        Parsed time object or None if not military format
    """
    match = re.match(r"^(\d{4})\s*(?:hours?|hrs?)$", raw_timestamp.strip(), re.IGNORECASE)
    if match:
        time_str = match.group(1)
        try:
            hours = int(time_str[:2])
            minutes = int(time_str[2:])
            if 0 <= hours <= 23 and 0 <= minutes <= 59:
                return time(hour=hours, minute=minutes)
        except ValueError:
            pass
    return None


def calculate_confidence(raw_timestamp: str, parsed_dt: Optional[datetime]) -> float:
    """
    Calculate confidence score for a parsed timestamp.

    Confidence is based on:
    - Whether parsing succeeded
    - Ambiguity of the original format
    - Completeness of the parsed result

    Args:
        raw_timestamp: Original timestamp string
        parsed_dt: Parsed datetime (or None if parsing failed)

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if parsed_dt is None:
        return 0.0

    # Start with base confidence
    confidence = 0.5

    # Boost for unambiguous date formats
    if has_unambiguous_date(raw_timestamp):
        confidence += 0.4  # High confidence for ISO or full month names

    # Moderate confidence for other parsed dates
    elif not is_time_only(raw_timestamp):
        confidence += 0.2

    # Lower confidence for time-only (missing date context)
    else:
        confidence += 0.1

    # Slight penalty for approximate times
    if re.search(r"(?:around|approximately|about|approx\.?)", raw_timestamp, re.IGNORECASE):
        confidence -= 0.2

    # Ensure bounds
    return max(0.0, min(1.0, confidence))


def normalize_timestamp(
    raw_timestamp: str,
    reference_date: Optional[date] = None,
) -> NormalizedTimestamp:
    """
    Normalize a raw timestamp to ISO-8601 format.

    Uses deterministic parsing with dateutil and dateparser.
    If parsing is ambiguous, returns null iso with low confidence.

    Args:
        raw_timestamp: Raw timestamp string exactly as found in text
        reference_date: Optional reference date for time-only timestamps.
                       If None, uses today's date for time-only parsing.

    Returns:
        NormalizedTimestamp with original, iso (or null), and confidence

    Note:
        This function does NOT guess missing dates or infer context.
        Ambiguous references like "yesterday" return null iso.
    """
    original = raw_timestamp.strip()

    # Handle empty input
    if not original:
        return NormalizedTimestamp(original=raw_timestamp, iso=None, confidence=0.0)

    # Check for ambiguous relative references
    if is_ambiguous_reference(original):
        return NormalizedTimestamp(original=original, iso=None, confidence=0.1)

    # Try parsing with dateutil first (more deterministic)
    parsed_dt: Optional[datetime] = None

    # Handle military time specially
    military_time = parse_military_time(original)
    if military_time:
        ref = reference_date or date.today()
        parsed_dt = datetime.combine(ref, military_time)
        confidence = 0.85 if reference_date else 0.6
        return NormalizedTimestamp(
            original=original,
            iso=parsed_dt.isoformat(),
            confidence=confidence,
        )

    # Handle time-only timestamps BEFORE dateutil (which would use today's date)
    if is_time_only(original):
        ref = reference_date or date.today()
        try:
            # Try to parse just the time
            time_match = re.match(
                r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?",
                original.strip(),
            )
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2))
                second = int(time_match.group(3)) if time_match.group(3) else 0
                ampm = time_match.group(4)

                if ampm:
                    if ampm.upper() == "PM" and hour != 12:
                        hour += 12
                    elif ampm.upper() == "AM" and hour == 12:
                        hour = 0

                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    parsed_dt = datetime.combine(
                        ref,
                        time(hour=hour, minute=minute, second=second),
                    )
                    confidence = 0.7 if reference_date else 0.5
                    return NormalizedTimestamp(
                        original=original,
                        iso=parsed_dt.isoformat(),
                        confidence=confidence,
                    )
        except (ValueError, AttributeError):
            pass

    # Try dateutil parser for full date/datetime strings
    try:
        # Use fuzzy=False for stricter parsing
        parsed_dt = dateutil_parser.parse(original, fuzzy=False)
    except (ParserError, ValueError, OverflowError):
        pass

    # If dateutil fails, try dateparser with strict settings
    if parsed_dt is None:
        try:
            parsed_dt = dateparser.parse(
                original,
                settings={
                    "STRICT_PARSING": True,
                    "PREFER_DATES_FROM": "past",  # Deterministic choice
                    "RETURN_AS_TIMEZONE_AWARE": False,
                },
            )
        except Exception:
            pass

    # Calculate confidence and return
    if parsed_dt is not None:
        confidence = calculate_confidence(original, parsed_dt)
        return NormalizedTimestamp(
            original=original,
            iso=parsed_dt.isoformat(),
            confidence=confidence,
        )

    # Parsing failed completely
    return NormalizedTimestamp(original=original, iso=None, confidence=0.0)


def normalize_timestamps(
    raw_timestamps: list[str],
    reference_date: Optional[date] = None,
) -> list[NormalizedTimestamp]:
    """
    Normalize a list of raw timestamps.

    Args:
        raw_timestamps: List of raw timestamp strings
        reference_date: Optional reference date for time-only timestamps

    Returns:
        List of NormalizedTimestamp objects in the same order
    """
    return [normalize_timestamp(ts, reference_date) for ts in raw_timestamps]
