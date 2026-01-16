"""
Stage 9: Timeline Reconstruction - Sorter

Deterministic chronological ordering of timeline events.

IMPORTANT:
- Sort by ISO-8601 timestamp
- Stable ordering (same input → same order)
- If timestamps equal → preserve input order
"""

from datetime import datetime
from typing import Optional

from .models import TimelineEvent


def parse_timestamp(iso_timestamp: str) -> Optional[datetime]:
    """
    Parse an ISO-8601 timestamp string to datetime.

    Args:
        iso_timestamp: ISO-8601 formatted string.

    Returns:
        datetime object or None if parsing fails.
    """
    try:
        # Handle various ISO formats
        if "T" in iso_timestamp:
            # Standard ISO format with time
            if "." in iso_timestamp:
                # Has microseconds
                return datetime.fromisoformat(iso_timestamp)
            else:
                return datetime.fromisoformat(iso_timestamp)
        else:
            # Date only
            return datetime.fromisoformat(iso_timestamp + "T00:00:00")
    except (ValueError, TypeError):
        return None


def sort_events(events: list[TimelineEvent]) -> list[TimelineEvent]:
    """
    Sort events chronologically with stable ordering.

    Uses stable sort to preserve input order for equal timestamps.

    Args:
        events: List of timeline events.

    Returns:
        New list of events sorted chronologically.
    """
    if not events:
        return []

    # Create indexed list for stable sort
    indexed_events = list(enumerate(events))

    def sort_key(item: tuple[int, TimelineEvent]) -> tuple[datetime, int]:
        """Sort key: (timestamp, original_index) for stable sort."""
        index, event = item
        parsed = parse_timestamp(event.timestamp)
        # Use minimum datetime for unparseable timestamps
        if parsed is None:
            parsed = datetime.min
        return (parsed, index)

    # Sort by timestamp, then by original index (stable)
    sorted_indexed = sorted(indexed_events, key=sort_key)

    # Extract just the events
    return [event for _, event in sorted_indexed]


def is_chronologically_ordered(events: list[TimelineEvent]) -> bool:
    """
    Check if events are in chronological order.

    Args:
        events: List of timeline events.

    Returns:
        True if events are sorted chronologically.
    """
    if len(events) <= 1:
        return True

    for i in range(len(events) - 1):
        ts1 = parse_timestamp(events[i].timestamp)
        ts2 = parse_timestamp(events[i + 1].timestamp)

        if ts1 is None or ts2 is None:
            continue  # Skip unparseable timestamps

        if ts1 > ts2:
            return False

    return True


def get_time_range(events: list[TimelineEvent]) -> tuple[Optional[str], Optional[str]]:
    """
    Get the earliest and latest timestamps from events.

    Args:
        events: List of timeline events.

    Returns:
        Tuple of (earliest_timestamp, latest_timestamp) or (None, None) if empty.
    """
    if not events:
        return None, None

    sorted_events = sort_events(events)
    return sorted_events[0].timestamp, sorted_events[-1].timestamp


def get_duration_minutes(start_iso: str, end_iso: str) -> Optional[int]:
    """
    Calculate duration in minutes between two ISO timestamps.

    Args:
        start_iso: Start timestamp in ISO format.
        end_iso: End timestamp in ISO format.

    Returns:
        Duration in minutes, or None if parsing fails.
    """
    start = parse_timestamp(start_iso)
    end = parse_timestamp(end_iso)

    if start is None or end is None:
        return None

    delta = end - start
    return int(delta.total_seconds() / 60)
