"""
Stage 11: RAG - Timeline Checker

Consults timeline for relevant events, gaps, and conflicts.

IMPORTANT:
- Timeline is READ-ONLY
- Identifies gaps to report as limitations
- Never modifies timeline
"""

from typing import Any

from .models import TimelineEvent


def find_relevant_events(
    question: str,
    timeline_events: list[dict[str, Any]],
    chunk_ids: list[str],
) -> list[TimelineEvent]:
    """
    Find timeline events relevant to the question.

    Args:
        question: Investigator question.
        timeline_events: All timeline events from Stage 9.
        chunk_ids: Retrieved chunk IDs.

    Returns:
        Relevant timeline events.
    """
    relevant = []

    # Events from retrieved chunks
    for event in timeline_events:
        event_chunk_id = event.get("chunk_id", "")
        if event_chunk_id in chunk_ids:
            relevant.append(
                TimelineEvent(
                    event_id=event.get("event_id", ""),
                    timestamp=event.get("timestamp", ""),
                    description=event.get("description", ""),
                    chunk_id=event_chunk_id,
                )
            )

    # Sort by timestamp
    relevant.sort(key=lambda e: e.timestamp)

    return relevant


def detect_timeline_gaps(
    events: list[TimelineEvent],
    gaps: list[dict[str, Any]],
) -> list[str]:
    """
    Identify timeline gaps that should be reported as limitations.

    Args:
        events: Relevant timeline events.
        gaps: Gap objects from Stage 9.

    Returns:
        List of gap descriptions for limitations.
    """
    limitations = []

    if not events or not gaps:
        return limitations

    # Get timestamp range of relevant events
    if len(events) >= 2:
        start_ts = events[0].timestamp
        end_ts = events[-1].timestamp

        # Find gaps within this range
        for gap in gaps:
            gap_start = gap.get("start", "")
            gap_end = gap.get("end", "")
            duration = gap.get("duration_minutes", 0)
            severity = gap.get("severity", "")

            # Check if gap is within relevant time range
            if gap_start >= start_ts and gap_end <= end_ts:
                limitations.append(
                    f"Timeline contains a {duration}-minute gap between {gap_start} and {gap_end} ({severity})"
                )

    return limitations


def events_to_context(events: list[TimelineEvent]) -> str:
    """
    Convert timeline events to context string.

    Args:
        events: Timeline events.

    Returns:
        Formatted context string.
    """
    if not events:
        return ""

    lines = ["Timeline of relevant events:"]
    for event in events:
        lines.append(f"- [{event.timestamp}] {event.description}")

    return "\n".join(lines)


def get_event_timestamps(events: list[TimelineEvent]) -> list[str]:
    """
    Extract timestamps from events.

    Args:
        events: Timeline events.

    Returns:
        List of timestamps.
    """
    return [e.timestamp for e in events]


def find_conflicting_timestamps(
    events: list[TimelineEvent],
    conflicts: list[dict[str, Any]],
) -> list[str]:
    """
    Find timestamp conflicts related to relevant events.

    Args:
        events: Relevant timeline events.
        conflicts: Conflict objects from Stage 9.

    Returns:
        List of conflict descriptions.
    """
    limitations = []

    event_chunk_ids = {e.chunk_id for e in events}

    for conflict in conflicts:
        conflict_timestamp = conflict.get("timestamp", "")
        conflicting_chunks = conflict.get("conflicting_chunks", [])

        # Check if any conflicting chunks are in our events
        if any(cid in event_chunk_ids for cid in conflicting_chunks):
            limitations.append(f"Conflicting information at {conflict_timestamp}")

    return limitations
