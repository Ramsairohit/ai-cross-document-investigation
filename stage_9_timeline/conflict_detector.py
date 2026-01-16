"""
Stage 9: Timeline Reconstruction - Conflict Detector

Conflict flagging for timeline events.

IMPORTANT:
- Flag conflicts when same timestamp has different speakers/locations
- Conflicts are FLAGGED but NOT resolved
- This stage does NOT decide truth
- No prioritization of sources
"""

from collections import defaultdict

from .models import TimelineConfig, TimelineConflict, TimelineEvent


def detect_conflicts(
    events: list[TimelineEvent],
    config: TimelineConfig | None = None,
) -> list[TimelineConflict]:
    """
    Detect conflicts in timeline events.

    A conflict is flagged when:
    - Same timestamp appears in multiple events
    - With different speakers

    Conflicts are FLAGGED but NOT resolved.

    Args:
        events: List of timeline events.
        config: Configuration for conflict detection.

    Returns:
        List of detected conflicts.
    """
    if config is None:
        config = TimelineConfig()

    # Group events by timestamp
    events_by_timestamp: dict[str, list[TimelineEvent]] = defaultdict(list)
    for event in events:
        events_by_timestamp[event.timestamp].append(event)

    conflicts: list[TimelineConflict] = []

    for timestamp, timestamp_events in events_by_timestamp.items():
        if len(timestamp_events) < 2:
            continue  # No conflict with single event

        # Check for speaker conflicts
        if config.detect_speaker_conflicts:
            speaker_conflict = detect_speaker_conflict(timestamp, timestamp_events)
            if speaker_conflict:
                conflicts.append(speaker_conflict)

    return conflicts


def detect_speaker_conflict(
    timestamp: str,
    events: list[TimelineEvent],
) -> TimelineConflict | None:
    """
    Detect if events at same timestamp have different speakers.

    Args:
        timestamp: The shared timestamp.
        events: Events at this timestamp.

    Returns:
        TimelineConflict if speakers differ, None otherwise.
    """
    # Get unique speakers (excluding None)
    speakers = set()
    for event in events:
        if event.speaker:
            speakers.add(event.speaker)

    # Only flag conflict if there are multiple different speakers
    if len(speakers) <= 1:
        return None

    # Calculate minimum confidence
    min_confidence = min(e.confidence for e in events)

    return TimelineConflict(
        timestamp=timestamp,
        conflicting_chunks=[e.chunk_id for e in events],
        conflicting_event_ids=[e.event_id for e in events],
        reason=f"Multiple events at same time with different speakers: {', '.join(sorted(speakers))}",
        confidence=min_confidence,
    )


def detect_document_conflict(
    timestamp: str,
    events: list[TimelineEvent],
) -> TimelineConflict | None:
    """
    Detect if events at same timestamp come from different documents.

    This may indicate conflicting statements from different sources.

    Args:
        timestamp: The shared timestamp.
        events: Events at this timestamp.

    Returns:
        TimelineConflict if documents differ, None otherwise.
    """
    # Get unique document IDs
    documents = set(e.document_id for e in events)

    # Only flag if multiple documents
    if len(documents) <= 1:
        return None

    # Calculate minimum confidence
    min_confidence = min(e.confidence for e in events)

    return TimelineConflict(
        timestamp=timestamp,
        conflicting_chunks=[e.chunk_id for e in events],
        conflicting_event_ids=[e.event_id for e in events],
        reason=f"Multiple events at same time from different documents: {', '.join(sorted(documents))}",
        confidence=min_confidence,
    )


def get_conflict_summary(conflicts: list[TimelineConflict]) -> dict[str, int]:
    """
    Get summary of conflicts.

    Args:
        conflicts: List of detected conflicts.

    Returns:
        Dictionary with conflict statistics.
    """
    return {
        "total": len(conflicts),
        "unique_timestamps": len(set(c.timestamp for c in conflicts)),
        "total_conflicting_chunks": sum(len(c.conflicting_chunks) for c in conflicts),
    }


def get_chunks_with_conflicts(conflicts: list[TimelineConflict]) -> set[str]:
    """
    Get all chunk IDs involved in conflicts.

    Args:
        conflicts: List of detected conflicts.

    Returns:
        Set of chunk IDs.
    """
    chunk_ids: set[str] = set()
    for conflict in conflicts:
        chunk_ids.update(conflict.conflicting_chunks)
    return chunk_ids
