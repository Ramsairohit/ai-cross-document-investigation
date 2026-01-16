"""
Stage 9: Timeline Reconstruction - Confidence

Confidence score handling for timeline events.

IMPORTANT:
- Event confidence = minimum of chunk and timestamp confidence
- Gap confidence = derived from surrounding events
- Conflict confidence = minimum of conflicting events
"""

from .models import TimelineConflict, TimelineEvent, TimelineGap


def calculate_event_confidence(chunk_confidence: float, timestamp_confidence: float) -> float:
    """
    Calculate event confidence as minimum of chunk and timestamp confidence.

    Takes the conservative approach - confidence is limited by the
    weakest link in the chain.

    Args:
        chunk_confidence: Confidence from the source chunk (0.0-1.0).
        timestamp_confidence: Confidence from timestamp normalization (0.0-1.0).

    Returns:
        Minimum of both confidence values.
    """
    return min(chunk_confidence, timestamp_confidence)


def calculate_gap_confidence(
    before_event: TimelineEvent | None,
    after_event: TimelineEvent | None,
) -> float:
    """
    Calculate gap confidence based on surrounding events.

    Gap confidence is the minimum of the events on either side,
    since evidence of the gap depends on both events being reliable.

    Args:
        before_event: Event before the gap.
        after_event: Event after the gap.

    Returns:
        Confidence score for the gap (0.0-1.0).
    """
    confidences = []

    if before_event is not None:
        confidences.append(before_event.confidence)
    if after_event is not None:
        confidences.append(after_event.confidence)

    if not confidences:
        return 0.0

    return min(confidences)


def calculate_conflict_confidence(events: list[TimelineEvent]) -> float:
    """
    Calculate conflict confidence as minimum of conflicting events.

    A conflict is only as reliable as the least reliable
    piece of evidence supporting it.

    Args:
        events: List of conflicting events.

    Returns:
        Minimum confidence of all events.
    """
    if not events:
        return 0.0

    return min(e.confidence for e in events)


def get_average_confidence(events: list[TimelineEvent]) -> float:
    """
    Get average confidence across all events.

    Args:
        events: List of timeline events.

    Returns:
        Average confidence score.
    """
    if not events:
        return 0.0

    return sum(e.confidence for e in events) / len(events)


def get_confidence_distribution(
    events: list[TimelineEvent],
) -> dict[str, int]:
    """
    Get distribution of confidence scores.

    Categorizes confidence into ranges:
    - high: >= 0.9
    - medium: 0.7 - 0.9
    - low: < 0.7

    Args:
        events: List of timeline events.

    Returns:
        Dictionary with counts by confidence range.
    """
    distribution = {
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    for event in events:
        if event.confidence >= 0.9:
            distribution["high"] += 1
        elif event.confidence >= 0.7:
            distribution["medium"] += 1
        else:
            distribution["low"] += 1

    return distribution


def get_low_confidence_events(
    events: list[TimelineEvent],
    threshold: float = 0.7,
) -> list[TimelineEvent]:
    """
    Get events with confidence below threshold.

    Args:
        events: List of timeline events.
        threshold: Confidence threshold (default 0.7).

    Returns:
        List of low confidence events.
    """
    return [e for e in events if e.confidence < threshold]
