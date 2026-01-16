"""
Stage 9: Timeline Reconstruction - Gap Detector

Gap identification between consecutive events.

IMPORTANT:
- Detect gaps between events
- Report gaps — do NOT explain them
- Configurable thresholds:
  < 60 min → ignored
  60-120 min → MODERATE
  > 120 min → SIGNIFICANT
"""

from .models import GapSeverity, TimelineConfig, TimelineEvent, TimelineGap
from .sorter import get_duration_minutes


def calculate_gap_severity(duration_minutes: int, config: TimelineConfig) -> GapSeverity:
    """
    Calculate gap severity based on duration.

    Args:
        duration_minutes: Gap duration in minutes.
        config: Timeline configuration.

    Returns:
        Gap severity level.
    """
    if duration_minutes >= config.significant_gap_minutes:
        return GapSeverity.SIGNIFICANT
    else:
        return GapSeverity.MODERATE


def detect_gaps(
    events: list[TimelineEvent],
    config: TimelineConfig | None = None,
) -> list[TimelineGap]:
    """
    Detect gaps between consecutive events.

    Only gaps >= threshold are reported.
    Gaps are detected but NOT explained.

    Args:
        events: Chronologically sorted list of events.
        config: Configuration for gap thresholds.

    Returns:
        List of detected gaps.
    """
    if config is None:
        config = TimelineConfig()

    if len(events) < 2:
        return []

    gaps: list[TimelineGap] = []

    for i in range(len(events) - 1):
        current_event = events[i]
        next_event = events[i + 1]

        # Calculate duration between events
        duration = get_duration_minutes(current_event.timestamp, next_event.timestamp)

        if duration is None:
            continue  # Skip if timestamps can't be parsed

        # Only report gaps >= threshold
        if duration >= config.gap_threshold_minutes:
            severity = calculate_gap_severity(duration, config)

            gap = TimelineGap(
                start=current_event.timestamp,
                end=next_event.timestamp,
                duration_minutes=duration,
                severity=severity,
                before_event_id=current_event.event_id,
                after_event_id=next_event.event_id,
            )
            gaps.append(gap)

    return gaps


def get_gap_summary(gaps: list[TimelineGap]) -> dict[str, int]:
    """
    Get summary of gaps by severity.

    Args:
        gaps: List of detected gaps.

    Returns:
        Dictionary with counts by severity.
    """
    summary = {
        "total": len(gaps),
        "moderate": 0,
        "significant": 0,
    }

    for gap in gaps:
        if gap.severity == GapSeverity.MODERATE:
            summary["moderate"] += 1
        elif gap.severity == GapSeverity.SIGNIFICANT:
            summary["significant"] += 1

    return summary


def get_total_gap_duration(gaps: list[TimelineGap]) -> int:
    """
    Get total duration of all gaps in minutes.

    Args:
        gaps: List of detected gaps.

    Returns:
        Total gap duration in minutes.
    """
    return sum(gap.duration_minutes for gap in gaps)


def find_largest_gap(gaps: list[TimelineGap]) -> TimelineGap | None:
    """
    Find the largest gap by duration.

    Args:
        gaps: List of detected gaps.

    Returns:
        Largest gap, or None if no gaps.
    """
    if not gaps:
        return None

    return max(gaps, key=lambda g: g.duration_minutes)
