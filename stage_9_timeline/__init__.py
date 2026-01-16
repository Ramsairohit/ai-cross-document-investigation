"""
Stage 9: Timeline Reconstruction

Reconstruct a chronological timeline of events from explicit timestamps,
detect gaps, and flag conflicts without resolving them.

HARD RULES (NON-NEGOTIABLE):
❌ No inference
❌ No timeline correction
❌ No guessing missing times
❌ No resolving contradictions
❌ No causal reasoning

✅ Deterministic ordering
✅ Explicit timestamps only
✅ Full provenance
✅ Gaps detected at thresholds
✅ Conflicts flagged (not resolved)

This stage ORDERS evidence — it does NOT interpret it.
"""

from .confidence import (
    calculate_conflict_confidence,
    calculate_event_confidence,
    calculate_gap_confidence,
    get_average_confidence,
    get_confidence_distribution,
    get_low_confidence_events,
)
from .conflict_detector import (
    detect_conflicts,
    detect_document_conflict,
    detect_speaker_conflict,
    get_chunks_with_conflicts,
    get_conflict_summary,
)
from .event_builder import (
    build_events,
    build_events_from_blocks,
    chunk_to_events,
    generate_event_id,
)
from .gap_detector import (
    calculate_gap_severity,
    detect_gaps,
    find_largest_gap,
    get_gap_summary,
    get_total_gap_duration,
)
from .models import (
    GapSeverity,
    TimelineConfig,
    TimelineConflict,
    TimelineEvent,
    TimelineGap,
    TimelineResult,
)
from .sorter import (
    get_duration_minutes,
    get_time_range,
    is_chronologically_ordered,
    parse_timestamp,
    sort_events,
)
from .timeline_pipeline import (
    TimelinePipeline,
    build_timeline_async,
    build_timeline_sync,
)

__all__ = [
    # Main Pipeline API
    "TimelinePipeline",
    "build_timeline_sync",
    "build_timeline_async",
    # Models
    "GapSeverity",
    "TimelineEvent",
    "TimelineGap",
    "TimelineConflict",
    "TimelineResult",
    "TimelineConfig",
    # Event Builder
    "generate_event_id",
    "chunk_to_events",
    "build_events",
    "build_events_from_blocks",
    # Sorter
    "parse_timestamp",
    "sort_events",
    "is_chronologically_ordered",
    "get_time_range",
    "get_duration_minutes",
    # Gap Detector
    "calculate_gap_severity",
    "detect_gaps",
    "get_gap_summary",
    "get_total_gap_duration",
    "find_largest_gap",
    # Conflict Detector
    "detect_conflicts",
    "detect_speaker_conflict",
    "detect_document_conflict",
    "get_conflict_summary",
    "get_chunks_with_conflicts",
    # Confidence
    "calculate_event_confidence",
    "calculate_gap_confidence",
    "calculate_conflict_confidence",
    "get_average_confidence",
    "get_confidence_distribution",
    "get_low_confidence_events",
]
