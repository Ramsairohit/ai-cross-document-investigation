"""
Unit tests for Stage 9: Timeline Reconstruction - Gap Detector

Tests for gap identification.
"""

import pytest

from stage_9_timeline.gap_detector import (
    calculate_gap_severity,
    detect_gaps,
    find_largest_gap,
    get_gap_summary,
    get_total_gap_duration,
)
from stage_9_timeline.models import GapSeverity, TimelineConfig, TimelineEvent


def make_event(event_id: str, timestamp: str) -> TimelineEvent:
    """Helper to create a test event."""
    return TimelineEvent(
        event_id=event_id,
        timestamp=timestamp,
        chunk_id=f"C_{event_id}",
        document_id="D1",
        page_range=[1, 1],
        description=f"Event {event_id}",
        confidence=0.9,
    )


class TestCalculateGapSeverity:
    """Tests for gap severity calculation."""

    def test_moderate_severity(self):
        """60-120 minutes should be MODERATE."""
        config = TimelineConfig()

        assert calculate_gap_severity(60, config) == GapSeverity.MODERATE
        assert calculate_gap_severity(90, config) == GapSeverity.MODERATE
        assert calculate_gap_severity(119, config) == GapSeverity.MODERATE

    def test_significant_severity(self):
        """Over 120 minutes should be SIGNIFICANT."""
        config = TimelineConfig()

        assert calculate_gap_severity(120, config) == GapSeverity.SIGNIFICANT
        assert calculate_gap_severity(180, config) == GapSeverity.SIGNIFICANT
        assert calculate_gap_severity(1440, config) == GapSeverity.SIGNIFICANT

    def test_custom_thresholds(self):
        """Should respect custom thresholds."""
        config = TimelineConfig(
            gap_threshold_minutes=30,
            significant_gap_minutes=60,
        )

        assert calculate_gap_severity(30, config) == GapSeverity.MODERATE
        assert calculate_gap_severity(59, config) == GapSeverity.MODERATE
        assert calculate_gap_severity(60, config) == GapSeverity.SIGNIFICANT


class TestDetectGaps:
    """Tests for gap detection."""

    def test_detects_significant_gap(self):
        """Should detect gap over 2 hours."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T12:00:00"),  # 4 hour gap
        ]

        gaps = detect_gaps(events)

        assert len(gaps) == 1
        assert gaps[0].duration_minutes == 240
        assert gaps[0].severity == GapSeverity.SIGNIFICANT

    def test_detects_moderate_gap(self):
        """Should detect gap between 1-2 hours."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T09:30:00"),  # 90 minute gap
        ]

        gaps = detect_gaps(events)

        assert len(gaps) == 1
        assert gaps[0].duration_minutes == 90
        assert gaps[0].severity == GapSeverity.MODERATE

    def test_ignores_small_gaps(self):
        """Should ignore gaps under threshold."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T08:30:00"),  # 30 minute gap
            make_event("E3", "2024-03-15T08:45:00"),  # 15 minute gap
        ]

        gaps = detect_gaps(events)

        assert len(gaps) == 0

    def test_multiple_gaps(self):
        """Should detect multiple gaps."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T10:00:00"),  # 2 hour gap
            make_event("E3", "2024-03-15T10:15:00"),  # 15 min, ignored
            make_event("E4", "2024-03-15T12:30:00"),  # 2hr 15min gap
        ]

        gaps = detect_gaps(events)

        assert len(gaps) == 2

    def test_includes_event_ids(self):
        """Should include before and after event IDs."""
        events = [
            make_event("EVT_001", "2024-03-15T08:00:00"),
            make_event("EVT_002", "2024-03-15T10:00:00"),
        ]

        gaps = detect_gaps(events)

        assert gaps[0].before_event_id == "EVT_001"
        assert gaps[0].after_event_id == "EVT_002"

    def test_empty_events(self):
        """Should handle empty event list."""
        gaps = detect_gaps([])
        assert gaps == []

    def test_single_event(self):
        """Should handle single event."""
        events = [make_event("E1", "2024-03-15T08:00:00")]
        gaps = detect_gaps(events)
        assert gaps == []

    def test_custom_threshold(self):
        """Should respect custom threshold."""
        config = TimelineConfig(gap_threshold_minutes=30)

        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T08:45:00"),  # 45 minute gap
        ]

        gaps = detect_gaps(events, config)

        assert len(gaps) == 1
        assert gaps[0].duration_minutes == 45


class TestGapSummary:
    """Tests for gap summary utilities."""

    def test_get_gap_summary(self):
        """Should return correct summary."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T09:30:00"),  # 90 min - MODERATE
            make_event("E3", "2024-03-15T12:30:00"),  # 180 min - SIGNIFICANT
        ]

        gaps = detect_gaps(events)
        summary = get_gap_summary(gaps)

        assert summary["total"] == 2
        assert summary["moderate"] == 1
        assert summary["significant"] == 1

    def test_get_total_gap_duration(self):
        """Should sum all gap durations."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T09:30:00"),  # 90 min
            make_event("E3", "2024-03-15T12:30:00"),  # 180 min
        ]

        gaps = detect_gaps(events)
        total = get_total_gap_duration(gaps)

        assert total == 270  # 90 + 180

    def test_find_largest_gap(self):
        """Should find the largest gap."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T09:30:00"),  # 90 min
            make_event("E3", "2024-03-15T12:30:00"),  # 180 min
        ]

        gaps = detect_gaps(events)
        largest = find_largest_gap(gaps)

        assert largest is not None
        assert largest.duration_minutes == 180

    def test_find_largest_gap_empty(self):
        """Should return None for no gaps."""
        largest = find_largest_gap([])
        assert largest is None
