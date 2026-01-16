"""
Unit tests for Stage 9: Timeline Reconstruction - Sorter

Tests for chronological ordering.
"""

import pytest

from stage_9_timeline.models import TimelineEvent
from stage_9_timeline.sorter import (
    get_duration_minutes,
    get_time_range,
    is_chronologically_ordered,
    parse_timestamp,
    sort_events,
)


def make_event(event_id: str, timestamp: str, confidence: float = 0.9) -> TimelineEvent:
    """Helper to create a test event."""
    return TimelineEvent(
        event_id=event_id,
        timestamp=timestamp,
        chunk_id=f"C_{event_id}",
        document_id="D1",
        page_range=[1, 1],
        description=f"Event {event_id}",
        confidence=confidence,
    )


class TestParseTimestamp:
    """Tests for timestamp parsing."""

    def test_full_iso(self):
        """Should parse full ISO timestamp."""
        dt = parse_timestamp("2024-03-15T20:15:00")
        assert dt is not None
        assert dt.year == 2024
        assert dt.month == 3
        assert dt.day == 15
        assert dt.hour == 20
        assert dt.minute == 15

    def test_date_only(self):
        """Should parse date-only timestamp."""
        dt = parse_timestamp("2024-03-15")
        assert dt is not None
        assert dt.year == 2024
        assert dt.hour == 0

    def test_invalid_timestamp(self):
        """Should return None for invalid timestamp."""
        dt = parse_timestamp("not a date")
        assert dt is None

    def test_empty_string(self):
        """Should return None for empty string."""
        dt = parse_timestamp("")
        assert dt is None


class TestSortEvents:
    """Tests for event sorting."""

    def test_sorts_chronologically(self):
        """Should sort events by timestamp."""
        events = [
            make_event("E3", "2024-03-15T12:00:00"),
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T10:00:00"),
        ]

        sorted_events = sort_events(events)

        assert sorted_events[0].event_id == "E1"
        assert sorted_events[1].event_id == "E2"
        assert sorted_events[2].event_id == "E3"

    def test_stable_sort(self):
        """Equal timestamps should preserve input order."""
        events = [
            make_event("E1", "2024-03-15T10:00:00"),
            make_event("E2", "2024-03-15T10:00:00"),
            make_event("E3", "2024-03-15T10:00:00"),
        ]

        sorted_events = sort_events(events)

        # Same timestamps - should preserve original order
        assert sorted_events[0].event_id == "E1"
        assert sorted_events[1].event_id == "E2"
        assert sorted_events[2].event_id == "E3"

    def test_empty_list(self):
        """Should handle empty list."""
        sorted_events = sort_events([])
        assert sorted_events == []

    def test_single_event(self):
        """Should handle single event."""
        events = [make_event("E1", "2024-03-15T10:00:00")]
        sorted_events = sort_events(events)
        assert len(sorted_events) == 1

    def test_deterministic(self):
        """Same input should always produce same order."""
        events = [
            make_event("E2", "2024-03-15T10:00:00"),
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E3", "2024-03-15T12:00:00"),
        ]

        results = [sort_events(events) for _ in range(50)]

        first_order = [e.event_id for e in results[0]]
        for result in results[1:]:
            order = [e.event_id for e in result]
            assert order == first_order

    def test_returns_new_list(self):
        """Should return new list, not mutate original."""
        events = [
            make_event("E2", "2024-03-15T10:00:00"),
            make_event("E1", "2024-03-15T08:00:00"),
        ]
        original_order = [e.event_id for e in events]

        sorted_events = sort_events(events)

        # Original should be unchanged
        assert [e.event_id for e in events] == original_order
        # Sorted should be different
        assert sorted_events is not events


class TestIsChronologicallyOrdered:
    """Tests for order checking."""

    def test_ordered_events(self):
        """Should return True for ordered events."""
        events = [
            make_event("E1", "2024-03-15T08:00:00"),
            make_event("E2", "2024-03-15T10:00:00"),
            make_event("E3", "2024-03-15T12:00:00"),
        ]
        assert is_chronologically_ordered(events) is True

    def test_unordered_events(self):
        """Should return False for unordered events."""
        events = [
            make_event("E1", "2024-03-15T12:00:00"),
            make_event("E2", "2024-03-15T08:00:00"),
        ]
        assert is_chronologically_ordered(events) is False

    def test_empty_list(self):
        """Should return True for empty list."""
        assert is_chronologically_ordered([]) is True

    def test_single_event(self):
        """Should return True for single event."""
        events = [make_event("E1", "2024-03-15T10:00:00")]
        assert is_chronologically_ordered(events) is True


class TestGetTimeRange:
    """Tests for time range extraction."""

    def test_returns_range(self):
        """Should return earliest and latest timestamps."""
        events = [
            make_event("E1", "2024-03-15T10:00:00"),
            make_event("E2", "2024-03-15T08:00:00"),
            make_event("E3", "2024-03-15T22:00:00"),
        ]

        earliest, latest = get_time_range(events)

        assert earliest == "2024-03-15T08:00:00"
        assert latest == "2024-03-15T22:00:00"

    def test_empty_list(self):
        """Should return None for empty list."""
        earliest, latest = get_time_range([])
        assert earliest is None
        assert latest is None


class TestGetDurationMinutes:
    """Tests for duration calculation."""

    def test_calculates_duration(self):
        """Should calculate duration in minutes."""
        duration = get_duration_minutes(
            "2024-03-15T08:00:00",
            "2024-03-15T10:30:00",
        )
        assert duration == 150  # 2.5 hours = 150 minutes

    def test_same_time(self):
        """Should return 0 for same time."""
        duration = get_duration_minutes(
            "2024-03-15T08:00:00",
            "2024-03-15T08:00:00",
        )
        assert duration == 0

    def test_invalid_timestamp(self):
        """Should return None for invalid timestamp."""
        duration = get_duration_minutes(
            "invalid",
            "2024-03-15T08:00:00",
        )
        assert duration is None
