"""
Unit tests for Stage 9: Timeline Reconstruction - Conflict Detector

Tests for conflict flagging (NOT resolution).
"""

import pytest

from stage_9_timeline.conflict_detector import (
    detect_conflicts,
    detect_speaker_conflict,
    get_chunks_with_conflicts,
    get_conflict_summary,
)
from stage_9_timeline.models import TimelineConfig, TimelineEvent


def make_event(
    event_id: str,
    timestamp: str,
    speaker: str | None = None,
    chunk_id: str | None = None,
) -> TimelineEvent:
    """Helper to create a test event."""
    return TimelineEvent(
        event_id=event_id,
        timestamp=timestamp,
        chunk_id=chunk_id or f"C_{event_id}",
        document_id="D1",
        page_range=[1, 1],
        description=f"Event {event_id}",
        speaker=speaker,
        confidence=0.9,
    )


class TestDetectSpeakerConflict:
    """Tests for speaker conflict detection."""

    def test_detects_different_speakers(self):
        """Should detect conflict with different speakers."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
        ]

        conflict = detect_speaker_conflict("2024-03-15T08:00:00", events)

        assert conflict is not None
        assert len(conflict.conflicting_chunks) == 2
        assert "Alice" in conflict.reason
        assert "Bob" in conflict.reason

    def test_no_conflict_same_speaker(self):
        """Should not flag conflict with same speaker."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C2"),
        ]

        conflict = detect_speaker_conflict("2024-03-15T08:00:00", events)

        assert conflict is None

    def test_no_conflict_null_speakers(self):
        """Should not flag conflict with null speakers."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker=None, chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker=None, chunk_id="C2"),
        ]

        conflict = detect_speaker_conflict("2024-03-15T08:00:00", events)

        assert conflict is None


class TestDetectConflicts:
    """Tests for conflict detection."""

    def test_detects_conflict(self):
        """Should detect conflicts at same timestamp with different speakers."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
            make_event("E3", "2024-03-15T10:00:00", speaker="Alice", chunk_id="C3"),
        ]

        conflicts = detect_conflicts(events)

        assert len(conflicts) == 1
        assert conflicts[0].timestamp == "2024-03-15T08:00:00"

    def test_no_conflict_different_times(self):
        """Should not flag conflict at different times."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice"),
            make_event("E2", "2024-03-15T09:00:00", speaker="Bob"),
        ]

        conflicts = detect_conflicts(events)

        assert len(conflicts) == 0

    def test_no_conflict_single_event(self):
        """Should not flag conflict for single event at timestamp."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice"),
        ]

        conflicts = detect_conflicts(events)

        assert len(conflicts) == 0

    def test_multiple_conflicts(self):
        """Should detect multiple conflicts."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
            make_event("E3", "2024-03-15T10:00:00", speaker="Charlie", chunk_id="C3"),
            make_event("E4", "2024-03-15T10:00:00", speaker="Dave", chunk_id="C4"),
        ]

        conflicts = detect_conflicts(events)

        assert len(conflicts) == 2

    def test_conflict_includes_confidence(self):
        """Conflict should include minimum confidence."""
        events = [
            TimelineEvent(
                event_id="E1",
                timestamp="2024-03-15T08:00:00",
                chunk_id="C1",
                document_id="D1",
                page_range=[1, 1],
                description="Event 1",
                speaker="Alice",
                confidence=0.9,
            ),
            TimelineEvent(
                event_id="E2",
                timestamp="2024-03-15T08:00:00",
                chunk_id="C2",
                document_id="D1",
                page_range=[1, 1],
                description="Event 2",
                speaker="Bob",
                confidence=0.7,
            ),
        ]

        conflicts = detect_conflicts(events)

        assert conflicts[0].confidence == 0.7  # minimum

    def test_respects_config(self):
        """Should respect detection config."""
        config = TimelineConfig(detect_speaker_conflicts=False)

        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
        ]

        conflicts = detect_conflicts(events, config)

        assert len(conflicts) == 0  # Disabled by config

    def test_empty_events(self):
        """Should handle empty event list."""
        conflicts = detect_conflicts([])
        assert conflicts == []


class TestConflictNotResolved:
    """Tests to verify conflicts are not resolved."""

    def test_all_chunks_preserved(self):
        """All conflicting chunks should be in result."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
            make_event("E3", "2024-03-15T08:00:00", speaker="Charlie", chunk_id="C3"),
        ]

        conflicts = detect_conflicts(events)

        assert len(conflicts) == 1
        assert set(conflicts[0].conflicting_chunks) == {"C1", "C2", "C3"}

    def test_no_priority_assigned(self):
        """No source should be prioritized - all are included equally."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Witness", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Detective", chunk_id="C2"),
        ]

        conflicts = detect_conflicts(events)

        # Both should be in conflicting_chunks, no preference
        assert "C1" in conflicts[0].conflicting_chunks
        assert "C2" in conflicts[0].conflicting_chunks


class TestConflictUtilities:
    """Tests for conflict utility functions."""

    def test_get_conflict_summary(self):
        """Should return correct summary."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
        ]

        conflicts = detect_conflicts(events)
        summary = get_conflict_summary(conflicts)

        assert summary["total"] == 1
        assert summary["unique_timestamps"] == 1
        assert summary["total_conflicting_chunks"] == 2

    def test_get_chunks_with_conflicts(self):
        """Should return all chunk IDs involved in conflicts."""
        events = [
            make_event("E1", "2024-03-15T08:00:00", speaker="Alice", chunk_id="C1"),
            make_event("E2", "2024-03-15T08:00:00", speaker="Bob", chunk_id="C2"),
            make_event("E3", "2024-03-15T10:00:00", speaker="Alice", chunk_id="C3"),
        ]

        conflicts = detect_conflicts(events)
        chunks = get_chunks_with_conflicts(conflicts)

        assert chunks == {"C1", "C2"}
