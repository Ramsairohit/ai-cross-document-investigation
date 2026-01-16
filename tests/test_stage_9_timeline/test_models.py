"""
Unit tests for Stage 9: Timeline Reconstruction - Data Models

Tests for event, gap, conflict, and timeline schemas.
"""

import pytest
from pydantic import ValidationError

from stage_9_timeline.models import (
    GapSeverity,
    TimelineConfig,
    TimelineConflict,
    TimelineEvent,
    TimelineGap,
    TimelineResult,
)


class TestGapSeverity:
    """Tests for GapSeverity enum."""

    def test_severity_values(self):
        """Should have correct severity values."""
        assert GapSeverity.MODERATE.value == "MODERATE"
        assert GapSeverity.SIGNIFICANT.value == "SIGNIFICANT"

    def test_severity_count(self):
        """Should have exactly 2 severity levels."""
        assert len(GapSeverity) == 2


class TestTimelineEvent:
    """Tests for TimelineEvent model."""

    def test_valid_event(self):
        """Should create valid timeline event."""
        event = TimelineEvent(
            event_id="EVT_001",
            timestamp="2024-03-15T20:15:00",
            chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            description="I heard a loud crash around 8:15 PM.",
            speaker="Clara Higgins",
            confidence=0.92,
        )
        assert event.event_id == "EVT_001"
        assert event.timestamp == "2024-03-15T20:15:00"
        assert event.description == "I heard a loud crash around 8:15 PM."

    def test_event_without_speaker(self):
        """Speaker should be optional."""
        event = TimelineEvent(
            event_id="EVT_001",
            timestamp="2024-03-15T20:15:00",
            chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            description="Some event.",
            confidence=0.9,
        )
        assert event.speaker is None

    def test_missing_required_field(self):
        """Should reject missing required fields."""
        with pytest.raises(ValidationError):
            TimelineEvent(
                event_id="EVT_001",
                chunk_id="CHUNK_001",
                document_id="DOC123",
                page_range=[2, 2],
                description="Missing timestamp.",
                confidence=0.9,
            )

    def test_confidence_range(self):
        """Should validate confidence range."""
        # Valid at bounds
        event_low = TimelineEvent(
            event_id="E1",
            timestamp="2024-01-01T00:00:00",
            chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            description="Test",
            confidence=0.0,
        )
        assert event_low.confidence == 0.0

        event_high = TimelineEvent(
            event_id="E1",
            timestamp="2024-01-01T00:00:00",
            chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            description="Test",
            confidence=1.0,
        )
        assert event_high.confidence == 1.0

        # Invalid
        with pytest.raises(ValidationError):
            TimelineEvent(
                event_id="E1",
                timestamp="2024-01-01T00:00:00",
                chunk_id="C1",
                document_id="D1",
                page_range=[1, 1],
                description="Test",
                confidence=1.5,
            )


class TestTimelineGap:
    """Tests for TimelineGap model."""

    def test_valid_gap(self):
        """Should create valid timeline gap."""
        gap = TimelineGap(
            start="2024-03-15T20:15:00",
            end="2024-03-15T22:00:00",
            duration_minutes=105,
            severity=GapSeverity.SIGNIFICANT,
        )
        assert gap.duration_minutes == 105
        assert gap.severity == GapSeverity.SIGNIFICANT

    def test_gap_with_event_ids(self):
        """Should include event IDs if provided."""
        gap = TimelineGap(
            start="2024-03-15T20:00:00",
            end="2024-03-15T21:30:00",
            duration_minutes=90,
            severity=GapSeverity.MODERATE,
            before_event_id="EVT_001",
            after_event_id="EVT_002",
        )
        assert gap.before_event_id == "EVT_001"
        assert gap.after_event_id == "EVT_002"


class TestTimelineConflict:
    """Tests for TimelineConflict model."""

    def test_valid_conflict(self):
        """Should create valid timeline conflict."""
        conflict = TimelineConflict(
            timestamp="2024-03-15T21:00:00",
            conflicting_chunks=["CHUNK_004", "CHUNK_007"],
            reason="Multiple events at same time with different speakers",
            confidence=0.85,
        )
        assert len(conflict.conflicting_chunks) == 2
        assert conflict.confidence == 0.85

    def test_conflict_requires_multiple_chunks(self):
        """Should require at least 2 conflicting chunks."""
        with pytest.raises(ValidationError):
            TimelineConflict(
                timestamp="2024-03-15T21:00:00",
                conflicting_chunks=["CHUNK_004"],  # Only 1
                reason="Not enough chunks",
                confidence=0.85,
            )


class TestTimelineResult:
    """Tests for TimelineResult model."""

    def test_valid_result(self):
        """Should create valid timeline result."""
        result = TimelineResult(
            case_id="24-890-H",
            events=[],
            gaps=[],
            conflicts=[],
            total_events=0,
            total_gaps=0,
            total_conflicts=0,
        )
        assert result.case_id == "24-890-H"
        assert result.total_events == 0

    def test_default_lists(self):
        """Lists should default to empty."""
        result = TimelineResult(case_id="001")
        assert result.events == []
        assert result.gaps == []
        assert result.conflicts == []


class TestTimelineConfig:
    """Tests for TimelineConfig model."""

    def test_default_values(self):
        """Should have correct default values."""
        config = TimelineConfig()
        assert config.gap_threshold_minutes == 60
        assert config.significant_gap_minutes == 120
        assert config.detect_speaker_conflicts is True

    def test_custom_threshold(self):
        """Should accept custom thresholds."""
        config = TimelineConfig(
            gap_threshold_minutes=30,
            significant_gap_minutes=90,
        )
        assert config.gap_threshold_minutes == 30
        assert config.significant_gap_minutes == 90
