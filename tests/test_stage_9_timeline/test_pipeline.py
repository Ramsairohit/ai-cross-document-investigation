"""
Unit tests for Stage 9: Timeline Reconstruction - Pipeline

Tests for the main pipeline API.
"""

import pytest

from stage_9_timeline.models import GapSeverity
from stage_9_timeline.timeline_pipeline import TimelinePipeline, build_timeline_sync


class TestTimelinePipeline:
    """Tests for TimelinePipeline class."""

    def test_build_timeline(self):
        """Should build complete timeline."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "I heard a crash at 8 PM.",
                "speaker": "Witness A",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "At 10 PM, I left the house.",
                "speaker": "Witness A",
                "chunk_confidence": 0.85,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8 PM", "iso": "2024-03-15T20:00:00", "confidence": 0.92}],
            "C2": [{"original": "10 PM", "iso": "2024-03-15T22:00:00", "confidence": 0.88}],
        }

        result = pipeline.build_timeline("24-890-H", chunks, timestamps_map)

        assert result.case_id == "24-890-H"
        assert result.total_events == 2
        assert result.chunks_processed == 2
        assert result.timestamps_processed == 2

    def test_events_are_sorted(self):
        """Events should be in chronological order."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Later event at 10 PM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Earlier event at 8 AM.",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "10 PM", "iso": "2024-03-15T22:00:00", "confidence": 0.9}],
            "C2": [{"original": "8 AM", "iso": "2024-03-15T08:00:00", "confidence": 0.9}],
        }

        result = pipeline.build_timeline("001", chunks, timestamps_map)

        # First event should be 8 AM
        assert result.events[0].timestamp == "2024-03-15T08:00:00"
        # Second event should be 10 PM
        assert result.events[1].timestamp == "2024-03-15T22:00:00"

    def test_detects_gaps(self):
        """Should detect gaps between events."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Morning event.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Evening event.",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8 AM", "iso": "2024-03-15T08:00:00", "confidence": 0.9}],
            "C2": [{"original": "6 PM", "iso": "2024-03-15T18:00:00", "confidence": 0.9}],
        }

        result = pipeline.build_timeline("001", chunks, timestamps_map)

        assert result.total_gaps == 1
        assert result.gaps[0].duration_minutes == 600  # 10 hours
        assert result.gaps[0].severity == GapSeverity.SIGNIFICANT

    def test_detects_conflicts(self):
        """Should detect conflicts at same timestamp."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "I was at the park.",
                "speaker": "Alice",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "I was at home.",
                "speaker": "Bob",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "3 PM", "iso": "2024-03-15T15:00:00", "confidence": 0.9}],
            "C2": [{"original": "3 PM", "iso": "2024-03-15T15:00:00", "confidence": 0.9}],
        }

        result = pipeline.build_timeline("001", chunks, timestamps_map)

        assert result.total_conflicts == 1
        assert len(result.conflicts[0].conflicting_chunks) == 2

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        pipeline = TimelinePipeline()

        result = pipeline.build_timeline("001", [], {})

        assert result.case_id == "001"
        assert result.total_events == 0
        assert result.total_gaps == 0
        assert result.total_conflicts == 0


class TestBuildTimelineSync:
    """Tests for sync helper function."""

    def test_convenience_function(self):
        """Should work as convenience function."""
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Test event.",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "noon", "iso": "2024-03-15T12:00:00", "confidence": 0.9}],
        }

        result = build_timeline_sync("001", chunks, timestamps_map)

        assert result.case_id == "001"
        assert result.total_events == 1


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input should always produce same output."""
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Event A at 10 AM.",
                "speaker": "Witness",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Event B at 2 PM.",
                "speaker": "Witness",
                "chunk_confidence": 0.85,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "10 AM", "iso": "2024-03-15T10:00:00", "confidence": 0.9}],
            "C2": [{"original": "2 PM", "iso": "2024-03-15T14:00:00", "confidence": 0.88}],
        }

        results = [build_timeline_sync("001", chunks, timestamps_map) for _ in range(50)]

        first = results[0]
        for result in results[1:]:
            assert result.total_events == first.total_events
            assert result.total_gaps == first.total_gaps
            assert result.total_conflicts == first.total_conflicts
            event_ids = [e.event_id for e in result.events]
            first_event_ids = [e.event_id for e in first.events]
            assert event_ids == first_event_ids

    def test_verify_determinism_method(self):
        """Pipeline should have determinism verification."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Test event.",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "noon", "iso": "2024-03-15T12:00:00", "confidence": 0.9}],
        }

        is_deterministic = pipeline.verify_determinism("001", chunks, timestamps_map, runs=100)

        assert is_deterministic is True


class TestNoInference:
    """Tests to verify no inference is made."""

    def test_description_is_exact_text(self):
        """Event description should be exact chunk text, no modification."""
        pipeline = TimelinePipeline()

        original_text = "I heard a loud crash around 8:15 PM."
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": original_text,
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8:15 PM", "iso": "2024-03-15T20:15:00", "confidence": 0.9}],
        }

        result = pipeline.build_timeline("001", chunks, timestamps_map)

        # Description should be EXACT text
        assert result.events[0].description == original_text

    def test_full_provenance_preserved(self):
        """Every event should have full provenance."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "CHUNK_XYZ",
                "document_id": "DOC_ABC",
                "page_range": [5, 5],
                "text": "Test event.",
                "speaker": "John Doe",
                "chunk_confidence": 0.87,
            },
        ]
        timestamps_map = {
            "CHUNK_XYZ": [
                {"original": "3:45 PM", "iso": "2024-03-15T15:45:00", "confidence": 0.91}
            ],
        }

        result = pipeline.build_timeline("001", chunks, timestamps_map)

        event = result.events[0]
        assert event.chunk_id == "CHUNK_XYZ"
        assert event.document_id == "DOC_ABC"
        assert event.page_range == [5, 5]
        assert event.speaker == "John Doe"
        assert event.raw_timestamp == "3:45 PM"
        assert 0.0 <= event.confidence <= 1.0
