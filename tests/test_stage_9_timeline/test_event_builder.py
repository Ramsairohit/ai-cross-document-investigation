"""
Unit tests for Stage 9: Timeline Reconstruction - Event Builder

Tests for chunk to event conversion.
"""

import pytest

from stage_9_timeline.event_builder import (
    build_events,
    chunk_to_events,
    generate_event_id,
)


class TestGenerateEventId:
    """Tests for event ID generation."""

    def test_format(self):
        """Should follow EVT_{case_id}_{index} format."""
        event_id = generate_event_id("24-890-H", 0)
        assert event_id == "EVT_24_890_H_0000"

    def test_deterministic(self):
        """Same input should always produce same output."""
        ids = [generate_event_id("24-890-H", 5) for _ in range(100)]
        assert all(id == ids[0] for id in ids)

    def test_different_indices(self):
        """Different indices should produce different IDs."""
        id1 = generate_event_id("001", 0)
        id2 = generate_event_id("001", 1)
        assert id1 != id2

    def test_different_cases(self):
        """Different case IDs should produce different IDs."""
        id1 = generate_event_id("001", 0)
        id2 = generate_event_id("002", 0)
        assert id1 != id2


class TestChunkToEvents:
    """Tests for chunk to event conversion."""

    def test_creates_event_per_timestamp(self):
        """Should create one event per timestamp."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "page_range": [1, 1],
            "text": "I heard something at 8:15 PM.",
            "speaker": "Witness A",
            "chunk_confidence": 0.9,
        }
        timestamps = [
            {"original": "8:15 PM", "iso": "2024-03-15T20:15:00", "confidence": 0.92},
        ]

        events = chunk_to_events(chunk, timestamps, 0, "001")

        assert len(events) == 1
        assert events[0].timestamp == "2024-03-15T20:15:00"
        assert events[0].description == "I heard something at 8:15 PM."
        assert events[0].speaker == "Witness A"

    def test_multiple_timestamps(self):
        """Should create multiple events for multiple timestamps."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "page_range": [2, 2],
            "text": "Between 8 PM and 10 PM I was home.",
            "speaker": None,
            "chunk_confidence": 0.85,
        }
        timestamps = [
            {"original": "8 PM", "iso": "2024-03-15T20:00:00", "confidence": 0.9},
            {"original": "10 PM", "iso": "2024-03-15T22:00:00", "confidence": 0.88},
        ]

        events = chunk_to_events(chunk, timestamps, 0, "001")

        assert len(events) == 2

    def test_skips_null_iso(self):
        """Should skip timestamps without valid ISO."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "page_range": [1, 1],
            "text": "Test.",
            "chunk_confidence": 0.9,
        }
        timestamps = [
            {"original": "around noon", "iso": None, "confidence": 0.3},
            {"original": "3 PM", "iso": "2024-03-15T15:00:00", "confidence": 0.9},
        ]

        events = chunk_to_events(chunk, timestamps, 0, "001")

        assert len(events) == 1
        assert events[0].timestamp == "2024-03-15T15:00:00"

    def test_confidence_calculation(self):
        """Event confidence should be min of chunk and timestamp."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "page_range": [1, 1],
            "text": "Test.",
            "chunk_confidence": 0.8,
        }
        timestamps = [
            {"original": "3 PM", "iso": "2024-03-15T15:00:00", "confidence": 0.95},
        ]

        events = chunk_to_events(chunk, timestamps, 0, "001")

        # min(0.8, 0.95) = 0.8
        assert events[0].confidence == 0.8

    def test_preserves_provenance(self):
        """Should preserve full provenance."""
        chunk = {
            "chunk_id": "CHUNK_XYZ",
            "document_id": "DOC_ABC",
            "page_range": [5, 5],
            "text": "Original text.",
            "speaker": "Speaker Name",
            "chunk_confidence": 0.9,
        }
        timestamps = [
            {"original": "2 PM", "iso": "2024-03-15T14:00:00", "confidence": 0.9},
        ]

        events = chunk_to_events(chunk, timestamps, 0, "001")

        assert events[0].chunk_id == "CHUNK_XYZ"
        assert events[0].document_id == "DOC_ABC"
        assert events[0].page_range == [5, 5]
        assert events[0].raw_timestamp == "2 PM"


class TestBuildEvents:
    """Tests for batch event building."""

    def test_builds_events_from_chunks(self):
        """Should build events from multiple chunks."""
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Event at 8 AM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Event at 9 AM.",
                "chunk_confidence": 0.85,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8 AM", "iso": "2024-03-15T08:00:00", "confidence": 0.9}],
            "C2": [{"original": "9 AM", "iso": "2024-03-15T09:00:00", "confidence": 0.88}],
        }

        events = build_events(chunks, timestamps_map, "001")

        assert len(events) == 2

    def test_skips_chunks_without_timestamps(self):
        """Should skip chunks without timestamps in map."""
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Event with time.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Event without time.",
                "chunk_confidence": 0.85,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8 AM", "iso": "2024-03-15T08:00:00", "confidence": 0.9}],
            # C2 not in map
        }

        events = build_events(chunks, timestamps_map, "001")

        assert len(events) == 1

    def test_empty_input(self):
        """Should handle empty input."""
        events = build_events([], {}, "001")
        assert events == []
