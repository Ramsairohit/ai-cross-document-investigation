"""
Unit tests for Stage 9: Timeline Reconstruction - Determinism

Comprehensive tests to verify that timeline building is deterministic.
The same input must always produce the identical timeline.
"""

import pytest

from stage_9_timeline.timeline_pipeline import TimelinePipeline


class TestDeterminism100Runs:
    """Tests for 100-run determinism verification."""

    @pytest.fixture
    def sample_data(self):
        """Sample test data for determinism testing."""
        chunks = [
            {
                "chunk_id": "C-001",
                "document_id": "W001-24-890-H",
                "page_range": [2, 2],
                "text": "I heard a loud crash around 8:15 PM.",
                "speaker": "Clara Higgins",
                "chunk_confidence": 0.93,
            },
            {
                "chunk_id": "C-002",
                "document_id": "W001-24-890-H",
                "page_range": [3, 3],
                "text": "At approximately 9:30 PM, I saw the suspect leave.",
                "speaker": "Marcus Vane",
                "chunk_confidence": 0.91,
            },
            {
                "chunk_id": "C-003",
                "document_id": "W002-24-890-H",
                "page_range": [1, 1],
                "text": "The incident occurred around 8:00 PM.",
                "speaker": "Officer Bennett",
                "chunk_confidence": 0.95,
            },
            {
                "chunk_id": "C-004",
                "document_id": "W002-24-890-H",
                "page_range": [2, 2],
                "text": "By midnight, all witnesses had been interviewed.",
                "speaker": "Officer Bennett",
                "chunk_confidence": 0.92,
            },
        ]
        timestamps_map = {
            "C-001": [{"original": "8:15 PM", "iso": "2024-03-15T20:15:00", "confidence": 0.92}],
            "C-002": [{"original": "9:30 PM", "iso": "2024-03-15T21:30:00", "confidence": 0.88}],
            "C-003": [{"original": "8:00 PM", "iso": "2024-03-15T20:00:00", "confidence": 0.95}],
            "C-004": [{"original": "midnight", "iso": "2024-03-16T00:00:00", "confidence": 0.90}],
        }
        return {"chunks": chunks, "timestamps_map": timestamps_map, "case_id": "24-890-H"}

    def test_100_run_determinism(self, sample_data):
        """
        Rebuild timeline 100 times and verify identical results.

        This is a CRITICAL test for forensic-grade requirements.
        """
        pipeline = TimelinePipeline()
        runs = 100

        results = []
        for _ in range(runs):
            result = pipeline.build_timeline(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                timestamps_map=sample_data["timestamps_map"],
            )
            results.append(result)

        # Compare all results to first
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.total_events == first.total_events, f"Event count differs at run {i}"
            assert result.total_gaps == first.total_gaps, f"Gap count differs at run {i}"
            assert result.total_conflicts == first.total_conflicts, (
                f"Conflict count differs at run {i}"
            )

    def test_event_order_identical_across_runs(self, sample_data):
        """Event order must be identical across all runs."""
        pipeline = TimelinePipeline()
        runs = 100

        event_orders = []
        for _ in range(runs):
            result = pipeline.build_timeline(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                timestamps_map=sample_data["timestamps_map"],
            )
            order = [e.event_id for e in result.events]
            event_orders.append(order)

        first_order = event_orders[0]
        for i, order in enumerate(event_orders[1:], 2):
            assert order == first_order, f"Event order differs at run {i}"

    def test_event_timestamps_identical_across_runs(self, sample_data):
        """Event timestamps must be identical across all runs."""
        pipeline = TimelinePipeline()
        runs = 100

        timestamp_lists = []
        for _ in range(runs):
            result = pipeline.build_timeline(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                timestamps_map=sample_data["timestamps_map"],
            )
            timestamps = [e.timestamp for e in result.events]
            timestamp_lists.append(timestamps)

        first_list = timestamp_lists[0]
        for i, ts_list in enumerate(timestamp_lists[1:], 2):
            assert ts_list == first_list, f"Timestamps differ at run {i}"

    def test_gaps_identical_across_runs(self, sample_data):
        """Gaps must be identical across all runs."""
        pipeline = TimelinePipeline()
        runs = 100

        gap_snapshots = []
        for _ in range(runs):
            result = pipeline.build_timeline(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                timestamps_map=sample_data["timestamps_map"],
            )
            snapshot = [(g.start, g.end, g.duration_minutes, g.severity.value) for g in result.gaps]
            gap_snapshots.append(snapshot)

        first_snapshot = gap_snapshots[0]
        for i, snapshot in enumerate(gap_snapshots[1:], 2):
            assert snapshot == first_snapshot, f"Gaps differ at run {i}"


class TestDeterminismEdgeCases:
    """Edge case tests for determinism."""

    def test_empty_input_determinism(self):
        """Empty input should produce identical empty results."""
        pipeline = TimelinePipeline()

        results = [pipeline.build_timeline("001", [], {}) for _ in range(100)]

        for result in results:
            assert result.total_events == 0
            assert result.total_gaps == 0
            assert result.total_conflicts == 0

    def test_single_event_determinism(self):
        """Single event should produce identical results."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Single event.",
                "chunk_confidence": 0.9,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "noon", "iso": "2024-03-15T12:00:00", "confidence": 0.9}],
        }

        results = [pipeline.build_timeline("001", chunks, timestamps_map) for _ in range(100)]

        first = results[0]
        for result in results[1:]:
            assert result.total_events == first.total_events
            assert result.events[0].event_id == first.events[0].event_id

    def test_equal_timestamps_stable_order(self):
        """Events with equal timestamps should preserve input order."""
        pipeline = TimelinePipeline()

        # Create chunks with same timestamp
        chunks = [
            {
                "chunk_id": f"C{i}",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": f"Event {i}.",
                "chunk_confidence": 0.9,
            }
            for i in range(5)
        ]
        timestamps_map = {
            f"C{i}": [{"original": "noon", "iso": "2024-03-15T12:00:00", "confidence": 0.9}]
            for i in range(5)
        }

        results = [pipeline.build_timeline("001", chunks, timestamps_map) for _ in range(100)]

        first_order = [e.chunk_id for e in results[0].events]
        for result in results[1:]:
            order = [e.chunk_id for e in result.events]
            assert order == first_order

    def test_verify_determinism_method_passes(self):
        """Built-in verify_determinism should return True."""
        pipeline = TimelinePipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Event 1 at 8 AM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Event 2 at 10 AM.",
                "chunk_confidence": 0.85,
            },
        ]
        timestamps_map = {
            "C1": [{"original": "8 AM", "iso": "2024-03-15T08:00:00", "confidence": 0.9}],
            "C2": [{"original": "10 AM", "iso": "2024-03-15T10:00:00", "confidence": 0.88}],
        }

        is_deterministic = pipeline.verify_determinism(
            case_id="001",
            chunks=chunks,
            timestamps_map=timestamps_map,
            runs=100,
        )

        assert is_deterministic is True
