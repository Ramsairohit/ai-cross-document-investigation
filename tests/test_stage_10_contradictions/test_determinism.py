"""
Unit tests for Stage 10: Contradiction Detection - Determinism

Comprehensive tests to verify that contradiction detection is deterministic.
"""

import pytest

from stage_10_contradictions.contradiction_pipeline import ContradictionPipeline


class TestDeterminism100Runs:
    """Tests for 100-run determinism verification."""

    @pytest.fixture
    def sample_data(self):
        """Sample test data for determinism testing."""
        chunks = [
            {
                "chunk_id": "C-001",
                "case_id": "24-890-H",
                "document_id": "W001",
                "page_range": [2, 2],
                "speaker": "Marcus Vane",
                "text": "I was not at the scene at 9 PM. I was at home.",
                "chunk_confidence": 0.93,
            },
            {
                "chunk_id": "C-002",
                "case_id": "24-890-H",
                "document_id": "W002",
                "page_range": [3, 3],
                "speaker": "Julian Thorne",
                "text": "I saw Marcus at the scene at 9 PM.",
                "chunk_confidence": 0.91,
            },
            {
                "chunk_id": "C-003",
                "case_id": "24-890-H",
                "document_id": "W003",
                "page_range": [1, 1],
                "speaker": "Clara Higgins",
                "text": "Marcus was definitely not at home that evening.",
                "chunk_confidence": 0.88,
            },
        ]
        return {"chunks": chunks, "case_id": "24-890-H"}

    def test_100_run_determinism(self, sample_data):
        """
        Detect contradictions 100 times and verify identical results.

        This is a CRITICAL test for forensic-grade requirements.
        """
        pipeline = ContradictionPipeline()
        runs = 100

        results = []
        for _ in range(runs):
            result = pipeline.detect_contradictions(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
            )
            results.append(result)

        # Compare all results to first
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.total_contradictions == first.total_contradictions, (
                f"Count differs at run {i}"
            )
            assert result.pairs_compared == first.pairs_compared, f"Pairs differs at run {i}"

    def test_contradiction_ids_identical(self, sample_data):
        """Contradiction IDs must be identical across runs."""
        pipeline = ContradictionPipeline()
        runs = 100

        id_lists = []
        for _ in range(runs):
            result = pipeline.detect_contradictions(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
            )
            ids = [c.contradiction_id for c in result.contradictions]
            id_lists.append(tuple(ids))

        first_ids = id_lists[0]
        for i, ids in enumerate(id_lists[1:], 2):
            assert ids == first_ids, f"IDs differ at run {i}"

    def test_types_identical(self, sample_data):
        """Contradiction types must be identical across runs."""
        pipeline = ContradictionPipeline()
        runs = 100

        type_lists = []
        for _ in range(runs):
            result = pipeline.detect_contradictions(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
            )
            types = [c.type.value for c in result.contradictions]
            type_lists.append(tuple(types))

        first_types = type_lists[0]
        for i, types in enumerate(type_lists[1:], 2):
            assert types == first_types, f"Types differ at run {i}"


class TestDeterminismEdgeCases:
    """Edge case tests for determinism."""

    def test_empty_input_determinism(self):
        """Empty input should produce identical empty results."""
        pipeline = ContradictionPipeline()

        results = [pipeline.detect_contradictions("001", []) for _ in range(100)]

        for result in results:
            assert result.total_contradictions == 0

    def test_single_chunk_determinism(self):
        """Single chunk should produce no contradictions."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Alice",
                "text": "I was at home.",
                "chunk_confidence": 0.9,
            },
        ]

        results = [pipeline.detect_contradictions("001", chunks) for _ in range(100)]

        for result in results:
            assert result.total_contradictions == 0

    def test_verify_determinism_passes(self):
        """Built-in verify_determinism should return True."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Marcus",
                "text": "I was not there.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D2",
                "page_range": [2, 2],
                "speaker": "Julian",
                "text": "I saw Marcus there.",
                "chunk_confidence": 0.85,
            },
        ]

        is_deterministic = pipeline.verify_determinism("001", chunks, runs=100)

        assert is_deterministic is True
