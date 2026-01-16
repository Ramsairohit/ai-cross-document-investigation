"""
Unit tests for Stage 10: Contradiction Detection - Pipeline

Tests for the main pipeline API.
"""

import pytest

from stage_10_contradictions.models import (
    ContradictionSeverity,
    ContradictionStatus,
    ContradictionType,
)
from stage_10_contradictions.contradiction_pipeline import (
    ContradictionPipeline,
    detect_contradictions_sync,
    generate_contradiction_id,
)


class TestGenerateContradictionId:
    """Tests for contradiction ID generation."""

    def test_format(self):
        """Should follow CONT_{case_id}_{index} format."""
        cont_id = generate_contradiction_id("24-890-H", 0)
        assert cont_id == "CONT_24_890_H_0000"

    def test_deterministic(self):
        """Same input should produce same output."""
        ids = [generate_contradiction_id("24-890-H", 5) for _ in range(100)]
        assert all(id == ids[0] for id in ids)


class TestContradictionPipeline:
    """Tests for ContradictionPipeline class."""

    def test_detect_contradictions(self):
        """Should detect contradictions."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Marcus",
                "text": "I was not at the park at 9 PM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D2",
                "page_range": [2, 2],
                "speaker": "Julian",
                "text": "I saw Marcus at the park at 9 PM.",
                "chunk_confidence": 0.85,
            },
        ]

        result = pipeline.detect_contradictions("001", chunks)

        assert result.case_id == "001"
        assert result.chunks_analyzed == 2

    def test_no_cross_case(self):
        """Should not detect contradictions across cases."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "CASE_A",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Alice",
                "text": "I was at home.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "case_id": "CASE_B",
                "document_id": "D2",
                "page_range": [1, 1],
                "speaker": "Bob",
                "text": "Alice was at the scene.",
                "chunk_confidence": 0.85,
            },
        ]

        result = pipeline.detect_contradictions("CASE_A", chunks)

        # Only CASE_A chunks analyzed, so no cross-case comparison
        # C2 is from CASE_B so won't be paired with C1
        for cont in result.contradictions:
            assert cont.case_id == "CASE_A"

    def test_status_always_flagged(self):
        """All contradictions should have FLAGGED status."""
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

        result = pipeline.detect_contradictions("001", chunks)

        for cont in result.contradictions:
            assert cont.status == ContradictionStatus.FLAGGED

    def test_empty_input(self):
        """Should handle empty input."""
        pipeline = ContradictionPipeline()

        result = pipeline.detect_contradictions("001", [])

        assert result.total_contradictions == 0
        assert result.chunks_analyzed == 0


class TestNoResolution:
    """Tests to verify contradictions are not resolved."""

    def test_both_chunks_preserved(self):
        """Both conflicting chunks should be preserved."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Marcus",
                "text": "I was not at the scene at 9 PM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D2",
                "page_range": [2, 2],
                "speaker": "Julian",
                "text": "I saw Marcus at the scene at 9 PM.",
                "chunk_confidence": 0.85,
            },
        ]

        result = pipeline.detect_contradictions("001", chunks)

        # If contradiction detected, both chunks should be referenced
        for cont in result.contradictions:
            assert cont.chunk_a is not None
            assert cont.chunk_b is not None
            # Text should be preserved exactly
            assert cont.chunk_a.text in [c["text"] for c in chunks]
            assert cont.chunk_b.text in [c["text"] for c in chunks]

    def test_no_truth_assigned(self):
        """No contradiction should assign truth."""
        pipeline = ContradictionPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Witness A",
                "text": "The suspect was not there.",
                "chunk_confidence": 0.95,
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D2",
                "page_range": [1, 1],
                "speaker": "Witness B",
                "text": "I saw the suspect there.",
                "chunk_confidence": 0.7,
            },
        ]

        result = pipeline.detect_contradictions("001", chunks)

        # Even with different confidences, no truth is assigned
        for cont in result.contradictions:
            # Status is always FLAGGED, never RESOLVED
            assert cont.status == ContradictionStatus.FLAGGED
            # Explanation should not claim which is true
            assert "true" not in cont.explanation.lower()
            assert "false" not in cont.explanation.lower()
            assert "correct" not in cont.explanation.lower()


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input should produce same output."""
        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "speaker": "Marcus",
                "text": "I was not at home at 9 PM.",
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D2",
                "page_range": [2, 2],
                "speaker": "Julian",
                "text": "I saw Marcus at home at 9 PM.",
                "chunk_confidence": 0.85,
            },
        ]

        results = [detect_contradictions_sync("001", chunks) for _ in range(50)]

        first = results[0]
        for result in results[1:]:
            assert result.total_contradictions == first.total_contradictions

    def test_verify_determinism_method(self):
        """Pipeline verify_determinism should return True."""
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

        is_deterministic = pipeline.verify_determinism("001", chunks, runs=100)

        assert is_deterministic is True
