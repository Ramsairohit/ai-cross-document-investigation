"""
Unit tests for Stage 5: Logical Chunking - Confidence Aggregation

Tests for conservative confidence scoring.
"""

import pytest

from stage_5_chunking.confidence import (
    aggregate_confidence,
    compute_chunk_confidence,
    validate_confidence,
)


class TestAggregateConfidence:
    """Tests for confidence aggregation."""

    def test_empty_list(self):
        """Empty list should return 1.0."""
        assert aggregate_confidence([]) == 1.0

    def test_single_value(self):
        """Single value should be returned."""
        assert aggregate_confidence([0.9]) == 0.9

    def test_minimum_is_returned(self):
        """Should return minimum value."""
        scores = [0.9, 0.85, 0.95, 0.88]
        assert aggregate_confidence(scores) == 0.85

    def test_all_same_value(self):
        """All same values should return that value."""
        scores = [0.8, 0.8, 0.8]
        assert aggregate_confidence(scores) == 0.8

    def test_includes_zero(self):
        """Zero should be returned if present."""
        scores = [0.9, 0.0, 0.8]
        assert aggregate_confidence(scores) == 0.0


class TestValidateConfidence:
    """Tests for confidence validation."""

    def test_valid_range(self):
        """Values in range should pass through."""
        assert validate_confidence(0.5) == 0.5
        assert validate_confidence(0.0) == 0.0
        assert validate_confidence(1.0) == 1.0

    def test_clamp_high(self):
        """Values > 1.0 should be clamped."""
        assert validate_confidence(1.5) == 1.0
        assert validate_confidence(100.0) == 1.0

    def test_clamp_low(self):
        """Values < 0.0 should be clamped."""
        assert validate_confidence(-0.5) == 0.0
        assert validate_confidence(-100.0) == 0.0


class TestComputeChunkConfidence:
    """Tests for chunk confidence computation."""

    def test_empty_blocks(self):
        """No blocks should return 1.0."""
        assert compute_chunk_confidence([]) == 1.0

    def test_single_block(self):
        """Single block confidence should pass through."""
        assert compute_chunk_confidence([0.9]) == 0.9

    def test_multiple_blocks(self):
        """Should return minimum."""
        confidences = [0.95, 0.88, 0.92]
        assert compute_chunk_confidence(confidences) == 0.88

    def test_with_invalid_values(self):
        """Should handle invalid values by clamping."""
        confidences = [0.9, 1.5, -0.1]
        result = compute_chunk_confidence(confidences)
        # After clamping: [0.9, 1.0, 0.0]
        # Minimum is 0.0
        assert result == 0.0

    def test_conservative_approach(self):
        """Should always be conservative (weakest link)."""
        # This tests the core requirement: court-defensible confidence
        high_confidence = [0.99, 0.98, 0.97]
        assert compute_chunk_confidence(high_confidence) == 0.97

        mixed_confidence = [0.99, 0.50, 0.97]
        assert compute_chunk_confidence(mixed_confidence) == 0.50
