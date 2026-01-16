"""
Unit tests for Stage 10: Contradiction Detection - Data Models

Tests for contradiction schemas and enums.
"""

import pytest
from pydantic import ValidationError

from stage_10_contradictions.models import (
    ChunkReference,
    Contradiction,
    ContradictionConfig,
    ContradictionResult,
    ContradictionSeverity,
    ContradictionStatus,
    ContradictionType,
)


class TestContradictionType:
    """Tests for ContradictionType enum."""

    def test_type_values(self):
        """Should have exactly 4 types."""
        assert len(ContradictionType) == 4
        assert ContradictionType.TIME_CONFLICT.value == "TIME_CONFLICT"
        assert ContradictionType.LOCATION_CONFLICT.value == "LOCATION_CONFLICT"
        assert ContradictionType.STATEMENT_VS_EVIDENCE.value == "STATEMENT_VS_EVIDENCE"
        assert ContradictionType.DENIAL_VS_ASSERTION.value == "DENIAL_VS_ASSERTION"


class TestContradictionSeverity:
    """Tests for ContradictionSeverity enum."""

    def test_severity_values(self):
        """Should have 4 severity levels."""
        assert len(ContradictionSeverity) == 4
        assert ContradictionSeverity.LOW.value == "LOW"
        assert ContradictionSeverity.MEDIUM.value == "MEDIUM"
        assert ContradictionSeverity.HIGH.value == "HIGH"
        assert ContradictionSeverity.CRITICAL.value == "CRITICAL"


class TestContradictionStatus:
    """Tests for ContradictionStatus enum."""

    def test_only_flagged(self):
        """Should only have FLAGGED status."""
        assert len(ContradictionStatus) == 1
        assert ContradictionStatus.FLAGGED.value == "FLAGGED"


class TestChunkReference:
    """Tests for ChunkReference model."""

    def test_valid_reference(self):
        """Should create valid chunk reference."""
        ref = ChunkReference(
            chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            speaker="Marcus Vane",
            text="I was at home at 9 PM.",
        )
        assert ref.chunk_id == "CHUNK_001"
        assert ref.speaker == "Marcus Vane"

    def test_optional_speaker(self):
        """Speaker should be optional."""
        ref = ChunkReference(
            chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[1, 1],
            text="Some text.",
        )
        assert ref.speaker is None


class TestContradiction:
    """Tests for Contradiction model."""

    def test_valid_contradiction(self):
        """Should create valid contradiction."""
        contradiction = Contradiction(
            contradiction_id="CONT_001",
            case_id="24-890-H",
            type=ContradictionType.LOCATION_CONFLICT,
            chunk_a=ChunkReference(
                chunk_id="CHUNK_001",
                document_id="DOC123",
                page_range=[2, 2],
                text="I was at home at 9 PM.",
            ),
            chunk_b=ChunkReference(
                chunk_id="CHUNK_007",
                document_id="DOC456",
                page_range=[3, 3],
                text="I saw Marcus at the scene at 9 PM.",
            ),
            confidence=0.91,
            severity=ContradictionSeverity.CRITICAL,
            explanation="Different locations claimed.",
        )
        assert contradiction.contradiction_id == "CONT_001"
        assert contradiction.status == ContradictionStatus.FLAGGED

    def test_status_always_flagged(self):
        """Status should default to FLAGGED."""
        contradiction = Contradiction(
            contradiction_id="CONT_001",
            case_id="001",
            type=ContradictionType.TIME_CONFLICT,
            chunk_a=ChunkReference(chunk_id="C1", document_id="D1", page_range=[1, 1], text="A"),
            chunk_b=ChunkReference(chunk_id="C2", document_id="D1", page_range=[1, 1], text="B"),
            confidence=0.9,
            severity=ContradictionSeverity.HIGH,
            explanation="Conflict.",
        )
        assert contradiction.status == ContradictionStatus.FLAGGED


class TestContradictionResult:
    """Tests for ContradictionResult model."""

    def test_valid_result(self):
        """Should create valid result."""
        result = ContradictionResult(
            case_id="24-890-H",
            contradictions=[],
            total_contradictions=0,
            chunks_analyzed=10,
            pairs_compared=5,
        )
        assert result.case_id == "24-890-H"
        assert result.total_contradictions == 0


class TestContradictionConfig:
    """Tests for ContradictionConfig model."""

    def test_default_values(self):
        """Should have correct defaults."""
        config = ContradictionConfig()
        assert config.use_nli is False
        assert config.min_confidence == 0.5
        assert config.require_entity_overlap is True
