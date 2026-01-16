"""
Unit tests for Stage 7: Vector Embeddings - Data Models

Tests for VectorRecord, EmbeddingResult, and EmbeddingConfig.
"""

import pytest

from stage_7_embeddings.models import EmbeddingConfig, EmbeddingResult, VectorRecord


class TestVectorRecord:
    """Tests for VectorRecord model."""

    def test_valid_record(self):
        """Should create valid vector record with all required fields."""
        record = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            speaker="Clara Higgins",
            confidence=0.91,
        )

        assert record.chunk_id == "C-001"
        assert record.vector_id == 0
        assert record.case_id == "24-890-H"
        assert record.document_id == "W001-24-890-H"
        assert record.page_range == [1, 1]
        assert record.speaker == "Clara Higgins"
        assert record.confidence == 0.91

    def test_speaker_optional(self):
        """Speaker should be optional."""
        record = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            confidence=0.91,
        )

        assert record.speaker is None

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        # Valid bounds
        record_low = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            confidence=0.0,
        )
        assert record_low.confidence == 0.0

        record_high = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            confidence=1.0,
        )
        assert record_high.confidence == 1.0

        # Invalid bounds
        with pytest.raises(ValueError):
            VectorRecord(
                chunk_id="C-001",
                vector_id=0,
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                confidence=1.5,
            )

    def test_page_range_validation(self):
        """Page range should have exactly 2 elements."""
        # Valid
        record = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 3],
            confidence=0.91,
        )
        assert record.page_range == [1, 3]

        # Invalid - too few elements
        with pytest.raises(ValueError):
            VectorRecord(
                chunk_id="C-001",
                vector_id=0,
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1],
                confidence=0.91,
            )

    def test_vector_id_non_negative(self):
        """Vector ID should be non-negative."""
        with pytest.raises(ValueError):
            VectorRecord(
                chunk_id="C-001",
                vector_id=-1,
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                confidence=0.91,
            )

    def test_json_serializable(self):
        """Record should be JSON serializable."""
        record = VectorRecord(
            chunk_id="C-001",
            vector_id=0,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            speaker="Clara Higgins",
            confidence=0.91,
        )

        json_data = record.model_dump_json()
        assert "C-001" in json_data
        assert "24-890-H" in json_data


class TestEmbeddingResult:
    """Tests for EmbeddingResult model."""

    def test_valid_result(self):
        """Should create valid embedding result."""
        result = EmbeddingResult(
            chunk_id="C-001",
            vector_id=0,
            embedding_dimension=384,
            success=True,
        )

        assert result.chunk_id == "C-001"
        assert result.vector_id == 0
        assert result.embedding_dimension == 384
        assert result.success is True

    def test_default_values(self):
        """Should use default values for optional fields."""
        result = EmbeddingResult(
            chunk_id="C-001",
            vector_id=0,
        )

        assert result.embedding_dimension == 384
        assert result.success is True

    def test_failed_result(self):
        """Should allow failed result."""
        result = EmbeddingResult(
            chunk_id="C-001",
            vector_id=0,
            success=False,
        )

        assert result.success is False


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig model."""

    def test_default_values(self):
        """Should have correct default values."""
        config = EmbeddingConfig()

        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.embedding_dimension == 384
        assert config.index_type == "Flat"
        assert config.normalize_embeddings is True

    def test_custom_values(self):
        """Should allow custom values."""
        config = EmbeddingConfig(
            model_name="custom-model",
            embedding_dimension=768,
            index_type="IVF",
            normalize_embeddings=False,
        )

        assert config.model_name == "custom-model"
        assert config.embedding_dimension == 768
        assert config.index_type == "IVF"
        assert config.normalize_embeddings is False
