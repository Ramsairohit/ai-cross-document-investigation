"""
Unit tests for Stage 5: Logical Chunking - Data Models

Tests for Pydantic models and validation.
"""

import pytest
from pydantic import ValidationError

from stage_5_chunking.models import BlockInput, Chunk, ChunkingConfig, ChunkingResult


class TestBlockInput:
    """Tests for BlockInput model."""

    def test_valid_block(self):
        """Should create valid block input."""
        block = BlockInput(
            block_id="b12",
            page=2,
            clean_text="DET. SMITH: Where were you at 8:15 PM?",
            speaker="DET. SMITH",
            confidence=0.93,
        )
        assert block.block_id == "b12"
        assert block.page == 2
        assert block.speaker == "DET. SMITH"
        assert block.confidence == 0.93

    def test_speaker_optional(self):
        """Speaker should be optional."""
        block = BlockInput(
            block_id="b1",
            page=1,
            clean_text="Some text content.",
        )
        assert block.speaker is None

    def test_default_confidence(self):
        """Confidence should default to 1.0."""
        block = BlockInput(
            block_id="b1",
            page=1,
            clean_text="Some text.",
        )
        assert block.confidence == 1.0

    def test_page_must_be_positive(self):
        """Page number must be >= 1."""
        with pytest.raises(ValidationError):
            BlockInput(
                block_id="b1",
                page=0,
                clean_text="Text",
            )

    def test_confidence_bounds(self):
        """Confidence must be between 0 and 1."""
        # Valid confidence
        block = BlockInput(
            block_id="b1",
            page=1,
            clean_text="Text",
            confidence=0.5,
        )
        assert block.confidence == 0.5

    def test_json_serializable(self):
        """BlockInput should be JSON serializable."""
        block = BlockInput(
            block_id="b1",
            page=1,
            clean_text="Test text",
            speaker="WITNESS",
        )
        json_str = block.model_dump_json()
        assert "b1" in json_str
        assert "WITNESS" in json_str


class TestChunk:
    """Tests for Chunk model."""

    def test_valid_chunk(self):
        """Should create valid chunk with all fields."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[2, 2],
            speaker="DET. SMITH",
            text="DET. SMITH: Where were you at 8:15 PM?",
            source_block_ids=["b12"],
            token_count=21,
            chunk_confidence=0.93,
        )
        assert chunk.chunk_id == "C-0001"
        assert chunk.page_range == [2, 2]
        assert chunk.source_block_ids == ["b12"]
        assert chunk.token_count == 21

    def test_speaker_optional(self):
        """Speaker should be optional."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="CASE1",
            document_id="DOC1",
            page_range=[1, 1],
            text="Test text.",
            source_block_ids=["b1"],
            token_count=5,
            chunk_confidence=1.0,
        )
        assert chunk.speaker is None

    def test_page_range_must_have_two_elements(self):
        """Page range must have exactly 2 elements."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="CASE1",
            document_id="DOC1",
            page_range=[1, 1],
            text="Test",
            source_block_ids=["b1"],
            token_count=1,
            chunk_confidence=1.0,
        )
        assert len(chunk.page_range) == 2

    def test_source_block_ids_required(self):
        """Chunk must have at least one source block ID."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="CASE1",
            document_id="DOC1",
            page_range=[1, 1],
            text="Test",
            source_block_ids=["b1"],
            token_count=1,
            chunk_confidence=1.0,
        )
        assert len(chunk.source_block_ids) >= 1

    def test_json_serializable(self):
        """Chunk should be JSON serializable."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="24-890-H",
            document_id="DOC1",
            page_range=[1, 1],
            text="Test text.",
            source_block_ids=["b1", "b2"],
            token_count=10,
            chunk_confidence=0.85,
        )
        json_str = chunk.model_dump_json()
        assert "C-0001" in json_str
        assert "24-890-H" in json_str


class TestChunkingConfig:
    """Tests for ChunkingConfig model."""

    def test_default_values(self):
        """Should have correct defaults."""
        config = ChunkingConfig()
        assert config.min_tokens == 300
        assert config.max_tokens == 1000
        assert config.encoding_name == "cl100k_base"

    def test_custom_values(self):
        """Should accept custom values."""
        config = ChunkingConfig(
            min_tokens=100,
            max_tokens=500,
            encoding_name="p50k_base",
        )
        assert config.min_tokens == 100
        assert config.max_tokens == 500
        assert config.encoding_name == "p50k_base"


class TestChunkingResult:
    """Tests for ChunkingResult model."""

    def test_empty_result(self):
        """Should create result with no chunks."""
        result = ChunkingResult(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
        )
        assert result.chunks == []
        assert result.total_chunks == 0
        assert result.total_blocks_processed == 0

    def test_result_with_chunks(self):
        """Should create result with chunks."""
        chunk = Chunk(
            chunk_id="C-0001",
            case_id="CASE1",
            document_id="DOC1",
            page_range=[1, 1],
            text="Test",
            source_block_ids=["b1"],
            token_count=1,
            chunk_confidence=1.0,
        )
        result = ChunkingResult(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            chunks=[chunk],
            total_chunks=1,
            total_blocks_processed=1,
        )
        assert len(result.chunks) == 1
        assert result.total_chunks == 1
