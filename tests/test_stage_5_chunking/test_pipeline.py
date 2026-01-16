"""
Unit tests for Stage 5: Logical Chunking - Pipeline

Tests for the chunking pipeline orchestrator.
"""

import pytest

from stage_5_chunking.chunking_pipeline import (
    ChunkingPipeline,
    process_cleaning_result_sync,
    process_document_async,
    process_document_sync,
)
from stage_5_chunking.models import BlockInput, ChunkingConfig, ChunkingResult


def make_cleaning_result() -> dict:
    """Create a mock Stage 4 CleaningResult."""
    return {
        "document_id": "W001-24-890-H",
        "case_id": "24-890-H",
        "source_file": "witness_statement.pdf",
        "cleaned_blocks": [
            {
                "block_id": "b1",
                "page": 1,
                "clean_text": "DET. SMITH: Where were you at 8:15 PM?",
                "speaker": "DET. SMITH",
                "confidence": 0.93,
            },
            {
                "block_id": "b2",
                "page": 1,
                "clean_text": "WITNESS: I was at home.",
                "speaker": "WITNESS",
                "confidence": 0.91,
            },
        ],
    }


class TestChunkingPipeline:
    """Tests for ChunkingPipeline class."""

    def test_init_default_config(self):
        """Should use default config."""
        pipeline = ChunkingPipeline()
        assert pipeline.config.min_tokens == 300
        assert pipeline.config.max_tokens == 1000

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = ChunkingConfig(min_tokens=100, max_tokens=500)
        pipeline = ChunkingPipeline(config=config)
        assert pipeline.config.min_tokens == 100
        assert pipeline.config.max_tokens == 500

    def test_process_document(self):
        """Should process document and return ChunkingResult."""
        pipeline = ChunkingPipeline()
        blocks = [
            BlockInput(
                block_id="b1",
                page=1,
                clean_text="Test content.",
                confidence=0.9,
            )
        ]
        result = pipeline.process_document(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
        )
        assert isinstance(result, ChunkingResult)
        assert result.document_id == "DOC1"
        assert result.case_id == "CASE1"
        assert result.total_chunks == 1
        assert result.total_blocks_processed == 1

    def test_process_cleaning_result(self):
        """Should process CleaningResult dict."""
        pipeline = ChunkingPipeline()
        cleaning_result = make_cleaning_result()
        result = pipeline.process_cleaning_result(cleaning_result)

        assert result.document_id == "W001-24-890-H"
        assert result.case_id == "24-890-H"
        assert result.total_blocks_processed == 2


class TestProcessDocumentSync:
    """Tests for synchronous document processing."""

    def test_basic_processing(self):
        """Should process document synchronously."""
        blocks = [
            {"block_id": "b1", "page": 1, "clean_text": "Test."},
        ]
        result = process_document_sync(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
        )
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 1

    def test_with_custom_config(self):
        """Should accept custom config."""
        blocks = [{"block_id": "b1", "page": 1, "clean_text": "Test."}]
        config = ChunkingConfig(max_tokens=500)
        result = process_document_sync(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
            config=config,
        )
        assert result is not None


class TestProcessDocumentAsync:
    """Tests for asynchronous document processing."""

    @pytest.mark.asyncio
    async def test_async_processing(self):
        """Should process document asynchronously."""
        blocks = [
            {"block_id": "b1", "page": 1, "clean_text": "Test content."},
        ]
        result = await process_document_async(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
        )
        assert isinstance(result, ChunkingResult)
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_async_with_custom_config(self):
        """Should accept custom config."""
        blocks = [{"block_id": "b1", "page": 1, "clean_text": "Test."}]
        config = ChunkingConfig(max_tokens=500)
        result = await process_document_async(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
            config=config,
        )
        assert result is not None


class TestProcessCleaningResultSync:
    """Tests for Stage 4 result processing."""

    def test_process_cleaning_result(self):
        """Should process CleaningResult dict."""
        cleaning_result = make_cleaning_result()
        result = process_cleaning_result_sync(cleaning_result)

        assert result.document_id == "W001-24-890-H"
        assert result.case_id == "24-890-H"
        assert result.source_file == "witness_statement.pdf"
        assert len(result.chunks) >= 1

    def test_empty_blocks(self):
        """Should handle empty blocks."""
        cleaning_result = {
            "document_id": "DOC1",
            "case_id": "CASE1",
            "source_file": "empty.pdf",
            "cleaned_blocks": [],
        }
        result = process_cleaning_result_sync(cleaning_result)
        assert result.total_chunks == 0


class TestStage6Compatibility:
    """Tests verifying Stage 6 (NER) compatibility."""

    def test_chunk_has_required_fields_for_ner(self):
        """Chunks should have all fields required by Stage 6 ChunkInput."""
        blocks = [
            BlockInput(
                block_id="b1",
                page=1,
                clean_text="I saw Marcus Vane at 420 Harrow Lane.",
                speaker="WITNESS",
                confidence=0.91,
            )
        ]
        result = process_document_sync(
            document_id="DOC1",
            case_id="24-890-H",
            source_file="statement.pdf",
            blocks=blocks,
        )
        chunk = result.chunks[0]

        # Stage 6 ChunkInput required fields
        assert hasattr(chunk, "chunk_id")
        assert hasattr(chunk, "document_id")
        assert hasattr(chunk, "case_id")
        assert hasattr(chunk, "page_range")
        assert hasattr(chunk, "speaker")
        assert hasattr(chunk, "text")

        # Verify field values
        assert chunk.chunk_id is not None
        assert chunk.case_id == "24-890-H"
        assert chunk.page_range == [1, 1]
        assert chunk.speaker == "WITNESS"
        assert "Marcus Vane" in chunk.text

    def test_chunk_can_be_converted_to_dict(self):
        """Chunk should be convertible to dict for Stage 6."""
        blocks = [BlockInput(block_id="b1", page=1, clean_text="Test.", confidence=0.9)]
        result = process_document_sync(
            document_id="DOC1",
            case_id="CASE1",
            source_file="test.pdf",
            blocks=blocks,
        )
        chunk_dict = result.chunks[0].model_dump()

        # Verify structure matches Stage 6 expectations
        assert "chunk_id" in chunk_dict
        assert "text" in chunk_dict
        assert "page_range" in chunk_dict
        assert isinstance(chunk_dict["page_range"], list)
        assert len(chunk_dict["page_range"]) == 2
