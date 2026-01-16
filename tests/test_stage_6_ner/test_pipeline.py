"""
Unit tests for Stage 6: NER - Pipeline

Tests for the NER pipeline orchestrator.
"""

import pytest

from stage_6_ner import (
    NERPipeline,
    NERResult,
    process_chunk_sync,
    process_chunks_sync,
)
from stage_6_ner.models import EntityType


class TestNERPipeline:
    """Tests for NERPipeline class."""

    def test_init(self):
        """Should initialize pipeline."""
        pipeline = NERPipeline()
        assert pipeline is not None

    def test_process_single_chunk(self):
        """Should process a single chunk."""
        pipeline = NERPipeline()
        chunk = {
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 1],
            "text": "I saw a gun at the scene.",
            "speaker": "WITNESS",
            "confidence": 0.9,
        }
        result = pipeline.process_chunk(chunk)

        assert isinstance(result, NERResult)
        assert result.chunk_id == "CHUNK_001"

    def test_process_multiple_chunks(self):
        """Should process multiple chunks independently."""
        pipeline = NERPipeline()
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [1, 1],
                "text": "First chunk with knife.",
                "speaker": None,
                "confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [2, 2],
                "text": "Second chunk at 420 Main Street.",
                "speaker": None,
                "confidence": 0.85,
            },
        ]
        results = pipeline.process_chunks(chunks)

        assert len(results) == 2
        assert results[0].chunk_id == "C1"
        assert results[1].chunk_id == "C2"

    def test_chunks_processed_independently(self):
        """Chunks should be processed independently (no cross-chunk analysis)."""
        pipeline = NERPipeline()

        # Same text in different chunks should produce same entities
        chunk1 = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Found a gun.",
            "speaker": None,
            "confidence": 1.0,
        }
        chunk2 = {
            "chunk_id": "C2",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [2, 2],
            "text": "Found a gun.",
            "speaker": None,
            "confidence": 1.0,
        }

        result1 = pipeline.process_chunk(chunk1)
        result2 = pipeline.process_chunk(chunk2)

        # Should have same number of entities (same text)
        assert result1.entity_count == result2.entity_count

        # But different chunk IDs
        for e in result1.entities:
            assert e.chunk_id == "C1"
        for e in result2.entities:
            assert e.chunk_id == "C2"


class TestProcessChunkSync:
    """Tests for process_chunk_sync function."""

    def test_basic_processing(self):
        """Should process chunk synchronously."""
        chunk = {
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 1],
            "text": "Blood evidence found.",
            "speaker": None,
            "confidence": 0.95,
        }
        result = process_chunk_sync(chunk)

        assert isinstance(result, NERResult)
        assert result.chunk_id == "CHUNK_001"

    def test_returns_ner_result(self):
        """Should return NERResult type."""
        result = process_chunk_sync(
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [1, 1],
                "text": "Test.",
                "speaker": None,
                "confidence": 1.0,
            }
        )
        assert isinstance(result, NERResult)


class TestProcessChunksSync:
    """Tests for process_chunks_sync function."""

    def test_empty_list(self):
        """Should handle empty chunk list."""
        results = process_chunks_sync([])
        assert results == []

    def test_multiple_chunks(self):
        """Should process multiple chunks."""
        chunks = [
            {
                "chunk_id": f"C{i}",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [i, i],
                "text": f"Text {i}.",
                "speaker": None,
                "confidence": 1.0,
            }
            for i in range(5)
        ]
        results = process_chunks_sync(chunks)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.chunk_id == f"C{i}"


class TestAsyncFunctions:
    """Tests for async processing functions."""

    @pytest.mark.asyncio
    async def test_process_chunk_async(self):
        """Should process chunk asynchronously."""
        from stage_6_ner import process_chunk_async

        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Async test with gun.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = await process_chunk_async(chunk)

        assert isinstance(result, NERResult)
        assert result.chunk_id == "C1"

    @pytest.mark.asyncio
    async def test_process_chunks_async(self):
        """Should process multiple chunks asynchronously."""
        from stage_6_ner import process_chunks_async

        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [1, 1],
                "text": "First async.",
                "speaker": None,
                "confidence": 1.0,
            },
            {
                "chunk_id": "C2",
                "document_id": "D1",
                "case_id": "CASE1",
                "page_range": [2, 2],
                "text": "Second async with knife.",
                "speaker": None,
                "confidence": 1.0,
            },
        ]
        results = await process_chunks_async(chunks)

        assert len(results) == 2


class TestNoInference:
    """Tests to verify no inference occurs."""

    def test_no_cross_chunk_entity_merging(self):
        """Should not merge entities across chunks."""
        pipeline = NERPipeline()

        # Same person mentioned in different chunks
        chunk1 = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "John Smith was there.",
            "speaker": None,
            "confidence": 1.0,
        }
        chunk2 = {
            "chunk_id": "C2",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [2, 2],
            "text": "John Smith left.",
            "speaker": None,
            "confidence": 1.0,
        }

        result1 = pipeline.process_chunk(chunk1)
        result2 = pipeline.process_chunk(chunk2)

        # Each chunk should have its own entity
        # They should NOT be merged or linked
        if result1.entities and result2.entities:
            # Entity IDs should be different
            ids1 = {e.entity_id for e in result1.entities}
            ids2 = {e.entity_id for e in result2.entities}
            assert ids1.isdisjoint(ids2)

    def test_no_role_inference_from_text(self):
        """Should not infer role from text content."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "The witness John testified that the suspect Bob ran.",
            "speaker": None,  # No metadata for role
            "confidence": 1.0,
        }
        result = process_chunk_sync(chunk)

        # Even though text says "witness" and "suspect", no role should be assigned
        # because speaker metadata is None
        person_entities = [e for e in result.entities if e.entity_type == EntityType.PERSON]
        for entity in person_entities:
            assert entity.role is None
