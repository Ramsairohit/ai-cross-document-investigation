"""
Unit tests for Stage 7: Vector Embeddings - Pipeline

Tests for the embedding pipeline orchestrator.
"""

import tempfile
from pathlib import Path

import pytest

from stage_6_ner.models import ChunkInput
from stage_7_embeddings import EmbeddingPipeline, EmbeddingResult, VectorRecord
from stage_7_embeddings.embedding_model import EmbeddingModelLoader


class TestEmbeddingPipeline:
    """Tests for EmbeddingPipeline class."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_init(self):
        """Should initialize pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))

            assert pipeline.get_vector_count() == 0

    def test_process_single_chunk(self):
        """Should process a single chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))
            chunk = ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="I heard a loud crash around 8:15 PM.",
                speaker="Clara Higgins",
                confidence=0.91,
            )

            result = pipeline.process_chunk(chunk)

            assert isinstance(result, EmbeddingResult)
            assert result.chunk_id == "C-001"
            assert result.vector_id == 0
            assert result.embedding_dimension == 384
            assert result.success is True
            assert pipeline.get_vector_count() == 1

    def test_process_multiple_chunks(self):
        """Should process multiple chunks independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))
            chunks = [
                ChunkInput(
                    chunk_id="C-001",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[1, 1],
                    text="First witness statement.",
                    confidence=0.91,
                ),
                ChunkInput(
                    chunk_id="C-002",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[1, 1],
                    text="Second witness statement.",
                    confidence=0.88,
                ),
            ]

            results = pipeline.process_chunks(chunks)

            assert len(results) == 2
            assert results[0].chunk_id == "C-001"
            assert results[0].vector_id == 0
            assert results[1].chunk_id == "C-002"
            assert results[1].vector_id == 1
            assert pipeline.get_vector_count() == 2

    def test_process_dict_input(self):
        """Should accept dict input."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))
            chunk = {
                "chunk_id": "C-001",
                "case_id": "24-890-H",
                "document_id": "W001-24-890-H",
                "page_range": [1, 1],
                "text": "I heard a loud crash.",
                "confidence": 0.91,
            }

            result = pipeline.process_chunk(chunk)

            assert result.success is True

    def test_metadata_correctly_linked(self):
        """Metadata should be correctly linked to vector positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))
            chunks = [
                ChunkInput(
                    chunk_id="C-001",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[1, 1],
                    text="First chunk.",
                    speaker="Alice",
                    confidence=0.91,
                ),
                ChunkInput(
                    chunk_id="C-002",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[2, 2],
                    text="Second chunk.",
                    speaker="Bob",
                    confidence=0.85,
                ),
            ]

            pipeline.process_chunks(chunks)

            # Check metadata linkage
            record1 = pipeline.store.get_metadata(0)
            record2 = pipeline.store.get_metadata(1)

            assert record1 is not None
            assert record1.chunk_id == "C-001"
            assert record1.speaker == "Alice"
            assert record1.vector_id == 0

            assert record2 is not None
            assert record2.chunk_id == "C-002"
            assert record2.speaker == "Bob"
            assert record2.vector_id == 1


class TestPipelinePersistence:
    """Tests for pipeline save/load functionality."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_save_and_load(self):
        """Should save and load pipeline state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)

            # Create and populate pipeline
            pipeline = EmbeddingPipeline(storage_dir)
            chunk = ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="Test chunk.",
                confidence=0.91,
            )
            pipeline.process_chunk(chunk)
            pipeline.save()

            # Load into new pipeline
            new_pipeline = EmbeddingPipeline(storage_dir)
            new_pipeline.load()

            assert new_pipeline.get_vector_count() == 1
            record = new_pipeline.store.get_metadata(0)
            assert record is not None
            assert record.chunk_id == "C-001"

    def test_metadata_persisted_as_json(self):
        """Metadata should be persisted as JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)
            pipeline = EmbeddingPipeline(storage_dir)

            chunk = ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="Test chunk.",
                confidence=0.91,
            )
            pipeline.process_chunk(chunk)
            pipeline.save()

            # Check JSON file exists
            metadata_path = storage_dir / "metadata.json"
            assert metadata_path.exists()

            # Check FAISS index exists
            index_path = storage_dir / "faiss.index"
            assert index_path.exists()


class TestOneVectorPerChunk:
    """Tests to verify exactly one vector per chunk."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_one_vector_per_chunk(self):
        """Should create exactly one vector per chunk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))

            for i in range(5):
                chunk = ChunkInput(
                    chunk_id=f"C-{i:03d}",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[1, 1],
                    text=f"Chunk number {i}.",
                    confidence=0.91,
                )
                pipeline.process_chunk(chunk)

            assert pipeline.get_vector_count() == 5
            assert len(pipeline.store.metadata) == 5

    def test_no_chunk_merging(self):
        """Should not merge similar chunks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = EmbeddingPipeline(Path(tmpdir))

            # Same text, different chunk IDs
            chunks = [
                ChunkInput(
                    chunk_id="C-001",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[1, 1],
                    text="Identical text content.",
                    confidence=0.91,
                ),
                ChunkInput(
                    chunk_id="C-002",
                    case_id="24-890-H",
                    document_id="W001-24-890-H",
                    page_range=[2, 2],
                    text="Identical text content.",
                    confidence=0.91,
                ),
            ]

            pipeline.process_chunks(chunks)

            # Should have 2 separate vectors
            assert pipeline.get_vector_count() == 2
            assert len(pipeline.store.metadata) == 2
