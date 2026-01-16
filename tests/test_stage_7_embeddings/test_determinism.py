"""
Unit tests for Stage 7: Vector Embeddings - Determinism

Critical tests to verify deterministic behavior required for court auditing.
Same input MUST produce identical output every time.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from stage_6_ner.models import ChunkInput
from stage_7_embeddings import EmbeddingPipeline
from stage_7_embeddings.embedder import embed_chunk
from stage_7_embeddings.embedding_model import EmbeddingModelLoader


class TestEmbeddingDeterminism:
    """Tests to verify embeddings are deterministic."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_same_input_same_output_10_times(self):
        """Same input should produce identical vectors 10 times."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="I heard a loud crash around 8:15 PM near the corner of Main Street.",
            speaker="Clara Higgins",
            confidence=0.91,
        )

        embeddings = [embed_chunk(chunk) for _ in range(10)]

        # All embeddings should be identical
        for i in range(1, len(embeddings)):
            np.testing.assert_array_equal(
                embeddings[0],
                embeddings[i],
                err_msg=f"Embedding {i} differs from embedding 0",
            )

    def test_same_input_same_output_100_times(self):
        """Same input should produce identical vectors 100 times."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="The suspect was wearing a dark blue jacket and black sneakers.",
            confidence=0.91,
        )

        first_embedding = embed_chunk(chunk)

        for _ in range(99):
            embedding = embed_chunk(chunk)
            np.testing.assert_array_equal(first_embedding, embedding)

    def test_determinism_across_model_reloads(self):
        """Embeddings should be identical even after model reload."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="The witness observed the incident from approximately 50 feet away.",
            confidence=0.91,
        )

        # First embedding
        embedding1 = embed_chunk(chunk)

        # Reset and reload model
        EmbeddingModelLoader.reset()

        # Second embedding
        embedding2 = embed_chunk(chunk)

        np.testing.assert_array_equal(embedding1, embedding2)


class TestIndexDeterminism:
    """Tests to verify FAISS indexing is deterministic."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_identical_index_ordering(self):
        """Re-indexing same chunks should produce identical ordering."""
        chunks = [
            ChunkInput(
                chunk_id=f"C-{i:03d}",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text=f"This is chunk number {i} with unique content.",
                confidence=0.91,
            )
            for i in range(5)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # First indexing
            pipeline1 = EmbeddingPipeline(Path(tmpdir) / "run1")
            results1 = pipeline1.process_chunks(chunks)

            # Second indexing
            pipeline2 = EmbeddingPipeline(Path(tmpdir) / "run2")
            results2 = pipeline2.process_chunks(chunks)

            # Check same ordering
            for r1, r2 in zip(results1, results2):
                assert r1.chunk_id == r2.chunk_id
                assert r1.vector_id == r2.vector_id

    def test_identical_vectors_in_index(self):
        """Re-indexing should produce identical vectors."""
        chunks = [
            ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="Critical evidence found at the scene.",
                confidence=0.91,
            ),
            ChunkInput(
                chunk_id="C-002",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[2, 2],
                text="Witness reported hearing gunshots.",
                confidence=0.88,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            # First indexing
            pipeline1 = EmbeddingPipeline(Path(tmpdir) / "run1")
            pipeline1.process_chunks(chunks)

            # Second indexing
            pipeline2 = EmbeddingPipeline(Path(tmpdir) / "run2")
            pipeline2.process_chunks(chunks)

            # Compare vectors
            for i in range(2):
                v1 = pipeline1.store.index_manager.reconstruct(i)
                v2 = pipeline2.store.index_manager.reconstruct(i)
                np.testing.assert_array_equal(v1, v2)


class TestPersistenceDeterminism:
    """Tests to verify persistence produces identical results."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_faiss_reload_preserves_vectors(self):
        """Loading FAISS index should preserve exact vectors."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="The firearm was recovered from the vehicle.",
            confidence=0.91,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)

            # Create, save
            pipeline = EmbeddingPipeline(storage_dir)
            pipeline.process_chunk(chunk)
            original_vector = pipeline.store.index_manager.reconstruct(0).copy()
            pipeline.save()

            # Load into new pipeline
            new_pipeline = EmbeddingPipeline(storage_dir)
            new_pipeline.load()
            loaded_vector = new_pipeline.store.index_manager.reconstruct(0)

            np.testing.assert_array_equal(original_vector, loaded_vector)

    def test_metadata_reload_preserves_linkage(self):
        """Loading metadata should preserve vector linkage."""
        chunks = [
            ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="First statement.",
                speaker="Alice",
                confidence=0.91,
            ),
            ChunkInput(
                chunk_id="C-002",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[2, 2],
                text="Second statement.",
                speaker="Bob",
                confidence=0.85,
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)

            # Create, save
            pipeline = EmbeddingPipeline(storage_dir)
            pipeline.process_chunks(chunks)
            pipeline.save()

            # Load into new pipeline
            new_pipeline = EmbeddingPipeline(storage_dir)
            new_pipeline.load()

            # Verify metadata linkage
            record0 = new_pipeline.store.get_metadata(0)
            record1 = new_pipeline.store.get_metadata(1)

            assert record0 is not None
            assert record0.chunk_id == "C-001"
            assert record0.vector_id == 0
            assert record0.speaker == "Alice"

            assert record1 is not None
            assert record1.chunk_id == "C-002"
            assert record1.vector_id == 1
            assert record1.speaker == "Bob"

    def test_multiple_save_load_preserves_determinism(self):
        """Multiple save/load cycles should preserve vectors."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="Forensic evidence requirement.",
            confidence=0.91,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            storage_dir = Path(tmpdir)
            pipeline = EmbeddingPipeline(storage_dir)
            pipeline.process_chunk(chunk)

            original_vector = pipeline.store.index_manager.reconstruct(0).copy()

            # Multiple save/load cycles
            for _ in range(5):
                pipeline.save()
                pipeline.load()

            final_vector = pipeline.store.index_manager.reconstruct(0)

            np.testing.assert_array_equal(original_vector, final_vector)
