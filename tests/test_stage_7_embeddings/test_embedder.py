"""
Unit tests for Stage 7: Vector Embeddings - Embedder

Tests for chunk embedding generation.
"""

import numpy as np
import pytest

from stage_6_ner.models import ChunkInput
from stage_7_embeddings.embedder import embed_chunk, embed_chunks, extract_metadata
from stage_7_embeddings.embedding_model import EmbeddingModelLoader


class TestEmbedChunk:
    """Tests for embed_chunk function."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_embed_chunk_input(self):
        """Should embed ChunkInput object."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="I heard a loud crash around 8:15 PM.",
            speaker="Clara Higgins",
            confidence=0.91,
        )

        embedding = embed_chunk(chunk)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_embed_dict_input(self):
        """Should embed dict input."""
        chunk = {
            "chunk_id": "C-001",
            "case_id": "24-890-H",
            "document_id": "W001-24-890-H",
            "page_range": [1, 1],
            "text": "I heard a loud crash around 8:15 PM.",
            "speaker": "Clara Higgins",
            "confidence": 0.91,
        }

        embedding = embed_chunk(chunk)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_text_not_modified(self):
        """Chunk text should not be modified during embedding."""
        original_text = "I heard a loud crash around 8:15 PM."
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text=original_text,
            confidence=0.91,
        )

        _ = embed_chunk(chunk)

        # Text should be unchanged
        assert chunk.text == original_text

    def test_deterministic_embedding(self):
        """Same chunk should produce same embedding."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="I heard a loud crash around 8:15 PM.",
            confidence=0.91,
        )

        embedding1 = embed_chunk(chunk)
        embedding2 = embed_chunk(chunk)

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_different_text_different_embedding(self):
        """Different text should produce different embeddings."""
        chunk1 = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="I heard a loud crash around 8:15 PM.",
            confidence=0.91,
        )
        chunk2 = ChunkInput(
            chunk_id="C-002",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="The suspect was wearing a blue jacket.",
            confidence=0.91,
        )

        embedding1 = embed_chunk(chunk1)
        embedding2 = embed_chunk(chunk2)

        # Should be different
        assert not np.array_equal(embedding1, embedding2)


class TestEmbedChunks:
    """Tests for embed_chunks function."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_embed_multiple_chunks(self):
        """Should embed multiple chunks."""
        chunks = [
            ChunkInput(
                chunk_id="C-001",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="First chunk text.",
                confidence=0.91,
            ),
            ChunkInput(
                chunk_id="C-002",
                case_id="24-890-H",
                document_id="W001-24-890-H",
                page_range=[1, 1],
                text="Second chunk text.",
                confidence=0.91,
            ),
        ]

        embeddings = embed_chunks(chunks)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 384)

    def test_chunks_processed_independently(self):
        """Each chunk should be processed independently."""
        chunk1 = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="First chunk text.",
            confidence=0.91,
        )
        chunk2 = ChunkInput(
            chunk_id="C-002",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 1],
            text="Second chunk text.",
            confidence=0.91,
        )

        # Embed individually
        emb1_single = embed_chunk(chunk1)
        emb2_single = embed_chunk(chunk2)

        # Embed as batch
        embeddings = embed_chunks([chunk1, chunk2])

        # Results should match
        np.testing.assert_array_equal(embeddings[0], emb1_single)
        np.testing.assert_array_equal(embeddings[1], emb2_single)


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extract_from_chunk_input(self):
        """Should extract metadata from ChunkInput."""
        chunk = ChunkInput(
            chunk_id="C-001",
            case_id="24-890-H",
            document_id="W001-24-890-H",
            page_range=[1, 2],
            speaker="Clara Higgins",
            text="Some text.",
            confidence=0.91,
        )

        metadata = extract_metadata(chunk)

        assert metadata["chunk_id"] == "C-001"
        assert metadata["case_id"] == "24-890-H"
        assert metadata["document_id"] == "W001-24-890-H"
        assert metadata["page_range"] == [1, 2]
        assert metadata["speaker"] == "Clara Higgins"
        assert metadata["confidence"] == 0.91

    def test_extract_from_dict(self):
        """Should extract metadata from dict."""
        chunk = {
            "chunk_id": "C-001",
            "case_id": "24-890-H",
            "document_id": "W001-24-890-H",
            "page_range": [1, 1],
            "speaker": None,
            "text": "Some text.",
            "confidence": 0.85,
        }

        metadata = extract_metadata(chunk)

        assert metadata["chunk_id"] == "C-001"
        assert metadata["speaker"] is None
        assert metadata["confidence"] == 0.85
