"""
Unit tests for Stage 11: RAG - Determinism

Tests for deterministic behavior of RAG system.
"""

import numpy as np
import pytest

from stage_11_rag.models import RAGConfig, RAGQuery
from stage_11_rag.retriever import retrieve_chunks


class TestDeterminism100Runs:
    """Tests for 100-run determinism verification."""

    @pytest.fixture
    def mock_index_and_data(self):
        """Create mock FAISS index and metadata."""
        import faiss

        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        # Use fixed seed for reproducibility
        np.random.seed(42)
        vectors = np.random.rand(5, dimension).astype(np.float32)
        index.add(vectors)

        metadata = [
            {
                "chunk_id": f"C{i}",
                "case_id": "001",
                "document_id": f"D{i}",
                "page_range": [i, i],
                "text": f"Text content {i}",
                "speaker": f"Speaker{i}",
            }
            for i in range(5)
        ]

        # Fixed query vector
        np.random.seed(123)
        query_vec = np.random.rand(dimension).astype(np.float32)

        return {
            "index": index,
            "metadata": metadata,
            "query_vec": query_vec,
        }

    def test_retrieval_determinism(self, mock_index_and_data):
        """Same query should return same chunks in same order."""
        index = mock_index_and_data["index"]
        metadata = mock_index_and_data["metadata"]
        query_vec = mock_index_and_data["query_vec"]

        config = RAGConfig(top_k=3)

        results = []
        for _ in range(100):
            chunks = retrieve_chunks(query_vec, index, metadata, config)
            chunk_ids = tuple(c.chunk_id for c in chunks)
            results.append(chunk_ids)

        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result == first, f"Result differs at run {i}"

    def test_scores_determinism(self, mock_index_and_data):
        """Same query should return same scores."""
        index = mock_index_and_data["index"]
        metadata = mock_index_and_data["metadata"]
        query_vec = mock_index_and_data["query_vec"]

        config = RAGConfig(top_k=3)

        scores_list = []
        for _ in range(100):
            chunks = retrieve_chunks(query_vec, index, metadata, config)
            scores = tuple(c.score for c in chunks)
            scores_list.append(scores)

        first = scores_list[0]
        for i, scores in enumerate(scores_list[1:], 2):
            assert scores == first, f"Scores differ at run {i}"


class TestDeterminismEdgeCases:
    """Edge case tests for determinism."""

    def test_empty_index_determinism(self):
        """Empty index should consistently return empty results."""
        import faiss

        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        query_vec = np.random.rand(dimension).astype(np.float32)

        results = []
        for _ in range(100):
            chunks = retrieve_chunks(query_vec, index, [], RAGConfig(top_k=5))
            results.append(len(chunks))

        assert all(r == 0 for r in results)
