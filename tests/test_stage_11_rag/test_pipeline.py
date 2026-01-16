"""
Unit tests for Stage 11: RAG - Pipeline

Tests for main RAG pipeline.
"""

import numpy as np
import pytest

from stage_11_rag.models import (
    RAGAnswer,
    RAGConfig,
    RAGQuery,
    RetrievedChunk,
)
from stage_11_rag.retriever import (
    filter_by_case,
    retrieve_chunks,
)
from stage_11_rag.llm_client import (
    calculate_answer_confidence,
    generate_stub_answer,
)


class TestRetrieveChunks:
    """Tests for chunk retrieval."""

    def test_retrieves_chunks(self):
        """Should retrieve chunks from mock index."""
        # Create mock index
        import faiss

        dimension = 384
        index = faiss.IndexFlatL2(dimension)

        # Add vectors
        vectors = np.random.rand(3, dimension).astype(np.float32)
        index.add(vectors)

        # Mock metadata
        metadata = [
            {
                "chunk_id": "C1",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "text": "Text 1",
            },
            {
                "chunk_id": "C2",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [2, 2],
                "text": "Text 2",
            },
            {
                "chunk_id": "C3",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [3, 3],
                "text": "Text 3",
            },
        ]

        # Query
        query_vec = np.random.rand(dimension).astype(np.float32)
        config = RAGConfig(top_k=2)

        chunks = retrieve_chunks(query_vec, index, metadata, config)

        assert len(chunks) <= 2


class TestFilterByCase:
    """Tests for case filtering."""

    def test_filters_to_case(self):
        """Should only return chunks from specified case."""
        chunks = [
            RetrievedChunk(
                chunk_id="C1",
                document_id="D1",
                case_id="CASE_A",
                page_range=[1, 1],
                text="Text",
                score=0.9,
            ),
            RetrievedChunk(
                chunk_id="C2",
                document_id="D2",
                case_id="CASE_B",
                page_range=[1, 1],
                text="Text",
                score=0.8,
            ),
        ]

        filtered = filter_by_case(chunks, "CASE_A")

        assert len(filtered) == 1
        assert filtered[0].case_id == "CASE_A"


class TestCalculateConfidence:
    """Tests for confidence calculation."""

    def test_no_sources_zero_confidence(self):
        """Zero sources should have zero confidence."""
        confidence = calculate_answer_confidence(
            sources_count=0,
            has_contradictions=False,
            has_gaps=False,
        )
        assert confidence == 0.0

    def test_contradictions_reduce_confidence(self):
        """Contradictions should reduce confidence."""
        without = calculate_answer_confidence(3, False, False)
        with_cont = calculate_answer_confidence(3, True, False)
        assert with_cont < without

    def test_gaps_reduce_confidence(self):
        """Gaps should reduce confidence."""
        without = calculate_answer_confidence(3, False, False)
        with_gaps = calculate_answer_confidence(3, False, True)
        assert with_gaps < without


class TestStubAnswer:
    """Tests for stub answer generation."""

    def test_insufficient_evidence(self):
        """Should indicate insufficient evidence for empty context."""
        answer = generate_stub_answer("Who did it?", "", None)
        assert "insufficient" in answer.lower() or "not contain" in answer.lower()

    def test_includes_source_references(self):
        """Should reference sources when available."""
        context = "[Source 1: C1]\nSome text.\n[Source 2: C2]\nMore text."
        answer = generate_stub_answer("What happened?", context, None)
        assert "Source" in answer


class TestNoCrossCase:
    """Tests to verify no cross-case access."""

    def test_filters_out_other_cases(self):
        """Should not include chunks from other cases."""
        chunks = [
            RetrievedChunk(
                chunk_id="C1",
                document_id="D1",
                case_id="CASE_A",
                page_range=[1, 1],
                text="Relevant",
                score=0.95,
            ),
            RetrievedChunk(
                chunk_id="C2",
                document_id="D2",
                case_id="CASE_B",
                page_range=[1, 1],
                text="From other case",
                score=0.99,
            ),
        ]

        filtered = filter_by_case(chunks, "CASE_A")

        # Even though C2 has higher score, it should be excluded
        assert all(c.case_id == "CASE_A" for c in filtered)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_query_same_confidence(self):
        """Same inputs should produce same confidence."""
        results = [calculate_answer_confidence(3, True, False) for _ in range(100)]
        assert all(r == results[0] for r in results)
