"""
Unit tests for Stage 11: RAG - Data Models

Tests for query, answer, and source schemas.
"""

import pytest

from stage_11_rag.models import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GraphFact,
    RAGAnswer,
    RAGConfig,
    RAGQuery,
    RetrievedChunk,
    SourceReference,
    TimelineEvent,
)


class TestRAGQuery:
    """Tests for RAGQuery model."""

    def test_valid_query(self):
        """Should create valid query."""
        query = RAGQuery(
            case_id="24-890-H",
            question="Who last spoke to the victim?",
        )
        assert query.case_id == "24-890-H"
        assert "victim" in query.question


class TestSourceReference:
    """Tests for SourceReference model."""

    def test_valid_source(self):
        """Should create valid source reference."""
        source = SourceReference(
            chunk_id="CHUNK_012",
            document_id="W001-24-890-H",
            page_range=[3, 3],
            excerpt="I spoke with Julian at 9 PM.",
        )
        assert source.chunk_id == "CHUNK_012"


class TestRAGAnswer:
    """Tests for RAGAnswer model."""

    def test_valid_answer(self):
        """Should create valid answer."""
        answer = RAGAnswer(
            answer="Based on evidence...",
            confidence=0.82,
            sources=[],
            limitations=[],
        )
        assert answer.confidence == 0.82

    def test_insufficient_evidence(self):
        """INSUFFICIENT_EVIDENCE_ANSWER should have zero confidence."""
        assert INSUFFICIENT_EVIDENCE_ANSWER.confidence == 0.0
        assert len(INSUFFICIENT_EVIDENCE_ANSWER.sources) == 0


class TestRAGConfig:
    """Tests for RAGConfig model."""

    def test_default_values(self):
        """Should have correct defaults."""
        config = RAGConfig()
        assert config.top_k == 5
        assert config.include_graph is True
        assert config.include_timeline is True
        assert config.include_contradictions is True


class TestRetrievedChunk:
    """Tests for RetrievedChunk model."""

    def test_valid_chunk(self):
        """Should create valid retrieved chunk."""
        chunk = RetrievedChunk(
            chunk_id="C1",
            document_id="D1",
            case_id="001",
            page_range=[1, 1],
            text="Some evidence text.",
            score=0.85,
        )
        assert chunk.score == 0.85


class TestGraphFact:
    """Tests for GraphFact model."""

    def test_valid_fact(self):
        """Should create valid fact."""
        fact = GraphFact(
            subject="Marcus",
            predicate="KNOWS",
            object="Julian",
        )
        assert fact.predicate == "KNOWS"


class TestTimelineEvent:
    """Tests for TimelineEvent model."""

    def test_valid_event(self):
        """Should create valid event."""
        event = TimelineEvent(
            event_id="EVT_001",
            timestamp="2024-03-15T20:00:00",
            description="Event occurred.",
            chunk_id="C1",
        )
        assert "2024" in event.timestamp
