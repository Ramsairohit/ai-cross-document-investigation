"""
Stage 11: Retrieval-Augmented Generation (RAG) - Data Models

Pydantic models for forensic-grade RAG system.

IMPORTANT: This stage speaks for the evidence.
It does NOT think, decide, or judge.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RAGQuery(BaseModel):
    """
    Investigator query for the RAG system.
    """

    case_id: str = Field(..., description="Case identifier")
    question: str = Field(..., description="Investigator's question")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "question": "Who last spoke to the victim?",
            }
        }


class SourceReference(BaseModel):
    """
    Reference to evidence source with full provenance.

    Every claim in an answer MUST cite a source.
    """

    chunk_id: str = Field(..., description="Source chunk identifier")
    document_id: str = Field(..., description="Source document identifier")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    excerpt: str = Field(..., description="Relevant excerpt from the chunk")
    speaker: Optional[str] = Field(default=None, description="Speaker if applicable")
    timestamp: Optional[str] = Field(default=None, description="Timestamp if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "CHUNK_012",
                "document_id": "W001-24-890-H",
                "page_range": [3, 3],
                "excerpt": "I spoke with Julian Thorne at approximately 9:10 PM.",
                "speaker": "Clara Higgins",
                "timestamp": "2024-03-15T21:10:00",
            }
        }


class RAGAnswer(BaseModel):
    """
    RAG system answer with citations and limitations.

    CRITICAL:
    - Every claim must cite a source
    - Contradictions must be in limitations
    - No hallucination allowed
    """

    answer: str = Field(..., description="Evidence-based answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: list[SourceReference] = Field(default_factory=list, description="Cited sources")
    limitations: list[str] = Field(
        default_factory=list, description="Limitations and contradictions"
    )
    query: Optional[str] = Field(default=None, description="Original query")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Based on available evidence, the last recorded interaction...",
                "confidence": 0.82,
                "sources": [],
                "limitations": ["Timeline contains a 45-minute gap"],
            }
        }


class RetrievedChunk(BaseModel):
    """
    Chunk retrieved from vector search with score.
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Document identifier")
    case_id: str = Field(..., description="Case identifier")
    page_range: list[int] = Field(..., description="Page range")
    text: str = Field(..., description="Chunk text")
    speaker: Optional[str] = Field(default=None, description="Speaker")
    score: float = Field(..., description="Similarity score")
    confidence: float = Field(default=1.0, description="Chunk confidence")


class GraphFact(BaseModel):
    """
    Fact retrieved from knowledge graph.
    """

    subject: str = Field(..., description="Subject entity")
    predicate: str = Field(..., description="Relationship type")
    object: str = Field(..., description="Object entity")
    source_chunk_id: Optional[str] = Field(default=None, description="Source chunk")


class TimelineEvent(BaseModel):
    """
    Event from timeline for context.
    """

    event_id: str = Field(..., description="Event identifier")
    timestamp: str = Field(..., description="ISO-8601 timestamp")
    description: str = Field(..., description="Event description")
    chunk_id: str = Field(..., description="Source chunk")


class RAGConfig(BaseModel):
    """
    Configuration for the RAG pipeline.
    """

    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    min_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum similarity score")
    include_graph: bool = Field(default=True, description="Include graph lookup")
    include_timeline: bool = Field(default=True, description="Include timeline check")
    include_contradictions: bool = Field(default=True, description="Include contradiction check")
    llm_model: str = Field(default="gpt-4", description="LLM model to use")
    max_context_tokens: int = Field(default=4000, description="Maximum context tokens")

    class Config:
        json_schema_extra = {
            "example": {
                "top_k": 5,
                "min_score": 0.3,
                "include_graph": True,
                "include_timeline": True,
                "include_contradictions": True,
            }
        }


# Insufficient evidence response
INSUFFICIENT_EVIDENCE_ANSWER = RAGAnswer(
    answer="The available evidence does not contain sufficient information to answer this question.",
    confidence=0.0,
    sources=[],
    limitations=["Insufficient evidence"],
)
