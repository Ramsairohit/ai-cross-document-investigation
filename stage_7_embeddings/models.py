"""
Stage 7: Vector Embeddings - Data Models

Pydantic models for forensic-grade vector storage with full provenance.
Every stored vector MUST preserve metadata for legal traceability.

IMPORTANT:
- Metadata is deterministically linked to FAISS vector positions
- No inference, no interpretation, no cross-chunk analysis
"""

from typing import Optional

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    """
    Vector metadata with full provenance.

    Every field traces back to the original chunk for chain-of-custody.
    The vector_id corresponds to the position in the FAISS index.
    """

    chunk_id: str = Field(..., description="Source chunk ID from Stage 5")
    vector_id: int = Field(..., ge=0, description="Position in FAISS index")
    case_id: str = Field(..., description="Case ID for chain-of-custody")
    document_id: str = Field(..., description="Source document ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    speaker: Optional[str] = Field(default=None, description="Speaker label if present in chunk")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Chunk confidence from Stage 5")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "C-001",
                "vector_id": 0,
                "case_id": "24-890-H",
                "document_id": "W001-24-890-H",
                "page_range": [1, 1],
                "speaker": "Clara Higgins",
                "confidence": 0.91,
            }
        }


class EmbeddingResult(BaseModel):
    """
    Result of embedding a single chunk.

    Returned by the pipeline to confirm successful embedding.
    """

    chunk_id: str = Field(..., description="Processed chunk ID")
    vector_id: int = Field(..., ge=0, description="Assigned position in FAISS index")
    embedding_dimension: int = Field(default=384, description="Dimension of the embedding vector")
    success: bool = Field(default=True, description="Whether embedding succeeded")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "C-001",
                "vector_id": 0,
                "embedding_dimension": 384,
                "success": True,
            }
        }


class EmbeddingConfig(BaseModel):
    """
    Configuration for the embedding pipeline.

    Allows customization of model and index settings.
    """

    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence Transformer model name",
    )
    embedding_dimension: int = Field(default=384, description="Expected embedding dimension")
    index_type: str = Field(
        default="Flat",
        description="FAISS index type: 'Flat' for exact, 'IVF' for approximate",
    )
    normalize_embeddings: bool = Field(
        default=True, description="Whether to L2-normalize embeddings"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "index_type": "Flat",
                "normalize_embeddings": True,
            }
        }
