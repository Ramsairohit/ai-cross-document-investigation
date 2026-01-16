"""
Stage 5: Logical Chunking - Data Models

Pydantic models for forensic-grade logical chunking.
These models define the exact schema for meaning-preserving chunks.

IMPORTANT: This stage defines boundaries only.
No NLP, no inference, no intelligence.
"""

from typing import Optional

from pydantic import BaseModel, Field


class BlockInput(BaseModel):
    """
    Input block from Stage 4 (Semantic Cleaning).

    This is the expected input format for chunking.
    Must match Stage 4 CleanedBlock schema.
    """

    block_id: str = Field(..., description="Original block ID from earlier stages")
    page: int = Field(..., ge=1, description="1-indexed page number")
    clean_text: str = Field(..., description="Semantically cleaned text content")
    speaker: Optional[str] = Field(default=None, description="Detected speaker label from Stage 3")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Confidence score from cleaning (0.0-1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "block_id": "b12",
                "page": 2,
                "clean_text": "DET. SMITH: Where were you at 8:15 PM?",
                "speaker": "DET. SMITH",
                "confidence": 0.93,
            }
        }


class Chunk(BaseModel):
    """
    Meaning-preserving chunk compatible with Stage 6 (NER) and Stage 7 (Embeddings).

    Every field is MANDATORY for legal traceability.
    This schema must match Stage 6 ChunkInput exactly.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier (e.g., 'C-0001')")
    case_id: str = Field(..., description="Case ID for chain-of-custody")
    document_id: str = Field(..., description="Source document ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page] - must be same page"
    )
    speaker: Optional[str] = Field(default=None, description="Speaker label for this chunk")
    text: str = Field(..., description="Exact concatenation of source block texts")
    source_block_ids: list[str] = Field(
        ..., min_length=1, description="Block IDs that compose this chunk (provenance)"
    )
    token_count: int = Field(..., ge=1, description="Exact token count using tiktoken cl100k_base")
    chunk_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Conservative aggregated confidence (minimum of sources)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "C-0001",
                "case_id": "24-890-H",
                "document_id": "W001-24-890-H",
                "page_range": [2, 2],
                "speaker": "DET. SMITH",
                "text": "DET. SMITH: Where were you at 8:15 PM?",
                "source_block_ids": ["b12"],
                "token_count": 21,
                "chunk_confidence": 0.93,
            }
        }


class ChunkingConfig(BaseModel):
    """
    Configuration for chunking behavior.

    Token limits are HARD CONSTRAINTS.
    """

    min_tokens: int = Field(default=300, ge=1, description="Minimum tokens per chunk (soft limit)")
    max_tokens: int = Field(default=1000, ge=1, description="Maximum tokens per chunk (HARD limit)")
    encoding_name: str = Field(
        default="cl100k_base", description="Tiktoken encoding for token counting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "min_tokens": 300,
                "max_tokens": 1000,
                "encoding_name": "cl100k_base",
            }
        }


class ChunkingResult(BaseModel):
    """
    Complete chunking output for a single document.

    This is the MANDATORY output schema for Stage 5.
    One ChunkingResult is produced per input document.
    """

    document_id: str = Field(..., description="Document ID from input")
    case_id: str = Field(..., description="Case ID from input")
    source_file: str = Field(..., description="Original source filename")
    chunks: list[Chunk] = Field(
        default_factory=list, description="Ordered list of meaning-preserving chunks"
    )
    total_chunks: int = Field(default=0, description="Total number of chunks produced")
    total_blocks_processed: int = Field(default=0, description="Total blocks consumed")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "W001-24-890-H",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "chunks": [],
                "total_chunks": 0,
                "total_blocks_processed": 0,
            }
        }
