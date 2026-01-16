"""
Stage 10: Contradiction Detection - Data Models

Pydantic models for forensic-grade contradiction detection.
These models define the exact schema for contradictions.

IMPORTANT: This stage FLAGS inconsistencies only.
It NEVER resolves, removes, or explains them.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class ContradictionType(str, Enum):
    """
    Strict enumeration of allowed contradiction types.

    Only these 4 types are permitted.
    DO NOT add new types without legal review.
    """

    TIME_CONFLICT = "TIME_CONFLICT"
    LOCATION_CONFLICT = "LOCATION_CONFLICT"
    STATEMENT_VS_EVIDENCE = "STATEMENT_VS_EVIDENCE"
    DENIAL_VS_ASSERTION = "DENIAL_VS_ASSERTION"


class ContradictionSeverity(str, Enum):
    """
    Severity levels for contradictions.

    LOW: Weak/indirect contradiction
    MEDIUM: Clear but indirect
    HIGH: Direct inconsistency
    CRITICAL: Mutually exclusive facts
    """

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ContradictionStatus(str, Enum):
    """
    Status of a contradiction.

    FLAGGED is the ONLY allowed status.
    Contradictions are NEVER resolved or removed.
    """

    FLAGGED = "FLAGGED"


class ChunkReference(BaseModel):
    """
    Reference to a chunk with key provenance fields.

    Used to preserve provenance in contradictions.
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    speaker: Optional[str] = Field(default=None, description="Speaker label")
    text: str = Field(..., description="Exact chunk text")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "page_range": [2, 2],
                "speaker": "Marcus Vane",
                "text": "I was at home at 9 PM.",
            }
        }


class Contradiction(BaseModel):
    """
    A detected contradiction between two chunks.

    Contradictions are FLAGGED but NEVER resolved.
    Full provenance is preserved for legal traceability.
    """

    contradiction_id: str = Field(
        ..., description="Unique contradiction identifier (e.g., 'CONT_001')"
    )
    case_id: str = Field(..., description="Case identifier")
    type: ContradictionType = Field(..., description="Type of contradiction")
    chunk_a: ChunkReference = Field(..., description="First chunk in contradiction")
    chunk_b: ChunkReference = Field(..., description="Second chunk in contradiction")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score (min of chunk confidences)"
    )
    severity: ContradictionSeverity = Field(..., description="Severity level")
    explanation: str = Field(
        ..., description="Factual explanation of the contradiction (no inference)"
    )
    status: ContradictionStatus = Field(
        default=ContradictionStatus.FLAGGED,
        description="Status - always FLAGGED",
    )
    shared_entities: list[str] = Field(
        default_factory=list, description="Entities shared between chunks"
    )
    timestamp: Optional[str] = Field(default=None, description="Relevant timestamp if applicable")

    class Config:
        json_schema_extra = {
            "example": {
                "contradiction_id": "CONT_001",
                "case_id": "24-890-H",
                "type": "LOCATION_CONFLICT",
                "chunk_a": {
                    "chunk_id": "CHUNK_001",
                    "document_id": "DOC123",
                    "page_range": [2, 2],
                    "speaker": "Marcus Vane",
                    "text": "I was at home at 9 PM.",
                },
                "chunk_b": {
                    "chunk_id": "CHUNK_007",
                    "document_id": "DOC456",
                    "page_range": [3, 3],
                    "speaker": "Julian Thorne",
                    "text": "I saw Marcus at the crime scene at 9 PM.",
                },
                "confidence": 0.91,
                "severity": "CRITICAL",
                "explanation": "Two chunks assert different locations for Marcus at 9 PM.",
                "status": "FLAGGED",
                "shared_entities": ["Marcus"],
                "timestamp": "2024-03-15T21:00:00",
            }
        }


class ContradictionResult(BaseModel):
    """
    Complete contradiction detection result for a case.

    Contains all detected contradictions without resolution.
    """

    case_id: str = Field(..., description="Case identifier")
    contradictions: list[Contradiction] = Field(
        default_factory=list, description="All detected contradictions"
    )
    total_contradictions: int = Field(default=0, description="Total number of contradictions")
    chunks_analyzed: int = Field(default=0, description="Number of chunks analyzed")
    pairs_compared: int = Field(default=0, description="Number of pairs compared")
    by_type: dict[str, int] = Field(default_factory=dict, description="Count by contradiction type")
    by_severity: dict[str, int] = Field(default_factory=dict, description="Count by severity")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "contradictions": [],
                "total_contradictions": 0,
                "chunks_analyzed": 10,
                "pairs_compared": 5,
                "by_type": {},
                "by_severity": {},
            }
        }


class ContradictionConfig(BaseModel):
    """
    Configuration for contradiction detection.
    """

    use_nli: bool = Field(default=False, description="Use NLI for confirmation (secondary)")
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Minimum confidence to report"
    )
    require_entity_overlap: bool = Field(
        default=True, description="Only compare chunks with shared entities"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "use_nli": False,
                "min_confidence": 0.5,
                "require_entity_overlap": True,
            }
        }
