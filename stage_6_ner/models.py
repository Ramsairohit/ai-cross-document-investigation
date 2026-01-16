"""
Stage 6: NER - Data Models

Pydantic models and enums for forensic-grade Named Entity Recognition.
These models define the exact schema for extracted entities.

IMPORTANT: Entities are labeled, NOT interpreted.
No inference, no conclusions, no cross-chunk analysis.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EntityType(str, Enum):
    """
    Strict enumeration of allowed entity types.

    DO NOT add new types without legal review.
    """

    PERSON = "PERSON"
    WITNESS = "WITNESS"
    SUSPECT = "SUSPECT"
    LOCATION = "LOCATION"
    TIME = "TIME"
    EVIDENCE = "EVIDENCE"
    WEAPON = "WEAPON"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"


class ExtractionSource(str, Enum):
    """Source of entity extraction."""

    SPACY = "spacy"
    RULE_BASED = "rule_based"
    METADATA = "metadata"


class ExtractedEntity(BaseModel):
    """
    A single extracted entity with full provenance.

    Every field is MANDATORY for legal traceability.
    If provenance is missing, the entity is invalid.
    """

    entity_id: str = Field(..., description="Unique entity identifier (e.g., 'ENT_001')")
    entity_type: EntityType = Field(..., description="Type from strict enum")
    text: str = Field(..., description="Exact text span of the entity")
    chunk_id: str = Field(..., description="Source chunk ID for provenance")
    document_id: str = Field(..., description="Source document ID")
    case_id: str = Field(..., description="Case ID for chain-of-custody")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    start_char: int = Field(..., ge=0, description="Start character offset in chunk text")
    end_char: int = Field(..., ge=0, description="End character offset in chunk text")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    source: ExtractionSource = Field(..., description="Extraction method used")
    role: Optional[str] = Field(
        default=None,
        description="Role if applicable (e.g., 'WITNESS' from metadata). Only from metadata, never inferred.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "entity_id": "ENT_001",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "page_range": [1, 1],
                "start_char": 6,
                "end_char": 17,
                "confidence": 0.92,
                "source": "spacy",
                "role": None,
            }
        }


class ChunkInput(BaseModel):
    """
    Input schema for Stage 5 chunks.

    This is the expected input format for NER processing.
    """

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Source document ID")
    case_id: str = Field(..., description="Case ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    speaker: Optional[str] = Field(
        default=None, description="Speaker label (e.g., 'WITNESS', 'SUSPECT')"
    )
    text: str = Field(..., description="Chunk text content")
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Chunk confidence from Stage 5"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "page_range": [1, 1],
                "speaker": "WITNESS",
                "text": "I saw Marcus Vane near 420 Harrow Lane at 8:15 PM.",
                "confidence": 0.93,
            }
        }


class NERResult(BaseModel):
    """
    Complete NER output for a single chunk.

    Contains all entities extracted from the chunk with full provenance.
    """

    chunk_id: str = Field(..., description="Source chunk ID")
    document_id: str = Field(..., description="Source document ID")
    case_id: str = Field(..., description="Case ID")
    entities: list[ExtractedEntity] = Field(
        default_factory=list, description="All entities extracted from this chunk"
    )
    entity_count: int = Field(default=0, description="Total number of entities extracted")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "entities": [
                    {
                        "entity_id": "ENT_001",
                        "entity_type": "PERSON",
                        "text": "Marcus Vane",
                        "chunk_id": "CHUNK_001",
                        "document_id": "DOC123",
                        "case_id": "24-890-H",
                        "page_range": [1, 1],
                        "start_char": 6,
                        "end_char": 17,
                        "confidence": 0.92,
                        "source": "spacy",
                        "role": None,
                    }
                ],
                "entity_count": 1,
            }
        }


# Mapping from spaCy labels to our EntityType
SPACY_LABEL_MAP: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "PER": EntityType.PERSON,
    "GPE": EntityType.LOCATION,  # Geopolitical entity
    "LOC": EntityType.LOCATION,
    "FAC": EntityType.LOCATION,  # Facility
    "ORG": EntityType.LOCATION,  # Organization (can be location-like)
    "DATE": EntityType.TIME,
    "TIME": EntityType.TIME,
}

# Keywords for rule-based weapon detection
WEAPON_KEYWORDS: set[str] = {
    "gun",
    "pistol",
    "revolver",
    "rifle",
    "shotgun",
    "firearm",
    "knife",
    "blade",
    "dagger",
    "machete",
    "sword",
    "bat",
    "baseball bat",
    "club",
    "hammer",
    "axe",
    "crowbar",
    "brass knuckles",
    "taser",
    "stun gun",
    "pepper spray",
    "mace",
}

# Keywords for rule-based evidence detection
EVIDENCE_KEYWORDS: set[str] = {
    "fingerprint",
    "fingerprints",
    "dna",
    "blood",
    "hair",
    "fiber",
    "fibers",
    "footprint",
    "footprints",
    "shell casing",
    "shell casings",
    "bullet",
    "bullets",
    "cufflink",
    "cufflinks",
    "wallet",
    "id card",
    "driver's license",
    "license plate",
    "surveillance",
    "cctv",
    "camera",
    "photograph",
    "photographs",
    "document",
    "receipt",
    "phone records",
    "text messages",
    "email",
    "emails",
}
