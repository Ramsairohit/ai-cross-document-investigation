"""
Stage 4: Semantic Cleaning - Data Models

Pydantic models for forensic-grade semantic cleaning.
These models define the exact schema for cleaned blocks.

IMPORTANT: This stage cleans text, it does NOT understand it.
No summarization, no inference, no NLP.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class NormalizedTimestamp(BaseModel):
    """
    A timestamp normalized to ISO-8601 format.

    Contains both the original raw string and the normalized ISO representation.
    If parsing is ambiguous, iso will be None with low confidence.
    """

    original: str = Field(..., description="Original timestamp string exactly as found in text")
    iso: Optional[str] = Field(
        default=None,
        description="ISO-8601 formatted timestamp (YYYY-MM-DDTHH:MM:SS) or null if ambiguous",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the normalization (0.0-1.0)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "original": "8:15 PM",
                "iso": "2024-03-15T20:15:00",
                "confidence": 0.90,
            }
        }


class CleanedBlock(BaseModel):
    """
    Semantically cleaned content block.

    Contains the cleaned text along with all structural metadata
    preserved from Stage 3, plus normalized timestamps.
    """

    block_id: str = Field(..., description="Original block ID from earlier stages")
    page: int = Field(..., ge=1, description="1-indexed page number")
    clean_text: str = Field(..., description="Semantically cleaned text content")
    speaker: Optional[str] = Field(default=None, description="Detected speaker label from Stage 3")
    section: Optional[str] = Field(default=None, description="Section name from Stage 3")
    is_header: bool = Field(default=False, description="True if block is a header")
    is_footer: bool = Field(default=False, description="True if block is a footer")
    raw_timestamps: list[str] = Field(
        default_factory=list,
        description="Raw timestamp strings exactly as found in text",
    )
    normalized_timestamps: list[NormalizedTimestamp] = Field(
        default_factory=list,
        description="Timestamps normalized to ISO-8601 format with confidence scores",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "block_id": "b1",
                "page": 1,
                "clean_text": "I heard a loud crash at 8:15 PM",
                "speaker": "WITNESS",
                "section": "STATEMENT",
                "is_header": False,
                "is_footer": False,
                "raw_timestamps": ["8:15 PM"],
                "normalized_timestamps": [
                    {
                        "original": "8:15 PM",
                        "iso": "2024-03-15T20:15:00",
                        "confidence": 0.90,
                    }
                ],
            }
        }


class CleaningResult(BaseModel):
    """
    Complete semantically cleaned document.

    This is the MANDATORY output schema for Stage 4 cleaning.
    One CleaningResult is produced per input document.
    """

    document_id: str = Field(..., description="Document ID from input")
    case_id: str = Field(..., description="Case ID from input")
    source_file: str = Field(..., description="Original source filename")
    cleaned_blocks: list[CleanedBlock] = Field(
        default_factory=list, description="Ordered list of semantically cleaned blocks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "cleaned_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "clean_text": "I heard a loud crash at 8:15 PM",
                        "speaker": "WITNESS",
                        "section": "STATEMENT",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                        "normalized_timestamps": [
                            {
                                "original": "8:15 PM",
                                "iso": "2024-03-15T20:15:00",
                                "confidence": 0.90,
                            }
                        ],
                    }
                ],
            }
        }


class CleaningConfig(BaseModel):
    """
    Configuration for semantic cleaning behavior.

    Allows customization of cleaning rules while maintaining
    deterministic behavior. All transformations must be reversible
    in terms of meaning preservation.
    """

    # Reference date for timestamp parsing (if known from document metadata)
    reference_date: Optional[datetime] = Field(
        default=None,
        description="Reference date for parsing time-only timestamps (e.g., document date)",
    )

    # Noise removal settings
    remove_ocr_artifacts: bool = Field(
        default=True, description="Remove common OCR artifact characters"
    )

    # Whitespace settings
    collapse_whitespace: bool = Field(default=True, description="Collapse multiple spaces into one")
    normalize_newlines: bool = Field(
        default=True, description="Normalize different newline formats to \\n"
    )
    trim_whitespace: bool = Field(default=True, description="Remove leading/trailing whitespace")

    class Config:
        json_schema_extra = {
            "example": {
                "reference_date": "2024-03-15T00:00:00",
                "remove_ocr_artifacts": True,
                "collapse_whitespace": True,
                "normalize_newlines": True,
                "trim_whitespace": True,
            }
        }
