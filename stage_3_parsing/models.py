"""
Stage 3: Structural Parsing - Data Models

Pydantic models for forensic-grade structural parsing.
These models define the exact schema for parsed blocks.

IMPORTANT: This stage identifies form, not meaning.
No summarization, interpretation, or inference.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ParsedBlock(BaseModel):
    """
    Structurally parsed content block.

    Contains the original block data plus structural annotations:
    speaker labels, section assignments, header/footer flags, and raw timestamps.
    """

    block_id: str = Field(..., description="Original block ID from Stage 2")
    page: int = Field(..., ge=1, description="1-indexed page number")
    text: str = Field(..., description="Text content with speaker label stripped if present")
    speaker: Optional[str] = Field(
        default=None, description="Detected speaker label (e.g., 'DET. SMITH') or null"
    )
    is_header: bool = Field(default=False, description="True if block is detected as a header")
    is_footer: bool = Field(default=False, description="True if block is detected as a footer")
    section: Optional[str] = Field(
        default=None, description="Current section name (e.g., 'INTERVIEW', 'STATEMENT')"
    )
    raw_timestamps: list[str] = Field(
        default_factory=list,
        description="Raw timestamp strings exactly as found in text (NOT normalized)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "block_id": "b1",
                "page": 1,
                "text": "Where were you on the night of March 15?",
                "speaker": "DET. SMITH",
                "is_header": False,
                "is_footer": False,
                "section": "INTERVIEW",
                "raw_timestamps": ["March 15"],
            }
        }


class StructuralParseResult(BaseModel):
    """
    Complete structurally parsed document.

    This is the MANDATORY output schema for Stage 3 parsing.
    One StructuralParseResult is produced per input document.
    """

    document_id: str = Field(..., description="Document ID from Stage 2 input")
    case_id: str = Field(..., description="Case ID from Stage 2 input")
    source_file: str = Field(..., description="Original source filename")
    parsed_blocks: list[ParsedBlock] = Field(
        default_factory=list, description="Ordered list of structurally parsed blocks"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "Where were you on the night of March 15?",
                        "speaker": "DET. SMITH",
                        "is_header": False,
                        "is_footer": False,
                        "section": "INTERVIEW",
                        "raw_timestamps": ["March 15"],
                    }
                ],
            }
        }


class ParsingConfig(BaseModel):
    """
    Configuration for structural parsing rules.

    Allows customization of detection patterns and thresholds
    while maintaining deterministic behavior.
    """

    # Header/footer detection
    min_page_repetition: int = Field(
        default=2,
        ge=2,
        description="Minimum pages a text must appear on to be considered header/footer",
    )

    # Section detection
    max_section_header_length: int = Field(
        default=50, ge=10, description="Maximum character length for section headers"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "min_page_repetition": 2,
                "max_section_header_length": 50,
            }
        }
