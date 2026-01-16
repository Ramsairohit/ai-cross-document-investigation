"""
Stage 2: Document Extraction - Data Models

Pydantic models for forensic-grade document extraction.
These models define the exact schema for inputs and outputs.

IMPORTANT: These models preserve evidence verbatim.
No summarization, grammar correction, or interpretation.
"""

from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ExtractionStatus(str, Enum):
    """Document extraction status based on confidence thresholds."""

    ACCEPTED = "ACCEPTED"
    FLAGGED_FOR_REVIEW = "FLAGGED_FOR_REVIEW"
    REJECTED = "REJECTED"


class ContentBlockType(str, Enum):
    """Types of content blocks extracted from documents."""

    paragraph = "paragraph"
    heading = "heading"
    table = "table"
    list_item = "list_item"
    caption = "caption"
    footnote = "footnote"
    code = "code"
    formula = "formula"
    unknown = "unknown"


class ContentBlock(BaseModel):
    """
    Individual content block extracted from a document.

    Each block represents a discrete piece of content with its
    type, text, page location, and extraction confidence.
    """

    block_id: str = Field(..., description="Unique identifier for this block (e.g., 'b1', 'b2')")
    type: str = Field(..., description="Content type: paragraph, heading, table, list_item, etc.")
    text: str = Field(..., description="Verbatim extracted text - NOT modified or corrected")
    page: int = Field(..., ge=1, description="1-indexed page number where this block appears")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence score (0.0 to 1.0)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "block_id": "b1",
                "type": "paragraph",
                "text": "I heard a loud crash around 8:15 PM...",
                "page": 1,
                "confidence": 0.94,
            }
        }


class ExtractionResult(BaseModel):
    """
    Structured output for a single extracted document.

    This is the MANDATORY output schema for Stage 2 extraction.
    One ExtractionResult is produced per input document.
    """

    document_id: str = Field(..., description="Auto-generated unique identifier for this document")
    case_id: str = Field(..., description="Case identifier this document belongs to")
    source_file: str = Field(..., description="Original filename of the source document")
    pages: int = Field(..., ge=1, description="Total number of pages in the document")
    extraction_method: Literal["docling", "docling+ocr"] = Field(
        ..., description="Extraction method used: 'docling' or 'docling+ocr'"
    )
    content_blocks: list[ContentBlock] = Field(
        default_factory=list, description="Ordered list of extracted content blocks"
    )
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall document extraction confidence (0.0 to 1.0)"
    )
    status: ExtractionStatus = Field(
        ..., description="Extraction status: ACCEPTED, FLAGGED_FOR_REVIEW, or REJECTED"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "case_id": "24-890-H",
                "source_file": "witness_statement_01.pdf",
                "pages": 3,
                "extraction_method": "docling",
                "content_blocks": [
                    {
                        "block_id": "b1",
                        "type": "paragraph",
                        "text": "I heard a loud crash around 8:15 PM...",
                        "page": 1,
                        "confidence": 0.94,
                    }
                ],
                "overall_confidence": 0.91,
                "status": "ACCEPTED",
            }
        }


class ExtractionRequest(BaseModel):
    """
    Input request for document extraction.

    Accepts a case ID and list of file paths to process.
    """

    case_id: str = Field(..., description="Case identifier for chain-of-custody tracking")
    uploaded_files: list[Union[str, Path]] = Field(
        ..., min_length=1, description="List of file paths to process"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "uploaded_files": [
                    "evidence/witness_statement_01.pdf",
                    "evidence/crime_scene_photo.jpg",
                ],
            }
        }


class AuditEvent(BaseModel):
    """
    Chain-of-custody audit log event.

    Every extraction operation MUST be logged with this structure
    for forensic traceability and legal admissibility.
    """

    event: Literal["DOCUMENT_EXTRACTED", "DOCUMENT_UPLOADED"] = Field(
        ..., description="Event type"
    )
    tool: Optional[str] = Field(
        default=None, description="Tool used for extraction (e.g., 'docling')"
    )
    tool_version: Optional[str] = Field(default=None, description="Version of the extraction tool")
    case_id: str = Field(..., description="Case identifier")
    document_id: str = Field(..., description="Generated document identifier")
    input_hash: str = Field(..., description="SHA256 hash of input file (format: 'sha256:xxxx')")
    output_hash: Optional[str] = Field(
        default=None, description="SHA256 hash of extracted content (format: 'sha256:xxxx')"
    )
    timestamp: str = Field(..., description="ISO-8601 formatted timestamp")
    operator: str = Field(default="SYSTEM", description="Operator who initiated the operation")
    original_filename: Optional[str] = Field(
        default=None, description="Original filename of uploaded document"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "event": "DOCUMENT_EXTRACTED",
                    "tool": "docling",
                    "tool_version": "2.67.0",
                    "case_id": "24-890-H",
                    "document_id": "doc-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "input_hash": "sha256:a1b2c3d4e5f6...",
                    "output_hash": "sha256:f6e5d4c3b2a1...",
                    "timestamp": "2026-01-10T21:56:24+05:30",
                    "operator": "SYSTEM",
                },
                {
                    "event": "DOCUMENT_UPLOADED",
                    "tool": None,
                    "tool_version": None,
                    "case_id": "24-890-H",
                    "document_id": "doc-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                    "input_hash": "sha256:a1b2c3d4e5f6...",
                    "output_hash": None,
                    "timestamp": "2026-01-10T21:56:24+05:30",
                    "operator": "SYSTEM",
                    "original_filename": "witness.pdf",
                },
            ]
        }


class DoclingPageResult(BaseModel):
    """
    Internal model for Docling page-level extraction results.

    Used internally to track which pages need OCR fallback.
    """

    page_number: int = Field(..., ge=1)
    has_text_layer: bool = Field(default=True)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    ocr_confidence: Optional[float] = Field(default=None)
    needs_ocr: bool = Field(default=False)
