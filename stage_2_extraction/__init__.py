"""
Stage 2: Document Extraction

Forensic-grade document extraction pipeline for AI Police Case Investigation.
Converts evidence files (PDF, DOCX, images) into structured, page-aware JSON output.

This module prepares data for chunking, NLP, timelines, and contradiction detection.
It does NOT analyze, reason, or make decisions about content.

Build it like it will be examined in court.
"""

from .extract_documents import DocumentExtractor, extract_documents
from .models import (
    AuditEvent,
    ContentBlock,
    ContentBlockType,
    ExtractionRequest,
    ExtractionResult,
    ExtractionStatus,
)

__all__ = [
    # Main extractor
    "DocumentExtractor",
    "extract_documents",
    # Data models
    "ExtractionRequest",
    "ExtractionResult",
    "ContentBlock",
    "ContentBlockType",
    "ExtractionStatus",
    "AuditEvent",
]

__version__ = "1.0.0"
