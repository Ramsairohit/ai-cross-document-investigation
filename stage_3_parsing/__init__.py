"""
Stage 3: Structural Parsing

Forensic-grade structural parsing pipeline for AI Police Case Investigation.
Transforms Stage 2 extraction output into form-aware parsed blocks.

This module identifies:
- Headers and footers
- Speaker labels
- Section boundaries
- Raw timestamp mentions

This stage understands document FORM, not meaning.
It prepares text so later stages can reason safely.

Build it like it will be cross-examined in court.
"""

from .header_footer import HeaderFooterResult, detect_headers_footers
from .models import ParsedBlock, ParsingConfig, StructuralParseResult
from .section_detection import (
    SectionDetectionResult,
    SectionTracker,
    assign_sections,
    detect_section,
)
from .speaker_detection import (
    SpeakerDetectionResult,
    detect_speaker,
    normalize_speaker_name,
)
from .structural_parser import (
    StructuralParser,
    parse_document,
    parse_document_sync,
)
from .timestamp_regex import extract_timestamps, extract_timestamps_with_positions

__all__ = [
    # Main parser
    "StructuralParser",
    "parse_document",
    "parse_document_sync",
    # Data models
    "ParsedBlock",
    "StructuralParseResult",
    "ParsingConfig",
    # Header/footer detection
    "detect_headers_footers",
    "HeaderFooterResult",
    # Speaker detection
    "detect_speaker",
    "normalize_speaker_name",
    "SpeakerDetectionResult",
    # Section detection
    "detect_section",
    "assign_sections",
    "SectionTracker",
    "SectionDetectionResult",
    # Timestamp extraction
    "extract_timestamps",
    "extract_timestamps_with_positions",
]

__version__ = "1.0.0"
