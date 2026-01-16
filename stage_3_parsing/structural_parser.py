"""
Stage 3: Structural Parsing - Main Orchestrator

Orchestrates all structural parsing components to transform
Stage 2 extraction output into form-aware parsed blocks.

This module:
1. Accepts ExtractionResult from Stage 2
2. Applies header/footer detection
3. Extracts speaker labels
4. Detects section boundaries
5. Extracts raw timestamps
6. Returns StructuralParseResult

IMPORTANT:
- Deterministic: same input → same output
- Async-safe: no shared mutable state
- No AI, NLP, or inference
"""

from typing import Any, Optional

from .header_footer import HeaderFooterResult, detect_headers_footers
from .models import ParsedBlock, ParsingConfig, StructuralParseResult
from .section_detection import assign_sections
from .speaker_detection import SpeakerDetectionResult, detect_speaker
from .timestamp_regex import extract_timestamps


class StructuralParser:
    """
    Main structural parsing orchestrator.

    Transforms Stage 2 content blocks into structurally parsed blocks
    with speaker labels, sections, timestamps, and header/footer flags.
    """

    def __init__(self, config: Optional[ParsingConfig] = None) -> None:
        """
        Initialize the structural parser.

        Args:
            config: Optional parsing configuration. Uses defaults if not provided.
        """
        self.config = config or ParsingConfig()

    def parse(self, extraction_result: dict[str, Any]) -> StructuralParseResult:
        """
        Parse a Stage 2 extraction result into structured blocks.

        Args:
            extraction_result: Dictionary containing:
                - document_id: str
                - case_id: str
                - source_file: str
                - content_blocks: list of block dicts

        Returns:
            StructuralParseResult with parsed blocks
        """
        document_id = extraction_result.get("document_id", "")
        case_id = extraction_result.get("case_id", "")
        source_file = extraction_result.get("source_file", "")
        content_blocks = extraction_result.get("content_blocks", [])

        # Parse blocks
        parsed_blocks = self._parse_blocks(content_blocks)

        return StructuralParseResult(
            document_id=document_id,
            case_id=case_id,
            source_file=source_file,
            parsed_blocks=parsed_blocks,
        )

    def _parse_blocks(self, content_blocks: list[dict[str, Any]]) -> list[ParsedBlock]:
        """
        Parse all content blocks.

        Applies detection modules in order:
        1. Header/footer detection (requires all blocks)
        2. Section assignment (requires all blocks, sequential)
        3. Speaker detection (per-block)
        4. Timestamp extraction (per-block)

        Args:
            content_blocks: List of Stage 2 content blocks

        Returns:
            List of parsed blocks
        """
        if not content_blocks:
            return []

        # Step 1: Detect headers and footers (needs all blocks)
        header_footer_results = detect_headers_footers(
            content_blocks,
            min_page_repetition=self.config.min_page_repetition,
        )

        # Step 2: Assign sections (needs all blocks, sequential)
        section_assignments = assign_sections(
            content_blocks,
            max_section_header_length=self.config.max_section_header_length,
        )

        # Step 3 & 4: Process each block individually
        parsed_blocks: list[ParsedBlock] = []

        for block in content_blocks:
            block_id = block.get("block_id", "")
            page = block.get("page", 1)
            text = block.get("text", "")

            # Get header/footer status
            hf_result = header_footer_results.get(
                block_id,
                HeaderFooterResult(is_header=False, is_footer=False),
            )

            # Get section assignment
            section = section_assignments.get(block_id)

            # Detect speaker and clean text
            speaker_result = detect_speaker(text)

            # Extract timestamps from cleaned text
            raw_timestamps = extract_timestamps(speaker_result.cleaned_text)

            parsed_blocks.append(
                ParsedBlock(
                    block_id=block_id,
                    page=page,
                    text=speaker_result.cleaned_text,
                    speaker=speaker_result.speaker,
                    is_header=hf_result.is_header,
                    is_footer=hf_result.is_footer,
                    section=section,
                    raw_timestamps=raw_timestamps,
                )
            )

        return parsed_blocks


async def parse_document(
    extraction_result: dict[str, Any],
    config: Optional[ParsingConfig] = None,
) -> StructuralParseResult:
    """
    Async-safe document parsing function.

    Convenience function for parsing a single document.
    The actual parsing is CPU-bound and synchronous, but this
    wrapper allows integration with async pipelines.

    Args:
        extraction_result: Stage 2 extraction result dictionary
        config: Optional parsing configuration

    Returns:
        StructuralParseResult with parsed blocks
    """
    parser = StructuralParser(config=config)
    return parser.parse(extraction_result)


def parse_document_sync(
    extraction_result: dict[str, Any],
    config: Optional[ParsingConfig] = None,
) -> StructuralParseResult:
    """
    Synchronous document parsing function.

    Args:
        extraction_result: Stage 2 extraction result dictionary
        config: Optional parsing configuration

    Returns:
        StructuralParseResult with parsed blocks
    """
    parser = StructuralParser(config=config)
    return parser.parse(extraction_result)
