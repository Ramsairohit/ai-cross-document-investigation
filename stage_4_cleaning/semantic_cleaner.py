"""
Stage 4: Semantic Cleaning - Main Orchestrator

Orchestrates all semantic cleaning components to transform
Stage 3 parsed output into clean, normalized representation.

This module:
1. Accepts StructuralParseResult from Stage 3
2. Applies encoding normalization
3. Applies whitespace normalization
4. Applies noise removal
5. Normalizes timestamps to ISO-8601
6. Returns CleaningResult

IMPORTANT:
- Deterministic: same input → same output
- Async-safe: no shared mutable state
- No AI, NLP, or inference
- Text is cleaned, NOT understood
"""

from datetime import date
from typing import Any, Optional, Union

from stage_3_parsing.models import StructuralParseResult

from .encoding_fix import fix_encoding
from .models import CleanedBlock, CleaningConfig, CleaningResult
from .noise_removal import remove_noise
from .timestamp_normalizer import normalize_timestamps
from .whitespace_normalizer import normalize_whitespace


class SemanticCleaner:
    """
    Main semantic cleaning orchestrator.

    Transforms Stage 3 parsed blocks into semantically cleaned blocks
    with normalized encoding, whitespace, and timestamps.
    """

    def __init__(self, config: Optional[CleaningConfig] = None) -> None:
        """
        Initialize the semantic cleaner.

        Args:
            config: Optional cleaning configuration. Uses defaults if not provided.
        """
        self.config = config or CleaningConfig()

    def clean(
        self,
        parse_result: Union[StructuralParseResult, dict[str, Any]],
    ) -> CleaningResult:
        """
        Clean a Stage 3 parse result.

        Args:
            parse_result: StructuralParseResult from Stage 3, or a dict with
                         the same schema

        Returns:
            CleaningResult with cleaned blocks
        """
        # Handle dict input
        if isinstance(parse_result, dict):
            document_id = parse_result.get("document_id", "")
            case_id = parse_result.get("case_id", "")
            source_file = parse_result.get("source_file", "")
            parsed_blocks = parse_result.get("parsed_blocks", [])
        else:
            document_id = parse_result.document_id
            case_id = parse_result.case_id
            source_file = parse_result.source_file
            parsed_blocks = [block.model_dump() for block in parse_result.parsed_blocks]

        # Get reference date for timestamp parsing
        reference_date = self.config.reference_date.date() if self.config.reference_date else None

        # Clean all blocks
        cleaned_blocks = self._clean_blocks(parsed_blocks, reference_date)

        return CleaningResult(
            document_id=document_id,
            case_id=case_id,
            source_file=source_file,
            cleaned_blocks=cleaned_blocks,
        )

    def _clean_blocks(
        self,
        parsed_blocks: list[dict[str, Any]],
        reference_date: Optional[date],
    ) -> list[CleanedBlock]:
        """
        Clean all parsed blocks.

        Applies cleaning steps in order:
        1. Encoding normalization
        2. Whitespace normalization
        3. Noise removal
        4. Timestamp normalization

        Args:
            parsed_blocks: List of parsed block dicts from Stage 3
            reference_date: Optional reference date for timestamp parsing

        Returns:
            List of cleaned blocks
        """
        if not parsed_blocks:
            return []

        cleaned_blocks: list[CleanedBlock] = []

        for block in parsed_blocks:
            cleaned_block = self._clean_single_block(block, reference_date)
            cleaned_blocks.append(cleaned_block)

        return cleaned_blocks

    def _clean_single_block(
        self,
        block: dict[str, Any],
        reference_date: Optional[date],
    ) -> CleanedBlock:
        """
        Clean a single parsed block.

        Args:
            block: Parsed block dict from Stage 3
            reference_date: Optional reference date for timestamp parsing

        Returns:
            CleanedBlock with normalized content
        """
        # Extract block data
        block_id = block.get("block_id", "")
        page = block.get("page", 1)
        text = block.get("text", "")
        speaker = block.get("speaker")
        section = block.get("section")
        is_header = block.get("is_header", False)
        is_footer = block.get("is_footer", False)
        raw_timestamps = block.get("raw_timestamps", [])

        # Step 1: Encoding normalization
        clean_text = fix_encoding(text)

        # Step 2: Whitespace normalization
        clean_text = normalize_whitespace(
            clean_text,
            collapse_spaces=self.config.collapse_whitespace,
            normalize_newline_format=self.config.normalize_newlines,
            collapse_newlines=True,
            trim=self.config.trim_whitespace,
        )

        # Step 3: Noise removal
        if self.config.remove_ocr_artifacts:
            clean_text = remove_noise(clean_text, aggressive=False)

        # Step 4: Timestamp normalization
        normalized_ts = normalize_timestamps(raw_timestamps, reference_date)

        return CleanedBlock(
            block_id=block_id,
            page=page,
            clean_text=clean_text,
            speaker=speaker,
            section=section,
            is_header=is_header,
            is_footer=is_footer,
            raw_timestamps=raw_timestamps,
            normalized_timestamps=normalized_ts,
        )

    def clean_text_only(self, text: str) -> str:
        """
        Clean text without creating a full block structure.

        Useful for cleaning individual text snippets.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Apply cleaning steps
        clean_text = fix_encoding(text)

        clean_text = normalize_whitespace(
            clean_text,
            collapse_spaces=self.config.collapse_whitespace,
            normalize_newline_format=self.config.normalize_newlines,
            collapse_newlines=True,
            trim=self.config.trim_whitespace,
        )

        if self.config.remove_ocr_artifacts:
            clean_text = remove_noise(clean_text, aggressive=False)

        return clean_text


async def clean_document(
    parse_result: Union[StructuralParseResult, dict[str, Any]],
    config: Optional[CleaningConfig] = None,
) -> CleaningResult:
    """
    Async-safe document cleaning function.

    Convenience function for cleaning a single document.
    The actual cleaning is CPU-bound and synchronous, but this
    wrapper allows integration with async pipelines.

    Args:
        parse_result: Stage 3 parse result (model or dict)
        config: Optional cleaning configuration

    Returns:
        CleaningResult with cleaned blocks
    """
    cleaner = SemanticCleaner(config=config)
    return cleaner.clean(parse_result)


def clean_document_sync(
    parse_result: Union[StructuralParseResult, dict[str, Any]],
    config: Optional[CleaningConfig] = None,
) -> CleaningResult:
    """
    Synchronous document cleaning function.

    Args:
        parse_result: Stage 3 parse result (model or dict)
        config: Optional cleaning configuration

    Returns:
        CleaningResult with cleaned blocks
    """
    cleaner = SemanticCleaner(config=config)
    return cleaner.clean(parse_result)
