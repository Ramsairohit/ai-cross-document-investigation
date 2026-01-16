"""
Stage 5: Logical Chunking - Pipeline Orchestrator

Main orchestrator for logical chunking.

This module:
1. Accepts CleaningResult from Stage 4
2. Applies chunking rules (page/speaker boundaries)
3. Produces ChunkingResult for Stage 6 & 7

IMPORTANT:
- Deterministic: same input → same output
- Async-safe: no shared mutable state
- Each document processed independently
"""

from typing import Any

from .chunker import chunk_blocks
from .models import BlockInput, Chunk, ChunkingConfig, ChunkingResult


class ChunkingPipeline:
    """
    Main chunking pipeline orchestrator.

    Processes documents and produces meaning-preserving chunks
    with full provenance tracking.
    """

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        """
        Initialize the chunking pipeline.

        Args:
            config: Chunking configuration. Uses defaults if None.
        """
        self._config = config or ChunkingConfig()

    @property
    def config(self) -> ChunkingConfig:
        """Get the current configuration."""
        return self._config

    def process_document(
        self,
        document_id: str,
        case_id: str,
        source_file: str,
        blocks: list[BlockInput | dict[str, Any]],
    ) -> ChunkingResult:
        """
        Process a document and produce chunks.

        Args:
            document_id: Document identifier.
            case_id: Case identifier.
            source_file: Original source filename.
            blocks: List of cleaned blocks from Stage 4.

        Returns:
            ChunkingResult with all chunks.
        """
        chunks = chunk_blocks(
            blocks=blocks,
            case_id=case_id,
            document_id=document_id,
            config=self._config,
        )

        return ChunkingResult(
            document_id=document_id,
            case_id=case_id,
            source_file=source_file,
            chunks=chunks,
            total_chunks=len(chunks),
            total_blocks_processed=len(blocks),
        )

    def process_cleaning_result(
        self,
        cleaning_result: dict[str, Any],
    ) -> ChunkingResult:
        """
        Process a Stage 4 CleaningResult directly.

        Args:
            cleaning_result: CleaningResult dict from Stage 4.

        Returns:
            ChunkingResult with all chunks.
        """
        return self.process_document(
            document_id=cleaning_result["document_id"],
            case_id=cleaning_result["case_id"],
            source_file=cleaning_result["source_file"],
            blocks=cleaning_result.get("cleaned_blocks", []),
        )


def process_document_sync(
    document_id: str,
    case_id: str,
    source_file: str,
    blocks: list[BlockInput | dict[str, Any]],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Synchronous document processing.

    Args:
        document_id: Document identifier.
        case_id: Case identifier.
        source_file: Original source filename.
        blocks: List of cleaned blocks from Stage 4.
        config: Chunking configuration.

    Returns:
        ChunkingResult with all chunks.
    """
    pipeline = ChunkingPipeline(config=config)
    return pipeline.process_document(
        document_id=document_id,
        case_id=case_id,
        source_file=source_file,
        blocks=blocks,
    )


async def process_document_async(
    document_id: str,
    case_id: str,
    source_file: str,
    blocks: list[BlockInput | dict[str, Any]],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Async-safe document processing.

    The actual chunking is CPU-bound and synchronous, but this
    wrapper allows integration with async pipelines.

    Args:
        document_id: Document identifier.
        case_id: Case identifier.
        source_file: Original source filename.
        blocks: List of cleaned blocks from Stage 4.
        config: Chunking configuration.

    Returns:
        ChunkingResult with all chunks.
    """
    # Chunking is pure computation, safe to run directly
    return process_document_sync(
        document_id=document_id,
        case_id=case_id,
        source_file=source_file,
        blocks=blocks,
        config=config,
    )


def process_cleaning_result_sync(
    cleaning_result: dict[str, Any],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Process Stage 4 CleaningResult synchronously.

    Args:
        cleaning_result: CleaningResult dict from Stage 4.
        config: Chunking configuration.

    Returns:
        ChunkingResult with all chunks.
    """
    pipeline = ChunkingPipeline(config=config)
    return pipeline.process_cleaning_result(cleaning_result)


async def process_cleaning_result_async(
    cleaning_result: dict[str, Any],
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """
    Process Stage 4 CleaningResult asynchronously.

    Args:
        cleaning_result: CleaningResult dict from Stage 4.
        config: Chunking configuration.

    Returns:
        ChunkingResult with all chunks.
    """
    return process_cleaning_result_sync(cleaning_result, config)
