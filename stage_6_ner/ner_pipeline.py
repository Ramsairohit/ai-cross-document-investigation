"""
Stage 6: NER - Pipeline Orchestrator

Main orchestrator for Named Entity Recognition.

This module:
1. Accepts chunks from Stage 5
2. Applies spaCy NER
3. Applies rule-based extraction
4. Merges and returns entities with full provenance

IMPORTANT:
- Deterministic: same input → same output
- Async-safe: no shared mutable state
- No cross-chunk analysis
- Each chunk processed independently
"""

from typing import Any, Sequence, Union

from .entity_extractor import extract_entities
from .models import ChunkInput, NERResult


class NERPipeline:
    """
    Main NER pipeline orchestrator.

    Processes chunks independently and extracts entities
    with full provenance tracking.
    """

    def __init__(self) -> None:
        """Initialize the NER pipeline."""
        # Pre-load spaCy model on first use
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Ensure the spaCy model is loaded."""
        if not self._model_loaded:
            from .spacy_loader import get_spacy_model

            get_spacy_model()  # Trigger loading
            self._model_loaded = True

    def process_chunk(
        self,
        chunk: Union[ChunkInput, dict[str, Any]],
    ) -> NERResult:
        """
        Process a single chunk and extract entities.

        Args:
            chunk: ChunkInput or dict with chunk data.

        Returns:
            NERResult with extracted entities.
        """
        self._ensure_model_loaded()
        return extract_entities(chunk)

    def process_chunks(
        self,
        chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
    ) -> list[NERResult]:
        """
        Process multiple chunks and extract entities.

        Each chunk is processed INDEPENDENTLY.
        No cross-chunk analysis is performed.

        Args:
            chunks: List of chunks to process.

        Returns:
            List of NERResult objects, one per chunk.
        """
        self._ensure_model_loaded()
        return [extract_entities(chunk) for chunk in chunks]


async def process_chunk_async(
    chunk: Union[ChunkInput, dict[str, Any]],
) -> NERResult:
    """
    Async-safe chunk processing.

    The actual NER is CPU-bound and synchronous, but this
    wrapper allows integration with async pipelines.

    Args:
        chunk: ChunkInput or dict with chunk data.

    Returns:
        NERResult with extracted entities.
    """
    pipeline = NERPipeline()
    return pipeline.process_chunk(chunk)


def process_chunk_sync(
    chunk: Union[ChunkInput, dict[str, Any]],
) -> NERResult:
    """
    Synchronous chunk processing.

    Args:
        chunk: ChunkInput or dict with chunk data.

    Returns:
        NERResult with extracted entities.
    """
    pipeline = NERPipeline()
    return pipeline.process_chunk(chunk)


async def process_chunks_async(
    chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
) -> list[NERResult]:
    """
    Async-safe batch chunk processing.

    Args:
        chunks: List of chunks to process.

    Returns:
        List of NERResult objects.
    """
    pipeline = NERPipeline()
    return pipeline.process_chunks(chunks)


def process_chunks_sync(
    chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
) -> list[NERResult]:
    """
    Synchronous batch chunk processing.

    Args:
        chunks: List of chunks to process.

    Returns:
        List of NERResult objects.
    """
    pipeline = NERPipeline()
    return pipeline.process_chunks(chunks)
