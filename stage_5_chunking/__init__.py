"""
Stage 5: Logical Chunking

Convert Stage 4 (Semantic Cleaning) output into meaning-preserving
chunks for Stage 6 (NER) and Stage 7 (Embeddings).

IMPORTANT: This stage defines meaning boundaries only.
No NLP, no inference, no intelligence.

Pipeline Position:
    Stage 4 (Semantic Cleaning)
            ↓
    STAGE 5 (Logical Chunking)   ← THIS MODULE
            ↓
    Stage 6 (NER)
    Stage 7 (Embeddings)

Usage:
    from stage_5_chunking import ChunkingPipeline, process_document_sync

    # Using pipeline class
    pipeline = ChunkingPipeline()
    result = pipeline.process_document(
        document_id="DOC123",
        case_id="24-890-H",
        source_file="statement.pdf",
        blocks=[...],  # From Stage 4
    )

    # Using function directly
    result = process_document_sync(
        document_id="DOC123",
        case_id="24-890-H",
        source_file="statement.pdf",
        blocks=[...],
    )

    # Process Stage 4 CleaningResult directly
    from stage_5_chunking import process_cleaning_result_sync
    result = process_cleaning_result_sync(cleaning_result_dict)
"""

from .chunker import chunk_blocks
from .chunking_pipeline import (
    ChunkingPipeline,
    process_cleaning_result_async,
    process_cleaning_result_sync,
    process_document_async,
    process_document_sync,
)
from .confidence import aggregate_confidence, compute_chunk_confidence
from .models import BlockInput, Chunk, ChunkingConfig, ChunkingResult
from .tokenizer import count_tokens, split_text_by_tokens

__all__ = [
    # Pipeline
    "ChunkingPipeline",
    "process_document_sync",
    "process_document_async",
    "process_cleaning_result_sync",
    "process_cleaning_result_async",
    # Core functions
    "chunk_blocks",
    "count_tokens",
    "split_text_by_tokens",
    "aggregate_confidence",
    "compute_chunk_confidence",
    # Models
    "BlockInput",
    "Chunk",
    "ChunkingConfig",
    "ChunkingResult",
]
