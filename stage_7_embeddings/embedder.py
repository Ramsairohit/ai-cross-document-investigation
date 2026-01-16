"""
Stage 7: Vector Embeddings - Chunk Embedder

Converts chunks to vector embeddings using SentenceTransformer.

CRITICAL RULES:
- Chunk text is NOT modified
- Entity annotations are NOT embedded
- Each chunk processed independently
- Deterministic: same input → same output
"""

from typing import Any, Union

import numpy as np

from stage_6_ner.models import ChunkInput

from .embedding_model import encode_text


def embed_chunk(
    chunk: Union[ChunkInput, dict[str, Any]],
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate deterministic embedding for chunk text.

    CRITICAL: Chunk text is NOT modified.
    Entity annotations are NOT embedded.

    Args:
        chunk: ChunkInput or dict with chunk data.
        normalize: Whether to L2-normalize the embedding.

    Returns:
        Embedding vector as numpy array.
    """
    # Convert dict to ChunkInput if needed
    if isinstance(chunk, dict):
        chunk = ChunkInput(**chunk)

    # Embed the raw text directly - NO MODIFICATION
    text = chunk.text

    # Generate embedding
    embedding = encode_text(text, normalize=normalize)

    return embedding


def embed_chunks(
    chunks: list[Union[ChunkInput, dict[str, Any]]],
    normalize: bool = True,
) -> np.ndarray:
    """
    Generate embeddings for multiple chunks.

    Each chunk is processed INDEPENDENTLY.
    No cross-chunk analysis is performed.

    Args:
        chunks: List of ChunkInput or dicts.
        normalize: Whether to L2-normalize embeddings.

    Returns:
        Array of embeddings (N x dimension).
    """
    embeddings = [embed_chunk(chunk, normalize=normalize) for chunk in chunks]
    return np.array(embeddings)


def extract_metadata(chunk: Union[ChunkInput, dict[str, Any]]) -> dict[str, Any]:
    """
    Extract metadata from chunk for storage.

    Args:
        chunk: ChunkInput or dict with chunk data.

    Returns:
        Metadata dict for VectorStore.
    """
    if isinstance(chunk, dict):
        chunk = ChunkInput(**chunk)

    return {
        "chunk_id": chunk.chunk_id,
        "case_id": chunk.case_id,
        "document_id": chunk.document_id,
        "page_range": chunk.page_range,
        "speaker": chunk.speaker,
        "confidence": chunk.confidence,
    }
