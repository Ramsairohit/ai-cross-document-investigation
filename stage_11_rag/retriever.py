"""
Stage 11: RAG - Vector Retriever

Retrieves relevant chunks using FAISS vector search.

IMPORTANT:
- Uses same embedding model as Stage 7
- Preserves chunk metadata
- No ranking explanation or filtering by interpretation
"""

from typing import Any, Union

import numpy as np

from .models import RAGConfig, RetrievedChunk


def search_index(
    index: Any,
    query_vector: np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for nearest neighbors.

    Args:
        index: FAISS index object.
        query_vector: Query embedding vector.
        k: Number of results to return.

    Returns:
        Tuple of (distances, indices).
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    query_vector = query_vector.astype(np.float32)

    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def retrieve_chunks(
    query_embedding: np.ndarray,
    index: Any,
    chunk_metadata: list[dict[str, Any]],
    config: RAGConfig | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve relevant chunks using vector similarity.

    Args:
        query_embedding: Embedded query vector.
        index: FAISS index.
        chunk_metadata: List of chunk metadata dicts (same order as index).
        config: RAG configuration.

    Returns:
        List of retrieved chunks with scores.
    """
    config = config or RAGConfig()

    # Search index
    distances, indices = search_index(index, query_embedding, config.top_k)

    # Build results
    results: list[RetrievedChunk] = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if idx < 0:  # FAISS returns -1 for missing results
            continue

        if idx >= len(chunk_metadata):
            continue

        meta = chunk_metadata[idx]

        # Convert distance to score (L2 distance -> similarity)
        # Lower distance = higher similarity
        score = 1.0 / (1.0 + float(dist))

        if score < config.min_score:
            continue

        chunk = RetrievedChunk(
            chunk_id=meta.get("chunk_id", f"CHUNK_{idx}"),
            document_id=meta.get("document_id", ""),
            case_id=meta.get("case_id", ""),
            page_range=meta.get("page_range", [1, 1]),
            text=meta.get("text", ""),
            speaker=meta.get("speaker"),
            score=score,
            confidence=meta.get("confidence", 1.0),
        )
        results.append(chunk)

    return results


def embed_query(
    question: str,
    embedder_fn: callable,
) -> np.ndarray:
    """
    Embed investigator question using same model as chunks.

    Args:
        question: Investigator question text.
        embedder_fn: Function to generate embeddings.

    Returns:
        Query embedding vector.
    """
    return embedder_fn(question)


def filter_by_case(
    chunks: list[RetrievedChunk],
    case_id: str,
) -> list[RetrievedChunk]:
    """
    Filter chunks to only include those from the specified case.

    CRITICAL: No cross-case access allowed.

    Args:
        chunks: Retrieved chunks.
        case_id: Case to filter for.

    Returns:
        Filtered chunks from the specified case only.
    """
    return [c for c in chunks if c.case_id == case_id]


def chunks_to_context(chunks: list[RetrievedChunk]) -> str:
    """
    Convert retrieved chunks to context string for LLM.

    Args:
        chunks: Retrieved chunks.

    Returns:
        Formatted context string with citations.
    """
    if not chunks:
        return "No relevant evidence found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        speaker_info = f" (Speaker: {chunk.speaker})" if chunk.speaker else ""
        context_parts.append(f"[Source {i}: {chunk.chunk_id}]{speaker_info}\n{chunk.text}\n")

    return "\n".join(context_parts)
