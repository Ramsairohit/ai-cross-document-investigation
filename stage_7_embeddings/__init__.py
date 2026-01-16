"""
Stage 7: Vector Embeddings

Forensic-grade vector embedding generation for immutable text chunks.

This stage:
- Generates semantic vector embeddings using SentenceTransformer
- Stores vectors in FAISS index with full provenance metadata
- Enables future retrieval (no reasoning, ranking, or interpretation)

CRITICAL RULES:
- Chunk text is NEVER modified
- Entity annotations are NOT embedded
- Chunks processed independently (no cross-chunk analysis)
- Deterministic: same input → same output
- Court-auditable artifacts
"""

from .embedding_model import (
    EmbeddingModelLoader,
    encode_text,
    get_embedding_dimension,
    get_embedding_model,
    is_model_loaded,
)
from .embedding_pipeline import (
    EmbeddingPipeline,
    embed_chunk_async,
    embed_chunk_sync,
    embed_chunks_async,
    embed_chunks_sync,
)
from .faiss_index import FAISSIndexManager
from .models import EmbeddingConfig, EmbeddingResult, VectorRecord
from .vector_store import VectorStore

__all__ = [
    # Pipeline
    "EmbeddingPipeline",
    "embed_chunk_sync",
    "embed_chunks_sync",
    "embed_chunk_async",
    "embed_chunks_async",
    # Models
    "VectorRecord",
    "EmbeddingResult",
    "EmbeddingConfig",
    # Components
    "VectorStore",
    "FAISSIndexManager",
    "EmbeddingModelLoader",
    # Utilities
    "get_embedding_model",
    "encode_text",
    "get_embedding_dimension",
    "is_model_loaded",
]
