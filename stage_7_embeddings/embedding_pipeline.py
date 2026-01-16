"""
Stage 7: Vector Embeddings - Pipeline Orchestrator

Main orchestrator for vector embedding generation and storage.

This module:
1. Accepts chunks from Stage 5 (after Stage 6 annotation)
2. Generates embeddings using SentenceTransformer
3. Stores vectors in FAISS index with metadata

IMPORTANT:
- Deterministic: same input → same output
- Async-safe: no shared mutable state
- No cross-chunk analysis
- Each chunk processed independently
- Chunks are processed AFTER Stage 6 is complete
"""

from pathlib import Path
from typing import Any, Sequence, Union

from stage_6_ner.models import ChunkInput

from .embedder import embed_chunk, extract_metadata
from .embedding_model import get_embedding_dimension
from .models import EmbeddingResult
from .vector_store import VectorStore


class EmbeddingPipeline:
    """
    Main embedding pipeline orchestrator.

    Processes chunks independently and stores embeddings
    with full provenance tracking.
    """

    def __init__(
        self,
        storage_dir: Path,
        index_type: str = "Flat",
    ):
        """
        Initialize the embedding pipeline.

        Args:
            storage_dir: Directory to persist vectors and metadata.
            index_type: FAISS index type ('Flat' or 'IVF').
        """
        self.storage_dir = Path(storage_dir)
        self.store = VectorStore(
            storage_dir=self.storage_dir,
            dimension=get_embedding_dimension(),
            index_type=index_type,
        )
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Ensure the embedding model is loaded."""
        if not self._model_loaded:
            from .embedding_model import get_embedding_model

            get_embedding_model()  # Trigger loading
            self._model_loaded = True

    def process_chunk(
        self,
        chunk: Union[ChunkInput, dict[str, Any]],
    ) -> EmbeddingResult:
        """
        Process a single chunk and store its embedding.

        Args:
            chunk: ChunkInput or dict with chunk data.

        Returns:
            EmbeddingResult with vector_id and status.
        """
        self._ensure_model_loaded()

        # Convert dict to ChunkInput if needed
        if isinstance(chunk, dict):
            chunk = ChunkInput(**chunk)

        # Generate embedding
        embedding = embed_chunk(chunk)

        # Extract metadata
        metadata = extract_metadata(chunk)

        # Store in vector store
        vector_id = self.store.add(
            chunk_id=metadata["chunk_id"],
            vector=embedding,
            case_id=metadata["case_id"],
            document_id=metadata["document_id"],
            page_range=metadata["page_range"],
            speaker=metadata.get("speaker"),
            confidence=metadata.get("confidence", 1.0),
        )

        return EmbeddingResult(
            chunk_id=chunk.chunk_id,
            vector_id=vector_id,
            embedding_dimension=len(embedding),
            success=True,
        )

    def process_chunks(
        self,
        chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
    ) -> list[EmbeddingResult]:
        """
        Process multiple chunks and store their embeddings.

        Each chunk is processed INDEPENDENTLY.
        No cross-chunk analysis is performed.

        Args:
            chunks: List of chunks to process.

        Returns:
            List of EmbeddingResult objects, one per chunk.
        """
        self._ensure_model_loaded()
        return [self.process_chunk(chunk) for chunk in chunks]

    def save(self) -> None:
        """
        Persist all vectors and metadata to disk.

        Creates:
        - faiss.index: The FAISS index file
        - metadata.json: JSON file with all VectorRecords
        """
        self.store.save()

    def load(self) -> None:
        """
        Load existing vectors and metadata from disk.

        Raises:
            FileNotFoundError: If storage files don't exist.
        """
        self.store.load()

    def get_vector_count(self) -> int:
        """Get the number of vectors stored."""
        return self.store.get_vector_count()


async def embed_chunk_async(
    chunk: Union[ChunkInput, dict[str, Any]],
    storage_dir: Path,
) -> EmbeddingResult:
    """
    Async-safe chunk embedding.

    The actual embedding is CPU-bound and synchronous, but this
    wrapper allows integration with async pipelines.

    Args:
        chunk: ChunkInput or dict with chunk data.
        storage_dir: Directory to persist vectors.

    Returns:
        EmbeddingResult with vector_id and status.
    """
    pipeline = EmbeddingPipeline(storage_dir)
    return pipeline.process_chunk(chunk)


def embed_chunk_sync(
    chunk: Union[ChunkInput, dict[str, Any]],
    storage_dir: Path,
) -> EmbeddingResult:
    """
    Synchronous chunk embedding.

    Args:
        chunk: ChunkInput or dict with chunk data.
        storage_dir: Directory to persist vectors.

    Returns:
        EmbeddingResult with vector_id and status.
    """
    pipeline = EmbeddingPipeline(storage_dir)
    return pipeline.process_chunk(chunk)


async def embed_chunks_async(
    chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
    storage_dir: Path,
) -> list[EmbeddingResult]:
    """
    Async-safe batch chunk embedding.

    Args:
        chunks: List of chunks to process.
        storage_dir: Directory to persist vectors.

    Returns:
        List of EmbeddingResult objects.
    """
    pipeline = EmbeddingPipeline(storage_dir)
    return pipeline.process_chunks(chunks)


def embed_chunks_sync(
    chunks: Sequence[Union[ChunkInput, dict[str, Any]]],
    storage_dir: Path,
) -> list[EmbeddingResult]:
    """
    Synchronous batch chunk embedding.

    Args:
        chunks: List of chunks to process.
        storage_dir: Directory to persist vectors.

    Returns:
        List of EmbeddingResult objects.
    """
    pipeline = EmbeddingPipeline(storage_dir)
    return pipeline.process_chunks(chunks)
