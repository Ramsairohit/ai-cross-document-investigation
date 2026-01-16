"""
Stage 7: Vector Embeddings - Vector Store

Coordinates FAISS index with metadata storage.
Ensures deterministic linkage between vectors and their provenance.

IMPORTANT:
- Metadata stored as JSON for audit trail
- vector_id in metadata matches FAISS position exactly
- Re-running produces identical metadata linkage
"""

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .faiss_index import FAISSIndexManager
from .models import VectorRecord


class VectorStore:
    """
    Combined vector + metadata storage.

    Manages the FAISS index and corresponding metadata,
    ensuring perfect linkage between vectors and provenance.
    """

    def __init__(
        self,
        storage_dir: Path,
        dimension: int = 384,
        index_type: str = "Flat",
    ):
        """
        Initialize the vector store.

        Args:
            storage_dir: Directory to persist index and metadata.
            dimension: Vector dimension (default: 384).
            index_type: FAISS index type ('Flat' or 'IVF').
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "faiss.index"
        self.metadata_path = self.storage_dir / "metadata.json"

        self.index_manager = FAISSIndexManager(
            dimension=dimension,
            index_type=index_type,  # type: ignore
        )
        self.metadata: list[VectorRecord] = []
        self._dimension = dimension

    def add(
        self,
        chunk_id: str,
        vector: np.ndarray,
        case_id: str,
        document_id: str,
        page_range: list[int],
        speaker: Optional[str] = None,
        confidence: float = 1.0,
    ) -> int:
        """
        Add a vector with its metadata to the store.

        Args:
            chunk_id: Source chunk ID.
            vector: Embedding vector.
            case_id: Case ID for chain-of-custody.
            document_id: Source document ID.
            page_range: [start_page, end_page].
            speaker: Speaker label if present.
            confidence: Chunk confidence score.

        Returns:
            The vector_id (position in index).
        """
        # Add vector to FAISS index
        vector_id = self.index_manager.add_vector(vector)

        # Create metadata record
        record = VectorRecord(
            chunk_id=chunk_id,
            vector_id=vector_id,
            case_id=case_id,
            document_id=document_id,
            page_range=page_range,
            speaker=speaker,
            confidence=confidence,
        )

        self.metadata.append(record)

        return vector_id

    def add_batch(
        self,
        vectors: np.ndarray,
        metadata_list: list[dict[str, Any]],
    ) -> list[int]:
        """
        Add multiple vectors with metadata.

        Args:
            vectors: Array of vectors (N x dimension).
            metadata_list: List of metadata dicts for each vector.

        Returns:
            List of vector_ids.
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Vectors and metadata must have same length")

        vector_ids = self.index_manager.add_vectors(vectors)

        for i, (vector_id, meta) in enumerate(zip(vector_ids, metadata_list)):
            record = VectorRecord(
                chunk_id=meta["chunk_id"],
                vector_id=vector_id,
                case_id=meta["case_id"],
                document_id=meta["document_id"],
                page_range=meta["page_range"],
                speaker=meta.get("speaker"),
                confidence=meta.get("confidence", 1.0),
            )
            self.metadata.append(record)

        return vector_ids

    def get_metadata(self, vector_id: int) -> Optional[VectorRecord]:
        """
        Get metadata for a specific vector_id.

        Args:
            vector_id: The vector's position in the index.

        Returns:
            VectorRecord or None if not found.
        """
        for record in self.metadata:
            if record.vector_id == vector_id:
                return record
        return None

    def get_metadata_by_chunk_id(self, chunk_id: str) -> Optional[VectorRecord]:
        """
        Get metadata by chunk_id.

        Args:
            chunk_id: The source chunk ID.

        Returns:
            VectorRecord or None if not found.
        """
        for record in self.metadata:
            if record.chunk_id == chunk_id:
                return record
        return None

    def save(self) -> None:
        """
        Persist both index and metadata to disk.

        Creates:
        - faiss.index: The FAISS index file
        - metadata.json: JSON file with all VectorRecords
        """
        # Save FAISS index
        self.index_manager.save(self.index_path)

        # Save metadata as JSON
        metadata_json = [record.model_dump() for record in self.metadata]
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_json, f, indent=2, ensure_ascii=False)

    def load(self) -> None:
        """
        Load both index and metadata from disk.

        Raises:
            FileNotFoundError: If index or metadata files don't exist.
        """
        # Load FAISS index
        self.index_manager.load(self.index_path)

        # Load metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            metadata_json = json.load(f)

        self.metadata = [VectorRecord(**record) for record in metadata_json]

    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        return self.index_manager.get_vector_count()

    def is_empty(self) -> bool:
        """Check if the store is empty."""
        return self.index_manager.is_empty()

    def clear(self) -> None:
        """Clear the store (recreate empty index)."""
        self.index_manager = FAISSIndexManager(
            dimension=self._dimension,
            index_type=self.index_manager.index_type,  # type: ignore
        )
        self.metadata = []
