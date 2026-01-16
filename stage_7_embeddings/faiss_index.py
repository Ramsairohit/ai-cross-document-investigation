"""
Stage 7: Vector Embeddings - FAISS Index Manager

CPU-only FAISS index management with deterministic persistence.
Supports Flat (exact) and IVF (approximate) index types.

IMPORTANT:
- Index is deterministic and reproducible
- Re-running indexing produces identical ordering
- Persistence to disk for audit trail
"""

from pathlib import Path
from typing import Literal, Optional

import faiss
import numpy as np


class FAISSIndexManager:
    """
    CPU-only FAISS index with deterministic persistence.

    Manages vector storage and retrieval with full persistence support.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_type: Literal["Flat", "IVF"] = "Flat",
        nlist: int = 100,
    ):
        """
        Initialize FAISS index manager.

        Args:
            dimension: Vector dimension (default: 384 for all-MiniLM-L6-v2).
            index_type: 'Flat' for exact search, 'IVF' for approximate.
            nlist: Number of clusters for IVF index.
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self._index: Optional[faiss.Index] = None
        self._vector_count: int = 0

        # Create index
        self._create_index()

    def _create_index(self) -> None:
        """Create the FAISS index based on configuration."""
        if self.index_type == "Flat":
            # Exact search with L2 distance
            self._index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Approximate search with IVF
            quantizer = faiss.IndexFlatL2(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

    def add_vector(self, vector: np.ndarray) -> int:
        """
        Add a single vector to the index.

        Args:
            vector: Vector to add (must match dimension).

        Returns:
            The vector's position (ID) in the index.
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        if vector.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector.shape[1]} != index dimension {self.dimension}"
            )

        # Ensure float32 for FAISS
        vector = vector.astype(np.float32)

        # Train IVF index if needed
        if self.index_type == "IVF" and not self._index.is_trained:
            # Need training data - use the vector itself for single vector
            self._index.train(vector)

        position = self._vector_count
        self._index.add(vector)
        self._vector_count += 1

        return position

    def add_vectors(self, vectors: np.ndarray) -> list[int]:
        """
        Add multiple vectors to the index.

        Args:
            vectors: Array of vectors (N x dimension).

        Returns:
            List of vector positions (IDs) in the index.
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} != index dimension {self.dimension}"
            )

        # Ensure float32 for FAISS
        vectors = vectors.astype(np.float32)

        # Train IVF index if needed
        if self.index_type == "IVF" and not self._index.is_trained:
            self._index.train(vectors)

        start_position = self._vector_count
        positions = list(range(start_position, start_position + len(vectors)))

        self._index.add(vectors)
        self._vector_count += len(vectors)

        return positions

    def save(self, path: Path) -> None:
        """
        Persist the index to disk.

        Args:
            path: File path to save the index.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    def load(self, path: Path) -> None:
        """
        Load the index from disk.

        Args:
            path: File path to load the index from.

        Raises:
            FileNotFoundError: If the index file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        self._index = faiss.read_index(str(path))
        self._vector_count = self._index.ntotal

    def get_vector_count(self) -> int:
        """Get the number of vectors in the index."""
        return self._vector_count

    def is_empty(self) -> bool:
        """Check if the index is empty."""
        return self._vector_count == 0

    def reconstruct(self, vector_id: int) -> np.ndarray:
        """
        Reconstruct a vector from the index by its ID.

        Args:
            vector_id: The vector's position in the index.

        Returns:
            The reconstructed vector.

        Note:
            Only works with Flat index or trained IVF index.
        """
        if vector_id < 0 or vector_id >= self._vector_count:
            raise ValueError(f"Invalid vector_id: {vector_id}")

        return self._index.reconstruct(vector_id)
