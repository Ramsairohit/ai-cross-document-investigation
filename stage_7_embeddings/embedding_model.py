"""
Stage 7: Vector Embeddings - Embedding Model Loader

Thread-safe singleton loader for SentenceTransformer models.
Model is loaded once and reused for performance.

IMPORTANT:
- Deterministic embeddings via controlled random seeds
- Model loaded lazily on first use
- Thread-safe for concurrent access
"""

import threading
from typing import Optional

import numpy as np


class EmbeddingModelLoader:
    """
    Thread-safe singleton loader for SentenceTransformer.

    Ensures the model is loaded only once across the application
    for optimal memory usage and performance.
    """

    _instance: Optional["EmbeddingModelLoader"] = None
    _lock: threading.Lock = threading.Lock()
    _model: Optional[object] = None  # SentenceTransformer type
    _model_name: str = "all-MiniLM-L6-v2"
    _embedding_dimension: int = 384

    def __new__(cls) -> "EmbeddingModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self) -> object:
        """
        Get the SentenceTransformer model, loading it if necessary.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self) -> object:
        """
        Load the SentenceTransformer model.

        Returns:
            Loaded SentenceTransformer model.
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self._model_name)
            return model
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            ) from e

    def encode(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text to a deterministic embedding vector.

        Args:
            text: Text to encode.
            normalize: Whether to L2-normalize the embedding.

        Returns:
            Embedding vector as numpy array.
        """
        model = self.get_model()

        # Set seeds for deterministic behavior
        np.random.seed(42)
        try:
            import torch

            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
        except ImportError:
            pass  # PyTorch not required if using CPU-only

        # Generate embedding
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )

        return embedding

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dimension

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.

        Useful for testing or when model needs to be reloaded.
        """
        with cls._lock:
            cls._model = None
            cls._instance = None


# Global instance for convenience
_loader = EmbeddingModelLoader()


def get_embedding_model() -> object:
    """
    Get the SentenceTransformer model (convenience function).

    Returns:
        Loaded SentenceTransformer model.
    """
    return _loader.get_model()


def encode_text(text: str, normalize: bool = True) -> np.ndarray:
    """
    Encode text to embedding (convenience function).

    Args:
        text: Text to encode.
        normalize: Whether to L2-normalize the embedding.

    Returns:
        Embedding vector as numpy array.
    """
    return _loader.encode(text, normalize)


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension.

    Returns:
        Embedding dimension (default: 384).
    """
    return _loader.embedding_dimension


def is_model_loaded() -> bool:
    """
    Check if the embedding model is already loaded.

    Returns:
        True if model is loaded, False otherwise.
    """
    return EmbeddingModelLoader._model is not None
