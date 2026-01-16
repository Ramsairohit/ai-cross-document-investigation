"""
Unit tests for Stage 7: Vector Embeddings - Embedding Model

Tests for the singleton SentenceTransformer loader.
"""

import threading

import numpy as np
import pytest

from stage_7_embeddings.embedding_model import (
    EmbeddingModelLoader,
    encode_text,
    get_embedding_dimension,
    get_embedding_model,
    is_model_loaded,
)


class TestEmbeddingModelLoader:
    """Tests for EmbeddingModelLoader singleton."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_singleton_pattern(self):
        """Should return same instance."""
        loader1 = EmbeddingModelLoader()
        loader2 = EmbeddingModelLoader()

        assert loader1 is loader2

    def test_model_loads(self):
        """Should load model successfully."""
        loader = EmbeddingModelLoader()
        model = loader.get_model()

        assert model is not None

    def test_embedding_dimension(self):
        """Should return correct embedding dimension."""
        loader = EmbeddingModelLoader()

        assert loader.embedding_dimension == 384

    def test_model_name(self):
        """Should return correct model name."""
        loader = EmbeddingModelLoader()

        assert loader.model_name == "all-MiniLM-L6-v2"

    def test_encode_returns_array(self):
        """Encode should return numpy array."""
        loader = EmbeddingModelLoader()
        embedding = loader.encode("Test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_encode_deterministic(self):
        """Same text should produce same embedding."""
        loader = EmbeddingModelLoader()
        text = "I heard a loud crash around 8:15 PM."

        embedding1 = loader.encode(text)
        embedding2 = loader.encode(text)

        np.testing.assert_array_equal(embedding1, embedding2)

    def test_reset(self):
        """Reset should clear singleton."""
        loader1 = EmbeddingModelLoader()
        _ = loader1.get_model()

        assert is_model_loaded()

        EmbeddingModelLoader.reset()

        assert not is_model_loaded()

    def test_thread_safety(self):
        """Should be thread-safe."""
        instances = []
        errors = []

        def get_instance():
            try:
                loader = EmbeddingModelLoader()
                instances.append(id(loader))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All instances should be the same
        assert len(set(instances)) == 1


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        EmbeddingModelLoader.reset()

    def teardown_method(self):
        """Reset singleton after each test."""
        EmbeddingModelLoader.reset()

    def test_get_embedding_model(self):
        """Should get model via convenience function."""
        model = get_embedding_model()

        assert model is not None

    def test_encode_text(self):
        """Should encode text via convenience function."""
        embedding = encode_text("Test text")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)

    def test_get_embedding_dimension(self):
        """Should get dimension via convenience function."""
        dim = get_embedding_dimension()

        assert dim == 384

    def test_is_model_loaded(self):
        """Should check if model is loaded."""
        assert not is_model_loaded()

        get_embedding_model()

        assert is_model_loaded()
