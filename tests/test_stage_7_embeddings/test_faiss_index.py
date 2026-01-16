"""
Unit tests for Stage 7: Vector Embeddings - FAISS Index

Tests for FAISS index creation, persistence, and retrieval.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from stage_7_embeddings.faiss_index import FAISSIndexManager


class TestFAISSIndexManager:
    """Tests for FAISSIndexManager."""

    def test_create_flat_index(self):
        """Should create Flat index."""
        manager = FAISSIndexManager(dimension=384, index_type="Flat")

        assert manager.dimension == 384
        assert manager.index_type == "Flat"
        assert manager.is_empty()

    def test_create_ivf_index(self):
        """Should create IVF index."""
        manager = FAISSIndexManager(dimension=384, index_type="IVF")

        assert manager.dimension == 384
        assert manager.index_type == "IVF"

    def test_add_single_vector(self):
        """Should add single vector and return position."""
        manager = FAISSIndexManager(dimension=384)
        vector = np.random.randn(384).astype(np.float32)

        position = manager.add_vector(vector)

        assert position == 0
        assert manager.get_vector_count() == 1
        assert not manager.is_empty()

    def test_add_multiple_vectors(self):
        """Should add multiple vectors and return positions."""
        manager = FAISSIndexManager(dimension=384)
        vectors = np.random.randn(5, 384).astype(np.float32)

        positions = manager.add_vectors(vectors)

        assert positions == [0, 1, 2, 3, 4]
        assert manager.get_vector_count() == 5

    def test_add_sequential_vectors(self):
        """Adding vectors sequentially should increment positions."""
        manager = FAISSIndexManager(dimension=384)

        pos1 = manager.add_vector(np.random.randn(384).astype(np.float32))
        pos2 = manager.add_vector(np.random.randn(384).astype(np.float32))
        pos3 = manager.add_vector(np.random.randn(384).astype(np.float32))

        assert pos1 == 0
        assert pos2 == 1
        assert pos3 == 2

    def test_dimension_mismatch(self):
        """Should reject vectors with wrong dimension."""
        manager = FAISSIndexManager(dimension=384)
        wrong_vector = np.random.randn(256).astype(np.float32)

        with pytest.raises(ValueError):
            manager.add_vector(wrong_vector)

    def test_save_and_load(self):
        """Should save and load index without loss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "test.index"

            # Create and populate index
            manager = FAISSIndexManager(dimension=384)
            vectors = np.random.randn(3, 384).astype(np.float32)
            manager.add_vectors(vectors)
            manager.save(index_path)

            # Load into new manager
            new_manager = FAISSIndexManager(dimension=384)
            new_manager.load(index_path)

            assert new_manager.get_vector_count() == 3

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        manager = FAISSIndexManager(dimension=384)

        with pytest.raises(FileNotFoundError):
            manager.load(Path("/nonexistent/path/index.faiss"))

    def test_reconstruct_vector(self):
        """Should reconstruct vector from index."""
        manager = FAISSIndexManager(dimension=384, index_type="Flat")
        original = np.random.randn(384).astype(np.float32)

        position = manager.add_vector(original)
        reconstructed = manager.reconstruct(position)

        np.testing.assert_array_almost_equal(original, reconstructed, decimal=5)

    def test_reconstruct_invalid_id(self):
        """Should raise error for invalid vector ID."""
        manager = FAISSIndexManager(dimension=384)
        manager.add_vector(np.random.randn(384).astype(np.float32))

        with pytest.raises(ValueError):
            manager.reconstruct(999)

    def test_empty_index_handling(self):
        """Empty index operations should work correctly."""
        manager = FAISSIndexManager(dimension=384)

        assert manager.is_empty()
        assert manager.get_vector_count() == 0


class TestFAISSPersistence:
    """Tests focused on persistence behavior."""

    def test_persistence_preserves_order(self):
        """Loaded index should preserve vector order."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "order.index"

            # Create specific vectors
            manager = FAISSIndexManager(dimension=4, index_type="Flat")
            v1 = np.array([1, 0, 0, 0], dtype=np.float32)
            v2 = np.array([0, 1, 0, 0], dtype=np.float32)
            v3 = np.array([0, 0, 1, 0], dtype=np.float32)

            manager.add_vector(v1)
            manager.add_vector(v2)
            manager.add_vector(v3)
            manager.save(index_path)

            # Load and verify order
            new_manager = FAISSIndexManager(dimension=4)
            new_manager.load(index_path)

            np.testing.assert_array_almost_equal(new_manager.reconstruct(0), v1)
            np.testing.assert_array_almost_equal(new_manager.reconstruct(1), v2)
            np.testing.assert_array_almost_equal(new_manager.reconstruct(2), v3)

    def test_multiple_save_load_cycles(self):
        """Index should survive multiple save/load cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = Path(tmpdir) / "cycles.index"
            manager = FAISSIndexManager(dimension=384)
            vector = np.random.randn(384).astype(np.float32)
            manager.add_vector(vector)

            for _ in range(3):
                manager.save(index_path)
                manager.load(index_path)

            assert manager.get_vector_count() == 1
