# tests/unit/test_ann_search.py
"""
Unit test for ANN Search Module (M7).
"""
import sys
import os
import unittest
import numpy as np
import tempfile
import shutil

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.modules.ann_search import ANNSearch


class TestANNSearch(unittest.TestCase):
    """Test cases for the ANNSearch module."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.ann_search = ANNSearch()

        # Create sample vectors for testing
        self.sample_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.3, 0.3, 0.3],
            ],
            dtype=np.float32,
        )

        # Normalize embeddings for inner product search
        norms = np.linalg.norm(self.sample_embeddings, axis=1, keepdims=True)
        self.normalized_embeddings = self.sample_embeddings / norms

        # Create sample item IDs
        self.sample_item_ids = np.array([101, 102, 103, 104, 105])

        # Create temporary directory for index files
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_build_index_flat(self):
        """Test building a flat index."""
        index = self.ann_search.build_index(
            self.normalized_embeddings, self.sample_item_ids, index_type="Flat"
        )

        # Check index structure
        self.assertIsInstance(index, dict)
        self.assertIn("index", index)
        self.assertIn("item_ids", index)
        self.assertIn("index_type", index)

        # Check item IDs were stored correctly
        np.testing.assert_array_equal(index["item_ids"], self.sample_item_ids)

        # Check index type
        self.assertEqual(index["index_type"], "Flat")

    def test_ann_search_exact_match(self):
        """Test searching for an exact match."""
        # Build index
        index = self.ann_search.build_index(
            self.normalized_embeddings, self.sample_item_ids, index_type="Flat"
        )

        # Search for the first vector
        query = self.normalized_embeddings[0]
        embeddings, ids = self.ann_search.ann_search(index, query, candidates=1)

        # Check if we found the exact match
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], self.sample_item_ids[0])

    def test_ann_search_approximate_match(self):
        """Test searching for an approximate match."""
        # Build index
        index = self.ann_search.build_index(
            self.normalized_embeddings, self.sample_item_ids, index_type="Flat"
        )
        # Create a query that is close to the fourth vector
        query = np.array([0.48, 0.52, 0.01], dtype=np.float32)
        query = query / np.linalg.norm(query)  # Normalize

        # Search for nearest neighbor
        embeddings, ids = self.ann_search.ann_search(index, query, candidates=1)

        # Check if we found the closest match (should be item 104)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], self.sample_item_ids[3])

    def test_ann_search_multiple_results(self):
        """Test searching for multiple nearest neighbors."""
        # Build index
        index = self.ann_search.build_index(
            self.normalized_embeddings, self.sample_item_ids, index_type="Flat"
        )

        # Create a query
        query = np.array([0.4, 0.4, 0.2], dtype=np.float32)
        query = query / np.linalg.norm(query)  # Normalize

        # Search for 3 nearest neighbors
        embeddings, ids = self.ann_search.ann_search(index, query, candidates=3)

        # Check if we got 3 results
        self.assertEqual(len(ids), 3)

    def test_build_index_empty(self):
        """Test building an index with empty embeddings raises an error."""
        empty_embeddings = np.array([], dtype=np.float32).reshape(0, 3)
        empty_ids = np.array([])

        with self.assertRaises(ValueError):
            self.ann_search.build_index(empty_embeddings, empty_ids)

    def test_build_index_mismatched_lengths(self):
        """Test building an index with mismatched embeddings and IDs raises an error."""
        with self.assertRaises(ValueError):
            self.ann_search.build_index(
                self.normalized_embeddings,
                self.sample_item_ids[:-1],  # One fewer ID than embeddings
            )

    def test_save_load_index(self):
        """Test saving and loading an index."""
        # Build index
        index = self.ann_search.build_index(
            self.normalized_embeddings, self.sample_item_ids, index_type="Flat"
        )

        # Save index
        index_path = os.path.join(self.temp_dir, "test_index")
        self.ann_search.save_index(index, index_path)

        # Check if files were created
        self.assertTrue(os.path.exists(f"{index_path}.faiss"))
        self.assertTrue(os.path.exists(f"{index_path}.meta.npy"))

        # Load index
        loaded_index = self.ann_search.load_index(index_path)

        # Check index structure
        self.assertIsInstance(loaded_index, dict)
        self.assertIn("index", loaded_index)
        self.assertIn("item_ids", loaded_index)
        self.assertIn("index_type", loaded_index)

        # Check item IDs were loaded correctly
        np.testing.assert_array_equal(loaded_index["item_ids"], self.sample_item_ids)

        # Check index type
        self.assertEqual(loaded_index["index_type"], "Flat")

    def test_load_index_nonexistent(self):
        """Test loading a non-existent index raises an error."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent_index")

        with self.assertRaises(IOError):
            self.ann_search.load_index(nonexistent_path)

    def test_ann_search_invalid_index(self):
        """Test searching with an invalid index raises an error."""
        invalid_index = {"not_an_index": True}
        query = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        with self.assertRaises(ValueError):
            self.ann_search.ann_search(invalid_index, query)


if __name__ == "__main__":
    unittest.main()
