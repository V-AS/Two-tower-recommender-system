# tests/unit/test_embedding_generation.py
"""
Unit test for Embedding Generation Module (M4).
Tests generation of user and item embeddings.
"""
import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.embedding_generation import EmbeddingGenerator
from src.modules.neural_network import create_user_tower, create_item_tower


class TestEmbeddingGenerator(unittest.TestCase):
    """Test cases for the EmbeddingGenerator module."""
    
    def setUp(self):
        """Set up test environment before each test method."""
        self.embedding_dim = 32
        self.user_input_dim = 4
        self.item_input_dim = 4
        
        # Create models for testing
        self.user_model = create_user_tower(
            input_dim=self.user_input_dim,
            embedding_dim=self.embedding_dim
        )
        
        self.item_model = create_item_tower(
            input_dim=self.item_input_dim,
            embedding_dim=self.embedding_dim
        )
        
        # Create sample data
        self.sample_user_features = np.random.rand(10, self.user_input_dim).astype(np.float32)
        self.sample_item_features = np.random.rand(10, self.item_input_dim).astype(np.float32)
        
        # Create a single user/item for single tests
        self.single_user = np.random.rand(self.user_input_dim).astype(np.float32)
        self.single_item = np.random.rand(self.item_input_dim).astype(np.float32)
        
        # Initialize generator
        self.generator = EmbeddingGenerator()
    
    def test_initialization(self):
        """Test initialization of the embedding generator."""
        # Initialize generator with models
        self.generator.initialize(self.user_model, self.item_model)
        
        # Check initialization status
        self.assertTrue(self.generator.is_initialized)
        
        # Check that models are stored
        self.assertIsNotNone(self.generator.user_model)
        self.assertIsNotNone(self.generator.item_model)
    
    def test_initialization_with_incompatible_models(self):
        """Test initialization with incompatible models raises an error."""
        # Create an incompatible item model (different embedding dimension)
        incompatible_item_model = create_item_tower(
            input_dim=self.item_input_dim,
            embedding_dim=self.embedding_dim + 10  # Different embedding dimension
        )
        
        # Initialization should raise a ValueError
        with self.assertRaises(ValueError):
            self.generator.initialize(self.user_model, incompatible_item_model)
    
    def test_generate_user_embedding_single(self):
        """Test generating a single user embedding."""
        # Initialize generator
        self.generator.initialize(self.user_model, self.item_model)
        
        # Generate embedding for a single user
        user_embedding = self.generator.generate_user_embedding([self.single_user])
        
        # Check output shape
        self.assertEqual(user_embedding.shape, (1, self.embedding_dim))
        
        # Check that the embedding is normalized (has unit norm)
        norm = np.linalg.norm(user_embedding[0])
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_generate_user_embedding_batch(self):
        """Test generating embeddings for a batch of users."""
        # Initialize generator
        self.generator.initialize(self.user_model, self.item_model)
        
        # Generate embeddings for a batch of users
        user_embeddings = self.generator.generate_user_embedding(self.sample_user_features)
        
        # Check output shape
        self.assertEqual(user_embeddings.shape, (len(self.sample_user_features), self.embedding_dim))
        
        # Check that all embeddings are normalized (have unit norm)
        for i in range(len(user_embeddings)):
            norm = np.linalg.norm(user_embeddings[i])
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_generate_item_embedding_single(self):
        """Test generating a single item embedding."""
        # Initialize generator
        self.generator.initialize(self.user_model, self.item_model)
        
        # Generate embedding for a single item
        item_embedding = self.generator.generate_item_embedding([self.single_item])
        
        # Check output shape
        self.assertEqual(item_embedding.shape, (1, self.embedding_dim))
        
        # Check that the embedding is normalized (has unit norm)
        norm = np.linalg.norm(item_embedding[0])
        self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_generate_item_embedding_batch(self):
        """Test generating embeddings for a batch of items."""
        # Initialize generator
        self.generator.initialize(self.user_model, self.item_model)
        
        # Generate embeddings for a batch of items
        item_embeddings = self.generator.generate_item_embedding(self.sample_item_features)
        
        # Check output shape
        self.assertEqual(item_embeddings.shape, (len(self.sample_item_features), self.embedding_dim))
        
        # Check that all embeddings are normalized (have unit norm)
        for i in range(len(item_embeddings)):
            norm = np.linalg.norm(item_embeddings[i])
            self.assertAlmostEqual(norm, 1.0, places=5)
    
    def test_generate_without_initialization(self):
        """Test generating embeddings without initialization raises an error."""
        # Try to generate embeddings without initializing
        with self.assertRaises(RuntimeError):
            self.generator.generate_user_embedding([self.single_user])
            
        with self.assertRaises(RuntimeError):
            self.generator.generate_item_embedding([self.single_item])
    
    
    def test_item_embedding_batching(self):
        """Test that item embeddings work with different batch sizes."""
        # Initialize generator
        self.generator.initialize(self.user_model, self.item_model)
        
        # Create a larger batch of items
        large_batch = np.random.rand(1500, self.item_input_dim).astype(np.float32)
        
        # Generate embeddings
        item_embeddings = self.generator.generate_item_embedding(large_batch)
        
        # Check output shape
        self.assertEqual(item_embeddings.shape, (1500, self.embedding_dim))
    


if __name__ == '__main__':
    unittest.main()