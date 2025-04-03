# tests/unit/test_neural_network.py
"""
Unit test for Neural Network Module (M6).
Tests tower architecture, initialization, and forward pass.
"""
import sys
import os
import unittest
import torch
import numpy as np

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.modules.neural_network import (
    TowerNetwork,
    create_user_tower,
    create_item_tower,
)


class TestNeuralNetwork(unittest.TestCase):
    """Test cases for the Neural Network module."""

    def setUp(self):
        """Set up test environment before each test method."""
        self.input_dim = 4
        self.embedding_dim = 32
        self.batch_size = 10

        # Create sample inputs
        self.sample_input = torch.rand(self.batch_size, self.input_dim)

    def test_tower_network_init(self):
        """Test initialization of tower network."""
        network = TowerNetwork(
            input_dim=self.input_dim, embedding_dim=self.embedding_dim
        )

        # Check that the network has the right attributes
        self.assertEqual(network.input_dim, self.input_dim)
        self.assertEqual(network.embedding_dim, self.embedding_dim)

        # Check model structure (sequential with expected layers)
        self.assertIsInstance(network.model, torch.nn.Sequential)

        # First layer should be Linear with expected dimensions
        self.assertIsInstance(network.model[0], torch.nn.Linear)
        self.assertEqual(network.model[0].in_features, self.input_dim)
        self.assertEqual(network.model[0].out_features, self.embedding_dim * 2)

        # Second layer should be ReLU
        self.assertIsInstance(network.model[1], torch.nn.ReLU)

        # Third layer should be Linear with expected dimensions
        self.assertIsInstance(network.model[2], torch.nn.Linear)
        self.assertEqual(network.model[2].in_features, self.embedding_dim * 2)
        self.assertEqual(network.model[2].out_features, self.embedding_dim)

    def test_forward_pass(self):
        """Test forward pass of tower network."""
        network = TowerNetwork(
            input_dim=self.input_dim, embedding_dim=self.embedding_dim
        )

        # Run forward pass
        output = network(self.sample_input)

        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))

        # Check that output is normalized (all vectors have unit norm)
        norms = torch.norm(output, dim=1)
        for norm in norms:
            self.assertAlmostEqual(norm.item(), 1.0, places=5)

    def test_forward_pass_with_noise(self):
        """Test forward pass with training noise."""
        network = TowerNetwork(
            input_dim=self.input_dim, embedding_dim=self.embedding_dim
        )

        # Set to training mode
        network.train()

        # Run forward pass with the same input twice
        output1 = network(self.sample_input)
        output2 = network(self.sample_input)

        # The outputs should be different due to training noise
        self.assertFalse(torch.allclose(output1, output2))

        # Set to evaluation mode
        network.eval()

        # Run forward pass with the same input twice
        output1 = network(self.sample_input)
        output2 = network(self.sample_input)

        # The outputs should be the same in evaluation mode
        self.assertTrue(torch.allclose(output1, output2))

    def test_create_user_tower(self):
        """Test create_user_tower function."""
        user_tower = create_user_tower(
            input_dim=self.input_dim,
            hidden_layers=[
                128,
                64,
            ],  # These should be ignored in the simplified version
            embedding_dim=self.embedding_dim,
        )

        # Check that it creates a TowerNetwork
        self.assertIsInstance(user_tower, TowerNetwork)

        # Check dimensions
        self.assertEqual(user_tower.input_dim, self.input_dim)
        self.assertEqual(user_tower.embedding_dim, self.embedding_dim)

        # Test with input
        output = user_tower(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))

    def test_create_item_tower(self):
        """Test create_item_tower function."""
        item_tower = create_item_tower(
            input_dim=self.input_dim,
            hidden_layers=[
                128,
                64,
            ],  # These should be ignored in the simplified version
            embedding_dim=self.embedding_dim,
        )

        # Check that it creates a TowerNetwork
        self.assertIsInstance(item_tower, TowerNetwork)

        # Check dimensions
        self.assertEqual(item_tower.input_dim, self.input_dim)
        self.assertEqual(item_tower.embedding_dim, self.embedding_dim)

        # Test with input
        output = item_tower(self.sample_input)
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))

    def test_weight_initialization(self):
        """Test that weights are properly initialized."""
        network = TowerNetwork(
            input_dim=self.input_dim, embedding_dim=self.embedding_dim
        )

        # Check first linear layer weights
        layer1 = network.model[0]

        # Weights should not be all zeros
        self.assertFalse(torch.allclose(layer1.weight, torch.zeros_like(layer1.weight)))

        # Biases should not be all zeros if initialized with uniform
        if layer1.bias is not None:
            self.assertFalse(torch.allclose(layer1.bias, torch.zeros_like(layer1.bias)))

    def test_different_inputs_different_outputs(self):
        """Test that different inputs produce different embeddings."""
        network = TowerNetwork(
            input_dim=self.input_dim, embedding_dim=self.embedding_dim
        )
        network.eval()

        # Create two different inputs
        input1 = torch.rand(1, self.input_dim)
        input2 = torch.rand(1, self.input_dim)

        self.assertFalse(torch.allclose(input1, input2))

        # Run forward pass
        output1 = network(input1)
        output2 = network(input2)

        # Outputs should be different
        self.assertFalse(torch.allclose(output1, output2))


if __name__ == "__main__":
    unittest.main()
