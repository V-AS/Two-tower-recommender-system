"""
Author: Yinying Huo
Date: 2025-04-03
Purpose: This module provides a deep neural network for generating embeddings for users and items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerNetwork(nn.Module):
    """tower network for embedding generation."""

    def __init__(self, input_dim, embedding_dim=64):
        """
        Initialize a minimal tower network.

        Args:
            input_dim (int): Dimension of input features
            embedding_dim (int): Dimension of the output embedding
        """
        super(TowerNetwork, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Use a single hidden layer with minimal complexity
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        # Initialize weights with more variance to encourage diversity
        self._init_weights()

    def _init_weights(self):
        """Custom initialization to prevent embedding collapse."""
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    # Add some randomness to biases
                    nn.init.uniform_(m.bias, -0.1, 0.1)

    def forward(self, x):
        """Forward pass through the network."""
        # Apply a small amount of noise to prevent identical outputs
        if self.training:
            noise = torch.randn_like(x) * 0.01
            x = x + noise

        embeddings = self.model(x)

        # L2 normalize the embeddings for cosine similarity
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)

        return normalized_embeddings


def create_user_tower(input_dim, hidden_layers=None, embedding_dim=64):
    """
    Create a simplified user tower network.

    Args:
        input_dim (int): Dimension of user features
        hidden_layers (list): Ignored
        embedding_dim (int): Dimension of the output embedding

    Returns:
        TowerNetwork: The user tower
    """
    return TowerNetwork(input_dim, embedding_dim)


def create_item_tower(input_dim, hidden_layers=None, embedding_dim=64):
    """
    Create a simplified item tower network.

    Args:
        input_dim (int): Dimension of item features
        hidden_layers (list): Ignored
        embedding_dim (int): Dimension of the output embedding

    Returns:
        TowerNetwork: The item tower
    """
    return TowerNetwork(input_dim, embedding_dim)
