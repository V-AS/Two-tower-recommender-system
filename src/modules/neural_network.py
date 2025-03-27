"""
Neural Network Architecture Module.
Defines the architecture of user and item towers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_HIDDEN_LAYERS = [256, 128]
DEFAULT_ACTIVATION = "relu"

class TowerNetwork(nn.Module):
    """Base tower network for embedding generation."""
    
    def __init__(self, input_dim, hidden_layers, embedding_dim, activation="relu"):
        """
        Initialize a tower network.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_layers (list): List of hidden layer dimensions
            embedding_dim (int): Dimension of the output embedding
            activation (str): Activation function to use
        """
        super(TowerNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.embedding_dim = embedding_dim
        
        # Create layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, embedding_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)


def create_user_tower(input_dim, hidden_layers=None, embedding_dim=128):
    """
    Create a user tower network.
    
    Args:
        input_dim (int): Dimension of user features
        hidden_layers (list): List of hidden layer dimensions
        embedding_dim (int): Dimension of the output embedding
        
    Returns:
        TowerNetwork: The user tower
    """
    if hidden_layers is None:
        hidden_layers = DEFAULT_HIDDEN_LAYERS
    
    return TowerNetwork(input_dim, hidden_layers, embedding_dim)


def create_item_tower(input_dim, hidden_layers=None, embedding_dim=128):
    """
    Create an item tower network.
    
    Args:
        input_dim (int): Dimension of item features
        hidden_layers (list): List of hidden layer dimensions
        embedding_dim (int): Dimension of the output embedding
        
    Returns:
        TowerNetwork: The item tower
    """
    if hidden_layers is None:
        hidden_layers = DEFAULT_HIDDEN_LAYERS
    
    return TowerNetwork(input_dim, hidden_layers, embedding_dim)