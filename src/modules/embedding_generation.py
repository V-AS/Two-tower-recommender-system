"""
Modified Embedding Generation Module.
Generates embeddings for users and items using trained models.
"""
import torch
import numpy as np

class EmbeddingGenerator:
    def __init__(self):
        """Initialize the EmbeddingGenerator."""
        self.user_model = None
        self.item_model = None
        self.is_initialized = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self, user_model, item_model):
        """
        Initialize with trained models.
        
        Args:
            user_model: Trained user tower model
            item_model: Trained item tower model
            
        Raises:
            ValueError: If models are incompatible
        """
        # Verify models are compatible
        try:
            user_out = user_model(torch.randn(1, user_model.input_dim).to(self.device))
            item_out = item_model(torch.randn(1, item_model.input_dim).to(self.device))
            
            if user_out.shape[1] != item_out.shape[1]:
                raise ValueError("User and item models must produce embeddings of the same dimension")
            
            self.user_model = user_model.to(self.device)
            self.item_model = item_model.to(self.device)
            self.is_initialized = True
        except Exception as e:
            raise ValueError(f"Failed to initialize embedding generator: {str(e)}")
    
    def generate_user_embedding(self, users):
        """
        Generate embeddings for users.
        
        Args:
            users: List of processed user feature vectors
            
        Returns:
            array: User embeddings
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("EmbeddingGenerator not initialized")
        
        # Set model to evaluation mode
        self.user_model.eval()
        
        # Convert to tensor
        users_tensor = torch.tensor(users, dtype=torch.float32).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.user_model(users_tensor)
        
        # Convert to numpy
        return embeddings.cpu().numpy()
    
    def generate_item_embedding(self, items):
        """
        Generate embeddings for items.
        
        Args:
            items: List of processed item feature vectors
            
        Returns:
            array: Item embeddings
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("EmbeddingGenerator not initialized")
        
        # Set model to evaluation mode
        self.item_model.eval()
        
        # Convert to tensor
        items_tensor = torch.tensor(items, dtype=torch.float32).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.item_model(items_tensor)
        
        # Convert to numpy
        return embeddings.cpu().numpy()