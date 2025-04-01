"""
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
        Generate embeddings for users with additional checks for diversity.
        
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
        user_array = np.array(users)
        users_tensor = torch.tensor(user_array, dtype=torch.float32).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            # Apply small random noise to user features to increase diversity
            if len(users) == 1:  # Only apply to single user inference (recommendation)
                # Add small random perturbation to encourage diversity
                noise_scale = 0.02
                
                # Make multiple versions of the user embedding with slight variations
                num_variations = 5
                all_embeddings = []
                
                for i in range(num_variations):
                    # Different noise for each variation
                    perturbed_input = users_tensor + torch.randn_like(users_tensor) * noise_scale
                    emb = self.user_model(perturbed_input)
                    all_embeddings.append(emb)
                
                # Average the embeddings for stability
                embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
            else:
                # Standard embedding generation for batch processing
                embeddings = self.user_model(users_tensor)

            # L2 normalize embeddings
            normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        if len(users) == 1:  # Only when generating during recommendation
            user_emb = normalized_embeddings[0].cpu().numpy()
            
            # print(f"User embedding stats - min: {user_emb.min():.6f}, max: {user_emb.max():.6f}, std: {user_emb.std():.6f}")
        
        # Convert to numpy
        return normalized_embeddings.cpu().numpy()


    
    def generate_item_embedding(self, items):
        """
        Generate embeddings for items with consistent normalization.
        
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
        items_array = np.array(items)
        
        
        items_tensor = torch.tensor(items_array, dtype=torch.float32).to(self.device)
        
        # Generate embeddings in smaller batches to prevent OOM errors
        batch_size = 1024
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(items), batch_size):
                batch = items_tensor[i:i+batch_size]
                batch_embeddings = self.item_model(batch)
                normalized_batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                all_embeddings.append(normalized_batch_embeddings)
        
        # Concatenate all batches
        if len(all_embeddings) > 1:
            normalized_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            normalized_embeddings = all_embeddings[0]
        
        # Convert to numpy
        return normalized_embeddings.cpu().numpy()