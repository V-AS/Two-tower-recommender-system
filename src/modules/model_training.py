"""
Simplified Model Training Module.
Focuses on ensuring embedding diversity.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from hardware.system_interface import save_training_history


# Default parameters
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 64
DEFAULT_REGULARIZATION = 0.0001

output_dir = "output"  # Directory to save training history

random_seed = 5241323
torch.manual_seed(random_seed)
class ModelTrainer:
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.user_model = None
        self.item_model = None
        self.is_initialized = False
        self.config = None
        self.optimizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def initialize(self, config):
        """
        Initialize the trainer with configuration.
        
        Args:
            config: Training configuration
            
        Raises:
            ValueError: If config contains invalid parameters
        """
        if 'user_architecture' not in config or 'item_architecture' not in config:
            raise ValueError("Config must contain user_architecture and item_architecture")
        
        self.config = config
        self.user_model = config['user_architecture'].to(self.device)
        self.item_model = config['item_architecture'].to(self.device)
        
        # Configure optimizer with separate parameter groups and learning rates
        user_params = list(self.user_model.parameters())
        item_params = list(self.item_model.parameters())
        
        # Use Adam with reduced weight decay
        lr = config.get('learning_rate', DEFAULT_LEARNING_RATE)
        weight_decay = config.get('regularization', DEFAULT_REGULARIZATION)
        
        self.optimizer = optim.Adam([
            {'params': user_params, 'lr': lr},
            {'params': item_params, 'lr': lr}
        ], weight_decay=weight_decay)
        
        self.is_initialized = True
    
    def train(self, dataset, epochs=10):
        """
        Train the two-tower model
        
        Args:
            dataset: Training dataset
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training history
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Rating prediction loss
        rating_criterion = nn.MSELoss()
        
        # Training history
        history = {
            'loss': [],
            'rating_loss': [],
            'diversity_loss': [],
            'val_loss': []
        }
        
        indices = np.arange(len(dataset['user_ids']))
        
        for epoch in range(epochs):
            # Shuffle data for this epoch
            np.random.shuffle(indices)
            
            epoch_losses = []
            rating_losses = []
            diversity_losses = []
            
            # Training in mini-batches
            self.user_model.train()
            self.item_model.train()
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i+batch_size]
                
                # Get batch data
                user_batch = torch.tensor(dataset['user_features'][batch_indices], dtype=torch.float32).to(self.device)
                item_batch = torch.tensor(dataset['item_features'][batch_indices], dtype=torch.float32).to(self.device)
                rating_batch = torch.tensor(dataset['ratings'][batch_indices], dtype=torch.float32).to(self.device)
                
                # Forward pass
                user_embeddings = self.user_model(user_batch)
                item_embeddings = self.item_model(item_batch)
                
                # 1. Rating prediction loss (dot product)
                predicted_ratings = torch.sum(user_embeddings * item_embeddings, dim=1)
                rating_loss = rating_criterion(predicted_ratings, rating_batch)
                
                # 2. Diversity loss
                # Calculate cosine similarity matrix for embeddings
                user_sim = torch.mm(user_embeddings, user_embeddings.t())
                item_sim = torch.mm(item_embeddings, item_embeddings.t())
                
                # Create identity matrix for comparison (diagonal should be 1, off-diagonal should be lower)
                identity = torch.eye(len(batch_indices)).to(self.device)
                
                # Calculate diversity loss (penalize high similarity between different embeddings)
                # We mask out the diagonal elements (self-similarity) with the identity matrix
                diversity_loss = torch.mean(torch.pow(user_sim * (1 - identity), 2)) + \
                                torch.mean(torch.pow(item_sim * (1 - identity), 2))
                
                # Total loss with diversity component
                loss = rating_loss + 0.7 * diversity_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.user_model.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.item_model.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                rating_losses.append(rating_loss.item())
                diversity_losses.append(diversity_loss.item() * 0.1)
            
            # Validation
            with torch.no_grad():
                self.user_model.eval()
                self.item_model.eval()
                
                # Sample validation data
                val_indices = np.random.choice(indices, min(1000, len(indices)), replace=False)
                
                val_user = torch.tensor(dataset['user_features'][val_indices], dtype=torch.float32).to(self.device)
                val_item = torch.tensor(dataset['item_features'][val_indices], dtype=torch.float32).to(self.device)
                val_rating = torch.tensor(dataset['ratings'][val_indices], dtype=torch.float32).to(self.device)
                
                val_user_embeddings = self.user_model(val_user)
                val_item_embeddings = self.item_model(val_item)
                
                # Check variation in embeddings
                user_emb_var = torch.var(val_user_embeddings).item()
                item_emb_var = torch.var(val_item_embeddings).item()
                
                # Compute validation loss
                val_pred_ratings = torch.sum(val_user_embeddings * val_item_embeddings, dim=1)
                val_loss = rating_criterion(val_pred_ratings, val_rating).item()
            
            # Record history
            avg_loss = np.mean(epoch_losses)
            avg_rating_loss = np.mean(rating_losses)
            avg_diversity_loss = np.mean(diversity_losses)
            history['loss'].append(avg_loss)
            history['rating_loss'].append(avg_rating_loss)
            history['diversity_loss'].append(avg_diversity_loss)
            history['val_loss'].append(val_loss)
            

            
            # Early stopping if embeddings have good variance
            if user_emb_var > 0.1 and item_emb_var > 0.1 and epoch > 5:
                print("Embeddings have sufficient variance - early stopping")
                break
        
        # Return trained model and history
        model = {
            'user_model': self.user_model,
            'item_model': self.item_model,
            'history': history
        }
        
        return model
    
    
    def evaluate(self, dataset):
        """
        Evaluate the model on test data.
        
        Args:
            dataset: Testing dataset
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        # Setup loss function
        criterion = nn.MSELoss()
        
        # Evaluation
        self.user_model.eval()
        self.item_model.eval()
        
        with torch.no_grad():
            user_test = torch.tensor(dataset['user_features'], dtype=torch.float32).to(self.device)
            item_test = torch.tensor(dataset['item_features'], dtype=torch.float32).to(self.device)
            rating_test = torch.tensor(dataset['ratings'], dtype=torch.float32).to(self.device)
            
            user_embeddings = self.user_model(user_test)
            item_embeddings = self.item_model(item_test)
            
            # Calculate embedding stats
            user_emb_var = torch.var(user_embeddings).item()
            item_emb_var = torch.var(item_embeddings).item()
            print(f"Test User Emb Var: {user_emb_var:.6f}, Item Emb Var: {item_emb_var:.6f}")
            
            predicted_ratings = torch.sum(user_embeddings * item_embeddings, dim=1)
            
            mse = criterion(predicted_ratings, rating_test).item()
            rmse = np.sqrt(mse)
            
            # Calculate additional metrics
            predictions = predicted_ratings.cpu().numpy()
            targets = rating_test.cpu().numpy()
            
            # Mean Absolute Error
            mae = np.mean(np.abs(predictions - targets))
            
            # Determine accuracy (% of predictions within 10% of true value)
            accuracy = np.mean(np.abs(predictions - targets) <= 0.1)
            
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'user_embedding_variance': user_emb_var,
            'item_embedding_variance': item_emb_var
        }
        
        return metrics
    
    def get_user_model(self):
        """Get the trained user model."""
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        return self.user_model
    
    def get_item_model(self):
        """Get the trained item model."""
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        return self.item_model
    
    def update_model(self, dataset, epochs=5):
        """
        Update the model with new data (incremental learning).
        
        Args:
            dataset: New training data
            epochs (int): Number of training epochs
            
        Returns:
            dict: Updated model and training history
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        # Train for a few epochs on new data
        return self.train(dataset, epochs=epochs)