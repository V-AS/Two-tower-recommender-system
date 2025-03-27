"""
Modified Model Training Module.
Handles training and evaluation of the two-tower model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_BATCH_SIZE = 128
DEFAULT_REGULARIZATION = 0.01

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
        
        # Configure optimizer
        params = list(self.user_model.parameters()) + list(self.item_model.parameters())
        lr = config.get('learning_rate', DEFAULT_LEARNING_RATE)
        weight_decay = config.get('regularization', DEFAULT_REGULARIZATION)
        
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.is_initialized = True
    
    def train(self, dataset, epochs=10):
        """
        Train the two-tower model with contrastive loss to better separate user preferences.
        
        Args:
            dataset: Training dataset
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training history
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Primary loss function for rating prediction
        rating_criterion = nn.MSELoss()
        
        # Contrastive loss margin
        margin = 0.5
    
        # Training history
        history = {
            'loss': [],
            'rating_loss': [],
            'contrastive_loss': [],
            'val_loss': []
        }
        
        indices = np.arange(len(dataset['user_ids']))
        
        for epoch in range(epochs):
            # Shuffle data for this epoch
            np.random.shuffle(indices)
            
            epoch_losses = []
            rating_losses = []
            contrastive_losses = []
            
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
                
                # 2. Contrastive loss to increase separation
                contrastive_loss = torch.tensor(0.0).to(self.device)
                
                # Create negatives by shuffling item embeddings
                shuffle_idx = torch.randperm(item_embeddings.size(0))
                negative_items = item_embeddings[shuffle_idx]
                negative_scores = torch.sum(user_embeddings * negative_items, dim=1)
                
                # Contrastive loss: push positives high and negatives low
                # For positives (real item for user), we want higher score for higher rating
                # For negatives (random item for user), we want lower score
                pos_weight = rating_batch  # Higher weight for higher ratings
                
                # Compute the contrastive loss term
                # We want predicted_ratings to be high and negative_scores to be low
                # with a margin between them proportional to the true rating
                for j in range(len(batch_indices)):
                    if rating_batch[j] >= 0.7:  # Only apply to items with high ratings (7+ out of 10)
                        # Calculate contrastive loss
                        contrastive_loss += max(0, margin - (predicted_ratings[j] - negative_scores[j]))
                
                # Normalize
                if len(batch_indices) > 0:
                    contrastive_loss = contrastive_loss / len(batch_indices)
                
                # Total loss
                loss = rating_loss + contrastive_loss
                
                # Backward pass and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
                rating_losses.append(rating_loss.item())
                contrastive_losses.append(contrastive_loss.item())
            
            # Validation
            with torch.no_grad():
                self.user_model.eval()
                self.item_model.eval()
                
                # Subsample validation data to keep it fast
                val_indices = np.random.choice(indices, min(5000, len(indices)), replace=False)
                
                val_user = torch.tensor(dataset['user_features'][val_indices], dtype=torch.float32).to(self.device)
                val_item = torch.tensor(dataset['item_features'][val_indices], dtype=torch.float32).to(self.device)
                val_rating = torch.tensor(dataset['ratings'][val_indices], dtype=torch.float32).to(self.device)
                
                val_user_embeddings = self.user_model(val_user)
                val_item_embeddings = self.item_model(val_item)
                
                # Check variation in user embeddings (diagnostic)
                user_emb_var = torch.std(val_user_embeddings, dim=0).mean().item()
                
                # Compute validation loss
                val_pred_ratings = torch.sum(val_user_embeddings * val_item_embeddings, dim=1)
                val_loss = rating_criterion(val_pred_ratings, val_rating).item()
            
            # Record history
            avg_loss = np.mean(epoch_losses)
            avg_rating_loss = np.mean(rating_losses)
            avg_contrastive_loss = np.mean(contrastive_losses)
            history['loss'].append(avg_loss)
            history['rating_loss'].append(avg_rating_loss)
            history['contrastive_loss'].append(avg_contrastive_loss)
            history['val_loss'].append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} (Rating: {avg_rating_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f}, User Embedding Variation: {user_emb_var:.6f}")
        
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
            
            predicted_ratings = torch.sum(user_embeddings * item_embeddings, dim=1)
            
            mse = criterion(predicted_ratings, rating_test).item()
            rmse = np.sqrt(mse)
            
            # Calculate additional metrics
            predictions = predicted_ratings.cpu().numpy()
            targets = rating_test.cpu().numpy()
            
            # Mean Absolute Error
            mae = np.mean(np.abs(predictions - targets))
            
            # Determine accuracy (% of predictions within 10% of true value)
            accuracy = np.mean(np.abs(predictions - targets) <= 0.5)
            
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy
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