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
        Train the two-tower model.
        
        Args:
            dataset: Training dataset with user_features, item_features, and ratings
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training history
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Setup loss function
        criterion = nn.MSELoss()
        
        # Training history
        history = {
            'loss': [],
            'val_loss': []
        }
        
        # Train with k-fold cross-validation
        k_folds = 5
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        # Split dataset indices

        indices = np.arange(len(dataset['user_ids']))
        
        for epoch in range(epochs):
            epoch_losses = []
            val_losses = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
                # Training
                self.user_model.train()
                self.item_model.train()
                
                # Mini-batch training
                for i in range(0, len(train_idx), batch_size):
                    batch_indices = train_idx[i:i+batch_size]
                    
                    # Get batch data
                    user_batch = torch.tensor(dataset['user_features'][batch_indices], dtype=torch.float32).to(self.device)
                    item_batch = torch.tensor(dataset['item_features'][batch_indices], dtype=torch.float32).to(self.device)
                    rating_batch = torch.tensor(dataset['ratings'][batch_indices], dtype=torch.float32).to(self.device)
                    
                    # Forward pass
                    user_embeddings = self.user_model(user_batch)
                    item_embeddings = self.item_model(item_batch)
                    
                    # Compute predicted ratings (dot product)
                    predicted_ratings = torch.sum(user_embeddings * item_embeddings, dim=1)
                    
                    # Compute loss
                    loss = criterion(predicted_ratings, rating_batch)
                    
                    # Backward pass and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_losses.append(loss.item())
                
                # Validation
                self.user_model.eval()
                self.item_model.eval()
                
                with torch.no_grad():
                    user_val = torch.tensor(dataset['user_features'][val_idx], dtype=torch.float32).to(self.device)
                    item_val = torch.tensor(dataset['item_features'][val_idx], dtype=torch.float32).to(self.device)
                    rating_val = torch.tensor(dataset['ratings'][val_idx], dtype=torch.float32).to(self.device)
                    
                    user_embeddings = self.user_model(user_val)
                    item_embeddings = self.item_model(item_val)
                    
                    predicted_ratings = torch.sum(user_embeddings * item_embeddings, dim=1)
                    val_loss = criterion(predicted_ratings, rating_val).item()
                    val_losses.append(val_loss)
            
            # Record history
            avg_loss = np.mean(epoch_losses)
            avg_val_loss = np.mean(val_losses)
            history['loss'].append(avg_loss)
            history['val_loss'].append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
        
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