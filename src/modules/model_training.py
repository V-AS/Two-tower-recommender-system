"""
Model Training Module.
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
    
    def train(self, train_data, epochs=10):
        """
        Train the two-tower model.
        
        Args:
            train_data: Training dataset
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training history
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        # Setup
        users_df = train_data['users']
        books_df = train_data['books']
        ratings_df = train_data['ratings']
        
        batch_size = self.config.get('batch_size', DEFAULT_BATCH_SIZE)
        
        # Prepare user features
        user_features = self._prepare_user_features(users_df)
        
        # Prepare item features
        item_features = self._prepare_item_features(books_df)
        
        # Create dataset from ratings
        dataset = self._create_train_dataset(ratings_df, user_features, item_features)
        
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
    
    def evaluate(self, test_data):
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Testing dataset
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        # Setup
        users_df = test_data['users']
        books_df = test_data['books']
        ratings_df = test_data['ratings']
        
        # Prepare user features
        user_features = self._prepare_user_features(users_df)
        
        # Prepare item features
        item_features = self._prepare_item_features(books_df)
        
        # Create dataset from ratings
        dataset = self._create_train_dataset(ratings_df, user_features, item_features)
        
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
            accuracy = np.mean(np.abs(predictions - targets) <= 0.1)
            
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy
        }
        
        return metrics
    
    def get_user_model(self):
        """
        Get the trained user model.
        
        Returns:
            The user model
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        return self.user_model
    
    def get_item_model(self):
        """
        Get the trained item model.
        
        Returns:
            The item model
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        return self.item_model
    
    def update_model(self, new_data):
        """
        Update the model with new data (incremental learning).
        
        Args:
            new_data: New training data
            
        Returns:
            dict: Updated model and training history
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("ModelTrainer not initialized")
        
        # Train for a few epochs on new data
        return self.train(new_data, epochs=5)
    
    def _prepare_user_features(self, users_df):
        """
        Converts the processed dataframe into a format suitable for the neural network model
        """
        
        # Extract relevant columns for user features
        user_features = {}
        
        # Create a mapping from User-ID-Encoded to feature vector
        for _, row in users_df.iterrows():
            user_id = row['User-ID-Encoded']
            
            # Get numeric features
            features = [row['Age-Normalized']]
            
            # Add one-hot encoded location features
            location_cols = [col for col in users_df.columns if col.startswith('location_')]
            for col in location_cols:
                features.append(row[col])
            
            user_features[user_id] = np.array(features)
        
        return user_features
    
    def _prepare_item_features(self, books_df):
        """
        Converts the processed dataframe into a format suitable for the neural network model
        """
        # Extract relevant columns for item features
        item_features = {}
        
        # Create a mapping from ISBN-Encoded to feature vector
        for _, row in books_df.iterrows():
            item_id = row['ISBN-Encoded']
            
            # Get numeric features if available
            features = []
            
            if 'Year-Normalized' in books_df.columns:
                features.append(row['Year-Normalized'])
            
            if 'Author-Popularity' in books_df.columns:
                features.append(row['Author-Popularity'])
            
            # If we don't have any features, use a default
            if not features:
                features = [0.5]  # Default feature
            
            item_features[item_id] = np.array(features)
        
        return item_features
    
    def _create_train_dataset(self, ratings_df, user_features, item_features):
        """Create a training dataset from ratings and features."""
        user_ids = ratings_df['User-ID-Encoded'].values
        item_ids = ratings_df['ISBN-Encoded'].values
        ratings = ratings_df['Normalized-Rating'].values
        
        # Get feature vectors for each user and item in ratings
        user_feature_vectors = np.array([user_features[user_id] for user_id in user_ids])
        item_feature_vectors = np.array([item_features[item_id] for item_id in item_ids])
        
        dataset = {
            'user_ids': user_ids,
            'item_ids': item_ids,
            'ratings': ratings,
            'user_features': user_feature_vectors,
            'item_features': item_feature_vectors
        }
        
        return dataset