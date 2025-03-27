"""
Improved Hardware-Hiding Module for system interface operations.
Handles file I/O operations for models and embeddings with better metadata preservation.
"""
import os
import json
import numpy as np
import torch

def save_model(model, path):
    """
    Save a trained model to the specified path.
    Saves both the state_dict and the model metadata.
    
    Args:
        model: The model to save
        path (str): The file path to save to
        
    Returns:
        bool: True if successful
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save model state dict
        model_state = model.state_dict()
        torch.save(model_state, path)
        
        # Save model metadata (input dimension, etc.)
        metadata = {
            'input_dim': model.input_dim,
            'hidden_layers': model.hidden_layers if hasattr(model, 'hidden_layers') else [],
            'embedding_dim': model.embedding_dim if hasattr(model, 'embedding_dim') else 128,
        }
        
        # Save metadata to a JSON file
        meta_path = f"{path}.meta"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
        return True
    except Exception as e:
        raise IOError(f"Failed to save model: {str(e)}")

def load_model(path):
    """
    Load a model from the specified path.
    Reconstructs the full model using the saved state_dict and metadata.
    
    Args:
        path (str): The file path to load from
        
    Returns:
        The loaded model
        
    Raises:
        IOError: If file cannot be read
        FormatError: If file format is invalid
    """
    try:
        if not os.path.exists(path):
            raise IOError(f"File not found: {path}")
        
        # First, try to load the metadata
        meta_path = f"{path}.meta"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                
                # Import model creation functions
                from modules.neural_network import create_user_tower, create_item_tower
                
                # Check if this is a user or item model based on filename
                if 'user' in os.path.basename(path).lower():
                    model = create_user_tower(
                        input_dim=metadata['input_dim'],
                        hidden_layers=metadata.get('hidden_layers', [256, 128]),
                        embedding_dim=metadata.get('embedding_dim', 128)
                    )
                else:
                    model = create_item_tower(
                        input_dim=metadata['input_dim'],
                        hidden_layers=metadata.get('hidden_layers', [256, 128]),
                        embedding_dim=metadata.get('embedding_dim', 128)
                    )
                
                # Load the state dict
                state_dict = torch.load(path, map_location='cpu')
                model.load_state_dict(state_dict)
                
                return model
            except Exception as e:
                print(f"Warning: Failed to load with metadata: {str(e)}")
                # Fall back to regular loading methods
        
        # Try loading with various compatibility settings
        try:
            # Try standard loading first (might work for older saves)
            return torch.load(path, map_location='cpu')
        except Exception:
            try:
                # Load state dict
                state_dict = torch.load(path, map_location='cpu')
                
                # If we get here, we have a state dict but no metadata
                # Try to determine input dimensions from weights
                if isinstance(state_dict, dict):
                    first_layer_key = [k for k in state_dict.keys() if 'weight' in k][0]
                    input_dim = state_dict[first_layer_key].shape[1]
                    
                    # Import model creation functions
                    from modules.neural_network import create_user_tower, create_item_tower
                    
                    # Create model based on filename
                    if 'user' in os.path.basename(path).lower():
                        model = create_user_tower(input_dim=input_dim)
                    else:
                        model = create_item_tower(input_dim=input_dim)
                    
                    # Load state dict
                    model.load_state_dict(state_dict)
                    return model
                else:
                    raise ValueError("Unexpected format for saved model")
            except Exception as e:
                print(f"Warning: Failed to reconstruct model: {str(e)}")
                # Return the state dict as a last resort
                return torch.load(path, map_location='cpu')
    except Exception as e:
        if "FormatError" in str(e):
            raise ValueError(f"Invalid model format: {str(e)}")
        raise IOError(f"Failed to load model: {str(e)}")

def save_embeddings(embeddings, path):
    """
    Save embeddings to the specified path.
    
    Args:
        embeddings: The embeddings to save
        path (str): The file path to save to
        
    Returns:
        bool: True if successful
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        np.save(path, embeddings)
        return True
    except Exception as e:
        raise IOError(f"Failed to save embeddings: {str(e)}")

def load_embeddings(path):
    """
    Load embeddings from the specified path.
    
    Args:
        path (str): The file path to load from
        
    Returns:
        The loaded embeddings
        
    Raises:
        IOError: If file cannot be read
        FormatError: If file format is invalid
    """
    try:
        if not os.path.exists(path):
            raise IOError(f"File not found: {path}")
        return np.load(path)
    except Exception as e:
        if "FormatError" in str(e):
            raise ValueError(f"Invalid embeddings format: {str(e)}")
        raise IOError(f"Failed to load embeddings: {str(e)}")