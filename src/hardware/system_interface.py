"""
Hardware-Hiding Module for system interface operations.
Handles file I/O operations for models and embeddings.
"""
import os
import numpy as np
import torch

def save_model(model, path):
    """
    Save a trained model to the specified path.
    
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
        torch.save(model, path)
        return True
    except Exception as e:
        raise IOError(f"Failed to save model: {str(e)}")

def load_model(path):
    """
    Load a model from the specified path.
    
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
        return torch.load(path)
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