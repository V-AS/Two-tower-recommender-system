"""
Improved Hardware-Hiding Module for system interface operations.
Handles file I/O operations for models and embeddings with better metadata preservation.
"""

import os
import json
import numpy as np
import torch


def save_training_history(history, path):
    """
    Save training history to a JSON file.

    Args:
        history (dict): Dictionary containing training metrics
            (e.g., {'loss': [...], 'val_loss': [...], ...})
        path (str): The file path to save to

    Returns:
        bool: True if successful

    Raises:
        IOError: If file cannot be written
    """
    try:
        # Create directory if it doesn't exist
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in history.items():
            if isinstance(value, list) or isinstance(value, np.ndarray):
                serializable_history[key] = [
                    float(v) if isinstance(v, (np.number, float)) else v for v in value
                ]
            else:
                serializable_history[key] = value

        # Save to JSON file
        with open(path, "w") as f:
            json.dump(serializable_history, f, indent=2)

        return True
    except Exception as e:
        raise IOError(f"Failed to save training history: {str(e)}")


def load_training_history(path):
    """
    Load training history from a JSON file.

    Args:
        path (str): The file path to load from

    Returns:
        dict: The loaded training history

    Raises:
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    try:
        if not os.path.exists(path):
            raise IOError(f"File not found: {path}")

        with open(path, "r") as f:
            history = json.load(f)

        return history
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in history file: {path}")
    except Exception as e:
        raise IOError(f"Failed to load training history: {str(e)}")


def save_model(model, path):
    try:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        # Save only model state dict
        torch.save(model.state_dict(), path)
        return True
    except Exception as e:
        raise IOError(f"Failed to save model: {str(e)}")


def load_model(path):
    """
    Load a model's state dictionary from the specified path.

    Args:
        path (str): The file path to load from

    Returns:
        dict: The loaded state dictionary

    Raises:
        IOError: If file cannot be read
        ValueError: If file format is invalid
    """
    try:
        if not os.path.exists(path):
            raise IOError(f"File not found: {path}")

        state_dict = torch.load(path, map_location="cpu", weights_only=False, strict=False)

        return state_dict

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
