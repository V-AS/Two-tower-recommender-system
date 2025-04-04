"""
Author: Yinying Huo
Date: 2025-04-03
Purpose: This module provides vector operations such as dot product and normalization.
"""

import numpy as np

EPSILON = 1e-8  # Small value to prevent division by zero


def dot_product(v1, v2):
    """
    Calculate the dot product of two vectors.

    Args:
        v1 (array-like): First vector
        v2 (array-like): Second vector

    Returns:
        float: Dot product result

    Raises:
        ValueError: If vectors have different dimensions
    """
    v1 = np.array(v1)
    v2 = np.array(v2)

    if v1.shape != v2.shape:
        raise ValueError(f"Dimension mismatch: {v1.shape} vs {v2.shape}")

    return np.dot(v1, v2)


def normalize(v):
    """
    Normalize a vector to unit length.

    Args:
        v (array-like): Vector to normalize

    Returns:
        array: Normalized vector
    """
    v = np.array(v)
    norm = np.linalg.norm(v)

    if norm < EPSILON:
        return v

    return v / norm
