# tests/unit/test_vector_operations.py
"""
Unit test for Vector Operations Module (M8).
Tests dot product calculation.
"""
import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.vector_operations import dot_product


class TestVectorOperations(unittest.TestCase):
    """Test cases for the VectorOperations module."""
    
    def test_dot_product_basic(self):
        """Test basic dot product calculation."""
        v1 = [1.0, 2.0, 3.0, 4.0]
        v2 = [5.0, 6.0, 7.0, 8.0]
        expected = 70.0  # 1*5 + 2*6 + 3*7 + 4*8
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_numpy_arrays(self):
        """Test dot product with numpy arrays."""
        v1 = np.array([1.0, 2.0, 3.0, 4.0])
        v2 = np.array([5.0, 6.0, 7.0, 8.0])
        expected = 70.0
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_empty_vectors(self):
        """Test dot product with empty vectors."""
        v1 = []
        v2 = []
        expected = 0.0
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_orthogonal_vectors(self):
        """Test dot product with orthogonal vectors."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        expected = 0.0
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_same_direction(self):
        """Test dot product with vectors in the same direction."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [2.0, 4.0, 6.0]  # v2 = 2*v1
        expected = 28.0  # 1*2 + 2*4 + 3*6
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_opposite_direction(self):
        """Test dot product with vectors in opposite directions."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [-1.0, -2.0, -3.0]  # v2 = -1*v1
        expected = -14.0  # 1*(-1) + 2*(-2) + 3*(-3)
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_dot_product_large_values(self):
        """Test dot product with large values."""
        v1 = [1e6, 2e6, 3e6]
        v2 = [4e6, 5e6, 6e6]
        expected = 32e12  # 1e6*4e6 + 2e6*5e6 + 3e6*6e6
        
        result = dot_product(v1, v2)
        
        # Use delta for large values
        self.assertAlmostEqual(result, expected, delta=1e6)
    
    def test_dot_product_small_values(self):
        """Test dot product with small values."""
        v1 = [1e-6, 2e-6, 3e-6]
        v2 = [4e-6, 5e-6, 6e-6]
        expected = 32e-12  # 1e-6*4e-6 + 2e-6*5e-6 + 3e-6*6e-6
        
        result = dot_product(v1, v2)
        
        self.assertAlmostEqual(result, expected, places=15)
    
    def test_dot_product_dimension_mismatch(self):
        """Test that dot product raises error for different dimensions."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [4.0, 5.0]
        
        with self.assertRaises(Exception):  # Expecting DimensionMismatchError
            dot_product(v1, v2)


if __name__ == '__main__':
    unittest.main()