# tests/system/test_data_validation.py
"""
System test for dataset validation (test-id1).
Verifies that the dataset meets the required format before training.
"""
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.modules.data_processing import DataProcessor


def test_dataset_validation(data_path):
    """Test dataset validation to ensure each user-item pair has associated reward."""
    print(f"Running Dataset Validation Test on {data_path}")
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Load data
    try:
        data = data_processor.load_data(data_path)
        print(f"Successfully loaded data with shape {data.shape}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return False
    
    # Validate the data
    is_valid = data_processor.validate_data(data)
    
    if is_valid:
        print("Dataset validation passed!")
        return True
    else:
        print("Dataset validation failed!")
        return False


if __name__ == "__main__":
    # Use the correct dataset path
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/recommender_data.csv"
    
    result = test_dataset_validation(data_path)
    
    # Exit with appropriate code for CI
    sys.exit(0 if result else 1)