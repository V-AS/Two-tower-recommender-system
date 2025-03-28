# tests/system/test_model_storage.py
"""
System test for model storage (test-id3).
Verifies that the model and pre-computed item embeddings are stored correctly.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hardware.system_interface import load_model, load_embeddings

def test_model_storage(output_dir):
    """Test that models and embeddings were saved correctly and can be loaded."""
    print(f"Running Model Storage Test on output in {output_dir}")
    
    # Define expected file paths
    user_model_path = os.path.join(output_dir, "user_model.pth")
    item_model_path = os.path.join(output_dir, "item_model.pth")
    item_embeddings_path = os.path.join(output_dir, "item_embeddings.npy")
    ann_index_path = os.path.join(output_dir, "ann_index")
    
    # Check that all required files exist
    all_files_exist = True
    
    if not os.path.exists(user_model_path):
        print(f"User model not found at {user_model_path}")
        all_files_exist = False
    
    if not os.path.exists(item_model_path):
        print(f"Item model not found at {item_model_path}")
        all_files_exist = False
    
    if not os.path.exists(item_embeddings_path):
        print(f"Item embeddings not found at {item_embeddings_path}")
        all_files_exist = False
    
    if not os.path.exists(f"{ann_index_path}.faiss"):
        print(f"ANN index not found at {ann_index_path}.faiss")
        all_files_exist = False
    
    if not all_files_exist:
        return False
    
    # Try to load models and embeddings
    try:
        # Load user model
        print("Loading user model...")
        user_model = load_model(user_model_path)
        
        # Load item model
        print("Loading item model...")
        item_model = load_model(item_model_path)
        
        # Load item embeddings
        print("Loading item embeddings...")
        item_embeddings = load_embeddings(item_embeddings_path)
        
        # Check if models have the expected attributes
        if not hasattr(user_model, 'input_dim'):
            print("User model missing input_dim attribute")
            return False
        
        if not hasattr(item_model, 'input_dim'):
            print("Item model missing input_dim attribute")
            return False
        
        # Check if embeddings have the expected shape
        if len(item_embeddings.shape) != 2:
            print(f"Item embeddings have unexpected shape: {item_embeddings.shape}")
            return False
        
        print("Model storage test passed!")
        return True
    
    except Exception as e:
        print(f"Error loading models or embeddings: {e}")
        return False

if __name__ == "__main__":
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    
    result = test_model_storage(output_dir)
    
    # Exit with appropriate code for CI
    sys.exit(0 if result else 1)