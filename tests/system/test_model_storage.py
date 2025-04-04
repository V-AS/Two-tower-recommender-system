# tests/system/test_model_storage.py
"""
System test for model storage (test-id3).
Verifies that the model and pre-computed item embeddings are stored correctly.
"""
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.hardware.system_interface import load_model, load_embeddings

import pytest


@pytest.fixture
def output_dir():
    return "output"


def test_model_storage(output_dir):
    """Test that models and embeddings were saved correctly and can be loaded."""
    print(f"Running Model Storage Test on files in {output_dir}")
    # Define expected file paths
    user_model_path = os.path.join(output_dir, "user_model.pth")
    item_model_path = os.path.join(output_dir, "item_model.pth")
    item_embeddings_path = os.path.join(output_dir, "item_embeddings.npy")
    ann_index_path = os.path.join(output_dir, "ann_index")

    # Check that all required files exist
    assert os.path.exists(user_model_path), f"User model not found at {user_model_path}"
    assert os.path.exists(item_model_path), f"Item model not found at {item_model_path}"
    assert os.path.exists(
        item_embeddings_path
    ), f"Item embeddings not found at {item_embeddings_path}"
    assert os.path.exists(
        f"{ann_index_path}.faiss"
    ), f"ANN index not found at {ann_index_path}.faiss"

    # Try to load models and embeddings
    try:
        # Load user model state dict
        print("Loading user model...")
        user_model_state = load_model(user_model_path)

        # Load item model state dict
        print("Loading item model...")
        item_model_state = load_model(item_model_path)

        # Load item embeddings
        print("Loading item embeddings...")
        item_embeddings = load_embeddings(item_embeddings_path)

        # Validate model state dicts contain expected elements
        assert isinstance(
            user_model_state, dict
        ), "User model is not a state dictionary"
        assert isinstance(
            item_model_state, dict
        ), "Item model is not a state dictionary"

        # Check if state dicts have expected layer parameters
        assert any(
            "weight" in key for key in user_model_state.keys()
        ), "User model state dict missing expected weight parameters"
        assert any(
            "weight" in key for key in item_model_state.keys()
        ), "Item model state dict missing expected weight parameters"

        # Check if embeddings have the expected shape
        assert (
            len(item_embeddings.shape) == 2
        ), f"Item embeddings have unexpected shape: {item_embeddings.shape}"

        # Print some stats about the loaded files for verification
        print(f"User model state dict contains {len(user_model_state)} parameters")
        print(f"Item model state dict contains {len(item_model_state)} parameters")
        print(f"Item embeddings shape: {item_embeddings.shape}")
        print("Model storage test passed!")
        return True
    except Exception as e:
        print(f"Error loading models or embeddings: {e}")
        raise


if __name__ == "__main__":
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"

    result = test_model_storage(output_dir)

    # Exit with appropriate code for CI
    sys.exit(0 if result else 1)
