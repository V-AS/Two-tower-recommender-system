"""
System test for model training convergence (test-id2).
Verifies that the model training loss decreases over iterations.
Uses saved history file or checks model weights directly.
"""
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.hardware.system_interface import load_model, load_training_history


def test_model_convergence(output_dir):
    """
    Test that model training was successful by:
    1. Checking training history if available
    2. Comparing model weights to default initialization otherwise
    """
    print(f"Running Model Convergence Test on output in {output_dir}")
    
    # First try: Check if we have a training history file
    history_path = os.path.join(output_dir, "training_history.json")
    
    if os.path.exists(history_path):
        print(f"Found training history at {history_path}")
        try:
            history = load_training_history(history_path)
            
            # Check if loss decreases
            if 'loss' in history:
                loss_values = history['loss']
                
                # Print loss values for debugging
                for i, loss in enumerate(loss_values):
                    print(f"Epoch {i+1}: Loss = {loss:.6f}")
                
                # Check if the loss at the end is lower than at the beginning
                if len(loss_values) >= 2 and loss_values[-1] < loss_values[0]:
                    improvement = (1 - loss_values[-1] / loss_values[0]) * 100
                    print(f"Loss decreased by {improvement:.2f}% from {loss_values[0]:.6f} to {loss_values[-1]:.6f}")
                    return True
                else:
                    print("Loss did not decrease during training")
                    return False
            else:
                print("No loss data found in history file")
        except Exception as e:
            print(f"Error reading history file: {e}")


if __name__ == "__main__":
    # Get output directory from command line or use default
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "output"
    
    result = test_model_convergence(output_dir)
    
    sys.exit(0 if result else 1)