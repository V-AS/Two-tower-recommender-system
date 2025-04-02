"""
Configuration Module.
Contains default parameters and configurations.
"""

# Default parameters for neural network architecture
NN_CONFIG = {"hidden_layers": [256, 128], "activation": "relu", "embedding_dim": 128}

# Default parameters for training
TRAINING_CONFIG = {
    "learning_rate": 0.01,
    "batch_size": 128,
    "epochs": 10,
    "regularization": 0.01,
    "train_ratio": 0.8,
}

# Default parameters for ANN search
ANN_CONFIG = {"index_type": "IVF", "nprobe": 10}

# Default parameters for recommendation
REC_CONFIG = {"num_recommendations": 10, "similarity_threshold": 0.5}

# File paths
DATA_PATHS = {
    "users": "data/processed/Users.csv",
    "books": "data/processed/Books.csv",
    "ratings": "data/processed/Ratings.csv",
}

# Output paths
OUTPUT_PATHS = {
    "user_model": "output/user_model.pth",
    "item_model": "output/item_model.pth",
    "item_embeddings": "output/item_embeddings.npy",
    "ann_index": "output/ann_index",
}
