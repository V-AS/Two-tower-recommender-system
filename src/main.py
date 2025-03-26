"""
Main execution script for the Two-Tower Embeddings Recommendation System.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch

from modules.data_processing import DataProcessor
from modules.neural_network import create_user_tower, create_item_tower
from modules.model_training import ModelTrainer
from modules.embedding_generation import EmbeddingGenerator
from modules.ann_search import ANNSearch
from modules.recommendation import Recommender
from hardware.system_interface import (
    save_model,
    load_model,
    save_embeddings,
    load_embeddings,
)
from utils.evaluation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-Tower Embedding Recommendation System"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save models and results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "recommend", "update"],
        default="train",
        help="Operation mode",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Training batch size"
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument("--user_id", type=str, help="User ID for recommendation mode")
    parser.add_argument(
        "--num_recommendations",
        type=int,
        default=10,
        help="Number of recommendations to generate",
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load and process data
    data_processor = DataProcessor()

    users_path = os.path.join(args.data_dir, "Users.csv")
    books_path = os.path.join(args.data_dir, "Books.csv")
    ratings_path = os.path.join(args.data_dir, "Ratings.csv")

    print("Loading data...")
    dataset = data_processor.load_data(users_path, books_path, ratings_path)

    # Validate data
    is_valid = data_processor.validate_data(dataset)
    if not is_valid:
        print("Invalid dataset format. Please check your data files.")
        return

    print("Preprocessing data...")
    processed_dataset = data_processor.preprocess_data(dataset)

    if args.mode == "train":
        # Split data
        print("Splitting data into training and testing sets...")
        train_data, test_data = data_processor.split_data(
            processed_dataset, train_ratio=0.8
        )

        # Create neural network architectures
        print("Creating neural network architectures...")

        # Create a sample feature vector to get the correct dimension
        trainer = ModelTrainer()

        sample_user_features = trainer._prepare_user_features(train_data["users"])
        sample_key = list(sample_user_features.keys())[0]
        user_input_dim = len(sample_user_features[sample_key])

        sample_item_features = trainer._prepare_item_features(train_data["books"])
        sample_key = list(sample_item_features.keys())[0]
        item_input_dim = len(sample_item_features[sample_key])

        print(f"Actual user feature dimension: {user_input_dim}")
        print(f"Actual item feature dimension: {item_input_dim}")

        user_tower = create_user_tower(
            input_dim=user_input_dim,
            hidden_layers=[256, 128],
            embedding_dim=args.embedding_dim,
        )

        item_tower = create_item_tower(
            input_dim=item_input_dim,
            hidden_layers=[256, 128],
            embedding_dim=args.embedding_dim,
        )

        # Initialize trainer
        print("Initializing model trainer...")
        trainer = ModelTrainer()
        trainer.initialize(
            {
                "user_architecture": user_tower,
                "item_architecture": item_tower,
                "learning_rate": 0.001,
                "batch_size": args.batch_size,
                "regularization": 0.01,
            }
        )

        # Train model
        print(f"Training model for {args.epochs} epochs...")
        model = trainer.train(train_data, epochs=args.epochs)

        # Evaluate model
        print("Evaluating model...")
        metrics = trainer.evaluate(test_data)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save models
        print("Saving models...")
        save_model(model["user_model"], os.path.join(args.output_dir, "user_model.pth"))
        save_model(model["item_model"], os.path.join(args.output_dir, "item_model.pth"))

        # Generate and save embeddings
        print("Generating embeddings...")
        embedding_generator = EmbeddingGenerator()
        embedding_generator.initialize(model["user_model"], model["item_model"])

        # Prepare item features for embedding generation

        item_features = []
        item_ids = []

        item_feature_dict = trainer._prepare_item_features(processed_dataset["books"])
        for item_id, features in item_feature_dict.items():
            item_features.append(features)
            item_ids.append(item_id)

        item_embeddings = embedding_generator.generate_item_embedding(
            np.array(item_features)
        )

        # Save item embeddings
        print("Saving item embeddings...")
        save_embeddings(
            item_embeddings, os.path.join(args.output_dir, "item_embeddings.npy")
        )
        np.save(os.path.join(args.output_dir, "item_ids.npy"), np.array(item_ids))

        # Build and save ANN index
        print("Building ANN index...")
        ann_search = ANNSearch()
        ann_index = ann_search.build_index(item_embeddings, np.array(item_ids))

        print("Saving ANN index...")
        ann_search.save_index(ann_index, os.path.join(args.output_dir, "ann_index"))

        print("Training complete!")

    elif args.mode == "evaluate":
        # Load models
        print("Loading models...")
        user_model = load_model(os.path.join(args.output_dir, "user_model.pth"))
        item_model = load_model(os.path.join(args.output_dir, "item_model.pth"))

        # Load ANN index
        print("Loading ANN index...")
        ann_search = ANNSearch()
        ann_index = ann_search.load_index(os.path.join(args.output_dir, "ann_index"))

        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        embedding_generator.initialize(user_model, item_model)

        # Initialize recommender
        recommender = Recommender()
        recommender.initialize(
            ann_index, embedding_generator, processed_dataset["books"]
        )

        # Evaluate recommendations
        print("Evaluating recommendations...")
        train_data, test_data = data_processor.split_data(
            processed_dataset, train_ratio=0.8
        )
        metrics = recommender.evaluate_recommendations(test_data)

        print("Recommendation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

    elif args.mode == "recommend":
        # Check if user ID is provided
        if args.user_id is None:
            print("Please provide a user ID with --user_id")
            return

        # Load models
        print("Loading models...")
        user_model = load_model(os.path.join(args.output_dir, "user_model.pth"))
        item_model = load_model(os.path.join(args.output_dir, "item_model.pth"))

        # Load ANN index
        print("Loading ANN index...")
        ann_search = ANNSearch()
        ann_index = ann_search.load_index(os.path.join(args.output_dir, "ann_index"))

        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        embedding_generator.initialize(user_model, item_model)

        # Initialize recommender
        recommender = Recommender()
        recommender.initialize(
            ann_index, embedding_generator, processed_dataset["books"]
        )

        # Get user
        users_df = processed_dataset["users"]

        # Find user by ID
        user_row = users_df[users_df["User-ID"] == args.user_id]
        if len(user_row) == 0:
            print(f"User with ID {args.user_id} not found")
            return

        # Get user features
        trainer = ModelTrainer()
        user_features = trainer._prepare_user_features(users_df)
        user_id_encoded = user_row["User-ID-Encoded"].values[0]
        user_feature = user_features[user_id_encoded]

        # Generate recommendations
        print(
            f"Generating {args.num_recommendations} recommendations for user {args.user_id}..."
        )
        recommendations = recommender.get_recommendations(
            user_feature, num_results=args.num_recommendations
        )

        # Display recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(
                f"{i}. {rec.get('title', 'Unknown')} by {rec.get('author', 'Unknown')} (Score: {rec['score']:.4f})"
            )

    elif args.mode == "update":
        # Load models
        print("Loading models...")
        user_model = load_model(os.path.join(args.output_dir, "user_model.pth"))
        item_model = load_model(os.path.join(args.output_dir, "item_model.pth"))

        # Initialize trainer
        print("Initializing model trainer...")
        trainer = ModelTrainer()
        trainer.initialize(
            {
                "user_architecture": user_model,
                "item_architecture": item_model,
                "learning_rate": 0.0005,  # Lower learning rate for fine-tuning
                "batch_size": args.batch_size,
                "regularization": 0.01,
            }
        )

        # Split data for incremental update (simulating new data)
        print("Preparing data for incremental update...")
        initial_data, new_data = data_processor.split_data(
            processed_dataset, train_ratio=0.7
        )

        # Update model with new data
        print("Updating model with new data...")
        updated_model = trainer.update_model(new_data)

        # Save updated models
        print("Saving updated models...")
        save_model(
            updated_model["user_model"],
            os.path.join(args.output_dir, "user_model_updated.pth"),
        )
        save_model(
            updated_model["item_model"],
            os.path.join(args.output_dir, "item_model_updated.pth"),
        )

        print("Model successfully updated!")


if __name__ == "__main__":
    main()
