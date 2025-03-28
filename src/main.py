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
    save_training_history
)
from utils.evaluation import evaluate_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Two-Tower Embedding Recommendation System"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/recommender_data.csv",
        help="Path to the processed data CSV file",
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
        "--embedding_dim", type=int, default=32, help="Embedding dimension"
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

    # Initialize data processor
    data_processor = DataProcessor()

    print("Loading data...")
    data = data_processor.load_data(args.data_path)

    # Validate data
    is_valid = data_processor.validate_data(data)
    if not is_valid:
        print("Invalid dataset format. Please check your data file.")
        return

    print("Preprocessing data...")
    processed_data = data_processor.preprocess_data(data)

    if args.mode == "train":
        # Split data
        print("Splitting data into training and testing sets...")
        train_data, test_data = data_processor.split_data(processed_data, train_ratio=0.8)

        # Prepare training dataset
        print("Preparing training dataset...")
        training_dataset = data_processor.create_training_data(train_data)
        
        # Create neural network architectures
        print("Creating neural network architectures...")
        
        # Get input dimensions from the feature arrays
        user_input_dim = training_dataset['user_features'].shape[1]
        item_input_dim = training_dataset['item_features'].shape[1]
        
        print(f"User feature dimension: {user_input_dim}")
        print(f"Item feature dimension: {item_input_dim}")

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
        model = trainer.train(training_dataset, epochs=args.epochs)

        # Prepare test dataset for evaluation
        testing_dataset = data_processor.create_training_data(test_data)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = trainer.evaluate(testing_dataset)
        print("Evaluation metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save models
        print("Saving models...")
        save_model(model["user_model"], os.path.join(args.output_dir, "user_model.pth"))
        save_model(model["item_model"], os.path.join(args.output_dir, "item_model.pth"))
        # Save training history
        save_training_history(
            model["training_history"],
            os.path.join(args.output_dir, "training_history.json")
        )
        # Generate and save embeddings
        print("Generating embeddings...")
        embedding_generator = EmbeddingGenerator()
        embedding_generator.initialize(model["user_model"], model["item_model"])
        
        # Get unique books for embedding generation
        unique_books = processed_data.drop_duplicates('Book-Title-Encoded')
        

        # Extract item features for all unique books
        item_features = np.column_stack((
            unique_books['Year-Normalized'].values,
            unique_books['Author-Frequency'].values,
            unique_books['Publisher-Frequency'].values,
            unique_books['Decade'].values
        ))
        
        item_ids = unique_books['Book-Title-Encoded'].values
        
        # Generate embeddings
        item_embeddings = embedding_generator.generate_item_embedding(item_features)

        # Save item embeddings
        print("Saving item embeddings...")
        save_embeddings(
            item_embeddings, os.path.join(args.output_dir, "item_embeddings.npy")
        )
        np.save(os.path.join(args.output_dir, "item_ids.npy"), np.array(item_ids))
        
        # Save book mapping for recommendation lookups
        book_mapping = data_processor.get_book_mapping(processed_data)
        np.save(os.path.join(args.output_dir, "book_mapping.npy"), book_mapping)

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
        
        # Split data for evaluation
        _, test_data = data_processor.split_data(processed_data, train_ratio=0.8)
        
        # Prepare test dataset
        testing_dataset = data_processor.create_training_data(test_data)
        
        # Initialize trainer for evaluation
        trainer = ModelTrainer()
        trainer.initialize(
            {
                "user_architecture": user_model,
                "item_architecture": item_model,
                "learning_rate": 0.001,
                "batch_size": args.batch_size,
                "regularization": 0.01,
            }
        )
        
        # Evaluate model
        print("Evaluating model...")
        metrics = trainer.evaluate(testing_dataset)
        print("Evaluation metrics:")
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
        
        # Load book mapping
        book_mapping = np.load(os.path.join(args.output_dir, "book_mapping.npy"), allow_pickle=True).item()

        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        embedding_generator.initialize(user_model, item_model)

        # Find user by ID
        user_row = processed_data[processed_data['User-ID'] == int(args.user_id)]
        if len(user_row) == 0:
            print(f"User with ID {args.user_id} not found")
            return

        # Get user features
        user_features = np.array([
            user_row['Age-Normalized'].values[0],
            user_row['Age-Group'].values[0],
            user_row['State-Frequency'].values[0],
            user_row['Country-Frequency'].values[0]
        ])
        
        # Initialize recommender
        recommender = Recommender()
        recommender.initialize(ann_index, embedding_generator, book_mapping)

        # Generate recommendations
        print(f"Generating {args.num_recommendations} recommendations for user {args.user_id}...")
        recommendations = recommender.get_recommendations(
            user_features, num_results=args.num_recommendations
        )

        # Display recommendations
        print("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec.get('title', 'Unknown')} by {rec.get('author', 'Unknown')} (Estimated Rating: {rec['score']:.4f})")

    elif args.mode == "update":
        # Load models
        print("Loading models...")
        user_model = load_model(os.path.join(args.output_dir, "user_model.pth"))
        item_model = load_model(os.path.join(args.output_dir, "item_model.pth"))

        # Split data for incremental update (simulating new data)
        print("Preparing data for incremental update...")
        _, new_data = data_processor.split_data(processed_data, train_ratio=0.7)
        
        # Create training dataset
        training_dataset = data_processor.create_training_data(new_data)

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

        # Update model with new data
        print("Updating model with new data...")
        updated_model = trainer.update_model(training_dataset, epochs=5)

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