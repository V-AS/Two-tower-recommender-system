"""
Evaluation Module.
Provides functions for model and recommendation evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(predictions, targets):
    """
    Evaluate model predictions against targets.
    
    Args:
        predictions: Predicted ratings
        targets: True ratings
        
    Returns:
        dict: Evaluation metrics
    """
    # Mean Squared Error
    mse = mean_squared_error(targets, predictions)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = mean_absolute_error(targets, predictions)
    
    # % of predictions within 10% of true value
    accuracy = np.mean(np.abs(predictions - targets) <= 0.1)
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy
    }
    
    return metrics

def evaluate_recommendations(recommendations, ground_truth, k_values=[5, 10]):
    """
    Evaluate recommendation quality.
    
    Args:
        recommendations: List of recommended item IDs for each user
        ground_truth: Dictionary of relevant item IDs for each user
        k_values: List of k values for metrics@k
        
    Returns:
        dict: Recommendation metrics
    """
    metrics = {}
    
    # Calculate Precision@k, Recall@k, NDCG@k
    precision_at_k = []
    recall_at_k = []
    ndcg_at_k = []
    
    for user_id in recommendations:
        if user_id not in ground_truth:
            continue
        
        rec_items = recommendations[user_id]
        relevant_items = ground_truth[user_id]
        
        for k in k_values:
            # Precision@k
            hits = sum(1 for item_id in rec_items[:k] if item_id in relevant_items)
            precision = hits / k if k > 0 else 0
            precision_at_k.append((k, precision))
            
            # Recall@k
            recall = hits / len(relevant_items) if relevant_items else 0
            recall_at_k.append((k, recall))
            
            # NDCG@k
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
            dcg = sum(1.0 / np.log2(i + 2) for i, item_id in enumerate(rec_items[:k]) if item_id in relevant_items)
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k.append((k, ndcg))
    
    # Aggregate metrics
    for k in k_values:
        metrics[f'precision@{k}'] = np.mean([p for k_val, p in precision_at_k if k_val == k])
        metrics[f'recall@{k}'] = np.mean([r for k_val, r in recall_at_k if k_val == k])
        metrics[f'ndcg@{k}'] = np.mean([n for k_val, n in ndcg_at_k if k_val == k])
    
    return metrics

def plot_learning_curves(history):
    """
    Plot learning curves from training history.
    
    Args:
        history: Training history with loss and validation loss
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('output/learning_curves.png')
    plt.close()