"""
Modified Recommendation Module.
Generates recommendations based on user embeddings and ANN search.
"""
import numpy as np

DEFAULT_NUM_RECOMMENDATIONS = 10
SIMILARITY_THRESHOLD = 0.5

class Recommender:
    def __init__(self):
        """Initialize the Recommender."""
        self.ann_index = None
        self.embedding_generator = None
        self.is_initialized = False
        self.book_lookup = {}  # Book ID to book details mapping
    
    def initialize(self, ann_index, embedding_generator=None, book_lookup=None):
        """
        Initialize the recommender.
        
        Args:
            ann_index: ANN index for item search
            embedding_generator: Optional embedding generator
            book_lookup: Optional book details lookup dictionary
            
        Raises:
            ValueError: If ann_index is invalid
        """
        if not isinstance(ann_index, dict) or 'index' not in ann_index:
            raise ValueError("Invalid ANN index")
        
        self.ann_index = ann_index
        self.embedding_generator = embedding_generator
        self.book_lookup = book_lookup if book_lookup is not None else {}
        
        self.is_initialized = True
    
    def get_recommendations(self, user, num_results=DEFAULT_NUM_RECOMMENDATIONS):
        """
        Get recommendations for a user.
        
        Args:
            user: User features or embedding
            num_results (int): Number of recommendations to return
            
        Returns:
            list: Ranked list of recommendations
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Recommender not initialized")
        
        # If user is not already an embedding and we have an embedding generator
        if self.embedding_generator is not None and isinstance(user, (list, np.ndarray)) and len(np.array(user).shape) == 1:
            # User is raw features, generate embedding
            user_embedding = self.embedding_generator.generate_user_embedding([user])[0]
        else:
            # User is already an embedding
            user_embedding = user
        
        # Query ANN index
        from modules.ann_search import ANNSearch
        ann_search = ANNSearch()
        results = ann_search.search(self.ann_index, user_embedding, k=num_results)
        
        # Enhance results with book details if available
        recommendations = []
        for item_id, score in results:
            rec = {
                'item_id': item_id,
                'score': score
            }
            
            # Add book details if available
            if item_id in self.book_lookup:
                rec.update(self.book_lookup[item_id])
            
            recommendations.append(rec)
        
        return recommendations
    
    def evaluate_recommendations(self, test_data, k_values=[5, 10]):
        """
        Evaluate recommendations on test data.
        
        Args:
            test_data: Test dataset
            k_values: List of k values for evaluation metrics
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Recommender not initialized")
        
        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator required for evaluation")
        
        # Calculate metrics for precision, recall, and NDCG at different k values
        metrics = {}
        precision_sum = {k: 0 for k in k_values}
        recall_sum = {k: 0 for k in k_values}
        ndcg_sum = {k: 0 for k in k_values}
        user_count = 0
        
        # Group test data by user
        user_groups = {}
        for user_id in test_data['user_ids']:
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(user_id)
        
        # For each user in test set
        for user_id, indices in user_groups.items():
            # Get user features
            user_feature = test_data['user_features'][indices[0]]
            
            # Get relevant items (items the user actually liked with rating >= 0.7)
            relevant_items = set()
            for idx in indices:
                if test_data['ratings'][idx] >= 0.7:  # 7/10 rating or higher
                    relevant_items.add(test_data['item_ids'][idx])
            
            if not relevant_items:
                continue
            
            # Get recommendations
            recommendations = self.get_recommendations(user_feature, num_results=max(k_values))
            rec_item_ids = [rec['item_id'] for rec in recommendations]
            
            # Calculate metrics for each k
            for k in k_values:
                # Precision@k
                hits = sum(1 for item_id in rec_item_ids[:k] if item_id in relevant_items)
                precision = hits / k if k > 0 else 0
                precision_sum[k] += precision
                
                # Recall@k
                recall = hits / len(relevant_items) if relevant_items else 0
                recall_sum[k] += recall
                
                # NDCG@k
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
                dcg = sum(1.0 / np.log2(i + 2) for i, item_id in enumerate(rec_item_ids[:k]) if item_id in relevant_items)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_sum[k] += ndcg
            
            user_count += 1
        
        # Calculate average metrics
        if user_count > 0:
            for k in k_values:
                metrics[f'precision@{k}'] = precision_sum[k] / user_count
                metrics[f'recall@{k}'] = recall_sum[k] / user_count
                metrics[f'ndcg@{k}'] = ndcg_sum[k] / user_count
        
        return metrics