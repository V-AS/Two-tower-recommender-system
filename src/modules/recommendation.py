"""
Recommendation Module.
Generates recommendations based on user embeddings and ANN search.
"""
import numpy as np
from tqdm import tqdm

DEFAULT_NUM_RECOMMENDATIONS = 10
SIMILARITY_THRESHOLD = 0.5

class Recommender:
    def __init__(self):
        """Initialize the Recommender."""
        self.ann_index = None
        self.embedding_generator = None
        self.is_initialized = False
        self.book_lookup = {}  # ISBN-Encoded to book details mapping
    
    def initialize(self, ann_index, embedding_generator=None, books_df=None):
        """
        Initialize the recommender.
        
        Args:
            ann_index: ANN index for item search
            embedding_generator: Optional embedding generator
            books_df: Optional books dataframe for lookup
            
        Raises:
            ValueError: If ann_index is invalid
        """
        if not isinstance(ann_index, dict) or 'index' not in ann_index:
            raise ValueError("Invalid ANN index")
        
        self.ann_index = ann_index
        self.embedding_generator = embedding_generator
        
        # Create book lookup if books_df is provided
        if books_df is not None:
            for _, row in books_df.iterrows():
                self.book_lookup[row['ISBN-Encoded']] = {
                    'isbn': row['ISBN'],
                    'title': row['Book-Title'],
                    'author': row['Book-Author']
                }
        
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
        if self.embedding_generator is not None and isinstance(user, (list, np.ndarray)) and len(user.shape) != 1:
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
    
    def evaluate_recommendations(self, test_data):
        """
        Evaluate recommendations on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            RuntimeError: If not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Recommender not initialized")
        
        if self.embedding_generator is None:
            raise RuntimeError("Embedding generator required for evaluation")
        
        # Extract relevant data
        users_df = test_data['users']
        ratings_df = test_data['ratings']
        
        # Prepare user features
        from modules.model_training import ModelTrainer
        trainer = ModelTrainer()
        user_features = trainer._prepare_user_features(users_df)
        
        # Get unique users in test set
        test_users = ratings_df['User-ID-Encoded'].unique()
        
        # Evaluation metrics
        precision_at_k = []
        recall_at_k = []
        ndcg_at_k = []
        
        k_values = [5, 10]
        
        # For each user in test set
        for user_id in tqdm(test_users, desc="Evaluating recommendations"):
            # Get user's actual ratings
            user_ratings = ratings_df[ratings_df['User-ID-Encoded'] == user_id]
            
            # Get user features
            user_feature = user_features[user_id]
            
            # Get recommendations
            recommendations = self.get_recommendations(user_feature, num_results=max(k_values))
            rec_item_ids = [rec['item_id'] for rec in recommendations]
            
            # Get items user rated highly (rating >= 8)
            relevant_items = set(user_ratings[user_ratings['Book-Rating'] >= 8]['ISBN-Encoded'].values)
            
            # Calculate metrics for each k
            for k in k_values:
                # Precision@k
                hits = sum(1 for item_id in rec_item_ids[:k] if item_id in relevant_items)
                precision = hits / k if k > 0 else 0
                precision_at_k.append((k, precision))
                
                # Recall@k
                recall = hits / len(relevant_items) if relevant_items else 0
                recall_at_k.append((k, recall))
                
                # NDCG@k
                idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
                dcg = sum(1.0 / np.log2(i + 2) for i, item_id in enumerate(rec_item_ids[:k]) if item_id in relevant_items)
                ndcg = dcg / idcg if idcg > 0 else 0
                ndcg_at_k.append((k, ndcg))
        
        # Aggregate metrics
        metrics = {}
        for k in k_values:
            metrics[f'precision@{k}'] = np.mean([p for k_val, p in precision_at_k if k_val == k])
            metrics[f'recall@{k}'] = np.mean([r for k_val, r in recall_at_k if k_val == k])
            metrics[f'ndcg@{k}'] = np.mean([n for k_val, n in ndcg_at_k if k_val == k])
        
        return metrics