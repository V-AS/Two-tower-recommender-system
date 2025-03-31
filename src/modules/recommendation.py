"""
Modified Recommendation Module.
Generates recommendations based on user embeddings and ANN search.
"""
import numpy as np
from modules.vector_operations import dot_product

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
        candidate_embeddings, candidate_ids = ann_search.ann_search(self.ann_index, user_embedding)
        
        refined_results = []
        query_vector = user_embedding[0]  # Remove the batch dimension
        
        for i, emb in enumerate(candidate_embeddings):
            item_id = candidate_ids[i]
            # Calculate exact dot product
            exact_score = dot_product(query_vector, emb)
            estimated_rating = (float(exact_score) + 1) * 5
            refined_results.append((item_id, estimated_rating))
            # Removed the debug print statement that was here
        
        # Sort by score (highest first) and take top final_k
        refined_results = sorted(refined_results, key=lambda x: x[1], reverse=True)[:num_results]
        
        # Enhance results with book details if available
        recommendations = []
        for item_id, score in refined_results:
            rec = {
                'item_id': item_id,
                'score': score
            }
            
            # Add book details if available
            if item_id in self.book_lookup:
                rec.update(self.book_lookup[item_id])
            
            recommendations.append(rec)
        
        return recommendations
    