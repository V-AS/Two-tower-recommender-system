"""
ANN Search Module.
Implements Approximate Nearest Neighbor search using FAISS.
Supports both exact search (IndexFlatIP) for small datasets and 
approximate nearest neighbor search (IndexIVFFlat) for larger datasets.
"""
import numpy as np
import faiss
import os

DEFAULT_SEARCH_NPROBE = 10
DEFAULT_INDEX_TYPE = "IVF"

class ANNSearch:
    def __init__(self):
        """Initialize the ANN Search module."""
        pass
    
    def build_index(self, embeddings, item_ids, index_type=DEFAULT_INDEX_TYPE):
        """
        Build an ANN index from embeddings.
        
        Args:
            embeddings (array-like): Item embeddings
            item_ids (array-like): IDs corresponding to embeddings
            index_type (str): Type of FAISS index to build
            
        Returns:
            dict: ANN index object containing the FAISS index and mappings
            
        Raises:
            ValueError: If parameters are invalid
        """
        if len(embeddings) != len(item_ids):
            raise ValueError("Number of embeddings and item IDs must match")
        
        if len(embeddings) == 0:
            raise ValueError("Cannot build index with empty embeddings")
        
        # Convert to numpy arrays
        embeddings = np.array(embeddings).astype('float32')
        item_ids = np.array(item_ids)
        
        # Get dimension
        d = embeddings.shape[1]
        
        # Create appropriate index based on type and size
        if index_type == "Flat":
            index = faiss.IndexFlatIP(d)  # Exact inner product search
        elif index_type == "IVF":
            nlist = min(int(np.sqrt(len(embeddings) * 5)), len(embeddings))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            # Need to train IVF index
            index.train(embeddings)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        index.add(embeddings)
        
        # Set search parameters
        if hasattr(index, 'nprobe'):
            index.nprobe = DEFAULT_SEARCH_NPROBE
        
        # Create the ANN index object
        ann_index = {
            'index': index,
            'item_ids': item_ids,
            'index_type': index_type
        }
        
        return ann_index
    
    def search(self, index, query, k=10):
        """
        Search the ANN index for nearest neighbors.
        
        Args:
            index (dict): ANN index object
            query (array-like): Query embedding
            k (int): Number of neighbors to retrieve
            
        Returns:
            list: List of (item_id, similarity_score) tuples
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(index, dict) or 'index' not in index or 'item_ids' not in index:
            raise ValueError("Invalid ANN index")
        
        if k <= 0:
            raise ValueError("k must be positive")
        
        # Convert query to numpy array
        query = np.array(query).astype('float32').reshape(1, -1)
        
        # Search the index
        distances, indices = index['index'].search(query, k)
        
        # Map indices to item IDs and create result tuples
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 for padded results
                item_id = index['item_ids'][idx]
                score = float(distances[0][i])
                results.append((item_id, score))
        
        return results
    
    def save_index(self, index, path):
        """
        Save an ANN index to disk.
        
        Args:
            index (dict): ANN index object
            path (str): Path to save to
            
        Returns:
            bool: True if successful
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save FAISS index
            faiss.write_index(index['index'], f"{path}.faiss")
            
            # Save metadata (item IDs and index type)
            metadata = {
                'item_ids': index['item_ids'],
                'index_type': index['index_type']
            }
            np.save(f"{path}.meta", metadata)
            
            return True
        except Exception as e:
            raise IOError(f"Failed to save index: {str(e)}")
    
    def load_index(self, path):
        """
        Load an ANN index from disk.
        
        Args:
            path (str): Path to load from
            
        Returns:
            dict: Loaded ANN index object
            
        Raises:
            IOError: If file cannot be read
            FormatError: If file format is invalid
        """
        try:
            # Check if files exist
            if not os.path.exists(f"{path}.faiss") or not os.path.exists(f"{path}.meta.npy"):
                raise IOError(f"Index files not found at {path}")
            
            # Load FAISS index
            index = faiss.read_index(f"{path}.faiss")
            
            # Load metadata
            metadata = np.load(f"{path}.meta.npy", allow_pickle=True).item()
            
            # Reconstruct ANN index object
            ann_index = {
                'index': index,
                'item_ids': metadata['item_ids'],
                'index_type': metadata['index_type']
            }
            
            return ann_index
        except Exception as e:
            if "Invalid" in str(e) or "format" in str(e).lower():
                raise ValueError(f"Invalid index format: {str(e)}")
            raise IOError(f"Failed to load index: {str(e)}")