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
DEFAULT_INDEX_TYPE = "Flat"

class ANNSearch:
    def __init__(self):
        """Initialize the ANN Search module."""
        pass
    
    def build_index(self, embeddings, item_ids, index_type=DEFAULT_INDEX_TYPE):
        """
        Build an improved ANN index from embeddings with better configuration.
        
        Args:
            embeddings (array-like): Item embeddings
            item_ids (array-like): IDs corresponding to embeddings
            index_type (str): Type of FAISS index to build
            
        Returns:
            dict: ANN index object containing the FAISS index and mappings
            
        Raises:
            ValueError: If parameters are invalid
        """
        try:
            # Existing validation
            if len(embeddings) != len(item_ids):
                raise ValueError("Number of embeddings and item IDs must match")
            
            if len(embeddings) == 0:
                raise ValueError("Cannot build index with empty embeddings")
            
            # Force contiguous arrays and check for NaN/Inf values
            embeddings = np.ascontiguousarray(embeddings).astype('float32')
            
            # Check for NaN or infinite values
            if np.isnan(embeddings).any() or np.isinf(embeddings).any():
                print("Warning: NaN or infinite values detected in embeddings")
                # Replace NaN/Inf with zeros
                embeddings = np.nan_to_num(embeddings)
            
            item_ids = np.array(item_ids)
        
            # Get dimension
            d = embeddings.shape[1]
        
            # Print statistics about the embeddings for diagnostic purposes
            print(f"Building index with {len(embeddings)} embeddings of dimension {d}")
            print(f"Embedding stats - min: {embeddings.min():.6f}, max: {embeddings.max():.6f}")
            print(f"Embedding std: {embeddings.std():.6f}, mean: {embeddings.mean():.6f}")
            
            # Normalize embeddings again to ensure they're ready for inner product search
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            # Create appropriate index based on type and size
            if index_type == "Flat":
                # Use IndexFlatIP for exact inner product search (cosine similarity on normalized vectors)
                index = faiss.IndexFlatIP(d)  
                index.add(embeddings)
            elif index_type == "IVF":
                # Improved IVF configuration
                # More clusters for better granularity - sqrt(n) is a common heuristic
                nlist = min(int(np.sqrt(len(embeddings)) * 4), len(embeddings) // 10 or 1)
                nlist = max(nlist, 8)  # Ensure at least 8 clusters
                
                quantizer = faiss.IndexFlatIP(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
                print(f"Training IVF index with {nlist} clusters...")
                
                # Need to train IVF index
                index.train(embeddings)
                index.add(embeddings)
                
                # Set nprobe higher for better recall
                index.nprobe = min(nlist // 4 + 1, 16)  # Higher nprobe for better accuracy
                print(f"Set nprobe to {index.nprobe}")
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            # Create the ANN index object
            ann_index = {
                'index': index,
                'item_ids': item_ids,
                'index_type': index_type
            }
            
            return ann_index
        except Exception as e:
            print(f"Error during index building: {str(e)}")
            raise
        
    def ann_search(self, index, query, candidates=100):
        """

        Use FAISS to get a larger set of candidate neighbors
        
        Args:
            index (dict): ANN index object
            query (array-like): Query embedding
            candidates (int): Number of candidates to retrieve in first stage
            
        Returns:
            list: List of (item_id, similarity_score) tuples
        """
        if not isinstance(index, dict) or 'index' not in index or 'item_ids' not in index:
            raise ValueError("Invalid ANN index")

        # Convert query to numpy array
        query = np.array(query).astype('float32').reshape(1, -1)

        # Stage 1: Get candidate set using FAISS search
        distances, indices = index['index'].search(query, candidates)
        
        # Get the embeddings for these candidates
        candidate_embeddings = []
        candidate_ids = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS returns -1 for padded results
                item_id = index['item_ids'][idx]
                candidate_ids.append(item_id)
                # For a Flat index, we can reconstruct the embedding
                if index['index_type'] == "Flat":
                    embedding = index['index'].reconstruct(int(idx))
                    candidate_embeddings.append(embedding)
        
        return candidate_embeddings, candidate_ids
        
    
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