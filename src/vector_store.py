"""
Vector Store Module
Handles FAISS vector database operations for similarity search
"""

import os
import logging
from typing import List, Tuple, Optional
import numpy as np
import faiss
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search
    """
    
    def __init__(self, dimension: int, index_type: str = "IndexFlatIP"):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Dimension of the embeddings
            index_type: Type of FAISS index to use
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata_list = []
        self._create_index()
    
    def _create_index(self):
        """
        Create FAISS index based on specified type
        """
        try:
            if self.index_type == "IndexFlatIP":
                # Inner Product (cosine similarity for normalized vectors)
                self.index = faiss.IndexFlatIP(self.dimension)
            elif self.index_type == "IndexFlatL2":
                # L2 distance
                self.index = faiss.IndexFlatL2(self.dimension)
            else:
                # Default to Inner Product
                self.index = faiss.IndexFlatIP(self.dimension)
            
            logger.info(f"Created FAISS index: {self.index_type} with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {str(e)}")
            raise
    
    def add_embeddings(self, embeddings: np.ndarray, metadata_list: List[dict]):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: NumPy array of embeddings
            metadata_list: List of metadata dictionaries corresponding to embeddings
        """
        if embeddings.shape[0] != len(metadata_list):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        try:
            # Add embeddings to index
            embeddings_float32 = embeddings.astype('float32')
            self.index.add(embeddings_float32)
            
            # Store metadata
            self.metadata_list.extend(metadata_list)
            
            logger.info(f"Added {len(embeddings)} embeddings to vector store")
            
        except Exception as e:
            logger.error(f"Failed to add embeddings to vector store: {str(e)}")
            raise
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[dict]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            k: Number of similar results to return
            
        Returns:
            Tuple of (scores, metadata_list) for top-k results
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty")
            return [], []
        
        try:
            # Ensure query embedding has correct shape and type
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            query_embedding = query_embedding.astype('float32')
            
            # Perform search
            scores, indices = self.index.search(query_embedding, min(k, self.index.ntotal))
            
            # Extract results
            result_scores = scores[0].tolist()
            result_metadata = [self.metadata_list[idx] for idx in indices[0] if idx < len(self.metadata_list)]
            
            logger.info(f"Found {len(result_metadata)} similar documents")
            return result_scores, result_metadata
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {str(e)}")
            return [], []
    
    def get_relevant_documents(self, query_embedding: np.ndarray, k: int = 5, 
                              score_threshold: float = 0.0) -> List[dict]:
        """
        Get relevant documents based on similarity search
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant document metadata with scores
        """
        scores, metadata_list = self.search(query_embedding, k)
        
        logger.info(f"Vector search found {len(scores)} results with scores: {scores}")
        
        # Filter by score threshold and add scores to metadata
        relevant_docs = []
        for i, (score, metadata) in enumerate(zip(scores, metadata_list)):
            logger.info(f"Result {i+1}: Score={score:.3f}, Metadata keys: {list(metadata.keys())}")
            if 'page_content' in metadata:
                logger.info(f"Content preview: {metadata['page_content'][:150]}...")
            
            if score >= score_threshold:
                enhanced_metadata = metadata.copy()
                enhanced_metadata['similarity_score'] = score
                relevant_docs.append(enhanced_metadata)
            else:
                logger.info(f"Filtered out result with score {score:.3f} (below threshold {score_threshold})")
        
        logger.info(f"Returning {len(relevant_docs)} documents after filtering")
        return relevant_docs
    
    def save_index(self, base_path: str = "vectorstore"):
        """
        Save FAISS index and metadata to files
        
        Args:
            base_path: Base directory to save files
        """
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(base_path, "faiss.index")
            faiss.write_index(self.index, index_path)
            
            # Save metadata
            metadata_path = os.path.join(base_path, "documents.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata_list, f)
            
            # Save index configuration
            config_path = os.path.join(base_path, "config.pkl")
            config = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'total_vectors': self.index.ntotal
            }
            with open(config_path, 'wb') as f:
                pickle.dump(config, f)
            
            logger.info(f"Vector store saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {str(e)}")
            raise
    
    def load_index(self, base_path: str = "vectorstore") -> bool:
        """
        Load FAISS index and metadata from files
        
        Args:
            base_path: Base directory containing saved files
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = os.path.join(base_path, "faiss.index")
            metadata_path = os.path.join(base_path, "documents.pkl")
            config_path = os.path.join(base_path, "config.pkl")
            
            if not all(os.path.exists(path) for path in [index_path, metadata_path, config_path]):
                logger.warning(f"Vector store files not found in {base_path}")
                return False
            
            # Load configuration
            with open(config_path, 'rb') as f:
                config = pickle.load(f)
            
            # Verify configuration matches
            if config['dimension'] != self.dimension:
                logger.error(f"Dimension mismatch: expected {self.dimension}, got {config['dimension']}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata_list = pickle.load(f)
            
            logger.info(f"Vector store loaded from {base_path}")
            logger.info(f"Loaded {self.index.ntotal} vectors with {len(self.metadata_list)} metadata entries")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary containing vector store statistics
        """
        return {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'total_vectors': self.index.ntotal if self.index else 0,
            'metadata_count': len(self.metadata_list),
            'is_trained': self.index.is_trained if self.index else False
        }
    
    def clear(self):
        """
        Clear all vectors and metadata from the store
        """
        self._create_index()
        self.metadata_list = []
        logger.info("Vector store cleared")


class VectorStoreManager:
    """
    High-level manager for vector store operations
    """
    
    def __init__(self, dimension: int, base_path: str = "vectorstore"):
        """
        Initialize vector store manager
        
        Args:
            dimension: Embedding dimension
            base_path: Base path for storing vector store files
        """
        self.dimension = dimension
        self.base_path = base_path
        self.vector_store = FAISSVectorStore(dimension)
        
        # Try to load existing index
        self.load_or_create()
    
    def load_or_create(self):
        """
        Load existing vector store or create new one
        """
        if not self.vector_store.load_index(self.base_path):
            logger.info("Creating new vector store")
            os.makedirs(self.base_path, exist_ok=True)
    
    def add_documents(self, embeddings: np.ndarray, metadata_list: List[dict]):
        """
        Add documents to vector store
        
        Args:
            embeddings: Document embeddings
            metadata_list: Document metadata
        """
        self.vector_store.add_embeddings(embeddings, metadata_list)
        self.save()
    
    def search_documents(self, query_embedding: np.ndarray, k: int = 5, 
                        score_threshold: float = 0.0) -> List[dict]:
        """
        Search for relevant documents
        
        Args:
            query_embedding: Query embedding
            k: Number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        return self.vector_store.get_relevant_documents(query_embedding, k, score_threshold)
    
    def save(self):
        """
        Save vector store to disk
        """
        self.vector_store.save_index(self.base_path)
    
    def get_statistics(self) -> dict:
        """
        Get vector store statistics
        
        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_stats()


def main():
    """
    Test the vector store
    """
    # Test with sample data
    dimension = 384  # all-MiniLM-L6-v2 dimension
    
    # Create sample embeddings
    sample_embeddings = np.random.rand(5, dimension).astype('float32')
    sample_metadata = [
        {'text': f'Document {i}', 'source': f'doc_{i}.txt'} 
        for i in range(5)
    ]
    
    # Test vector store
    vector_store = FAISSVectorStore(dimension)
    vector_store.add_embeddings(sample_embeddings, sample_metadata)
    
    # Test search
    query_embedding = np.random.rand(1, dimension).astype('float32')
    scores, results = vector_store.search(query_embedding, k=3)
    
    print(f"Search results: {len(results)} documents found")
    for i, (score, metadata) in enumerate(zip(scores, results)):
        print(f"Result {i+1}: Score={score:.3f}, Metadata={metadata}")
    
    # Test save/load
    vector_store.save_index("test_vectorstore")
    print("Vector store saved successfully")


if __name__ == "__main__":
    main()