"""
Embeddings Module
Handles text embedding generation using Hugging Face sentence-transformers
"""

import os
from typing import List, Optional
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles text embedding generation using Hugging Face sentence-transformers
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Name of the Hugging Face model to use for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.dimension = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the sentence transformer model
        """
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            sample_embedding = self.model.encode(["test"])
            self.dimension = sample_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            logger.warning("Empty text list provided")
            return np.array([])
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Normalize embeddings for better similarity search
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            logger.info(f"Generated {embeddings.shape[0]} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array representing the embedding
        """
        embeddings = self.generate_embeddings([text], show_progress=False)
        return embeddings[0] if len(embeddings) > 0 else np.array([])
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """
        Save embeddings to file
        
        Args:
            embeddings: NumPy array of embeddings
            file_path: Path to save the embeddings
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            logger.info(f"Embeddings saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings to {file_path}: {str(e)}")
            raise
    
    def load_embeddings(self, file_path: str) -> Optional[np.ndarray]:
        """
        Load embeddings from file
        
        Args:
            file_path: Path to the embeddings file
            
        Returns:
            NumPy array of embeddings or None if file doesn't exist
        """
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Embeddings file not found: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                embeddings = pickle.load(f)
            
            logger.info(f"Loaded embeddings from {file_path}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings from {file_path}: {str(e)}")
            return None
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'dimension': self.dimension,
            'max_sequence_length': getattr(self.model, 'max_seq_length', 'Unknown')
        }


class DocumentEmbedder:
    """
    Combines document processing with embedding generation
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """
        Initialize document embedder
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator
        """
        self.embedding_generator = embedding_generator
    
    def embed_documents(self, documents: List, batch_size: int = 32) -> tuple:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of Document objects with page_content
            batch_size: Batch size for embedding generation
            
        Returns:
            Tuple of (embeddings, metadata_list)
        """
        if not documents:
            logger.warning("No documents provided for embedding")
            return np.array([]), []
        
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts, batch_size)
        
        # Extract metadata and include page_content
        metadata_list = []
        for doc in documents:
            metadata = doc.metadata.copy()  # Copy existing metadata
            metadata['page_content'] = doc.page_content  # Add the actual text content
            metadata_list.append(metadata)
        
        logger.info(f"Embedded {len(documents)} documents")
        return embeddings, metadata_list
    
    def save_document_embeddings(self, embeddings: np.ndarray, metadata_list: List[dict], 
                                base_path: str = "vectorstore"):
        """
        Save document embeddings and metadata
        
        Args:
            embeddings: NumPy array of embeddings
            metadata_list: List of metadata dictionaries
            base_path: Base directory to save files
        """
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save embeddings
            embeddings_path = os.path.join(base_path, "embeddings.pkl")
            self.embedding_generator.save_embeddings(embeddings, embeddings_path)
            
            # Save metadata
            metadata_path = os.path.join(base_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata_list, f)
            
            logger.info(f"Document embeddings and metadata saved to {base_path}")
            
        except Exception as e:
            logger.error(f"Failed to save document embeddings: {str(e)}")
            raise
    
    def load_document_embeddings(self, base_path: str = "vectorstore") -> tuple:
        """
        Load document embeddings and metadata
        
        Args:
            base_path: Base directory containing saved files
            
        Returns:
            Tuple of (embeddings, metadata_list)
        """
        try:
            # Load embeddings
            embeddings_path = os.path.join(base_path, "embeddings.pkl")
            embeddings = self.embedding_generator.load_embeddings(embeddings_path)
            
            # Load metadata
            metadata_path = os.path.join(base_path, "metadata.pkl")
            metadata_list = []
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    metadata_list = pickle.load(f)
            
            logger.info(f"Loaded document embeddings from {base_path}")
            return embeddings, metadata_list
            
        except Exception as e:
            logger.error(f"Failed to load document embeddings: {str(e)}")
            return None, []


def main():
    """
    Test the embedding generator
    """
    # Initialize embedding generator
    embedder = EmbeddingGenerator()
    
    # Test with sample texts
    sample_texts = [
        "This is a sample document about machine learning.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text."
    ]
    
    # Generate embeddings
    embeddings = embedder.generate_embeddings(sample_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Print model info
    model_info = embedder.get_model_info()
    print(f"Model info: {model_info}")


if __name__ == "__main__":
    main()