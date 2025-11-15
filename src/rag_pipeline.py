"""
RAG Pipeline Module
Combines document processing, embeddings, vector search, and LLM generation
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator, DocumentEmbedder
from .vector_store import VectorStoreManager
from .llm_integration import LLMManager, create_rag_prompt, LLMResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """
    Response structure for RAG queries
    """
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model_used: str
    response_time: float
    metadata: Optional[Dict[str, Any]] = None


class RAGPipeline:
    """
    Complete RAG pipeline implementation
    """
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 vector_store_path: str = "vectorstore"):
        """
        Initialize RAG pipeline
        
        Args:
            embedding_model: Hugging Face embedding model name
            chunk_size: Text chunk size for processing
            chunk_overlap: Overlap between chunks
            vector_store_path: Path to store vector database
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store_path = vector_store_path
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """
        Initialize all pipeline components
        """
        try:
            # Document processor
            logger.info("Initializing document processor...")
            self.document_processor = DocumentProcessor(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            # Embedding generator
            logger.info("Initializing embedding generator...")
            self.embedding_generator = EmbeddingGenerator(self.embedding_model)
            self.document_embedder = DocumentEmbedder(self.embedding_generator)
            
            # Vector store manager
            logger.info("Initializing vector store...")
            self.vector_store_manager = VectorStoreManager(
                dimension=self.embedding_generator.dimension,
                base_path=self.vector_store_path
            )
            
            # LLM manager
            logger.info("Initializing LLM manager...")
            self.llm_manager = LLMManager()
            
            logger.info("RAG pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {str(e)}")
            raise
    
    def add_documents(self, documents_path: str) -> Dict[str, Any]:
        """
        Add documents to the RAG system
        
        Args:
            documents_path: Path to directory containing documents
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing documents from {documents_path}")
            
            # Process documents
            documents = self.document_processor.process_documents_from_directory(documents_path)
            
            if not documents:
                return {
                    'success': False,
                    'message': 'No documents found or processed',
                    'documents_processed': 0,
                    'chunks_created': 0
                }
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings, metadata_list = self.document_embedder.embed_documents(documents)
            
            # Add to vector store
            logger.info("Adding to vector store...")
            self.vector_store_manager.add_documents(embeddings, metadata_list)
            
            processing_time = time.time() - start_time
            
            # Get unique source files
            unique_files = set(metadata.get('source', 'unknown') for metadata in metadata_list)
            
            result = {
                'success': True,
                'message': 'Documents processed successfully',
                'documents_processed': len(unique_files),
                'chunks_created': len(documents),
                'processing_time': processing_time,
                'files_processed': list(unique_files)
            }
            
            logger.info(f"Successfully processed {len(unique_files)} documents into {len(documents)} chunks")
            return result
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing documents: {str(e)}',
                'documents_processed': 0,
                'chunks_created': 0
            }
    
    def query(self, 
              question: str, 
              k: int = 5, 
              score_threshold: float = 0.0,
              llm_name: Optional[str] = None,
              **llm_kwargs) -> RAGResponse:
        """
        Process a query through the RAG pipeline
        
        Args:
            question: User question
            k: Number of relevant documents to retrieve
            score_threshold: Minimum similarity score for retrieval
            llm_name: Specific LLM to use (optional)
            **llm_kwargs: Additional parameters for LLM generation
            
        Returns:
            RAGResponse object with answer and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(question)
            
            # Search for relevant documents
            relevant_docs = self.vector_store_manager.search_documents(
                query_embedding, k=k, score_threshold=score_threshold
            )
            
            # Debug: Log retrieved documents
            logger.info(f"Retrieved {len(relevant_docs)} documents")
            for i, doc in enumerate(relevant_docs):
                logger.info(f"Document {i+1}: Score={doc.get('similarity_score', 0):.3f}, Content preview: {doc.get('page_content', '')[:100]}...")
            
            if not relevant_docs:
                logger.warning("No relevant documents found!")
                return RAGResponse(
                    answer="I couldn't find relevant information to answer your question. Please try rephrasing or add more documents to the knowledge base.",
                    sources=[],
                    query=question,
                    model_used="none",
                    response_time=time.time() - start_time,
                    metadata={'error': 'No relevant documents found'}
                )
            
            # Prepare context from relevant documents
            context_parts = []
            sources_info = []
            
            for i, doc in enumerate(relevant_docs):
                # Add document content to context (without confusing "Document N:" prefix)
                content = doc.get('page_content', '').strip()
                if content:
                    context_parts.append(content)
                
                # Prepare source information
                source_info = {
                    'content': doc.get('page_content', '')[:200] + "...",  # First 200 chars
                    'source': doc.get('source', 'Unknown'),
                    'filename': doc.get('filename', 'Unknown'),
                    'similarity_score': doc.get('similarity_score', 0.0)
                }
                sources_info.append(source_info)
            
            # Join context with clear separators
            context = "\n\n---\n\n".join(context_parts)
            
            # Create RAG prompt
            prompt = create_rag_prompt(context, question)
            
            # Debug: Log the context being sent
            logger.info(f"Final context ({len(context)} chars): {context}")
            logger.info(f"Full prompt being sent to LLM: {prompt}")
            
            # Generate response using LLM
            llm_response = self.llm_manager.generate(prompt, llm_name=llm_name, **llm_kwargs)
            
            response_time = time.time() - start_time
            
            return RAGResponse(
                answer=llm_response.content,
                sources=sources_info,
                query=question,
                model_used=llm_response.model,
                response_time=response_time,
                metadata={
                    'relevant_docs_count': len(relevant_docs),
                    'context_length': len(context),
                    'llm_usage': llm_response.usage
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return RAGResponse(
                answer=f"An error occurred while processing your question: {str(e)}",
                sources=[],
                query=question,
                model_used="error",
                response_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status and statistics
        
        Returns:
            Dictionary containing system information
        """
        try:
            vector_stats = self.vector_store_manager.get_statistics()
            available_llms = self.llm_manager.get_available_llms()
            
            return {
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_generator.dimension,
                'vector_store_stats': vector_stats,
                'available_llms': available_llms,
                'default_llm': self.llm_manager.default_llm,
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'vector_store_path': self.vector_store_path
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}
    
    def clear_knowledge_base(self):
        """
        Clear all documents from the knowledge base
        """
        try:
            self.vector_store_manager.vector_store.clear()
            logger.info("Knowledge base cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise
    
    def add_single_document(self, file_path: str) -> Dict[str, Any]:
        """
        Add a single document to the knowledge base
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Processing single document: {file_path}")
            
            # Extract text from document
            text = self.document_processor.process_document(file_path)
            
            if not text:
                return {
                    'success': False,
                    'message': 'Could not extract text from document',
                    'chunks_created': 0
                }
            
            # Create metadata
            metadata = {
                'source': file_path,
                'filename': os.path.basename(file_path),
                'file_type': os.path.splitext(file_path)[1].lower()
            }
            
            # Chunk the document
            chunks = self.document_processor.chunk_text(text, metadata)
            
            if not chunks:
                return {
                    'success': False,
                    'message': 'Could not create chunks from document',
                    'chunks_created': 0
                }
            
            # Generate embeddings and add to vector store
            embeddings, metadata_list = self.document_embedder.embed_documents(chunks)
            self.vector_store_manager.add_documents(embeddings, metadata_list)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'message': 'Document processed successfully',
                'chunks_created': len(chunks),
                'processing_time': processing_time,
                'filename': os.path.basename(file_path)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            return {
                'success': False,
                'message': f'Error processing document: {str(e)}',
                'chunks_created': 0
            }


def main():
    """
    Test the RAG pipeline
    """
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Get system status
    status = rag.get_system_status()
    print("System Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # Test with documents (if available)
    docs_path = "data/docs"
    if os.path.exists(docs_path):
        print(f"\nProcessing documents from {docs_path}...")
        result = rag.add_documents(docs_path)
        print(f"Processing result: {result}")
        
        # Test query
        if result['success']:
            test_query = "What is the main topic of the documents?"
            print(f"\nTesting query: {test_query}")
            
            response = rag.query(test_query)
            print(f"Answer: {response.answer}")
            print(f"Sources: {len(response.sources)} documents")
            print(f"Response time: {response.response_time:.2f}s")
    else:
        print(f"Documents directory {docs_path} not found")


if __name__ == "__main__":
    main()