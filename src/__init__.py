"""
RAG Assistant Package
Initialization file for the RAG assistant modules
"""

from .document_processor import DocumentProcessor
from .embeddings import EmbeddingGenerator, DocumentEmbedder
from .vector_store import FAISSVectorStore, VectorStoreManager
from .llm_integration import (
    LLMManager, 
    OpenAILLM, 
    HuggingFaceLLM, 
    create_rag_prompt,
    create_greeting_prompt
)
from .rag_pipeline import RAGPipeline, RAGResponse

__version__ = "1.0.0"
__author__ = "RAG Assistant Team"

__all__ = [
    "DocumentProcessor",
    "EmbeddingGenerator", 
    "DocumentEmbedder",
    "FAISSVectorStore",
    "VectorStoreManager",
    "LLMManager",
    "OpenAILLM",
    "HuggingFaceLLM",
    "create_rag_prompt",
    "create_greeting_prompt",
    "RAGPipeline",
    "RAGResponse"
]