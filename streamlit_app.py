"""
Streamlit RAG Assistant Interface
Interactive web application for the RAG-powered assistant
"""

import streamlit as st
import os
import sys
import tempfile
import time
from typing import List, Dict, Any
import logging

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_pipeline import RAGPipeline
from src.llm_integration import create_greeting_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RAG-Powered Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #e8f4fd;
        border-left: 4px solid #28a745;
    }
    .source-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag_pipeline():
    """
    Initialize and cache the RAG pipeline
    """
    try:
        pipeline = RAGPipeline(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            chunk_size=500,
            chunk_overlap=50
        )
        return pipeline
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {str(e)}")
        return None


def save_uploaded_file(uploaded_file) -> str:
    """
    Save uploaded file to temporary directory
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        Path to saved file
    """
    try:
        # Create temp directory if it doesn't exist
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Failed to save uploaded file: {str(e)}")
        return None


def display_source_info(sources: List[Dict[str, Any]]):
    """
    Display source information in an organized way
    
    Args:
        sources: List of source dictionaries
    """
    if not sources:
        st.info("No sources found for this response.")
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources, 1):
        with st.expander(f"Source {i}: {source.get('filename', 'Unknown')} (Score: {source.get('similarity_score', 0):.3f})"):
            st.write("**Content Preview:**")
            st.write(source.get('content', 'No content available'))
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**File:** {source.get('filename', 'Unknown')}")
            with col2:
                st.write(f"**Similarity Score:** {source.get('similarity_score', 0):.3f}")


def main():
    """
    Main Streamlit application
    """
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG-Powered Assistant</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize RAG pipeline
    rag_pipeline = initialize_rag_pipeline()
    
    if rag_pipeline is None:
        st.error("Failed to initialize the RAG system. Please check your configuration.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("📋 System Status")
        
        # Get system status
        with st.spinner("Loading system status..."):
            system_status = rag_pipeline.get_system_status()
        
        if 'error' not in system_status:
            st.success("✅ System Online")
            
            # Display key metrics
            vector_stats = system_status.get('vector_store_stats', {})
            st.metric("Documents in Knowledge Base", vector_stats.get('total_vectors', 0))
            st.metric("Embedding Dimension", system_status.get('embedding_dimension', 'Unknown'))
            
            # Available LLMs
            available_llms = system_status.get('available_llms', [])
            if available_llms:
                st.write("**Available LLMs:**")
                for llm in available_llms:
                    is_default = llm == system_status.get('default_llm')
                    st.write(f"• {llm} {'(default)' if is_default else ''}")
            else:
                st.warning("No LLMs configured")
        else:
            st.error("❌ System Error")
            st.write(system_status['error'])
        
        st.markdown("---")
        
        # Document Management Section
        st.header("📁 Document Management")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files to add to the knowledge base"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Files", type="primary"):
                process_uploaded_files(uploaded_files, rag_pipeline)
        
        # Clear knowledge base
        if st.button("Clear Knowledge Base", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                try:
                    rag_pipeline.clear_knowledge_base()
                    st.success("Knowledge base cleared successfully!")
                    st.session_state['confirm_clear'] = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to clear knowledge base: {str(e)}")
            else:
                st.session_state['confirm_clear'] = True
                st.warning("Click again to confirm clearing the knowledge base")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Simple greeting button
        if st.button("👋 Welcome"):
            greeting_message = create_greeting_prompt()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": greeting_message,
                "sources": []
            })
            st.rerun()
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
            # Add welcome message for first-time users
            greeting_message = create_greeting_prompt()
            st.session_state.messages.append({
                "role": "assistant", 
                "content": greeting_message,
                "sources": []
            })
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.write(message["content"])
                    if "sources" in message and message["sources"]:
                        display_source_info(message["sources"])
                else:
                    st.write(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me anything about your documents..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = rag_pipeline.query(
                            prompt,
                            k=5,
                            temperature=0.7,
                            max_tokens=500
                        )
                        
                        # Display response
                        st.write(response.answer)
                        
                        # Display sources
                        if response.sources:
                            display_source_info(response.sources)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources
                        })
                    
                    except Exception as e:
                        error_message = "I encountered an error while processing your request. Please try rephrasing your question."
                        st.error("❌ An error occurred while processing your request.")
                        st.write(error_message)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "sources": []
                        })
    
    with col2:
        st.header("📊 Query Settings")
        
        # Query parameters
        with st.expander("Advanced Settings", expanded=False):
            k_docs = st.slider("Number of documents to retrieve", 1, 10, 5)
            temperature = st.slider("Response creativity", 0.0, 1.0, 0.7, 0.1)
            max_tokens = st.slider("Maximum response length", 100, 1000, 500, 50)
            
            # Store settings in session state
            st.session_state.update({
                'k_docs': k_docs,
                'temperature': temperature,
                'max_tokens': max_tokens
            })
        
        # Recent queries
        if st.session_state.messages:
            st.subheader("📝 Recent Queries")
            user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
            for i, msg in enumerate(reversed(user_messages[-5:]), 1):  # Show last 5
                if st.button(f"{i}. {msg['content'][:50]}...", key=f"recent_{i}"):
                    # Re-run the query
                    with st.spinner("Re-processing query..."):
                        response = rag_pipeline.query(
                            msg['content'],
                            k=st.session_state.get('k_docs', 5),
                            temperature=st.session_state.get('temperature', 0.7),
                            max_tokens=st.session_state.get('max_tokens', 500)
                        )
                    
                    # Add to chat history
                    st.session_state.messages.extend([
                        {"role": "user", "content": msg['content']},
                        {
                            "role": "assistant", 
                            "content": response.answer,
                            "sources": response.sources
                        }
                    ])
                    st.rerun()
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


def process_uploaded_files(uploaded_files, rag_pipeline):
    """
    Process uploaded files and add them to the knowledge base
    
    Args:
        uploaded_files: List of uploaded file objects
        rag_pipeline: RAG pipeline instance
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    processed_files = 0
    failed_files = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file
            file_path = save_uploaded_file(uploaded_file)
            
            if file_path:
                # Process the document
                result = rag_pipeline.add_single_document(file_path)
                
                if result['success']:
                    processed_files += 1
                    st.success(f"✅ {uploaded_file.name}: {result['chunks_created']} chunks created")
                else:
                    failed_files += 1
                    st.error(f"❌ {uploaded_file.name}: {result['message']}")
                
                # Clean up temp file
                try:
                    os.remove(file_path)
                except:
                    pass
            else:
                failed_files += 1
                st.error(f"❌ Failed to save {uploaded_file.name}")
            
        except Exception as e:
            failed_files += 1
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
        
        # Update progress
        progress = (i + 1) / total_files
        progress_bar.progress(progress)
    
    # Final status
    status_text.text("Processing complete!")
    
    if processed_files > 0:
        st.success(f"Successfully processed {processed_files} files!")
    
    if failed_files > 0:
        st.warning(f"Failed to process {failed_files} files.")
    
    # Auto-refresh system status
    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    main()