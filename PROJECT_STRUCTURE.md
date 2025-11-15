# RAG System - Clean Project Structure

## 📁 Directory Structure
```
HCL/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── streamlit_app.py            # Main web interface
├── PROJECT_STRUCTURE.md        # This file
├── .env.example                # Environment variables template
├── 
├── src/                        # Core RAG modules
│   ├── __init__.py
│   ├── document_processor.py   # PDF processing with OCR
│   ├── embeddings.py           # Hugging Face embeddings
│   ├── vector_store.py         # FAISS vector database
│   ├── llm_integration.py      # Groq/OpenAI/HuggingFace LLMs
│   └── rag_pipeline.py         # Main RAG orchestration
├──
├── data/                       # PDF documents for processing
│   ├── PG-Ordinances.pdf
│   ├── 2024-August-M.Tech.( CSE)-Regulations1.pdf
│   ├── 2024-May-M.Tech.(CS & AI)-Regulation.pdf
│   └── Regulations MTech (R).pdf
├──
└── vectorstore/                # FAISS vector database files
    ├── index.faiss
    └── index.pkl
```

## 🧹 Cleaned Up Files
The following unnecessary files were removed to maintain a clean codebase:

### Debug Files (Removed)
- `debug_llm_response.py` - LLM response testing
- `debug_rag.py` - RAG pipeline debugging  
- `debug_rag_fixed.py` - Fixed debugging script

### Test Files (Removed)
- `test_pdf_rag.py` - PDF processing tests
- `test_rag.py` - General RAG tests
- `test_single_pdf.py` - Single PDF testing
- `run_rag.py` - Legacy RAG runner
- `test_document.txt` - Test document

### Obsolete Directories (Removed)
- `mini-rag/` - Old implementation directory
- `__pycache__/` - Python cache directories

## 🚀 Core Components Retained

### Essential Files
- **streamlit_app.py**: Main web interface with chat functionality
- **requirements.txt**: Optimized dependencies (commented out dev tools)
- **README.md**: Complete project documentation
- **.env.example**: Environment configuration template

### Core Modules (src/)
- **rag_pipeline.py**: Central orchestration engine
- **document_processor.py**: PDF processing with OCR support
- **embeddings.py**: Sentence transformer embeddings
- **vector_store.py**: FAISS vector search capabilities
- **llm_integration.py**: Multi-provider LLM support (Groq/OpenAI)

### Data & Storage
- **data/**: PDF documents (M.Tech regulations)
- **vectorstore/**: Pre-computed FAISS embeddings

## 📊 Project Stats
- **Total Files**: 8 core files + data
- **Lines of Code**: ~1,500 (estimated)
- **Dependencies**: 15 essential packages
- **Features**: Complete RAG pipeline with web interface

## 🎯 Next Steps
1. The codebase is now production-ready
2. All debug/test files removed
3. Clean, maintainable structure
4. Ready for deployment or further development