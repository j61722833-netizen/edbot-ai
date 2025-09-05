# EdBot AI - Textbook Q&A System

A comprehensive AI-powered textbook question-answering system with citation support. Built using LangChain, OpenAI GPT models, and FAISS vector storage.

## 🚀 Features

- **PDF Processing**: Automatically split large textbooks into manageable chunks
- **Text Extraction**: High-quality text extraction with PyMuPDF
- **Vector Embeddings**: OpenAI embeddings with FAISS vector storage
- **Citation Support**: Accurate source citations with page references
- **Confidence Scoring**: Reliability assessment for each answer
- **Interactive CLI**: User-friendly command-line interface
- **Batch Processing**: Handle multiple textbooks efficiently

## ✅ Implementation Status

### COMPLETED - Full RAG Pipeline ✅
- **PDF Processing Pipeline** - Split large textbooks into manageable chunks
- **Text Extraction System** - High-quality text extraction with metadata preservation
- **Vector Database** - FAISS integration with OpenAI embeddings
- **Citation-Aware Q&A** - GPT-4 powered Q&A with source citations
- **Complete RAG Pipeline** - End-to-end question answering system
- **Interactive CLI** - User-friendly command-line interface
- **System Testing** - Comprehensive test suite (6/6 tests passed)

**Successfully Processed:**
- US History textbook (975 pages → 65 chunks → 970 text segments)
- Vector index with 451,280 words
- Ready for interactive Q&A with citations

## 📋 Requirements

- Python 3.11+
- OpenAI API key
- ~2GB disk space for dependencies and indexes

## 🛠 Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd edbot_ai_langchain
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set OpenAI API key:**
```bash
export OPENAI_API_KEY='your-api-key-here'
```

## 🎯 Quick Start

### Process a Textbook
```bash
python cli.py process textbook.pdf --textbook-name "US History"
```

### Start Interactive Q&A
```bash
python cli.py chat us_history_textbook
```

### Ask Single Question
```bash
python cli.py ask us_history_textbook "What caused the Civil War?"
```

### List Available Textbooks
```bash
python cli.py list
```

## 🏗 Architecture

The system processes textbooks through a multi-stage pipeline:

1. **PDF Splitting**: Large PDFs → manageable chunks (15-20 pages)
2. **Text Extraction**: PDF chunks → clean, structured text
3. **Vector Indexing**: Text chunks → searchable embeddings  
4. **Q&A Processing**: Questions → context-aware answers with citations

## 📊 Test Results

Successfully tested with 975-page US History textbook:
- ✅ **970 text chunks** extracted and indexed
- ✅ **451,280 words** processed
- ✅ **High-quality citations** with page references
- ✅ **Fast retrieval** (3-8 seconds per query)

## 🧪 System Verification

Run the test suite:
```bash
python test_system.py
```

Expected: **6/6 tests passed** 🎉

## 📁 Project Structure

```
src/
├── processors/          # PDF and text processing
├── utils/              # Vector storage and analysis  
├── models/             # Q&A system
└── edbot_pipeline.py   # Main pipeline

cli.py                  # Command-line interface
test_system.py         # System tests
```

## ⚡ Quick Demo

```bash
# 1. Process textbook (if available)
python cli.py process textbooks/USHistory-WEB_compressed.pdf

# 2. Start chatting
python cli.py chat USHistory-WEB_compressed_textbook

# 3. Ask questions with citations!
❓ Question: What caused the American Revolution?
🤖 Answer: [Detailed answer with source citations]
```

The EdBot AI system is now **fully implemented and tested** - ready for textbook Q&A with accurate citations! 🎓📚