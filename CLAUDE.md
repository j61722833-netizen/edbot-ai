# Claude Code Instructions for EdBot AI LangChain Project

## Project Overview
This project implements a dual-model LangChain system for textbook Q&A with citation and fact-checking capabilities. The system consists of:

1. **Citation Q&A Model**: Answers questions based on textbook content with source citations
2. **Fact-Checking Model**: Verifies claims and citations for accuracy using Chain of Verification (CoVe)

## Architecture Report
A comprehensive architecture report has been created: `langchain_dual_model_architecture_report.md`
- Contains detailed technical specifications
- Includes implementation roadmap and workflow options
- Provides technology stack recommendations
- Contains cost estimates and evaluation metrics

## Current Project Status
- **Phase**: Foundation setup phase (not yet started)
- **Codebase**: Empty - starting from scratch
- **Next Steps**: Begin Phase 1 implementation

## Implementation Roadmap

### Phase 1: Foundation Setup (Current Priority)
**Tasks to implement:**
- [ ] Set up Python virtual environment and dependencies
- [ ] Install LangChain, LangGraph, and vector database packages
- [ ] Create project structure with modular components
- [ ] Implement basic RAG pipeline with FAISS
- [ ] Create textbook document processor
- [ ] Develop citation output schema using Pydantic

**Key files to create:**
```
/workspace/edbot_ai_langchain/
├── requirements.txt           # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── qa_model.py       # Citation Q&A model
│   │   └── fact_checker.py   # Fact-checking model
│   ├── processors/
│   │   ├── __init__.py
│   │   └── textbook_processor.py  # Document processing
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── citation_schema.py     # Pydantic schemas
│   └── utils/
│       ├── __init__.py
│       └── vector_store.py        # Vector database utilities
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration settings
├── tests/
│   └── __init__.py
└── examples/
    └── sample_textbooks/     # Test textbook samples
```

### Required Dependencies
```txt
langchain>=0.1.0
langchain-openai
langchain-community
langgraph
langsmith
faiss-cpu
openai
pydantic>=2.0
python-dotenv
streamlit  # for UI (later phases)
pytest  # for testing
```

### Environment Variables Needed
```
OPENAI_API_KEY=your_openai_api_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
SERPER_API_KEY=your_serper_key  # for external search
```

### Recommended Model Configuration
- **Primary Q&A Model**: GPT-4 Turbo or Claude 3.5 Sonnet
- **Fact-Checking Model**: GPT-4o or Claude 3 Opus
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector Store**: Start with FAISS, migrate to Pinecone for production

## Development Guidelines

### Code Standards
- Use type hints throughout
- Implement comprehensive error handling
- Add logging for debugging and monitoring
- Follow modular design principles
- Write unit tests for core components

### Testing Strategy
- Test with multiple textbook formats (PDF, HTML, plain text)
- Validate citation accuracy with known sources
- Test fact-checking with deliberately false claims
- Performance testing for latency requirements

### Key Implementation Notes
1. **Citation System**: Use paragraph-level indexing with unique IDs
2. **Fact-Checking**: Implement "Factored method" for independent verification
3. **Agent Coordination**: Use LangGraph for multi-agent workflows
4. **External Sources**: Integrate Serper API for web search verification

## Commands to Run

### Setup Commands
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Start development server (later phases)
streamlit run app.py
```

### Lint and Type Check Commands
```bash
# Type checking
mypy src/

# Code formatting
black src/
isort src/

# Linting
flake8 src/
```

## Next Claude Session Instructions

When you start working on this project:

1. **Begin with Phase 1 tasks** - Set up the foundation components
2. **Create the project structure** as outlined above
3. **Implement basic RAG pipeline** with citation capabilities first
4. **Test with sample textbook content** before moving to fact-checking
5. **Follow the roadmap sequentially** - don't skip phases

## Important Considerations

### Security
- Never commit API keys to the repository
- Use environment variables for all sensitive configuration
- Implement input validation for user queries

### Performance
- Implement caching for repeated queries
- Consider async processing for fact-checking
- Monitor token usage and costs

### Scalability
- Design for multiple textbook sources
- Plan for production vector database migration
- Consider user management and rate limiting

## Helpful Resources
- LangChain Documentation: https://python.langchain.com/
- LangGraph Tutorials: https://langchain-ai.github.io/langgraph/
- Chain of Verification Papers and Implementations (referenced in architecture report)

## Success Metrics
- Citation accuracy > 95%
- Fact-checking precision > 90%
- End-to-end response time < 10 seconds
- System uptime > 99%

---
*This file should be updated as the project evolves and new requirements emerge.*