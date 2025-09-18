# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is Session 04 of the AI Engineering course focused on Production RAG with LangGraph and LangChain. The repository contains educational materials for building RAG systems using local AI infrastructure.

## Development Setup

### Environment Setup
```bash
# Install dependencies and create virtual environment
uv sync

# Start local AI stack (Ollama + Qdrant)
docker compose up -d

# Verify services are running
docker compose ps
```

### Jupyter Notebook Workflow
1. Ensure virtual environment is activated (created by `uv sync`)
2. Start with `Ollama_Setup_and_Testing.ipynb` to verify local LLM setup
3. Complete main assignment in `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb`
4. Run evaluation tasks in `LangSmith_and_Evaluation.ipynb`

### Ollama Model Management
```bash
# Pull required models
ollama pull gpt-oss:20b        # Chat model
ollama pull embeddinggemma:latest  # Embeddings

# Verify Ollama is running
ollama -v  # Should be 0.11.10 or greater
```

## Architecture

### Core Components
- **LangGraph StateGraph**: Manages stateful RAG pipeline with retrieve/generate nodes
- **Qdrant Vector Database**: Stores document embeddings (runs in Docker)
- **Ollama**: Local LLM inference server (gpt-oss:20b for chat, embeddinggemma for embeddings)
- **Document Processing**: PyMuPDF loader for PDF ingestion, RecursiveCharacterTextSplitter for chunking

### State Management
The application uses a `TypedDict` state object:
```python
class State(TypedDict):
    question: str
    context: list[Document]
    response: str
```

### RAG Pipeline Flow
1. **Retrieve Node**: Takes user question, queries Qdrant for relevant document chunks
2. **Generate Node**: Uses retrieved context + question to generate response via Ollama
3. **State Transitions**: Each node updates the shared state object

### Local AI Stack (Docker Compose)
- **Qdrant**: Vector database on ports 6333 (HTTP) and 6334 (gRPC)
- **Ollama**: LLM server on port 11434
- **GPU Support**: Configured for NVIDIA GPUs (uncomment CPU-only section if needed)

## Key Dependencies

### Python Package Management
- Uses `uv` for fast, reliable dependency management
- Python 3.12 required
- Key packages: `langgraph`, `langchain-*`, `langchain-qdrant`, `langchain-ollama`

### External Services
- **Ollama**: Local LLM inference (must be installed separately)
- **Docker**: Required for Qdrant vector database
- **OpenAI API**: Used in evaluation notebook for LangSmith integration

## Assignment Structure

### Breakout Room #1 (Main Assignment)
1. Install LangGraph and understand States/Nodes
2. Build basic graph structure
3. Implement simple RAG graph with retrieve/generate nodes
4. Extend graph with complex flows

### Breakout Room #2 (Evaluation)
1. Set up OpenAI API and dependencies
2. Rebuild LangGraph RAG with OpenAI models
3. Configure LangSmith for monitoring
4. Create testing datasets and run evaluations

## Development Notes

### Text Processing
- Uses tiktoken for token counting (cl100k_base encoding)
- Default chunk size: 750 tokens with no overlap
- Embedding dimension for embeddinggemma: Check model documentation

### Vector Database
- Qdrant configured with COSINE distance metric
- In-memory client for development (scale to persistent for production)
- Collection name: "ai_usage_knowledge_index"

### LangChain Integration
- Heavy use of LCEL (LangChain Expression Language) for pipeline composition
- Runnables provide consistent interface across components
- Built-in support for streaming, batching, and parallelization

## Git Workflow

### Assignment Submission
```bash
# Create assignment branch
git checkout -b s04-assignment

# Complete notebooks and commit changes
git add .
git commit -m "Complete Session 04 assignment"
git push origin s04-assignment
```

### Response Format Options
1. Create separate markdown document (e.g., "ACTIVITIES_QUESTIONS.md")
2. Respond inline in notebook markdown cells with `##### ✅ Answer:` headers

## Troubleshooting

### Common Issues
- Ensure Ollama is running and models are pulled before starting notebooks
- Verify Docker services are healthy: `docker compose ps`
- Check Python version compatibility (requires 3.12.*)
- For async issues in Jupyter: `nest_asyncio.apply()` is included in notebooks