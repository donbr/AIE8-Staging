# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Start Docker services (Ollama and Qdrant)
docker-compose up -d

# Pull required Ollama models
ollama pull gpt-oss:20b        # Chat model
ollama pull embeddinggemma:latest  # Embedding model (768 dimensions)

# Verify Ollama is running
ollama -v  # Should show version 0.11.10+
curl http://localhost:11434/api/tags  # Should list available models

# Check Docker services status
docker-compose ps

# Access Qdrant dashboard
# http://localhost:6333/dashboard
```

### Working with Notebooks
```bash
# Start Jupyter with the correct kernel
jupyter notebook

# Or run Python scripts directly
python lcel-langchain-langgraph.py
```

Key notebooks:
- `Ollama_Setup_and_Testing.ipynb`: Verify Ollama setup before main assignment
- `Assignment_Introduction_to_LCEL_and_LangGraph_LangChain_Powered_RAG.ipynb`: Main assignment notebook

## Project Architecture

This is a Production RAG implementation that processes AI usage research papers to answer questions about how people use AI:

### Core Components
- **LangGraph**: Stateful graph-based workflows with nodes for retrieval and generation
- **Ollama Models**:
  - `gpt-oss:20b` for chat/generation (temperature=0.6)
  - `embeddinggemma:latest` for embeddings (768 dimensions)
- **Qdrant**: Vector database on port 6333 (dashboard at http://localhost:6333/dashboard)
  - Collection: `ai_usage_knowledge_index`
  - Distance metric: Cosine similarity
- **Document Processing**:
  - PyMuPDFLoader for PDF ingestion from `data/` directory
  - RecursiveCharacterTextSplitter (750 tokens, no overlap)

### LangGraph State Flow
```python
State(TypedDict):
  question: str        # User query
  context: list[Document]  # Retrieved documents
  response: str        # Generated answer
```

Flow: START → retrieve → generate → END

### Optional LangSmith Integration
Set environment variables for tracing:
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your-key>`
- `LANGCHAIN_PROJECT=RAG-Assignment`

## Key Implementation Details

- **Embedding dimension**: 768 (required for Question #1)
- **Retriever configuration**: Returns top 5 documents (k=5)
- **RAG prompt template**: Uses context-aware template with fallback to "I don't know"
- **Vector store**: Can use in-memory (`:memory:`) or persistent Qdrant (localhost:6333)

## Cursor Rules Configuration (September 2025)

This project uses the modern Cursor rules system with `.cursor/rules/` directory containing `.mdc` files:

### Rule Types Available
- **Always Rules**: `project-fundamentals.mdc` - Core project context always included
- **Auto-Attached Rules**: `langgraph-patterns.mdc` - Activated when working with Python/notebook files
- **Agent-Requested Rules**: `development-workflow.mdc` - AI decides when to use based on context
- **Manual Rules**: `debugging-helpers.mdc` - Use `@debugging-helpers` to activate

### Using Rules in Cursor
```
@debugging-helpers    # Activate manual debugging rules
@development-workflow # Not needed - agent will request automatically
```

### Rule Structure
```
.cursor/rules/
├── project-fundamentals.mdc     # Always applied
├── langgraph-patterns.mdc       # Auto-attached to *.py, *.ipynb
├── development-workflow.mdc     # Agent-requested
└── debugging-helpers.mdc        # Manual activation
```

Each `.mdc` file contains:
- **Frontmatter**: Metadata with `alwaysApply`, `globs`, `description`
- **Content**: Markdown with specific project guidance

## Assignment Notes

### Breakout Room #1 Tasks
1. Install and run Ollama
2. Pull required models
3. Test embeddings and inference with LangChain connectors

### Breakout Room #2 Tasks
1. Install LangGraph
2. Understand States and Nodes
3. Build Basic Graph
4. Implement Simple RAG Graph
5. Extend Graph with Complex Flows

### Important Reminders
- Always verify Ollama is running before LLM operations
- The embedding dimension (768) must be correctly set for vector store initialization
- Use Docker Compose for consistent service management
- Test both embeddings and model inference before proceeding with main assignment