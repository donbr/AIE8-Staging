# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Environment Setup
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies from pyproject.toml
uv pip sync
# OR for development
uv pip install -e .
```

### Running Jupyter Notebooks
```bash
jupyter notebook
# OR
jupyter lab
```

## Architecture

This repository contains educational materials for Session 8 of AI Engineering focusing on evaluating RAG (Retrieval-Augmented Generation) systems and agents using the Ragas framework.

### Key Components

1. **RAG Evaluation Pipeline** (`Evaluating_RAG_with_Ragas_(2025)_AI_Makerspace.ipynb`)
   - Synthetic dataset generation using Ragas
   - LangChain/LangGraph RAG implementation with Qdrant vector store
   - Evaluation using multiple Ragas metrics (Context Recall, Faithfulness, Factual Correctness, etc.)
   - Iterative improvements using reranking (Cohere) and chunking strategies

2. **Agent Evaluation** (`Evaluating_Agents_with_Ragas_(2025)_AI_Makerspace.ipynb`)
   - ReAct agent implementation with LangGraph
   - Metal price API tool integration
   - Agent performance evaluation using Tool Call Accuracy, Goal Accuracy, and Topic Adherence

### Data Files
- `data/AIE7_Projects_with_Domains.csv` - Project dataset
- `data/howpeopleuseai.pdf` - PDF document for RAG testing

## Key Dependencies

- **LLM Framework**: LangChain (0.3.14), LangGraph (0.2.61)
- **Evaluation**: Ragas (0.2.10)
- **Vector Store**: Qdrant
- **Reranking**: Cohere (5.12.0)
- **LLMs**: OpenAI GPT models
- **Document Processing**: PyMuPDF
- **Package Management**: uv

## API Keys Required

Set the following environment variables:
- `OPENAI_API_KEY` - For LLM and embeddings
- `COHERE_API_KEY` - For reranking (optional)
- `METAL_API_KEY` - For metal price API in agent example

## Assignment Structure

The assignment involves two main notebooks:
1. Evaluating RAG systems with synthetic data generation and iterative improvements
2. Evaluating agent systems with tool usage and goal achievement metrics

Advanced build option includes implementing semantic chunking strategies for improved RAG performance.