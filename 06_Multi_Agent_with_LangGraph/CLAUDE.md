# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent LangGraph application for AIE8 Session 6. **Status**: 50% migrated from Jupyter notebook to production library with dual architecture support.

**Use Case**: Hierarchical agent teams that combine web research + PDF knowledge retrieval to generate professional documents.

## Essential Development Commands

### Testing & Validation
```bash
# Primary test commands - run these to validate changes
python validate_notebook_compatibility.py    # Ensure notebook still works (7 checks)
python validate_phase2_components.py         # Test modular components (5 categories)

# Unit tests (Phase 1 only - Phase 2 has import issues)
PYTHONPATH=src python -m pytest tests/test_configuration.py -v    # 17 tests

# Run all validations
python validate_notebook_compatibility.py && python validate_phase2_components.py
```

### Development Setup
```bash
# Environment setup
uv sync                           # Install dependencies
export OPENAI_API_KEY="..."       # Required for LLM calls
export TAVILY_API_KEY="..."       # Required for web search

# Notebook development
jupyter lab                       # Start Jupyter for notebook work
```

### Using the Modular Library
```bash
# Test modular components
python -c "
from src.config.configuration import Configuration
from src.graphs import get_available_graphs
config = Configuration()
graphs = get_available_graphs()
print(f'✅ Config: {config.research_model}')
print(f'✅ Available graphs: {list(graphs.keys())}')
"
```

## Architecture Overview

### Dual Implementation
The system supports both **notebook** (original) and **modular library** (new) patterns:

#### Production Library Structure (50% Complete)
```
src/
├── config/         # Pydantic configuration + externalized prompts
├── core/           # LLM factories and shared utilities
├── agents/         # MessagesState-based agent definitions
├── graphs/         # Modular graph implementations (simple_rag, research_team, writing_team, meta_supervisor)
├── tools/          # Organized tool implementations (search, file_management)
└── compat.py       # Backward compatibility layer
```

#### Notebook Architecture (Fully Functional)
Three-level hierarchy maintained for backward compatibility:
1. **Meta-Supervisor** → Routes between research and writing teams
2. **Research Team** → Search agent + RAG agent + supervisor
3. **Writing Team** → NoteTaker + DocWriter + CopyEditor + supervisor

### Key State Classes (New)
- `SimpleRAGState` - Basic retrieve + generate workflow
- `ResearchTeamState` - Multi-agent research coordination
- `DocWritingState` - Document creation workflow
- `MetaSupervisorState` - Top-level routing

## Configuration Management

The system uses a **Pydantic Configuration class** that externalizes all hardcoded notebook values:

```python
# All configuration through environment variables or defaults
from src.config.configuration import Configuration

config = Configuration(
    research_model="gpt-4o-mini",      # Defaults to this
    writing_model="gpt-4o-mini",
    supervisor_model="gpt-4o",         # More reasoning power needed
    search_api="tavily",               # SearchAPI.TAVILY enum
    max_research_iterations=6,         # Control research depth
    chunk_size=750                     # RAG chunk size
)

# Environment override support
export RESEARCH_MODEL="gpt-4o"  # Overrides default
```

## Important Files

### Core Implementation
- `Multi_Agent_RAG_LangGraph.ipynb` - **Main implementation** (source of truth)
- `data/howpeopleuseai.pdf` - RAG knowledge source
- `content/data/` - Dynamic agent workspaces (UUID-prefixed)

### Migration Infrastructure
- `src/compat.py` - **Critical**: Maintains notebook compatibility during migration
- `validate_notebook_compatibility.py` - **Must pass** before any changes
- `validate_phase2_components.py` - Component structure validation

### Documentation
- `MIGRATION_STATUS.md` - Detailed migration progress and testing status
- `TESTING.md` - Comprehensive testing procedures and troubleshooting

## Testing Strategy

### Validation Hierarchy
1. **Notebook Compatibility** - Highest priority, must always pass
2. **Component Structure** - Validates modular organization
3. **Unit Tests** - Phase 1 configuration system (17 tests passing)

### Common Test Failures
- **Import errors**: Phase 2 unit tests have relative import issues - use validation scripts instead
- **Missing API keys**: Set dummy values for testing: `export OPENAI_API_KEY="test"`
- **PYTHONPATH issues**: Use `PYTHONPATH=src python -m pytest` for unit tests

## Migration Status

### ✅ Completed (Phases 1-2)
- **Configuration System**: Pydantic-based config with environment support
- **State Management**: MessagesState patterns following LangGraph 2025 practices
- **Modular Architecture**: Organized agents, graphs, tools into separate modules
- **Backward Compatibility**: Notebook functionality fully preserved

### 🔄 Planned (Phases 3-4)
- LangGraph Platform deployment readiness
- Async parallel execution patterns
- Production infrastructure (Docker, monitoring)

### Critical Constraints
- **Maintain Notebook Compatibility**: Any changes must pass `validate_notebook_compatibility.py`
- **Test Before Changes**: Always run validation scripts before modifications
- **Use Configuration Class**: All new hardcoded values must go through `Configuration`

## Assignment Context

This is AIE8 Session 6 coursework demonstrating:
- Multi-agent hierarchical architectures
- LangGraph workflow orchestration
- RAG + web search integration
- Production refactoring practices

**Assignment Submission**: Must work as Jupyter notebook for grading, but modular library provides foundation for advanced builds.

## Environment Requirements

```bash
# Required
OPENAI_API_KEY="sk-..."     # OpenAI API access
TAVILY_API_KEY="tvly-..."   # Web search capability

# Optional
LANGCHAIN_API_KEY="..."     # LangSmith tracing
```

Python 3.11+ required (specified in pyproject.toml).