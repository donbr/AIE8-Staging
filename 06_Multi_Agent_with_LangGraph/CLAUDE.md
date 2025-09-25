# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-Agent LangGraph application for AIE8 Session 6. **Status**: 95% complete with quadruple architecture support: notebook + modular library + MCP microservices + audio transcription workflows.

**Use Case**: Hierarchical agent teams that combine web research, PDF knowledge retrieval, and audio transcription to generate professional documents with multi-modal content processing.

## Essential Development Commands

### Testing & Validation
```bash
# Primary test commands - run these to validate changes
python validate_notebook_compatibility.py    # Ensure notebook still works (7 checks)
python validate_phase2_components.py         # Test modular components (5 categories)
python validate_mcp_servers.py               # Test MCP servers (21 tests, 90.5% success)

# Unit tests
PYTHONPATH=src python -m pytest tests/test_configuration.py -v    # 17+ tests (includes audio config)
PYTHONPATH=src python -m pytest tests/test_mcp_servers.py -v      # Comprehensive MCP tests (40+ tests)

# Audio transcription integration test
python test_audio_transcription_integration.py                    # End-to-end transcription validation

# Run all validations
python validate_notebook_compatibility.py && python validate_phase2_components.py && python validate_mcp_servers.py

# Run specific test categories
pytest tests/test_mcp_servers.py::TestTavilyMCPServer -v          # Tavily server tests
pytest tests/test_mcp_servers.py::TestVectorStoreMCPServer -v     # Vector store tests
pytest tests/test_mcp_servers.py::TestAudioTranscriptionMCPServer -v  # Audio transcription tests
pytest tests/test_mcp_servers.py::TestMCPIntegration -v           # Integration tests
```

### Development Setup
```bash
# Environment setup
uv sync                           # Install dependencies
export OPENAI_API_KEY="..."       # Required for LLM calls
export TAVILY_API_KEY="..."       # Required for web search

# MCP Server Configuration (Optional)
export QDRANT_URL="http://localhost:6333"    # Vector store URL
export EMBEDDING_DIMENSION="1536"            # OpenAI embedding dimension
export ENABLE_MCP="true"                     # Enable MCP servers
export MCP_FALLBACK_LANGCHAIN="true"        # Fallback to LangChain if MCP fails

# Audio Transcription Configuration (Optional)
export TRANSCRIPTION_BACKEND="auto"          # faster-whisper, openai-whisper, openai-api, auto
export TRANSCRIPTION_MODEL="turbo"           # Model size for Whisper
export ENABLE_VAD="true"                     # Voice Activity Detection
export CLEAN_TRANSCRIPT="true"              # LLM-based post-processing

# Notebook development
jupyter lab                       # Start Jupyter for notebook work

### Audio Transcription Commands
```bash
# Extract transcript from default audio file (primary tool)
python extract_transcript.py                           # Uses reference/GMT20250710-230109_Recording.m4a

# Extract transcript from custom audio file
python extract_transcript.py /path/to/your/audio.wav   # Supports .mp3, .wav, .m4a, .flac, .aac, .ogg, .webm

# Alternative transcription approaches (for debugging)
python transcribe_audio.py                             # Direct faster-whisper approach
python transcribe_with_openai.py                       # OpenAI API approach
python test_audio_transcription_integration.py         # Integration test suite
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

# Single test for specific component
PYTHONPATH=src python -m pytest tests/test_configuration.py::TestConfiguration::test_model_selection -v

# Debug import issues in Phase 2
python -c "
import sys
sys.path.insert(0, 'src')
from agents.state import ResearchTeamState
print('✅ State classes import correctly')
"

# Test MCP integration
python -c "
from src.config.configuration import Configuration
config = Configuration()
print(f'✅ MCP Enabled: {config.is_mcp_enabled()}')
print(f'✅ Search Preference: {config.get_search_preference()}')
print(f'✅ Transcription Backend: {config.get_transcription_backend()}')
"

# Test audio transcription workflow
python -c "
from src.graphs.transcription_workflow import transcribe_single_file
import asyncio
# result = asyncio.run(transcribe_single_file('path/to/audio.wav'))  # Uncomment to test
print('✅ Audio transcription workflow ready')
"
```

### MCP Server Management
```bash
# MCP server status and management
python -m mcp_servers.manager status                    # Check all server status
python -m mcp_servers.manager start --server tavily-search    # Start specific server
python -m mcp_servers.manager stop --server vector-store      # Stop specific server
python -m mcp_servers.manager health                    # Health check all servers

# Manual MCP server testing
python -m mcp_servers.tavily_search                     # Run Tavily server directly
python -m mcp_servers.vector_store                      # Run Vector store server directly
python -m mcp_servers.arxiv_researcher                  # Run ArXiv server directly
python -m mcp_servers.audio_transcription               # Run Audio transcription server directly
```

### Audio Transcription Workflows
```bash
# Install transcription dependencies
pip install -r mcp_servers/requirements.txt             # Install all MCP server dependencies

# Single file transcription (programmatic)
python -c "
import asyncio
from src.graphs.transcription_workflow import transcribe_single_file
result = asyncio.run(transcribe_single_file(
    'reference/GMT20250710-230109_Recording.m4a',
    transcription_config={'backend': 'auto', 'clean_transcript': True}
))
print(f'Transcribed {len(result[\"final_documents\"])} documents')
"

# Batch transcription
python -c "
import asyncio
from src.graphs.transcription_workflow import transcribe_batch_files
result = asyncio.run(transcribe_batch_files(['file1.wav', 'file2.mp3']))
print(f'Processed {result[\"total_files\"]} files, {result[\"successful_transcriptions\"]} succeeded')
"

# Run integration test suite
python test_audio_transcription_integration.py          # Comprehensive validation (7 tests)
```

## Architecture Overview

### Quadruple Architecture Pattern
The system supports **notebook** (original), **modular library** (Phase 2), **MCP microservices** (Phase 3), and **audio transcription workflows** (Phase 4) patterns:

#### Production Library Structure (95% Complete)
```
src/
├── config/         # Pydantic configuration + externalized prompts + audio transcription settings
├── core/           # LLM factories and shared utilities
├── agents/         # MessagesState-based agent definitions
├── graphs/         # Modular graph implementations + transcription workflows
│   └── transcription_workflow.py  # Audio transcription graph (NEW)
├── tools/          # Organized tool implementations
├── mcp_integration.py  # MCP compatibility layer + audio transcription tools
└── compat.py       # Backward compatibility layer

mcp_servers/        # MCP microservices
├── config.py       # MCP server configuration
├── manager.py      # Server lifecycle management
├── tavily_search.py    # Enhanced web search server
├── vector_store.py     # Persistent vector database
├── arxiv_researcher.py # Academic paper research (BONUS ACTIVITY)
├── audio_transcription.py  # Multi-backend audio transcription server (NEW)
└── requirements.txt    # MCP dependencies + transcription libraries
```

#### Notebook Architecture (Fully Functional)
Three-level hierarchy maintained for backward compatibility:
1. **Meta-Supervisor** → Routes between research and writing teams
2. **Research Team** → Search agent + RAG agent + supervisor
3. **Writing Team** → NoteTaker + DocWriter + CopyEditor + supervisor

#### MCP Microservices Architecture
Enhanced tool functionality through standardized Model Context Protocol servers:

1. **Tavily Search Server**: Direct API integration with caching, structured results, news search
2. **Vector Store Server**: Persistent Qdrant with collection management, embedding automation
3. **ArXiv Researcher Server**: Academic paper search, trend analysis, citation networks
4. **Audio Transcription Server**: Multi-backend transcription with LLM post-processing (NEW)
5. **Server Manager**: Lifecycle management, health monitoring, graceful degradation

**Key Design Principles**:
- **Hierarchical Coordination**: Each team has its own supervisor for local decisions
- **Tool Specialization**: Research agents use enhanced MCP tools or LangChain fallbacks
- **State Isolation**: Teams maintain separate workspaces in `content/data/` with UUID prefixes
- **Configuration-Driven**: All models, temperatures, and parameters externalized to Pydantic config
- **Microservice Pattern**: Independent, scalable MCP servers with health monitoring
- **Graceful Degradation**: Automatic fallback to LangChain tools when MCP unavailable

### Key State Classes (MessagesState-based)
- `SimpleRAGState` - Basic retrieve + generate workflow
- `ResearchTeamState` - Multi-agent research coordination with iteration tracking
- `DocWritingState` - Document creation workflow with file management
- `MetaSupervisorState` - Top-level routing with team coordination
- `AudioTranscriptionState` - Multi-modal audio processing workflow (NEW)

**Migration from TypedDict**: All states now extend LangGraph's `MessagesState` for better framework integration and conversation history management.

## MCP Architecture Details

### MCP Server Implementation Pattern

Each MCP server follows a standardized pattern for consistency and maintainability:

```python
# Server structure example (mcp_servers/audio_transcription.py)
class AudioTranscriptionMCPServer:
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.cache: Dict[str, tuple[TranscriptionResult, datetime]] = {}
        self.cache_ttl = timedelta(hours=24)
        self._check_backends()  # Auto-detect available transcription backends

    async def _transcribe_audio(self, file_path: str, backend: str = "auto", **kwargs) -> List[TextContent]:
        # Multi-backend implementation with intelligent fallback
        pass
```

### Integration Points

**Factory Integration** (`src/core/factory.py`):
```python
def create_search_tool(config: Configuration, **kwargs):
    if config.is_mcp_enabled():
        try:
            # Attempt MCP tool creation
            mcp_tool = await create_mcp_search_tool(config, **kwargs)
            return mcp_tool
        except Exception:
            if not config.mcp_fallback_to_langchain:
                raise

    # Fallback to LangChain tool
    return create_langchain_search_tool(config, **kwargs)
```

**Configuration Integration** (`src/config/configuration.py`):
```python
class Configuration(BaseModel):
    # MCP-specific fields
    use_mcp_servers: bool = True
    mcp_fallback_to_langchain: bool = True

    def is_mcp_enabled(self) -> bool:
        return self.use_mcp_servers and os.getenv("ENABLE_MCP", "true").lower() == "true"

    def should_use_mcp_for_tool(self, tool_name: str) -> bool:
        # Tool-specific MCP preference logic
        pass
```

### Testing Architecture

**Multi-level Testing Strategy**:
1. **Unit Tests**: Individual MCP server components with mocked dependencies
2. **Integration Tests**: MCP-LangChain compatibility layer
3. **Validation Scripts**: End-to-end functionality verification
4. **Compatibility Tests**: Ensure notebook still works with MCP integration

**Key Test Categories**:
- Server initialization and configuration
- Tool functionality and result formatting
- Async operation handling
- Error handling and fallback mechanisms
- Caching and performance optimizations

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
    chunk_size=750,                    # RAG chunk size

    # MCP Configuration
    use_mcp_servers=True,              # Enable MCP server integration
    mcp_fallback_to_langchain=True,    # Graceful fallback
    mcp_timeout_seconds=30,            # MCP operation timeout
    mcp_auto_start_servers=True,       # Auto-start servers when needed

    # Audio Transcription Configuration (NEW)
    transcription_backend="auto",      # Backend selection: auto, faster-whisper, openai-whisper, openai-api
    transcription_model="turbo",       # Model size: tiny, base, small, medium, large, turbo
    enable_vad=True,                   # Voice Activity Detection
    clean_transcript=True,             # LLM-based post-processing
    transcription_cache_hours=24       # Cache duration for transcription results
)

# Environment override support
export RESEARCH_MODEL="gpt-4o"        # Overrides default
export ENABLE_MCP="false"             # Disable MCP servers
export MCP_FALLBACK_LANGCHAIN="true"  # Enable fallback
export TRANSCRIPTION_BACKEND="faster-whisper"  # Override transcription backend
export TRANSCRIPTION_MODEL="large-v3"    # Override model size
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
- `validate_mcp_servers.py` - **NEW**: MCP server functionality validation (21 tests)

### MCP Infrastructure
- `mcp_servers/config.py` - MCP server configuration management
- `mcp_servers/manager.py` - Server lifecycle and health monitoring
- `mcp_servers/audio_transcription.py` - Multi-backend audio transcription server (NEW)
- `src/mcp_integration.py` - LangChain compatibility layer + audio transcription tools
- `tests/test_mcp_servers.py` - Comprehensive MCP test suite (40+ tests)

### Audio Transcription Infrastructure (NEW)
- `extract_transcript.py` - **Primary transcription tool** with multi-backend fallback and WSL2 optimization
- `transcribe_audio.py` - Direct faster-whisper implementation with CPU optimization
- `transcribe_with_openai.py` - OpenAI API implementation with file size validation
- `src/graphs/transcription_workflow.py` - Complete audio transcription workflow graphs
- `test_audio_transcription_integration.py` - End-to-end integration test suite

### Documentation
- `MIGRATION_STATUS.md` - Detailed migration progress and testing status
- `MCP_TESTING_SUMMARY.md` - **NEW**: MCP integration and testing summary
- `TESTING.md` - Comprehensive testing procedures and troubleshooting

## Testing Strategy

### Validation Hierarchy
1. **Notebook Compatibility** - Highest priority, must always pass (7/7 tests)
2. **Component Structure** - Validates modular organization (5/5 tests)
3. **MCP Server Functionality** - **NEW**: Validates MCP integration (19/21 tests, 90.5% success)
4. **Unit Tests** - Configuration system and MCP components (38+ tests total)

### Common Test Failures
- **Import errors**: Phase 2 unit tests have relative import issues - use validation scripts instead
- **Missing API keys**: Set dummy values for testing: `export OPENAI_API_KEY="test"`
- **PYTHONPATH issues**: Use `PYTHONPATH=src python -m pytest` for unit tests
- **Notebook execution**: If notebook fails, check API keys and ensure all cells run in sequence
- **MCP ArXiv tests**: Require `pip install arxiv` for full functionality
- **MCP server connections**: Ensure external services (Qdrant) are running or use mocked tests
- **Audio transcription WSL2 issues**: GPU detection failures resolved by `extract_transcript.py` with CPU-only processing
- **Audio transcription timeouts**: Large files (>50MB) may require compression or chunking - use `extract_transcript.py`
- **Audio dependencies**: Missing `faster-whisper` or `openai-whisper` - install with `uv add faster-whisper`

### Development Workflow
```bash
# Before making changes
python validate_notebook_compatibility.py && python validate_phase2_components.py && python validate_mcp_servers.py

# After making changes to configuration
PYTHONPATH=src python -m pytest tests/test_configuration.py -v

# After making changes to MCP servers
python validate_mcp_servers.py
PYTHONPATH=src python -m pytest tests/test_mcp_servers.py -v

# After making changes to modular components
python validate_phase2_components.py

# Before committing
python validate_notebook_compatibility.py  # Must pass
python validate_mcp_servers.py             # Should achieve >85% success
```

## Migration Status

### ✅ Completed (Phases 1-4)
- **Configuration System**: Pydantic-based config with environment support + audio transcription settings
- **State Management**: MessagesState patterns following LangGraph 2025 practices
- **Modular Architecture**: Organized agents, graphs, tools into separate modules
- **Backward Compatibility**: Notebook functionality fully preserved
- **MCP Microservices**: Production-ready MCP servers with comprehensive testing
- **Enhanced Tool Integration**: Direct API access, caching, structured results
- **ArXiv Research Capability**: Implements notebook bonus activity with academic paper analysis
- **Audio Transcription System**: Multi-backend transcription with LLM post-processing (NEW)

### ✅ Phase 3-4 Achievements (MCP + Audio Integration)
- **Tavily Search Server**: Direct API integration with 15-minute caching and structured results
- **Vector Store Server**: Persistent Qdrant integration with collection management and batch operations
- **ArXiv Researcher Server**: Academic paper search, trend analysis, and citation networks
- **Audio Transcription Server**: Multi-backend support (Faster-Whisper, OpenAI Whisper, OpenAI API) (NEW)
- **MCP Configuration System**: Centralized server configuration with environment variable support
- **Integration Layer**: LangChain-compatible wrappers with automatic fallback mechanisms
- **Comprehensive Testing**: 90.5% success rate with 40+ comprehensive test cases
- **Health Monitoring**: Server lifecycle management with graceful degradation
- **LLM Post-Processing**: Transcript cleaning and document generation workflows (NEW)

### 🔄 Planned (Phase 5)
- LangGraph Platform deployment readiness
- Advanced async parallel execution patterns
- Production infrastructure (Docker, Kubernetes, monitoring)
- Redis-based distributed caching
- Prometheus metrics and observability
- Real-time audio transcription streaming
- Multi-language transcription optimization

### Critical Constraints
- **Maintain Notebook Compatibility**: Any changes must pass `validate_notebook_compatibility.py`
- **Test Before Changes**: Always run validation scripts before modifications
- **Use Configuration Class**: All new hardcoded values must go through `Configuration`
- **Backward Compatibility Layer**: Use `src/compat.py` to maintain notebook interface during migration

### Known Limitations
- **Phase 2 Unit Tests**: Have import issues, use validation scripts instead
- **Real API Testing**: Limited due to costs, validated through notebook execution
- **CI/CD**: No automated pipeline yet - manual testing required

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

# Optional LangChain
LANGCHAIN_API_KEY="..."     # LangSmith tracing

# MCP Server Configuration (Optional)
ENABLE_MCP="true"                           # Enable/disable MCP servers
MCP_FALLBACK_LANGCHAIN="true"              # Fallback to LangChain if MCP fails
QDRANT_URL="http://localhost:6333"         # Vector store URL
EMBEDDING_DIMENSION="1536"                 # OpenAI embedding dimension
MCP_TIMEOUT_SECONDS="30"                   # MCP operation timeout
MCP_AUTO_START_SERVERS="true"             # Auto-start servers when needed

# Development/Testing
OPENAI_API_KEY="test"      # Dummy key for testing (when not using real APIs)
TAVILY_API_KEY="test"      # Dummy key for testing
```

**Dependencies**:
- Python 3.11+ (specified in pyproject.toml)
- Optional: `faster-whisper` for optimized transcription (`pip install faster-whisper`)
- Optional: `openai-whisper` for fallback transcription (`pip install openai-whisper`)
- Optional: `arxiv` library for full ArXiv server functionality (`pip install arxiv`)
- Optional: Qdrant server for persistent vector storage (Docker: `docker run -p 6333:6333 qdrant/qdrant`)
- Required: FFmpeg for audio format conversion (pre-installed on most systems)