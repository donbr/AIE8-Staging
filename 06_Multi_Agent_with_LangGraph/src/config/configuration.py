"""Configuration management for the Multi-Agent LangGraph system.

Following production patterns from open_deep_research for enterprise-grade configuration.
"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

# Import MCP configuration if available
try:
    from ..mcp_servers.config import MCPConfiguration, get_mcp_config
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    MCPConfiguration = None


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


class TranscriptionBackend(Enum):
    """Enumeration of available transcription backends."""

    FASTER_WHISPER = "faster-whisper"
    OPENAI_WHISPER = "openai-whisper"
    OPENAI_API = "openai-api"
    AUTO = "auto"


class Configuration(BaseModel):
    """Main configuration class for the Multi-Agent Research system.

    Externalizes all hardcoded values from the original notebook implementation
    and provides type-safe configuration management with environment variable fallbacks.
    """

    # General Configuration
    max_structured_output_retries: int = Field(
        default=3,
        description="Maximum number of retries for structured output calls from models"
    )
    allow_clarification: bool = Field(
        default=True,
        description="Whether to allow agents to ask clarifying questions"
    )
    max_concurrent_research_units: int = Field(
        default=5,
        description="Maximum number of research units to run concurrently"
    )

    # Model Configuration (extracted from notebook hardcoded values)
    research_model: str = Field(
        default="gpt-4o-mini",
        description="Model for conducting research tasks"
    )
    writing_model: str = Field(
        default="gpt-4o-mini",
        description="Model for document writing tasks"
    )
    supervisor_model: str = Field(
        default="gpt-4o",
        description="Model for supervisor agents (needs more reasoning capability)"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model for text embeddings"
    )

    # Model Token Limits
    research_model_max_tokens: int = Field(
        default=10000,
        description="Maximum output tokens for research model"
    )
    writing_model_max_tokens: int = Field(
        default=10000,
        description="Maximum output tokens for writing model"
    )
    supervisor_model_max_tokens: int = Field(
        default=8192,
        description="Maximum output tokens for supervisor model"
    )

    # Research Configuration
    search_api: SearchAPI = Field(
        default=SearchAPI.TAVILY,
        description="Search API to use for research"
    )
    max_research_iterations: int = Field(
        default=6,
        description="Maximum number of research iterations for Research Supervisor"
    )
    max_react_tool_calls: int = Field(
        default=10,
        description="Maximum number of tool calling iterations in a single step"
    )
    max_content_length: int = Field(
        default=50000,
        description="Maximum character length for content before summarization"
    )

    # RAG Configuration (extracted from notebook chunk_size=750, etc.)
    chunk_size: int = Field(
        default=750,
        description="Text chunk size for document splitting"
    )
    chunk_overlap: int = Field(
        default=0,
        description="Overlap between text chunks"
    )
    retrieval_top_k: int = Field(
        default=5,
        description="Number of top documents to retrieve"
    )

    # Document Writing Configuration
    max_document_iterations: int = Field(
        default=5,
        description="Maximum number of document writing iterations"
    )
    working_directory_prefix: str = Field(
        default="content/data",
        description="Base directory for agent working spaces"
    )

    # Temperature Settings
    research_temperature: float = Field(
        default=0.1,
        description="Temperature for research model (low for consistency)"
    )
    writing_temperature: float = Field(
        default=0.3,
        description="Temperature for writing model (moderate for creativity)"
    )
    supervisor_temperature: float = Field(
        default=0.0,
        description="Temperature for supervisor model (deterministic routing)"
    )

    # MCP (Model Context Protocol) Configuration
    use_mcp_servers: bool = Field(
        default=True,
        description="Use MCP servers for tools when available"
    )
    mcp_fallback_to_langchain: bool = Field(
        default=True,
        description="Fall back to LangChain tools if MCP servers are unavailable"
    )
    mcp_timeout_seconds: int = Field(
        default=30,
        description="Timeout for MCP server operations"
    )
    mcp_auto_start_servers: bool = Field(
        default=True,
        description="Automatically start MCP servers when needed"
    )
    mcp_debug: bool = Field(
        default=False,
        description="Enable MCP debugging output"
    )

    # Audio Transcription Configuration
    transcription_backend: TranscriptionBackend = Field(
        default=TranscriptionBackend.AUTO,
        description="Preferred transcription backend (auto selects best available)"
    )
    transcription_model: str = Field(
        default="turbo",
        description="Whisper model size (tiny, base, small, medium, large, large-v2, large-v3, turbo)"
    )
    transcription_language: Optional[str] = Field(
        default=None,
        description="Expected language for transcription (ISO 639-1 code, auto-detect if None)"
    )
    transcription_task: str = Field(
        default="transcribe",
        description="Transcription task: 'transcribe' or 'translate' to English"
    )
    enable_vad: bool = Field(
        default=True,
        description="Enable Voice Activity Detection to filter out silence"
    )
    enable_word_timestamps: bool = Field(
        default=False,
        description="Include word-level timestamps in transcription"
    )
    clean_transcript: bool = Field(
        default=True,
        description="Apply post-processing to clean transcripts (remove filler words, etc.)"
    )
    transcription_cache_hours: int = Field(
        default=24,
        description="Hours to cache transcription results"
    )
    max_audio_file_size_mb: int = Field(
        default=500,
        description="Maximum audio file size for transcription (MB)"
    )
    audio_chunk_duration_minutes: int = Field(
        default=30,
        description="Duration to split long audio files for processing (minutes)"
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig.

        Supports both runtime configuration and environment variable fallbacks
        following the LangGraph Platform pattern.
        """
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())

        # Build values dict with proper precedence: env vars > configurable > defaults
        values: dict[str, Any] = {}
        for field_name in field_names:
            env_value = os.environ.get(field_name.upper())
            if env_value is not None:
                # Convert string env values to appropriate types
                field_info = cls.model_fields[field_name]
                if field_info.annotation in (int, Optional[int]):
                    values[field_name] = int(env_value)
                elif field_info.annotation in (float, Optional[float]):
                    values[field_name] = float(env_value)
                elif field_info.annotation in (bool, Optional[bool]):
                    values[field_name] = env_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    values[field_name] = env_value
            elif field_name in configurable:
                values[field_name] = configurable[field_name]

        return cls(**values)

    def get_model_name(self, role: str) -> str:
        """Get model name for a specific role with fallback logic."""
        model_map = {
            "research": self.research_model,
            "writing": self.writing_model,
            "supervisor": self.supervisor_model,
        }
        return model_map.get(role, self.research_model)

    def get_model_max_tokens(self, role: str) -> int:
        """Get max tokens for a specific role."""
        token_map = {
            "research": self.research_model_max_tokens,
            "writing": self.writing_model_max_tokens,
            "supervisor": self.supervisor_model_max_tokens,
        }
        return token_map.get(role, self.research_model_max_tokens)

    def get_temperature(self, role: str) -> float:
        """Get temperature setting for a specific role."""
        temp_map = {
            "research": self.research_temperature,
            "writing": self.writing_temperature,
            "supervisor": self.supervisor_temperature,
        }
        return temp_map.get(role, 0.1)

    @property
    def mcp_config(self) -> Optional["MCPConfiguration"]:
        """Get MCP configuration if available."""
        if MCP_AVAILABLE and self.use_mcp_servers:
            return get_mcp_config()
        return None

    def is_mcp_enabled(self) -> bool:
        """Check if MCP servers are enabled and available."""
        return MCP_AVAILABLE and self.use_mcp_servers

    def should_use_mcp_for_tool(self, tool_type: str) -> bool:
        """Check if a specific tool type should use MCP."""
        if not self.is_mcp_enabled():
            return False

        mcp_cfg = self.mcp_config
        if not mcp_cfg:
            return False

        # Map tool types to MCP server types
        tool_server_map = {
            "search": "tavily_search",
            "vector_store": "vector_store",
            "document_processor": "document_processor",
            "arxiv": "arxiv_researcher",
            "web_research": "web_researcher",
            "workspace": "workspace_manager",
            "audio_transcription": "audio_transcription"
        }

        server_type = tool_server_map.get(tool_type)
        if not server_type:
            return False

        # Check if the corresponding MCP server is enabled
        from ..mcp_servers.config import MCPServerType
        try:
            return mcp_cfg.is_server_enabled(MCPServerType(server_type))
        except (ValueError, AttributeError):
            return False

    def get_search_preference(self) -> str:
        """Get search tool preference (MCP or LangChain)."""
        if self.should_use_mcp_for_tool("search"):
            return "mcp"
        elif self.mcp_fallback_to_langchain:
            return "langchain"
        else:
            return "none"

    def get_transcription_preference(self) -> str:
        """Get transcription tool preference (MCP or direct)."""
        if self.should_use_mcp_for_tool("audio_transcription"):
            return "mcp"
        else:
            return "direct"

    def get_transcription_backend(self) -> str:
        """Get the preferred transcription backend as string."""
        return self.transcription_backend.value

    def is_transcription_enabled(self) -> bool:
        """Check if audio transcription is enabled and available."""
        # Check if we have necessary dependencies or MCP server
        return (
            self.should_use_mcp_for_tool("audio_transcription") or
            self.transcription_backend != TranscriptionBackend.AUTO
        )

    def get_transcription_config(self) -> dict:
        """Get transcription configuration as dictionary."""
        return {
            "backend": self.get_transcription_backend(),
            "model": self.transcription_model,
            "language": self.transcription_language,
            "task": self.transcription_task,
            "enable_vad": self.enable_vad,
            "enable_word_timestamps": self.enable_word_timestamps,
            "clean_transcript": self.clean_transcript,
            "cache_hours": self.transcription_cache_hours,
            "max_file_size_mb": self.max_audio_file_size_mb,
            "chunk_duration_minutes": self.audio_chunk_duration_minutes
        }

    model_config = {
        "arbitrary_types_allowed": True,
        "use_enum_values": True
    }