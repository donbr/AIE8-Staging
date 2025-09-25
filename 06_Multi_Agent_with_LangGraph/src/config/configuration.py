"""Configuration management for the Multi-Agent LangGraph system.

Following production patterns from open_deep_research for enterprise-grade configuration.
"""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    TAVILY = "tavily"
    NONE = "none"


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

    model_config = {
        "arbitrary_types_allowed": True,
        "use_enum_values": True
    }