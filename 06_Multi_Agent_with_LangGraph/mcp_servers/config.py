"""MCP Server Configuration Management.

Centralized configuration for all MCP servers in the Multi-Agent LangGraph system.
Integrates with the existing Pydantic configuration system.
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class MCPServerType(str, Enum):
    """Available MCP server types."""
    TAVILY_SEARCH = "tavily_search"
    VECTOR_STORE = "vector_store"
    DOCUMENT_PROCESSOR = "document_processor"
    ARXIV_RESEARCHER = "arxiv_researcher"
    WEB_RESEARCHER = "web_researcher"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    WORKSPACE_MANAGER = "workspace_manager"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""
    name: str
    type: MCPServerType
    enabled: bool = True
    transport_uri: str = "stdio://"
    api_key_env_var: Optional[str] = None
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class MCPConfiguration(BaseModel):
    """Overall MCP server configuration."""

    # Server definitions
    servers: List[MCPServerConfig] = Field(default_factory=lambda: [
        MCPServerConfig(
            name="tavily-search",
            type=MCPServerType.TAVILY_SEARCH,
            api_key_env_var="TAVILY_API_KEY",
            custom_settings={
                "cache_ttl_minutes": 15,
                "max_results_default": 5,
                "rate_limit_per_minute": 60
            }
        ),
        MCPServerConfig(
            name="vector-store",
            type=MCPServerType.VECTOR_STORE,
            custom_settings={
                "collection_name": "research_documents",
                "embedding_dimension": 1536,
                "similarity_threshold": 0.7
            }
        ),
        MCPServerConfig(
            name="document-processor",
            type=MCPServerType.DOCUMENT_PROCESSOR,
            custom_settings={
                "chunk_size": 750,
                "chunk_overlap": 50,
                "supported_formats": ["pdf", "docx", "txt", "html"]
            }
        ),
        MCPServerConfig(
            name="arxiv-researcher",
            type=MCPServerType.ARXIV_RESEARCHER,
            enabled=False,  # Enable after implementation
            custom_settings={
                "max_papers_per_search": 10,
                "include_abstracts": True,
                "download_pdfs": True
            }
        )
    ])

    # Global MCP settings
    enable_mcp: bool = Field(
        default=True,
        description="Enable MCP server integration"
    )

    fallback_to_langchain: bool = Field(
        default=True,
        description="Fall back to LangChain tools if MCP servers are unavailable"
    )

    mcp_timeout_seconds: int = Field(
        default=30,
        description="Timeout for MCP server operations"
    )

    debug_mcp: bool = Field(
        default=False,
        description="Enable MCP debugging output"
    )

    @classmethod
    def from_env(cls) -> "MCPConfiguration":
        """Create configuration from environment variables."""
        return cls(
            enable_mcp=os.getenv("ENABLE_MCP", "true").lower() == "true",
            fallback_to_langchain=os.getenv("MCP_FALLBACK_LANGCHAIN", "true").lower() == "true",
            mcp_timeout_seconds=int(os.getenv("MCP_TIMEOUT_SECONDS", "30")),
            debug_mcp=os.getenv("DEBUG_MCP", "false").lower() == "true"
        )

    def get_server_config(self, server_type: MCPServerType) -> Optional[MCPServerConfig]:
        """Get configuration for a specific server type."""
        for server in self.servers:
            if server.type == server_type and server.enabled:
                return server
        return None

    def get_enabled_servers(self) -> List[MCPServerConfig]:
        """Get all enabled server configurations."""
        return [server for server in self.servers if server.enabled]

    def is_server_enabled(self, server_type: MCPServerType) -> bool:
        """Check if a specific server type is enabled."""
        return self.get_server_config(server_type) is not None


# Global MCP configuration instance
mcp_config = MCPConfiguration.from_env()


def get_mcp_config() -> MCPConfiguration:
    """Get the global MCP configuration."""
    return mcp_config


def update_mcp_config(new_config: MCPConfiguration):
    """Update the global MCP configuration."""
    global mcp_config
    mcp_config = new_config