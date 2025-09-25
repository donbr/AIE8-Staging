"""MCP Integration Layer for Multi-Agent LangGraph System.

Provides compatibility layer between MCP servers and existing LangChain-based tools,
ensuring backward compatibility while enabling enhanced MCP functionality.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from functools import wraps

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# MCP imports (optional)
try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters
    from mcp.types import CallToolResult, TextContent
    MCP_CLIENT_AVAILABLE = True
except ImportError:
    MCP_CLIENT_AVAILABLE = False
    ClientSession = None
    StdioServerParameters = None


class MCPToolWrapper(BaseTool):
    """Wrapper that makes MCP tools compatible with LangChain agents."""

    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    mcp_server_name: str = Field(description="MCP server name")
    mcp_tool_name: str = Field(description="MCP tool name")
    client_session: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _get_client_session(self):
        """Get or create MCP client session."""
        if self.client_session is None:
            # This would be managed by the MCP server manager
            # For now, return None to trigger fallback
            return None
        return self.client_session

    def _run(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Run the MCP tool synchronously."""
        try:
            # Use asyncio to run the async method
            return asyncio.run(self._arun(query, **kwargs))
        except Exception as e:
            logging.getLogger(f"mcp.tool.{self.name}").error(f"Error running MCP tool {self.name}: {e}")
            return f"Error: {str(e)}"

    async def _arun(
        self,
        query: str,
        **kwargs: Any,
    ) -> str:
        """Run the MCP tool asynchronously."""
        session = await self._get_client_session()
        if session is None:
            return "MCP server not available"

        try:
            # Call the MCP tool
            arguments = {"query": query, **kwargs}
            result = await session.call_tool(
                name=self.mcp_tool_name,
                arguments=arguments
            )

            # Process the result
            if isinstance(result, CallToolResult):
                content_parts = []
                for content in result.content:
                    if isinstance(content, TextContent):
                        content_parts.append(content.text)
                    else:
                        content_parts.append(str(content))
                return "\n".join(content_parts)
            else:
                return str(result)

        except Exception as e:
            logging.getLogger(f"mcp.tool.{self.name}").error(f"Error calling MCP tool {self.mcp_tool_name}: {e}")
            return f"MCP Error: {str(e)}"


class MCPToolFactory:
    """Factory for creating LangChain-compatible MCP tools."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_sessions: Dict[str, Any] = {}

    async def create_mcp_search_tool(
        self,
        server_name: str = "tavily-search",
        tool_name: str = "tavily_web_search",
        max_results: int = 5
    ) -> MCPToolWrapper:
        """Create MCP-based search tool."""
        return MCPToolWrapper(
            name="tavily_search",
            description="Search the web using Tavily API via MCP server for up-to-date information",
            mcp_server_name=server_name,
            mcp_tool_name=tool_name,
            client_session=await self._get_session(server_name)
        )

    async def create_mcp_vector_tool(
        self,
        server_name: str = "vector-store",
        tool_name: str = "vector_search"
    ) -> MCPToolWrapper:
        """Create MCP-based vector store tool."""
        return MCPToolWrapper(
            name="vector_retrieval",
            description="Retrieve relevant documents from vector store via MCP server",
            mcp_server_name=server_name,
            mcp_tool_name=tool_name,
            client_session=await self._get_session(server_name)
        )

    async def create_mcp_document_tool(
        self,
        server_name: str = "document-processor",
        tool_name: str = "process_document"
    ) -> MCPToolWrapper:
        """Create MCP-based document processing tool."""
        return MCPToolWrapper(
            name="document_processor",
            description="Process and extract content from documents via MCP server",
            mcp_server_name=server_name,
            mcp_tool_name=tool_name,
            client_session=await self._get_session(server_name)
        )

    async def create_mcp_arxiv_tool(
        self,
        server_name: str = "arxiv-researcher",
        tool_name: str = "search_arxiv"
    ) -> MCPToolWrapper:
        """Create MCP-based ArXiv research tool."""
        return MCPToolWrapper(
            name="arxiv_search",
            description="Search and retrieve academic papers from ArXiv via MCP server",
            mcp_server_name=server_name,
            mcp_tool_name=tool_name,
            client_session=await self._get_session(server_name)
        )

    async def create_mcp_audio_transcription_tool(
        self,
        server_name: str = "audio-transcription",
        tool_name: str = "transcribe_audio"
    ) -> MCPToolWrapper:
        """Create MCP-based audio transcription tool."""
        return MCPToolWrapper(
            name="audio_transcription",
            description="Transcribe audio files using multiple backend options via MCP server",
            mcp_server_name=server_name,
            mcp_tool_name=tool_name,
            client_session=await self._get_session(server_name)
        )

    async def _get_session(self, server_name: str) -> Optional[Any]:
        """Get or create MCP client session for a server."""
        if not MCP_CLIENT_AVAILABLE:
            return None

        if server_name not in self.active_sessions:
            # This would typically connect to the actual MCP server
            # For now, we'll return None to trigger fallback behavior
            self.logger.warning(f"MCP session for {server_name} not implemented yet")
            return None

        return self.active_sessions.get(server_name)

    async def close_all_sessions(self):
        """Close all active MCP sessions."""
        for session in self.active_sessions.values():
            if hasattr(session, 'close'):
                await session.close()
        self.active_sessions.clear()


# Global MCP tool factory
_mcp_factory: Optional[MCPToolFactory] = None


def get_mcp_factory() -> MCPToolFactory:
    """Get the global MCP tool factory."""
    global _mcp_factory
    if _mcp_factory is None:
        _mcp_factory = MCPToolFactory()
    return _mcp_factory


def mcp_fallback(langchain_fallback_func):
    """Decorator to provide LangChain fallback for MCP tools."""
    @wraps(langchain_fallback_func)
    async def wrapper(*args, **kwargs):
        from .config.configuration import Configuration

        config = kwargs.get('config')
        if not isinstance(config, Configuration):
            # If no config provided, create default
            config = Configuration()

        # Try MCP first if enabled
        if config.is_mcp_enabled():
            try:
                mcp_result = await _try_mcp_tool(*args, **kwargs)
                if mcp_result is not None:
                    return mcp_result
            except Exception as e:
                logging.getLogger(__name__).warning(f"MCP tool failed: {e}")

        # Fall back to LangChain if enabled
        if config.mcp_fallback_to_langchain:
            return langchain_fallback_func(*args, **kwargs)
        else:
            raise RuntimeError("MCP tool failed and LangChain fallback is disabled")

    return wrapper


async def _try_mcp_tool(*args, **kwargs) -> Optional[Any]:
    """Try to execute MCP tool - placeholder implementation."""
    # This would implement the actual MCP tool execution
    # For now, return None to trigger fallback
    return None


class MCPCompatibilityLayer:
    """Compatibility layer for integrating MCP with existing LangChain tools."""

    def __init__(self, config: "Configuration"):
        self.config = config
        self.mcp_factory = get_mcp_factory()
        self.logger = logging.getLogger(__name__)

    async def get_search_tool(self, **kwargs):
        """Get search tool (MCP or LangChain based on config)."""
        preference = self.config.get_search_preference()

        if preference == "mcp":
            try:
                return await self.mcp_factory.create_mcp_search_tool(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create MCP search tool: {e}")
                if not self.config.mcp_fallback_to_langchain:
                    raise

        # Fall back to LangChain tool
        if preference != "none":
            from .core.factory import create_search_tool
            return create_search_tool(self.config, **kwargs)

        return None

    async def get_vector_tool(self, **kwargs):
        """Get vector retrieval tool."""
        if self.config.should_use_mcp_for_tool("vector_store"):
            try:
                return await self.mcp_factory.create_mcp_vector_tool(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create MCP vector tool: {e}")
                if not self.config.mcp_fallback_to_langchain:
                    raise

        # Fallback would be to existing RAG tool
        return None

    async def get_document_tool(self, **kwargs):
        """Get document processing tool."""
        if self.config.should_use_mcp_for_tool("document_processor"):
            try:
                return await self.mcp_factory.create_mcp_document_tool(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create MCP document tool: {e}")
                if not self.config.mcp_fallback_to_langchain:
                    raise

        # Fallback would be to existing file management tools
        return None

    async def get_arxiv_tool(self, **kwargs):
        """Get ArXiv research tool (implements notebook bonus activity)."""
        if self.config.should_use_mcp_for_tool("arxiv"):
            try:
                return await self.mcp_factory.create_mcp_arxiv_tool(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create MCP ArXiv tool: {e}")
                if not self.config.mcp_fallback_to_langchain:
                    raise

        # No LangChain fallback for ArXiv - this is a new MCP-only feature
        return None

    async def get_audio_transcription_tool(self, **kwargs):
        """Get audio transcription tool."""
        preference = self.config.get_transcription_preference()

        if preference == "mcp":
            try:
                return await self.mcp_factory.create_mcp_audio_transcription_tool(**kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create MCP audio transcription tool: {e}")
                if not self.config.mcp_fallback_to_langchain:
                    raise

        # For audio transcription, we can fall back to direct implementation
        # This would use the transcription libraries directly without MCP
        return self._create_direct_transcription_tool(**kwargs)

    def _create_direct_transcription_tool(self, **kwargs):
        """Create direct transcription tool (non-MCP fallback)."""
        # This would implement direct transcription using faster-whisper, etc.
        # For now, return a placeholder that uses the transcription server directly
        class DirectTranscriptionTool(BaseTool):
            name: str = "direct_audio_transcription"
            description: str = "Transcribe audio files using direct library access"

            def _run(self, file_path: str, **kwargs: Any) -> str:
                try:
                    # This would implement direct transcription
                    return f"Direct transcription of {file_path} (not yet implemented)"
                except Exception as e:
                    return f"Direct transcription error: {str(e)}"

            async def _arun(self, file_path: str, **kwargs: Any) -> str:
                return self._run(file_path, **kwargs)

        return DirectTranscriptionTool()

    async def close(self):
        """Close all MCP connections."""
        await self.mcp_factory.close_all_sessions()


# Global compatibility layer instance
_compatibility_layer: Optional[MCPCompatibilityLayer] = None


def get_compatibility_layer(config: "Configuration") -> MCPCompatibilityLayer:
    """Get the global MCP compatibility layer."""
    global _compatibility_layer
    if _compatibility_layer is None:
        _compatibility_layer = MCPCompatibilityLayer(config)
    return _compatibility_layer


# Convenience functions for common MCP tools
async def get_mcp_search_tool(config: "Configuration", **kwargs):
    """Get MCP search tool with fallback."""
    layer = get_compatibility_layer(config)
    return await layer.get_search_tool(**kwargs)


async def get_mcp_vector_tool(config: "Configuration", **kwargs):
    """Get MCP vector tool with fallback."""
    layer = get_compatibility_layer(config)
    return await layer.get_vector_tool(**kwargs)


async def get_mcp_document_tool(config: "Configuration", **kwargs):
    """Get MCP document tool with fallback."""
    layer = get_compatibility_layer(config)
    return await layer.get_document_tool(**kwargs)


async def get_mcp_arxiv_tool(config: "Configuration", **kwargs):
    """Get MCP ArXiv tool."""
    layer = get_compatibility_layer(config)
    return await layer.get_arxiv_tool(**kwargs)


async def get_mcp_audio_transcription_tool(config: "Configuration", **kwargs):
    """Get MCP audio transcription tool with fallback."""
    layer = get_compatibility_layer(config)
    return await layer.get_audio_transcription_tool(**kwargs)