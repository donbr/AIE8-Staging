"""Factory functions for creating LLM instances and other services.

Provides clean abstraction for service creation based on configuration,
following production patterns for dependency injection and service management.
"""

from typing import Optional, Any, Union
import asyncio
import logging

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

from ..config.configuration import Configuration, SearchAPI

# MCP integration (optional)
try:
    from ..mcp_integration import get_compatibility_layer
    MCP_INTEGRATION_AVAILABLE = True
except ImportError:
    MCP_INTEGRATION_AVAILABLE = False
    get_compatibility_layer = None


def create_chat_model(
    config: Configuration,
    role: str = "research",
    **kwargs
) -> BaseChatModel:
    """Create a chat model instance based on configuration and role.

    Args:
        config: Configuration instance with model settings
        role: Model role ("research", "writing", "supervisor")
        **kwargs: Additional arguments to pass to the model

    Returns:
        Configured chat model instance
    """
    model_name = config.get_model_name(role)
    max_tokens = config.get_model_max_tokens(role)
    temperature = config.get_temperature(role)

    # Override with any provided kwargs
    model_kwargs = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs
    }

    # Currently supporting OpenAI models, can be extended for other providers
    return ChatOpenAI(**model_kwargs)


def create_embedding_model(config: Configuration) -> OpenAIEmbeddings:
    """Create an embedding model instance based on configuration.

    Args:
        config: Configuration instance with embedding settings

    Returns:
        Configured embedding model instance
    """
    return OpenAIEmbeddings(model=config.embedding_model)


def create_search_tool(
    config: Configuration,
    max_results: Optional[int] = None
) -> Optional[Union[TavilySearchResults, Any]]:
    """Create a search tool instance based on configuration.

    Args:
        config: Configuration instance with search API settings
        max_results: Maximum number of search results (optional)

    Returns:
        Configured search tool instance or None if search is disabled
    """
    logger = logging.getLogger(__name__)

    # Try MCP integration first if enabled
    if MCP_INTEGRATION_AVAILABLE and config.is_mcp_enabled():
        search_preference = config.get_search_preference()
        if search_preference == "mcp":
            try:
                # Use asyncio to get MCP tool
                compatibility_layer = get_compatibility_layer(config)
                mcp_tool = asyncio.run(compatibility_layer.get_search_tool(
                    max_results=max_results or 5
                ))
                if mcp_tool is not None:
                    logger.info("Using MCP search tool")
                    return mcp_tool
            except Exception as e:
                logger.warning(f"Failed to create MCP search tool: {e}")
                if not config.mcp_fallback_to_langchain:
                    raise

    # Fall back to LangChain implementation
    logger.info("Using LangChain search tool")

    # Handle both enum and string values due to use_enum_values=True
    search_api_value = config.search_api
    if isinstance(search_api_value, SearchAPI):
        search_api_value = search_api_value.value

    if search_api_value == SearchAPI.NONE.value:
        return None

    if search_api_value == SearchAPI.TAVILY.value:
        search_kwargs = {}
        if max_results:
            search_kwargs["max_results"] = max_results

        return TavilySearchResults(**search_kwargs)

    elif search_api_value == SearchAPI.OPENAI.value:
        # TODO: Implement OpenAI search integration
        raise NotImplementedError("OpenAI search not yet implemented")

    elif search_api_value == SearchAPI.ANTHROPIC.value:
        # TODO: Implement Anthropic search integration
        raise NotImplementedError("Anthropic search not yet implemented")

    else:
        raise ValueError(f"Unsupported search API: {search_api_value}")


def create_research_llm(config: Configuration, **kwargs) -> BaseChatModel:
    """Create LLM instance specifically configured for research tasks.

    Args:
        config: Configuration instance
        **kwargs: Additional arguments for the model

    Returns:
        Chat model configured for research
    """
    return create_chat_model(config, role="research", **kwargs)


def create_writing_llm(config: Configuration, **kwargs) -> BaseChatModel:
    """Create LLM instance specifically configured for writing tasks.

    Args:
        config: Configuration instance
        **kwargs: Additional arguments for the model

    Returns:
        Chat model configured for writing
    """
    return create_chat_model(config, role="writing", **kwargs)


def create_supervisor_llm(config: Configuration, **kwargs) -> BaseChatModel:
    """Create LLM instance specifically configured for supervisor tasks.

    Args:
        config: Configuration instance
        **kwargs: Additional arguments for the model

    Returns:
        Chat model configured for supervision/routing
    """
    return create_chat_model(config, role="supervisor", **kwargs)


# Backward compatibility functions for notebook integration
def get_research_llm(config: Configuration) -> BaseChatModel:
    """Backward compatible function name for research LLM.

    Matches the original notebook variable: research_llm
    """
    return create_research_llm(config)


def get_authoring_llm(config: Configuration) -> BaseChatModel:
    """Backward compatible function name for authoring LLM.

    Matches the original notebook variable: authoring_llm
    """
    return create_writing_llm(config)


def get_super_llm(config: Configuration) -> BaseChatModel:
    """Backward compatible function name for supervisor LLM.

    Matches the original notebook variable: super_llm
    """
    return create_supervisor_llm(config)


def get_embedding_model(config: Configuration) -> OpenAIEmbeddings:
    """Backward compatible function name for embedding model.

    Matches the original notebook variable: embedding_model
    """
    return create_embedding_model(config)


def get_generator_llm(config: Configuration) -> BaseChatModel:
    """Backward compatible function name for generator LLM.

    Matches the original notebook variable: generator_llm
    """
    return create_chat_model(config, role="research", model="gpt-4o-nano")  # Special case for RAG