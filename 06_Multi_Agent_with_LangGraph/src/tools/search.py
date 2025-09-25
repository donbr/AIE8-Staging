"""Search tools for the Multi-Agent LangGraph system.

Provides search capabilities using various search APIs with proper configuration management.
Extracted from notebook and modernized for production use.
"""

from typing import Optional, Dict, Any, List

from ..config.configuration import Configuration, SearchAPI


def create_tavily_search_tool(max_results: int = 5):
    """Create a Tavily search tool.

    Args:
        max_results: Maximum number of search results to return

    Returns:
        TavilySearchResults tool instance
    """
    from langchain_community.tools.tavily_search import TavilySearchResults

    return TavilySearchResults(max_results=max_results)


def create_search_tool_from_config(config: Configuration) -> Optional[Any]:
    """Create search tool based on configuration.

    Args:
        config: Configuration instance

    Returns:
        Search tool instance or None if search is disabled
    """
    # This is already implemented in core.factory, but provided here for completeness
    from ..core.factory import create_search_tool

    return create_search_tool(config)


def get_search_tool_info(search_api: str) -> Dict[str, Any]:
    """Get information about a specific search tool.

    Args:
        search_api: Search API name

    Returns:
        Dictionary with tool information
    """
    info_map = {
        SearchAPI.TAVILY.value: {
            "name": "Tavily Search",
            "description": "Web search using Tavily API",
            "requires_api_key": True,
            "env_var": "TAVILY_API_KEY",
            "max_results_default": 5
        },
        SearchAPI.OPENAI.value: {
            "name": "OpenAI Search (Future)",
            "description": "Search using OpenAI capabilities",
            "requires_api_key": True,
            "env_var": "OPENAI_API_KEY",
            "status": "Not implemented"
        },
        SearchAPI.ANTHROPIC.value: {
            "name": "Anthropic Search (Future)",
            "description": "Search using Anthropic capabilities",
            "requires_api_key": True,
            "env_var": "ANTHROPIC_API_KEY",
            "status": "Not implemented"
        },
        SearchAPI.NONE.value: {
            "name": "No Search",
            "description": "Search disabled",
            "requires_api_key": False,
            "status": "Disabled"
        }
    }

    return info_map.get(search_api, {"name": "Unknown", "status": "Unsupported"})


# Backward compatibility for notebook
def create_notebook_compatible_search_tool():
    """Create search tool that matches notebook pattern.

    Returns:
        TavilySearchResults with default settings
    """
    return create_tavily_search_tool(max_results=5)


# Export the main search tool for direct import
tavily_tool = None  # Will be set by lazy initialization


def get_tavily_tool():
    """Get or create the global Tavily tool instance."""
    global tavily_tool
    if tavily_tool is None:
        tavily_tool = create_tavily_search_tool()
    return tavily_tool