"""Tool modules for the Multi-Agent LangGraph system.

Provides organized tool creation functions for search, file management, and RAG capabilities.
All tools are extracted from notebook implementation and modernized for production use.
"""

from .search import (
    create_tavily_search_tool,
    create_search_tool_from_config,
    get_search_tool_info,
    create_notebook_compatible_search_tool,
    get_tavily_tool
)

from .file_management import (
    create_file_management_tools,
    create_rag_tool,
    create_reference_tool,
    get_file_tools_info,
    create_notebook_compatible_file_tools
)

__all__ = [
    # Search tools
    "create_tavily_search_tool",
    "create_search_tool_from_config",
    "get_search_tool_info",
    "create_notebook_compatible_search_tool",
    "get_tavily_tool",

    # File management tools
    "create_file_management_tools",
    "create_rag_tool",
    "create_reference_tool",
    "get_file_tools_info",
    "create_notebook_compatible_file_tools",
]


def get_available_tools():
    """Get information about all available tool types."""
    return {
        "search_tools": {
            "tavily_search": {
                "description": "Web search using Tavily API",
                "factory_function": "create_tavily_search_tool",
                "requires_api_key": True,
                "env_var": "TAVILY_API_KEY"
            }
        },
        "file_tools": {
            "create_outline": {
                "description": "Create structured outlines from list of points",
                "inputs": ["points: List[str]", "file_name: str"]
            },
            "write_document": {
                "description": "Write content to a new document file",
                "inputs": ["content: str", "file_name: str"]
            },
            "read_document": {
                "description": "Read existing document with optional line range",
                "inputs": ["file_name: str", "start: Optional[int]", "end: Optional[int]"]
            },
            "edit_document": {
                "description": "Edit document by inserting text at specific lines",
                "inputs": ["file_name: str", "inserts: Dict[int, str]"]
            }
        },
        "rag_tools": {
            "rag_retriever": {
                "description": "Retrieve relevant documents from vector store",
                "factory_function": "create_rag_tool",
                "inputs": ["query: str"]
            },
            "reference_tool": {
                "description": "Reference previous responses and work",
                "factory_function": "create_reference_tool",
                "inputs": ["query: str"]
            }
        }
    }


def create_tools_from_config(config, working_directory=None, retriever=None):
    """Create all tools from configuration.

    Args:
        config: Configuration instance
        working_directory: Path to working directory (for file tools)
        retriever: Vector store retriever (for RAG tools)

    Returns:
        Dictionary of organized tools
    """
    tools = {
        "search": None,
        "file_tools": [],
        "rag_tool": None,
        "reference_tool": None
    }

    # Create search tool
    tools["search"] = create_search_tool_from_config(config)

    # Create file management tools if working directory provided
    if working_directory:
        from pathlib import Path
        working_dir = Path(working_directory)
        tools["file_tools"] = create_file_management_tools(working_dir)
        tools["reference_tool"] = create_reference_tool(working_dir)

    # Create RAG tool if retriever provided
    if retriever:
        tools["rag_tool"] = create_rag_tool(retriever)

    return tools


# Backward compatibility for notebook
def create_notebook_compatible_tools(working_directory_path: str, retriever=None):
    """Create tools that match notebook implementation patterns.

    Args:
        working_directory_path: String path to working directory
        retriever: Optional retriever for RAG tool

    Returns:
        Tuple of (search_tool, file_tools, reference_tool, rag_tool)
    """
    search_tool = create_notebook_compatible_search_tool()
    file_tools, reference_tool = create_notebook_compatible_file_tools(working_directory_path)
    rag_tool = create_rag_tool(retriever) if retriever else None

    return search_tool, file_tools, reference_tool, rag_tool