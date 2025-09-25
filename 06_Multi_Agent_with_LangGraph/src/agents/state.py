"""State management classes for the Multi-Agent LangGraph system.

Modernized to use MessagesState pattern following LangGraph 2025 best practices.
Migrated from TypedDict to MessagesState for better message handling and compatibility.
"""

from typing import List, Dict, Optional
from typing_extensions import Annotated
from pydantic import Field

from langgraph.graph import MessagesState
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document


class SimpleRAGState(MessagesState):
    """State for simple RAG operations.

    Replaces the original TypedDict-based State class for basic RAG workflows.
    MessagesState automatically handles message list management.
    """
    question: str = Field(default="", description="User's question for RAG")
    context: List[Document] = Field(default_factory=list, description="Retrieved documents")
    response: str = Field(default="", description="Generated response")


class ResearchTeamState(MessagesState):
    """State for research team multi-agent workflows.

    Migrated from TypedDict to MessagesState for better message handling.
    MessagesState automatically manages the messages list with proper annotations.
    """
    team_members: List[str] = Field(
        default_factory=list,
        description="Available team members for research tasks"
    )
    next: str = Field(
        default="",
        description="Next agent to route to (or FINISH)"
    )


class DocWritingState(MessagesState):
    """State for document writing team multi-agent workflows.

    Migrated from TypedDict to MessagesState following LangGraph 2025 patterns.
    Includes file management capabilities for document creation workflows.
    """
    team_members: List[str] = Field(
        default_factory=list,
        description="Available team members for document writing"
    )
    next: str = Field(
        default="",
        description="Next agent to route to (or FINISH)"
    )
    current_files: Dict[str, str] = Field(
        default_factory=dict,
        description="Currently managed files and their contents"
    )


class MetaSupervisorState(MessagesState):
    """State for meta-supervisor graph coordination.

    Top-level state for coordinating between research and writing teams.
    Simplified compared to team-specific states as it focuses on routing.
    """
    next: str = Field(
        default="",
        description="Next team/graph to route to (research_team, writing_team, or FINISH)"
    )


# Backward compatibility aliases for existing notebook code
State = SimpleRAGState  # For basic RAG workflow compatibility


def convert_typeddict_to_messages_state(state_dict: dict, state_class) -> dict:
    """Convert TypedDict state format to MessagesState format.

    Helper function for backward compatibility during migration.
    Ensures existing notebook code continues to work with new state classes.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    # MessagesState handles messages automatically, so we just need to ensure
    # other fields are properly formatted
    converted_state = {}

    for key, value in state_dict.items():
        if key == "messages":
            # MessagesState handles this automatically
            converted_state[key] = value
        elif key in ["team_members", "context"] and not isinstance(value, list):
            # Ensure list fields are properly formatted
            converted_state[key] = value if isinstance(value, list) else [value]
        elif key == "current_files" and not isinstance(value, dict):
            # Ensure dict fields are properly formatted
            converted_state[key] = value if isinstance(value, dict) else {}
        else:
            converted_state[key] = value

    return converted_state


def get_state_schema_info():
    """Get information about available state schemas.

    Useful for debugging and understanding state structure during migration.
    """
    schemas = {
        "SimpleRAGState": {
            "base": "MessagesState",
            "fields": ["question", "context", "response"],
            "replaces": "Original State(TypedDict)"
        },
        "ResearchTeamState": {
            "base": "MessagesState",
            "fields": ["team_members", "next"],
            "replaces": "Original ResearchTeamState(TypedDict)"
        },
        "DocWritingState": {
            "base": "MessagesState",
            "fields": ["team_members", "next", "current_files"],
            "replaces": "Original DocWritingState(TypedDict)"
        },
        "MetaSupervisorState": {
            "base": "MessagesState",
            "fields": ["next"],
            "replaces": "Original State(TypedDict) for meta-supervisor"
        }
    }
    return schemas