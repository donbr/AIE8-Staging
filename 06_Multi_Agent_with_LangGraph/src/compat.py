"""Backward compatibility layer for the notebook implementation.

This module provides the same interface as the original notebook hardcoded variables
but uses the new configuration system underneath. This allows the notebook to continue
working while we gradually refactor to the new architecture.
"""

import os
from pathlib import Path
from typing import Optional

from .config.configuration import Configuration
from .core.factory import (
    get_research_llm,
    get_authoring_llm,
    get_super_llm,
    get_embedding_model,
    get_generator_llm,
    create_search_tool
)
from .agents.state import (
    SimpleRAGState,
    ResearchTeamState,
    DocWritingState,
    MetaSupervisorState,
    convert_typeddict_to_messages_state
)

# Global configuration instance (lazy loaded)
_config: Optional[Configuration] = None


def get_config() -> Configuration:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Configuration()
    return _config


def reset_config():
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None


# Backward compatible variables that match the original notebook
# These can be imported directly to maintain notebook compatibility

def _lazy_init():
    """Lazy initialization function to set up all notebook variables."""
    config = get_config()

    # LLM instances (matching original notebook variable names)
    global research_llm, authoring_llm, super_llm, generator_llm, embedding_model
    global search_tool, chunk_size, chunk_overlap

    research_llm = get_research_llm(config)
    authoring_llm = get_authoring_llm(config)
    super_llm = get_super_llm(config)
    generator_llm = get_generator_llm(config)
    embedding_model = get_embedding_model(config)

    # Tools
    search_tool = create_search_tool(config)

    # RAG configuration
    chunk_size = config.chunk_size
    chunk_overlap = config.chunk_overlap

    return config


# Initialize on import
config = _lazy_init()

# Export all the backward-compatible variables
__all__ = [
    'config',
    'research_llm',
    'authoring_llm',
    'super_llm',
    'generator_llm',
    'embedding_model',
    'search_tool',
    'chunk_size',
    'chunk_overlap',
    'get_config',
    'reset_config',
    # State classes for MessagesState migration
    'SimpleRAGState',
    'ResearchTeamState',
    'DocWritingState',
    'MetaSupervisorState',
    'State',  # Alias for SimpleRAGState
    'tiktoken_len',
    'create_agent',
    'create_team_supervisor'
]

# State class aliases for backward compatibility
State = SimpleRAGState  # Primary alias for basic RAG workflows


# Notebook helper functions with new configuration support
def tiktoken_len(text: str) -> int:
    """Calculate token length using tiktoken (unchanged from notebook)."""
    import tiktoken
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)


def create_agent(llm, tools, system_prompt: str):
    """Create an agent using the original notebook pattern.

    Enhanced to use the new prompt formatting system.
    """
    from langchain.agents import AgentExecutor, create_openai_functions_agent
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from .config.prompts import BASE_AGENT_INSTRUCTION

    # Enhance system prompt with base instructions
    enhanced_prompt = f"{system_prompt}\n\n{BASE_AGENT_INSTRUCTION}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return executor


def create_team_supervisor(llm, system_prompt: str, members: list):
    """Create a team supervisor using the original notebook pattern."""
    from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": members},
                    ],
                }
            },
            "required": ["next"],
        },
    }

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(members + ["FINISH"]), team_members=", ".join(members))

    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )