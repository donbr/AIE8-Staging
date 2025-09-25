"""Meta-supervisor graph implementation using MessagesState.

Top-level coordination between research and document writing teams.
Modernized to use MessagesState patterns and subgraph composition.
"""

from typing import Dict, Any, Callable, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents.state import MetaSupervisorState
from ..agents.factory import create_supervisor_from_config, create_supervisor_node
from ..config.configuration import Configuration
from ..config.prompts import RESEARCH_SUPERVISOR_PROMPT  # Placeholder - should have meta supervisor prompt


def get_last_message(state: MetaSupervisorState) -> str:
    """Extract the last message content from state.

    Helper function to get the most recent message for subgraph processing.

    Args:
        state: Current meta-supervisor state

    Returns:
        Content of the last message
    """
    messages = state.get("messages", [])
    if messages:
        return messages[-1].content
    return ""


def join_graph(response: Dict[str, Any]) -> Dict[str, Any]:
    """Join subgraph response back to parent graph.

    Extracts the last message from subgraph response to include in parent state.

    Args:
        response: Response from subgraph execution

    Returns:
        Dictionary with messages for parent graph
    """
    messages = response.get("messages", [])
    if messages:
        return {"messages": [messages[-1]]}
    return {"messages": []}


def should_continue_meta(state: MetaSupervisorState) -> str:
    """Conditional edge function for meta-supervisor routing.

    Determines whether to route to research team, writing team, or finish.

    Args:
        state: Current meta-supervisor state

    Returns:
        Next team name or END
    """
    next_team = state.get("next", "")

    if next_team == "FINISH":
        return END
    elif next_team in ["research_team", "Research team"]:
        return "research_team"
    elif next_team in ["writing_team", "Response team"]:
        return "writing_team"
    else:
        return END


def create_meta_supervisor_node(
    config: Configuration,
    teams: List[str],
    **kwargs
) -> callable:
    """Create meta-supervisor node from configuration.

    Args:
        config: System configuration
        teams: List of team names to route between
        **kwargs: Additional supervisor configuration

    Returns:
        Meta-supervisor node function
    """
    # Use appropriate prompt for meta-supervisor
    meta_supervisor_prompt = kwargs.get(
        "system_prompt",
        "You are a supervisor tasked with managing a conversation between "
        "research and writing teams. Route to the appropriate team based on the request."
    )

    supervisor = create_supervisor_from_config(
        config=config,
        system_prompt=meta_supervisor_prompt,
        members=teams,
        role="supervisor"
    )

    return create_supervisor_node(supervisor)


def create_meta_supervisor_graph(
    config: Configuration,
    research_graph: StateGraph,
    writing_graph: StateGraph,
    with_checkpointer: bool = True,
    **kwargs
) -> StateGraph:
    """Create meta-supervisor graph using MessagesState patterns.

    Top-level coordination graph that routes between research and writing teams.
    Modernized with:
    - MessagesState for better message handling
    - Subgraph composition
    - Proper message joining between levels
    - Optional checkpointing

    Args:
        config: System configuration
        research_graph: Compiled research team graph
        writing_graph: Compiled document writing team graph
        with_checkpointer: Whether to include checkpointing
        **kwargs: Additional graph configuration

    Returns:
        Compiled meta-supervisor graph
    """
    # Team names for routing
    teams = ["research_team", "writing_team"]

    # Create meta-supervisor node
    supervisor_node = create_meta_supervisor_node(
        config=config,
        teams=teams,
        **kwargs
    )

    # Create subgraph wrapper functions
    def research_team_node(state):
        """Research team subgraph node."""
        # Extract message for subgraph
        message_content = get_last_message(state)

        # Invoke research subgraph
        result = research_graph.invoke({"messages": [message_content]})

        # Join response back to meta-supervisor state
        return join_graph(result)

    def writing_team_node(state):
        """Writing team subgraph node."""
        # Extract message for subgraph
        message_content = get_last_message(state)

        # Invoke writing subgraph
        result = writing_graph.invoke({"messages": [message_content]})

        # Join response back to meta-supervisor state
        return join_graph(result)

    # Build the graph
    graph = StateGraph(MetaSupervisorState)

    # Add team nodes
    graph.add_node("research_team", research_team_node)
    graph.add_node("writing_team", writing_team_node)

    # Add meta-supervisor node
    graph.add_node("MetaSupervisor", supervisor_node)

    # Define the workflow edges
    # Start with meta-supervisor
    graph.add_edge(START, "MetaSupervisor")

    # Conditional edges from supervisor to teams or END
    graph.add_conditional_edges(
        "MetaSupervisor",
        should_continue_meta,
        {
            "research_team": "research_team",
            "writing_team": "writing_team",
            "FINISH": END
        }
    )

    # Teams return to supervisor
    graph.add_edge("research_team", "MetaSupervisor")
    graph.add_edge("writing_team", "MetaSupervisor")

    # Compile with optional checkpointing
    if with_checkpointer:
        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)
    else:
        compiled_graph = graph.compile()

    return compiled_graph


def create_meta_nodes_for_notebook(
    research_chain,
    authoring_chain,
    super_supervisor_agent
):
    """Create meta-supervisor nodes that match the original notebook implementation.

    Provides backward compatibility for existing notebook code.

    Args:
        research_chain: Compiled research team chain
        authoring_chain: Compiled authoring team chain
        super_supervisor_agent: Super supervisor agent

    Returns:
        Tuple of node functions for notebook compatibility
    """
    def get_last_message_compat(state):
        """Notebook-compatible get_last_message."""
        messages = state.get("messages", [])
        if messages:
            return messages[-1].content
        return ""

    def join_graph_compat(response):
        """Notebook-compatible join_graph."""
        messages = response.get("messages", [])
        if messages:
            return {"messages": [messages[-1]]}
        return {"messages": []}

    def super_supervisor_node(state):
        """Super supervisor node for notebook compatibility."""
        result = super_supervisor_agent.invoke(state)
        return {"next": result["next"]}

    return (
        get_last_message_compat,
        join_graph_compat,
        super_supervisor_node
    )


def create_complete_multi_agent_system(
    config: Configuration,
    search_tool,
    rag_tool,
    file_tools: List[Any],
    working_directory_path: str,
    **kwargs
) -> StateGraph:
    """Create complete multi-agent system with all graphs.

    Factory function to create the entire hierarchical multi-agent system
    with proper configuration and tool setup.

    Args:
        config: System configuration
        search_tool: Search tool instance
        rag_tool: RAG tool instance
        file_tools: List of file management tools
        working_directory_path: Path to working directory
        **kwargs: Additional configuration options

    Returns:
        Compiled complete multi-agent system graph
    """
    from .research_team import create_research_team_graph
    from .writing_team import create_document_writing_graph

    # Create team graphs
    research_graph = create_research_team_graph(
        config=config,
        search_tool=search_tool,
        rag_tool=rag_tool,
        with_checkpointer=False,  # Individual teams don't need checkpointing
        **kwargs
    )

    writing_graph = create_document_writing_graph(
        config=config,
        file_tools=file_tools,
        working_directory_path=working_directory_path,
        with_checkpointer=False,  # Individual teams don't need checkpointing
        **kwargs
    )

    # Create meta-supervisor graph
    return create_meta_supervisor_graph(
        config=config,
        research_graph=research_graph,
        writing_graph=writing_graph,
        with_checkpointer=True,  # Main graph should have checkpointing
        **kwargs
    )


# Backward compatibility helpers
def get_meta_supervisor_info():
    """Get information about meta-supervisor graph structure."""
    return {
        "teams": ["research_team", "writing_team"],
        "supervisor": "MetaSupervisor",
        "state_class": "MetaSupervisorState",
        "routing": "Routes between research and writing teams",
        "features": ["Subgraph composition", "Message joining", "Hierarchical coordination"]
    }