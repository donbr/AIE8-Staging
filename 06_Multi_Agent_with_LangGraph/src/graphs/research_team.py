"""Research team graph implementation using MessagesState.

Multi-agent research workflow with search and RAG agents coordinated by a supervisor.
Modernized to use MessagesState patterns and configuration-driven agent creation.
"""

from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents.state import ResearchTeamState
from ..agents.factory import (
    create_agents_from_config,
    create_supervisor_from_config,
    create_agent_node,
    create_supervisor_node
)
from ..config.configuration import Configuration
from ..config.prompts import (
    SEARCH_AGENT_PROMPT,
    RAG_AGENT_PROMPT,
    RESEARCH_SUPERVISOR_PROMPT
)


def create_research_agent_nodes(
    config: Configuration,
    search_tool,
    rag_tool,
    **kwargs
) -> Dict[str, callable]:
    """Create research agent nodes from configuration.

    Args:
        config: System configuration
        search_tool: Search tool instance (e.g., Tavily)
        rag_tool: RAG tool instance for document retrieval
        **kwargs: Additional agent configuration options

    Returns:
        Dictionary of agent node functions
    """
    # Define agent configurations
    agent_configs = [
        {
            "name": "Search",
            "role": "research",
            "system_prompt": SEARCH_AGENT_PROMPT,
            "tools": [search_tool] if search_tool else []
        },
        {
            "name": "HowPeopleUseAIRetriever",
            "role": "research",
            "system_prompt": RAG_AGENT_PROMPT,
            "tools": [rag_tool] if rag_tool else []
        }
    ]

    # Create agents using configuration
    agents = create_agents_from_config(
        config=config,
        agent_configs=agent_configs,
        shared_tools=kwargs.get("shared_tools", [])
    )

    # Convert to node functions
    nodes = {}
    for name, agent in agents.items():
        nodes[name] = create_agent_node(agent, name)

    return nodes


def create_research_supervisor_node(
    config: Configuration,
    team_members: List[str],
    **kwargs
) -> callable:
    """Create research supervisor node from configuration.

    Args:
        config: System configuration
        team_members: List of team member names to route between
        **kwargs: Additional supervisor configuration

    Returns:
        Supervisor node function
    """
    supervisor = create_supervisor_from_config(
        config=config,
        system_prompt=RESEARCH_SUPERVISOR_PROMPT,
        members=team_members,
        role="supervisor"
    )

    return create_supervisor_node(supervisor)


def should_continue(state: ResearchTeamState) -> str:
    """Conditional edge function for research team routing.

    Determines whether to continue with team members or finish.

    Args:
        state: Current research team state

    Returns:
        Next node name or END
    """
    next_agent = state.get("next", "")

    if next_agent == "FINISH":
        return END
    else:
        return next_agent


def create_research_team_graph(
    config: Configuration,
    search_tool,
    rag_tool,
    with_checkpointer: bool = True,
    **kwargs
) -> StateGraph:
    """Create research team graph using MessagesState patterns.

    Modernized multi-agent research workflow with:
    - MessagesState for better message handling
    - Configuration-driven agent creation
    - Proper supervisor routing
    - Optional checkpointing

    Args:
        config: System configuration
        search_tool: Search tool instance
        rag_tool: RAG tool instance
        with_checkpointer: Whether to include checkpointing
        **kwargs: Additional graph configuration

    Returns:
        Compiled research team graph
    """
    # Team member names
    team_members = ["Search", "HowPeopleUseAIRetriever"]

    # Create agent nodes
    agent_nodes = create_research_agent_nodes(
        config=config,
        search_tool=search_tool,
        rag_tool=rag_tool,
        **kwargs
    )

    # Create supervisor node
    supervisor_node = create_research_supervisor_node(
        config=config,
        team_members=team_members,
        **kwargs
    )

    # Build the graph
    graph = StateGraph(ResearchTeamState)

    # Add agent nodes
    for name, node in agent_nodes.items():
        graph.add_node(name, node)

    # Add supervisor node
    graph.add_node("ResearchSupervisor", supervisor_node)

    # Define the workflow edges
    # Start with supervisor
    graph.add_edge(START, "ResearchSupervisor")

    # Conditional edges from supervisor to agents or END
    graph.add_conditional_edges(
        "ResearchSupervisor",
        should_continue,
        {member: member for member in team_members} | {"FINISH": END}
    )

    # All agents return to supervisor
    for member in team_members:
        graph.add_edge(member, "ResearchSupervisor")

    # Compile with optional checkpointing
    if with_checkpointer:
        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)
    else:
        compiled_graph = graph.compile()

    return compiled_graph


def create_research_nodes_for_notebook(
    search_agent,
    research_agent,
    research_supervisor_agent
):
    """Create research nodes that match the original notebook implementation.

    Provides backward compatibility for existing notebook code.

    Args:
        search_agent: Search agent executor
        research_agent: Research agent executor
        research_supervisor_agent: Research supervisor chain

    Returns:
        Tuple of (search_node, research_node, supervisor_node) functions
    """
    # Create node wrappers for notebook compatibility
    def search_node(state):
        result = search_agent.invoke({"messages": state["messages"]})
        return {"messages": [result["output"]]}

    def research_node(state):
        result = research_agent.invoke({"messages": state["messages"]})
        return {"messages": [result["output"]]}

    def supervisor_agent_node(state):
        result = research_supervisor_agent.invoke(state)
        return {"next": result["next"]}

    return search_node, research_node, supervisor_agent_node


# Backward compatibility helpers
def get_research_team_members() -> List[str]:
    """Get standard research team member names."""
    return ["Search", "HowPeopleUseAIRetriever"]


def get_research_graph_info():
    """Get information about research team graph structure."""
    return {
        "team_members": get_research_team_members(),
        "supervisor": "ResearchSupervisor",
        "state_class": "ResearchTeamState",
        "routing": "Conditional edges based on supervisor decisions"
    }