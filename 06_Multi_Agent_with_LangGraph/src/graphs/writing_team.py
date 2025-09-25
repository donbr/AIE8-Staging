"""Document writing team graph implementation using MessagesState.

Multi-agent document writing workflow with specialized writing agents coordinated by a supervisor.
Modernized to use MessagesState patterns and configuration-driven agent creation.
"""

from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from ..agents.state import DocWritingState
from ..agents.factory import (
    create_agents_from_config,
    create_supervisor_from_config,
    create_agent_node,
    create_supervisor_node
)
from ..config.configuration import Configuration
from ..config.prompts import (
    DOC_WRITER_PROMPT,
    COPY_EDITOR_PROMPT,
    AUTHORING_SUPERVISOR_PROMPT
)


def create_writing_agent_nodes(
    config: Configuration,
    file_tools: List[Any],
    **kwargs
) -> Dict[str, callable]:
    """Create document writing agent nodes from configuration.

    Args:
        config: System configuration
        file_tools: List of file management tools
        **kwargs: Additional agent configuration options

    Returns:
        Dictionary of agent node functions
    """
    # Define agent configurations
    agent_configs = [
        {
            "name": "DocWriter",
            "role": "writing",
            "system_prompt": DOC_WRITER_PROMPT,
            "tools": file_tools
        },
        {
            "name": "NoteTaker",
            "role": "writing",
            "system_prompt": "You are an expert at taking notes and creating outlines. "
                           "Help organize information and create structured documents.",
            "tools": file_tools
        },
        {
            "name": "CopyEditor",
            "role": "writing",
            "system_prompt": COPY_EDITOR_PROMPT,
            "tools": file_tools
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


def create_writing_supervisor_node(
    config: Configuration,
    team_members: List[str],
    **kwargs
) -> callable:
    """Create document writing supervisor node from configuration.

    Args:
        config: System configuration
        team_members: List of team member names to route between
        **kwargs: Additional supervisor configuration

    Returns:
        Supervisor node function
    """
    supervisor = create_supervisor_from_config(
        config=config,
        system_prompt=AUTHORING_SUPERVISOR_PROMPT,
        members=team_members,
        role="supervisor"
    )

    return create_supervisor_node(supervisor)


def should_continue_writing(state: DocWritingState) -> str:
    """Conditional edge function for writing team routing.

    Determines whether to continue with team members or finish.

    Args:
        state: Current document writing state

    Returns:
        Next node name or END
    """
    next_agent = state.get("next", "")

    if next_agent == "FINISH":
        return END
    else:
        return next_agent


def create_prelude_node(working_directory_path):
    """Create prelude node for document writing context.

    Provides file system context to agents about current working directory.

    Args:
        working_directory_path: Path to the working directory

    Returns:
        Prelude node function
    """
    def prelude(state: DocWritingState) -> Dict[str, Any]:
        """Update state with current working directory context."""
        from pathlib import Path

        working_dir = Path(working_directory_path)
        written_files = []

        if not working_dir.exists():
            working_dir.mkdir(parents=True, exist_ok=True)

        try:
            written_files = [
                str(f.relative_to(working_dir))
                for f in working_dir.rglob("*")
                if f.is_file()
            ]
        except Exception:
            written_files = []

        # Update current_files in state
        files_info = {
            "working_directory": str(working_dir),
            "files": written_files,
            "file_count": len(written_files)
        }

        return {"current_files": files_info}

    return prelude


def create_document_writing_graph(
    config: Configuration,
    file_tools: List[Any],
    working_directory_path: str,
    with_checkpointer: bool = True,
    **kwargs
) -> StateGraph:
    """Create document writing team graph using MessagesState patterns.

    Modernized multi-agent document writing workflow with:
    - MessagesState for better message handling
    - Configuration-driven agent creation
    - File management capabilities
    - Proper supervisor routing
    - Optional checkpointing

    Args:
        config: System configuration
        file_tools: List of file management tools
        working_directory_path: Path to working directory
        with_checkpointer: Whether to include checkpointing
        **kwargs: Additional graph configuration

    Returns:
        Compiled document writing team graph
    """
    # Team member names
    team_members = ["DocWriter", "NoteTaker", "CopyEditor"]

    # Create agent nodes
    agent_nodes = create_writing_agent_nodes(
        config=config,
        file_tools=file_tools,
        **kwargs
    )

    # Create supervisor node
    supervisor_node = create_writing_supervisor_node(
        config=config,
        team_members=team_members,
        **kwargs
    )

    # Create prelude node for file context
    prelude_node = create_prelude_node(working_directory_path)

    # Build the graph
    graph = StateGraph(DocWritingState)

    # Add prelude node for context
    graph.add_node("prelude", prelude_node)

    # Add agent nodes
    for name, node in agent_nodes.items():
        graph.add_node(name, node)

    # Add supervisor node
    graph.add_node("AuthoringSupervisor", supervisor_node)

    # Define the workflow edges
    # Start with prelude to set up file context
    graph.add_edge(START, "prelude")

    # Prelude leads to supervisor
    graph.add_edge("prelude", "AuthoringSupervisor")

    # Conditional edges from supervisor to agents or END
    graph.add_conditional_edges(
        "AuthoringSupervisor",
        should_continue_writing,
        {member: member for member in team_members} | {"FINISH": END}
    )

    # All agents return to supervisor
    for member in team_members:
        graph.add_edge(member, "AuthoringSupervisor")

    # Compile with optional checkpointing
    if with_checkpointer:
        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)
    else:
        compiled_graph = graph.compile()

    return compiled_graph


def create_writing_nodes_for_notebook(
    doc_writer,
    note_taker,
    copy_editor,
    authoring_supervisor,
    working_directory
):
    """Create writing nodes that match the original notebook implementation.

    Provides backward compatibility for existing notebook code.

    Args:
        doc_writer: Document writer agent
        note_taker: Note taker agent
        copy_editor: Copy editor agent
        authoring_supervisor: Authoring supervisor chain
        working_directory: Working directory path

    Returns:
        Tuple of node functions for notebook compatibility
    """
    # Create prelude function matching notebook pattern
    def prelude(state):
        from pathlib import Path

        working_dir = Path(working_directory)
        written_files = []

        if not working_dir.exists():
            working_dir.mkdir(parents=True, exist_ok=True)

        try:
            written_files = [
                str(f.relative_to(working_dir)) for f in working_dir.rglob("*")
            ]
        except:
            pass

        return {"current_files": f"[{', '.join(written_files)}]"}

    # Create agent node wrappers
    def doc_writing_node(state):
        prelude_result = prelude(state)
        updated_state = {**state, **prelude_result}
        result = doc_writer.invoke({"messages": updated_state["messages"]})
        return {"messages": [result["output"]]}

    def note_taking_node(state):
        prelude_result = prelude(state)
        updated_state = {**state, **prelude_result}
        result = note_taker.invoke({"messages": updated_state["messages"]})
        return {"messages": [result["output"]]}

    def copy_editing_node(state):
        prelude_result = prelude(state)
        updated_state = {**state, **prelude_result}
        result = copy_editor.invoke({"messages": updated_state["messages"]})
        return {"messages": [result["output"]]}

    def authoring_supervisor_node(state):
        result = authoring_supervisor.invoke(state)
        return {"next": result["next"]}

    return (
        doc_writing_node,
        note_taking_node,
        copy_editing_node,
        authoring_supervisor_node,
        prelude
    )


# Backward compatibility helpers
def get_writing_team_members() -> List[str]:
    """Get standard writing team member names."""
    return ["DocWriter", "NoteTaker", "CopyEditor"]


def get_writing_graph_info():
    """Get information about document writing team graph structure."""
    return {
        "team_members": get_writing_team_members(),
        "supervisor": "AuthoringSupervisor",
        "state_class": "DocWritingState",
        "routing": "Conditional edges based on supervisor decisions",
        "features": ["File management", "Working directory context", "Document lifecycle"]
    }