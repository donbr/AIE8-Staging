"""Graph modules for the Multi-Agent LangGraph system.

Modernized graph implementations using MessagesState patterns and LangGraph 2025 best practices.
All graphs are extracted from the original notebook and refactored for better modularity.
"""

from .simple_rag import (
    create_simple_rag_graph,
    create_rag_graph_from_config,
    create_rag_nodes_from_notebook
)

from .research_team import (
    create_research_team_graph,
    create_research_agent_nodes,
    create_research_supervisor_node,
    create_research_nodes_for_notebook,
    get_research_team_members
)

from .writing_team import (
    create_document_writing_graph,
    create_writing_agent_nodes,
    create_writing_supervisor_node,
    create_writing_nodes_for_notebook,
    get_writing_team_members
)

from .meta_supervisor import (
    create_meta_supervisor_graph,
    create_complete_multi_agent_system,
    create_meta_nodes_for_notebook,
    get_last_message,
    join_graph
)

from .transcription_workflow import (
    create_transcription_workflow_graph,
    create_batch_transcription_graph,
    create_simple_transcription_graph,
    transcribe_single_file,
    transcribe_batch_files,
    AudioTranscriptionState
)

__all__ = [
    # Simple RAG
    "create_simple_rag_graph",
    "create_rag_graph_from_config",
    "create_rag_nodes_from_notebook",

    # Research Team
    "create_research_team_graph",
    "create_research_agent_nodes",
    "create_research_supervisor_node",
    "create_research_nodes_for_notebook",
    "get_research_team_members",

    # Writing Team
    "create_document_writing_graph",
    "create_writing_agent_nodes",
    "create_writing_supervisor_node",
    "create_writing_nodes_for_notebook",
    "get_writing_team_members",

    # Meta Supervisor
    "create_meta_supervisor_graph",
    "create_complete_multi_agent_system",
    "create_meta_nodes_for_notebook",
    "get_last_message",
    "join_graph",

    # Audio Transcription Workflow
    "create_transcription_workflow_graph",
    "create_batch_transcription_graph",
    "create_simple_transcription_graph",
    "transcribe_single_file",
    "transcribe_batch_files",
    "AudioTranscriptionState",
]


def get_available_graphs():
    """Get information about all available graph types."""
    return {
        "simple_rag": {
            "description": "Basic RAG workflow with retrieve and generate nodes",
            "state_class": "SimpleRAGState",
            "factory_function": "create_simple_rag_graph"
        },
        "research_team": {
            "description": "Multi-agent research team with search and RAG agents",
            "state_class": "ResearchTeamState",
            "factory_function": "create_research_team_graph",
            "agents": ["Search", "HowPeopleUseAIRetriever"],
            "supervisor": "ResearchSupervisor"
        },
        "writing_team": {
            "description": "Multi-agent document writing team with specialized roles",
            "state_class": "DocWritingState",
            "factory_function": "create_document_writing_graph",
            "agents": ["DocWriter", "NoteTaker", "CopyEditor"],
            "supervisor": "AuthoringSupervisor"
        },
        "meta_supervisor": {
            "description": "Top-level coordination between research and writing teams",
            "state_class": "MetaSupervisorState",
            "factory_function": "create_meta_supervisor_graph",
            "teams": ["research_team", "writing_team"],
            "supervisor": "MetaSupervisor"
        },
        "complete_system": {
            "description": "Full hierarchical multi-agent system",
            "state_class": "MetaSupervisorState",
            "factory_function": "create_complete_multi_agent_system",
            "features": ["All teams", "Full coordination", "Production ready"]
        },
        "transcription_workflow": {
            "description": "Complete audio transcription workflow with LLM post-processing",
            "state_class": "AudioTranscriptionState",
            "factory_function": "create_transcription_workflow_graph",
            "features": ["Multi-backend transcription", "LLM cleaning", "Document generation"],
            "nodes": ["Preprocessor", "Transcriber", "Processor", "Generator"]
        },
        "batch_transcription": {
            "description": "Batch audio transcription for multiple files",
            "state_class": "AudioTranscriptionState",
            "factory_function": "create_batch_transcription_graph",
            "features": ["Batch processing", "Error handling", "Progress tracking"]
        },
        "simple_transcription": {
            "description": "Basic audio transcription without LLM processing",
            "state_class": "AudioTranscriptionState",
            "factory_function": "create_simple_transcription_graph",
            "features": ["Fast processing", "Basic transcription", "Minimal dependencies"]
        }
    }


def create_graph_from_config(
    graph_type: str,
    config,
    **kwargs
):
    """Factory function to create any graph type from configuration.

    Args:
        graph_type: Type of graph to create ('simple_rag', 'research_team', etc.)
        config: Configuration instance
        **kwargs: Graph-specific arguments

    Returns:
        Compiled graph of the specified type

    Raises:
        ValueError: If graph_type is not supported
    """
    available = get_available_graphs()

    if graph_type not in available:
        raise ValueError(
            f"Unsupported graph type: {graph_type}. "
            f"Available types: {list(available.keys())}"
        )

    if graph_type == "simple_rag":
        return create_rag_graph_from_config(config, **kwargs)
    elif graph_type == "research_team":
        return create_research_team_graph(config, **kwargs)
    elif graph_type == "writing_team":
        return create_document_writing_graph(config, **kwargs)
    elif graph_type == "meta_supervisor":
        return create_meta_supervisor_graph(config, **kwargs)
    elif graph_type == "complete_system":
        return create_complete_multi_agent_system(config, **kwargs)
    elif graph_type == "transcription_workflow":
        return create_transcription_workflow_graph(config, **kwargs)
    elif graph_type == "batch_transcription":
        return create_batch_transcription_graph(config, **kwargs)
    elif graph_type == "simple_transcription":
        return create_simple_transcription_graph(config, **kwargs)
    else:
        raise ValueError(f"Graph creation not implemented for: {graph_type}")


# Backward compatibility exports for notebook
def get_notebook_compatible_functions():
    """Get functions that provide notebook compatibility."""
    return {
        "rag": {
            "factory": create_rag_nodes_from_notebook,
            "description": "Create RAG nodes matching notebook pattern"
        },
        "research": {
            "factory": create_research_nodes_for_notebook,
            "description": "Create research team nodes matching notebook pattern"
        },
        "writing": {
            "factory": create_writing_nodes_for_notebook,
            "description": "Create writing team nodes matching notebook pattern"
        },
        "meta": {
            "factory": create_meta_nodes_for_notebook,
            "description": "Create meta-supervisor nodes matching notebook pattern"
        }
    }