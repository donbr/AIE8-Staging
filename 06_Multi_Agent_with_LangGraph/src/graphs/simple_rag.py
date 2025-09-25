"""Simple RAG graph implementation using MessagesState.

Modernized version of the basic RAG workflow from the notebook,
refactored to use MessagesState and proper LangGraph 2025 patterns.
"""

from typing import Dict, Any, List

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document

from ..agents.state import SimpleRAGState
from ..config.configuration import Configuration
from ..core.factory import create_embedding_model


def create_retrieve_node(retriever):
    """Create a retrieve node function for the RAG graph.

    Args:
        retriever: Vector store retriever instance

    Returns:
        Callable retrieve function for graph node
    """
    def retrieve(state: SimpleRAGState) -> Dict[str, Any]:
        """Retrieve documents based on the question."""
        question = state.get("question", "")
        if not question:
            return {"context": []}

        try:
            retrieved_docs = retriever.invoke(question)
            return {"context": retrieved_docs}
        except Exception as e:
            # Handle retrieval errors gracefully
            return {"context": []}

    return retrieve


def create_generate_node(llm, prompt_template):
    """Create a generate node function for the RAG graph.

    Args:
        llm: Language model for generation
        prompt_template: Prompt template for generation

    Returns:
        Callable generate function for graph node
    """
    from langchain_core.output_parsers import StrOutputParser

    def generate(state: SimpleRAGState) -> Dict[str, Any]:
        """Generate response based on question and context."""
        question = state.get("question", "")
        context = state.get("context", [])

        if not question:
            return {"response": "No question provided."}

        try:
            # Create the generation chain
            generator_chain = prompt_template | llm | StrOutputParser()

            # Generate response
            response = generator_chain.invoke({
                "query": question,
                "context": context
            })

            return {"response": response}
        except Exception as e:
            return {"response": f"Error generating response: {str(e)}"}

    return generate


def create_simple_rag_graph(
    retriever,
    llm,
    prompt_template,
    with_checkpointer: bool = True
) -> StateGraph:
    """Create a simple RAG graph using MessagesState patterns.

    Modernized version of the notebook's basic RAG implementation with:
    - MessagesState for better state management
    - Proper error handling
    - Optional checkpointing
    - Clean node separation

    Args:
        retriever: Vector store retriever for document retrieval
        llm: Language model for generation
        prompt_template: Prompt template for generation
        with_checkpointer: Whether to include checkpointing

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create node functions
    retrieve = create_retrieve_node(retriever)
    generate = create_generate_node(llm, prompt_template)

    # Build the graph using MessagesState
    graph = StateGraph(SimpleRAGState)

    # Add nodes
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)

    # Define the sequence: retrieve -> generate
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")

    # Compile with optional checkpointing
    if with_checkpointer:
        memory = MemorySaver()
        compiled_graph = graph.compile(checkpointer=memory)
    else:
        compiled_graph = graph.compile()

    return compiled_graph


def create_rag_graph_from_config(
    config: Configuration,
    retriever,
    prompt_template,
    **kwargs
) -> StateGraph:
    """Create RAG graph using configuration-driven approach.

    Factory function that uses the configuration system to create
    the RAG graph with appropriate models and settings.

    Args:
        config: System configuration
        retriever: Vector store retriever
        prompt_template: Generation prompt template
        **kwargs: Additional arguments for graph creation

    Returns:
        Compiled RAG graph
    """
    from ..core.factory import create_chat_model

    # Create LLM from configuration
    llm = create_chat_model(config, role="research")

    # Create the graph
    return create_simple_rag_graph(
        retriever=retriever,
        llm=llm,
        prompt_template=prompt_template,
        **kwargs
    )


def create_rag_nodes_from_notebook(retriever, generator_llm, chat_prompt):
    """Create RAG nodes that match the original notebook implementation.

    Provides backward compatibility for existing notebook code.

    Args:
        retriever: Qdrant or other retriever instance
        generator_llm: Generator LLM instance
        chat_prompt: Chat prompt template

    Returns:
        Tuple of (retrieve_node, generate_node) functions
    """
    retrieve = create_retrieve_node(retriever)
    generate = create_generate_node(generator_llm, chat_prompt)

    return retrieve, generate


# Backward compatibility function names
def retrieve(state):
    """Backward compatibility retrieve function."""
    # This should be replaced with proper retriever injection
    # For now, maintain interface for notebook compatibility
    raise NotImplementedError(
        "Use create_retrieve_node() to create retrieve functions with proper retriever injection"
    )


def generate(state):
    """Backward compatibility generate function."""
    # This should be replaced with proper LLM injection
    # For now, maintain interface for notebook compatibility
    raise NotImplementedError(
        "Use create_generate_node() to create generate functions with proper LLM injection"
    )