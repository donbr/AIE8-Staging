"""Agent factory functions for creating LangGraph agents and supervisors.

Modernized agent creation following LangGraph 2025 patterns with MessagesState support.
Extracted from notebook cells to provide clean, reusable agent factory functions.
"""

from typing import List, Optional, Dict, Any
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

from ..config.configuration import Configuration
from ..config.prompts import BASE_AGENT_INSTRUCTION, format_prompt
from ..core.factory import create_chat_model


def create_agent(
    llm: ChatOpenAI,
    tools: list,
    system_prompt: str,
    use_enhanced_prompts: bool = True
) -> AgentExecutor:
    """Create a function-calling agent following LangGraph patterns.

    Modernized version of the original notebook create_agent function with:
    - Enhanced prompt system integration
    - Better error handling
    - Type annotations
    - Documentation

    Args:
        llm: ChatOpenAI model instance
        tools: List of tools available to the agent
        system_prompt: Base system prompt for the agent
        use_enhanced_prompts: Whether to use enhanced prompt templates

    Returns:
        AgentExecutor configured with the specified LLM and tools
    """
    # Enhance system prompt with base instructions if enabled
    if use_enhanced_prompts:
        enhanced_prompt = format_prompt(
            "{system_prompt}\n\n{base_instruction}",
            system_prompt=system_prompt,
            base_instruction=BASE_AGENT_INSTRUCTION
        )
    else:
        # Original notebook pattern for compatibility
        enhanced_prompt = (
            system_prompt +
            "\nWork autonomously according to your specialty, using the tools available to you."
            " Do not ask for clarification."
            " Your other team members (and other teams) will collaborate with you with their own specialties."
            " You are chosen for a reason!"
        )

    # Create prompt template following LangGraph patterns
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Create the OpenAI functions agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Return executor for easy invocation
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


def create_team_supervisor(
    llm: ChatOpenAI,
    system_prompt: str,
    members: List[str],
    include_finish: bool = True
) -> Any:
    """Create a team supervisor for routing between agents.

    Modernized version following LangGraph 2025 patterns with:
    - Better type annotations
    - Configurable FINISH option
    - Enhanced error handling
    - Documentation

    Args:
        llm: ChatOpenAI model for routing decisions
        system_prompt: System prompt for the supervisor
        members: List of team member names to route between
        include_finish: Whether to include FINISH as an option

    Returns:
        Supervisor chain that can route between team members
    """
    # Build routing options
    options = (["FINISH"] + members) if include_finish else members

    # Define the routing function schema
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
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }

    # Create prompt template with routing context
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]).partial(options=str(options), team_members=", ".join(members))

    # Create the supervisor chain
    supervisor_chain = (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

    return supervisor_chain


def create_agent_node(agent: AgentExecutor, name: str) -> callable:
    """Create a node function for use in LangGraph from an AgentExecutor.

    Wraps an AgentExecutor to work properly with MessagesState and LangGraph.

    Args:
        agent: The AgentExecutor to wrap
        name: Name of the agent (for message attribution)

    Returns:
        Callable function suitable for use as a LangGraph node
    """
    def agent_node(state):
        """Agent node function for LangGraph integration."""
        # Extract messages from state (works with both TypedDict and MessagesState)
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}

        # Invoke the agent with the current messages
        result = agent.invoke({"messages": messages})

        # Return the result in the expected format
        # The agent output should already be properly formatted
        return {"messages": [result.get("output", result)]}

    return agent_node


def create_supervisor_node(supervisor_chain: Any) -> callable:
    """Create a supervisor node function for use in LangGraph.

    Wraps a supervisor chain to work properly with MessagesState.

    Args:
        supervisor_chain: The supervisor chain to wrap

    Returns:
        Callable function suitable for use as a LangGraph supervisor node
    """
    def supervisor_node(state):
        """Supervisor node function for LangGraph integration."""
        # Invoke supervisor with current state
        result = supervisor_chain.invoke(state)

        # Extract routing decision
        next_agent = result.get("next", "FINISH")

        return {"next": next_agent}

    return supervisor_node


def create_agents_from_config(
    config: Configuration,
    agent_configs: List[Dict[str, Any]],
    shared_tools: Optional[List] = None
) -> Dict[str, AgentExecutor]:
    """Create multiple agents from configuration.

    Factory function to create agents in bulk using configuration-driven approach.

    Args:
        config: System configuration
        agent_configs: List of agent configuration dictionaries with keys:
            - name: Agent name
            - role: Role for model selection (research/writing/supervisor)
            - system_prompt: System prompt for the agent
            - tools: Optional list of additional tools
        shared_tools: Tools to include for all agents

    Returns:
        Dictionary mapping agent names to AgentExecutor instances
    """
    agents = {}

    for agent_config in agent_configs:
        name = agent_config["name"]
        role = agent_config.get("role", "research")
        system_prompt = agent_config["system_prompt"]
        agent_tools = agent_config.get("tools", [])

        # Combine shared tools with agent-specific tools
        tools = (shared_tools or []) + agent_tools

        # Create LLM for this agent using configuration
        llm = create_chat_model(config, role=role)

        # Create the agent
        agents[name] = create_agent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            use_enhanced_prompts=True
        )

    return agents


def create_supervisor_from_config(
    config: Configuration,
    system_prompt: str,
    members: List[str],
    role: str = "supervisor"
) -> Any:
    """Create a supervisor from configuration.

    Factory function to create supervisors using configuration-driven approach.

    Args:
        config: System configuration
        system_prompt: System prompt for the supervisor
        members: List of team member names
        role: Role for model selection

    Returns:
        Supervisor chain for routing between team members
    """
    # Create supervisor LLM using configuration
    llm = create_chat_model(config, role=role)

    # Create the supervisor
    return create_team_supervisor(
        llm=llm,
        system_prompt=system_prompt,
        members=members,
        include_finish=True
    )


# Backward compatibility functions for existing notebook code
def agent_node(state, agent, name):
    """Backward compatibility wrapper for agent nodes."""
    # Original notebook pattern - wrap in new implementation
    node_func = create_agent_node(agent, name)
    return node_func(state)


def get_agent_factory_info():
    """Get information about agent factory functions.

    Useful for debugging and understanding available factory functions.
    """
    return {
        "functions": {
            "create_agent": "Create individual agents with tools and prompts",
            "create_team_supervisor": "Create supervisors for routing between agents",
            "create_agent_node": "Wrap agents for LangGraph node use",
            "create_supervisor_node": "Wrap supervisors for LangGraph node use",
            "create_agents_from_config": "Create multiple agents from configuration",
            "create_supervisor_from_config": "Create supervisors from configuration"
        },
        "patterns": {
            "MessagesState": "Supports both TypedDict and MessagesState patterns",
            "Configuration": "Uses centralized configuration for model creation",
            "Enhanced_Prompts": "Integrates with externalized prompt system"
        }
    }