"""Tests for agent state classes and factory functions."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agents.state import SimpleRAGState, ResearchTeamState, DocWritingState, MetaSupervisorState
from agents.factory import (
    create_agent_from_config,
    create_supervisor_from_config,
    create_agents_from_config,
    create_agent_node,
    create_supervisor_node
)
from config.configuration import Configuration, LLMProvider


class TestStateClasses(unittest.TestCase):
    """Test MessagesState-based state classes."""

    def test_simple_rag_state_creation(self):
        """Test SimpleRAGState creation and default values."""
        state = SimpleRAGState()
        self.assertEqual(state.get("messages", []), [])
        self.assertIn("messages", state)

    def test_research_team_state_creation(self):
        """Test ResearchTeamState with team-specific fields."""
        state = ResearchTeamState()
        self.assertEqual(state.get("messages", []), [])
        self.assertEqual(state.get("team_members", []), [])
        self.assertEqual(state.get("next", ""), "")

    def test_doc_writing_state_creation(self):
        """Test DocWritingState with file tracking fields."""
        state = DocWritingState()
        self.assertEqual(state.get("messages", []), [])
        self.assertEqual(state.get("current_files", {}), {})
        self.assertEqual(state.get("next", ""), "")

    def test_meta_supervisor_state_creation(self):
        """Test MetaSupervisorState creation."""
        state = MetaSupervisorState()
        self.assertEqual(state.get("messages", []), [])
        self.assertEqual(state.get("next", ""), "")

    def test_state_message_handling(self):
        """Test that states can handle message updates."""
        state = ResearchTeamState()
        state.update({"messages": ["test message"]})
        self.assertEqual(state["messages"], ["test message"])

    def test_state_team_members_handling(self):
        """Test that research state handles team members."""
        state = ResearchTeamState()
        members = ["SearchAgent", "RAGAgent"]
        state.update({"team_members": members})
        self.assertEqual(state["team_members"], members)


class TestAgentFactory(unittest.TestCase):
    """Test agent factory functions."""

    def setUp(self):
        """Set up test configuration."""
        self.config = Configuration(
            llm_provider=LLMProvider.OPENAI,
            research_model="gpt-4o-mini",
            writing_model="gpt-4o-mini",
            supervisor_model="gpt-4o"
        )

    @patch('agents.factory.create_chat_openai_from_config')
    def test_create_agent_from_config(self, mock_llm_factory):
        """Test agent creation from configuration."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm_factory.return_value = mock_llm

        # Create agent
        agent = create_agent_from_config(
            config=self.config,
            system_prompt="Test prompt",
            tools=[],
            role="research"
        )

        # Verify LLM factory was called with correct parameters
        mock_llm_factory.assert_called_once_with(self.config, "research")
        self.assertIsNotNone(agent)

    @patch('agents.factory.create_chat_openai_from_config')
    def test_create_supervisor_from_config(self, mock_llm_factory):
        """Test supervisor creation with routing function."""
        # Mock LLM
        mock_llm = Mock()
        mock_llm_factory.return_value = mock_llm

        # Create supervisor
        supervisor = create_supervisor_from_config(
            config=self.config,
            system_prompt="Supervisor prompt",
            members=["Agent1", "Agent2"],
            role="supervisor"
        )

        # Verify LLM factory was called
        mock_llm_factory.assert_called_once_with(self.config, "supervisor")
        self.assertIsNotNone(supervisor)

    @patch('agents.factory.create_agent_from_config')
    def test_create_agents_from_config(self, mock_agent_factory):
        """Test bulk agent creation from configuration."""
        # Mock agent creation
        mock_agent = Mock()
        mock_agent_factory.return_value = mock_agent

        # Define agent configurations
        agent_configs = [
            {
                "name": "TestAgent1",
                "role": "research",
                "system_prompt": "Test prompt 1",
                "tools": []
            },
            {
                "name": "TestAgent2",
                "role": "writing",
                "system_prompt": "Test prompt 2",
                "tools": []
            }
        ]

        # Create agents
        agents = create_agents_from_config(
            config=self.config,
            agent_configs=agent_configs
        )

        # Verify correct number of agents created
        self.assertEqual(len(agents), 2)
        self.assertIn("TestAgent1", agents)
        self.assertIn("TestAgent2", agents)

        # Verify agent factory was called correctly
        self.assertEqual(mock_agent_factory.call_count, 2)

    def test_create_agent_node(self):
        """Test agent node wrapper creation."""
        # Mock agent
        mock_agent = Mock()
        mock_agent.invoke.return_value = {"output": "test response"}

        # Create node
        node = create_agent_node(mock_agent, "TestAgent")

        # Test node invocation
        test_state = {"messages": ["test message"]}
        result = node(test_state)

        # Verify agent was called and result formatted correctly
        mock_agent.invoke.assert_called_once_with(test_state)
        self.assertEqual(result["messages"], ["test response"])

    def test_create_supervisor_node(self):
        """Test supervisor node wrapper creation."""
        # Mock supervisor
        mock_supervisor = Mock()
        mock_supervisor.invoke.return_value = {"next": "Agent1"}

        # Create node
        node = create_supervisor_node(mock_supervisor)

        # Test node invocation
        test_state = {"messages": ["test message"]}
        result = node(test_state)

        # Verify supervisor was called and result formatted correctly
        mock_supervisor.invoke.assert_called_once_with(test_state)
        self.assertEqual(result["next"], "Agent1")

    def test_agent_node_error_handling(self):
        """Test agent node handles errors gracefully."""
        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.invoke.side_effect = Exception("Test error")

        # Create node
        node = create_agent_node(mock_agent, "TestAgent")

        # Test node handles error
        test_state = {"messages": ["test message"]}
        with self.assertRaises(Exception):
            node(test_state)


if __name__ == "__main__":
    unittest.main()