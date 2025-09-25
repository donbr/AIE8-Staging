"""Tests for graph creation modules."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphs import (
    create_simple_rag_graph,
    create_research_team_graph,
    create_document_writing_graph,
    create_meta_supervisor_graph,
    create_complete_multi_agent_system,
    get_available_graphs
)
from config.configuration import Configuration, LLMProvider


class TestGraphModules(unittest.TestCase):
    """Test graph creation and composition."""

    def setUp(self):
        """Set up test configuration and mocks."""
        self.config = Configuration(
            llm_provider=LLMProvider.OPENAI,
            research_model="gpt-4o-mini",
            writing_model="gpt-4o-mini",
            supervisor_model="gpt-4o"
        )

        # Mock tools
        self.mock_search_tool = Mock()
        self.mock_rag_tool = Mock()
        self.mock_file_tools = [Mock(), Mock(), Mock()]

    def test_get_available_graphs(self):
        """Test that available graphs information is correctly structured."""
        graphs = get_available_graphs()

        # Verify expected graph types exist
        expected_graphs = ["simple_rag", "research_team", "writing_team", "meta_supervisor", "complete_system"]
        for graph_type in expected_graphs:
            self.assertIn(graph_type, graphs)

        # Verify structure of graph info
        for graph_type, info in graphs.items():
            self.assertIn("description", info)
            self.assertIn("state_class", info)
            self.assertIn("factory_function", info)

    @patch('graphs.simple_rag.StateGraph')
    @patch('graphs.simple_rag.create_rag_agent_nodes')
    def test_create_simple_rag_graph(self, mock_nodes, mock_graph_class):
        """Test simple RAG graph creation."""
        # Mock StateGraph
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        mock_graph.compile.return_value = Mock()

        # Mock nodes
        mock_nodes.return_value = {"retrieve": Mock(), "generate": Mock()}

        # Create graph
        retriever = Mock()
        graph = create_simple_rag_graph(self.config, retriever)

        # Verify graph was created and compiled
        mock_graph_class.assert_called_once()
        mock_graph.compile.assert_called_once()
        self.assertIsNotNone(graph)

    @patch('graphs.research_team.StateGraph')
    @patch('graphs.research_team.create_research_agent_nodes')
    @patch('graphs.research_team.create_research_supervisor_node')
    def test_create_research_team_graph(self, mock_supervisor, mock_agents, mock_graph_class):
        """Test research team graph creation."""
        # Mock StateGraph and nodes
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        mock_graph.compile.return_value = Mock()

        mock_agents.return_value = {"Search": Mock(), "HowPeopleUseAIRetriever": Mock()}
        mock_supervisor.return_value = Mock()

        # Create graph
        graph = create_research_team_graph(
            config=self.config,
            search_tool=self.mock_search_tool,
            rag_tool=self.mock_rag_tool
        )

        # Verify components were created
        mock_agents.assert_called_once()
        mock_supervisor.assert_called_once()
        mock_graph.compile.assert_called_once()
        self.assertIsNotNone(graph)

    @patch('graphs.writing_team.StateGraph')
    @patch('graphs.writing_team.create_writing_agent_nodes')
    @patch('graphs.writing_team.create_writing_supervisor_node')
    @patch('graphs.writing_team.create_prelude_node')
    def test_create_document_writing_graph(self, mock_prelude, mock_supervisor, mock_agents, mock_graph_class):
        """Test document writing team graph creation."""
        # Mock StateGraph and nodes
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        mock_graph.compile.return_value = Mock()

        mock_agents.return_value = {
            "DocWriter": Mock(),
            "NoteTaker": Mock(),
            "CopyEditor": Mock()
        }
        mock_supervisor.return_value = Mock()
        mock_prelude.return_value = Mock()

        # Create graph
        graph = create_document_writing_graph(
            config=self.config,
            file_tools=self.mock_file_tools,
            working_directory_path="/tmp/test"
        )

        # Verify components were created
        mock_agents.assert_called_once()
        mock_supervisor.assert_called_once()
        mock_prelude.assert_called_once_with("/tmp/test")
        mock_graph.compile.assert_called_once()
        self.assertIsNotNone(graph)

    @patch('graphs.meta_supervisor.StateGraph')
    @patch('graphs.meta_supervisor.create_meta_supervisor_node')
    def test_create_meta_supervisor_graph(self, mock_supervisor, mock_graph_class):
        """Test meta-supervisor graph creation."""
        # Mock StateGraph
        mock_graph = Mock()
        mock_graph_class.return_value = mock_graph
        mock_graph.compile.return_value = Mock()

        mock_supervisor.return_value = Mock()

        # Mock subgraphs
        mock_research_graph = Mock()
        mock_writing_graph = Mock()
        mock_research_graph.invoke.return_value = {"messages": ["research result"]}
        mock_writing_graph.invoke.return_value = {"messages": ["writing result"]}

        # Create graph
        graph = create_meta_supervisor_graph(
            config=self.config,
            research_graph=mock_research_graph,
            writing_graph=mock_writing_graph
        )

        # Verify components were created
        mock_supervisor.assert_called_once()
        mock_graph.compile.assert_called_once()
        self.assertIsNotNone(graph)

    @patch('graphs.meta_supervisor.create_research_team_graph')
    @patch('graphs.meta_supervisor.create_document_writing_graph')
    @patch('graphs.meta_supervisor.create_meta_supervisor_graph')
    def test_create_complete_multi_agent_system(self, mock_meta, mock_writing, mock_research):
        """Test complete multi-agent system creation."""
        # Mock individual graphs
        mock_research_graph = Mock()
        mock_writing_graph = Mock()
        mock_complete_graph = Mock()

        mock_research.return_value = mock_research_graph
        mock_writing.return_value = mock_writing_graph
        mock_meta.return_value = mock_complete_graph

        # Create complete system
        system = create_complete_multi_agent_system(
            config=self.config,
            search_tool=self.mock_search_tool,
            rag_tool=self.mock_rag_tool,
            file_tools=self.mock_file_tools,
            working_directory_path="/tmp/test"
        )

        # Verify all components were created
        mock_research.assert_called_once()
        mock_writing.assert_called_once()
        mock_meta.assert_called_once()
        self.assertIsNotNone(system)

    def test_graph_routing_functions(self):
        """Test conditional edge routing functions."""
        from graphs.research_team import should_continue_research
        from graphs.writing_team import should_continue_writing
        from graphs.meta_supervisor import should_continue_meta

        # Test research team routing
        research_state = {"next": "Search"}
        self.assertEqual(should_continue_research(research_state), "Search")

        research_state = {"next": "FINISH"}
        self.assertEqual(should_continue_research(research_state), "FINISH")

        # Test writing team routing
        writing_state = {"next": "DocWriter"}
        self.assertEqual(should_continue_writing(writing_state), "DocWriter")

        writing_state = {"next": "FINISH"}
        self.assertEqual(should_continue_writing(writing_state), "FINISH")

        # Test meta-supervisor routing
        meta_state = {"next": "research_team"}
        self.assertEqual(should_continue_meta(meta_state), "research_team")

        meta_state = {"next": "writing_team"}
        self.assertEqual(should_continue_meta(meta_state), "writing_team")

        meta_state = {"next": "FINISH"}
        self.assertEqual(should_continue_meta(meta_state), "FINISH")

    def test_helper_functions(self):
        """Test graph helper functions."""
        from graphs.meta_supervisor import get_last_message, join_graph

        # Test get_last_message
        state = {"messages": ["msg1", "msg2", "msg3"]}
        self.assertEqual(get_last_message(state), "msg3")

        empty_state = {"messages": []}
        self.assertEqual(get_last_message(empty_state), "")

        # Test join_graph
        response = {"messages": ["result1", "result2"]}
        joined = join_graph(response)
        self.assertEqual(joined, {"messages": ["result2"]})

        empty_response = {"messages": []}
        joined_empty = join_graph(empty_response)
        self.assertEqual(joined_empty, {"messages": []})


if __name__ == "__main__":
    unittest.main()