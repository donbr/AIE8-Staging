"""Tests for tool organization modules."""

import unittest
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import shutil
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tools import (
    create_tavily_search_tool,
    create_search_tool_from_config,
    get_search_tool_info,
    create_file_management_tools,
    create_rag_tool,
    create_reference_tool,
    get_available_tools,
    create_tools_from_config
)
from config.configuration import Configuration, SearchAPI


class TestSearchTools(unittest.TestCase):
    """Test search tool creation and configuration."""

    def setUp(self):
        """Set up test configuration."""
        self.config = Configuration(search_api=SearchAPI.TAVILY)

    @patch('tools.search.TavilySearchResults')
    def test_create_tavily_search_tool(self, mock_tavily):
        """Test Tavily search tool creation."""
        # Mock TavilySearchResults
        mock_tool = Mock()
        mock_tavily.return_value = mock_tool

        # Create tool
        tool = create_tavily_search_tool(max_results=3)

        # Verify tool was created with correct parameters
        mock_tavily.assert_called_once_with(max_results=3)
        self.assertEqual(tool, mock_tool)

    @patch('tools.search.create_tavily_search_tool')
    def test_create_search_tool_from_config_tavily(self, mock_create_tavily):
        """Test search tool creation from Tavily configuration."""
        # Mock tool
        mock_tool = Mock()
        mock_create_tavily.return_value = mock_tool

        # Create tool from config
        tool = create_search_tool_from_config(self.config)

        # Verify Tavily tool was created
        mock_create_tavily.assert_called_once_with(max_results=5)
        self.assertEqual(tool, mock_tool)

    def test_create_search_tool_from_config_none(self):
        """Test search tool creation when disabled."""
        config = Configuration(search_api=SearchAPI.NONE)
        tool = create_search_tool_from_config(config)
        self.assertIsNone(tool)

    def test_get_search_tool_info(self):
        """Test search tool information retrieval."""
        # Test Tavily info
        tavily_info = get_search_tool_info(SearchAPI.TAVILY.value)
        self.assertEqual(tavily_info["name"], "Tavily Search")
        self.assertTrue(tavily_info["requires_api_key"])
        self.assertEqual(tavily_info["env_var"], "TAVILY_API_KEY")

        # Test None info
        none_info = get_search_tool_info(SearchAPI.NONE.value)
        self.assertEqual(none_info["name"], "No Search")
        self.assertFalse(none_info["requires_api_key"])

        # Test unknown
        unknown_info = get_search_tool_info("unknown")
        self.assertEqual(unknown_info["name"], "Unknown")


class TestFileTools(unittest.TestCase):
    """Test file management tool creation."""

    def setUp(self):
        """Set up test directory."""
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_create_file_management_tools(self):
        """Test file management tools creation."""
        tools = create_file_management_tools(self.test_dir)

        # Verify all expected tools are created
        self.assertEqual(len(tools), 4)
        tool_names = [tool.name for tool in tools]
        expected_names = ["create_outline", "write_document", "read_document", "edit_document"]
        for name in expected_names:
            self.assertIn(name, tool_names)

    def test_file_tools_functionality(self):
        """Test basic file tool functionality."""
        tools = create_file_management_tools(self.test_dir)
        tool_dict = {tool.name: tool for tool in tools}

        # Test write_document
        write_result = tool_dict["write_document"].invoke({
            "content": "Test content",
            "file_name": "test.txt"
        })
        self.assertIn("Document saved", write_result)

        # Verify file was created
        test_file = self.test_dir / "test.txt"
        self.assertTrue(test_file.exists())

        # Test read_document
        read_result = tool_dict["read_document"].invoke({
            "file_name": "test.txt"
        })
        self.assertEqual(read_result, "Test content")

        # Test create_outline
        outline_result = tool_dict["create_outline"].invoke({
            "points": ["Point 1", "Point 2", "Point 3"],
            "file_name": "outline.txt"
        })
        self.assertIn("Outline saved", outline_result)

        # Test edit_document
        edit_result = tool_dict["edit_document"].invoke({
            "file_name": "test.txt",
            "inserts": {1: "New first line\n"}
        })
        self.assertIn("Document edited", edit_result)

    def test_create_rag_tool(self):
        """Test RAG tool creation."""
        # Mock retriever
        mock_retriever = Mock()
        mock_doc = Mock()
        mock_doc.page_content = "Test document content"
        mock_retriever.invoke.return_value = [mock_doc]

        # Create RAG tool
        rag_tool = create_rag_tool(mock_retriever, "test_rag_tool")

        # Test tool invocation
        result = rag_tool.invoke({"query": "test query"})

        # Verify retriever was called and result formatted
        mock_retriever.invoke.assert_called_once_with("test query")
        self.assertIn("Document 1: Test document content", result)

    def test_create_rag_tool_no_results(self):
        """Test RAG tool with no results."""
        # Mock retriever with no results
        mock_retriever = Mock()
        mock_retriever.invoke.return_value = []

        # Create RAG tool
        rag_tool = create_rag_tool(mock_retriever)

        # Test tool invocation
        result = rag_tool.invoke({"query": "test query"})

        # Verify no results message
        self.assertEqual(result, "No relevant documents found.")

    def test_create_reference_tool(self):
        """Test reference tool creation and functionality."""
        # Create test files
        (self.test_dir / "test1.txt").write_text("This is test content with keyword")
        (self.test_dir / "test2.md").write_text("Another document with keyword here")

        # Create reference tool
        ref_tool = create_reference_tool(self.test_dir)

        # Test tool invocation
        result = ref_tool.invoke({"query": "keyword"})

        # Verify results contain references from both files
        self.assertIn("test1.txt", result)
        self.assertIn("test2.md", result)
        self.assertIn("keyword", result)

    def test_create_reference_tool_no_files(self):
        """Test reference tool with no files."""
        # Create reference tool in empty directory
        ref_tool = create_reference_tool(self.test_dir)

        # Test tool invocation
        result = ref_tool.invoke({"query": "keyword"})

        # Verify no files message
        self.assertIn("No previous responses found", result)


class TestToolIntegration(unittest.TestCase):
    """Test tool integration and configuration."""

    def setUp(self):
        """Set up test configuration and directory."""
        self.config = Configuration(search_api=SearchAPI.TAVILY)
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_available_tools(self):
        """Test available tools information."""
        tools_info = get_available_tools()

        # Verify structure
        self.assertIn("search_tools", tools_info)
        self.assertIn("file_tools", tools_info)
        self.assertIn("rag_tools", tools_info)

        # Verify search tools info
        search_info = tools_info["search_tools"]["tavily_search"]
        self.assertEqual(search_info["description"], "Web search using Tavily API")
        self.assertTrue(search_info["requires_api_key"])

        # Verify file tools info
        file_info = tools_info["file_tools"]["write_document"]
        self.assertEqual(file_info["description"], "Write content to a new document file")
        self.assertIn("content: str", file_info["inputs"])

    @patch('tools.search.create_search_tool_from_config')
    def test_create_tools_from_config(self, mock_search):
        """Test comprehensive tool creation from configuration."""
        # Mock search tool
        mock_search_tool = Mock()
        mock_search.return_value = mock_search_tool

        # Mock retriever
        mock_retriever = Mock()

        # Create tools from config
        tools = create_tools_from_config(
            config=self.config,
            working_directory=self.test_dir,
            retriever=mock_retriever
        )

        # Verify all tool types were created
        self.assertEqual(tools["search"], mock_search_tool)
        self.assertEqual(len(tools["file_tools"]), 4)
        self.assertIsNotNone(tools["rag_tool"])
        self.assertIsNotNone(tools["reference_tool"])

        # Verify search tool factory was called
        mock_search.assert_called_once_with(self.config)

    def test_create_tools_from_config_minimal(self):
        """Test tool creation with minimal configuration."""
        tools = create_tools_from_config(config=self.config)

        # Verify only search tool was created
        self.assertIsNotNone(tools["search"])
        self.assertEqual(tools["file_tools"], [])
        self.assertIsNone(tools["rag_tool"])
        self.assertIsNone(tools["reference_tool"])


if __name__ == "__main__":
    unittest.main()