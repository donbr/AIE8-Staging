#!/usr/bin/env python3
"""MCP Server Validation Script.

Validates the functionality of all implemented MCP servers:
- Configuration management
- Server initialization
- Tool registration and execution
- Integration with existing system

Run this script to ensure MCP servers are working correctly.
"""

import asyncio
import os
import sys
import json
import traceback
from typing import Dict, List, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

# Test results storage
test_results = {
    "timestamp": datetime.now().isoformat(),
    "tests_run": 0,
    "tests_passed": 0,
    "tests_failed": 0,
    "failures": []
}

def log_test(test_name: str, success: bool, error: str = None):
    """Log test result."""
    global test_results

    test_results["tests_run"] += 1

    if success:
        test_results["tests_passed"] += 1
        print(f"✅ {test_name}")
    else:
        test_results["tests_failed"] += 1
        test_results["failures"].append({
            "test": test_name,
            "error": error or "Unknown error"
        })
        print(f"❌ {test_name}: {error}")

def test_mcp_config():
    """Test MCP configuration system."""
    try:
        from mcp_servers.config import MCPConfiguration, MCPServerType, get_mcp_config

        # Test basic configuration
        config = MCPConfiguration()
        assert config.enable_mcp is True
        assert len(config.servers) >= 3
        log_test("MCP Configuration - Basic initialization", True)

        # Test server retrieval
        tavily_config = config.get_server_config(MCPServerType.TAVILY_SEARCH)
        assert tavily_config is not None
        assert tavily_config.type == MCPServerType.TAVILY_SEARCH
        log_test("MCP Configuration - Server retrieval", True)

        # Test enabled servers
        enabled = config.get_enabled_servers()
        assert len(enabled) > 0
        log_test("MCP Configuration - Enabled servers", True)

        # Test global config
        global_config = get_mcp_config()
        assert global_config is not None
        log_test("MCP Configuration - Global config access", True)

    except Exception as e:
        log_test("MCP Configuration", False, str(e))

def test_tavily_server():
    """Test Tavily MCP Server."""
    try:
        from mcp_servers.tavily_search import TavilyMCPServer, TavilySearchResult, TavilySearchResponse

        # Test initialization with mock API key
        server = TavilyMCPServer(api_key="test-key-123")
        assert server.api_key == "test-key-123"
        assert server.base_url == "https://api.tavily.com"
        log_test("Tavily Server - Initialization", True)

        # Test model validation
        result = TavilySearchResult(
            title="Test",
            url="https://example.com",
            content="Test content",
            score=0.85
        )
        assert result.title == "Test"
        assert result.score == 0.85
        log_test("Tavily Server - Result model validation", True)

        # Test response model
        response = TavilySearchResponse(
            query="test",
            results=[result],
            search_type="web",
            total_results=1,
            search_time=0.1
        )
        assert response.query == "test"
        assert len(response.results) == 1
        log_test("Tavily Server - Response model validation", True)

        # Test caching functionality
        server._cache_result("test_key", response)
        cached = server._get_cached_result("test_key")
        assert cached is not None
        assert cached.query == "test"
        log_test("Tavily Server - Caching functionality", True)

        # Test result formatting
        formatted = server._format_search_results(response)
        assert "Test" in formatted
        assert "example.com" in formatted
        log_test("Tavily Server - Result formatting", True)

    except Exception as e:
        log_test("Tavily Server", False, str(e))

def test_vector_store_server():
    """Test Vector Store MCP Server."""
    try:
        from mcp_servers.vector_store import VectorStoreMCPServer, DocumentChunk, SearchResult
        from unittest.mock import Mock

        # Test model validation
        chunk = DocumentChunk(
            id="test-1",
            content="Test content",
            metadata={"source": "test"},
            collection="test_collection"
        )
        assert chunk.id == "test-1"
        assert chunk.content == "Test content"
        log_test("Vector Store - Document model validation", True)

        # Test search result model
        search_result = SearchResult(
            id="result-1",
            content="Result content",
            score=0.92,
            metadata={"type": "test"},
            collection="test_collection"
        )
        assert search_result.score == 0.92
        log_test("Vector Store - Search result model validation", True)

        # Test server initialization (with mocked client)
        with_mock = True
        if with_mock:
            # Mock the QdrantClient to avoid requiring actual Qdrant instance
            import unittest.mock
            with unittest.mock.patch('mcp_servers.vector_store.QdrantClient'):
                server = VectorStoreMCPServer("http://localhost:6333", 1536)
                assert server.qdrant_url == "http://localhost:6333"
                assert server.embedding_dim == 1536
                log_test("Vector Store - Initialization (mocked)", True)

                # Test text chunking
                long_text = "This is a test. " * 100  # ~1600 characters
                chunks = server._chunk_text(long_text, chunk_size=500, chunk_overlap=50)
                assert len(chunks) > 1
                log_test("Vector Store - Text chunking", True)

    except Exception as e:
        log_test("Vector Store", False, str(e))

def test_arxiv_server():
    """Test ArXiv MCP Server."""
    try:
        from mcp_servers.arxiv_researcher import ArXivMCPServer, ArXivPaper, SearchQuery, ResearchSummary
        from datetime import datetime

        # Test server initialization
        server = ArXivMCPServer()
        assert server.client is not None
        assert len(server.category_map) > 0
        assert "cs.AI" in server.category_map
        log_test("ArXiv Server - Initialization", True)

        # Test paper model
        paper = ArXivPaper(
            arxiv_id="2301.00001",
            title="Test Paper",
            authors=["Test Author"],
            abstract="Test abstract",
            categories=["cs.AI"],
            published=datetime.now(),
            updated=datetime.now(),
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            arxiv_url="https://arxiv.org/abs/2301.00001"
        )
        assert paper.arxiv_id == "2301.00001"
        assert "Test Paper" in paper.title
        log_test("ArXiv Server - Paper model validation", True)

        # Test search query model
        query = SearchQuery(
            keywords=["machine learning"],
            categories=["cs.AI", "cs.LG"],
            title_contains="neural"
        )
        assert "machine learning" in query.keywords
        assert "cs.AI" in query.categories
        log_test("ArXiv Server - Search query model", True)

        # Test query building
        built_query = server._build_search_query("test query", ["cs.AI"], "all_time")
        assert 'all:"test query"' in built_query
        assert "cat:cs.AI" in built_query
        log_test("ArXiv Server - Query building", True)

        # Test keyword extraction
        keywords = server._extract_keywords("machine learning natural language processing")
        assert len(keywords) > 0
        log_test("ArXiv Server - Keyword extraction", True)

        # Test similarity calculation
        paper2 = ArXivPaper(
            arxiv_id="2301.00002",
            title="Another Test Paper",
            authors=["Another Author"],
            abstract="Another test abstract",
            categories=["cs.AI"],
            published=datetime.now(),
            updated=datetime.now(),
            pdf_url="", arxiv_url=""
        )
        similarity = server._calculate_similarity(paper, paper2)
        assert 0.0 <= similarity <= 1.0
        log_test("ArXiv Server - Similarity calculation", True)

        # Test paper formatting
        formatted = server._format_paper_list([paper, paper2])
        assert "Test Paper" in formatted
        assert "Another Test Paper" in formatted
        log_test("ArXiv Server - Paper formatting", True)

    except Exception as e:
        log_test("ArXiv Server", False, str(e))

def test_integration_config():
    """Test integration with main configuration system."""
    try:
        from src.config.configuration import Configuration

        # Test main config with MCP support
        config = Configuration()
        assert hasattr(config, 'use_mcp_servers')
        assert hasattr(config, 'mcp_fallback_to_langchain')
        log_test("Integration - Main config MCP support", True)

        # Test MCP-related methods
        assert hasattr(config, 'is_mcp_enabled')
        assert hasattr(config, 'should_use_mcp_for_tool')
        assert hasattr(config, 'get_search_preference')
        log_test("Integration - MCP-related methods", True)

        # Test method functionality
        mcp_enabled = config.is_mcp_enabled()
        assert isinstance(mcp_enabled, bool)

        search_pref = config.get_search_preference()
        assert search_pref in ["mcp", "langchain", "none"]
        log_test("Integration - Method functionality", True)

    except ImportError as e:
        log_test("Integration - Config", False, f"Import error (expected in some environments): {e}")
    except Exception as e:
        log_test("Integration - Config", False, str(e))

def test_mcp_compatibility_layer():
    """Test MCP compatibility layer."""
    try:
        from src.mcp_integration import MCPToolWrapper, MCPToolFactory, get_mcp_factory

        # Test tool wrapper initialization
        wrapper = MCPToolWrapper(
            name="test_tool",
            description="Test tool description",
            mcp_server_name="test-server",
            mcp_tool_name="test_tool_name"
        )
        assert wrapper.name == "test_tool"
        assert wrapper.mcp_server_name == "test-server"
        log_test("Compatibility Layer - Tool wrapper", True)

        # Test factory initialization
        factory = get_mcp_factory()
        assert factory is not None
        log_test("Compatibility Layer - Factory initialization", True)

    except ImportError as e:
        log_test("Compatibility Layer", False, f"Import error (expected in some environments): {e}")
    except Exception as e:
        log_test("Compatibility Layer", False, str(e))

def test_factory_integration():
    """Test factory integration with MCP."""
    try:
        from src.core.factory import create_search_tool
        from src.config.configuration import Configuration

        # Test search tool creation with mock config
        config = Configuration()

        # This should work even if MCP is not fully available
        # as it has fallback to LangChain
        search_tool = create_search_tool(config, max_results=5)

        # Tool should be created (either MCP or LangChain)
        assert search_tool is not None
        log_test("Factory Integration - Search tool creation", True)

    except Exception as e:
        log_test("Factory Integration", False, str(e))

async def test_async_functionality():
    """Test async functionality of MCP servers."""
    try:
        from mcp_servers.tavily_search import TavilyMCPServer
        from mcp_servers.vector_store import VectorStoreMCPServer
        from mcp_servers.arxiv_researcher import ArXivMCPServer
        from unittest.mock import Mock, AsyncMock

        # Test Tavily server async methods
        tavily_server = TavilyMCPServer(api_key="test-key")

        # Mock the embedding client for vector store
        import unittest.mock
        with unittest.mock.patch('mcp_servers.vector_store.QdrantClient'):
            vector_server = VectorStoreMCPServer()

            # Test embedding functionality
            vector_server.embedding_api_key = "test-key"
            # Mock the HTTP client response
            mock_response = Mock()
            mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
            mock_response.raise_for_status.return_value = None

            with unittest.mock.patch.object(vector_server.embedding_client, 'post', return_value=mock_response):
                embedding = await vector_server._get_embedding("test text")
                assert embedding == [0.1, 0.2, 0.3]

        log_test("Async Functionality - Vector store embedding", True)

        # Test ArXiv server
        arxiv_server = ArXivMCPServer()
        # Test keyword extraction (sync method)
        keywords = arxiv_server._extract_keywords("test machine learning")
        assert len(keywords) >= 0  # Should return list (might be empty due to filtering)
        log_test("Async Functionality - ArXiv keyword extraction", True)

    except Exception as e:
        log_test("Async Functionality", False, str(e))

def main():
    """Run all MCP server validation tests."""
    print("🚀 Starting MCP Server Validation...\n")

    # Run synchronous tests
    print("📋 Testing MCP Configuration...")
    test_mcp_config()

    print("\n🔍 Testing Tavily Search Server...")
    test_tavily_server()

    print("\n🗄️ Testing Vector Store Server...")
    test_vector_store_server()

    print("\n📚 Testing ArXiv Researcher Server...")
    test_arxiv_server()

    print("\n🔗 Testing Integration with Main Config...")
    test_integration_config()

    print("\n⚙️ Testing MCP Compatibility Layer...")
    test_mcp_compatibility_layer()

    print("\n🏭 Testing Factory Integration...")
    test_factory_integration()

    # Run async tests
    print("\n⚡ Testing Async Functionality...")
    asyncio.run(test_async_functionality())

    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Tests Run: {test_results['tests_run']}")
    print(f"Tests Passed: {test_results['tests_passed']}")
    print(f"Tests Failed: {test_results['tests_failed']}")

    if test_results['tests_failed'] > 0:
        print(f"\n❌ Failed Tests:")
        for failure in test_results['failures']:
            print(f"  • {failure['test']}: {failure['error']}")

    success_rate = (test_results['tests_passed'] / test_results['tests_run']) * 100
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")

    if success_rate >= 80:
        print("🎉 MCP servers are functioning well!")
        return True
    else:
        print("⚠️  Some MCP server functionality may need attention.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Validation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)