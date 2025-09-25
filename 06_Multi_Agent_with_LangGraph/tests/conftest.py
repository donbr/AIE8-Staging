"""Pytest configuration and fixtures for MCP server testing.

Provides shared fixtures and configuration for all MCP server tests.
"""

import pytest
import os
import sys
import asyncio
import tempfile
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    return {
        "TAVILY_API_KEY": "test-tavily-key-123",
        "OPENAI_API_KEY": "test-openai-key-456",
        "QDRANT_URL": "http://localhost:6333",
        "EMBEDDING_DIMENSION": "1536"
    }


@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for API calls."""
    client = AsyncMock()
    response = Mock()
    response.json.return_value = {"test": "data"}
    response.raise_for_status.return_value = None
    client.post.return_value = response
    client.get.return_value = response
    return client


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for vector store testing."""
    client = Mock()

    # Mock collections
    mock_collections = Mock()
    mock_collections.collections = [
        Mock(name="test_collection"),
        Mock(name="research_documents")
    ]
    client.get_collections.return_value = mock_collections

    # Mock collection info
    mock_collection_info = Mock()
    mock_collection_info.vectors_count = 100
    mock_collection_info.points_count = 80
    mock_collection_info.status = "green"
    client.get_collection.return_value = mock_collection_info

    # Mock search results
    mock_search_result = Mock()
    mock_search_result.id = "test-doc-1"
    mock_search_result.score = 0.95
    mock_search_result.payload = {
        "content": "Test document content",
        "metadata": {"source": "test.pdf"}
    }
    client.search.return_value = [mock_search_result]

    # Mock upsert operation
    client.upsert.return_value = True

    # Mock delete operation
    client.delete.return_value = True

    # Mock create collection
    client.create_collection.return_value = True

    return client


@pytest.fixture
def mock_arxiv_client():
    """Mock ArXiv client for paper search testing."""
    from datetime import datetime

    client = Mock()

    # Mock search result
    mock_result = Mock()
    mock_result.entry_id = "https://arxiv.org/abs/2301.00001"
    mock_result.title = "Test Paper: Machine Learning Approach"
    mock_result.authors = [Mock(name="Test Author 1"), Mock(name="Test Author 2")]
    mock_result.summary = "This is a test abstract for machine learning research."
    mock_result.categories = ["cs.AI", "cs.LG"]
    mock_result.published = datetime(2023, 1, 1)
    mock_result.updated = datetime(2023, 1, 2)
    mock_result.pdf_url = "https://arxiv.org/pdf/2301.00001.pdf"

    # Mock client results method
    def mock_results(search):
        return [mock_result]

    client.results = mock_results

    return client


@pytest.fixture
def sample_papers():
    """Sample papers for testing."""
    from datetime import datetime
    from mcp_servers.arxiv_researcher import ArXivPaper

    return [
        ArXivPaper(
            arxiv_id="2301.00001",
            title="Machine Learning for Natural Language Processing",
            authors=["Alice Smith", "Bob Johnson"],
            abstract="This paper presents novel approaches to NLP using machine learning.",
            categories=["cs.CL", "cs.LG"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 1),
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            arxiv_url="https://arxiv.org/abs/2301.00001"
        ),
        ArXivPaper(
            arxiv_id="2301.00002",
            title="Deep Learning Architectures for Computer Vision",
            authors=["Charlie Brown", "Diana Wilson"],
            abstract="We explore deep learning methods for computer vision tasks.",
            categories=["cs.CV", "cs.LG"],
            published=datetime(2023, 1, 2),
            updated=datetime(2023, 1, 2),
            pdf_url="https://arxiv.org/pdf/2301.00002.pdf",
            arxiv_url="https://arxiv.org/abs/2301.00002"
        )
    ]


@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    from mcp_servers.tavily_search import TavilySearchResult

    return [
        TavilySearchResult(
            title="Machine Learning Fundamentals",
            url="https://example.com/ml-fundamentals",
            content="A comprehensive guide to machine learning fundamentals covering algorithms, data preprocessing, and model evaluation.",
            score=0.95,
            published_date="2023-12-01"
        ),
        TavilySearchResult(
            title="Deep Learning with Python",
            url="https://example.com/deep-learning-python",
            content="Learn deep learning using Python and popular frameworks like TensorFlow and PyTorch.",
            score=0.88,
            published_date="2023-11-15"
        )
    ]


@pytest.fixture
def sample_documents():
    """Sample documents for vector store testing."""
    from mcp_servers.vector_store import DocumentChunk
    from datetime import datetime

    return [
        DocumentChunk(
            id="doc-1",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            metadata={"source": "ml_basics.pdf", "page": 1},
            collection="research_documents"
        ),
        DocumentChunk(
            id="doc-2",
            content="Natural language processing deals with the interaction between computers and human language.",
            metadata={"source": "nlp_intro.pdf", "page": 1},
            collection="research_documents"
        )
    ]


# Test configuration
@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch, mock_env_vars):
    """Set up test environment with mocked environment variables."""
    for key, value in mock_env_vars.items():
        monkeypatch.setenv(key, value)


# Mark async tests
@pytest.fixture(autouse=True)
def setup_asyncio():
    """Set up asyncio for tests."""
    # Ensure asyncio loop is available
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


# Pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test (requires external services)"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test (no external dependencies)"
    )
    config.addinivalue_line(
        "markers",
        "async: mark test as async test"
    )