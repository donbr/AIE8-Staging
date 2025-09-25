"""Comprehensive tests for MCP servers.

Tests the functionality of all implemented MCP servers:
- Tavily Search Server
- Vector Store Server
- ArXiv Researcher Server
- MCP Configuration and Management
"""

import asyncio
import json
import os
import tempfile
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

# Test imports for MCP servers
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mcp_servers.config import MCPConfiguration, MCPServerConfig, MCPServerType
from mcp_servers.manager import MCPServerManager
from mcp_servers.tavily_search import TavilyMCPServer, TavilySearchResult, TavilySearchResponse
from mcp_servers.vector_store import VectorStoreMCPServer, DocumentChunk, SearchResult
from mcp_servers.arxiv_researcher import ArXivMCPServer, ArXivPaper
from mcp_servers.audio_transcription import (
    AudioTranscriptionMCPServer,
    TranscriptionBackend,
    TranscriptionResult,
    TranscriptionSegment
)


class TestMCPConfiguration:
    """Test MCP configuration management."""

    def test_default_configuration(self):
        """Test default MCP configuration."""
        config = MCPConfiguration()

        assert config.enable_mcp is True
        assert config.fallback_to_langchain is True
        assert config.mcp_timeout_seconds == 30
        assert len(config.servers) >= 3  # Should have at least our core servers

    def test_server_config_creation(self):
        """Test individual server configuration."""
        server_config = MCPServerConfig(
            name="test-server",
            type=MCPServerType.TAVILY_SEARCH,
            api_key_env_var="TEST_API_KEY"
        )

        assert server_config.name == "test-server"
        assert server_config.type == MCPServerType.TAVILY_SEARCH
        assert server_config.enabled is True
        assert server_config.api_key_env_var == "TEST_API_KEY"

    def test_get_server_config(self):
        """Test getting specific server configuration."""
        config = MCPConfiguration()

        tavily_config = config.get_server_config(MCPServerType.TAVILY_SEARCH)
        assert tavily_config is not None
        assert tavily_config.type == MCPServerType.TAVILY_SEARCH

    def test_get_enabled_servers(self):
        """Test getting only enabled servers."""
        config = MCPConfiguration()
        enabled_servers = config.get_enabled_servers()

        assert len(enabled_servers) > 0
        for server in enabled_servers:
            assert server.enabled is True


class TestTavilyMCPServer:
    """Test Tavily Search MCP Server functionality."""

    @pytest.fixture
    def mock_api_key(self):
        """Mock API key for testing."""
        return "test-tavily-key-123"

    @pytest.fixture
    def tavily_server(self, mock_api_key):
        """Create Tavily server instance for testing."""
        with patch.dict(os.environ, {'TAVILY_API_KEY': mock_api_key}):
            return TavilyMCPServer(api_key=mock_api_key)

    def test_server_initialization(self, mock_api_key):
        """Test Tavily server initializes correctly."""
        server = TavilyMCPServer(api_key=mock_api_key)

        assert server.api_key == mock_api_key
        assert server.base_url == "https://api.tavily.com"
        assert isinstance(server.cache, dict)
        assert server.cache_ttl == timedelta(minutes=15)

    def test_server_initialization_no_api_key(self):
        """Test server raises error without API key."""
        with pytest.raises(ValueError, match="TAVILY_API_KEY environment variable is required"):
            TavilyMCPServer(api_key=None)

    @pytest.mark.asyncio
    async def test_web_search_cached_result(self, tavily_server):
        """Test web search returns cached result."""
        # Setup cache with mock result
        mock_response = TavilySearchResponse(
            query="test query",
            results=[
                TavilySearchResult(
                    title="Test Result",
                    url="https://example.com",
                    content="Test content",
                    score=0.95
                )
            ],
            search_type="web",
            total_results=1,
            search_time=0.1
        )

        cache_key = "web:test query:5:basic"
        tavily_server.cache[cache_key] = (mock_response, datetime.now())

        result = await tavily_server._web_search("test query")

        assert len(result) == 1
        assert "Cached Results" in result[0].text
        assert "Test Result" in result[0].text

    @pytest.mark.asyncio
    async def test_web_search_api_call(self, tavily_server):
        """Test web search makes API call when not cached."""
        # Mock HTTP response
        mock_response_data = {
            "results": [
                {
                    "title": "API Result",
                    "url": "https://api-example.com",
                    "content": "API content",
                    "score": 0.85,
                    "published_date": "2024-01-01"
                }
            ],
            "response_time": 0.2
        }

        with patch.object(tavily_server.client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            result = await tavily_server._web_search("api test query")

            assert len(result) == 1
            assert "API Result" in result[0].text
            assert "api-example.com" in result[0].text

            # Verify API was called with correct parameters
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://api.tavily.com/search"

    def test_cache_functionality(self, tavily_server):
        """Test caching mechanism."""
        # Test cache hit
        mock_response = TavilySearchResponse(
            query="cache test",
            results=[],
            search_type="web",
            total_results=0,
            search_time=0.0
        )

        cache_key = "test_key"
        tavily_server._cache_result(cache_key, mock_response)

        cached_result = tavily_server._get_cached_result(cache_key)
        assert cached_result is not None
        assert cached_result.query == "cache test"

        # Test cache miss
        missing_result = tavily_server._get_cached_result("missing_key")
        assert missing_result is None

    def test_result_formatting(self, tavily_server):
        """Test search result formatting."""
        response = TavilySearchResponse(
            query="format test",
            results=[
                TavilySearchResult(
                    title="Format Test Result",
                    url="https://format.example.com",
                    content="This is test content for formatting",
                    score=0.92,
                    published_date="2024-01-01"
                )
            ],
            search_type="web",
            total_results=1,
            search_time=0.15
        )

        formatted = tavily_server._format_search_results(response)

        assert "🔍 **Search Results for: format test**" in formatted
        assert "📊 Found 1 results in 0.15s" in formatted
        assert "Format Test Result" in formatted
        assert "format.example.com" in formatted
        assert "Score: 0.92" in formatted


class TestVectorStoreMCPServer:
    """Test Vector Store MCP Server functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client for testing."""
        return Mock()

    @pytest.fixture
    def vector_server(self, mock_qdrant_client):
        """Create Vector Store server instance for testing."""
        with patch('mcp_servers.vector_store.QdrantClient', return_value=mock_qdrant_client):
            server = VectorStoreMCPServer("http://localhost:6333", 1536)
            server.client = mock_qdrant_client
            return server

    def test_server_initialization(self, vector_server):
        """Test Vector Store server initializes correctly."""
        assert vector_server.qdrant_url == "http://localhost:6333"
        assert vector_server.embedding_dim == 1536
        assert vector_server.embedding_model == "text-embedding-3-small"
        assert len(vector_server.default_collections) >= 4

    def test_document_chunk_model(self):
        """Test DocumentChunk model validation."""
        chunk = DocumentChunk(
            id="test-chunk-1",
            content="This is test content for the chunk",
            metadata={"source": "test.pdf", "page": 1},
            collection="test_collection"
        )

        assert chunk.id == "test-chunk-1"
        assert "test content" in chunk.content
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.collection == "test_collection"

    def test_text_chunking(self, vector_server):
        """Test text chunking functionality."""
        long_text = "This is a long piece of text. " * 50  # ~1400 characters

        chunks = vector_server._chunk_text(long_text, chunk_size=100, chunk_overlap=20)

        assert len(chunks) > 1
        assert len(chunks[0]) <= 100

        # Test overlap
        if len(chunks) > 1:
            # There should be some overlap between consecutive chunks
            assert chunks[0][-20:] in chunks[1] or chunks[1][:20] in chunks[0]

    def test_short_text_chunking(self, vector_server):
        """Test chunking of short text."""
        short_text = "This is a short text."

        chunks = vector_server._chunk_text(short_text, chunk_size=100, chunk_overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    @pytest.mark.asyncio
    async def test_get_embedding_cached(self, vector_server):
        """Test embedding retrieval with caching."""
        # Setup cache
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        vector_server.embedding_cache["test text"] = test_embedding

        embedding = await vector_server._get_embedding("test text")

        assert embedding == test_embedding

    @pytest.mark.asyncio
    async def test_get_embedding_api_call(self, vector_server):
        """Test embedding API call."""
        mock_response_data = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }

        with patch.object(vector_server.embedding_client, 'post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = mock_response_data
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            vector_server.embedding_api_key = "test-key"

            embedding = await vector_server._get_embedding("new text")

            assert embedding == [0.1, 0.2, 0.3]
            assert "new text" in vector_server.embedding_cache

    @pytest.mark.asyncio
    async def test_ensure_collection_exists(self, vector_server):
        """Test collection existence checking."""
        # Mock collections response
        mock_collections = Mock()
        mock_collections.collections = [Mock(name="existing_collection")]
        vector_server.client.get_collections.return_value = mock_collections

        # Test existing collection
        await vector_server._ensure_collection_exists("existing_collection")
        vector_server.client.create_collection.assert_not_called()

        # Test new collection
        await vector_server._ensure_collection_exists("new_collection")
        # Should have been called when trying to create the new collection

    @pytest.mark.asyncio
    async def test_list_collections(self, vector_server):
        """Test listing collections."""
        # Mock collections
        mock_collection_info = Mock()
        mock_collection_info.vectors_count = 100
        mock_collection_info.points_count = 80
        mock_collection_info.status = "green"

        mock_collections = Mock()
        mock_collections.collections = [Mock(name="test_collection")]

        vector_server.client.get_collections.return_value = mock_collections
        vector_server.client.get_collection.return_value = mock_collection_info

        result = await vector_server._list_collections()

        assert len(result) == 1
        assert "test_collection" in result[0].text

    @pytest.mark.asyncio
    async def test_health_status(self, vector_server):
        """Test health status retrieval."""
        # Mock successful health check
        mock_collections = Mock()
        mock_collections.collections = []
        vector_server.client.get_collections.return_value = mock_collections

        health_json = await vector_server._get_health_status()
        health_data = json.loads(health_json)

        assert health_data["status"] == "healthy"
        assert "qdrant_url" in health_data
        assert "timestamp" in health_data


class TestArXivMCPServer:
    """Test ArXiv Researcher MCP Server functionality."""

    @pytest.fixture
    def arxiv_server(self):
        """Create ArXiv server instance for testing."""
        return ArXivMCPServer()

    def test_server_initialization(self, arxiv_server):
        """Test ArXiv server initializes correctly."""
        assert arxiv_server.client is not None
        assert isinstance(arxiv_server.category_map, dict)
        assert "cs.AI" in arxiv_server.category_map
        assert isinstance(arxiv_server.paper_cache, dict)

    def test_arxiv_paper_model(self):
        """Test ArXivPaper model validation."""
        paper = ArXivPaper(
            arxiv_id="2301.00001",
            title="Test Paper Title",
            authors=["John Doe", "Jane Smith"],
            abstract="This is a test abstract for the paper.",
            categories=["cs.AI", "cs.LG"],
            published=datetime(2023, 1, 1),
            updated=datetime(2023, 1, 2),
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            arxiv_url="https://arxiv.org/abs/2301.00001"
        )

        assert paper.arxiv_id == "2301.00001"
        assert "Test Paper" in paper.title
        assert len(paper.authors) == 2
        assert "cs.AI" in paper.categories

    def test_build_search_query(self, arxiv_server):
        """Test search query building."""
        query = arxiv_server._build_search_query(
            "machine learning",
            ["cs.AI", "cs.LG"],
            "all_time"
        )

        assert 'all:"machine learning"' in query
        assert "cat:cs.AI OR cat:cs.LG" in query

    def test_keyword_extraction(self, arxiv_server):
        """Test keyword extraction from text."""
        text = "This paper presents a novel approach to natural language processing using deep learning techniques."

        keywords = arxiv_server._extract_keywords(text)

        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Should extract meaningful words (4+ chars)
        meaningful_words = [w for w in keywords if len(w) >= 4]
        assert len(meaningful_words) > 0

    def test_keyword_grouping(self, arxiv_server):
        """Test keyword grouping for search queries."""
        keywords = ["machine", "learning", "neural", "network", "algorithm", "optimization"]

        groups = arxiv_server._group_keywords(keywords, group_size=3)

        assert len(groups) == 2  # 6 keywords / 3 per group
        assert len(groups[0]) == 3
        assert len(groups[1]) == 3

    def test_similarity_calculation(self, arxiv_server):
        """Test paper similarity calculation."""
        paper1 = ArXivPaper(
            arxiv_id="1",
            title="Machine Learning for Natural Language Processing",
            authors=["Author A"],
            abstract="This paper discusses machine learning techniques for NLP applications.",
            categories=["cs.CL", "cs.LG"],
            published=datetime.now(),
            updated=datetime.now(),
            pdf_url="", arxiv_url=""
        )

        paper2 = ArXivPaper(
            arxiv_id="2",
            title="Deep Learning in Natural Language Understanding",
            authors=["Author B"],
            abstract="We explore deep learning methods for language understanding tasks.",
            categories=["cs.CL", "cs.AI"],
            published=datetime.now(),
            updated=datetime.now(),
            pdf_url="", arxiv_url=""
        )

        similarity = arxiv_server._calculate_similarity(paper1, paper2)

        # Should have positive similarity due to shared keywords and categories
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.1  # Should have some similarity

    def test_paper_formatting(self, arxiv_server):
        """Test paper list formatting."""
        papers = [
            ArXivPaper(
                arxiv_id="2301.00001",
                title="Test Paper 1",
                authors=["Author 1", "Author 2"],
                abstract="First test abstract.",
                categories=["cs.AI"],
                published=datetime(2023, 1, 1),
                updated=datetime(2023, 1, 1),
                pdf_url="", arxiv_url=""
            ),
            ArXivPaper(
                arxiv_id="2301.00002",
                title="Test Paper 2",
                authors=["Author 3"],
                abstract="Second test abstract.",
                categories=["cs.LG"],
                published=datetime(2023, 1, 2),
                updated=datetime(2023, 1, 2),
                pdf_url="", arxiv_url=""
            )
        ]

        formatted = arxiv_server._format_paper_list(papers)

        assert "📊 Found 2 papers" in formatted
        assert "Test Paper 1" in formatted
        assert "Test Paper 2" in formatted
        assert "2301.00001" in formatted
        assert "2301.00002" in formatted

    @pytest.mark.asyncio
    async def test_search_arxiv_no_results(self, arxiv_server):
        """Test ArXiv search with no results."""
        with patch.object(arxiv_server.client, 'results', return_value=[]):
            result = await arxiv_server._search_arxiv("nonexistent query")

            assert len(result) == 1
            assert "No papers found" in result[0].text

    def test_brief_summary_generation(self, arxiv_server):
        """Test brief summary generation."""
        papers = [
            ArXivPaper(
                arxiv_id="1",
                title="Machine Learning Paper",
                authors=["ML Author"],
                abstract="Machine learning research abstract",
                categories=["cs.LG", "cs.AI"],
                published=datetime.now(),
                updated=datetime.now(),
                pdf_url="", arxiv_url=""
            ),
            ArXivPaper(
                arxiv_id="2",
                title="Natural Language Processing Study",
                authors=["NLP Author"],
                abstract="NLP research abstract with language processing",
                categories=["cs.CL", "cs.AI"],
                published=datetime.now(),
                updated=datetime.now(),
                pdf_url="", arxiv_url=""
            )
        ]

        summary = arxiv_server._generate_brief_summary(papers)

        assert "📚 **Research Summary (2 papers)**" in summary
        assert "Key Topics Covered" in summary
        assert "Trending Research Terms" in summary


class TestAudioTranscriptionMCPServer:
    """Test Audio Transcription MCP Server functionality."""

    @pytest.fixture
    def transcription_server(self):
        """Create transcription server for testing."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            server = AudioTranscriptionMCPServer()
            return server

    @pytest.fixture
    def mock_audio_file(self):
        """Create a mock audio file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Write minimal WAV header (44 bytes) + some dummy audio data
            wav_header = (
                b'RIFF' +  # ChunkID
                (100).to_bytes(4, 'little') +  # ChunkSize
                b'WAVE' +  # Format
                b'fmt ' +  # Subchunk1ID
                (16).to_bytes(4, 'little') +  # Subchunk1Size
                (1).to_bytes(2, 'little') +   # AudioFormat (PCM)
                (1).to_bytes(2, 'little') +   # NumChannels (mono)
                (16000).to_bytes(4, 'little') +  # SampleRate
                (32000).to_bytes(4, 'little') +  # ByteRate
                (2).to_bytes(2, 'little') +   # BlockAlign
                (16).to_bytes(2, 'little') +  # BitsPerSample
                b'data' +  # Subchunk2ID
                (64).to_bytes(4, 'little')    # Subchunk2Size
            )
            # Add some dummy audio data (silence)
            dummy_audio = b'\x00\x00' * 32  # 64 bytes of silence
            tmp.write(wav_header + dummy_audio)
            tmp.flush()
            return tmp.name

    def test_server_initialization(self, transcription_server):
        """Test server initialization."""
        assert transcription_server.server is not None
        assert transcription_server.client is not None
        assert isinstance(transcription_server.cache, dict)
        assert isinstance(transcription_server.available_backends, dict)
        assert transcription_server.supported_formats == {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.webm'}

    def test_backend_availability_checking(self, transcription_server):
        """Test backend availability detection."""
        # Test that backend checking works
        backends = transcription_server.available_backends

        assert TranscriptionBackend.FASTER_WHISPER in backends
        assert TranscriptionBackend.OPENAI_WHISPER in backends
        assert TranscriptionBackend.OPENAI_API in backends

        # At least one backend should be available or disabled based on environment
        assert isinstance(backends[TranscriptionBackend.FASTER_WHISPER], bool)
        assert isinstance(backends[TranscriptionBackend.OPENAI_WHISPER], bool)
        assert isinstance(backends[TranscriptionBackend.OPENAI_API], bool)

    def test_recommended_backend_selection(self, transcription_server):
        """Test recommended backend selection logic."""
        recommended = transcription_server._get_recommended_backend()

        # Should return a valid backend name or 'none'
        valid_backends = [b.value for b in TranscriptionBackend] + ['none']
        assert recommended in valid_backends

    def test_cache_key_generation(self, transcription_server, mock_audio_file):
        """Test cache key generation."""
        cache_key1 = transcription_server._generate_cache_key(
            mock_audio_file, "faster-whisper", "turbo", "en", "transcribe"
        )
        cache_key2 = transcription_server._generate_cache_key(
            mock_audio_file, "faster-whisper", "turbo", "en", "transcribe"
        )
        cache_key3 = transcription_server._generate_cache_key(
            mock_audio_file, "faster-whisper", "base", "en", "transcribe"
        )

        # Same parameters should generate same key
        assert cache_key1 == cache_key2
        # Different parameters should generate different keys
        assert cache_key1 != cache_key3
        # Keys should be MD5 hashes (32 chars)
        assert len(cache_key1) == 32

    def test_cache_operations(self, transcription_server):
        """Test cache storage and retrieval."""
        # Create a mock transcription result
        result = TranscriptionResult(
            filename="test.wav",
            duration=60.0,
            language="en",
            language_probability=0.95,
            backend_used="faster-whisper",
            segments=[],
            full_text="This is a test transcription.",
            processing_time=5.0,
            model_used="turbo"
        )

        cache_key = "test_cache_key"

        # Test caching
        transcription_server._cache_result(cache_key, result)
        assert cache_key in transcription_server.cache

        # Test retrieval
        cached_result = transcription_server._get_cached_result(cache_key)
        assert cached_result is not None
        assert cached_result.full_text == result.full_text

        # Test expiration (simulate old timestamp)
        old_timestamp = datetime.now() - timedelta(hours=25)
        transcription_server.cache[cache_key] = (result, old_timestamp)

        expired_result = transcription_server._get_cached_result(cache_key)
        assert expired_result is None
        assert cache_key not in transcription_server.cache

    def test_transcript_cleaning(self, transcription_server):
        """Test transcript post-processing and cleaning."""
        raw_transcript = "Um, this is like, you know, a test transcript. Actually, it has, uh, some filler words."

        cleaned = transcription_server._clean_transcript(raw_transcript)

        # Should remove filler words
        assert "um" not in cleaned.lower()
        assert "uh" not in cleaned.lower()
        assert "like" not in cleaned.lower()
        assert "you know" not in cleaned.lower()
        assert "actually" not in cleaned.lower()

        # Should preserve meaningful content
        assert "test transcript" in cleaned
        assert "filler words" in cleaned

    def test_transcription_result_formatting(self, transcription_server):
        """Test transcription result formatting."""
        segments = [
            TranscriptionSegment(
                start=0.0,
                end=2.5,
                text="Hello world",
                confidence=0.95
            ),
            TranscriptionSegment(
                start=2.5,
                end=5.0,
                text="This is a test",
                confidence=0.88
            )
        ]

        result = TranscriptionResult(
            filename="test_audio.wav",
            duration=5.0,
            language="en",
            language_probability=0.95,
            backend_used="faster-whisper",
            segments=segments,
            full_text="Hello world. This is a test.",
            processing_time=2.3,
            model_used="turbo"
        )

        formatted = transcription_server._format_transcription_result(result)

        # Check that all key information is included
        assert "test_audio.wav" in formatted
        assert "5.00s" in formatted  # Duration
        assert "en" in formatted     # Language
        assert "95%" in formatted    # Language probability
        assert "faster-whisper" in formatted  # Backend
        assert "turbo" in formatted  # Model
        assert "2.30s" in formatted  # Processing time
        assert "Hello world. This is a test." in formatted  # Full text

    @pytest.mark.asyncio
    async def test_transcribe_audio_invalid_file(self, transcription_server):
        """Test transcription with non-existent file."""
        result = await transcription_server._transcribe_audio(
            file_path="/nonexistent/path.wav",
            backend="auto"
        )

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_transcribe_audio_no_backend_available(self, transcription_server):
        """Test transcription when no backends are available."""
        # Mock all backends as unavailable
        transcription_server.available_backends = {
            TranscriptionBackend.FASTER_WHISPER: False,
            TranscriptionBackend.OPENAI_WHISPER: False,
            TranscriptionBackend.OPENAI_API: False
        }

        result = await transcription_server._transcribe_audio(
            file_path="dummy.wav",
            backend="auto"
        )

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "not available" in result[0].text

    @pytest.mark.asyncio
    async def test_convert_audio_format_missing_file(self, transcription_server):
        """Test audio format conversion with missing file."""
        result = await transcription_server._convert_audio_format(
            input_path="/nonexistent/file.m4a",
            output_format="wav"
        )

        assert len(result) == 1
        assert "Error" in result[0].text
        assert "not found" in result[0].text

    @pytest.mark.asyncio
    async def test_batch_transcribe_empty_list(self, transcription_server):
        """Test batch transcription with empty file list."""
        result = await transcription_server._batch_transcribe(
            file_paths=[],
            output_format="txt",
            backend="auto",
            model="turbo"
        )

        assert len(result) == 1
        assert "Batch Transcription Complete (0 files)" in result[0].text

    def test_transcription_segment_model(self):
        """Test TranscriptionSegment model validation."""
        # Valid segment
        segment = TranscriptionSegment(
            start=0.0,
            end=5.0,
            text="Hello world",
            confidence=0.95,
            speaker="Speaker1"
        )

        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Hello world"
        assert segment.confidence == 0.95
        assert segment.speaker == "Speaker1"

    def test_transcription_result_model(self):
        """Test TranscriptionResult model validation."""
        segments = [
            TranscriptionSegment(start=0.0, end=2.0, text="First segment"),
            TranscriptionSegment(start=2.0, end=4.0, text="Second segment")
        ]

        result = TranscriptionResult(
            filename="test.wav",
            duration=4.0,
            language="en",
            language_probability=0.98,
            backend_used="faster-whisper",
            segments=segments,
            full_text="First segment. Second segment.",
            processing_time=1.5,
            model_used="turbo"
        )

        assert result.filename == "test.wav"
        assert result.duration == 4.0
        assert result.language == "en"
        assert result.language_probability == 0.98
        assert result.backend_used == "faster-whisper"
        assert len(result.segments) == 2
        assert result.full_text == "First segment. Second segment."
        assert result.processing_time == 1.5
        assert result.model_used == "turbo"

    def test_backend_enum(self):
        """Test TranscriptionBackend enum values."""
        assert TranscriptionBackend.FASTER_WHISPER.value == "faster-whisper"
        assert TranscriptionBackend.OPENAI_WHISPER.value == "openai-whisper"
        assert TranscriptionBackend.OPENAI_API.value == "openai-api"

    @pytest.mark.asyncio
    async def test_mcp_tools_registration(self, transcription_server):
        """Test that MCP tools are properly registered."""
        # Since we can't easily test the full MCP server here,
        # we'll just verify the server has the expected structure
        assert hasattr(transcription_server, 'server')
        assert hasattr(transcription_server, '_register_tools')
        assert hasattr(transcription_server, '_register_resources')

        # Test that supported formats are properly configured
        expected_formats = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.webm'}
        assert transcription_server.supported_formats == expected_formats

    def test_cache_size_calculation(self, transcription_server):
        """Test cache size calculation."""
        # Add some mock entries to cache
        for i in range(5):
            result = TranscriptionResult(
                filename=f"test{i}.wav",
                duration=60.0,
                language="en",
                language_probability=0.95,
                backend_used="faster-whisper",
                segments=[],
                full_text=f"Test transcription {i}",
                processing_time=5.0,
                model_used="turbo"
            )
            transcription_server.cache[f"key{i}"] = (result, datetime.now())

        size_mb = transcription_server._calculate_cache_size()
        assert size_mb == 0.5  # 5 entries * 0.1MB estimated per entry

    def test_oldest_cache_entry(self, transcription_server):
        """Test oldest cache entry tracking."""
        # Empty cache
        oldest = transcription_server._get_oldest_cache_entry()
        assert oldest is None

        # Add entries with different timestamps
        now = datetime.now()
        older = now - timedelta(hours=1)
        oldest_time = now - timedelta(hours=2)

        result = TranscriptionResult(
            filename="test.wav", duration=60.0, language="en",
            language_probability=0.95, backend_used="test",
            segments=[], full_text="test", processing_time=1.0, model_used="test"
        )

        transcription_server.cache["new"] = (result, now)
        transcription_server.cache["old"] = (result, older)
        transcription_server.cache["oldest"] = (result, oldest_time)

        oldest = transcription_server._get_oldest_cache_entry()
        assert oldest == oldest_time.isoformat()


class TestMCPIntegration:
    """Test MCP integration and compatibility layer."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config = Mock()
        config.is_mcp_enabled.return_value = True
        config.get_search_preference.return_value = "mcp"
        config.should_use_mcp_for_tool.return_value = True
        config.mcp_fallback_to_langchain = True
        return config

    def test_mcp_configuration_integration(self):
        """Test MCP configuration integration with main config."""
        # Test importing MCP config in main configuration
        try:
            from src.config.configuration import Configuration
            config = Configuration()

            # Test MCP-related properties
            assert hasattr(config, 'use_mcp_servers')
            assert hasattr(config, 'mcp_fallback_to_langchain')
            assert hasattr(config, 'is_mcp_enabled')
            assert hasattr(config, 'should_use_mcp_for_tool')

        except ImportError:
            # If MCP integration isn't available, that's expected in some environments
            pytest.skip("MCP integration not available")

    @pytest.mark.asyncio
    async def test_mcp_tool_factory(self):
        """Test MCP tool factory functionality."""
        try:
            from src.mcp_integration import MCPToolFactory

            factory = MCPToolFactory()

            # Test creating MCP search tool
            search_tool = await factory.create_mcp_search_tool()
            assert search_tool.name == "tavily_search"
            assert "search" in search_tool.description.lower()

            # Test creating vector tool
            vector_tool = await factory.create_mcp_vector_tool()
            assert vector_tool.name == "vector_retrieval"
            assert "vector" in vector_tool.description.lower()

            # Test creating arxiv tool
            arxiv_tool = await factory.create_mcp_arxiv_tool()
            assert arxiv_tool.name == "arxiv_search"
            assert "arxiv" in arxiv_tool.description.lower()

            # Test creating audio transcription tool
            audio_tool = await factory.create_mcp_audio_transcription_tool()
            assert audio_tool.name == "audio_transcription"
            assert "transcribe" in audio_tool.description.lower()

        except ImportError:
            pytest.skip("MCP integration not available")

    @pytest.mark.asyncio
    async def test_mcp_compatibility_layer_audio_transcription(self, mock_config):
        """Test MCP compatibility layer for audio transcription."""
        try:
            from src.mcp_integration import MCPCompatibilityLayer

            layer = MCPCompatibilityLayer(mock_config)

            # Test getting transcription tool
            transcription_tool = await layer.get_audio_transcription_tool()

            # Should return either an MCP tool or direct implementation
            assert transcription_tool is not None
            assert hasattr(transcription_tool, 'name')
            assert "transcription" in transcription_tool.name.lower()

        except ImportError:
            pytest.skip("MCP integration not available")


@pytest.mark.integration
class TestMCPServerManager:
    """Integration tests for MCP Server Manager."""

    @pytest.fixture
    def server_manager(self):
        """Create server manager for testing."""
        return MCPServerManager()

    def test_manager_initialization(self, server_manager):
        """Test server manager initialization."""
        assert server_manager.config is not None
        assert isinstance(server_manager.running_servers, dict)
        assert isinstance(server_manager.server_health, dict)
        assert len(server_manager.server_modules) >= 3

    def test_get_server_status(self, server_manager):
        """Test getting server status."""
        status = server_manager.get_server_status()

        assert isinstance(status, dict)
        # Should have status for each configured server
        assert len(status) > 0

        for server_name, info in status.items():
            assert "enabled" in info
            assert "type" in info
            assert "running" in info
            assert "healthy" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])