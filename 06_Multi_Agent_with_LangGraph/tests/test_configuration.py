"""Unit tests for the configuration system.

Tests the Pydantic Configuration class and factory functions to ensure
proper validation, defaults, and environment variable handling.
"""

import os
import pytest
from unittest.mock import patch

from src.config.configuration import Configuration, SearchAPI
from src.core.factory import (
    create_chat_model,
    create_embedding_model,
    create_search_tool,
    create_research_llm,
    create_writing_llm,
    create_supervisor_llm
)


class TestConfiguration:
    """Test the Configuration class."""

    def test_default_configuration(self):
        """Test that default configuration values are set correctly."""
        config = Configuration()

        # Test model defaults
        assert config.research_model == "gpt-4o-mini"
        assert config.writing_model == "gpt-4o-mini"
        assert config.supervisor_model == "gpt-4o"
        assert config.embedding_model == "text-embedding-3-small"

        # Test research configuration
        assert config.search_api == SearchAPI.TAVILY
        assert config.max_research_iterations == 6
        assert config.max_concurrent_research_units == 5

        # Test RAG configuration
        assert config.chunk_size == 750
        assert config.chunk_overlap == 0

        # Test temperature settings
        assert config.research_temperature == 0.1
        assert config.writing_temperature == 0.3
        assert config.supervisor_temperature == 0.0

    def test_environment_variable_override(self):
        """Test that environment variables override defaults via from_runnable_config."""
        with patch.dict(os.environ, {
            'RESEARCH_MODEL': 'gpt-4',
            'CHUNK_SIZE': '1000',
            'MAX_RESEARCH_ITERATIONS': '10'
        }):
            config = Configuration.from_runnable_config()
            assert config.research_model == "gpt-4"
            assert config.chunk_size == 1000
            assert config.max_research_iterations == 10

    def test_get_model_name(self):
        """Test model name retrieval by role."""
        config = Configuration()

        assert config.get_model_name("research") == "gpt-4o-mini"
        assert config.get_model_name("writing") == "gpt-4o-mini"
        assert config.get_model_name("supervisor") == "gpt-4o"
        assert config.get_model_name("unknown") == "gpt-4o-mini"  # fallback

    def test_get_model_max_tokens(self):
        """Test max tokens retrieval by role."""
        config = Configuration()

        assert config.get_model_max_tokens("research") == 10000
        assert config.get_model_max_tokens("writing") == 10000
        assert config.get_model_max_tokens("supervisor") == 8192
        assert config.get_model_max_tokens("unknown") == 10000  # fallback

    def test_get_temperature(self):
        """Test temperature retrieval by role."""
        config = Configuration()

        assert config.get_temperature("research") == 0.1
        assert config.get_temperature("writing") == 0.3
        assert config.get_temperature("supervisor") == 0.0
        assert config.get_temperature("unknown") == 0.1  # fallback

    def test_search_api_enum(self):
        """Test SearchAPI enum values are converted to strings."""
        config = Configuration(search_api=SearchAPI.ANTHROPIC)
        # Due to use_enum_values=True, enum is converted to string value
        assert config.search_api == "anthropic"

    def test_validation(self):
        """Test Pydantic validation."""
        # Test that negative values are handled
        config = Configuration(max_research_iterations=-1)
        assert config.max_research_iterations == -1  # Pydantic allows this

        # Test invalid enum
        with pytest.raises(ValueError):
            Configuration(search_api="invalid_api")

    def test_from_runnable_config(self):
        """Test configuration creation from RunnableConfig."""
        runnable_config = {
            "configurable": {
                "research_model": "gpt-3.5-turbo",
                "chunk_size": 500
            }
        }

        config = Configuration.from_runnable_config(runnable_config)
        assert config.research_model == "gpt-3.5-turbo"
        assert config.chunk_size == 500
        # Other values should be defaults
        assert config.writing_model == "gpt-4o-mini"


class TestFactory:
    """Test the factory functions."""

    def test_create_chat_model(self):
        """Test chat model creation."""
        config = Configuration()
        model = create_chat_model(config, role="research")

        # Check that it's a ChatOpenAI instance
        from langchain_openai import ChatOpenAI
        assert isinstance(model, ChatOpenAI)

    def test_create_embedding_model(self):
        """Test embedding model creation."""
        config = Configuration()
        model = create_embedding_model(config)

        from langchain_openai.embeddings import OpenAIEmbeddings
        assert isinstance(model, OpenAIEmbeddings)

    def test_create_search_tool_tavily(self):
        """Test search tool creation for Tavily."""
        config = Configuration(search_api=SearchAPI.TAVILY)
        tool = create_search_tool(config)

        from langchain_community.tools.tavily_search import TavilySearchResults
        assert isinstance(tool, TavilySearchResults)

    def test_create_search_tool_none(self):
        """Test search tool creation when disabled."""
        config = Configuration(search_api=SearchAPI.NONE)
        tool = create_search_tool(config)
        assert tool is None

    def test_create_search_tool_unsupported(self):
        """Test search tool creation for unsupported APIs."""
        config = Configuration(search_api=SearchAPI.OPENAI)

        with pytest.raises(NotImplementedError):
            create_search_tool(config)

    def test_role_specific_llms(self):
        """Test role-specific LLM creation functions."""
        config = Configuration()

        research_llm = create_research_llm(config)
        writing_llm = create_writing_llm(config)
        supervisor_llm = create_supervisor_llm(config)

        from langchain_openai import ChatOpenAI
        assert isinstance(research_llm, ChatOpenAI)
        assert isinstance(writing_llm, ChatOpenAI)
        assert isinstance(supervisor_llm, ChatOpenAI)

    def test_custom_model_kwargs(self):
        """Test passing custom kwargs to model creation."""
        config = Configuration()
        model = create_chat_model(config, role="research", temperature=0.5)

        # The factory should use the custom temperature
        assert model.temperature == 0.5


class TestBackwardCompatibility:
    """Test backward compatibility with the original notebook."""

    def test_compat_imports(self):
        """Test that compatibility imports work."""
        from src.compat import (
            config,
            research_llm,
            authoring_llm,
            super_llm,
            embedding_model,
            chunk_size,
            chunk_overlap
        )

        # Check types
        assert isinstance(config, Configuration)
        assert chunk_size == 750
        assert chunk_overlap == 0

    def test_tiktoken_len_function(self):
        """Test the tiktoken_len backward compatibility function."""
        from src.compat import tiktoken_len

        # Test with simple text
        length = tiktoken_len("Hello world")
        assert isinstance(length, int)
        assert length > 0


if __name__ == "__main__":
    pytest.main([__file__])