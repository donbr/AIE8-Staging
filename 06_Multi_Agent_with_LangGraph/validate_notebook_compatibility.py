#!/usr/bin/env python3
"""
Validation script to ensure notebook compatibility with new configuration system.

This script tests that all the notebook's key functionality still works with
the new configuration system through the backward compatibility layer.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test that basic imports work from the compatibility layer."""
    print("Testing basic imports...")
    try:
        from src.compat import (
            config,
            research_llm,
            authoring_llm,
            super_llm,
            generator_llm,
            embedding_model,
            chunk_size,
            chunk_overlap,
            tiktoken_len
        )
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_configuration_values():
    """Test that configuration values are properly set."""
    print("Testing configuration values...")
    try:
        from src.compat import config, chunk_size, chunk_overlap

        # Test that configuration instance exists
        assert config is not None, "Configuration instance should exist"

        # Test that notebook values are properly set
        assert chunk_size == 750, f"Expected chunk_size=750, got {chunk_size}"
        assert chunk_overlap == 0, f"Expected chunk_overlap=0, got {chunk_overlap}"

        # Test model configurations
        assert config.research_model == "gpt-4o-mini", f"Expected research_model='gpt-4o-mini', got {config.research_model}"
        assert config.supervisor_model == "gpt-4o", f"Expected supervisor_model='gpt-4o', got {config.supervisor_model}"

        print("✅ Configuration values correct")
        return True
    except Exception as e:
        print(f"❌ Configuration values test failed: {e}")
        return False

def test_llm_instances():
    """Test that LLM instances are properly created."""
    print("Testing LLM instances...")
    try:
        from src.compat import research_llm, authoring_llm, super_llm, generator_llm
        from langchain_openai import ChatOpenAI

        # Test that all LLMs are ChatOpenAI instances
        assert isinstance(research_llm, ChatOpenAI), "research_llm should be ChatOpenAI instance"
        assert isinstance(authoring_llm, ChatOpenAI), "authoring_llm should be ChatOpenAI instance"
        assert isinstance(super_llm, ChatOpenAI), "super_llm should be ChatOpenAI instance"
        assert isinstance(generator_llm, ChatOpenAI), "generator_llm should be ChatOpenAI instance"

        print("✅ LLM instances created correctly")
        return True
    except Exception as e:
        print(f"❌ LLM instances test failed: {e}")
        return False

def test_embedding_model():
    """Test that embedding model is properly created."""
    print("Testing embedding model...")
    try:
        from src.compat import embedding_model
        from langchain_openai.embeddings import OpenAIEmbeddings

        assert isinstance(embedding_model, OpenAIEmbeddings), "embedding_model should be OpenAIEmbeddings instance"

        print("✅ Embedding model created correctly")
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_search_tool():
    """Test that search tool is properly created."""
    print("Testing search tool...")
    try:
        from src.compat import search_tool
        from langchain_community.tools.tavily_search import TavilySearchResults

        # Search tool should be TavilySearchResults (could be None if search is disabled)
        if search_tool is not None:
            assert isinstance(search_tool, TavilySearchResults), "search_tool should be TavilySearchResults instance"
            print("✅ Search tool created correctly")
        else:
            print("✅ Search tool is None (search disabled)")
        return True
    except Exception as e:
        print(f"❌ Search tool test failed: {e}")
        return False

def test_helper_functions():
    """Test that notebook helper functions work."""
    print("Testing helper functions...")
    try:
        from src.compat import tiktoken_len, create_agent, create_team_supervisor

        # Test tiktoken_len function
        length = tiktoken_len("Hello world")
        assert isinstance(length, int), "tiktoken_len should return int"
        assert length > 0, "tiktoken_len should return positive value"

        # Test that create_agent and create_team_supervisor functions exist
        assert callable(create_agent), "create_agent should be callable"
        assert callable(create_team_supervisor), "create_team_supervisor should be callable"

        print("✅ Helper functions work correctly")
        return True
    except Exception as e:
        print(f"❌ Helper functions test failed: {e}")
        return False

def test_prompt_externalization():
    """Test that prompts are properly externalized."""
    print("Testing prompt externalization...")
    try:
        from src.config.prompts import (
            RESEARCH_SUPERVISOR_PROMPT,
            SEARCH_AGENT_PROMPT,
            RAG_AGENT_PROMPT,
            DOC_WRITER_PROMPT,
            COPY_EDITOR_PROMPT,
            BASE_AGENT_INSTRUCTION,
            get_current_date,
            format_prompt
        )

        # Test that prompts exist and are strings
        assert isinstance(RESEARCH_SUPERVISOR_PROMPT, str), "RESEARCH_SUPERVISOR_PROMPT should be string"
        assert isinstance(BASE_AGENT_INSTRUCTION, str), "BASE_AGENT_INSTRUCTION should be string"
        assert len(RESEARCH_SUPERVISOR_PROMPT) > 0, "RESEARCH_SUPERVISOR_PROMPT should not be empty"

        # Test prompt formatting
        date = get_current_date()
        assert isinstance(date, str), "get_current_date should return string"

        formatted = format_prompt("Hello {name}", name="World")
        assert "Hello World" in formatted, "format_prompt should work correctly"

        print("✅ Prompt externalization works correctly")
        return True
    except Exception as e:
        print(f"❌ Prompt externalization test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("🚀 Starting notebook compatibility validation...\n")

    tests = [
        test_basic_imports,
        test_configuration_values,
        test_llm_instances,
        test_embedding_model,
        test_search_tool,
        test_helper_functions,
        test_prompt_externalization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()  # Empty line between tests

    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All compatibility tests passed! Notebook should work with new configuration system.")
        return True
    else:
        print("💥 Some tests failed. Notebook compatibility issues detected.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)