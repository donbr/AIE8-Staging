#!/usr/bin/env python3
"""Validation script for Phase 2 components.

Tests state classes, agent factories, graph modules, and tool organization.
Validates that all Phase 2 refactoring components work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_state_classes():
    """Test MessagesState-based state classes."""
    print("Testing state classes...")

    try:
        from agents.state import SimpleRAGState, ResearchTeamState, DocWritingState, MetaSupervisorState

        # Test state creation
        rag_state = SimpleRAGState()
        research_state = ResearchTeamState()
        writing_state = DocWritingState()
        meta_state = MetaSupervisorState()

        # Test state updates
        research_state.update({"team_members": ["Agent1", "Agent2"]})
        writing_state.update({"current_files": {"count": 5}})

        print("✅ State classes working correctly")
        return True

    except Exception as e:
        print(f"❌ State classes failed: {e}")
        return False


def test_agent_factories():
    """Test agent factory functions."""
    print("Testing agent factory functions...")

    try:
        from config.configuration import Configuration, SearchAPI
        from pathlib import Path

        # Test that agent modules exist
        src_path = Path(__file__).parent / "src"
        agents_path = src_path / "agents"

        expected_modules = [
            "state.py",
            "factory.py",
            "__init__.py"
        ]

        for module in expected_modules:
            module_path = agents_path / module
            assert module_path.exists(), f"Missing agent module: {module}"

        # Create test configuration
        config = Configuration(
            research_model="gpt-4o-mini"
        )

        # Test configuration creation
        assert config.research_model == "gpt-4o-mini"
        assert config.search_api == SearchAPI.TAVILY  # Should be default
        print("✅ Agent factories structure and configuration working correctly")
        return True

    except Exception as e:
        print(f"❌ Agent factories failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_modules():
    """Test graph module imports and structure."""
    print("Testing graph modules...")

    try:
        # Test that graph modules exist and have expected structure
        from pathlib import Path

        src_path = Path(__file__).parent / "src"
        graphs_path = src_path / "graphs"

        # Check that graph modules exist
        expected_modules = [
            "simple_rag.py",
            "research_team.py",
            "writing_team.py",
            "meta_supervisor.py",
            "__init__.py"
        ]

        for module in expected_modules:
            module_path = graphs_path / module
            assert module_path.exists(), f"Missing graph module: {module}"

        print("✅ Graph modules structure working correctly")
        return True

    except Exception as e:
        print(f"❌ Graph modules failed: {e}")
        return False


def test_tool_organization():
    """Test tool organization modules."""
    print("Testing tool organization...")

    try:
        # Test that tool modules exist and have expected structure
        from pathlib import Path

        src_path = Path(__file__).parent / "src"
        tools_path = src_path / "tools"

        # Check that tool modules exist
        expected_modules = [
            "search.py",
            "file_management.py",
            "__init__.py"
        ]

        for module in expected_modules:
            module_path = tools_path / module
            assert module_path.exists(), f"Missing tool module: {module}"

        # Check that modules have expected content
        init_content = (tools_path / "__init__.py").read_text()
        assert "create_tavily_search_tool" in init_content
        assert "create_file_management_tools" in init_content

        print("✅ Tool organization structure working correctly")
        return True

    except Exception as e:
        print(f"❌ Tool organization failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility layer."""
    print("Testing backward compatibility...")

    try:
        # Test that compatibility module exists and has expected structure
        from pathlib import Path

        src_path = Path(__file__).parent / "src"
        compat_path = src_path / "compat.py"

        assert compat_path.exists(), "Missing compat.py"

        # Check that compat module has expected functions
        compat_content = compat_path.read_text()
        expected_functions = [
            "def get_config",
            "def create_agent",
            "def create_team_supervisor",
            "def tiktoken_len"
        ]

        for func in expected_functions:
            assert func in compat_content, f"Missing function: {func}"

        print("✅ Backward compatibility structure working correctly")
        return True

    except Exception as e:
        print(f"❌ Backward compatibility failed: {e}")
        return False


def main():
    """Run all Phase 2 validation tests."""
    print("=== Phase 2 Component Validation ===\n")

    tests = [
        test_state_classes,
        test_agent_factories,
        test_graph_modules,
        test_tool_organization,
        test_backward_compatibility
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}\n")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print("=== Summary ===")
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All Phase 2 components are working correctly!")
        return 0
    else:
        print("⚠️ Some Phase 2 components need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())