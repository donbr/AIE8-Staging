# Testing Procedures

## Quick Start

```bash
# Run all current tests
python validate_notebook_compatibility.py    # Phase 1 integration tests
python validate_phase2_components.py         # Phase 2 component tests
PYTHONPATH=src python -m pytest tests/test_configuration.py -v  # Phase 1 unit tests
```

## Test Categories

### 1. Unit Tests
**Purpose**: Test individual components in isolation
**Location**: `tests/` directory
**Run Command**: `PYTHONPATH=src python -m pytest tests/ -v`

#### Available Unit Test Files
- `test_configuration.py` - Configuration system (17 tests) ✅ WORKING
- `test_agents.py` - Agent factories and state classes ⚠️ IMPORT ISSUES
- `test_graphs.py` - Graph modules ⚠️ IMPORT ISSUES
- `test_tools.py` - Tool organization ⚠️ IMPORT ISSUES

#### Working Unit Tests
```bash
# Configuration tests (fully working)
PYTHONPATH=src python -m pytest tests/test_configuration.py -v

# Expected output: 17 tests passing
# Tests cover: configuration creation, environment overrides,
# factory functions, backward compatibility
```

### 2. Integration Tests
**Purpose**: Test cross-component compatibility
**Location**: Root directory validation scripts
**Run Command**: `python <script_name>.py`

#### Phase 1 Integration Tests
```bash
# Test notebook compatibility with new configuration system
python validate_notebook_compatibility.py

# Expected output:
# 🚀 Starting notebook compatibility validation...
# ✅ Basic imports successful
# ✅ Configuration values correct
# ✅ LLM instances created correctly
# ✅ Embedding model created correctly
# ✅ Search tool created correctly
# ✅ Helper functions work correctly
# ✅ Prompt externalization works correctly
# 📊 Results: 7 passed, 0 failed
# 🎉 All compatibility tests passed!
```

#### Phase 2 Integration Tests
```bash
# Test all Phase 2 components
python validate_phase2_components.py

# Expected output:
# === Phase 2 Component Validation ===
# ✅ State classes working correctly
# ✅ Agent factories structure and configuration working correctly
# ✅ Graph modules structure working correctly
# ✅ Tool organization structure working correctly
# ✅ Backward compatibility structure working correctly
# === Summary ===
# Passed: 5/5
# 🎉 All Phase 2 components are working correctly!
```

### 3. Notebook Validation
**Purpose**: Ensure original notebook still works
**Location**: Jupyter notebook files
**Run Command**: Manual execution or nbconvert

#### Manual Notebook Testing
```bash
# Start Jupyter and run notebook cells
jupyter lab
# Navigate to Multi_Agent_RAG_LangGraph.ipynb
# Run all cells to ensure functionality
```

#### Automated Notebook Testing (Future)
```bash
# Execute notebook programmatically (not yet implemented)
jupyter nbconvert --execute Multi_Agent_RAG_LangGraph.ipynb --to notebook
```

## Test Execution Guide

### Pre-Test Setup
```bash
# Ensure dependencies are installed
uv sync

# Set required environment variables
export OPENAI_API_KEY="your-key-here"
export TAVILY_API_KEY="your-key-here"

# Optional: Set custom models for testing
export RESEARCH_MODEL="gpt-4o-mini"
export WRITING_MODEL="gpt-4o-mini"
export SUPERVISOR_MODEL="gpt-4o"
```

### Running Tests by Phase

#### Phase 1 Only
```bash
# Unit tests
PYTHONPATH=src python -m pytest tests/test_configuration.py -v

# Integration tests
python validate_notebook_compatibility.py
```

#### Phase 2 Only
```bash
# Component validation
python validate_phase2_components.py
```

#### All Current Tests
```bash
# Run everything (recommended)
echo "=== Phase 1 Unit Tests ==="
PYTHONPATH=src python -m pytest tests/test_configuration.py -v

echo -e "\n=== Phase 1 Integration Tests ==="
python validate_notebook_compatibility.py

echo -e "\n=== Phase 2 Component Tests ==="
python validate_phase2_components.py
```

### Interpreting Test Results

#### Success Indicators
- **Unit Tests**: All tests show `PASSED`, no `FAILED` or `ERROR`
- **Integration Tests**: All ✅ checkmarks, final success message
- **Component Tests**: `Passed: X/X` with 100% success rate

#### Warning Signs
- **Deprecation Warnings**: Expected for LangChain components, not critical
- **Import Warnings**: May indicate version mismatches
- **Timeout Warnings**: Could indicate API connectivity issues

#### Failure Investigation
```bash
# Run with verbose output
PYTHONPATH=src python -m pytest tests/test_configuration.py -v -s

# Run specific test
PYTHONPATH=src python -m pytest tests/test_configuration.py::TestConfiguration::test_default_configuration -v

# Check Python path issues
python -c "import sys; print('\n'.join(sys.path))"
```

## Test Maintenance

### Adding New Tests

#### Unit Test Template
```python
# tests/test_new_feature.py
import unittest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass

    def test_feature_works(self):
        """Test that the feature works correctly."""
        # Arrange
        expected = "expected_value"

        # Act
        result = your_function()

        # Assert
        self.assertEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
```

#### Integration Test Template
```python
# validate_new_feature.py
def test_new_feature():
    """Test new feature integration."""
    try:
        # Test implementation here
        print("✅ New feature working correctly")
        return True
    except Exception as e:
        print(f"❌ New feature failed: {e}")
        return False

def main():
    results = [test_new_feature()]
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"🎉 All tests passed! ({passed}/{total})")
        return 0
    else:
        print(f"⚠️ Some tests failed ({passed}/{total})")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

### Updating Tests After Changes

#### When Configuration Changes
1. Update `test_configuration.py` with new fields
2. Update `validate_notebook_compatibility.py` if backward compatibility affected
3. Verify all tests still pass

#### When Adding New Modules
1. Create new test file in `tests/` directory
2. Add validation to appropriate `validate_*.py` script
3. Update this documentation

#### When Modifying Existing Functionality
1. Update existing unit tests
2. Run all tests to check for regressions
3. Update integration tests if interfaces changed

### Known Issues and Workarounds

#### Import Issues with Phase 2 Unit Tests
**Problem**: Relative imports in `src/` modules cause pytest failures
**Symptoms**: `ImportError: attempted relative import beyond top-level package`
**Workaround**: Use validation scripts instead of direct pytest
**Future Fix**: Restructure imports or package configuration

```bash
# This fails:
PYTHONPATH=src python -m pytest tests/test_agents.py

# Use this instead:
python validate_phase2_components.py
```

#### Environment Variable Dependencies
**Problem**: Some tests require API keys
**Symptoms**: Tests fail with missing environment variables
**Workaround**: Set dummy values for testing

```bash
# For testing without real API calls
export OPENAI_API_KEY="test-key"
export TAVILY_API_KEY="test-key"
```

#### LangChain Deprecation Warnings
**Problem**: Deprecated classes cause warning noise
**Symptoms**: Warnings about TavilySearchResults
**Status**: Expected, not critical, will be addressed in Phase 3

### Performance Testing

#### Current Performance Baselines
- Configuration creation: < 10ms
- Agent factory creation: < 50ms (without LLM calls)
- Module imports: < 100ms total

#### Running Performance Tests
```bash
# Time imports
time python -c "import sys; sys.path.insert(0, 'src'); from config.configuration import Configuration; Configuration()"

# Profile test execution
python -m cProfile -s cumulative validate_phase2_components.py
```

## Troubleshooting

### Common Test Failures

#### "Module not found" errors
```bash
# Check Python path
echo $PYTHONPATH

# Set correct path
export PYTHONPATH=/path/to/project/src

# Or use explicit path
PYTHONPATH=src python -m pytest tests/test_configuration.py
```

#### "Configuration validation" errors
```bash
# Check environment variables
env | grep -E "(OPENAI|TAVILY|RESEARCH|WRITING)"

# Set required variables
export OPENAI_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
```

#### Import conflicts
```bash
# Clear Python cache
find . -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete

# Restart Python interpreter
```

### Getting Help

1. **Check this documentation** for common issues
2. **Run validation scripts** before reporting issues
3. **Include full error output** when asking for help
4. **Specify which test category** is failing

### Test Environment Requirements

#### Python Version
- **Required**: Python 3.11+
- **Tested**: Python 3.11.13
- **Check**: `python --version`

#### Dependencies
- **Core**: See `pyproject.toml` for full list
- **Testing**: `pytest`, `unittest` (built-in)
- **Validation**: No additional dependencies

#### System Requirements
- **Memory**: 1GB+ available (for LLM model loading)
- **Network**: Required for API key validation tests
- **Disk**: 100MB+ for test artifacts

## Test Coverage Goals

### Current Coverage
- **Phase 1**: ~80% of functionality tested
- **Phase 2**: ~60% of functionality tested (structural validation only)
- **Integration**: ~70% of critical paths tested

### Target Coverage (Future)
- **Unit Tests**: 90%+ of public methods
- **Integration Tests**: 95%+ of user workflows
- **Performance Tests**: All critical paths benchmarked
- **Security Tests**: All input validation covered