# **CLAUDE.md: A Guideline for Refactoring LangGraph Notebooks to Production-Ready Libraries**

**Objective:** To evolve the current LangGraph Jupyter Notebook prototype into a modular, configurable, and deployable Python library. This refactoring will enhance maintainability, scalability, and testability, preparing the application for production environments.

---

### **Phase 1: Foundational Project Structuring**

The first step is to move away from the single-file notebook environment and establish a standard Python project structure.

1.  **Create the Directory Structure:**
    Organize the project logically to separate concerns.

    ```
   .
    ├── src/
    │   ├── agents/         # Logic for individual agents/nodes
    │   ├── pipelines/      # Sub-graphs for distinct workflows (e.g., research, writing)
    │   ├── tools/          # Definitions for custom tools
    │   ├── app/            # Server entrypoint and API definitions
    │   ├── config.py       # Centralized configuration
    │   └── state.py        # State definitions for the graph(s)
    ├── tests/
    │   └── evaluate.py     # End-to-end evaluation script
    ├──.env                # For secrets and environment variables
    ├──.env.example        # Template for environment variables
    ├── pyproject.toml      # Project dependencies and metadata
    └── README.md           # Project documentation
    ```

2.  **Migrate Notebook Code:**
    *   Transfer all Python code from the notebook into the appropriate files within the `src/` directory.
    *   The `pyproject.toml` file should manage all dependencies, replacing any `pip install` cells.

---

### **Phase 2: Configuration and Abstraction**

Hardcoding values is the primary source of fragility in prototypes. We will externalize all configurations.

1.  **Centralize Configuration (`src/config.py`):**
    *   Move all hardcoded values—such as model names, prompts, temperature settings, retry counts, and API URLs—into a dedicated `config.py` file.
    *   This allows for easy adjustments without touching the core application logic. As seen in `open_deep_research`, you should define configurations for different LLM roles (e.g., `SummarizationModel`, `ResearchModel`, `FinalReportModel`).[1]

2.  **Manage Secrets (`.env`):**
    *   All API keys (OpenAI, Tavily, LangSmith, etc.) and other secrets must be moved to a `.env` file.
    *   Use a library like `python-dotenv` to load these variables at runtime.
    *   The `.env` file should be included in `.gitignore`, and a `.env.example` file should be committed to the repository to guide new developers.

3.  **Abstract Initializations:**
    *   Create factory functions in your configuration module (e.g., `init_chat_model()`, `get_search_tool()`). These functions will read from the config and environment variables to instantiate and return the correct clients. This makes swapping between `OpenAI` and a local `Ollama` instance a simple configuration change.[2]

---

### **Phase 3: Modularizing the LangGraph Workflow**

Deconstruct the monolithic graph definition into logical, reusable components.

1.  **Define Typed State (`src/state.py`):**
    *   The graph's state is its central nervous system. Define it explicitly using Python's `TypedDict` or, preferably, a Pydantic `BaseModel`.
    *   A strongly-typed state object prevents runtime errors, clarifies data flow, and serves as documentation for what information is available at each step of the process.[3]

2.  **Isolate Nodes (`src/agents/`):**
    *   Each logical block of code from the notebook that will become a node in the graph (e.g., `call_tavily`, `summarize_results`, `critique_draft`) should be refactored into its own Python function in a dedicated module like `src/agents/researcher.py`.
    *   Each node function should accept the state object as its sole argument and return a dictionary representing the partial state update. This enforces a clean, predictable pattern.

3.  **Separate Graph Construction:**
    *   Create a dedicated module (e.g., `src/pipelines/research_graph.py`) whose only responsibility is to import the state and node functions and assemble the `StatefulGraph`.
    *   This file will contain all the `graph.add_node()`, `graph.add_edge()`, and `graph.set_entry_point()` calls. This separation makes the application's control flow explicit and easy to understand, distinguishing the *what* (nodes) from the *how* (graph structure).[4]

4.  **Decompose into Subgraphs (for complex systems):**
    *   For multi-phase processes like the "Scope -> Research -> Write" architecture, each phase should be its own LangGraph instance (a subgraph).[3]
    *   The main application graph then becomes a "Hierarchical Agent Team" supervisor, where its nodes are other LangGraph graphs. This is a powerful pattern for managing complexity.[5]

---

### **Phase 4: Implementing Advanced Agentic Patterns**

Leverage established patterns to improve performance and reliability.

1.  **Supervisor Pattern for Parallelism:**
    *   To accelerate research, implement a supervisor agent that breaks a task into sub-queries and delegates them to multiple, parallel worker agents.[3]
    *   The supervisor's tools should be functions that invoke the worker agents. Use `asyncio.gather()` to execute these workers concurrently for true performance gains. The supervisor then aggregates the results.

2.  **Reflection and Critique Loops:**
    *   Formalize any self-correction logic. Create distinct nodes for "generation" and "critique." A conditional edge routes the output of the generator to the critique node, and the critique is then fed back into the generator. This iterative refinement loop is a core agentic pattern that dramatically improves output quality.[6]

---

### **Phase 5: Tooling, Deployment, and Observability**

Prepare the library for real-world use.

1.  **Standardize Tool Definitions (`src/tools/`):**
    *   Define all external tools in a dedicated `src/tools/` directory.
    *   Use Pydantic models for tool input schemas to get automatic validation and clear function signatures for the LLM.

2.  **Create a Server Entrypoint (`src/app/server.py`):**
    *   The final step in the library is to expose the compiled graph. Use `langgraph-cli` to serve the application.
    *   Your entrypoint script will import the graph object and make it available for the server. This decouples the agent logic from the serving mechanism.[7]

3.  **Integrate LangSmith for Observability:**
    *   Ensure LangSmith is configured via environment variables.
    *   Assign meaningful names to your graph nodes (`graph.add_node("summarize_web_content",...)`). These names appear directly in the LangSmith traces, making debugging complex, non-deterministic runs significantly easier.[8, 9]

---

### **Phase 6: Evaluation and Testing**

A production system must be rigorously tested.

1.  **Establish an Evaluation Harness (`tests/evaluate.py`):**
    *   Create a script that runs the entire agentic system against a predefined set of questions or tasks.
    *   This script should connect to a LangSmith dataset containing your test cases. Each run will be logged as an experiment, allowing you to track performance regressions or improvements as you modify prompts, models, or logic.[1]

2.  **Write Unit Tests:**
    *   For deterministic components, such as tool data transformations or specific prompt-formatting functions, write traditional unit tests to ensure their correctness in isolation.

By following these phases, you will transform the Jupyter Notebook from a promising prototype into a robust, professional-grade AI library. This structured approach is the standard for building reliable and scalable multi-agent systems.