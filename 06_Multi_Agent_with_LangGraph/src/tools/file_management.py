"""File management tools for the Multi-Agent LangGraph system.

Document creation, editing, and reading tools extracted from notebook implementation.
Modernized with proper error handling and configuration management.
"""

from typing import Dict, List, Optional, Any
from typing_extensions import Annotated
from pathlib import Path

from langchain_core.tools import tool


def create_file_management_tools(working_directory: Path) -> List[Any]:
    """Create file management tools for the specified working directory.

    Args:
        working_directory: Path to the working directory

    Returns:
        List of file management tools
    """
    # Ensure working directory exists
    working_directory = Path(working_directory)
    working_directory.mkdir(parents=True, exist_ok=True)

    @tool
    def create_outline(
        points: Annotated[List[str], "List of main points or sections."],
        file_name: Annotated[str, "File path to save the outline."],
    ) -> Annotated[str, "Path of the saved outline file."]:
        """Create and save an outline."""
        try:
            file_path = working_directory / file_name
            with file_path.open("w", encoding="utf-8") as file:
                for i, point in enumerate(points, 1):
                    file.write(f"{i}. {point}\n")
            return f"Outline saved to {file_path}"
        except Exception as e:
            return f"Error creating outline: {str(e)}"

    @tool
    def write_document(
        content: Annotated[str, "Text content to be written into the document."],
        file_name: Annotated[str, "File path to save the document."],
    ) -> Annotated[str, "Path of the saved document file."]:
        """Create and save a text document."""
        try:
            file_path = working_directory / file_name
            with file_path.open("w", encoding="utf-8") as file:
                file.write(content)
            return f"Document saved to {file_path}"
        except Exception as e:
            return f"Error writing document: {str(e)}"

    @tool
    def read_document(
        file_name: Annotated[str, "File path to read the document."],
        start: Annotated[Optional[int], "The start line. Default is 0"] = None,
        end: Annotated[Optional[int], "The end line. Default is None"] = None,
    ) -> str:
        """Read the specified document."""
        try:
            file_path = working_directory / file_name
            if not file_path.exists():
                return f"File {file_path} does not exist."

            with file_path.open("r", encoding="utf-8") as file:
                lines = file.readlines()

            # Handle line range selection
            if start is not None or end is not None:
                start = start or 0
                end = end or len(lines)
                lines = lines[start:end]

            return "".join(lines).strip()
        except Exception as e:
            return f"Error reading document: {str(e)}"

    @tool
    def edit_document(
        file_name: Annotated[str, "Path of the document to be edited."],
        inserts: Annotated[
            Dict[int, str],
            "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
        ] = {},
    ) -> Annotated[str, "Path of the edited document file."]:
        """Edit a document by inserting text at specified line numbers."""
        try:
            file_path = working_directory / file_name
            if not file_path.exists():
                return f"File {file_path} does not exist."

            # Read existing content
            with file_path.open("r", encoding="utf-8") as file:
                lines = file.readlines()

            # Apply inserts in reverse order to maintain line numbering
            for line_num in sorted(inserts.keys(), reverse=True):
                insert_text = inserts[line_num]
                if not insert_text.endswith("\n"):
                    insert_text += "\n"

                # Convert to 0-based indexing
                index = line_num - 1
                if 0 <= index <= len(lines):
                    lines.insert(index, insert_text)

            # Write back to file
            with file_path.open("w", encoding="utf-8") as file:
                file.writelines(lines)

            return f"Document edited and saved to {file_path}"
        except Exception as e:
            return f"Error editing document: {str(e)}"

    return [create_outline, write_document, read_document, edit_document]


def create_rag_tool(retriever, tool_name: str = "how_people_use_ai_retriever"):
    """Create a RAG tool for document retrieval.

    Args:
        retriever: Vector store retriever instance
        tool_name: Name for the RAG tool

    Returns:
        RAG retrieval tool
    """
    @tool(name=tool_name)
    def retrieve_documents(
        query: Annotated[str, "Query to search for in the document collection"]
    ) -> str:
        """Retrieve relevant documents based on the query."""
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant documents found."

            # Format the retrieved documents
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
                formatted_docs.append(f"Document {i}: {content}")

            return "\n\n".join(formatted_docs)
        except Exception as e:
            return f"Error retrieving documents: {str(e)}"

    return retrieve_documents


def create_reference_tool(working_directory: Path):
    """Create a tool for referencing previous responses.

    Args:
        working_directory: Path to working directory

    Returns:
        Reference tool
    """
    @tool
    def reference_previous_responses(
        query: Annotated[str, "Query to search for in previous responses"]
    ) -> str:
        """Reference previous responses and work done in the working directory."""
        try:
            working_dir = Path(working_directory)
            if not working_dir.exists():
                return "Working directory does not exist."

            # Find all text files in working directory
            text_files = list(working_dir.glob("*.txt")) + list(working_dir.glob("*.md"))

            if not text_files:
                return "No previous responses found in working directory."

            # Simple keyword search across files
            results = []
            query_lower = query.lower()

            for file_path in text_files:
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        content = f.read()

                    if query_lower in content.lower():
                        # Extract relevant context (first 300 chars containing the query)
                        content_lower = content.lower()
                        start_idx = max(0, content_lower.find(query_lower) - 150)
                        end_idx = min(len(content), start_idx + 300)
                        context = content[start_idx:end_idx]

                        results.append(f"From {file_path.name}: ...{context}...")
                except Exception:
                    continue

            return "\n\n".join(results) if results else f"No references found for '{query}' in previous responses."

        except Exception as e:
            return f"Error referencing previous responses: {str(e)}"

    return reference_previous_responses


def get_file_tools_info():
    """Get information about available file management tools."""
    return {
        "create_outline": {
            "description": "Create structured outlines from list of points",
            "inputs": ["points: List[str]", "file_name: str"],
            "output": "File path confirmation"
        },
        "write_document": {
            "description": "Write content to a new document file",
            "inputs": ["content: str", "file_name: str"],
            "output": "File path confirmation"
        },
        "read_document": {
            "description": "Read existing document with optional line range",
            "inputs": ["file_name: str", "start: Optional[int]", "end: Optional[int]"],
            "output": "Document content"
        },
        "edit_document": {
            "description": "Edit document by inserting text at specific lines",
            "inputs": ["file_name: str", "inserts: Dict[int, str]"],
            "output": "Edit confirmation"
        },
        "rag_tool": {
            "description": "Retrieve relevant documents from vector store",
            "inputs": ["query: str"],
            "output": "Formatted retrieved documents"
        },
        "reference_tool": {
            "description": "Reference previous responses and work",
            "inputs": ["query: str"],
            "output": "Previous work context"
        }
    }


# Backward compatibility functions
def create_notebook_compatible_file_tools(working_directory_path: str):
    """Create file tools that match notebook implementation.

    Args:
        working_directory_path: String path to working directory

    Returns:
        Tuple of (file_tools, reference_tool)
    """
    working_dir = Path(working_directory_path)
    file_tools = create_file_management_tools(working_dir)
    reference_tool = create_reference_tool(working_dir)

    return file_tools, reference_tool