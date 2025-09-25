"""Audio Transcription Workflow Graph.

A specialized LangGraph workflow for audio transcription that combines:
- Audio preprocessing and format conversion
- Multi-backend transcription with intelligent fallback
- Post-processing and cleaning using LLM agents
- Integration with the existing research and writing teams
"""

from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from ..agents.state import MessagesState
from ..config.configuration import Configuration
from ..core.factory import create_llm_factory


class AudioTranscriptionState(MessagesState):
    """State for audio transcription workflow."""

    # Input configuration
    audio_file_paths: List[str] = []
    transcription_config: Dict[str, Any] = {}

    # Processing results
    transcription_results: List[Dict[str, Any]] = []
    processed_transcripts: List[Dict[str, Any]] = []
    final_documents: List[Dict[str, Any]] = []

    # Workflow control
    current_file_index: int = 0
    processing_errors: List[str] = []
    should_continue: bool = True


async def audio_preprocessor_node(state: AudioTranscriptionState) -> AudioTranscriptionState:
    """Preprocess audio files for transcription.

    Handles format conversion, file size validation, and chunking if needed.
    """
    from ..mcp_integration import get_mcp_audio_transcription_tool

    # Get configuration
    config = Configuration()
    transcription_tool = await get_mcp_audio_transcription_tool(config)

    processed_files = []
    errors = []

    for file_path in state.audio_file_paths:
        try:
            # Validate file exists
            if not Path(file_path).exists():
                errors.append(f"Audio file not found: {file_path}")
                continue

            # Check file size
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            if file_size_mb > config.max_audio_file_size_mb:
                errors.append(f"File too large ({file_size_mb:.1f}MB): {file_path}")
                continue

            # Convert format if needed
            file_suffix = Path(file_path).suffix.lower()
            supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.webm'}

            if file_suffix not in supported_formats:
                # Use conversion tool
                convert_result = await transcription_tool.invoke({
                    "tool_name": "convert_audio_format",
                    "arguments": {
                        "input_path": file_path,
                        "output_format": "wav",
                        "sample_rate": 16000
                    }
                })

                if "Error" in convert_result:
                    errors.append(f"Conversion failed for {file_path}: {convert_result}")
                    continue

                # Update file path to converted file
                converted_path = str(Path(file_path).with_suffix('.wav'))
                processed_files.append(converted_path)
            else:
                processed_files.append(file_path)

        except Exception as e:
            errors.append(f"Preprocessing error for {file_path}: {str(e)}")

    # Update state
    state.audio_file_paths = processed_files
    state.processing_errors.extend(errors)

    # Add status message
    if processed_files:
        status_msg = f"✅ Preprocessed {len(processed_files)} audio files for transcription"
        if errors:
            status_msg += f" ({len(errors)} errors encountered)"
        state.messages.append(AIMessage(content=status_msg))
    else:
        state.messages.append(AIMessage(content="❌ No valid audio files to process"))
        state.should_continue = False

    return state


async def transcription_node(state: AudioTranscriptionState) -> AudioTranscriptionState:
    """Transcribe audio files using the configured backend."""
    from ..mcp_integration import get_mcp_audio_transcription_tool

    config = Configuration()
    transcription_tool = await get_mcp_audio_transcription_tool(config)

    results = []
    errors = []

    # Get transcription configuration
    transcription_config = state.transcription_config or config.get_transcription_config()

    for i, file_path in enumerate(state.audio_file_paths):
        try:
            state.messages.append(AIMessage(
                content=f"🎯 Starting transcription of file {i+1}/{len(state.audio_file_paths)}: {Path(file_path).name}"
            ))

            # Perform transcription
            result = await transcription_tool.invoke({
                "tool_name": "transcribe_audio",
                "arguments": {
                    "file_path": file_path,
                    "backend": transcription_config.get("backend", "auto"),
                    "model": transcription_config.get("model", "turbo"),
                    "language": transcription_config.get("language"),
                    "task": transcription_config.get("task", "transcribe"),
                    "enable_vad": transcription_config.get("enable_vad", True),
                    "word_timestamps": transcription_config.get("enable_word_timestamps", False),
                    "clean_text": transcription_config.get("clean_transcript", True)
                }
            })

            if "Error" in result:
                errors.append(f"Transcription failed for {file_path}: {result}")
                continue

            # Parse result and store
            results.append({
                "file_path": file_path,
                "filename": Path(file_path).name,
                "transcription": result,
                "config_used": transcription_config.copy()
            })

            state.messages.append(AIMessage(
                content=f"✅ Completed transcription of {Path(file_path).name}"
            ))

        except Exception as e:
            error_msg = f"Transcription error for {file_path}: {str(e)}"
            errors.append(error_msg)
            state.messages.append(AIMessage(content=f"❌ {error_msg}"))

    # Update state
    state.transcription_results = results
    state.processing_errors.extend(errors)

    if results:
        state.messages.append(AIMessage(
            content=f"🎉 Completed transcription of {len(results)} audio files"
        ))
    else:
        state.messages.append(AIMessage(
            content="❌ No successful transcriptions completed"
        ))
        state.should_continue = False

    return state


async def transcript_processor_node(state: AudioTranscriptionState) -> AudioTranscriptionState:
    """Process and clean transcripts using LLM agents.

    Uses the existing research and writing agents to:
    - Analyze transcript content for key topics
    - Clean and structure the text
    - Apply consistent formatting
    """
    config = Configuration()
    llm_factory = create_llm_factory(config)

    # Create specialized LLM for transcript processing
    processing_llm = llm_factory.create_llm(
        model_type="writing",  # Use writing model for text processing
        temperature=0.3,       # Moderate creativity for text improvement
        max_tokens=config.get_model_max_tokens("writing")
    )

    processed_transcripts = []

    for result in state.transcription_results:
        try:
            raw_transcript = result["transcription"]
            filename = result["filename"]

            # Create processing prompt
            processing_prompt = f"""You are an expert transcript editor. Your task is to clean and improve this audio transcription while preserving the original meaning and content.

Please perform the following improvements:

1. **Structure and Formatting:**
   - Organize content into logical paragraphs
   - Add appropriate punctuation and capitalization
   - Remove excessive repetition and filler words

2. **Content Enhancement:**
   - Identify and highlight key topics, decisions, or action items
   - Maintain speaker intent and meaning
   - Preserve technical terms and proper nouns accurately

3. **Readability:**
   - Improve sentence flow and clarity
   - Fix grammatical errors
   - Ensure consistent formatting

4. **Summary:**
   - Provide a brief summary of main topics discussed
   - List any action items or key decisions mentioned

**Original Transcript from {filename}:**
{raw_transcript}

**Please provide your improved version:**"""

            # Process with LLM
            response = await processing_llm.ainvoke([HumanMessage(content=processing_prompt)])

            processed_transcripts.append({
                "filename": filename,
                "file_path": result["file_path"],
                "raw_transcript": raw_transcript,
                "processed_transcript": response.content,
                "processing_config": result["config_used"]
            })

            state.messages.append(AIMessage(
                content=f"📝 Processed and cleaned transcript for {filename}"
            ))

        except Exception as e:
            error_msg = f"Transcript processing error for {result['filename']}: {str(e)}"
            state.processing_errors.append(error_msg)
            state.messages.append(AIMessage(content=f"❌ {error_msg}"))

    state.processed_transcripts = processed_transcripts

    if processed_transcripts:
        state.messages.append(AIMessage(
            content=f"✨ Successfully processed {len(processed_transcripts)} transcripts"
        ))
    else:
        state.messages.append(AIMessage(
            content="❌ No transcripts were successfully processed"
        ))

    return state


async def document_generator_node(state: AudioTranscriptionState) -> AudioTranscriptionState:
    """Generate final documents from processed transcripts.

    Creates structured documents with:
    - Executive summary
    - Full transcript
    - Key topics and insights
    - Action items (if any)
    """
    config = Configuration()
    llm_factory = create_llm_factory(config)

    # Use writing model for document generation
    writing_llm = llm_factory.create_llm(
        model_type="writing",
        temperature=0.3,
        max_tokens=config.get_model_max_tokens("writing")
    )

    final_documents = []

    for transcript in state.processed_transcripts:
        try:
            # Create document generation prompt
            doc_prompt = f"""You are a professional document writer. Create a comprehensive, well-structured document based on this processed transcript.

**Document Structure Required:**

# Transcript Analysis: {transcript['filename']}

## Executive Summary
[Provide a concise 2-3 sentence summary of the main content]

## Key Topics
[List and briefly describe the main topics covered]

## Full Transcript
[Include the complete processed transcript with clear formatting]

## Insights and Analysis
[Provide analysis of important points, themes, or patterns]

## Action Items
[List any action items, decisions, or next steps mentioned]

## Technical Details
[Note any technical information, tools, or methodologies discussed]

---

**Processed Transcript Content:**
{transcript['processed_transcript']}

Please create the structured document following the format above:"""

            # Generate document
            response = await writing_llm.ainvoke([HumanMessage(content=doc_prompt)])

            final_documents.append({
                "filename": transcript['filename'],
                "file_path": transcript['file_path'],
                "document_content": response.content,
                "raw_transcript": transcript['raw_transcript'],
                "processed_transcript": transcript['processed_transcript'],
                "generation_config": transcript['processing_config']
            })

            state.messages.append(AIMessage(
                content=f"📄 Generated structured document for {transcript['filename']}"
            ))

        except Exception as e:
            error_msg = f"Document generation error for {transcript['filename']}: {str(e)}"
            state.processing_errors.append(error_msg)
            state.messages.append(AIMessage(content=f"❌ {error_msg}"))

    state.final_documents = final_documents

    if final_documents:
        state.messages.append(AIMessage(
            content=f"📚 Generated {len(final_documents)} structured documents from transcripts"
        ))
    else:
        state.messages.append(AIMessage(
            content="❌ No documents were successfully generated"
        ))

    return state


def should_continue_processing(state: AudioTranscriptionState) -> str:
    """Determine if processing should continue based on current state."""
    if not state.should_continue:
        return END

    # Check if we have files to process
    if not state.audio_file_paths:
        return END

    # If we have errors but also some results, continue
    if state.transcription_results or state.processed_transcripts:
        return "continue"

    return END


def create_transcription_workflow_graph(config: Configuration) -> StateGraph:
    """Create the complete audio transcription workflow graph.

    Args:
        config: Configuration instance with transcription settings

    Returns:
        Compiled LangGraph StateGraph for audio transcription workflow
    """

    # Create workflow graph
    workflow = StateGraph(AudioTranscriptionState)

    # Add nodes
    workflow.add_node("preprocessor", audio_preprocessor_node)
    workflow.add_node("transcriber", transcription_node)
    workflow.add_node("processor", transcript_processor_node)
    workflow.add_node("generator", document_generator_node)

    # Define workflow edges
    workflow.add_edge("__start__", "preprocessor")
    workflow.add_conditional_edges(
        "preprocessor",
        should_continue_processing,
        {
            "continue": "transcriber",
            END: END
        }
    )
    workflow.add_conditional_edges(
        "transcriber",
        should_continue_processing,
        {
            "continue": "processor",
            END: END
        }
    )
    workflow.add_edge("processor", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()


def create_batch_transcription_graph(config: Configuration) -> StateGraph:
    """Create a specialized graph for batch transcription of multiple files.

    Args:
        config: Configuration instance

    Returns:
        Compiled graph optimized for batch processing
    """
    return create_transcription_workflow_graph(config)


def create_simple_transcription_graph(config: Configuration) -> StateGraph:
    """Create a simplified transcription graph for basic use cases.

    Args:
        config: Configuration instance

    Returns:
        Compiled graph with basic transcription only (no LLM processing)
    """
    workflow = StateGraph(AudioTranscriptionState)

    # Add only essential nodes
    workflow.add_node("preprocessor", audio_preprocessor_node)
    workflow.add_node("transcriber", transcription_node)

    # Simple workflow
    workflow.add_edge("__start__", "preprocessor")
    workflow.add_conditional_edges(
        "preprocessor",
        should_continue_processing,
        {
            "continue": "transcriber",
            END: END
        }
    )
    workflow.add_edge("transcriber", END)

    return workflow.compile()


# Convenience functions for different transcription workflows
async def transcribe_single_file(
    file_path: str,
    config: Optional[Configuration] = None,
    transcription_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transcribe a single audio file using the full workflow.

    Args:
        file_path: Path to the audio file
        config: Configuration instance (optional)
        transcription_config: Custom transcription settings (optional)

    Returns:
        Dictionary with transcription results and processed documents
    """
    if config is None:
        config = Configuration()

    graph = create_transcription_workflow_graph(config)

    initial_state = AudioTranscriptionState(
        messages=[HumanMessage(content=f"Transcribing audio file: {file_path}")],
        audio_file_paths=[file_path],
        transcription_config=transcription_config or {}
    )

    result = await graph.ainvoke(initial_state)

    return {
        "transcription_results": result.transcription_results,
        "processed_transcripts": result.processed_transcripts,
        "final_documents": result.final_documents,
        "processing_errors": result.processing_errors,
        "messages": [msg.content for msg in result.messages if isinstance(msg, AIMessage)]
    }


async def transcribe_batch_files(
    file_paths: List[str],
    config: Optional[Configuration] = None,
    transcription_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Transcribe multiple audio files in batch.

    Args:
        file_paths: List of audio file paths
        config: Configuration instance (optional)
        transcription_config: Custom transcription settings (optional)

    Returns:
        Dictionary with batch transcription results
    """
    if config is None:
        config = Configuration()

    graph = create_batch_transcription_graph(config)

    initial_state = AudioTranscriptionState(
        messages=[HumanMessage(content=f"Batch transcribing {len(file_paths)} audio files")],
        audio_file_paths=file_paths,
        transcription_config=transcription_config or {}
    )

    result = await graph.ainvoke(initial_state)

    return {
        "transcription_results": result.transcription_results,
        "processed_transcripts": result.processed_transcripts,
        "final_documents": result.final_documents,
        "processing_errors": result.processing_errors,
        "total_files": len(file_paths),
        "successful_transcriptions": len(result.transcription_results),
        "messages": [msg.content for msg in result.messages if isinstance(msg, AIMessage)]
    }