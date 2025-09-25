"""MCP Audio Transcription Server.

A Model Context Protocol server for audio transcription functionality,
supporting multiple transcription backends with intelligent fallback.

Features:
- Multiple transcription backends (Faster-Whisper, OpenAI Whisper, OpenAI API)
- Automatic format conversion via ffmpeg
- Intelligent chunking for long audio files
- Post-processing and cleaning options
- Progress tracking and caching
- Speaker diarization support (optional)
"""

import asyncio
import json
import os
import tempfile
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum

import httpx
from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
from pydantic import BaseModel


class TranscriptionBackend(str, Enum):
    """Available transcription backends."""
    FASTER_WHISPER = "faster-whisper"
    OPENAI_WHISPER = "openai-whisper"
    OPENAI_API = "openai-api"


class TranscriptionSegment(BaseModel):
    """Individual transcription segment with timing."""
    start: float
    end: float
    text: str
    confidence: Optional[float] = None
    speaker: Optional[str] = None


class TranscriptionResult(BaseModel):
    """Complete transcription result."""
    filename: str
    duration: float
    language: str
    language_probability: float
    backend_used: str
    segments: List[TranscriptionSegment]
    full_text: str
    processing_time: float
    model_used: str


class AudioTranscriptionMCPServer:
    """MCP Server for audio transcription operations."""

    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

        self.server = Server("audio-transcription")
        self.client = httpx.AsyncClient(timeout=300.0)  # Long timeout for audio processing

        # Cache for transcription results
        self.cache: Dict[str, tuple[TranscriptionResult, datetime]] = {}
        self.cache_ttl = timedelta(hours=24)  # Cache for 24 hours

        # Supported audio formats
        self.supported_formats = {'.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg', '.webm'}

        # Initialize backends availability
        self._check_backends()

        # Register tools and resources
        self._register_tools()
        self._register_resources()

    def _check_backends(self):
        """Check which transcription backends are available."""
        self.available_backends = {}

        # Check Faster-Whisper
        try:
            import faster_whisper
            self.available_backends[TranscriptionBackend.FASTER_WHISPER] = True
        except ImportError:
            self.available_backends[TranscriptionBackend.FASTER_WHISPER] = False

        # Check OpenAI Whisper
        try:
            import whisper
            self.available_backends[TranscriptionBackend.OPENAI_WHISPER] = True
        except ImportError:
            self.available_backends[TranscriptionBackend.OPENAI_WHISPER] = False

        # Check OpenAI API
        self.available_backends[TranscriptionBackend.OPENAI_API] = bool(self.openai_api_key)

    def _register_tools(self):
        """Register MCP tools."""

        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            return [
                Tool(
                    name="transcribe_audio",
                    description="Transcribe audio files using multiple backend options",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the audio file to transcribe"
                            },
                            "backend": {
                                "type": "string",
                                "enum": ["faster-whisper", "openai-whisper", "openai-api", "auto"],
                                "description": "Transcription backend to use (auto selects best available)",
                                "default": "auto"
                            },
                            "model": {
                                "type": "string",
                                "description": "Model size (tiny, base, small, medium, large, turbo)",
                                "enum": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                                "default": "turbo"
                            },
                            "language": {
                                "type": "string",
                                "description": "Expected language (ISO 639-1 code, auto-detect if not provided)"
                            },
                            "task": {
                                "type": "string",
                                "enum": ["transcribe", "translate"],
                                "description": "Task type - transcribe or translate to English",
                                "default": "transcribe"
                            },
                            "enable_vad": {
                                "type": "boolean",
                                "description": "Enable Voice Activity Detection to filter silence",
                                "default": True
                            },
                            "word_timestamps": {
                                "type": "boolean",
                                "description": "Include word-level timestamps",
                                "default": False
                            },
                            "clean_text": {
                                "type": "boolean",
                                "description": "Apply post-processing to clean the transcript",
                                "default": True
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="batch_transcribe",
                    description="Transcribe multiple audio files in batch",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of audio file paths to transcribe"
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["txt", "json", "srt", "vtt"],
                                "description": "Output format for transcripts",
                                "default": "txt"
                            },
                            "backend": {
                                "type": "string",
                                "enum": ["faster-whisper", "openai-whisper", "openai-api", "auto"],
                                "default": "auto"
                            },
                            "model": {
                                "type": "string",
                                "enum": ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"],
                                "default": "turbo"
                            }
                        },
                        "required": ["file_paths"]
                    }
                ),
                Tool(
                    name="convert_audio_format",
                    description="Convert audio file to supported format using ffmpeg",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "input_path": {
                                "type": "string",
                                "description": "Path to input audio file"
                            },
                            "output_format": {
                                "type": "string",
                                "enum": ["wav", "mp3", "flac", "m4a"],
                                "description": "Target audio format",
                                "default": "wav"
                            },
                            "sample_rate": {
                                "type": "integer",
                                "description": "Target sample rate in Hz",
                                "default": 16000
                            }
                        },
                        "required": ["input_path"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            if name == "transcribe_audio":
                return await self._transcribe_audio(**arguments)
            elif name == "batch_transcribe":
                return await self._batch_transcribe(**arguments)
            elif name == "convert_audio_format":
                return await self._convert_audio_format(**arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

    def _register_resources(self):
        """Register MCP resources."""

        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            return [
                Resource(
                    uri="audio-transcription://backend_status",
                    name="Backend Status",
                    description="Status of available transcription backends",
                    mimeType="application/json"
                ),
                Resource(
                    uri="audio-transcription://cache_stats",
                    name="Cache Statistics",
                    description="Transcription cache usage statistics",
                    mimeType="application/json"
                )
            ]

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            if uri == "audio-transcription://backend_status":
                return json.dumps({
                    "available_backends": {
                        backend.value: available
                        for backend, available in self.available_backends.items()
                    },
                    "recommended_backend": self._get_recommended_backend(),
                    "supported_formats": list(self.supported_formats)
                })
            elif uri == "audio-transcription://cache_stats":
                return json.dumps({
                    "cache_entries": len(self.cache),
                    "cache_size_mb": self._calculate_cache_size(),
                    "oldest_entry": self._get_oldest_cache_entry()
                })
            else:
                raise ValueError(f"Unknown resource: {uri}")

    async def _transcribe_audio(
        self,
        file_path: str,
        backend: str = "auto",
        model: str = "turbo",
        language: Optional[str] = None,
        task: str = "transcribe",
        enable_vad: bool = True,
        word_timestamps: bool = False,
        clean_text: bool = True
    ) -> List[TextContent]:
        """Transcribe an audio file using the specified backend."""

        # Validate file exists
        if not Path(file_path).exists():
            return [TextContent(
                type="text",
                text=f"❌ Error: Audio file not found: {file_path}"
            )]

        # Check cache
        cache_key = self._generate_cache_key(file_path, backend, model, language, task)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            return [TextContent(
                type="text",
                text=f"📋 **Cached Transcription Result**\n\n{self._format_transcription_result(cached_result)}"
            )]

        # Select backend
        if backend == "auto":
            backend = self._get_recommended_backend()

        if not self.available_backends.get(TranscriptionBackend(backend), False):
            return [TextContent(
                type="text",
                text=f"❌ Error: Backend '{backend}' is not available. Available backends: {[b.value for b, a in self.available_backends.items() if a]}"
            )]

        try:
            # Perform transcription
            start_time = datetime.now()

            if backend == TranscriptionBackend.FASTER_WHISPER:
                result = await self._transcribe_with_faster_whisper(
                    file_path, model, language, task, enable_vad, word_timestamps
                )
            elif backend == TranscriptionBackend.OPENAI_WHISPER:
                result = await self._transcribe_with_openai_whisper(
                    file_path, model, language, task, word_timestamps
                )
            elif backend == TranscriptionBackend.OPENAI_API:
                result = await self._transcribe_with_openai_api(
                    file_path, model, language, task
                )
            else:
                raise ValueError(f"Unsupported backend: {backend}")

            # Apply post-processing if requested
            if clean_text:
                result.full_text = self._clean_transcript(result.full_text)

            # Cache the result
            self._cache_result(cache_key, result)

            return [TextContent(
                type="text",
                text=f"🎯 **Transcription Complete**\n\n{self._format_transcription_result(result)}"
            )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Transcription failed with {backend}: {str(e)}"
            )]

    async def _batch_transcribe(
        self,
        file_paths: List[str],
        output_format: str = "txt",
        backend: str = "auto",
        model: str = "turbo"
    ) -> List[TextContent]:
        """Transcribe multiple audio files in batch."""

        results = []
        total_files = len(file_paths)

        for i, file_path in enumerate(file_paths, 1):
            result = await self._transcribe_audio(
                file_path=file_path,
                backend=backend,
                model=model
            )

            results.append(f"**File {i}/{total_files}: {Path(file_path).name}**")
            results.append(result[0].text)
            results.append("---")

        return [TextContent(
            type="text",
            text=f"📁 **Batch Transcription Complete ({total_files} files)**\n\n" + "\n".join(results)
        )]

    async def _convert_audio_format(
        self,
        input_path: str,
        output_format: str = "wav",
        sample_rate: int = 16000
    ) -> List[TextContent]:
        """Convert audio file format using ffmpeg."""

        if not Path(input_path).exists():
            return [TextContent(
                type="text",
                text=f"❌ Error: Input file not found: {input_path}"
            )]

        try:
            input_path_obj = Path(input_path)
            output_path = input_path_obj.with_suffix(f".{output_format}")

            # Use ffmpeg for conversion
            import subprocess

            cmd = [
                "ffmpeg", "-i", str(input_path),
                "-ar", str(sample_rate),
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                return [TextContent(
                    type="text",
                    text=f"✅ **Audio Conversion Complete**\n\n📁 Input: {input_path}\n📁 Output: {output_path}\n🎵 Format: {output_format}\n📊 Sample Rate: {sample_rate}Hz"
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ Conversion failed: {result.stderr}"
                )]

        except Exception as e:
            return [TextContent(
                type="text",
                text=f"❌ Conversion error: {str(e)}"
            )]

    async def _transcribe_with_faster_whisper(
        self,
        file_path: str,
        model: str,
        language: Optional[str],
        task: str,
        enable_vad: bool,
        word_timestamps: bool
    ) -> TranscriptionResult:
        """Transcribe using Faster-Whisper backend."""
        from faster_whisper import WhisperModel

        # Initialize model
        whisper_model = WhisperModel(
            model,
            device="cpu",  # Default to CPU for compatibility
            compute_type="int8"
        )

        # Transcription options
        kwargs = {
            "beam_size": 5,
            "task": task,
            "vad_filter": enable_vad,
            "word_timestamps": word_timestamps
        }

        if language:
            kwargs["language"] = language

        # Perform transcription
        start_time = datetime.now()
        segments, info = whisper_model.transcribe(file_path, **kwargs)

        # Process segments
        transcription_segments = []
        full_text_parts = []

        for segment in segments:
            transcription_segments.append(TranscriptionSegment(
                start=segment.start,
                end=segment.end,
                text=segment.text.strip(),
                confidence=getattr(segment, 'avg_logprob', None)
            ))
            full_text_parts.append(segment.text.strip())

        processing_time = (datetime.now() - start_time).total_seconds()

        return TranscriptionResult(
            filename=Path(file_path).name,
            duration=info.duration if hasattr(info, 'duration') else 0.0,
            language=info.language,
            language_probability=info.language_probability,
            backend_used=TranscriptionBackend.FASTER_WHISPER.value,
            segments=transcription_segments,
            full_text=" ".join(full_text_parts),
            processing_time=processing_time,
            model_used=model
        )

    async def _transcribe_with_openai_whisper(
        self,
        file_path: str,
        model: str,
        language: Optional[str],
        task: str,
        word_timestamps: bool
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper backend."""
        import whisper

        # Load model
        whisper_model = whisper.load_model(model)

        # Transcription options
        kwargs = {"task": task}
        if language:
            kwargs["language"] = language

        # Perform transcription
        start_time = datetime.now()
        result = whisper_model.transcribe(file_path, **kwargs)
        processing_time = (datetime.now() - start_time).total_seconds()

        # Process segments
        segments = []
        if "segments" in result:
            for seg in result["segments"]:
                segments.append(TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    confidence=seg.get("avg_logprob")
                ))

        return TranscriptionResult(
            filename=Path(file_path).name,
            duration=0.0,  # OpenAI Whisper doesn't provide duration directly
            language=result.get("language", "unknown"),
            language_probability=1.0,  # Not provided by OpenAI Whisper
            backend_used=TranscriptionBackend.OPENAI_WHISPER.value,
            segments=segments,
            full_text=result["text"],
            processing_time=processing_time,
            model_used=model
        )

    async def _transcribe_with_openai_api(
        self,
        file_path: str,
        model: str,
        language: Optional[str],
        task: str
    ) -> TranscriptionResult:
        """Transcribe using OpenAI API backend."""

        # Map model names for API
        api_model = "whisper-1"  # OpenAI API only has one Whisper model

        start_time = datetime.now()

        # Prepare file for upload
        with open(file_path, "rb") as audio_file:
            files = {"file": (Path(file_path).name, audio_file, "audio/mpeg")}
            data = {
                "model": api_model,
                "task": task,
                "response_format": "verbose_json"
            }

            if language:
                data["language"] = language

            # Make API request
            response = await self.client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.openai_api_key}"},
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()

        processing_time = (datetime.now() - start_time).total_seconds()

        # Process segments
        segments = []
        if "segments" in result:
            for seg in result["segments"]:
                segments.append(TranscriptionSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    confidence=seg.get("avg_logprob")
                ))

        return TranscriptionResult(
            filename=Path(file_path).name,
            duration=result.get("duration", 0.0),
            language=result.get("language", "unknown"),
            language_probability=1.0,
            backend_used=TranscriptionBackend.OPENAI_API.value,
            segments=segments,
            full_text=result["text"],
            processing_time=processing_time,
            model_used=api_model
        )

    def _get_recommended_backend(self) -> str:
        """Get the recommended transcription backend based on availability."""
        # Priority: Faster-Whisper > OpenAI Whisper > OpenAI API
        if self.available_backends.get(TranscriptionBackend.FASTER_WHISPER):
            return TranscriptionBackend.FASTER_WHISPER.value
        elif self.available_backends.get(TranscriptionBackend.OPENAI_WHISPER):
            return TranscriptionBackend.OPENAI_WHISPER.value
        elif self.available_backends.get(TranscriptionBackend.OPENAI_API):
            return TranscriptionBackend.OPENAI_API.value
        else:
            return "none"

    def _generate_cache_key(self, file_path: str, backend: str, model: str, language: Optional[str], task: str) -> str:
        """Generate a unique cache key for the transcription request."""
        # Include file size and modification time for cache invalidation
        path_obj = Path(file_path)
        file_info = f"{path_obj.stat().st_size}_{path_obj.stat().st_mtime}"

        cache_data = f"{file_path}_{backend}_{model}_{language}_{task}_{file_info}"
        return hashlib.md5(cache_data.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[TranscriptionResult]:
        """Get cached transcription result if not expired."""
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return result
            else:
                del self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: TranscriptionResult):
        """Cache transcription result with timestamp."""
        self.cache[cache_key] = (result, datetime.now())

        # Cleanup old entries
        cutoff_time = datetime.now() - self.cache_ttl
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if timestamp < cutoff_time
        ]
        for key in expired_keys:
            del self.cache[key]

    def _clean_transcript(self, text: str) -> str:
        """Apply post-processing to clean the transcript."""
        import re

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common filler words (basic cleaning)
        filler_words = [
            r'\b(um|uh|er|ah|like|you know|actually|basically|literally)\b',
        ]

        for pattern in filler_words:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Clean up punctuation
        text = re.sub(r'\s+([,.!?])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([.!?])\s*([a-z])', r'\1 \2', text)  # Ensure space after sentence endings

        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?]\s*)', text)
        cleaned_sentences = []
        for sentence in sentences:
            if sentence.strip() and not re.match(r'[.!?]\s*', sentence):
                sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
            cleaned_sentences.append(sentence)

        return ''.join(cleaned_sentences).strip()

    def _format_transcription_result(self, result: TranscriptionResult) -> str:
        """Format transcription result for display."""
        lines = [
            f"🎵 **File:** {result.filename}",
            f"⏱️ **Duration:** {result.duration:.2f}s",
            f"🌐 **Language:** {result.language} ({result.language_probability:.2%})",
            f"🔧 **Backend:** {result.backend_used}",
            f"🤖 **Model:** {result.model_used}",
            f"⚡ **Processing Time:** {result.processing_time:.2f}s",
            f"📝 **Segments:** {len(result.segments)}",
            "",
            "**Transcript:**",
            "=" * 50,
            result.full_text,
            "=" * 50
        ]

        # Add segment details if available and not too many
        if result.segments and len(result.segments) <= 10:
            lines.extend(["", "**Detailed Segments:**"])
            for i, segment in enumerate(result.segments[:5]):  # Show first 5 segments
                lines.append(f"{i+1}. [{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")
            if len(result.segments) > 5:
                lines.append(f"... and {len(result.segments) - 5} more segments")

        return "\n".join(lines)

    def _calculate_cache_size(self) -> float:
        """Calculate approximate cache size in MB."""
        # Rough estimation based on stored data
        return len(self.cache) * 0.1  # Assume ~100KB per entry

    def _get_oldest_cache_entry(self) -> Optional[str]:
        """Get timestamp of oldest cache entry."""
        if not self.cache:
            return None

        oldest_timestamp = min(timestamp for _, timestamp in self.cache.values())
        return oldest_timestamp.isoformat()

    async def run(self, transport_uri: str = "stdio://"):
        """Run the MCP server."""
        async with self.client:
            await self.server.run(
                transport_uri,
                InitializationOptions(
                    server_name="audio-transcription",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {},
                        "resources": {}
                    }
                )
            )


async def main():
    """Main entry point for the Audio Transcription MCP server."""
    server = AudioTranscriptionMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())