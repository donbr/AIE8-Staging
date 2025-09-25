#!/usr/bin/env python3
"""Transcribe audio using OpenAI's Whisper API.

This script uses OpenAI's cloud API for fast transcription.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def transcribe_with_openai_api():
    """Transcribe using OpenAI's Whisper API."""

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "test":
        print("❌ OPENAI_API_KEY not set or set to test value")
        print("💡 Set your OpenAI API key: export OPENAI_API_KEY='sk-...'")
        return False

    # Check if audio file exists
    audio_file = Path("reference/GMT20250710-230109_Recording.m4a")
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False

    file_size_mb = audio_file.stat().st_size / (1024*1024)
    print(f"🎵 Found audio file: {audio_file.name} ({file_size_mb:.1f} MB)")

    if file_size_mb > 25:
        print("⚠️  File is larger than OpenAI's 25MB limit")
        print("💡 Consider using the faster-whisper approach or splitting the file")
        return False

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        print("🎯 Starting OpenAI Whisper transcription...")
        start_time = datetime.now()

        with open(audio_file, "rb") as audio:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="verbose_json",
                task="transcribe"
            )

        processing_time = (datetime.now() - start_time).total_seconds()

        # Extract transcript and segments
        full_transcript = response.text
        segments = getattr(response, 'segments', [])

        print(f"\n🎉 Transcription completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"🌐 Language detected: {getattr(response, 'language', 'unknown')}")
        print(f"⏰ Audio duration: {getattr(response, 'duration', 0):.1f} seconds")
        print(f"📄 Total segments: {len(segments)}")
        print(f"📝 Total words: ~{len(full_transcript.split())} words")

        # Save transcript to file
        output_file = Path("content/data/GMT20250710-230109_Recording_transcript_openai.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Audio Transcription (OpenAI Whisper API)\n")
            f.write(f"**File:** {audio_file.name}\n")
            f.write(f"**Duration:** {getattr(response, 'duration', 'unknown')} seconds\n")
            f.write(f"**Language:** {getattr(response, 'language', 'unknown')}\n")
            f.write(f"**Processing Time:** {processing_time:.1f} seconds\n")
            f.write(f"**Model:** OpenAI Whisper-1\n")
            f.write(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Full Transcript\n\n")
            f.write(full_transcript)
            f.write("\n\n")

            if segments:
                f.write("## Detailed Segments\n\n")
                for i, segment in enumerate(segments, 1):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '')
                    f.write(f"{i:3d}. [{start:6.1f}s - {end:6.1f}s] {text}\n")

        print(f"💾 Transcript saved to: {output_file}")

        # Display preview
        preview_length = 500
        preview_text = full_transcript[:preview_length]
        if len(full_transcript) > preview_length:
            preview_text += "..."

        print(f"\n📋 Transcript Preview:")
        print("=" * 60)
        print(preview_text)
        print("=" * 60)

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure openai is installed: uv add openai")
        return False
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return False

if __name__ == "__main__":
    success = transcribe_with_openai_api()
    sys.exit(0 if success else 1)