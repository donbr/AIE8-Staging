#!/usr/bin/env python3
"""Direct audio transcription using faster-whisper.

This script transcribes the GMT20250710-230109_Recording.m4a file directly
using faster-whisper for speed and efficiency.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

def transcribe_audio_file():
    """Transcribe the audio file using faster-whisper."""

    # Check if audio file exists
    audio_file = Path("reference/GMT20250710-230109_Recording.m4a")
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        return False

    print(f"🎵 Found audio file: {audio_file.name} ({audio_file.stat().st_size / (1024*1024):.1f} MB)")

    try:
        from faster_whisper import WhisperModel

        # Initialize model - using 'turbo' for best balance of speed and accuracy
        print("🤖 Loading Whisper turbo model...")
        model = WhisperModel("turbo", device="cpu", compute_type="int8")
        print("✅ Model loaded successfully")

        # Transcribe with optimal settings
        print("🎯 Starting transcription...")
        start_time = datetime.now()

        segments, info = model.transcribe(
            str(audio_file),
            beam_size=5,
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=False,  # Faster processing
            task="transcribe"
        )

        # Collect all segments
        print("📝 Processing transcription segments...")
        transcript_segments = []
        full_text_parts = []

        for segment in segments:
            transcript_segments.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
            full_text_parts.append(segment.text.strip())

        processing_time = (datetime.now() - start_time).total_seconds()

        # Create full transcript
        full_transcript = " ".join(full_text_parts)

        # Display results
        print(f"\n🎉 Transcription completed successfully!")
        print(f"⏱️  Processing time: {processing_time:.1f} seconds")
        print(f"🌐 Language detected: {info.language} ({info.language_probability:.1%} confidence)")
        print(f"⏰ Audio duration: {info.duration:.1f} seconds")
        print(f"📄 Total segments: {len(transcript_segments)}")
        print(f"📝 Total words: ~{len(full_transcript.split())} words")

        # Save transcript to file
        output_file = Path("content/data/GMT20250710-230109_Recording_transcript.txt")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# Audio Transcription\n")
            f.write(f"**File:** {audio_file.name}\n")
            f.write(f"**Duration:** {info.duration:.1f} seconds\n")
            f.write(f"**Language:** {info.language} ({info.language_probability:.1%} confidence)\n")
            f.write(f"**Processing Time:** {processing_time:.1f} seconds\n")
            f.write(f"**Model:** Whisper Turbo (faster-whisper)\n")
            f.write(f"**Transcribed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Full Transcript\n\n")
            f.write(full_transcript)
            f.write("\n\n")

            f.write("## Detailed Segments\n\n")
            for i, segment in enumerate(transcript_segments, 1):
                f.write(f"{i:3d}. [{segment['start']:6.1f}s - {segment['end']:6.1f}s] {segment['text']}\n")

        print(f"💾 Transcript saved to: {output_file}")

        # Display first part of transcript
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
        print("💡 Make sure faster-whisper is installed: uv add faster-whisper")
        return False
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        return False

if __name__ == "__main__":
    success = transcribe_audio_file()
    sys.exit(0 if success else 1)