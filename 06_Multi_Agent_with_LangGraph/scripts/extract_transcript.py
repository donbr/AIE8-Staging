#!/usr/bin/env python3
"""
Robust Audio Transcript Extraction Script

This script implements multiple fallback strategies to successfully transcribe
the GMT20250710-230109_Recording.m4a file, addressing WSL2 GPU detection issues
and large file processing challenges.

Usage:
    python extract_transcript.py

The script will:
1. Try faster-whisper with CPU-only optimizations
2. Fallback to openai-whisper if faster-whisper fails
3. Fallback to OpenAI API with file chunking if local methods fail
4. Generate a clean, formatted transcript with timestamps
"""

import os
import sys
import subprocess
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('transcript_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TranscriptExtractor:
    def __init__(self, audio_file: Path, output_dir: Path = None):
        self.audio_file = audio_file
        self.output_dir = output_dir or Path("content/data")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure CPU-only processing
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NVIDIA_VISIBLE_DEVICES"] = ""

    def check_dependencies(self) -> Dict[str, bool]:
        """Check availability of transcription dependencies."""
        deps = {}

        # Check faster-whisper
        try:
            import faster_whisper
            deps['faster_whisper'] = True
            logger.info("✅ faster-whisper available")
        except ImportError:
            deps['faster_whisper'] = False
            logger.warning("❌ faster-whisper not available")

        # Check openai-whisper
        try:
            import whisper
            deps['openai_whisper'] = True
            logger.info("✅ openai-whisper available")
        except ImportError:
            deps['openai_whisper'] = False
            logger.warning("❌ openai-whisper not available")

        # Check OpenAI API
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            deps['openai_api'] = bool(api_key and api_key != "test")
            if deps['openai_api']:
                logger.info("✅ OpenAI API available")
            else:
                logger.warning("❌ OpenAI API key not set")
        except ImportError:
            deps['openai_api'] = False
            logger.warning("❌ openai library not available")

        # Check ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                 capture_output=True, text=True)
            deps['ffmpeg'] = result.returncode == 0
            if deps['ffmpeg']:
                logger.info("✅ ffmpeg available")
            else:
                logger.warning("❌ ffmpeg not available")
        except FileNotFoundError:
            deps['ffmpeg'] = False
            logger.warning("❌ ffmpeg not found")

        return deps

    def analyze_audio_file(self) -> Dict[str, Any]:
        """Analyze the audio file properties."""
        if not self.audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {self.audio_file}")

        file_size_mb = self.audio_file.stat().st_size / (1024 * 1024)

        # Try to get duration using ffprobe
        duration = None
        try:
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(self.audio_file)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                duration = float(result.stdout.strip())
        except (subprocess.SubprocessError, ValueError, FileNotFoundError):
            logger.warning("Could not determine audio duration")

        info = {
            'file_path': str(self.audio_file),
            'size_mb': file_size_mb,
            'duration_seconds': duration,
            'duration_minutes': duration / 60 if duration else None,
            'format': self.audio_file.suffix
        }

        logger.info(f"📊 Audio file analysis:")
        logger.info(f"   Size: {file_size_mb:.1f} MB")
        logger.info(f"   Duration: {info['duration_minutes']:.1f} minutes" if duration else "   Duration: Unknown")
        logger.info(f"   Format: {info['format']}")

        return info

    def compress_audio(self, target_size_mb: float = 20.0) -> Optional[Path]:
        """Compress audio file to target size using ffmpeg."""
        if not self.check_dependencies().get('ffmpeg', False):
            logger.warning("ffmpeg not available, skipping compression")
            return None

        try:
            compressed_file = self.output_dir / f"compressed_{self.audio_file.name}"

            # Calculate target bitrate based on duration and target size
            audio_info = self.analyze_audio_file()
            duration = audio_info.get('duration_seconds')

            if not duration:
                # Use conservative bitrate
                bitrate = "32k"
            else:
                # Calculate bitrate to achieve target size
                target_bits = target_size_mb * 8 * 1024 * 1024  # Convert MB to bits
                target_bitrate = int(target_bits / duration)
                bitrate = f"{min(target_bitrate, 128000)}k"  # Cap at 128k

            logger.info(f"🎵 Compressing audio to ~{target_size_mb}MB (bitrate: {bitrate})")

            cmd = [
                'ffmpeg', '-y', '-i', str(self.audio_file),
                '-codec:a', 'aac', '-b:a', bitrate,
                '-ac', '1',  # Convert to mono
                '-ar', '16000',  # 16kHz sample rate
                str(compressed_file)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and compressed_file.exists():
                new_size = compressed_file.stat().st_size / (1024 * 1024)
                logger.info(f"✅ Compression successful: {new_size:.1f}MB")
                return compressed_file
            else:
                logger.error(f"❌ Compression failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error("❌ Compression timeout")
            return None
        except Exception as e:
            logger.error(f"❌ Compression error: {e}")
            return None

    def split_audio_for_openai(self, chunk_duration_minutes: int = 10) -> List[Path]:
        """Split audio into chunks suitable for OpenAI API (under 25MB each)."""
        if not self.check_dependencies().get('ffmpeg', False):
            logger.warning("ffmpeg not available, cannot split audio")
            return []

        try:
            audio_info = self.analyze_audio_file()
            total_duration = audio_info.get('duration_seconds')

            if not total_duration:
                logger.error("Cannot split audio without duration information")
                return []

            chunk_duration_seconds = chunk_duration_minutes * 60
            num_chunks = int(total_duration / chunk_duration_seconds) + 1

            logger.info(f"🔪 Splitting audio into {num_chunks} chunks of {chunk_duration_minutes} minutes each")

            chunk_files = []
            for i in range(num_chunks):
                start_time = i * chunk_duration_seconds
                chunk_file = self.output_dir / f"chunk_{i:03d}_{self.audio_file.stem}.wav"

                cmd = [
                    'ffmpeg', '-y', '-i', str(self.audio_file),
                    '-ss', str(start_time),
                    '-t', str(chunk_duration_seconds),
                    '-codec:a', 'pcm_s16le',  # Uncompressed WAV for quality
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',  # Mono
                    str(chunk_file)
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

                if result.returncode == 0 and chunk_file.exists():
                    chunk_size = chunk_file.stat().st_size / (1024 * 1024)
                    if chunk_size > 0.1:  # Skip tiny files
                        chunk_files.append(chunk_file)
                        logger.info(f"   ✅ Chunk {i+1}: {chunk_size:.1f}MB")
                    else:
                        chunk_file.unlink()  # Remove tiny file

            return chunk_files

        except Exception as e:
            logger.error(f"❌ Audio splitting error: {e}")
            return []

    def transcribe_with_faster_whisper(self, audio_file: Path, timeout_minutes: int = 15) -> Optional[Dict[str, Any]]:
        """Transcribe using faster-whisper with optimized CPU settings."""
        try:
            from faster_whisper import WhisperModel

            logger.info("🚀 Attempting transcription with faster-whisper")

            # Try different models in order of preference
            models_to_try = ["base", "small", "medium"]

            for model_name in models_to_try:
                try:
                    logger.info(f"   Loading {model_name} model...")

                    # Force CPU-only with explicit parameters
                    model = WhisperModel(
                        model_name,
                        device="cpu",
                        device_index=0,
                        compute_type="int8",
                        cpu_threads=4,
                        num_workers=1
                    )

                    logger.info(f"   ✅ Model loaded, starting transcription...")
                    start_time = time.time()

                    segments, info = model.transcribe(
                        str(audio_file),
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        word_timestamps=False,
                        task="transcribe",
                        temperature=0.0
                    )

                    # Collect segments with timeout protection
                    transcript_segments = []
                    full_text_parts = []

                    segment_start_time = time.time()
                    for segment in segments:
                        # Check for timeout
                        if time.time() - segment_start_time > timeout_minutes * 60:
                            logger.warning(f"⏰ Segment processing timeout after {timeout_minutes} minutes")
                            break

                        transcript_segments.append({
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text.strip()
                        })
                        full_text_parts.append(segment.text.strip())

                    processing_time = time.time() - start_time
                    full_transcript = " ".join(full_text_parts)

                    result = {
                        "backend": "faster-whisper",
                        "model": model_name,
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "duration": info.duration,
                        "segments": transcript_segments,
                        "full_text": full_transcript,
                        "processing_time": processing_time,
                        "word_count": len(full_transcript.split())
                    }

                    logger.info(f"✅ faster-whisper transcription successful!")
                    logger.info(f"   Model: {model_name}")
                    logger.info(f"   Processing time: {processing_time:.1f}s")
                    logger.info(f"   Language: {info.language} ({info.language_probability:.1%})")
                    logger.info(f"   Segments: {len(transcript_segments)}")
                    logger.info(f"   Word count: {len(full_transcript.split())}")

                    return result

                except Exception as model_error:
                    logger.warning(f"   ❌ Model {model_name} failed: {model_error}")
                    continue

            logger.error("❌ All faster-whisper models failed")
            return None

        except ImportError:
            logger.error("❌ faster-whisper not available")
            return None
        except Exception as e:
            logger.error(f"❌ faster-whisper error: {e}")
            return None

    def transcribe_with_openai_whisper(self, audio_file: Path) -> Optional[Dict[str, Any]]:
        """Transcribe using OpenAI's whisper library."""
        try:
            import whisper

            logger.info("🚀 Attempting transcription with openai-whisper")

            # Try smaller models first
            models_to_try = ["base", "small"]

            for model_name in models_to_try:
                try:
                    logger.info(f"   Loading {model_name} model...")
                    model = whisper.load_model(model_name)

                    logger.info("   Starting transcription...")
                    start_time = time.time()

                    result = model.transcribe(str(audio_file))

                    processing_time = time.time() - start_time

                    # Convert to our standard format
                    segments = []
                    if "segments" in result:
                        for seg in result["segments"]:
                            segments.append({
                                "start": seg.get("start", 0),
                                "end": seg.get("end", 0),
                                "text": seg.get("text", "").strip()
                            })

                    transcript_result = {
                        "backend": "openai-whisper",
                        "model": model_name,
                        "language": result.get("language", "unknown"),
                        "language_probability": 1.0,  # OpenAI whisper doesn't provide this
                        "duration": max([seg["end"] for seg in segments]) if segments else 0,
                        "segments": segments,
                        "full_text": result.get("text", ""),
                        "processing_time": processing_time,
                        "word_count": len(result.get("text", "").split())
                    }

                    logger.info(f"✅ openai-whisper transcription successful!")
                    logger.info(f"   Model: {model_name}")
                    logger.info(f"   Processing time: {processing_time:.1f}s")
                    logger.info(f"   Language: {transcript_result['language']}")
                    logger.info(f"   Word count: {transcript_result['word_count']}")

                    return transcript_result

                except Exception as model_error:
                    logger.warning(f"   ❌ Model {model_name} failed: {model_error}")
                    continue

            logger.error("❌ All openai-whisper models failed")
            return None

        except ImportError:
            logger.error("❌ openai-whisper not available")
            return None
        except Exception as e:
            logger.error(f"❌ openai-whisper error: {e}")
            return None

    def transcribe_with_openai_api(self, audio_files: List[Path]) -> Optional[Dict[str, Any]]:
        """Transcribe using OpenAI API with multiple files."""
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or api_key == "test":
                logger.error("❌ OpenAI API key not set")
                return None

            client = OpenAI(api_key=api_key)
            logger.info("🚀 Attempting transcription with OpenAI API")

            all_segments = []
            all_text_parts = []
            total_processing_time = 0

            for i, audio_file in enumerate(audio_files):
                file_size_mb = audio_file.stat().st_size / (1024 * 1024)

                if file_size_mb > 25:
                    logger.warning(f"   ⚠️ File {i+1} is {file_size_mb:.1f}MB, skipping (>25MB limit)")
                    continue

                logger.info(f"   Processing chunk {i+1}/{len(audio_files)} ({file_size_mb:.1f}MB)")

                try:
                    start_time = time.time()

                    with open(audio_file, "rb") as audio:
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio,
                            response_format="verbose_json",
                            task="transcribe"
                        )

                    chunk_processing_time = time.time() - start_time
                    total_processing_time += chunk_processing_time

                    # Adjust segment timestamps for chunk offset
                    chunk_start_offset = i * 10 * 60  # Assuming 10-minute chunks

                    if hasattr(response, 'segments') and response.segments:
                        for segment in response.segments:
                            all_segments.append({
                                "start": segment.get("start", 0) + chunk_start_offset,
                                "end": segment.get("end", 0) + chunk_start_offset,
                                "text": segment.get("text", "").strip()
                            })

                    all_text_parts.append(response.text)
                    logger.info(f"   ✅ Chunk {i+1} completed ({chunk_processing_time:.1f}s)")

                except Exception as chunk_error:
                    logger.error(f"   ❌ Chunk {i+1} failed: {chunk_error}")
                    continue

            if not all_text_parts:
                logger.error("❌ No chunks were successfully transcribed")
                return None

            full_transcript = " ".join(all_text_parts)

            result = {
                "backend": "openai-api",
                "model": "whisper-1",
                "language": getattr(response, 'language', 'unknown'),
                "language_probability": 1.0,
                "duration": max([seg["end"] for seg in all_segments]) if all_segments else 0,
                "segments": all_segments,
                "full_text": full_transcript,
                "processing_time": total_processing_time,
                "word_count": len(full_transcript.split()),
                "chunks_processed": len([t for t in all_text_parts if t.strip()])
            }

            logger.info(f"✅ OpenAI API transcription successful!")
            logger.info(f"   Chunks processed: {result['chunks_processed']}/{len(audio_files)}")
            logger.info(f"   Total processing time: {total_processing_time:.1f}s")
            logger.info(f"   Language: {result['language']}")
            logger.info(f"   Word count: {result['word_count']}")

            return result

        except ImportError:
            logger.error("❌ openai library not available")
            return None
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return None

    def clean_and_format_transcript(self, transcript: str) -> str:
        """Clean and format the raw transcript."""
        import re

        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', transcript.strip())

        # Fix common transcription issues
        cleaned = re.sub(r'\buh+\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bum+\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\ber+\b', '', cleaned, flags=re.IGNORECASE)

        # Fix sentence spacing
        cleaned = re.sub(r'\s*\.\s*', '. ', cleaned)
        cleaned = re.sub(r'\s*\?\s*', '? ', cleaned)
        cleaned = re.sub(r'\s*!\s*', '! ', cleaned)

        # Capitalize sentences
        sentences = re.split(r'[.!?]+', cleaned)
        formatted_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                formatted_sentences.append(sentence)

        return '. '.join(formatted_sentences) + '.' if formatted_sentences else ''

    def save_transcript(self, result: Dict[str, Any]) -> Path:
        """Save the transcript to a formatted file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = self.output_dir / f"transcript_{timestamp}_{result['backend'].replace('-', '_')}.txt"

        # Clean and format the transcript
        cleaned_text = self.clean_and_format_transcript(result['full_text'])

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Audio Transcription Report\n\n")
            f.write(f"**Source File:** {self.audio_file.name}\n")
            f.write(f"**Backend:** {result['backend']}\n")
            f.write(f"**Model:** {result['model']}\n")
            f.write(f"**Language:** {result['language']}")

            if result.get('language_probability'):
                f.write(f" ({result['language_probability']:.1%} confidence)")
            f.write("\n")

            f.write(f"**Duration:** {result.get('duration', 0):.1f} seconds\n")
            f.write(f"**Processing Time:** {result['processing_time']:.1f} seconds\n")
            f.write(f"**Word Count:** {result['word_count']} words\n")
            f.write(f"**Segments:** {len(result.get('segments', []))}\n")

            if result.get('chunks_processed'):
                f.write(f"**Chunks Processed:** {result['chunks_processed']}\n")

            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Add cleaned transcript
            f.write("## Cleaned Transcript\n\n")
            f.write(cleaned_text)
            f.write("\n\n")

            # Add raw transcript
            f.write("## Raw Transcript\n\n")
            f.write(result['full_text'])
            f.write("\n\n")

            # Add segments if available
            if result.get('segments'):
                f.write("## Detailed Segments\n\n")
                for i, segment in enumerate(result['segments'], 1):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '')
                    f.write(f"{i:3d}. [{start:6.1f}s - {end:6.1f}s] {text}\n")

        return output_file

    def extract_transcript(self) -> bool:
        """Main extraction process with fallback strategies."""
        logger.info("🎯 Starting transcript extraction")
        logger.info("=" * 60)

        # Check dependencies
        deps = self.check_dependencies()
        available_backends = [k for k, v in deps.items() if v and k != 'ffmpeg']

        if not available_backends:
            logger.error("❌ No transcription backends available!")
            logger.info("Install dependencies with: pip install faster-whisper openai")
            return False

        logger.info(f"📊 Available backends: {', '.join(available_backends)}")

        # Analyze audio file
        try:
            audio_info = self.analyze_audio_file()
        except FileNotFoundError as e:
            logger.error(f"❌ {e}")
            return False

        # Strategy 1: Try faster-whisper with original file
        if deps.get('faster_whisper', False):
            result = self.transcribe_with_faster_whisper(self.audio_file)
            if result:
                output_file = self.save_transcript(result)
                logger.info(f"🎉 SUCCESS! Transcript saved to: {output_file}")
                return True

        # Strategy 2: Try with compressed audio
        if deps.get('faster_whisper', False) and audio_info['size_mb'] > 30:
            logger.info("\n🔄 Trying with compressed audio...")
            compressed_file = self.compress_audio(target_size_mb=15)
            if compressed_file:
                result = self.transcribe_with_faster_whisper(compressed_file)
                if result:
                    output_file = self.save_transcript(result)
                    logger.info(f"🎉 SUCCESS! Transcript saved to: {output_file}")
                    return True

        # Strategy 3: Try openai-whisper
        if deps.get('openai_whisper', False):
            logger.info("\n🔄 Trying openai-whisper...")
            # Use compressed file if available, otherwise original
            test_file = compressed_file if 'compressed_file' in locals() and compressed_file else self.audio_file
            result = self.transcribe_with_openai_whisper(test_file)
            if result:
                output_file = self.save_transcript(result)
                logger.info(f"🎉 SUCCESS! Transcript saved to: {output_file}")
                return True

        # Strategy 4: Try OpenAI API with chunking
        if deps.get('openai_api', False):
            logger.info("\n🔄 Trying OpenAI API with file chunking...")
            chunk_files = self.split_audio_for_openai()
            if chunk_files:
                result = self.transcribe_with_openai_api(chunk_files)
                if result:
                    output_file = self.save_transcript(result)
                    logger.info(f"🎉 SUCCESS! Transcript saved to: {output_file}")

                    # Clean up chunk files
                    for chunk_file in chunk_files:
                        try:
                            chunk_file.unlink()
                        except Exception:
                            pass

                    return True

        logger.error("❌ All transcription strategies failed!")
        logger.info("\n💡 Troubleshooting suggestions:")
        logger.info("1. Install dependencies: pip install faster-whisper openai")
        logger.info("2. Set OpenAI API key: export OPENAI_API_KEY='sk-...'")
        logger.info("3. Install ffmpeg for audio processing")
        logger.info("4. Check the transcript_extraction.log file for detailed errors")

        return False

def main():
    """Main entry point."""
    print("🎯 Audio Transcript Extraction Tool")
    print("=" * 50)

    # Default audio file path
    audio_file = Path("reference/GMT20250710-230109_Recording.m4a")

    # Allow command line argument for custom audio file
    if len(sys.argv) > 1:
        audio_file = Path(sys.argv[1])
        print(f"📁 Using custom audio file: {audio_file}")
    else:
        print(f"📁 Using default audio file: {audio_file}")

    # Check if audio file exists
    if not audio_file.exists():
        print(f"❌ Audio file not found: {audio_file}")
        print("💡 Usage: python extract_transcript.py [path/to/audio/file]")
        sys.exit(1)

    extractor = TranscriptExtractor(audio_file)
    success = extractor.extract_transcript()

    if success:
        print("\n✅ Transcript extraction completed successfully!")
    else:
        print("\n❌ Transcript extraction failed. Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()