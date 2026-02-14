"""
YouTube Transcript Extractor
A robust, production-ready tool for extracting transcripts from YouTube videos.

Features:
- Multiple fallback strategies for maximum success rate
- Proper resource management and cleanup
- Configurable retry logic and rate limiting
- Support for multiple output formats (TXT, SRT, VTT, JSON)
- Comprehensive error handling and logging
- Optional caching to avoid redundant API calls
"""

import os
import re
import time
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum
import tempfile
import shutil

# Optional dependencies - checked at runtime
try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
except ImportError:
    YouTubeTranscriptApi = None
    TranscriptsDisabled = None
    NoTranscriptFound = None

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None


class TranscriptFormat(Enum):
    """Supported transcript output formats."""
    TXT = "txt"
    SRT = "srt"
    VTT = "vtt"
    JSON = "json"


class TranscriptionMethod(Enum):
    """Available transcription methods."""
    YOUTUBE_API = "youtube_api"
    WHISPER = "whisper"
    GOOGLE_SPEECH = "google_speech"


@dataclass
class VideoInfo:
    """Container for video metadata."""
    video_id: str
    title: str
    author: str
    duration: int
    url: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """Container for a transcript segment with timing."""
    text: str
    start: float
    duration: float

    def end(self) -> float:
        return self.start + self.duration


@dataclass
class TranscriptResult:
    """Container for complete transcript result."""
    video_info: VideoInfo
    segments: List[TranscriptSegment]
    method: TranscriptionMethod
    timestamp: str
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_info": self.video_info.to_dict(),
            "segments": [asdict(seg) for seg in self.segments],
            "method": self.method.value,
            "timestamp": self.timestamp,
            "full_text": self.full_text
        }


class TranscriberConfig:
    """Configuration for the transcriber."""
    
    def __init__(
        self,
        chunk_duration_seconds: int = 30,
        max_workers: int = 4,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        whisper_model: str = "base",
        audio_quality: str = "192",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        log_level: int = logging.INFO
    ):
        self.chunk_duration_seconds = chunk_duration_seconds
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".youtube_transcriber_cache"
        self.enable_cache = enable_cache
        self.whisper_model = whisper_model
        self.audio_quality = audio_quality
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.log_level = log_level

    def ensure_cache_dir(self):
        """Ensure cache directory exists."""
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)


class DependencyChecker:
    """Check and report on optional dependencies."""
    
    CORE_DEPENDENCIES = {
        "yt-dlp": yt_dlp,
    }
    
    TRANSCRIPTION_DEPENDENCIES = {
        "youtube-transcript-api": YouTubeTranscriptApi,
        "SpeechRecognition": sr,
        "pydub": AudioSegment,
        "openai-whisper": whisper,
    }
    
    @classmethod
    def check_core(cls) -> List[str]:
        """Check core dependencies required for basic functionality."""
        return [name for name, module in cls.CORE_DEPENDENCIES.items() if module is None]
    
    @classmethod
    def check_all(cls) -> Dict[str, bool]:
        """Check all dependencies and return availability status."""
        return {
            **{name: module is not None for name, module in cls.CORE_DEPENDENCIES.items()},
            **{name: module is not None for name, module in cls.TRANSCRIPTION_DEPENDENCIES.items()}
        }
    
    @classmethod
    def print_status(cls):
        """Print dependency status."""
        print("\n=== Dependency Status ===")
        all_deps = cls.check_all()
        for name, available in all_deps.items():
            status = "✓ Installed" if available else "✗ Not installed"
            print(f"{name:30} {status}")
        print()


class YouTubeURLParser:
    """Parse and validate YouTube URLs."""
    
    PATTERNS = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'  # Direct video ID
    ]
    
    @classmethod
    def extract_video_id(cls, url_or_id: str) -> Optional[str]:
        """Extract video ID from URL or validate direct ID."""
        url_or_id = url_or_id.strip()
        
        for pattern in cls.PATTERNS:
            match = re.search(pattern, url_or_id)
            if match:
                return match.group(1)
        
        return None
    
    @classmethod
    def build_url(cls, video_id: str) -> str:
        """Build standard YouTube URL from video ID."""
        return f"https://www.youtube.com/watch?v={video_id}"


@contextmanager
def temporary_directory():
    """Context manager for temporary directory that ensures cleanup."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


class YouTubeTranscriber:
    """
    Main transcriber class with multiple fallback strategies.
    
    Attempts transcription in order:
    1. YouTube's built-in captions (fastest, free)
    2. Whisper AI (most accurate, requires download)
    3. Google Speech Recognition (fallback, requires download)
    """
    
    def __init__(self, config: Optional[TranscriberConfig] = None):
        self.config = config or TranscriberConfig()
        self._setup_logging()
        self.config.ensure_cache_dir()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=self.config.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _get_cache_key(self, video_id: str, language: str) -> str:
        """Generate cache key for a video transcript."""
        return hashlib.md5(f"{video_id}_{language}".encode()).hexdigest()
    
    def _get_from_cache(self, video_id: str, language: str) -> Optional[TranscriptResult]:
        """Retrieve transcript from cache if available."""
        if not self.config.enable_cache:
            return None
        
        cache_key = self._get_cache_key(video_id, language)
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reconstruct TranscriptResult from cached data
                video_info = VideoInfo(**data['video_info'])
                segments = [TranscriptSegment(**seg) for seg in data['segments']]
                method = TranscriptionMethod(data['method'])
                
                return TranscriptResult(
                    video_info=video_info,
                    segments=segments,
                    method=method,
                    timestamp=data['timestamp'],
                    full_text=data['full_text']
                )
            except Exception as e:
                self.logger.warning(f"Cache read failed: {e}")
        
        return None
    
    def _save_to_cache(self, result: TranscriptResult, language: str):
        """Save transcript result to cache."""
        if not self.config.enable_cache:
            return
        
        cache_key = self._get_cache_key(result.video_info.video_id, language)
        cache_file = self.config.cache_dir / f"{cache_key}.json"
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache write failed: {e}")
    
    def get_video_info(self, video_id: str) -> VideoInfo:
        """Fetch video metadata using yt-dlp."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp is required but not installed")
        
        url = YouTubeURLParser.build_url(video_id)
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return VideoInfo(
                    video_id=video_id,
                    title=info.get('title', 'Unknown'),
                    author=info.get('uploader', 'Unknown'),
                    duration=info.get('duration', 0),
                    url=url
                )
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            # Return minimal info
            return VideoInfo(
                video_id=video_id,
                title="Unknown",
                author="Unknown",
                duration=0,
                url=url
            )
    
    def _try_youtube_captions(
        self,
        video_id: str,
        language: str = "en"
    ) -> Optional[List[TranscriptSegment]]:
        """Attempt to fetch existing captions from YouTube."""
        if YouTubeTranscriptApi is None:
            self.logger.debug("youtube-transcript-api not available")
            return None
        
        try:
            self.logger.info("Attempting to fetch YouTube captions...")
            
            # Try specified language first, then fall back to any available
            languages = [language, f"{language}-US", f"{language}-GB"]
            
            for lang in languages:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=[lang]
                    )
                    
                    segments = [
                        TranscriptSegment(
                            text=item['text'],
                            start=item['start'],
                            duration=item['duration']
                        )
                        for item in transcript_list
                    ]
                    
                    self.logger.info(f"✓ Found YouTube captions in language: {lang}")
                    return segments
                
                except (NoTranscriptFound, TranscriptsDisabled):
                    continue
                except Exception as e:
                    self.logger.debug(f"Failed to get captions for {lang}: {e}")
                    continue
            
            # Try getting any available transcript
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                segments = [
                    TranscriptSegment(
                        text=item['text'],
                        start=item['start'],
                        duration=item['duration']
                    )
                    for item in transcript_list
                ]
                self.logger.info("✓ Found YouTube captions (default language)")
                return segments
            except:
                pass
        
        except Exception as e:
            self.logger.debug(f"YouTube captions not available: {e}")
        
        return None
    
    def _download_audio(self, video_id: str, output_dir: Path) -> Optional[Path]:
        """Download audio from YouTube video."""
        if yt_dlp is None:
            raise RuntimeError("yt-dlp is required but not installed")
        
        output_path = output_dir / "audio.mp3"
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': self.config.audio_quality,
            }],
            'outtmpl': str(output_dir / 'audio'),
            'quiet': True,
            'no_warnings': True,
        }
        
        url = YouTubeURLParser.build_url(video_id)
        
        try:
            self.logger.info("Downloading audio...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            # yt-dlp may add .mp3 extension
            if not output_path.exists():
                output_path = output_dir / "audio.mp3"
            
            if output_path.exists():
                self.logger.info(f"✓ Audio downloaded: {output_path}")
                return output_path
            else:
                self.logger.error("Audio file not found after download")
                return None
        
        except Exception as e:
            self.logger.error(f"Audio download failed: {e}")
            return None
    
    def _transcribe_with_whisper(
        self,
        audio_path: Path,
        language: str = "en"
    ) -> Optional[List[TranscriptSegment]]:
        """Transcribe audio using OpenAI Whisper."""
        if whisper is None or torch is None:
            self.logger.debug("Whisper not available")
            return None
        
        try:
            self.logger.info(f"Transcribing with Whisper ({self.config.whisper_model} model)...")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {device}")
            
            model = whisper.load_model(self.config.whisper_model, device=device)
            result = model.transcribe(
                str(audio_path),
                language=language.split('-')[0],  # Use primary language code
                verbose=False
            )
            
            # Convert Whisper segments to our format
            segments = [
                TranscriptSegment(
                    text=seg['text'].strip(),
                    start=seg['start'],
                    duration=seg['end'] - seg['start']
                )
                for seg in result['segments']
            ]
            
            self.logger.info(f"✓ Whisper transcription complete ({len(segments)} segments)")
            return segments
        
        except Exception as e:
            self.logger.error(f"Whisper transcription failed: {e}")
            return None
    
    def _transcribe_with_google(
        self,
        audio_path: Path,
        language: str = "en-US"
    ) -> Optional[List[TranscriptSegment]]:
        """Transcribe audio using Google Speech Recognition."""
        if sr is None or AudioSegment is None:
            self.logger.debug("SpeechRecognition or pydub not available")
            return None
        
        try:
            self.logger.info("Transcribing with Google Speech Recognition...")
            
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            
            # Load and convert audio
            audio = AudioSegment.from_file(str(audio_path))
            chunk_duration_ms = self.config.chunk_duration_seconds * 1000
            
            segments = []
            current_time = 0
            
            # Process in chunks
            for i in range(0, len(audio), chunk_duration_ms):
                chunk = audio[i:i + chunk_duration_ms]
                chunk_file = audio_path.parent / f"chunk_{i}.wav"
                
                try:
                    chunk.export(str(chunk_file), format="wav")
                    
                    with sr.AudioFile(str(chunk_file)) as source:
                        audio_data = recognizer.record(source)
                        
                        # Try recognition with retry
                        for attempt in range(self.config.max_retries):
                            try:
                                text = recognizer.recognize_google(audio_data, language=language)
                                
                                if text.strip():
                                    segments.append(TranscriptSegment(
                                        text=text.strip(),
                                        start=current_time,
                                        duration=len(chunk) / 1000.0
                                    ))
                                break
                            
                            except sr.RequestError as e:
                                if attempt < self.config.max_retries - 1:
                                    self.logger.warning(f"Request failed, retrying... ({e})")
                                    time.sleep(self.config.retry_delay * (attempt + 1))
                                else:
                                    self.logger.error(f"Max retries reached: {e}")
                            
                            except sr.UnknownValueError:
                                # No speech detected in this chunk
                                break
                
                finally:
                    # Clean up chunk file
                    if chunk_file.exists():
                        chunk_file.unlink()
                
                current_time += len(chunk) / 1000.0
            
            if segments:
                self.logger.info(f"✓ Google Speech transcription complete ({len(segments)} segments)")
                return segments
            else:
                self.logger.warning("No speech detected in audio")
                return None
        
        except Exception as e:
            self.logger.error(f"Google Speech transcription failed: {e}")
            return None
    
    def transcribe(
        self,
        youtube_url: str,
        language: str = "en",
        force_method: Optional[TranscriptionMethod] = None
    ) -> Optional[TranscriptResult]:
        """
        Main transcription method with fallback strategy.
        
        Args:
            youtube_url: YouTube URL or video ID
            language: Language code (e.g., 'en', 'en-US', 'es', 'fr')
            force_method: Force a specific transcription method
        
        Returns:
            TranscriptResult if successful, None otherwise
        """
        start_time = time.time()
        
        # Extract video ID
        video_id = YouTubeURLParser.extract_video_id(youtube_url)
        if not video_id:
            self.logger.error(f"Invalid YouTube URL: {youtube_url}")
            return None
        
        self.logger.info(f"Processing video ID: {video_id}")
        
        # Check cache
        cached_result = self._get_from_cache(video_id, language)
        if cached_result:
            self.logger.info("✓ Using cached transcript")
            return cached_result
        
        # Get video info
        video_info = self.get_video_info(video_id)
        self.logger.info(f"Video: {video_info.title} by {video_info.author}")
        
        segments = None
        method_used = None
        
        # Try methods in order based on force_method or default strategy
        if force_method == TranscriptionMethod.YOUTUBE_API or force_method is None:
            segments = self._try_youtube_captions(video_id, language)
            if segments:
                method_used = TranscriptionMethod.YOUTUBE_API
        
        # If YouTube captions failed, try audio-based methods
        if not segments and force_method != TranscriptionMethod.YOUTUBE_API:
            with temporary_directory() as temp_dir:
                audio_path = self._download_audio(video_id, temp_dir)
                
                if audio_path:
                    # Try Whisper first (better quality)
                    if force_method == TranscriptionMethod.WHISPER or force_method is None:
                        segments = self._transcribe_with_whisper(audio_path, language)
                        if segments:
                            method_used = TranscriptionMethod.WHISPER
                    
                    # Fall back to Google Speech
                    if not segments and force_method != TranscriptionMethod.WHISPER:
                        lang_code = f"{language}-US" if '-' not in language else language
                        segments = self._transcribe_with_google(audio_path, lang_code)
                        if segments:
                            method_used = TranscriptionMethod.GOOGLE_SPEECH
        
        if not segments:
            self.logger.error("All transcription methods failed")
            return None
        
        # Create result
        full_text = " ".join(seg.text for seg in segments)
        result = TranscriptResult(
            video_info=video_info,
            segments=segments,
            method=method_used,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            full_text=full_text
        )
        
        # Save to cache
        self._save_to_cache(result, language)
        
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Transcription complete in {elapsed:.2f}s using {method_used.value}")
        
        return result


class TranscriptFormatter:
    """Format transcripts into different output formats."""
    
    @staticmethod
    def to_text(result: TranscriptResult, include_metadata: bool = True) -> str:
        """Format as plain text."""
        lines = []
        
        if include_metadata:
            lines.extend([
                f"Title: {result.video_info.title}",
                f"Author: {result.video_info.author}",
                f"Duration: {result.video_info.duration} seconds",
                f"URL: {result.video_info.url}",
                f"Transcribed: {result.timestamp}",
                f"Method: {result.method.value}",
                "=" * 70,
                ""
            ])
        
        lines.append(result.full_text)
        return "\n".join(lines)
    
    @staticmethod
    def to_srt(result: TranscriptResult) -> str:
        """Format as SRT subtitle file."""
        lines = []
        
        for i, seg in enumerate(result.segments, 1):
            start_time = TranscriptFormatter._format_timestamp_srt(seg.start)
            end_time = TranscriptFormatter._format_timestamp_srt(seg.end())
            
            lines.extend([
                str(i),
                f"{start_time} --> {end_time}",
                seg.text,
                ""
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def to_vtt(result: TranscriptResult) -> str:
        """Format as WebVTT subtitle file."""
        lines = ["WEBVTT", ""]
        
        for seg in result.segments:
            start_time = TranscriptFormatter._format_timestamp_vtt(seg.start)
            end_time = TranscriptFormatter._format_timestamp_vtt(seg.end())
            
            lines.extend([
                f"{start_time} --> {end_time}",
                seg.text,
                ""
            ])
        
        return "\n".join(lines)
    
    @staticmethod
    def to_json(result: TranscriptResult, pretty: bool = True) -> str:
        """Format as JSON."""
        indent = 2 if pretty else None
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=indent)
    
    @staticmethod
    def _format_timestamp_srt(seconds: float) -> str:
        """Format timestamp for SRT format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_timestamp_vtt(seconds: float) -> str:
        """Format timestamp for VTT format (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class TranscriptWriter:
    """Write transcripts to files in various formats."""
    
    @staticmethod
    def write(
        result: TranscriptResult,
        output_path: str,
        format: TranscriptFormat = TranscriptFormat.TXT,
        include_metadata: bool = True
    ) -> bool:
        """
        Write transcript to file.
        
        Args:
            result: TranscriptResult to write
            output_path: Output file path
            format: Output format
            include_metadata: Include video metadata (for TXT format only)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Format content based on requested format
            if format == TranscriptFormat.TXT:
                content = TranscriptFormatter.to_text(result, include_metadata)
            elif format == TranscriptFormat.SRT:
                content = TranscriptFormatter.to_srt(result)
            elif format == TranscriptFormat.VTT:
                content = TranscriptFormatter.to_vtt(result)
            elif format == TranscriptFormat.JSON:
                content = TranscriptFormatter.to_json(result)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logging.info(f"✓ Transcript saved to: {output_file}")
            return True
        
        except Exception as e:
            logging.error(f"Failed to write transcript: {e}")
            return False


# Example usage and CLI interface
def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract transcripts from YouTube videos with multiple fallback methods"
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="YouTube URL or video ID"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (default: transcript_<video_id>.txt)"
    )
    parser.add_argument(
        "-f", "--format",
        choices=["txt", "srt", "vtt", "json"],
        default="txt",
        help="Output format (default: txt)"
    )
    parser.add_argument(
        "-l", "--language",
        default="en",
        help="Language code (default: en)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=["youtube_api", "whisper", "google_speech"],
        help="Force specific transcription method"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependency status and exit"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Check dependencies if requested
    if args.check_deps:
        DependencyChecker.print_status()
        return
    
    # Check core dependencies
    missing = DependencyChecker.check_core()
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Install with: pip install yt-dlp")
        print("\nNote: FFmpeg must also be installed on your system.")
        return
    
    # Get URL if not provided
    url = args.url
    if not url:
        url = input("Enter YouTube URL or video ID: ").strip()
    
    if not url:
        print("Error: No URL provided")
        return
    
    # Setup configuration
    config = TranscriberConfig(
        enable_cache=not args.no_cache,
        log_level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    # Create transcriber
    transcriber = YouTubeTranscriber(config)
    
    # Parse method if specified
    force_method = None
    if args.method:
        force_method = TranscriptionMethod(args.method)
    
    # Transcribe
    result = transcriber.transcribe(url, args.language, force_method)
    
    if not result:
        print("Error: Transcription failed")
        return
    
    # Determine output path
    output_path = args.output
    if not output_path:
        output_path = f"transcript_{result.video_info.video_id}.{args.format}"
    
    # Write output
    format_enum = TranscriptFormat(args.format)
    success = TranscriptWriter.write(result, output_path, format_enum)
    
    if success:
        print(f"\n✓ Transcript saved to: {output_path}")
        print(f"Method used: {result.method.value}")
        print(f"\nPreview:\n{result.full_text[:500]}...")
    else:
        print("Error: Failed to write output file")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)