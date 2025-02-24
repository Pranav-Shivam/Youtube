import os
import re
import time
import logging
import torch
import concurrent.futures
from tqdm import tqdm

# Optional dependencies that will be checked at runtime
# Import them here to make them available throughout the code
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
    from youtube_transcript_api import YouTubeTranscriptApi
except ImportError:
    YouTubeTranscriptApi = None

try:
    import whisper
except ImportError:
    whisper = None


class YouTubeTranscriber:
    """A class for extracting and generating transcripts from YouTube videos."""

    def __init__(self, log_level=logging.INFO):
        """Initialize the transcriber with logging configuration."""
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_video_id(self, youtube_url):
        """Extract the video ID from a YouTube URL."""
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]

        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)

        return None

    def get_video_info(self, video_id):
        """Get basic information about the video using yt-dlp."""
        try:
            if yt_dlp is None:
                raise ImportError("yt-dlp not installed")

            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                return {
                    "title": info.get('title', 'Unknown'),
                    "author": info.get('uploader', 'Unknown'),
                    "length": info.get('duration', 0),
                }
        except Exception as e:
            self.logger.error(f"Could not get video info: {e}")
            return {"title": "Unknown", "author": "Unknown", "length": 0}

    def get_transcript_from_youtube_api(self, video_id, language=None):
        """Try to get transcript using YouTube's transcript API."""
        try:
            if YouTubeTranscriptApi is None:
                raise ImportError("youtube_transcript_api not installed")

            # If language is specified, try to get that language transcript
            if language:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
                    return " ".join([item['text'] for item in transcript_list])
                except:
                    # If specific language fails, continue to get default transcript
                    pass

            # Get available transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            return " ".join([item['text'] for item in transcript_list])
        except Exception as e:
            self.logger.error(f"Could not get transcript from YouTube API: {e}")
            return None

    def download_audio(self, video_id, output_path="audio.mp3"):
        """Download the audio using yt-dlp."""
        try:
            if yt_dlp is None:
                raise ImportError("yt-dlp not installed")

            # Create folder for the output if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': output_path.replace('.mp3', ''),
                'quiet': False,
                'no_warnings': True,
            }

            self.logger.info(f"Downloading audio for video ID: {video_id}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])

            return output_path.replace('.mp3', '') + '.mp3'
        except Exception as e:
            self.logger.error(f"Error downloading audio: {e}")
            return None

    def transcribe_audio_chunk(self, chunk_info):
        """Transcribe a single audio chunk."""
        chunk_file, chunk_index, recognizer, language = chunk_info

        try:
            if sr is None:
                raise ImportError("speech_recognition not installed")

            with sr.AudioFile(chunk_file) as source:
                audio_data = recognizer.record(source)
                try:
                    chunk_text = recognizer.recognize_google(audio_data, language=language)
                    return {"index": chunk_index, "text": chunk_text}
                except sr.UnknownValueError:
                    return {"index": chunk_index, "text": ""}
                except sr.RequestError as e:
                    self.logger.error(f"Error with speech recognition service: {e}")
                    # Try one more time with a delay
                    time.sleep(2)
                    try:
                        chunk_text = recognizer.recognize_google(audio_data, language=language)
                        return {"index": chunk_index, "text": chunk_text}
                    except:
                        return {"index": chunk_index, "text": ""}
        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk_index}: {e}")
            return {"index": chunk_index, "text": ""}

    def transcribe_audio(self, audio_file, language="en-US"):
        """Transcribe audio file using speech recognition with parallel processing."""
        try:
            if sr is None:
                raise ImportError("speech_recognition not installed")
            if AudioSegment is None:
                raise ImportError("pydub not installed")

            recognizer = sr.Recognizer()

            # Try to increase recognition quality
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True

            # Convert mp3 to wav for speech_recognition
            audio = AudioSegment.from_file(audio_file)
            temp_wav = "temp.wav"
            audio.export(temp_wav, format="wav")

            # Process in 30-second chunks
            chunk_duration = 30 * 1000  # 30 seconds in milliseconds
            chunk_files = []

            # Split audio into chunks
            for i in range(0, len(audio), chunk_duration):
                chunk = audio[i:i+chunk_duration]
                chunk_file = f"chunk_{i}.wav"
                chunk.export(chunk_file, format="wav")
                chunk_files.append((chunk_file, i // chunk_duration, recognizer, language))

            transcript_pieces = []

            # Use progress bar to show transcription progress
            with tqdm(total=len(chunk_files), desc="Transcribing", unit="chunk") as pbar:
                # Use concurrent processing with error handling
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(chunk_files))) as executor:
                    future_to_chunk = {executor.submit(self.transcribe_audio_chunk, chunk_info): chunk_info for chunk_info in chunk_files}

                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_file = future_to_chunk[future][0]
                        try:
                            result = future.result()
                            transcript_pieces.append(result)
                        except Exception as e:
                            self.logger.error(f"Error processing chunk: {e}")
                        finally:
                            # Make sure to clean up chunk files
                            if os.path.exists(chunk_file):
                                os.remove(chunk_file)
                        pbar.update(1)

            # Clean up temp file
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

            # Sort pieces by index and join
            transcript_pieces.sort(key=lambda x: x["index"])
            full_text = " ".join([piece["text"] for piece in transcript_pieces if piece["text"]])

            return full_text
        except Exception as e:
            self.logger.error(f"Error in transcription process: {e}")
            # Clean up any remaining files
            if os.path.exists("temp.wav"):
                os.remove("temp.wav")
            for i in range(0, len(audio), chunk_duration if 'chunk_duration' in locals() else 30000):
                chunk_file = f"chunk_{i}.wav"
                if os.path.exists(chunk_file):
                    os.remove(chunk_file)
            return ""

    def save_transcript(self, transcript, video_info, output_file="transcript.txt"):
        """Save the transcript to a file with video info."""
        try:
            # Create folder for the output if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                # Write video info header
                f.write(f"Title: {video_info.get('title', 'Unknown')}\n")
                f.write(f"Author: {video_info.get('author', 'Unknown')}\n")
                f.write(f"Duration: {video_info.get('length', 0)} seconds\n")
                f.write(f"Transcribed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("\n" + "=" * 50 + "\n\n")

                # Write transcript
                f.write(transcript)

            self.logger.info(f"Transcript saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving transcript: {e}")
            return False

    def use_whisper_if_available(self, audio_file, language="en"):
        """Try to use Whisper if available, as it's more accurate."""
        try:
            if whisper is None:
                raise ImportError("whisper not installed")

            self.logger.info("Using Whisper for transcription (higher accuracy)")
            language_code = language.split('-')[0]  # Extract primary language code (e.g., 'en' from 'en-US')

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = whisper.load_model("base", device=device)
            result = model.transcribe(audio_file, language=language_code)

            return result["text"]
        except ImportError:
            self.logger.info("Whisper not available, using Google Speech Recognition")
            return None
        except Exception as e:
            self.logger.error(f"Error using Whisper: {e}")
            return None

    def get_video_transcript(self, youtube_url, output_file=None, language="en-US"):
        """Main function to get transcript from a YouTube video."""
        start_time = time.time()

        # Extract video ID and create default output file if not provided
        video_id = self.extract_video_id(youtube_url)
        if not video_id:
            self.logger.error("Invalid YouTube URL")
            return None

        if output_file is None:
            output_file = f"transcript_{video_id}.txt"

        # Get video information using yt-dlp
        video_info = self.get_video_info(video_id)
        self.logger.info(f"Processing video: {video_info.get('title', 'Unknown')}")

        # Try to get transcript from YouTube API first
        self.logger.info("Attempting to get existing captions...")
        transcript = self.get_transcript_from_youtube_api(video_id, language)

        # If YouTube API transcript not available, transcribe the audio
        if not transcript:
            self.logger.info("No captions found. Transcribing audio...")
            audio_file = self.download_audio(video_id, f"audio_{video_id}.mp3")

            if audio_file and os.path.exists(audio_file):
                # Try Whisper first if available (better quality)
                transcript = self.use_whisper_if_available(audio_file, language)

                # Fall back to Google Speech Recognition if Whisper fails or isn't available
                if not transcript:
                    transcript = self.transcribe_audio(audio_file, language)

                # Clean up the audio file
                try:
                    os.remove(audio_file)
                except:
                    pass

        if transcript:
            self.save_transcript(transcript, video_info, output_file)

            # Log completion time
            elapsed_time = time.time() - start_time
            self.logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

            return transcript
        else:
            self.logger.error("Could not generate transcript")
            return None


class DependencyChecker:
    """A class to check for required dependencies."""

    @staticmethod
    def check_dependencies():
        """Check if all required libraries are installed."""
        missing_libs = []
        if yt_dlp is None:
            missing_libs.append("yt-dlp")
        if sr is None:
            missing_libs.append("SpeechRecognition")
        if AudioSegment is None:
            missing_libs.append("pydub")

        return missing_libs


# Example usage
if __name__ == "__main__":
    try:
        # Check for required libraries
        dependency_checker = DependencyChecker()
        missing_libs = dependency_checker.check_dependencies()

        if missing_libs:
            print("Missing required libraries. Please install them with:")
            print(f"pip install {' '.join(missing_libs)}")
            print("\nAlso make sure you have ffmpeg installed on your system.")
            exit(1)

        # Initialize transcriber
        transcriber = YouTubeTranscriber()

        # Get video URL and generate transcript
        video_url = input("Enter the YouTube video link: ")
        transcript = transcriber.get_video_transcript(video_url)

        if transcript:
            print("\nPreview of transcript:")
            print(transcript)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print(f"An error occurred: {e}")
