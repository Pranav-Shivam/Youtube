import os
import re
import time
import json
import tiktoken
import requests
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Import the YouTubeTranscriber from the existing file
from youtube_encode import YouTubeTranscriber

class YouTubeNotesCreator:
    def __init__(self):
        self.transcriber = YouTubeTranscriber()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.notes_dir = self._ensure_dir("notes")
        self.cache_dir = self._ensure_dir("cache")
        self.user_defined_context: str =""
        self.code_required: bool = False
        self.ollama_config = {
            "model": "deepseek-r1:7b",
            "base_url": "http://localhost:11434",
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 1024
            }
        }
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.token_size = 1500
        self.chunk_overlap = 100

    def _ensure_dir(self, dir_name: str) -> str:
        path = os.path.join(self.base_dir, dir_name)
        os.makedirs(path, exist_ok=True)
        return path

    def _safe_filename(self, name: str) -> str:
        if not name:
            return "untitled"
        safe_name = re.sub(r'[^\w\s-]', '', name)
        safe_name = re.sub(r'\s+', '_', safe_name).strip()
        return safe_name[:100] if len(safe_name) > 100 else safe_name

    def _token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4

    def _chunk_content(self, content: str) -> list:
        if not content:
            return []
            
        main_content_parts = content.split('='*50)
        main_content = main_content_parts[1].strip() if len(main_content_parts) > 1 else content
        
        sentences = re.split(r'(?<=[.!?])\s+', main_content)
        chunks, current = [], ""
        
        for sentence in sentences:
            potential = f"{current} {sentence}".strip()
            token_count = self._token_count(potential)
            
            if token_count > self.token_size:
                if current:
                    chunks.append({"section": current})
                    overlap_words = current.split()[-min(len(current.split()), self.chunk_overlap):]
                    current = " ".join(overlap_words) + " " + sentence
                else:
                    current = sentence
            else:
                current = potential
                
        if current.strip():
            chunks.append({"section": current})
            
        return chunks

    def _extract_metadata(self, content: str, field: str) -> str:
        if not content:
            return ""
        match = re.search(fr"{field}:\s*(.+?)(?:\n|$)", content)
        return match.group(1).strip() if match else ""

    def _create_prompt(self, context: str, chunk: str) -> str:
        user_defined_context = self.user_defined_context
        code_required = self.code_required
        context_text = f"Context: {context}\n\n" if context else ""
        user_defined_context_text = f"Additional Context (User Defined): {user_defined_context}\n\n" if user_defined_context else ""
        code_instruction = "Provide relevant code snippets where applicable." if code_required else "Code is not required."

        return f"""{context_text}{user_defined_context_text}
    I need you to extract key information from the following text chunk from a video transcript and generate detailed, well-structured notes.

    ### Focus Areas:
    - **Main ideas and key takeaways**
    - **Important details and insights**
    - **Actionable points, if applicable**
    - **Comprehensive explanations without missing critical information**
    - **Easy, medium, and hard examples to explain the topic**
    - **{code_instruction}**

    ### Formatting Guidelines:
    - **Use headings for topics and sections**
    - **Use bullet points for key details**
    - **Use numbered lists for ordered steps or sequences**

    ### Text Chunk:
    {chunk}

    Please provide **well-structured and detailed notes** that effectively summarize and capture the essence of this content.
    """

    
    def _generate_notes(self, text: str, context: Optional[str] = None) -> str:
        if not text:
            return ""
            
        prompt = self._create_prompt(context= context, chunk= text, user_defined_context= "", code_required= True)
        
        try:
            response = requests.post(
                f"{self.ollama_config['base_url']}/api/generate",
                json={
                    "model": self.ollama_config["model"], 
                    "prompt": prompt, 
                    "stream": False, 
                    "options": self.ollama_config["options"]
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            return ""
        except Exception:
            return ""

    def create_notes_from_url(self, youtube_url: str, user_defined_context: str, code_required: bool) -> Optional[str]:
        print(f"Processing YouTube URL: {youtube_url}")
        self.code_required = code_required
        self.user_defined_context = user_defined_context
        
        # 1. Extract video ID
        video_id = self.transcriber.extract_video_id(youtube_url)
        if not video_id:
            print("Invalid YouTube URL")
            return None
            
        # 2. Get video info
        print("Fetching video information...")
        video_info = self.transcriber.get_video_info(video_id)
        
        # 3. Download audio
        print("Downloading audio...")
        audio_file = os.path.join(self.cache_dir, f"temp_audio_{video_id}.mp3")
        self.transcriber.download_audio(video_id, audio_file)
        
        # 4. Transcribe audio
        print("Transcribing audio...")
        transcript = self.transcriber.use_whisper_if_available(audio_file, "en")
        
        # 5. Create transcript file
        print("Saving transcript...")
        transcript_file = os.path.join(self.cache_dir, f"transcript_{video_id}.txt")
        self.transcriber.save_transcript(transcript, video_info, transcript_file)
        
        # 6. Generate markdown notes
        print("Generating notes...")
        markdown_file = self.generate_markdown_notes(transcript_file)
        
        # 7. Clean up
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
            if os.path.exists(transcript_file):
                os.remove(transcript_file)
        except Exception:
            pass
            
        return markdown_file

    def generate_markdown_notes(self, transcript_file: str) -> Optional[str]:
        if not os.path.exists(transcript_file):
            print(f"Transcript file not found: {transcript_file}")
            return None
            
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading transcript file: {e}")
            return None
            
        # Extract metadata
        title = self._extract_metadata(content, "Title") or "Unknown Video"
        author = self._extract_metadata(content, "Author") or "Unknown Author"
        duration = self._extract_metadata(content, "Duration") or "Unknown Duration"
        
        # Create output file
        output_file = os.path.join(self.notes_dir, f"{self._safe_filename(title)}.md")
        
        # Chunk the content
        chunks = self._chunk_content(content)
        if not chunks:
            print("No content to process")
            return None
            
        # Generate notes for each chunk
        print(f"Processing {len(chunks)} chunks of content...")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write metadata
                f.write(f"# Notes: {title}\n\n")
                f.write("## Metadata\n")
                f.write(f"- Title: {title}\n")
                f.write(f"- Author: {author}\n")
                f.write(f"- Duration: {duration}\n")
                f.write(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Write notes
                f.write("## Generated Notes\n\n")
                
                previous_notes = ""
                for i, chunk in enumerate(chunks, 1):
                    print(f"Processing chunk {i}/{len(chunks)}...")
                    context = "\n".join(previous_notes.split("\n")[-10:]) if previous_notes else None
                    response = self._generate_notes(chunk["section"], context)
                    if '<think>' in response:
                        cleaned_response = re.sub(r'<think>.*?</think>\s*\n*', '', response, flags=re.DOTALL)
                        final_answer = cleaned_response.strip()
                    else:
                        final_answer = response.strip()
                    notes = final_answer
                    if notes:
                        previous_notes = notes
                        f.write(f"{notes}\n\n---\n\n" if i < len(chunks) else notes)
                    time.sleep(1)  # Small delay between chunks
                    
            print(f"Notes generated successfully: {output_file}")
            return output_file
        except Exception as e:
            print(f"Error generating notes: {e}")
            return None

def main():
    
    url = str(input("Enter the YouTube URL: "))
    user_defined_context = str(input("Enter User Defined Context: "))
    code_required = bool(input("Enter User Defined Context Either True/False: "))
    user_defined_llms = input("Enter or choose which LLM you want to use (deepseek, nomic, llama3, openai, gemini): ").strip().lower()

    llm_mapping = {
        "deepseek": "deepseek-r1:7b",
        "llama3": "llama3.2:latest",
        "openai": "OpenAI API",
        "gemini": "Gemini API"
    }

    selected_llm = llm_mapping.get(user_defined_llms, "Invalid choice, please select a valid LLM")
    print(f"Selected LLM: {selected_llm}")
    creator = YouTubeNotesCreator()
    notes_file = creator.create_notes_from_url(url, user_defined_context, code_required)
    
    if notes_file:
        print(f"\nNotes created successfully: {notes_file}")
    else:
        print("\nFailed to create notes")

if __name__ == "__main__":
    main()