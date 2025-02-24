To create a `requirements.txt` file for the provided code, we need to include all third-party Python dependencies used. Here's the requirements list:

```
yt-dlp
SpeechRecognition
pydub
youtube-transcript-api
openai-whisper
torch
tqdm
```

**Important Notes:**
1. FFmpeg is required for audio processing but must be installed separately at the system level
2. For better GPU performance with Whisper, consider installing PyTorch with CUDA support from [pytorch.org](https://pytorch.org/)
3. Some systems might need additional dependencies for `pydub` (like libavcodec)

You can install these requirements with:
```bash
pip install -r requirements.txt
```