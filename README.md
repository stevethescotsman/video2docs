# Video2Docs

Converts video recordings to structured Word documents. Upload a local video (or YouTube URL), and the app transcribes the audio with Azure Whisper, uses GPT-4.1 to organise the content into sections with bullet points, and captures one screenshot per section — aligned to the transcript.

Output: a `.docx` with titled sections, bullet point summaries, and a relevant screenshot per section.

## Requirements

- Python 3.10+ (3.11 recommended)
- FFmpeg on PATH — install via `brew install ffmpeg` (Mac) or [ffmpeg.org](https://ffmpeg.org/download.html) (Windows)
- Azure OpenAI resource with a GPT-4.1 deployment
- Azure OpenAI resource with a Whisper deployment (for transcription)

## Setup

```shell
git clone https://github.com/stevethescotsman/video2docs.git
cd video2docs
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy the example env file and fill in your credentials:

```shell
cp .env.example .env
```

Edit `.env` — the required fields are:

```env
# Azure OpenAI (GPT-4.1 for summarisation)
OPENAI_API_KEY=<your Azure OpenAI key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
VIDEO2DOCS_LLM_MODEL=gpt-5.4

# Azure Whisper (transcription)
AZURE_WHISPER_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com
AZURE_WHISPER_KEY=<your Whisper key>

# Web UI login
ADMIN_USER=admin
ADMIN_PASS=<choose a password>
SECRET_KEY=<random string>
PORT=5001
```

> **Note:** `.env` is gitignored — your keys are never committed.

## Run

```shell
python -m src.webapp
```

Open [http://localhost:5001](http://localhost:5001) and log in with the credentials from your `.env`.

On Windows: double-click `run_web.bat`. On Mac/Linux: `./run_web.sh`.

## How to convert a video

1. Log in
2. Click **New Conversion**
3. Upload a video file (mp4, mov, avi, mkv, webm) or paste a YouTube URL
4. Choose output format (DOCX recommended) and click **Convert**
5. The conversion runs in the background — the page shows live progress
6. When done, click **Download** to get the document

A 14-minute screen recording takes roughly 2–3 minutes (transcription dominates). Repeat conversions of the same video use a local cache and complete in ~20 seconds.

## How it works

1. **Transcription** — Azure Whisper transcribes the audio with per-segment timestamps
2. **Section detection** — GPT-4.1 reads the timestamped transcript and divides it into logical sections, each with a heading, summary sentence, and bullet points
3. **Screenshots** — one frame is extracted from the video at the midpoint of each section's time range, so screenshots match the content they sit next to
4. **Document generation** — sections are written out with text first, screenshot below

## Supported models

Edit `config/llm_models.json` to change the available models in the UI dropdown. The default is `gpt-4.1`. Any model deployed to your Azure OpenAI resource can be used.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Port 5001 already in use | Change `PORT=` in `.env` |
| `ffmpeg not found` | Install ffmpeg and ensure it's on PATH |
| Transcription fails | Check `AZURE_WHISPER_ENDPOINT` and `AZURE_WHISPER_KEY` in `.env` |
| LLM times out | Check `AZURE_OPENAI_ENDPOINT` and `OPENAI_API_KEY` in `.env` |
| Screenshots missing | LLM may have returned timestamps past the video end — known edge case, usually affects the last section only |

Logs: `/tmp/webapp.log`

## License

MIT — see LICENSE file.
