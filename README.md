# Meeting Assistant

Turns an English meeting recording into a transcript and a structured meeting record. Built for the Inter IIT Tech Meet 15.0 Bootcamp (Phase 2, ML problem statement).

For each recording it produces:

- the **raw transcript** from speech recognition, with timestamps, unchanged
- a **refined transcript** with misheard technical terms and names corrected
- a **meeting record**: summary, minutes by topic, key decisions, action items (owner and deadline, or `unspecified`) and open questions, as Markdown and JSON
- **subtitles** (`.srt`) from the raw timestamps

There is no speaker detection: the transcript and record never label who said what.

How it works, which models are used and why: see [TECHNICAL.md](TECHNICAL.md).

## Requirements

- Python 3.10 or newer
- About 1 GB of free disk space for the speech-recognition model, which is downloaded on first use
- Internet access: for the first model download, and for every run, to call the Gemini API
- A Gemini API key, from <https://aistudio.google.com/apikey>
- A GPU is optional. On the CPU-only development laptop, 14–25 second clips took 6–9 seconds to transcribe, including loading the model; full-length meetings have not been timed yet. An NVIDIA GPU with CUDA is used automatically if present.

### FFmpeg

**You do not need to install FFmpeg.** Audio is decoded with [PyAV](https://pyav.basswood-io.com/), whose wheels include the FFmpeg libraries. It reads wav, mp3, m4a, flac, ogg and aac out of the box.

A separate FFmpeg install is only useful for converting a file in another format, such as a video, before uploading it:

| OS | Install |
|---|---|
| Windows | `winget install Gyan.FFmpeg` |
| macOS | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |

```bash
ffmpeg -i meeting.mp4 -vn -ac 1 -ar 16000 meeting.wav
```

## Setup

1. Get the code and open a terminal in the project folder.

2. Create and activate a virtual environment.

   Windows (PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   macOS / Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create your `.env` file from the template and add your API key:
   ```bash
   cp .env.example .env        # Windows: copy .env.example .env
   ```
   Then edit `.env`:
   ```ini
   GEMINI_API_KEY=your-gemini-key
   ```
   `.env` is listed in `.gitignore`. Never commit it or put keys in code.

All other settings in `.env` are optional; see [Configuration](#configuration).

## Run

### Web app

```bash
streamlit run app.py
```

Open the URL it prints (usually <http://localhost:8501>), then:

1. Optionally, enter names and terms used in the meeting in the sidebar under **Custom vocabulary**, comma-separated (for example `Priya, Kubernetes, PyTorch`). Refinement uses these spellings when a word sounds like one of them.
2. Upload a recording and click **Process**.
3. Watch each stage's status. If the file is empty, silent, unreadable or in an unsupported format, you get an error straight away and nothing else runs.
4. Browse the tabs: Raw Transcript, Refined Transcript, Diff, Corrections, Minutes and Decisions, Action Items. Use the search box to highlight words in the transcripts.
5. Download any of the five output files.

The first run downloads the Whisper model, so the Transcribing step takes longer that time.

### Command line

The whole pipeline, writing all output files to a folder:

```bash
python pipeline.py meeting.wav --vocab "Priya, Kubernetes" --out output
```

Each stage can also be run on its own:

```bash
python validate.py meeting.wav                         # stage 0: check the file
python stt.py meeting.wav --json raw.json              # stage 1: raw transcript
python refine.py raw.json --vocab "Priya" --json refined.json   # stage 2
python minutes.py refined.json --json record.json      # stage 3
```

### Output files

| File | Contents |
|---|---|
| `raw_transcript.txt` | Whisper's text exactly as produced |
| `refined_transcript.txt` | The transcript after term correction |
| `meeting_record.md` | Summary, minutes, key decisions, action items, open questions, review notes |
| `meeting_record.json` | The same record as JSON, with identical decisions and action items |
| `subtitles.srt` | Subtitles from the raw segments and their timestamps |

### Tests

```bash
python -m pytest
```

The tests use fake models, so they need neither an API key nor the Whisper download.

## Configuration

All settings go in `.env`. Leave a value empty to use the default.

| Variable | Default | Meaning |
|---|---|---|
| `GEMINI_API_KEY` | (required) | Gemini API key for the two LLM stages |
| `WHISPER_MODEL_SIZE` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`: larger is more accurate and slower |
| `WHISPER_DEVICE` | `auto` | `auto` uses a CUDA GPU if present, else the CPU |
| `WHISPER_COMPUTE_TYPE` | `int8` on CPU, `float16` on GPU | Precision for Whisper |
| `WHISPER_BEAM_SIZE` | `5` | Lower is faster, higher can be more accurate |
| `REFINE_MODEL` | `gemini-3.5-flash-lite` | Gemini model for stage 2 (refinement) |
| `REFINE_EFFORT` | `medium` | Thinking level: `minimal` / `low` / `medium` / `high` |
| `REFINE_CHUNK_CHARS` | `6000` | Longer transcripts are refined in chunks of this size |
| `MINUTES_MODEL` | `gemini-3.1-flash-lite` | Gemini model for stage 3 (documentation); keep it different from `REFINE_MODEL` |
| `MINUTES_EFFORT` | `high` | Thinking level: `minimal` / `low` / `medium` / `high` |

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| `Stage 2 (refinement) failed: No Gemini API key found.` | Create `.env` and set `GEMINI_API_KEY` (see Setup, step 4). |
| `... is temporarily unavailable (503: ... high demand ...)` | Google's servers are busy. The request was already retried automatically; try again in a few minutes, or set another model in `.env`. |
| `... quota or rate limit was reached ...` | The key's quota for that model is used up, or the model is not in your plan (Pro models usually need billing). Wait, or choose another model. |
| `Stage 1 (transcription) failed: No speech was detected in this recording.` | The file has sound but no recognisable speech. Check it is the right recording. |
| `Model '...' is not available` | Check `REFINE_MODEL` / `MINUTES_MODEL` in `.env`. Old names such as `claude-...` or `gemini-2.5-flash` no longer work. |
| A yellow warning that part of the refined text was rejected | Refinement changed the length, numbers or negations of that part, so the raw text is shown there instead. The rest of the results are still valid. |
| Hugging Face warnings about symlinks or `HF_TOKEN` on first run | Harmless. They come from the model download. |
| Transcription is slow | Use `WHISPER_MODEL_SIZE=base`, or `WHISPER_BEAM_SIZE=1`, or a GPU. |

## Limitations

- English recordings only.
- No speaker identification. An action item only gets an owner when the transcript names that person, so "I'll do it" stays `unspecified`.
- The LLM stages need internet access and an API key; validation and transcription run fully offline once the model is downloaded.
