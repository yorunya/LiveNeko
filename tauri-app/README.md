# LiveNeko Desktop App (Tauri)

A Windows desktop GUI around the LiveNeko multi-modal video summarization
pipelines. Uses Python workers and bundled assets, and exposes a 4-stage
pipeline with batch queue, progress, cancellation, Markdown results, and
settings.

## Requirements
- Windows 10/11 with WebView2 (built-in on Win11).
- System Python (3.10+) with CUDA-enabled `torch`, `torchaudio`, `torchvision`,
  `transformers`, `numpy`, `soundfile`, `funasr`, and DeepFilterNet (`df`).
- `ffmpeg` on `PATH`.
- No Python or Git Bash is bundled — the app shells out to the system Python.
- Summarization runs in-process via `openai-rust2` against one of:
  an OpenAI-compatible API, a local **Ollama** server, or a **llama.cpp**
  server. No `llama-cpp-python` / `openai` Python packages are needed.

## Bundled (in the installer)
- `yt-dlp.exe` and the DeepFilterNet3 model (audio denoising via CUDA Python).
- VAD/ASR/SPK model weights (`model/SenseVoiceSmall`, `model/fsmn-vad`,
  `model/cam++`), speaker references (`spk/`), `prompt.md`, and the Python
  worker scripts (`scripts/`).

## How to run (dev)
```bash
cd tauri-app
npm install
npm run tauri dev
```

## How to build the installer
```bash
cd tauri-app
npm install
npm run tauri build        # -> src-tauri/target/release/bundle/nsis/LiveNeko_*_x64-setup.exe
```
The installer includes the audio models (~0.9 GB), so the resulting setup exe
is ~1 GB.

## First launch
1. The app runs an environment + asset check once; the result is cached in
   `env_report.json` (`env_checked` in config) so later launches skip it.
2. If no VideoNeko model directory is set, a wizard asks you to pick one
   (the directory holding your fine-tuned `config.json` + `model.safetensors`
   + `preprocessor_config.json`).
3. Pick a summarization engine: an OpenAI-compatible API, a local Ollama
   server, or a llama.cpp server (base URL + model).
4. Settings are saved to `%APPDATA%\com.liveneko.desktop\config.json`.

## Pipeline
Each queued video runs through the 4 stages with progress and logs. At the start
of a run the app launches **resident model servers** (`audio_server.py` +
`visual_server.py`) over stdin/stdout IPC — the DeepFilterNet/VAD/ASR/SPK and
VideoNeko models load **once** and are reused for every queued video (no
per-video reload).

1. `1/4 Video Input` — for a Bilibili URL, the app first probes how many videos
   the URL yields (`yt-dlp --print %(title)s`). A multi-part URL is downloaded
   fully (`--yes-playlist`, parts ordered `001_…`, `002_…`) but the parts are
   **not** merged with ffmpeg (that was slow). Local files are used as-is. The
   download resolution is set in Settings (360P / 480P / 720P / 1080P, default
   720P); the matching `-f "bestvideo[height<=N]+bestaudio/best[height<=N]"`
   selector is passed to yt-dlp.
2. `2/4 ASR` — ffmpeg extracts 48 kHz mono; the resident `audio_server.py`
   denoises with DeepFilterNet (CUDA), downsamples to 16 kHz in-memory, then
   runs VAD/ASR/SPK and returns `asr.txt`.
3. `3/4 Visual` — ffmpeg (hardware-accelerated) decodes 1 fps frames; the
   visual server (VideoNeko, resident) classifies them and returns `visual.txt`.
   Stages 2 and 3 run **in parallel** per part. Multi-part inputs are analysed
   one part at a time (in `p0N` order), then the per-part `asr.txt`/`visual.txt`
   are merged and their timestamps realigned onto the full video timeline.
4. `4/4 Summary` — the app calls the configured LLM endpoint in-process
   (`openai-rust2`) combining the video **title** + both files → `summary.md`.
   The title (the video's prefix) is sent to the model so it can identify the
   main activities; `prompt.md` explains the title formats.

The servers are shut down when the queue finishes (or when Stop Analysis is
pressed — their PIDs are killed too).

Results land in `%APPDATA%\com.liveneko.desktop\results\<title>\` — one directory
per video named after its resolved title. For URL downloads the title is probed
with `yt-dlp --print %(title)s` and the video is downloaded **into that
directory**; for local files the directory is created from the file name. Each
directory holds `summary.md`, `asr.txt`, `visual.txt`, (when the model reasoned)
`thinking.txt`, and the video file. Stop Analysis kills the active subprocess
(Python model servers) and cleans up.

The Results page renders the summary as HTML: timestamped entries are shown as
styled blocks, the model's thinking is folded inside a collapsible
`<details>` (closed by default), and each result has **Re-summarize** (re-runs
only stage 4 using the stored `asr.txt` + `visual.txt` with the engine/model/API
currently set in Settings) and **Delete** (removes the whole `results/<title>/`
directory) buttons.
