# LiveNeko Desktop App (Tauri)

A Windows desktop GUI around the LiveNeko multi-modal video summarization
pipelines. Uses Python workers and bundled assets, and exposes a 4-stage
pipeline with batch queue, progress, cancellation, Markdown results, and
settings.

## Requirements
- Windows 10/11 with WebView2 (built-in on Win11).
- System Python (3.10+) with CUDA-enabled `torch`, `torchaudio`, `torchvision`,
  `transformers`, `numpy`, `soundfile`, and `funasr`.
- `ffmpeg` on `PATH`.
- No Python or Git Bash is bundled — the app shells out to the system Python.
- Summarization runs in-process via `openai-rust2` against one of:
  an OpenAI-compatible API, a local **Ollama** server, or a **llama.cpp**
  server. No `llama-cpp-python` / `openai` Python packages are needed.
- `huggingface_hub` / `modelscope` are needed only to download FunASR models
  from inside the app; an existing local model directory works without them.

## FunASR models (user-provided, NOT bundled)
The app no longer bundles ASR/VAD/SPK weights. On first launch (and in
Settings) you configure:

- **ASR (required)** — one of:
  - `SenseVoiceSmall`
  - `Fun-ASR-Nano`
  - `Paraformer-zh-streaming`
  - `Qwen3-ASR` over an online API (default endpoint:
    `https://dashscope.aliyuncs.com/compatible-mode/v1`, model
    `qwen3-asr-flash`). Both the OpenAI-compatible `/chat/completions` API and
    the native DashScope multimodal-generation API
    (`/api/v1/services/aigc/multimodal-generation/generation`) are supported;
    the app derives the native URL from the base URL and automatically picks
    the request shape that works for the chosen model (e.g.
    `qwen-audio-3.0-asr-flash` / `fun-asr-flash-*` are native-only).
- **VAD (required)** — an `fsmn-vad` compatible model directory.
- **SPK (optional)** — a `cam++` compatible model directory. Without it,
  speaker identification is disabled and every utterance is labelled `other`.

Each model can be:
- an existing local directory (validated for the expected files/config and
  model type), or
- downloaded from **Hugging Face** or **ModelScope** into
  `<app_data>/funasr-models/<asr|vad|spk>` (or a directory you pick).

ASR/VAD model paths (and the Qwen3-ASR endpoint fields) are validated before a
run starts; the Qwen3-ASR connection can be probed with *Test connection*.

## Bundled (in the installer)
- `yt-dlp`-style downloading is implemented in Rust (no exe bundled).
- The DeepFilterNet3 ONNX model (`model/DeepFilterNet3_onnx.tar.gz` for
  in-process Rust denoising), `prompt.md`, and the Python worker scripts
  (`scripts/`).

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
Cross-compiling the Windows installer from Linux is documented in the repo
root `AGENTS.md` (`cargo xwin` shim + `--target x86_64-pc-windows-msvc`).

## First launch
1. The app runs an environment + asset check once; the result is cached in
   `env_report.json` (`env_checked` in config) so later launches skip it.
2. **Configure the FunASR models** — choose an ASR type (or the Qwen3-ASR API),
   a local directory or a Hugging Face / ModelScope download, and the required
   `fsmn-vad` VAD model. The SPK (`cam++`) model is optional.
3. If no VideoNeko model directory is set, the wizard asks you to pick one
   (the directory holding your fine-tuned `config.json` + `model.safetensors`
   + `preprocessor_config.json`).
4. Pick a summarization engine: an OpenAI-compatible API, a local Ollama
   server, or a llama.cpp server (base URL + model).
5. Settings are saved to `%APPDATA%\com.liveneko.desktop\config.json`.

## Pipeline
Each queued video runs through the 4 stages with progress and logs. At the start
of a run the app launches **resident model servers** (`audio_server.py` +
`visual_server.py`) over stdin/stdout IPC — the VAD/ASR/SPK and VideoNeko
models load **once** and are reused for every queued video (no per-video
reload). The Rust backend writes the configured model paths / ASR backend to
`work/audio_server.json` and passes it to `audio_server.py --config`.

1. `1/4 Video Input` — for a Bilibili URL, the app first probes how many videos
   the URL yields. A multi-part URL is downloaded fully (`--yes-playlist`, parts
   ordered `001_…`, `002_…`) but the parts are **not** merged with ffmpeg (that
   was slow). Local files are used as-is. The download resolution is set in
   Settings (360P / 480P / 720P / 1080P, default 720P).
2. `2/4 ASR` — ffmpeg extracts 48 kHz mono; Rust denoises it in-process with
   DeepFilterNet (ONNX via the `df` crate), downsamples to 16 kHz, then the
   resident `audio_server.py` runs the configured ASR (local FunASR model or
   Qwen3-ASR API) plus optional speaker labelling and returns `asr.txt`.
3. `3/4 Visual` — ffmpeg (hardware-accelerated) decodes 1 fps frames; the
   visual server (VideoNeko, resident) classifies them and returns `visual.txt`.
   Stages 2 and 3 run **in parallel** per part. Multi-part inputs are analysed
   one part at a time (in `p0N` order), then the per-part `asr.txt`/`visual.txt`
   are merged and their timestamps realigned onto the full video timeline.
4. `4/4 Summary` — the app calls the configured LLM endpoint in-process
   (`openai-rust2`) combining the video **title** + both files → `summary.md`.

The servers are shut down when the queue finishes (or when Stop Analysis is
pressed — their PIDs are killed too).

Results land in `%APPDATA%\com.liveneko.desktop\results\<title>\` — one directory
per video named after its resolved title. Each directory holds `summary.md`,
`asr.txt`, `visual.txt`, (when the model reasoned) `thinking.txt`, and the video
file.

The Results page renders the summary as HTML: timestamped entries are shown as
styled blocks, the model's thinking is folded inside a collapsible
`<details>` (closed by default), and each result has **Re-summarize** and
**Delete** buttons.
