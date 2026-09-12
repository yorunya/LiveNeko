# LiveNeko — Livestream Video Summarization Desktop App

[English](README.md) | [简体中文](README-zh.md)

LiveNeko is a Windows desktop application that automatically summarizes
livestream VODs (e.g. Bilibili) by combining **speech recognition**, **visual
scene detection**, and an **LLM summary engine**. It downloads a video from a
URL or uses a local file as input, and produces a timestamped Markdown summary.

## Requirements

- Windows 10/11 with WebView2 (built in on Windows 11).
- System **Python 3.10+** on `PATH` with CUDA-enabled libraries — the app calls
  the system Python, it is not bundled:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers numpy soundfile funasr
  ```
  `huggingface_hub` / `modelscope` are needed only to download models from
  inside the app.
- `ffmpeg` on `PATH`:
  ```bash
  winget install Gyan.FFmpeg
  winget install Python.Python.3.12
  ```

## Models (user-provided — not bundled)

On first launch a wizard (also available in **Settings**) configures:

- **ASR** (required): a local directory, or an in-app Hugging Face / ModelScope
  download of `SenseVoiceSmall`, `Fun-ASR-Nano`, or
  `Paraformer-zh-streaming`; or the online **Qwen3-ASR** API.
- **VAD**: nothing to set up — the app bundles the Silero VAD model and runs it
  natively on CPU.
- **SPK** (optional): a `cam++` model directory for speaker identification.
  Without it, every utterance is labelled `other`.
- **Visual model (VideoNeko)**: the directory holding your fine-tuned ViT
  `config.json` + `model.safetensors` + `preprocessor_config.json`.

Video downloading is implemented in-process (no `yt-dlp.exe` required). Choose
the summarization engine in Settings: an OpenAI-compatible API, a local
**Ollama** server, or a **llama.cpp** server.

## Build

```bash
cd tauri-app
npm install
npm run tauri build
```

The installer is written to
`tauri-app/src-tauri/target/release/bundle/nsis/LiveNeko_<version>_x64-setup.exe`
(e.g. `LiveNeko_0.1.10_x64-setup.exe`, ~9.6 MB, unsigned — since it is
unsigned, Windows shows a SmartScreen warning on first run).

Settings live in `%APPDATA%\com.liveneko.desktop\config.json` (always there).
Everything else — `results\<title>\` (`summary.md`, `asr.txt`, `visual.txt`),
`funasr-models\` and `spk\` — goes to the **data directory**, which is chosen in
the first-run wizard and can be changed later in Settings; changing it moves the
existing data. It defaults to `%APPDATA%\com.liveneko.desktop\`.

## Finetune your model

The visual base model is `google/vit-base-patch16-224`; the app uses the
fine-tuned weights you point it at. `sample.py` extracts 1 fps frames from
`data/` into `dataset/<tag>/`, and `train.py` fine-tunes on `dataset/` into
`model/` — select that directory as the Visual model in the app. A sample
fine-tune for the VTuber `Ace Taffy` is available in the releases; `spk/` holds a
reference voiceprint for `Ace Taffy` (used by the standalone `AudioNeko.py`
pipeline, not by the app).
