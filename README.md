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

## Models

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

Choose the summarization engine in Settings: an OpenAI-compatible API, a local
**Ollama** server, or a **llama.cpp** server.

## Build

```bash
cd tauri-app
npm install
npm run tauri build
```

Settings live in `%APPDATA%\com.liveneko.desktop\config.json` (always there).
Everything else — `results\<title>\` (`summary.md`, `asr.txt`, `visual.txt`),
`funasr-models\` and `spk\` — goes to the **data directory**, which is chosen in
the first-run wizard and can be changed later in Settings; changing it moves the
existing data. It defaults to `%APPDATA%\com.liveneko.desktop\`.

## Finetune your model

The visual base model is `google/vit-base-patch16-224`; the app uses the
fine-tuned weights you point it at. Two repo scripts do the whole job:
`sample.py` extracts training frames from annotated clips, `train.py`
fine-tunes on them.

### Step 0 — Prerequisites

- Python 3.10+ with a CUDA build of PyTorch (a CPU fallback works but is slow):
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install transformers opencv-python pillow tqdm
  ```
- ~350 MB of disk for the dataset frames and a few GB of free VRAM (or RAM on
  CPU).
- The first run of `train.py` downloads the base model from Hugging Face
  (~350 MB). If the network is blocked, prefix the command with `proxychains`
  as in the script docstrings; the download is cached in `~/.cache/huggingface`
  and never needed again.

### Step 1 — Put annotated clips into `data/`

Every clip is an `.mp4` with a same-stem `.txt` next to it:

```text
data/
├── taffy_stream_0102.mp4
├── taffy_stream_0102.txt
├── other_stream_0311.mp4
└── other_stream_0311.txt
```

The `.txt` holds either **one label for the whole video**:

```text
watch anime
```

…or **time-segmented labels**, one per line:

```text
[00:00:00-00:01:39] game
[00:01:41-00:03:56] live2d
[00:03:57-00:05:54] game
```

Segment bounds are **start-inclusive / end-exclusive**; seconds not covered by
any segment are skipped during sampling. Timestamps may be `hh:mm:ss`,
`mm:ss`, or bare seconds. The label set is derived from the annotations — to
add a category (e.g. `sing`), add a labelled clip containing it. Mix both
formats freely across clips.

### Step 2 — Extract frames: `python sample.py`

```bash
python sample.py
```

This samples the clips at 1 fps (evenly thinned so no video contributes more
than `MAX_FRAMES_PER_VIDEO` = 300 frames) and writes JPEGs per label:

```text
dataset/
├── game/taffy_stream_0102_s000000.jpg …
└── live2d/taffy_stream_0102_s000101.jpg …
```

Expected console output, one line per clip:

```text
taffy_stream_0102.mp4 (3 segments): 214 frames, 45s outside segments skipped
other_stream_0311.mp4 [game]: 292 frames
```

Skipped-seconds are normal: they are stretches no segment covers. If a clip is
missing its `.txt` it is skipped with a warning. Parameters are the constants
at the top of `sample.py`: `DATA_DIR`, `DATASET_DIR`, `MAX_FRAMES_PER_VIDEO`
(keep the cap — clip lengths differ greatly and uncapped sampling skews
classes). You can fix labeling mistakes afterwards by moving/deleting frames in
`dataset/<tag>/` directly — no need to re-cut videos.

### Step 3 — Fine-tune: `python train.py`

```bash
python train.py
```

It reads `dataset/` (one class per subfolder; empty folders are skipped with a
warning), replaces the ViT classifier head with your label set, and trains:

```text
Device: cuda
Labels: {'game': 0, 'live2d': 1}
game: 512 frames
live2d: 388 frames
Epoch 1/5  train_loss=0.5123  val_acc=0.901
…
Epoch 5/5  train_loss=0.0187  val_acc=0.987
```

Watch `val_acc` climb — that number is your quality signal. Parameters are the
constants at the top of `train.py`: `EPOCHS` (5), `BATCH_SIZE` (32), `LR`
(2e-5), `VAL_FRACTION` (0.4 — 40 % of frames are held out for validation;
very small datasets should lower it), `SEED` (42, reproducible split). Takes a
few minutes on a GPU. The fine-tuned weights and processor are written to
`model/`:

```text
model/config.json
model/model.safetensors
model/preprocessor_config.json
```

### Step 4 — Verify

1. The `model/` directory above is exactly what the app expects: select it as
   the **Visual model** in the first-run wizard or **Settings → VideoNeko
   Model** (the path is stored in `config.json`).
2. End-to-end check in the app: run a summarization on any stream VOD and
   open `results\<title>\visual.txt` in the data directory — one
   `[hh:mm:ss-hh:mm:ss] tag` per line, the same syntax as the Step 1
   annotations:
   ```text
   [00:00:00-00:01:38] game
   [00:01:38-00:03:55] live2d
   ```
   Check that the tags match what you see in the video.
3. Re-training: add clips to `data/`, re-run `sample.py`, then `train.py`
   (it overwrites `model/`). Note `sample.py` only adds/overwrites frames —
   if you changed or removed annotations, delete `dataset/` (or the affected
   `dataset/<tag>/*.jpg`) first so stale frames do not leak into training.
   Re-pick the directory in the app only if you moved it.

A sample fine-tune for the VTuber `Ace Taffy` is available in the releases.
