"""Environment checker for the LiveNeko Tauri app.

Prints a JSON report of the Python environment: whether CUDA is available and
whether the required Python libraries are importable. The interpreter version
and ffmpeg presence are probed by the Rust backend directly, so they are not
repeated here. The libraries needed for the local pipeline
(audio_server.py/visual_server.py) are reported together; noise reduction and
VAD run natively in Rust, so no denoiser/VAD libraries are needed here. The
LLM summarization runs in-process via
openai-rust2, so llama_cpp/openai are not required here. `huggingface_hub` and
`modelscope` are only needed to download FunASR models and are reported
separately.
"""
import importlib
import json

REQUIRED_LIBS = [
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "numpy",
    "soundfile",
    "funasr",
]

# Optional: needed by the model downloader (Settings / first launch).
DOWNLOAD_LIBS = [
    "huggingface_hub",
    "modelscope",
]


def lib_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "ok")
    except Exception as exc:
        return f"missing ({type(exc).__name__})"


def main():
    libs = {name: lib_version(name) for name in REQUIRED_LIBS}
    download_libs = {name: lib_version(name) for name in DOWNLOAD_LIBS}
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    print(json.dumps({"cuda": cuda, "libraries": libs, "downloadLibraries": download_libs},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
