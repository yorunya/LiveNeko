"""Environment checker for the LiveNeko Tauri app.

Prints a JSON report of the Python environment: whether CUDA is available and whether the required Python libraries are importable. The interpreter version and ffmpeg presence are probed by the Rust backend directly, so they are not repeated here. The libraries needed for the local pipeline (audio_server.py visual_server.py) are reported together; DeepFilterNet (df) powers the CUDA audio denoiser. The LLM summarization now runs in-process via openai-rust2, so llama_cpp/openai are not required here.
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
    "df",
]


def lib_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "ok")
    except Exception as exc:
        return f"missing ({type(exc).__name__})"


def main():
    libs = {name: lib_version(name) for name in REQUIRED_LIBS}
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        pass
    print(json.dumps({"cuda": cuda, "libraries": libs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
