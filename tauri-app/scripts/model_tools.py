"""FunASR model helper for the LiveNeko Tauri app.

Used by the Rust backend for three jobs. None of them imports torch/funasr, so
they are fast and can never trigger a model load:

  check      validate one or more configured model directories
             (path, files, config, and whether the directory looks like the
             selected model type)
  download   fetch a model snapshot from Hugging Face or ModelScope
  check-api  probe a Qwen3-ASR (OpenAI-compatible) endpoint with a short
             silent WAV and report whether the connection/credentials work

Every subcommand prints exactly one JSON object on stdout; diagnostics go to
stderr. The Rust side parses stdout.

Usage:
  python model_tools.py check --config <checks.json>
  python model_tools.py download --source huggingface --model-id <id> --dest <dir>
  python model_tools.py check-api --base-url <url> --api-key <key> --model <name>
"""

import argparse
import io
import json
import os
import sys
import wave


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def emit(obj):
    """Print one JSON object on stdout (the IPC/result channel)."""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(msg):
    sys.stderr.write(str(msg) + "\n")
    sys.stderr.flush()


class ToolError(Exception):
    pass


WEIGHT_EXTS = (".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".onnx", ".gguf")
CONFIG_NAMES = ("config.yaml", "config.yml", "configuration.json", "config.json")

TYPE_HINTS = {
    "sensevoice-small": ("sensevoice", "sense_voice"),
    "fun-asr-nano": ("funasrnano", "fun_asr_nano", "fun-asr-nano", "qwen3"),
    "paraformer-zh-streaming": ("paraformerstreaming", "paraformer_streaming"),
    "cam++": ("campplus", "cam++", "cam_plus"),
}


def _walk_files(root):
    """Yield (relative_path, absolute_path, size) for real files below root."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                size = 0
            rel = os.path.relpath(path, root)
            yield rel, path, size


def _read_config_text(root):
    for name in CONFIG_NAMES:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    return name, f.read()
            except OSError:
                return name, ""
    return None, ""


def check_dir(kind, model_type, directory):
    """Validate one model directory. Returns a JSON-serializable dict."""
    out = {
        "kind": kind,
        "type": model_type,
        "dir": directory or "",
        "ok": False,
        "errors": [],
        "warnings": [],
        "files": 0,
        "bytes": 0,
    }
    directory = (directory or "").strip()
    if not directory:
        out["errors"].append("model directory is not set")
        return out
    if not os.path.isdir(directory):
        out["errors"].append(f"directory does not exist: {directory}")
        return out

    file_names = []
    total = 0
    weights = []
    for rel, path, size in _walk_files(directory):
        file_names.append(rel)
        total += 1
        out["bytes"] += size
        if os.path.splitext(rel)[1].lower() in WEIGHT_EXTS and size > 0:
            weights.append(rel)
    out["files"] = total
    if total == 0:
        out["errors"].append("directory contains no files")
        return out

    config_name, config_text = _read_config_text(directory)
    if not config_name:
        out["errors"].append(
            "missing model config (expected config.yaml or configuration.json)"
        )
    if not weights:
        out["errors"].append(
            "no model weight file found (*.pt/*.safetensors/*.bin/*.onnx)"
        )

    # Type compatibility: check the config text for the expected class/name.
    hints = TYPE_HINTS.get(model_type, ())
    lower = config_text.lower()
    if config_name and hints and not any(h in lower for h in hints):
        # Clearly a different supported type? Then this is an error, not a warning.
        others = [
            other
            for other, other_hints in TYPE_HINTS.items()
            if other != model_type and any(h in lower for h in other_hints)
        ]
        if others:
            out["errors"].append(
                f"model directory looks like {', '.join(others)}, not '{model_type}'"
            )
        else:
            out["warnings"].append(
                f"config does not mention the selected model type '{model_type}' — "
                "make sure the directory matches the chosen type"
            )

    # Type specific expectations.
    if model_type == "sensevoice-small":
        if "model.pt" not in file_names:
            out["warnings"].append("SenseVoiceSmall usually ships a model.pt weight file")
    elif model_type == "paraformer-zh-streaming":
        if "model.pt" not in file_names and not weights:
            out["errors"].append("Paraformer checkpoint weight file is missing")
    elif model_type == "cam++":
        if not any("campplus" in os.path.basename(w).lower() for w in weights):
            out["warnings"].append(
                "cam++ usually ships a campplus*.bin weight file"
            )

    out["ok"] = not out["errors"]
    return out


def check_batch(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        checks = json.load(f)
    items = {}
    errors = []
    warnings = []
    for slot in ("asr", "spk"):
        spec = checks.get(slot)
        if spec is None or not spec.get("enabled", False):
            items[slot] = {"ok": True, "skipped": True, "errors": [], "warnings": []}
            continue
        if spec.get("kind") == "api":
            slot_errors = []
            if not (spec.get("baseUrl") or "").strip():
                slot_errors.append("API base URL is not set")
            if not (spec.get("model") or "").strip():
                slot_errors.append("API model name is not set")
            item = {
                "kind": "api",
                "ok": not slot_errors,
                "errors": slot_errors,
                "warnings": [] if (spec.get("apiKey") or "").strip()
                else ["API key is empty (some endpoints do not require one)"],
            }
        else:
            item = check_dir(spec.get("kind", slot), spec.get("type", ""), spec.get("dir", ""))
        items[slot] = item
        errors.extend(item.get("errors", []))
        warnings.extend(item.get("warnings", []))
    return {"ok": len(errors) == 0, "items": items, "errors": errors, "warnings": warnings}


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def download_hf(model_id, dest, revision):
    os.makedirs(dest, exist_ok=True)
    revision = revision or None
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
    except ImportError:
        raise ToolError(
            "huggingface_hub is not installed — run: python -m pip install -U huggingface_hub"
        )
    kwargs = {"repo_id": model_id}
    if revision:
        kwargs["revision"] = revision
    try:
        files = [f for f in list_repo_files(**kwargs) if f != ".gitattributes"]
    except TypeError:
        files = [f for f in list_repo_files(model_id) if f != ".gitattributes"]
    if not files:
        raise ToolError(f"repository '{model_id}' has no files (check the model id)")
    emit({"progress": 1})
    total = len(files)
    for i, name in enumerate(files):
        try:
            hf_hub_download(
                repo_id=model_id, filename=name, revision=revision, local_dir=dest
            )
        except TypeError:
            # older huggingface_hub without local_dir support
            return download_hf_snapshot(model_id, dest, revision)
        emit({"progress": max(1, int((i + 1) * 100 / total))})
    return {"path": dest, "files": total}


def download_hf_snapshot(model_id, dest, revision):
    from huggingface_hub import snapshot_download
    import shutil

    revision = revision or None
    emit({"progress": 1})
    kwargs = {"repo_id": model_id, "local_dir": dest}
    if revision:
        kwargs["revision"] = revision
    try:
        path = snapshot_download(**kwargs)
    except TypeError:
        path = snapshot_download(repo_id=model_id)
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copytree(path, dest, dirs_exist_ok=True)
    emit({"progress": 100})
    return {"path": dest, "files": None}


def download_ms(model_id, dest, revision):
    os.makedirs(dest, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        raise ToolError(
            "modelscope is not installed — run: python -m pip install -U modelscope"
        )
    import shutil

    revision = revision or "master"
    emit({"progress": 2})
    try:
        path = snapshot_download(model_id, revision=revision, local_dir=dest)
    except TypeError:
        # older modelscope only knows cache_dir; download then copy the snapshot.
        path = snapshot_download(model_id, revision=revision, cache_dir=dest)
        if os.path.abspath(path) != os.path.abspath(dest):
            shutil.copytree(path, dest, dirs_exist_ok=True)
    emit({"progress": 100})
    return {"path": dest, "files": None}


def run_download(source, model_id, dest, revision):
    source = (source or "").lower()
    if source in ("hf", "huggingface"):
        return download_hf(model_id, dest, revision)
    if source in ("ms", "modelscope"):
        return download_ms(model_id, dest, revision)
    raise ToolError(f"unknown model source '{source}' (expected huggingface or modelscope)")


# --------------------------------------------------------------------------
# Qwen3-ASR API check
# --------------------------------------------------------------------------

def _silence_wav(seconds=0.3):
    """Raw WAV bytes for a short silent 16 kHz mono clip."""
    buf = io.BytesIO()
    frames = b"\x00\x00" * int(16000 * seconds)
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(frames)
    return buf.getvalue()


def check_api(base_url, api_key, model, language):
    # Imported lazily so `check`/`download` do not need the API helper.
    from qwen3_asr_client import check_connection, wav_data_url

    import urllib.error
    import urllib.request

    base = (base_url or "").rstrip("/")
    models_ok = False
    if base:
        # Cheap credential probe first: OpenAI-compatible /models where available.
        try:
            headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
            request = urllib.request.Request(base + "/models", headers=headers)
            with urllib.request.urlopen(request, timeout=30) as resp:
                if resp.status == 200:
                    models_ok = True
                    log("credentials accepted by GET /models")
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                return {"ok": False, "errors": [f"authentication failed (HTTP {exc.code})"]}
            log(f"GET /models returned HTTP {exc.code}; testing an ASR request instead")
        except Exception as exc:  # noqa: BLE001 - some endpoints have no /models route
            log(f"GET /models failed ({exc}); testing an ASR request instead")

    result = check_connection(base, api_key, model, wav_data_url(_silence_wav()), language=language)
    if result.get("ok"):
        log(f"ASR request OK via {result.get('method')}")
        return result
    # A model-level rejection of the silent probe (empty-body HTTP 400) does not
    # mean the connection is broken: qwen-audio-3.0-asr-flash, for example,
    # requires ~2 seconds of real speech and rejects silence at any length.
    if models_ok and not result.get("fatal", True):
        log("credentials verified; the model rejected the silent probe clip")
        return {
            "ok": True,
            "method": "models",
            "warning": "probe_audio_rejected",
        }
    # A native DashScope URL has no `/models` route, but a non-fatal model-level
    # rejection still proves the endpoint is reachable.
    if "/api/v1/services/" in base and not result.get("fatal", True):
        log("endpoint verified; the model rejected the silent probe clip")
        return {
            "ok": True,
            "method": "native",
            "warning": "probe_audio_rejected",
        }
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_check = sub.add_parser("check", help="validate configured model directories")
    p_check.add_argument("--config", required=True, help="path to a checks JSON file")

    p_download = sub.add_parser("download", help="download a model snapshot")
    p_download.add_argument("--source", required=True, help="huggingface | modelscope")
    p_download.add_argument("--model-id", required=True)
    p_download.add_argument("--dest", required=True)
    p_download.add_argument("--revision", default="")

    p_api = sub.add_parser("check-api", help="probe a Qwen3-ASR API endpoint")
    p_api.add_argument("--base-url", required=True)
    p_api.add_argument("--api-key", default="")
    p_api.add_argument("--model", required=True)
    p_api.add_argument("--language", default="")

    args = parser.parse_args()
    try:
        if args.command == "check":
            result = check_batch(args.config)
        elif args.command == "download":
            result = run_download(args.source, args.model_id, args.dest, args.revision)
            result["ok"] = True
        elif args.command == "check-api":
            result = check_api(args.base_url, args.api_key, args.model, args.language)
        else:  # pragma: no cover - argparse rejects unknown commands
            raise ToolError(f"unknown command {args.command}")
    except ToolError as exc:
        emit({"ok": False, "errors": [str(exc)]})
        return
    except Exception as exc:  # noqa: BLE001 - surface every failure as JSON
        emit({"ok": False, "errors": [f"{type(exc).__name__}: {exc}"]})
        return
    emit(result)


if __name__ == "__main__":
    main()
