"""Resident IPC visual worker for the LiveNeko Tauri app.

Launched once by the Rust backend at the start of a pipeline run. Loads the
VideoNeko ViT model a single time, then stays alive reading JSON requests on stdin and writing JSON responses on stdout until told to shut down. The model is NOT reloaded between requests (switching models is done via a "load" cmd).

This worker does ONLY model inference. Video decoding and resizing are done by the Rust backend via ffmpeg (hardware-accelerated), which writes the sampled frames to a raw RGB24 blob (one frame = height*width*3 bytes). The Rust backend also owns the smoothing, interval merging, timestamp formatting, and result-file writing; it receives raw per-second label predictions back.

Protocol (newline-delimited JSON on stdin/stdout):
  Request:  {"cmd":"predict","id":"<id>","input":"<frames.raw>"}
  Response: {"cmd":"predict","id":"<id>","ok":true,"preds":["<label>", ...]}
            {"cmd":"predict","id":"<id>","ok":false,"error":"..."}
  Switch model: {"cmd":"load","model_dir":"<dir>"}   (optional, for per-seq models)
  Shutdown: {"cmd":"shutdown"}
Progress is emitted on stdout as {"progress":N} lines while working.
"""
import argparse
import contextlib
import json
import logging
import sys

import numpy as np
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BATCH_SIZE = 512

# module-level state so switching models replaces the in-memory model
_model = None
_id2label = {}
_size_hw = None
_mean = None
_std = None
_device = None


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def quiet_stdout():
    # Redirect any stray prints (e.g. transformers progress bars) to stderr so stdout stays a clean JSON channel for IPC.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


def load_model(model_dir):
    global _model, _id2label, _size_hw, _mean, _std
    with quiet_stdout():
        processor = ViTImageProcessor.from_pretrained(model_dir)
        _model = ViTForImageClassification.from_pretrained(model_dir).to(_device)
    _model.eval()
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}
    size = processor.size
    _size_hw = (size.get("height") or size["shortest_edge"],
                size.get("width") or size["shortest_edge"])
    _mean = torch.tensor(processor.image_mean, device=_device).view(1, 3, 1, 1)
    _std = torch.tensor(processor.image_std, device=_device).view(1, 3, 1, 1)
    log.info(f"Labels: {_id2label}")


def predict_batch(buf, n):
    # buf is already RGB (ffmpeg writes rgb24); just normalize.
    x = buf[:n].to(_device, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = (x - _mean) / _std
    if _device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = _model(pixel_values=x).logits
    else:
        with torch.inference_mode():
            logits = _model(pixel_values=x).logits
    return logits.argmax(-1).tolist()


def predict_video(blob_path, h, w, on_progress=None):
    """Read the raw RGB24 frame blob and classify it in batches."""
    frame_bytes = h * w * 3
    data = np.fromfile(blob_path, dtype=np.uint8)
    n = data.size // frame_bytes
    data = data[:n * frame_bytes].reshape(n, h, w, 3)

    preds = []
    buf = torch.empty((BATCH_SIZE, h, w, 3), dtype=torch.uint8)
    if _device.type == "cuda":
        buf = buf.pin_memory()
    total = max(n, 1)
    for i in range(0, n, BATCH_SIZE):
        batch = data[i: i + BATCH_SIZE]
        m = batch.shape[0]
        buf[:m].copy_(torch.from_numpy(batch))
        preds.extend(predict_batch(buf, m))
        if on_progress:
            on_progress(min(100, int((i + BATCH_SIZE) / total * 100)))
    return preds


def handle_predict(req):
    rid = req.get("id", "")
    input_blob = req.get("input", "")
    if not input_blob:
        send({"cmd": "predict", "id": rid, "ok": False,
              "error": "missing input"})
        return
    try:
        h, w = _size_hw
        # Rust reports 0..20% for the ffmpeg decode; classify fills 20..100%.
        preds = predict_video(
            input_blob, h, w, on_progress=lambda n: send({"progress": 20 + int(n * 0.80)})
        )
        labels = [_id2label[p] for p in preds]
        send({"cmd": "predict", "id": rid, "ok": True, "preds": labels})
    except Exception as e:
        log.exception("predict failed")
        send({"cmd": "predict", "id": rid, "ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="dir with fine-tuned ViT weights")
    args = ap.parse_args()

    global _device
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {_device}")

    # The Rust backend writes UTF-8 bytes to stdin; on Windows the default stdin encoding is the system locale (e.g. GBK), which would corrupt the CJK characters in paths. Force UTF-8 for the IPC channel.
    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        load_model(args.model_dir)
    except Exception as e:
        log.exception("model load failed")
        send({"cmd": "ready", "ok": False, "error": str(e)})
        return
    send({"cmd": "ready", "ok": True, "engine": "visual"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send({"cmd": "unknown", "id": "", "ok": False, "error": "bad json"})
            continue
        cmd = req.get("cmd", "")
        if cmd == "shutdown":
            log.info("shutdown")
            break
        elif cmd == "load":
            try:
                load_model(req.get("model_dir", ""))
                send({"cmd": "load", "id": req.get("id", ""), "ok": True})
            except Exception as e:
                send({"cmd": "load", "id": req.get("id", ""), "ok": False, "error": str(e)})
        elif cmd == "predict":
            handle_predict(req)
        else:
            send({"cmd": cmd, "id": req.get("id", ""), "ok": False,
                  "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
