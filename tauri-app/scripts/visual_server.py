"""Resident IPC visual worker for the LiveNeko Tauri app.

Launched once by the Rust backend at the start of a pipeline run. Loads the
VideoNeko ViT model a single time, then stays alive reading JSON requests on stdin and writing JSON responses on stdout until told to shut down. The model is NOT reloaded between requests.

This worker does ONLY model inference. Video decoding and resizing are done by the Rust backend via ffmpeg (hardware-accelerated), which writes the sampled frames to a raw RGB24 blob (one frame = height*width*3 bytes). The Rust backend also owns the smoothing, interval merging, timestamp formatting, and result-file writing; it receives raw per-second label predictions back.

Protocol (newline-delimited JSON on stdin/stdout):
  Request:  {"cmd":"predict","id":"<id>","input":"<frames.raw>"}
  Response: {"cmd":"predict","id":"<id>","ok":true,"preds":["<label>", ...]}
            {"cmd":"predict","id":"<id>","ok":false,"error":"..."}
  Shutdown: {"cmd":"shutdown"}
Progress is emitted on stdout as {"progress":N} lines while working.
"""
import argparse
import contextlib
import json
import logging
import os
import re
import sys

import numpy as np
import torch
from safetensors import safe_open
from transformers import ViTForImageClassification, ViTImageProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---- Static VRAM budget ----
# The app only ever runs on 8 GB cards. Of the 8 GB, ~2 GB belong to the
# OS/display/driver, leaving USABLE_VRAM for the pipeline. Every model the
# pipeline holds at once is charged against that figure: this worker's ViT
# (taken from its checkpoint size) plus the bundled audio models the audio
# worker keeps resident in parallel, plus a fixed overhead for the two CUDA
# contexts, the audio ASR activations and ffmpeg's NVDEC session. The batch
# size is computed once from these constants — no runtime probing or retry.
USABLE_VRAM = 6 * 1024 ** 3
AUDIO_MODELS_VRAM = int((892.9 + 26.7) * 1024 ** 2)  # SenseVoiceSmall + cam++ (VAD is the native CPU Silero model in Rust)
OTHER_VRAM_OVERHEAD = int(1.5 * 1024 ** 3)
BYTES_PER_FRAME_AT_224 = int(5.5 * 1024 ** 2)  # measured ViT-base fp16 activations at 224x224
BATCH_SIZE = 128  # cap; the computed size never exceeds this
MIN_BATCH = 8

# module-level state so switching models replaces the in-memory model
_model = None
_id2label = {}
_size_hw = None
_mean = None
_std = None
_device = None
# Reusable CPU staging buffer for batched H2D copies. Reusing one buffer
# (instead of allocating per request) avoids the pinned alloc/free churn that
# can make Windows re-register a recycled address range and fail with
# "CUDA error: resource already mapped".
_staging = None
_staging_key = None
_pinned_ok = True
_model_dir = None
_batch = None


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


# transformers v5 restructured the ViT parameter paths; checkpoints saved by
# v4 (e.g. by train.py) keep the legacy names. Pure 1:1 renames — the tensors
# themselves are identical.
_LEGACY_VIT_KEY_REWRITES = (
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.query\."), r"vit.layers.\1.attention.q_proj."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.key\."), r"vit.layers.\1.attention.k_proj."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.attention\.attention\.value\."), r"vit.layers.\1.attention.v_proj."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.attention\.output\.dense\."), r"vit.layers.\1.attention.o_proj."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.intermediate\.dense\."), r"vit.layers.\1.mlp.fc1."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.output\.dense\."), r"vit.layers.\1.mlp.fc2."),
    (re.compile(r"^vit\.encoder\.layer\.(\d+)\.(layernorm_before|layernorm_after)\."), r"vit.layers.\1.\2."),
)


def _rewrite_legacy_vit_keys(keys):
    out = []
    for key in keys:
        for pattern, repl in _LEGACY_VIT_KEY_REWRITES:
            rewritten = pattern.sub(repl, key)
            if rewritten != key:
                key = rewritten
                break
        out.append(key)
    return out


class _DirectLoadUnsupported(Exception):
    pass


def _load_vit_direct(model_dir, device):
    """Build the model on the meta device (no weight memory anywhere) and read
    the safetensors weights straight into GPU memory; the parameters then
    adopt those tensors as-is (load_state_dict assign=True, no copies)."""
    config = ViTForImageClassification.config_class.from_pretrained(model_dir)
    with torch.device("meta"):
        model = ViTForImageClassification(config)
    expected = set(model.state_dict().keys())
    # Read the key layout from the file header first — nothing is materialized.
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        ckpt_keys = list(f.keys())
    remap = None
    if set(ckpt_keys) != expected:
        renamed = _rewrite_legacy_vit_keys(ckpt_keys)
        if set(renamed) != expected:
            raise _DirectLoadUnsupported("checkpoint key layout not recognized")
        remap = dict(zip(ckpt_keys, renamed))
    # Stream one tensor at a time: read from the (mmap-backed) file, transfer
    # to the GPU, drop the CPU copy. Only a single ~10 MB staging tensor is
    # ever resident in system memory, never the whole ~340 MB checkpoint.
    state = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for key in ckpt_keys:
            target = remap[key] if remap else key
            state[target] = f.get_tensor(key).to(device)
    model.load_state_dict(state, strict=True, assign=True)
    unloaded = [n for n, t in model.named_parameters() if t.is_meta]
    unloaded += [n for n, t in model.named_buffers() if t.is_meta]
    if unloaded:
        raise _DirectLoadUnsupported(f"tensors left unloaded: {unloaded[:3]}")
    return model


def _load_vit(model_dir, device):
    """Load the fine-tuned ViT straight onto `device`.

    On CUDA the safetensors weights are read directly into GPU memory, so the
    ~340 MB checkpoint never occupies system RAM. from_pretrained() instead
    materializes the full model on the CPU first and only then copies it to
    the GPU, which can exhaust the Windows pagefile (os error 1455) while the
    audio worker is already resident. Anything this direct path cannot handle
    (CPU target, no model.safetensors, unrecognized key layout) falls back to
    from_pretrained()."""
    st_path = os.path.join(model_dir, "model.safetensors")
    if device.type == "cuda" and os.path.exists(st_path):
        try:
            return _load_vit_direct(model_dir, device)
        except (_DirectLoadUnsupported, RuntimeError, KeyError) as e:
            log.warning(f"direct GPU load failed ({e}); "
                        "falling back to from_pretrained (checkpoint will "
                        "transiently occupy system memory)")
    return ViTForImageClassification.from_pretrained(model_dir).to(device)


def load_model(model_dir):
    global _model, _id2label, _size_hw, _mean, _std, _model_dir
    with quiet_stdout():
        processor = ViTImageProcessor.from_pretrained(model_dir)
        _model = _load_vit(model_dir, _device)
    _model_dir = model_dir
    _model.eval()
    _id2label = {int(k): v for k, v in _model.config.id2label.items()}
    size = processor.size
    _size_hw = (size.get("height") or size["shortest_edge"],
                size.get("width") or size["shortest_edge"])
    _mean = torch.tensor(processor.image_mean, device=_device).view(1, 3, 1, 1)
    _std = torch.tensor(processor.image_std, device=_device).view(1, 3, 1, 1)
    log.info(f"Labels: {_id2label}")


def staging_buffer(h, w):
    """One reusable BATCH_SIZE staging tensor, rebuilt only when the frame
    size changes. Pinned while CUDA is available; falls back to a normal
    buffer if pinning ever fails ("resource already mapped" driver/allocator
    race on Windows) — only the H2D copies get slower, results are identical."""
    global _staging, _staging_key, _pinned_ok
    key = (BATCH_SIZE, h, w, 3)
    if _staging is None or _staging_key != key:
        if _pinned_ok and _device.type == "cuda":
            try:
                _staging = torch.empty(key, dtype=torch.uint8, pin_memory=True)
            except RuntimeError as e:
                log.warning(f"pinned staging buffer unavailable ({e}); "
                            "falling back to unpinned memory for this session")
                _pinned_ok = False
                _staging = None
        if _staging is None:
            _staging = torch.empty(key, dtype=torch.uint8)
        _staging_key = key
    return _staging


def _forward_batch(frames):
    """Normalize a uint8 (N, H, W, 3) GPU tensor and classify it."""
    x = frames.permute(0, 3, 1, 2).float() / 255.0
    x = (x - _mean) / _std
    if _device.type == "cuda":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = _model(pixel_values=x).logits
    else:
        with torch.inference_mode():
            logits = _model(pixel_values=x).logits
    return logits.argmax(-1).tolist()


def compute_batch_size(h, w):
    """Fixed batch size for the 6 GB usable-VRAM budget: usable minus every
    model resident at once (this ViT + the three audio models) minus fixed
    overhead, divided by the per-frame activation cost at the model's input
    size. Pure arithmetic on constants — nothing is probed at runtime."""
    try:
        weights = os.path.getsize(os.path.join(_model_dir, "model.safetensors"))
    except OSError:
        weights = 327 * 1024 ** 2  # fine-tuned ViT-base checkpoint, typical size
    per_frame = BYTES_PER_FRAME_AT_224 * (h * w) / (224 * 224)
    budget = USABLE_VRAM - AUDIO_MODELS_VRAM - weights - OTHER_VRAM_OVERHEAD
    return max(MIN_BATCH, min(BATCH_SIZE, int(budget // per_frame)))


def predict_batch(buf, n):
    # buf is already RGB (ffmpeg writes rgb24); normalize and classify.
    x = buf[:n].to(_device, non_blocking=True)
    return _forward_batch(x)


def predict_video(blob_path, h, w, on_progress=None):
    """Read the raw RGB24 frame blob and classify it in fixed-size batches
    computed once from the static 6 GB usable-VRAM budget."""
    global _batch
    frame_bytes = h * w * 3
    data = np.fromfile(blob_path, dtype=np.uint8)
    n = data.size // frame_bytes
    data = data[:n * frame_bytes].reshape(n, h, w, 3)

    if _batch is None:
        _batch = compute_batch_size(h, w)
        log.info(f"Batch size: {_batch} (static, from {USABLE_VRAM / 2**30:.0f} GB "
                 f"usable VRAM budget)")
    batch = _batch
    buf = staging_buffer(h, w)
    preds = []
    i = 0
    total = max(n, 1)
    while i < n:
        m = min(batch, n - i)
        buf[:m].copy_(torch.from_numpy(data[i:i + m]))
        preds.extend(predict_batch(buf, m))
        i += m
        if on_progress:
            on_progress(min(100, int(i / total * 100)))
    if _device.type == "cuda":
        # Release the activation cache; the audio worker shares this GPU.
        torch.cuda.empty_cache()
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
        elif cmd == "predict":
            handle_predict(req)
        else:
            send({"cmd": cmd, "id": req.get("id", ""), "ok": False,
                  "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
