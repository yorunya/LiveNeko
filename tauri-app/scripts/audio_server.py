"""Resident IPC audio worker for the LiveNeko Tauri app.

Launched once by the Rust backend at the start of a pipeline run. Loads the
DeepFilterNet (denoise), VAD, ASR and speaker models a single time, then stays alive reading JSON requests on stdin and writing JSON responses on stdout until told to shut down. Models are NOT reloaded between requests.

The `process` command runs the whole audio chain in one request: denoise the raw 48 kHz wav -> downsample to 16 kHz in-memory -> VAD -> ASR -> speaker labelling, returning raw utterances. This worker does ONLY model inference; the Rust backend owns ffmpeg extraction, text parsing, formatting, and result-file writing.

Protocol (newline-delimited JSON on stdin/stdout):
  Request:  {"cmd":"process","id":"<id>","input":"<raw 48khz.wav>"}
  Response: {"cmd":"process","id":"<id>","ok":true,
             "utterances":[[start_ms,end_ms,"taffy"|"other","<raw tagged text>"], ...]}
            {"cmd":"process","id":"<id>","ok":false,"error":"..."}
  Shutdown: {"cmd":"shutdown"}
Progress is emitted on stdout as {"progress":N} lines: 0..20 for the denoise phase, 20..100 for the ASR phase.
"""
import argparse
import concurrent.futures
import contextlib
import glob
import json
import logging
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPK_LABEL = "taffy"
OTHER_LABEL = "other"
SPK_THRESHOLD = 0.50
SPK_CHUNK_S = 10.0
SPK_MIN_S = 2
SAMPLE_RATE = 16000
ASR_BATCH = 64
# Max seconds of audio packed into one ASR forward pass (VRAM < 8 GB).
ASR_BATCH_SIZE_S = 300
SPK_MIN_SAMPLES = int(SPK_MIN_S * SAMPLE_RATE)
SPK_CHUNK_SAMPLES = int(SPK_CHUNK_S * SAMPLE_RATE)

# Denoise in CHUNK_S-second pieces to keep VRAM flat.
CHUNK_S = 32


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def quiet_stdout():
    """Redirect stray prints (df/funasr banners) to stderr so stdout stays a
    clean JSON channel for IPC."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


# ---- denoise (DeepFilterNet) ----


def enhance_chunked(model, df_state, audio, on_progress=None):
    sr = df_state.sr()
    total = audio.shape[-1]
    chunk = int(sr * CHUNK_S)
    out = torch.zeros_like(audio)
    n = max(1, (total + chunk - 1) // chunk)
    for i in range(0, total, chunk):
        seg = audio[:, i: i + chunk]
        with torch.no_grad():
            out[:, i: i + chunk] = enhance(model, df_state, seg, atten_lim_db=18.0)
        if n > 1 and on_progress:
            on_progress(min(100, int((i + chunk) / total * 100)))
    return out


def filter_downsample(in_path, model, df_state, on_progress=None):
    """Denoise a raw 48 kHz wav and return a 16 kHz float32 numpy array."""
    audio, _ = load_audio(in_path, sr=df_state.sr())
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    audio = audio.cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    enhanced = enhance_chunked(model, df_state, audio, on_progress)
    enhanced = enhanced.squeeze(0).detach()
    src_sr = df_state.sr()
    if src_sr != SAMPLE_RATE:
        if torch.cuda.is_available():
            enhanced = torchaudio.functional.resample(enhanced.cuda(), src_sr, SAMPLE_RATE).cpu()
        else:
            enhanced = torchaudio.functional.resample(enhanced, src_sr, SAMPLE_RATE)
    return enhanced.numpy()


# ---- VAD / ASR / speaker ----


def load_speech(wav_path, vad_model):
    speech, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate {sr} for {wav_path}")
    segments = vad_model.generate(input=speech, fs=SAMPLE_RATE)[0]["value"]
    return speech, segments


def speaker_embeddings(chunks, spk_model):
    results = spk_model.generate(input=chunks)
    embeddings = []
    for res in results:
        e = res["spk_embedding"]
        if torch.is_tensor(e):
            e = e.detach().cpu().numpy()
        e = np.asarray(e, dtype=np.float32).ravel()
        embeddings.append(e / np.linalg.norm(e))
    return embeddings


def build_reference(wav_path, vad_model, spk_model):
    speech, segments = load_speech(wav_path, vad_model)
    chunks = []
    step = SPK_CHUNK_SAMPLES
    for start_ms, end_ms in segments:
        a, b = int(start_ms * SAMPLE_RATE / 1000), int(end_ms * SAMPLE_RATE / 1000)
        for i in range(a, b, step):
            chunk = speech[i: i + step]
            if len(chunk) >= SPK_MIN_SAMPLES:
                chunks.append(chunk)
    if not chunks:
        raise RuntimeError(f"No speech found in reference {wav_path}")
    ref = np.mean(speaker_embeddings(chunks, spk_model), axis=0)
    ref /= np.linalg.norm(ref)
    return ref


def _speaker_embeddings_matrix(chunks, long_idx, spk_model):
    return np.stack(speaker_embeddings([chunks[j] for j in long_idx], spk_model))


def transcribe_samples(speech, segments, asr_model, spk_model, ref_matrix, on_progress=None):
    log.info(f"VAD: {len(segments)} utterances")
    utterances = []
    total = max(len(segments), 1)

    batches = []
    for i in range(0, len(segments), ASR_BATCH):
        batch = segments[i: i + ASR_BATCH]
        chunks = [speech[int(s * SAMPLE_RATE / 1000): int(e * SAMPLE_RATE / 1000)]
                  for s, e in batch]
        long_idx = [j for j, c in enumerate(chunks) if len(c) >= SPK_MIN_SAMPLES]
        batches.append((batch, chunks, long_idx))

    def finalize(batch, long_idx, results, spk_emb):
        sims = {}
        if spk_emb is not None:
            sims = dict(zip(long_idx, np.max(ref_matrix @ spk_emb.T, axis=0)))
        for j, ((start_ms, end_ms), res) in enumerate(zip(batch, results)):
            speaker = SPK_LABEL if sims.get(j, -1.0) > SPK_THRESHOLD else OTHER_LABEL
            utterances.append([int(start_ms), int(end_ms), speaker, res["text"]])

    # Pipeline ASR (main thread) and SPK (worker thread): the SPK pass of batch i
    # overlaps the ASR pass of batch i+1.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as spk_executor:
        pending = None
        for idx, (batch, chunks, long_idx) in enumerate(batches):
            results = asr_model.generate(
                input=chunks, language="zh", use_itn=True, batch_size_s=ASR_BATCH_SIZE_S
            )
            if pending is not None:
                p_batch, p_long_idx, p_results, p_future = pending
                spk_emb = p_future.result() if p_future is not None else None
                finalize(p_batch, p_long_idx, p_results, spk_emb)
            if long_idx:
                future = spk_executor.submit(_speaker_embeddings_matrix, chunks, long_idx, spk_model)
            else:
                future = None
            pending = (batch, long_idx, results, future)
            if on_progress:
                on_progress(min(100, int((idx + 1) * ASR_BATCH / total * 100)))

        if pending is not None:
            batch, long_idx, results, future = pending
            spk_emb = future.result() if future is not None else None
            finalize(batch, long_idx, results, spk_emb)

    return utterances


def process(req, model, df_state, vad_model, asr_model, spk_model, ref_matrix):
    rid = req.get("id", "")
    input_wav = req.get("input", "")
    if not input_wav:
        send({"cmd": "process", "id": rid, "ok": False, "error": "missing input"})
        return
    try:
        def progress(n):
            send({"progress": int(n)})

        # Denoise + downsample: report 0..20%.
        speech = filter_downsample(input_wav, model, df_state,
                                   on_progress=lambda p: progress(p * 0.20))
        # VAD on the in-memory 16 kHz audio.
        segments = vad_model.generate(input=speech, fs=SAMPLE_RATE)[0]["value"]
        # ASR + speaker labelling: report 20..100%.
        utterances = transcribe_samples(
            speech, segments, asr_model, spk_model, ref_matrix,
            on_progress=lambda p: progress(20 + p * 0.80),
        )
        send({"cmd": "process", "id": rid, "ok": True, "utterances": utterances})
    except Exception as e:
        log.exception("process failed")
        send({"cmd": "process", "id": rid, "ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True,
                    help="dir containing SenseVoiceSmall/fsmn-vad/cam++")
    ap.add_argument("--ref-dir", required=True,
                    help="dir with pre-resampled 16 kHz reference wav files")
    ap.add_argument("--filter-model-dir", required=True,
                    help="dir containing the DeepFilterNet3 model")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    log.info("Loading models...")
    try:
        with quiet_stdout():
            from df.enhance import enhance, init_df, load_audio
            model, df_state, _ = init_df(
                args.filter_model_dir,
                post_filter=False,
                log_level="ERROR",
                log_file=None,
                config_allow_defaults=True,
                epoch="best",
            )
            if torch.cuda.is_available():
                model = model.to("cuda")
            model.eval()
            globals()["load_audio"] = load_audio
            globals()["enhance"] = enhance

            vad_model = AutoModel(model=os.path.join(args.model_dir, "fsmn-vad"),
                                  device=device, disable_update=True, disable_pbar=True)
            asr_model = AutoModel(model=os.path.join(args.model_dir, "SenseVoiceSmall"),
                                  device=device, disable_update=True, disable_pbar=True)
            spk_model = AutoModel(model=os.path.join(args.model_dir, "cam++"),
                                  device=device, disable_update=True, disable_pbar=True)

            ref_paths = sorted(glob.glob(os.path.join(args.ref_dir, "*.wav")))
            if not ref_paths:
                log.error(f"No reference wav files found in {args.ref_dir}/")
                send({"cmd": "ready", "ok": False, "error": "no reference media"})
                return
            ref_matrix = np.stack([build_reference(p, vad_model, spk_model) for p in ref_paths])
    except Exception as e:
        log.exception("model load failed")
        send({"cmd": "ready", "ok": False, "error": str(e)})
        return
    log.info(f"Reference voiceprints: {len(ref_paths)} file(s)")

    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    send({"cmd": "ready", "ok": True, "engine": "audio"})

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
        elif cmd == "process":
            process(req, model, df_state, vad_model, asr_model, spk_model, ref_matrix)
        else:
            send({"cmd": cmd, "id": req.get("id", ""), "ok": False,
                  "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
