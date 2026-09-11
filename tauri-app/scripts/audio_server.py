"""Resident IPC audio worker for the LiveNeko Tauri app.

Launched once by the Rust backend at the start of a pipeline run. Loads the
configured ASR/SPK models a single time, then stays alive reading JSON
requests on stdin and writing JSON responses on stdout until told to shut
down. Models are NOT reloaded between requests.

VAD is NOT done here: the Rust backend runs the bundled native Silero VAD
(tauri-app/src-tauri/src/silero_vad.rs, CPU ONNX) on the denoised audio and
sends the utterance segments with every request. FunASR ASR/SPK models are
user-provided (local directories or downloaded from Hugging Face / ModelScope
by the app); none are bundled. The worker is configured with a JSON file
written by the Rust backend:

  {
    "asr": {"backend": "local", "type": "sensevoice-small",
            "dir": "<asr model dir>", "language": "zh"},
    "spk": {"dir": "<cam++ dir>"} | null,
    "ref": {"file": "<reference wav>", "name": "taffy",
            "segments": [[start_ms, end_ms], ...]} | null
  }

  For the online ASR backend the `asr` object is:
    {"backend": "qwen3-api", "baseUrl": "...", "apiKey": "...",
     "model": "qwen3-asr-flash", "language": "zh"}

ASR types:
  - sensevoice-small        SenseVoiceSmall (emotion/event tags preserved)
  - fun-asr-nano            Fun-ASR-Nano-2512
  - paraformer-zh-streaming Streaming Paraformer (used chunk-by-chunk)
  - qwen3-api               Qwen3-ASR over an OpenAI-compatible endpoint

The `process` command expects a 16 kHz mono WAV (decoded by ffmpeg, with the
optional noise-reduction filters configured in Settings) plus the Silero VAD
segments. This worker only performs ASR/SPK.

Protocol (newline-delimited JSON on stdin/stdout):
  Request:  {"cmd":"process","id":"<id>","input":"<filtered 16khz.wav>",
             "segments":[[start_ms,end_ms], ...]}
  Response: {"cmd":"process","id":"<id>","ok":true,
             "utterances":[[start_ms,end_ms,"<speaker>"|"other","<raw tagged text>"], ...]}
            {"cmd":"process","id":"<id>","ok":false,"error":"..."}
  Shutdown: {"cmd":"shutdown"}
Progress is emitted on stdout as {"progress":N} lines: 0..100 for the ASR phase.

The SPK model and reference voiceprints are both optional. Without a SPK model,
or without reference WAVs, no speaker is identified and every utterance is
labelled "other".
"""
import argparse
import concurrent.futures
import contextlib
import io
import json
import logging
import os
import sys

try:
    import numpy as np
    import soundfile as sf
    import torch
    from funasr import AutoModel
    from qwen3_asr_client import AsrApiError, request_asr, wav_data_url
    _IMPORT_ERROR = None
except Exception as _exc:  # noqa: BLE001 - reported over IPC in main()
    np = sf = torch = None
    AutoModel = None
    AsrApiError = request_asr = wav_data_url = None
    _IMPORT_ERROR = _exc

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

OTHER_LABEL = "other"
SPK_THRESHOLD = 0.50
SPK_CHUNK_S = 10.0
SPK_MIN_S = 2
SAMPLE_RATE = 16000
ASR_BATCH = 64
# Max seconds of audio packed into one ASR forward pass (VRAM < 8 GB).
ASR_BATCH_SIZE_S = 300
# Fun-ASR-Nano is an LLM decoder: keep the packed batch smaller.
ASR_BATCH_SIZE_S_NANO = 120
# Streaming Paraformer chunking (600 ms chunks, look-backs per the FunASR docs).
PARAFORMER_CHUNK_SIZE = [0, 10, 5]
PARAFORMER_STRIDE = PARAFORMER_CHUNK_SIZE[1] * 960
PARAFORMER_ENC_LOOK_BACK = 4
PARAFORMER_DEC_LOOK_BACK = 1
SPK_MIN_SAMPLES = int(SPK_MIN_S * SAMPLE_RATE)
SPK_CHUNK_SAMPLES = int(SPK_CHUNK_S * SAMPLE_RATE)

LOCAL_ASR_TYPES = ("sensevoice-small", "fun-asr-nano", "paraformer-zh-streaming")


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


@contextlib.contextmanager
def quiet_stdout():
    """Redirect stray prints (funasr banners) to stderr so stdout stays a
    clean JSON channel for IPC."""
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = real_stdout


# ---- ASR backends ----

class LocalAsr:
    """FunASR AutoModel ASR loaded from a local model directory."""

    def __init__(self, model_type, model_dir, device, language=""):
        if model_type not in LOCAL_ASR_TYPES:
            raise RuntimeError(f"unsupported local ASR type: {model_type}")
        if not model_dir or not os.path.isdir(model_dir):
            raise RuntimeError(f"ASR model directory not found: {model_dir}")
        self.name = model_type
        self.language = (language or "").strip()
        log.info(f"Loading ASR model '{model_type}' from {model_dir}")
        self.model = AutoModel(
            model=model_dir,
            device=device,
            disable_update=True,
            disable_pbar=True,
            trust_remote_code=True,
        )

    def _stream_paraformer(self, samples):
        cache = {}
        parts = []
        for start in range(0, len(samples), PARAFORMER_STRIDE):
            end = min(start + PARAFORMER_STRIDE, len(samples))
            result = self.model.generate(
                input=samples[start:end],
                cache=cache,
                is_final=end == len(samples),
                chunk_size=PARAFORMER_CHUNK_SIZE,
                encoder_chunk_look_back=PARAFORMER_ENC_LOOK_BACK,
                decoder_chunk_look_back=PARAFORMER_DEC_LOOK_BACK,
                batch_size=1,
            )
            if result and result[0].get("text"):
                parts.append(result[0]["text"])
        return "".join(parts)

    def transcribe(self, chunks):
        """Transcribe a list of float32 mono 16 kHz numpy chunks."""
        if not chunks:
            return []
        if self.name == "paraformer-zh-streaming":
            return [self._stream_paraformer(c) for c in chunks]

        kwargs = {}
        if self.name == "sensevoice-small":
            # Preserve the historical behavior: Chinese + inverse text normalization.
            kwargs["language"] = self.language or "zh"
            kwargs["use_itn"] = True
            kwargs["batch_size_s"] = ASR_BATCH_SIZE_S
        else:  # fun-asr-nano
            if self.language:
                kwargs["language"] = self.language
            kwargs["itn"] = True
            kwargs["batch_size_s"] = ASR_BATCH_SIZE_S_NANO
        results = self.model.generate(input=chunks, **kwargs)
        return [r.get("text", "") for r in results]


class Qwen3ApiAsr:
    """Qwen3-ASR over an OpenAI-compatible or native DashScope endpoint.

    The endpoint/shape selection lives in `qwen3_asr_client.py`; this class
    only turns local samples into a WAV data URL and forwards the result.
    """

    def __init__(self, base_url, api_key, model, language="", timeout=180, retries=2):
        self.name = "qwen3-api"
        self.base_url = (base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.model = (model or "").strip()
        self.language = (language or "").strip()
        self.timeout = timeout
        self.retries = retries
        if not self.base_url:
            raise RuntimeError("Qwen3-ASR API base URL is not configured")
        if not self.model:
            raise RuntimeError("Qwen3-ASR API model name is not configured")

    def _request(self, samples):
        buf = io.BytesIO()
        sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        text, _method = request_asr(
            self.base_url,
            self.api_key,
            self.model,
            wav_data_url(buf.getvalue()),
            language=self.language,
            timeout=self.timeout,
            retries=self.retries,
        )
        return text

    def transcribe(self, chunks):
        texts = []
        failures = 0
        first_error = None
        for chunk in chunks:
            try:
                texts.append(self._request(chunk))
            except AsrApiError as exc:
                if exc.fatal:
                    raise RuntimeError(str(exc)) from exc
                # Models such as qwen-audio-3.0-asr-flash reject very short or
                # silent clips with `HTTP 400 {}`. Skip those chunks, report
                # them in the log, and only abort when *every* chunk failed.
                failures += 1
                first_error = first_error or exc
                log.warning(
                    "Qwen3-ASR rejected a %.2fs audio chunk: %s",
                    len(chunk) / SAMPLE_RATE,
                    exc,
                )
                texts.append("")
        if chunks and failures == len(chunks) and first_error is not None:
            raise RuntimeError(str(first_error))
        return texts


def build_asr(config, device):
    asr_cfg = config.get("asr") or {}
    backend = asr_cfg.get("backend", "local")
    if backend == "qwen3-api":
        return Qwen3ApiAsr(
            asr_cfg.get("baseUrl", ""),
            asr_cfg.get("apiKey", ""),
            asr_cfg.get("model", ""),
            asr_cfg.get("language", ""),
        )
    return LocalAsr(
        asr_cfg.get("type", ""),
        asr_cfg.get("dir", ""),
        device,
        asr_cfg.get("language", ""),
    )


# ---- SPK ----

def load_spk(spk_dir, device):
    if not spk_dir or not os.path.isdir(spk_dir):
        raise RuntimeError(f"SPK model directory not found: {spk_dir}")
    log.info(f"Loading SPK model from {spk_dir}")
    return AutoModel(
        model=spk_dir,
        device=device,
        disable_update=True,
        disable_pbar=True,
        trust_remote_code=True,
    )


def speaker_embeddings(chunks, spk):
    results = spk.inference(input=chunks, model=spk.model, kwargs=spk.kwargs)
    embeddings = []
    for res in results:
        e = res["spk_embedding"]
        if torch.is_tensor(e):
            e = e.detach().cpu().numpy()
        e = np.asarray(e, dtype=np.float32).ravel()
        embeddings.append(e / np.linalg.norm(e))
    return embeddings


def build_reference(wav_path, segments, spk):
    """Build one reference voiceprint from the wav's pre-computed VAD segments
    (delivered by the Rust backend, which runs the native Silero VAD)."""
    speech, sr = sf.read(wav_path, dtype="float32")
    if sr != SAMPLE_RATE:
        raise RuntimeError(f"Unexpected sample rate {sr} for {wav_path}; expected {SAMPLE_RATE}")
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
    ref = np.mean(speaker_embeddings(chunks, spk), axis=0)
    ref /= np.linalg.norm(ref)
    return ref


def _speaker_embeddings_matrix(chunks, long_idx, spk):
    return np.stack(speaker_embeddings([chunks[j] for j in long_idx], spk))


# ---- transcription pipeline ----

def transcribe_samples(speech, segments, asr, spk, ref_matrix, speaker_name, on_progress=None):
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
        if ref_matrix is not None and spk_emb is not None:
            sims = dict(zip(long_idx, np.max(ref_matrix @ spk_emb.T, axis=0)))
        for j, ((start_ms, end_ms), res) in enumerate(zip(batch, results)):
            speaker = speaker_name if sims.get(j, -1.0) > SPK_THRESHOLD else OTHER_LABEL
            # `res` is the worker result dict ({"text": ...}); the Rust IPC
            # contract expects a plain string as the 4th utterance element.
            text = res.get("text", "") if isinstance(res, dict) else str(res)
            utterances.append([int(start_ms), int(end_ms), speaker, text])

    # Pipeline ASR (main thread) and SPK (worker thread): the SPK pass of batch i
    # overlaps the ASR pass of batch i+1. Without a SPK model or reference
    # voiceprint the SPK pass is skipped entirely.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as spk_executor:
        pending = None
        for idx, (batch, chunks, long_idx) in enumerate(batches):
            texts = asr.transcribe(chunks)
            results = [{"text": text} for text in texts]
            if pending is not None:
                p_batch, p_long_idx, p_results, p_future = pending
                spk_emb = p_future.result() if p_future is not None else None
                finalize(p_batch, p_long_idx, p_results, spk_emb)
            if long_idx and ref_matrix is not None and spk is not None:
                future = spk_executor.submit(_speaker_embeddings_matrix, chunks, long_idx, spk)
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


def parse_segments(raw):
    """Validate the Rust-provided VAD segments: a list of [start_ms, end_ms]
    integer pairs with 0 <= start < end."""
    if not isinstance(raw, list):
        raise RuntimeError("request is missing the VAD segments list")
    segments = []
    for item in raw:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 2
            or not all(isinstance(v, int) for v in item)
        ):
            raise RuntimeError(f"malformed VAD segment: {item!r}")
        start_ms, end_ms = item
        if start_ms < 0 or end_ms <= start_ms:
            raise RuntimeError(f"malformed VAD segment: {item!r}")
        segments.append((start_ms, end_ms))
    return segments


def process(req, asr, spk, ref_matrix, speaker_name):
    rid = req.get("id", "")
    input_wav = req.get("input", "")
    if not input_wav:
        send({"cmd": "process", "id": rid, "ok": False, "error": "missing input"})
        return
    try:
        def progress(n):
            send({"progress": int(n)})

        speech, sr = sf.read(input_wav, dtype="float32")
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"Unexpected sample rate {sr} for {input_wav}; expected {SAMPLE_RATE}")
        segments = parse_segments(req.get("segments"))
        # ASR + speaker labelling: report 20..100 so the Rust backend keeps the
        # 0..20 range for DeepFilterNet denoising (VAD runs in Rust between the
        # two and needs no progress range of its own).
        utterances = transcribe_samples(
            speech, segments, asr, spk, ref_matrix, speaker_name,
            on_progress=lambda p: progress(20 + p * 0.8),
        )
        send({"cmd": "process", "id": rid, "ok": True, "utterances": utterances})
    except Exception as e:
        log.exception("process failed")
        send({"cmd": "process", "id": rid, "ok": False, "error": str(e)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True,
                    help="path to the JSON audio-model configuration written by the app")
    args = ap.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        send({"cmd": "ready", "ok": False, "error": f"cannot read model config: {e}"})
        return

    if AutoModel is None or torch is None or sf is None or np is None:
        send({
            "cmd": "ready",
            "ok": False,
            "error": f"required Python libraries are not importable: {_IMPORT_ERROR}",
        })
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    spk = None
    ref_matrix = None
    speaker_name = ""
    try:
        with quiet_stdout():
            asr = build_asr(config, device)
            spk_cfg = config.get("spk") or {}
            if spk_cfg.get("dir"):
                spk = load_spk(spk_cfg["dir"], device)
            ref_cfg = config.get("ref") or {}
            ref_file = ref_cfg.get("file", "")
            if spk is not None and ref_file:
                speaker_name = ref_cfg.get("name", "speaker")
                segments = parse_segments(ref_cfg.get("segments"))
                ref_matrix = build_reference(ref_file, segments, spk)[None, :]
    except Exception as e:
        log.exception("model load failed")
        send({"cmd": "ready", "ok": False, "error": str(e)})
        return

    if spk is None:
        log.info("No SPK model configured: speaker identification is disabled")
    elif ref_matrix is not None:
        log.info(f"Reference voiceprint for speaker '{speaker_name}'")
    else:
        log.info("No speaker reference: utterances will not be tagged with a specific speaker")

    try:
        sys.stdin.reconfigure(encoding="utf-8")
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    send({"cmd": "ready", "ok": True, "engine": "audio", "asr": asr.name,
          "spk": spk is not None})

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
            process(req, asr, spk, ref_matrix, speaker_name)
        else:
            send({"cmd": cmd, "id": req.get("id", ""), "ok": False,
                  "error": f"unknown cmd {cmd}"})


if __name__ == "__main__":
    main()
