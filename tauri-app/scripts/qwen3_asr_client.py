"""Stdlib-only client for Qwen3-ASR online endpoints.

Qwen3-ASR is reachable through two different DashScope/Model-Studio APIs and
the exact request shape depends on the endpoint and the model:

1. OpenAI-compatible mode:
     POST {base}/chat/completions
     {"model": ..., "messages": [{"role":"user","content":[
        {"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,..."}}]}],
      "stream": false, "asr_options": {"language": "zh"}}
   Works for `qwen3-asr-flash` and most third-party OpenAI-compatible servers.

2. Native DashScope multimodal generation:
     POST {origin}/api/v1/services/aigc/multimodal-generation/generation
   with header `X-DashScope-SSE: disable`. Depending on the model the audio
   content is either the OpenAI-style `input_audio` object
   (`qwen-audio-3.0-asr-flash`, `fun-asr-flash-*`) or an `audio` string
   (`qwen3-asr-flash`):
     {"type":"input_audio","input_audio":{"data":"data:audio/wav;base64,..."}}
     {"type":"audio","audio":"data:audio/wav;base64,..."}
   and `parameters: {"format":"wav","sample_rate":"16000"}`.

`build_requests()` produces the ordered candidate requests for a configured
base URL (native first on DashScope/Model-Studio hosts, OpenAI-compatible first
elsewhere). `request_asr()` tries each candidate and raises `AsrApiError` with a
per-candidate status/body summary when all of them fail, so the UI reports the
actual server error instead of a bare "HTTP 400: {}".
"""

import base64
import json
import time
import urllib.error
import urllib.request

NATIVE_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
COMPATIBLE_SUFFIX = "/compatible-mode/v1"
TIMEOUT = 180
RETRIES = 2
SAMPLE_RATE = 16000


class AsrApiError(RuntimeError):
    """Raised when every candidate Qwen3-ASR request failed.

    `fatal` marks errors that cannot be caused by a single rejected audio clip
    (authentication failures, unknown model/endpoint, malformed parameters), so
    the pipeline aborts instead of skipping every chunk.
    """

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def wav_data_url(wav_bytes):
    """Wrap raw WAV file bytes into an OpenAI-style base64 data URL."""
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")


def derive_native_url(base_url):
    """Return the native DashScope generation URL for `base_url`, or None.

    - `https://host/compatible-mode/v1` -> `https://host/api/v1/services/...`
    - a URL that already points at `/api/v1/services/...` is returned as-is
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return None
    if "/api/v1/services/" in base:
        return base
    if base.endswith(COMPATIBLE_SUFFIX):
        return base[: -len(COMPATIBLE_SUFFIX)] + NATIVE_PATH
    return None


def _is_dashscope_like(base_url):
    base = (base_url or "").lower()
    return "aliyuncs.com" in base or "dashscope" in base or "/api/v1/services/" in base


def _prefer_input_audio(model):
    name = (model or "").lower()
    return any(k in name for k in ("audio-", "qwen-audio", "fun-asr"))


def _content_input_audio(data_url):
    return [{"type": "input_audio", "input_audio": {"data": data_url}}]


def _content_audio(data_url):
    return [{"type": "audio", "audio": data_url}]


def _chat_payload(model, data_url, language):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": _content_input_audio(data_url)}],
        "stream": False,
    }
    if language:
        payload["asr_options"] = {"language": language}
    return payload


def _native_payload(model, data_url, language):
    parameters = {"format": "wav", "sample_rate": str(SAMPLE_RATE)}
    if language:
        parameters["asr_options"] = {"language": language}
    return {"model": model, "parameters": parameters}


def build_requests(base_url, model, data_url, language=""):
    """Return the ordered `(name, url, payload, extra_headers)` candidates."""
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise AsrApiError("Qwen3-ASR API base URL is empty")
    if not (model or "").strip():
        raise AsrApiError("Qwen3-ASR API model name is empty")

    native = derive_native_url(base)
    native_base = "/api/v1/services/" in base
    dashscope = _is_dashscope_like(base)

    native_candidates = []
    if native:
        shapes = (
            [_content_input_audio(data_url), _content_audio(data_url)]
            if _prefer_input_audio(model)
            else [_content_audio(data_url), _content_input_audio(data_url)]
        )
        for index, content in enumerate(shapes):
            payload = _native_payload(model, data_url, language)
            payload["input"] = {"messages": [{"role": "user", "content": content}]}
            native_candidates.append(
                (
                    f"native[{index}]",
                    native,
                    payload,
                    {"X-DashScope-SSE": "disable"},
                )
            )

    # When the configured URL already *is* the native endpoint there is no
    # meaningful `/chat/completions` sibling, so only native shapes are tried.
    if native_base:
        return native_candidates

    chat_candidate = (
        "chat.completions",
        base + "/chat/completions",
        _chat_payload(model, data_url, language),
        {},
    )

    if dashscope:
        return native_candidates + [chat_candidate]
    return [chat_candidate] + native_candidates


def _short_error(detail):
    """Compact, informative rendering of an HTTP error body."""
    detail = (detail or "").strip()
    if not detail:
        return "<empty response body>"
    try:
        obj = json.loads(detail)
    except Exception:  # noqa: BLE001 - non-JSON error body
        return detail[:300]
    source = obj.get("error") if isinstance(obj.get("error"), dict) else obj
    message = source.get("message") or obj.get("message")
    code = source.get("code") or obj.get("code")
    request_id = obj.get("request_id") or obj.get("id")
    parts = []
    if code:
        parts.append(str(code))
    if message:
        parts.append(str(message))
    if request_id:
        parts.append(f"request_id={request_id}")
    return "; ".join(parts)[:400] if parts else detail[:300]


def extract_text(payload):
    """Extract the recognised text from any of the response shapes."""
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message") or {}
        return _flatten_content(message.get("content"))
    output = payload.get("output") or {}
    if isinstance(output, dict):
        out_choices = output.get("choices") or []
        if out_choices:
            message = out_choices[0].get("message") or {}
            return _flatten_content(message.get("content"))
        sentence = output.get("sentence")
        if isinstance(sentence, dict) and sentence.get("text"):
            return sentence["text"]
        if output.get("text"):
            return output["text"]
    # Model-Studio `qwen-audio-3.0-asr-flash` shape
    sentence = payload.get("sentence")
    if isinstance(sentence, dict) and sentence.get("text"):
        return sentence["text"]
    return payload.get("text") or ""


def _flatten_content(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
            else:
                parts.append(str(part))
        return "".join(parts)
    if isinstance(content, dict):
        return content.get("text") or ""
    return str(content)


def _post(url, payload, api_key, extra_headers, timeout):
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def request_asr(base_url, api_key, model, data_url, language="", timeout=TIMEOUT, retries=RETRIES):
    """Transcribe one WAV data URL. Returns `(text, method)`.

    Raises `AsrApiError` listing every candidate's error when all fail.
    """
    candidates = build_requests(base_url, model, data_url, language)
    errors = []
    fatal = False
    for name, url, payload, extra_headers in candidates:
        for attempt in range(retries + 1):
            try:
                text = _post(url, payload, api_key, extra_headers, timeout)
                return extract_text(json.loads(text)), name
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
                compact = _short_error(detail)
                errors.append(f"{name} HTTP {exc.code}: {compact}")
                if exc.code in (401, 403):
                    raise AsrApiError(
                        "Qwen3-ASR authentication failed — " + errors[-1], fatal=True
                    ) from exc
                if exc.code >= 500 and attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                # 404/422 and structured 4xx errors are configuration problems;
                # an empty-body 400 usually just rejects this audio clip.
                if exc.code in (404, 422) or (
                    exc.code == 400 and detail.strip() not in ("", "{}")
                ):
                    fatal = True
                break  # a 4xx for this shape will not change on retry; try the next
            except Exception as exc:  # noqa: BLE001 - network/timeouts are retryable
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise AsrApiError("Qwen3-ASR request failed — " + " | ".join(errors), fatal=fatal)


def check_connection(base_url, api_key, model, data_url, language="", timeout=60):
    """Probe the configured endpoint with a short audio clip.

    Returns `{"ok": True, "method": ..., "text": ...}` or
    `{"ok": False, "fatal": bool, "errors": [...]}`.
    """
    try:
        text, method = request_asr(
            base_url, api_key, model, data_url, language=language, timeout=timeout, retries=1
        )
        return {"ok": True, "method": method, "text": (text or "")[:200]}
    except AsrApiError as exc:
        return {"ok": False, "fatal": exc.fatal, "errors": [str(exc)]}
