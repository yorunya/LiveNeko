//! In-process, unauthenticated video downloader for Bilibili and YouTube.
//!
//! Mirrors the relevant parts of the yt-dlp reference implementation:
//! - Bilibili: `x/player/wbi/playurl` with WBI signing + danmaku fingerprint
//!   params (`BilibiliBaseIE._download_playinfo`), DASH best-video+best-audio
//!   pick (including dolby/FLAC audio tracks), and legacy single-segment
//!   `durl` fallback.
//! - YouTube: Innertube `youtubei/v1/player` with the default client list
//!   `('visionos', 'web')` (`YoutubeIE._DEFAULT_CLIENTS`), client headers +
//!   visitor data, `bv*+ba/b` format-selection order (adaptive merge first,
//!   progressive only as a fallback), DRM/OTF formats skipped. Like yt-dlp,
//!   signatureCipher solving and PO-token minting are NOT implemented.
//! - Transport: ranged, resumable, retried chunked downloads (yt-dlp's
//!   http downloader behavior) with ffmpeg `-c copy -movflags +faststart`
//!   merging (same flags as yt-dlp's FFmpegMergerPP).
//! - Optional browser cookies (yt-dlp `--cookies-from-browser` equivalent,
//!   see `cookies.rs`): sent on all requests; a SESSDATA cookie marks the
//!   Bilibili session as logged in (drops `try_look`), YouTube "web" client
//!   calls carry cookies + SAPISIDHASH authorization.

use crate::cookies::{self, Cookie};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// HTTP plumbing (no external HTTP client crate)
// ---------------------------------------------------------------------------

/// A minimal blocking HTTP client backed by `curl` (bundled with Windows 10+).
/// This avoids pulling a full HTTP/TLS stack into the binary while still
struct HttpClient {
    /// Additional headers that every request should carry.
    default_headers: Vec<(String, String)>,
}

/// Shared progress/log callbacks handed to the downloader (`Arc` so the caller
/// can keep a copy while the downloader holds one).
type ProgressCb = Arc<Mutex<Box<dyn FnMut(u8) + Send>>>;
type LogCb = Arc<Mutex<Box<dyn FnMut(String) + Send>>>;

impl HttpClient {
    fn new() -> Self {
        Self {
            default_headers: vec![
                (
                    "User-Agent".into(),
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36".into(),
                ),
                ("Accept-Language".into(), "en-US,en;q=0.9".into()),
            ],
        }
    }

    fn build_cmd(&self, url: &str, extra_headers: &[(&str, &str)]) -> std::process::Command {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-sSL")
            .arg("--compressed")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--max-time")
            .arg("600");
        for (k, v) in &self.default_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        for (k, v) in extra_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        cmd.arg(url);
        crate::pipeline::hide_console(&mut cmd);
        cmd
    }

    fn get_text(&self, url: &str, extra_headers: &[(&str, &str)]) -> Result<String, String> {
        let out = self
            .build_cmd(url, extra_headers)
            .output()
            .map_err(|e| format!("curl: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "curl failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        String::from_utf8(out.stdout).map_err(|e| format!("invalid utf-8 response: {e}"))
    }

    fn get_json(&self, url: &str, extra_headers: &[(&str, &str)]) -> Result<Value, String> {
        let text = self.get_text(url, extra_headers)?;
        serde_json::from_str(&text).map_err(|e| format!("invalid json: {e}"))
    }

    /// POST JSON and return parsed JSON.
    fn post_json(
        &self,
        url: &str,
        body: &Value,
        extra_headers: &[(&str, &str)],
    ) -> Result<Value, String> {
        let mut headers = extra_headers.to_vec();
        headers.push(("Content-Type", "application/json"));
        let mut cmd = self.build_cmd(url, &headers);
        cmd.arg("-X").arg("POST").arg("-d").arg(body.to_string());
        let out = cmd.output().map_err(|e| format!("curl: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "curl failed ({}): {}",
                out.status,
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        serde_json::from_slice(&out.stdout).map_err(|e| format!("invalid json: {e}"))
    }

    /// Download a URL to a local file with ranged, resumable, retried chunks,
    /// reporting progress (0..100) via `on_progress`. Mirrors yt-dlp's http
    /// downloader: 10 MB range chunks, resume from a `.part` file, per-chunk
    /// retries, stall detection. The `.part` file is keyed by the URL (so a
    /// re-signed URL for a different quality starts fresh) and kept on
    /// failure/cancel for a later resume.
    fn download_file(
        &self,
        url: &str,
        dest: &Path,
        extra_headers: &[(&str, &str)],
        cancel: &Arc<AtomicBool>,
        on_progress: ProgressCb,
    ) -> Result<(), String> {
        let part = part_path(url);
        let mut start = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
        let report = |on_progress: &ProgressCb, p: u8| {
            let mut cb = on_progress.lock().unwrap();
            (cb)(p);
        };

        // Probe the total size with a 1-byte ranged request so we can chunk
        // the transfer and report exact progress.
        let total = match self.probe_total(url, start, extra_headers) {
            Ok(Some(t)) => t,
            Ok(None) => {
                // Server gave no usable length: fall back to one plain
                // streaming GET continued onto the part file.
                return self.stream_plain(url, &part, dest, extra_headers, cancel, &report, &on_progress);
            }
            Err(e) => return Err(e),
        };
        if start > total {
            // Part is longer than the remote file (content changed): restart.
            let _ = std::fs::remove_file(&part);
            start = 0;
        }
        if start >= total {
            // Part already complete from a previous run.
            let _ = std::fs::remove_file(dest);
            finalize_part(&part, dest)?;
            report(&on_progress, 100);
            return Ok(());
        }

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part)
            .map_err(|e| format!("open part file: {e}"))?;
        use std::io::Write;
        while start < total {
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled".into());
            }
            let end = (start + DL_CHUNK_SIZE).min(total) - 1;
            let expected = (end - start + 1) as usize;
            let mut chunk_done = false;
            for attempt in 0..DL_MAX_ATTEMPTS {
                if cancel.load(Ordering::SeqCst) {
                    return Err("cancelled".into());
                }
                match self.fetch_range(url, start, end, extra_headers) {
                    Ok(bytes) if bytes.len() == expected => {
                        file.write_all(&bytes)
                            .map_err(|e| format!("write part: {e}"))?;
                        start += expected as u64;
                        chunk_done = true;
                        break;
                    }
                    Ok(_) | Err(_) => {
                        // Retry with a small backoff, like yt-dlp's RetryManager.
                        std::thread::sleep(std::time::Duration::from_millis(
                            500 * (attempt as u64 + 1),
                        ));
                    }
                }
            }
            if !chunk_done {
                return Err(format!(
                    "download failed after {DL_MAX_ATTEMPTS} retries at byte {start}/{total} (partial data kept for resume)"
                ));
            }
            let p = ((start as f64 / total as f64) * 100.0).min(100.0) as u8;
            report(&on_progress, p);
        }
        drop(file);
        let _ = std::fs::remove_file(dest);
        finalize_part(&part, dest)?;
        report(&on_progress, 100);
        Ok(())
    }

    /// One ranged GET; returns the body bytes. Aborts when the transfer
    /// stalls (<1 KiB/s for 60 s) instead of a fixed total timeout.
    fn fetch_range(
        &self,
        url: &str,
        start: u64,
        end: u64,
        extra_headers: &[(&str, &str)],
    ) -> Result<Vec<u8>, String> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-sS")
            .arg("-L")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--speed-time")
            .arg("60")
            .arg("--speed-limit")
            .arg("1024")
            .arg("--range")
            .arg(format!("{start}-{end}"));
        for (k, v) in &self.default_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        for (k, v) in extra_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        cmd.arg(url);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd
            .output()
            .map_err(|e| format!("curl: {e}"))?;
        if !out.status.success() {
            return Err(format!("curl range fetch failed: {}", out.status));
        }
        Ok(out.stdout)
    }

    /// Probe the file's total size via a 1-byte ranged GET.
    /// `Ok(Some(total))` when known, `Ok(None)` when the server does not
    /// support ranges or report a length.
    fn probe_total(
        &self,
        url: &str,
        start: u64,
        extra_headers: &[(&str, &str)],
    ) -> Result<Option<u64>, String> {
        let body_tmp = std::env::temp_dir().join(format!(
            "liveneko_probe_{}",
            std::process::id()
        ));
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-sS")
            .arg("-L")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--max-time")
            .arg("60")
            .arg("-D")
            .arg("-")
            .arg("-o")
            .arg(&body_tmp)
            .arg("--range")
            .arg(format!("{start}-{}", start + 1023));
        for (k, v) in &self.default_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        for (k, v) in extra_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        cmd.arg(url);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("curl: {e}"));
        let _ = std::fs::remove_file(&body_tmp);
        let out = out?;
        if !out.status.success() {
            return Err(format!("curl probe failed: {}", out.status));
        }
        let headers = String::from_utf8_lossy(&out.stdout);
        let mut status = 0u16;
        let mut content_length = 0u64;
        let mut total_from_range: Option<u64> = None;
        for line in headers.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("HTTP/") {
                if let Some(code_str) = rest.split_whitespace().nth(1) {
                    status = code_str.parse().unwrap_or(0);
                }
            } else if let Some(v) = strip_header(line, "content-length") {
                content_length = v.trim().parse().unwrap_or(0);
            } else if let Some(v) = strip_header(line, "content-range") {
                // "bytes start-end/total" (total may be "*")
                if let Some(total) = v.rsplit('/').next() {
                    total_from_range = total.trim().parse().ok();
                }
            }
        }
        match status {
            206 => Ok(total_from_range),
            // 200 = server ignored the Range header: restart from scratch
            200 => Ok(if content_length > 0 {
                Some(content_length)
            } else {
                None
            }),
            // 416 = start beyond EOF; recover the size from "bytes */N"
            416 => Ok(total_from_range),
            _ => Ok(None),
        }
    }

    /// Plain streaming download for servers without range/length support.
    #[allow(clippy::too_many_arguments)]
    fn stream_plain(
        &self,
        url: &str,
        part: &Path,
        dest: &Path,
        extra_headers: &[(&str, &str)],
        cancel: &Arc<AtomicBool>,
        report: &dyn Fn(&ProgressCb, u8),
        on_progress: &ProgressCb,
    ) -> Result<(), String> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-sS")
            .arg("-L")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--speed-time")
            .arg("60")
            .arg("--speed-limit")
            .arg("1024")
            .arg("-C")
            .arg("-")
            .arg("-o")
            .arg(part);
        for (k, v) in &self.default_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        for (k, v) in extra_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        cmd.arg(url);
        crate::pipeline::hide_console(&mut cmd);
        let mut child = cmd
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .map_err(|e| format!("curl: {e}"))?;
        loop {
            if cancel.load(Ordering::SeqCst) {
                let _ = child.kill();
                let _ = child.wait();
                break Err("cancelled".into());
            }
            match child.try_wait().map_err(|e| format!("curl wait: {e}"))? {
                Some(status) => {
                    if status.success() {
                        break Ok(());
                    } else {
                        break Err(format!("curl download failed: {status}"));
                    }
                }
                None => std::thread::sleep(std::time::Duration::from_millis(200)),
            }
        }?;
        let _ = std::fs::remove_file(dest);
        finalize_part(part, dest)?;
        report(on_progress, 100);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MD5 (pure Rust, no external crate — needed for Bilibili WBI signing)
// ---------------------------------------------------------------------------

mod md5 {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10,
        15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    pub fn hex_digest(data: &[u8]) -> String {
        let mut a0: u32 = 0x67452301;
        let mut b0: u32 = 0xefcdab89;
        let mut c0: u32 = 0x98badcfe;
        let mut d0: u32 = 0x10325476;

        let mut msg = data.to_vec();
        let bit_len = (data.len() as u64).wrapping_mul(8);
        msg.push(0x80);
        while msg.len() % 64 != 56 {
            msg.push(0);
        }
        msg.extend_from_slice(&bit_len.to_le_bytes());

        for chunk in msg.as_chunks::<64>().0 {
            let mut m = [0u32; 16];
            for (i, w) in m.iter_mut().enumerate() {
                *w = u32::from_le_bytes([
                    chunk[i * 4],
                    chunk[i * 4 + 1],
                    chunk[i * 4 + 2],
                    chunk[i * 4 + 3],
                ]);
            }
            let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
            for i in 0..64 {
                let (mut f, g) = match i / 16 {
                    0 => ((b & c) | (!b & d), i),
                    1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                    2 => (b ^ c ^ d, (3 * i + 5) % 16),
                    _ => (c ^ (b | !d), (7 * i) % 16),
                };
                f = f.wrapping_add(a).wrapping_add(K[i]).wrapping_add(m[g]);
                a = d;
                d = c;
                c = b;
                b = b.wrapping_add(f.rotate_left(S[i]));
            }
            a0 = a0.wrapping_add(a);
            b0 = b0.wrapping_add(b);
            c0 = c0.wrapping_add(c);
            d0 = d0.wrapping_add(d);
        }
        let digest = [a0, b0, c0, d0]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<u8>>();
        digest.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

// ---------------------------------------------------------------------------
// URL helpers
// ---------------------------------------------------------------------------

fn urlencode(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

/// Extract a query parameter value from a URL string.
fn query_param(url: &str, key: &str) -> Option<String> {
    let q = url.split_once('?')?.1;
    for pair in q.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next() == Some(key) {
            return it.next().map(urlencoding_decode);
        }
    }
    None
}

fn urlencoding_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.bytes();
    while let Some(b) = it.next() {
        if b == b'%' {
            let h = it.next().unwrap_or(b'0');
            let l = it.next().unwrap_or(b'0');
            let hv = (h as char).to_digit(16).unwrap_or(0);
            let lv = (l as char).to_digit(16).unwrap_or(0);
            out.push((hv * 16 + lv) as u8 as char);
        } else if b == b'+' {
            out.push(' ');
        } else {
            out.push(b as char);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Shared downloader context
// ---------------------------------------------------------------------------

pub struct Downloader {
    http: HttpClient,
    /// ffmpeg path for merging video+audio (usually just "ffmpeg" on PATH).
    ffmpeg: String,
    cancel: Arc<AtomicBool>,
    /// Progress callback: (percent 0..100).
    on_progress: ProgressCb,
    /// Log callback.
    on_log: LogCb,
    /// Optional browser cookies (yt-dlp --cookies-from-browser equivalent).
    cookies: Option<Arc<Vec<Cookie>>>,
}

impl Downloader {
    pub fn new(
        cancel: Arc<AtomicBool>,
        on_progress: ProgressCb,
        on_log: LogCb,
        cookies: Option<Arc<Vec<Cookie>>>,
    ) -> Self {
        Self {
            http: HttpClient::new(),
            ffmpeg: "ffmpeg".into(),
            cancel,
            on_progress,
            on_log,
            cookies,
        }
    }

    fn log(&self, msg: impl Into<String>) {
        let mut f = self.on_log.lock().unwrap();
        (f)(msg.into());
    }

    fn progress(&self, p: u8) {
        let mut f = self.on_progress.lock().unwrap();
        (f)(p);
    }

    /// `("Cookie", "name=value; ...")` for the request host, when any browser
    /// cookie matches (yt-dlp sends cookies by the same domain matching).
    fn cookie_pair(&self, url: &str) -> Option<(&'static str, String)> {
        let cs = self.cookies.as_ref()?;
        cookies::cookie_header(cs, host_of(url)).map(|v| ("Cookie", v))
    }

    /// Whether a logged-in Bilibili session was imported (SESSDATA cookie),
    /// the same check as `BilibiliBaseIE.is_logged_in`.
    fn bilibili_logged_in(&self) -> bool {
        self.cookies
            .as_ref()
            .map(|cs| cookies::find_cookie(cs, "www.bilibili.com", "SESSDATA").is_some())
            .unwrap_or(false)
    }

    /// yt-dlp `_get_sid_authorization_header`: SAPISIDHASH / SAPISID1PHASH /
    /// SAPISID3PHASH tokens over the imported YouTube cookies.
    fn sid_authorization(&self) -> Option<String> {
        let cs = self.cookies.as_ref()?;
        let get = |name: &str| cookies::find_cookie(cs, "www.youtube.com", name).map(str::to_string);
        let origin = "https://www.youtube.com";
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
            .to_string();
        let make = |scheme: &str, sid: String| {
            format!("{scheme} {ts}_{}", sha1_hex(&format!("{ts} {sid} {origin}")))
        };
        let mut out: Vec<String> = Vec::new();
        if let Some(sid) = get("SAPISID").or_else(|| get("__Secure-3PAPISID")) {
            out.push(make("SAPISIDHASH", sid));
        }
        if let Some(sid) = get("__Secure-1PAPISID") {
            out.push(make("SAPISID1PHASH", sid));
        }
        if let Some(sid) = get("__Secure-3PAPISID") {
            out.push(make("SAPISID3PHASH", sid));
        }
        if out.is_empty() {
            None
        } else {
            Some(out.join(" "))
        }
    }

    // -----------------------------------------------------------------------
    // Public entry points (used by pipeline.rs)
    // -----------------------------------------------------------------------

    /// Probe the title(s) a URL yields, one per line.
    /// For Bilibili this uses `__INITIAL_STATE__`; for YouTube the
    /// `ytInitialPlayerResponse` / `youtubei/v1/player` fallback.
    pub fn probe_titles(&self, url: &str) -> Result<Vec<String>, String> {
        if url.contains("bilibili.com") {
            self.bilibili_probe_titles(url)
        } else if url.contains("youtube.com") || url.contains("youtu.be") {
            self.youtube_probe_titles(url)
        } else {
            Err("unsupported URL (only Bilibili and YouTube are supported)".into())
        }
    }

    /// Structured variant of `probe_titles` for the multi-part picker: returns
    /// the video's main title and its parts as `(1-based page, part title)`
    /// pairs in playlist order. YouTube URLs always yield a single part; a
    /// Bilibili `?p=N` URL yields only that page (the picker stays closed and
    /// the legacy single-page download path runs).
    pub fn probe_parts(&self, url: &str) -> Result<(String, Vec<(u32, String)>), String> {
        if url.contains("bilibili.com") {
            let meta = self.bilibili_meta(url)?;
            let part_id = query_param(url, "p").and_then(|v| v.parse::<u32>().ok());
            let parts = meta
                .pages
                .iter()
                .filter(|(p, _, _)| part_id.is_none_or(|pid| *p == pid))
                .map(|(page, _, part)| (*page, part.clone()))
                .collect();
            Ok((meta.title, parts))
        } else if url.contains("youtube.com") || url.contains("youtu.be") {
            let titles = self.youtube_probe_titles(url)?;
            let title = titles.into_iter().next().unwrap_or_else(|| "video".into());
            Ok((title.clone(), vec![(1, title)]))
        } else {
            Err("unsupported URL (only Bilibili and YouTube are supported)".into())
        }
    }

    /// Download a single video (or all parts of an anthology) into `out_dir`.
    /// `quality` is the maximum height (360/480/720/1080).
    /// `parts` selects specific anthology pages (1-based, yt-dlp
    /// `--playlist-items` equivalent): None = all parts, and the selection is
    /// applied in playlist order regardless of the order given.
    /// Returns the list of downloaded media files in order.
    pub fn download(
        &self,
        url: &str,
        out_dir: &Path,
        quality: u32,
        parts: Option<&[u32]>,
    ) -> Result<Vec<PathBuf>, String> {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("create dir: {e}"))?;
        if url.contains("bilibili.com") {
            self.bilibili_download(url, out_dir, quality, parts)
        } else if url.contains("youtube.com") || url.contains("youtu.be") {
            self.youtube_download(url, out_dir, quality)
        } else {
            Err("unsupported URL (only Bilibili and YouTube are supported)".into())
        }
    }

    // -----------------------------------------------------------------------
    // Bilibili
    // -----------------------------------------------------------------------

    fn bilibili_probe_titles(&self, url: &str) -> Result<Vec<String>, String> {
        let meta = self.bilibili_meta(url)?;
        if meta.pages.len() > 1 {
            Ok(meta
                .pages
                .iter()
                .enumerate()
                .map(|(i, (_, _, part))| format!("{} p{:02} {}", meta.title, i + 1, part))
                .collect())
        } else {
            Ok(vec![meta.title])
        }
    }

    fn bilibili_download(
        &self,
        url: &str,
        out_dir: &Path,
        quality: u32,
        parts: Option<&[u32]>,
    ) -> Result<Vec<PathBuf>, String> {
        self.log(format!("[downloader] downloading {url}"));
        let meta = self.bilibili_meta(url)?;

        // If a specific ?p= is requested, keep only that page.
        let part_id = query_param(url, "p").and_then(|v| v.parse::<u32>().ok());
        let mut selected: Vec<(u32, u64, String)> = if let Some(pid) = part_id {
            meta.pages
                .iter()
                .filter(|(p, _, _)| *p == pid)
                .cloned()
                .collect()
        } else {
            meta.pages.clone()
        };
        // --playlist-items equivalent: keep only the requested pages, in
        // playlist order; requested pages that no longer exist are skipped
        // with a log line like yt-dlp's missing-entry warning.
        if let Some(want) = parts.filter(|w| !w.is_empty()) {
            let (keep, missing) = filter_pages_by_parts(&selected, want);
            for p in &missing {
                self.log(format!("[downloader] page p{p} not found, skipping"));
            }
            selected = keep;
        }
        if selected.is_empty() {
            return Err(format!("no video page p{} found", part_id.unwrap_or(0)));
        }
        let multiple = selected.len() > 1;
        self.log(format!("URL yields {} video(s)", selected.len()));

        let mut files = Vec::new();
        let total = selected.len() as u32;
        for (idx, (_page, cid, part_title)) in selected.iter().enumerate() {
            if self.cancel.load(Ordering::SeqCst) {
                return Err("cancelled".into());
            }
            let play_info = self.bilibili_playurl(&meta.bvid, *cid)?;
            let filename = if multiple {
                format!(
                    "{:03}_{} [{}].mp4",
                    idx + 1,
                    sanitize_filename(part_title),
                    meta.bvid
                )
            } else {
                format!("{} [{}].mp4", sanitize_filename(&meta.title), meta.bvid)
            };
            // Per-part 0..100 maps onto the overall download so a multi-part
            // URL advances monotonically: part k spans ((k-1)*100/n, k*100/n]
            // and the last part ends at exactly 100.
            let part_progress = Arc::new(Mutex::new(Box::new({
                let on_progress = self.on_progress.clone();
                move |p: u8| {
                    let overall = ((idx as u32 * 100 + p as u32) / total).min(100) as u8;
                    let mut f = on_progress.lock().unwrap();
                    (f)(overall);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            let dest = out_dir.join(filename);
            self.bilibili_download_playinfo(&play_info, &dest, url, quality, &part_progress)?;
            files.push(dest);
        }
        Ok(files)
    }

    /// Resolve a Bilibili URL into bvid/title/pages. Prefers the unsigned
    /// `x/web-interface/view` API, which keeps working when the HTML page is
    /// risk-blocked (HTTP 412 / code -352); falls back to scraping
    /// `__INITIAL_STATE__` from the page (also covers festival pages).
    fn bilibili_meta(&self, url: &str) -> Result<BiliMeta, String> {
        let mut api_headers: Vec<(&str, String)> =
            vec![("Referer", "https://www.bilibili.com/".into())];
        if let Some(pair) = self.cookie_pair("https://api.bilibili.com/") {
            api_headers.push(pair);
        }
        let api_headers_ref: Vec<(&str, &str)> =
            api_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

        if let Some((id, is_bvid)) = bilibili_id_from_url(url) {
            let q = if is_bvid {
                format!("bvid={}", urlencode(&id))
            } else {
                format!("aid={}", urlencode(id.trim_start_matches("av")))
            };
            if let Ok(v) = self
                .http
                .get_json(&format!("https://api.bilibili.com/x/web-interface/view?{q}"), &api_headers_ref)
            {
                let code = v.get("code").and_then(Value::as_i64).unwrap_or(-1);
                if code == 0
                    && let Some(d) = v.get("data")
                {
                    let bvid = d
                            .get("bvid")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        let title = d
                            .get("title")
                            .and_then(Value::as_str)
                            .unwrap_or("video")
                            .to_string();
                        let mut pages = Vec::new();
                        if let Some(arr) = d.get("pages").and_then(Value::as_array) {
                            for p in arr {
                                let page =
                                    p.get("page").and_then(Value::as_u64).unwrap_or(1) as u32;
                                let cid = p.get("cid").and_then(Value::as_u64).unwrap_or(0);
                                let part = p
                                    .get("part")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string();
                                if cid != 0 {
                                    pages.push((page, cid, part));
                                }
                            }
                        }
                        if pages.is_empty() {
                            let cid = d.get("cid").and_then(Value::as_u64).unwrap_or(0);
                            if cid != 0 {
                                pages.push((1, cid, title.clone()));
                            }
                        }
                    if !bvid.is_empty() && !pages.is_empty() {
                        return Ok(BiliMeta { bvid, title, pages });
                    }
                }
            }
        }

        // Fallback: scrape the HTML page (festival pages etc.).
        let mut page_headers: Vec<(&str, String)> = vec![("Referer", "https://www.bilibili.com/".into())];
        if let Some(pair) = self.cookie_pair(url) {
            page_headers.push(pair);
        }
        let page_headers_ref: Vec<(&str, &str)> =
            page_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let html = self.http.get_text(url, &page_headers_ref)?;
        let initial = extract_json_object(&html, "window.__INITIAL_STATE__").ok_or_else(|| {
            if html.contains("v_voucher") || html.contains("err-code\">412") {
                "bilibili risk control rejected the request (rate limited); try again later"
                    .to_string()
            } else {
                "could not find __INITIAL_STATE__ on the bilibili page".to_string()
            }
        })?;
        // Festival pages have a different layout.
        let is_festival = initial.get("videoData").is_none();
        let video_data = if is_festival {
            initial
                .get("videoInfo")
                .cloned()
                .ok_or("no videoInfo in initial state")?
        } else {
            initial
                .get("videoData")
                .cloned()
                .ok_or("no videoData in initial state")?
        };
        let bvid = video_data
            .get("bvid")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let title = video_data
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("video")
            .to_string();
        let mut pages = Vec::new();
        if let Some(arr) = video_data.get("pages").and_then(Value::as_array) {
            for p in arr {
                let page = p.get("page").and_then(Value::as_u64).unwrap_or(1) as u32;
                let cid = p.get("cid").and_then(Value::as_u64).unwrap_or(0);
                let part = p
                    .get("part")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if cid != 0 {
                    pages.push((page, cid, part));
                }
            }
        }
        if pages.is_empty() {
            let cid = video_data.get("cid").and_then(Value::as_u64).unwrap_or(0);
            if cid != 0 {
                pages.push((1, cid, title.clone()));
            }
        }
        if bvid.is_empty() || pages.is_empty() {
            return Err("could not resolve bilibili video metadata".into());
        }
        Ok(BiliMeta { bvid, title, pages })
    }

    /// Bilibili WBI signing (see `BilibiliBaseIE._sign_wbi`). The key is cached for 30 s per session, matching yt-dlp's `_WBI_KEY_CACHE_TIMEOUT`.
    fn bilibili_wbi_key(&self) -> Result<String, String> {
        use std::sync::Mutex;
        use std::time::{Duration, Instant};
        static CACHE: Mutex<Option<(String, Instant)>> = Mutex::new(None);
        {
            let guard = CACHE.lock().unwrap();
            if let Some((key, ts)) = guard.as_ref()
                && ts.elapsed() < Duration::from_secs(30)
            {
                return Ok(key.clone());
            }
        }
        let nav_headers: Vec<(&str, String)> = {
            let mut h = vec![("Referer", "https://www.bilibili.com/".to_string())];
            if let Some(pair) = self.cookie_pair("https://api.bilibili.com/") {
                h.push(pair);
            }
            h
        };
        let nav_headers_ref: Vec<(&str, &str)> =
            nav_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let nav = self
            .http
            .get_json("https://api.bilibili.com/x/web-interface/nav", &nav_headers_ref)?;
        let img = nav
            .pointer("/data/wbi_img/img_url")
            .and_then(Value::as_str)
            .unwrap_or("");
        let sub = nav
            .pointer("/data/wbi_img/sub_url")
            .and_then(Value::as_str)
            .unwrap_or("");
        let img_key = img
            .rsplit('/')
            .next()
            .unwrap_or("")
            .split('.')
            .next()
            .unwrap_or("");
        let sub_key = sub
            .rsplit('/')
            .next()
            .unwrap_or("")
            .split('.')
            .next()
            .unwrap_or("");
        let lookup = format!("{img_key}{sub_key}");
        const MIXIN_KEY_ENC_TAB: [usize; 64] = [
            46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42,
            19, 29, 28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60,
            51, 30, 4, 22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
        ];
        let key: String = MIXIN_KEY_ENC_TAB
            .iter()
            .filter_map(|&i| lookup.chars().nth(i))
            .take(32)
            .collect();
        *CACHE.lock().unwrap() = Some((key.clone(), Instant::now()));
        Ok(key)
    }

    /// Randomised danmaku/fingerprint params that Bilibili expects on playurl requests (mirrors `BilibiliBaseIE._dm_params`). Sending these materially reduces `-352` risk-control rejections on unauthenticated requests.
    fn bilibili_dm_params(&self) -> Vec<(String, String)> {
        struct Rng(u64);
        impl Rng {
            fn next(&mut self) -> u32 {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (self.0 >> 33) as u32
            }
        }
        let mut rng = Rng(0x9E3779B97F4A7C15
            ^ (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0) as u64));

        const ALPHA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        fn pad(rng: &mut Rng, n: usize) -> String {
            let mut v = String::new();
            for _ in 0..n {
                v.push(ALPHA[(rng.next() as usize) % 64] as char);
            }
            v
        }
        let dm_img_str = {
            let n = rng.next();
            let s = pad(&mut rng, 16 + (n as usize) % 49);
            s[..s.len() - 2].to_string()
        };
        let dm_cover_img_str = {
            let n = rng.next();
            let s = pad(&mut rng, 32 + (n as usize) % 97);
            s[..s.len() - 2].to_string()
        };
        let (w, h) = match rng.next() % 18 {
            0..=2 => (1920i64, 1080i64),
            3..=5 => (1366, 768),
            6..=7 => (1536, 864),
            8 => (1280, 720),
            9 => (2560, 1440),
            10..=11 => (1440, 900),
            _ => (1600, 900),
        };
        let rnd_wh = (rng.next() % 114) as i64;
        let wh = format!(
            "[{}, {}, {}]",
            2 * w + 2 * h + 3 * rnd_wh,
            4 * w - h + rnd_wh,
            rnd_wh
        );
        let scroll_top = (rng.next() % 101) as i64;
        let rnd_of = (rng.next() % 514) as i64;
        let of = format!(
            "[{}, {}, {}]",
            3 * scroll_top + 2 * 10 + rnd_of,
            4 * scroll_top - 4 * 10 + 2 * rnd_of,
            rnd_of
        );
        vec![
            ("dm_img_list".to_string(), "[]".to_string()),
            ("dm_img_str".to_string(), dm_img_str),
            ("dm_cover_img_str".to_string(), dm_cover_img_str),
            (
                "dm_img_inter".to_string(),
                format!("{{\"ds\":[],\"wh\":{wh},\"of\":{of}}}"),
            ),
        ]
    }

    /// Fetch the playurl (DASH formats) for one cid, WBI-signed and carrying the danmaku fingerprint params, exactly as yt-dlp's `_download_playinfo` does. `try_look` is only sent for anonymous sessions (yt-dlp drops it when logged in); browser cookies are forwarded so logged-in qualities are returned.
    fn bilibili_playurl(&self, bvid: &str, cid: u64) -> Result<Value, String> {
        let wbi_key = self.bilibili_wbi_key()?;
        let wts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut params: HashMap<String, String> = [
            ("bvid".to_string(), bvid.to_string()),
            ("cid".to_string(), cid.to_string()),
            ("fnval".to_string(), "4048".to_string()),
            ("wts".to_string(), wts.to_string()),
        ]
        .into_iter()
        .collect();
        // yt-dlp only sends try_look when NOT logged in (`is_logged_in` = SESSDATA).
        if !self.bilibili_logged_in() {
            params.insert("try_look".to_string(), "1".to_string());
        }
        for (k, v) in self.bilibili_dm_params() {
            params.insert(k, v);
        }
        // Remove chars that must be stripped
        let mut sorted: Vec<(String, String)> = params
            .iter()
            .map(|(k, v)| {
                let clean: String = v.chars().filter(|c| !"!'()*".contains(*c)).collect();
                (k.clone(), clean)
            })
            .collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        let query = sorted
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&");
        let w_rid = md5::hex_digest(format!("{query}{wbi_key}").as_bytes());
        params.insert("w_rid".into(), w_rid);
        let mut qs = sorted
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&");
        qs.push_str(&format!("&w_rid={}", urlencode(&params["w_rid"])));

        let url = format!("https://api.bilibili.com/x/player/wbi/playurl?{qs}");
        let mut req_headers: Vec<(&str, String)> = vec![
            ("Referer", "https://www.bilibili.com/".into()),
            ("Origin", "https://www.bilibili.com".into()),
        ];
        if let Some(pair) = self.cookie_pair("https://api.bilibili.com/") {
            req_headers.push(pair);
        }
        let req_headers_ref: Vec<(&str, &str)> =
            req_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let raw = self.http.get_json(&url, &req_headers_ref)?;
        let code = raw.get("code").and_then(Value::as_i64).unwrap_or(-1);
        if code != 0 {
            let msg = raw
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown error");
            return Err(format!("bilibili playurl failed (code {code}): {msg}"));
        }
        raw.get("data")
            .cloned()
            .ok_or_else(|| "bilibili playurl: no data".into())
    }

    fn bilibili_download_playinfo(
        &self,
        play_info: &Value,
        dest: &Path,
        referer: &str,
        quality: u32,
        on_progress: &ProgressCb,
    ) -> Result<(), String> {
        let report = |p: u8| {
            let mut f = on_progress.lock().unwrap();
            (f)(p);
        };
        // CDN request headers: yt-dlp downloads DASH media with
        // `http_headers: {'Referer': url}` (+ cookies for matching domains).
        let cdn_headers = |media_url: &str| -> Vec<(&'static str, String)> {
            let mut h: Vec<(&'static str, String)> = vec![("Referer", referer.to_string())];
            if let Some(pair) = self.cookie_pair(media_url) {
                h.push(pair);
            }
            h
        };

        if let Some(dash) = play_info.get("dash") {
            let videos = dash
                .get("video")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            // yt-dlp includes `dash.audio`, `dash.dolby.audio` and `dash.flac.audio`
            // in the format list; the best one is picked by bandwidth.
            let mut audios = dash
                .get("audio")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            for extra in ["dolby", "flac"] {
                if let Some(list) = dash
                    .get(extra)
                    .and_then(|d| d.get("audio"))
                    .and_then(Value::as_array)
                {
                    audios.extend(list.iter().cloned());
                }
            }

            let best_video = videos
                .iter()
                .filter(|v| {
                    v.get("height")
                        .and_then(Value::as_u64)
                        .map(|h| h <= quality as u64)
                        .unwrap_or(false)
                })
                .max_by_key(|v| v.get("height").and_then(Value::as_u64).unwrap_or(0))
                .or_else(|| {
                    videos
                        .iter()
                        .max_by_key(|v| v.get("height").and_then(Value::as_u64).unwrap_or(0))
                })
                .cloned()
                .ok_or("bilibili: no suitable video format")?;
            let best_audio = audios
                .iter()
                .max_by_key(|a| a.get("bandwidth").and_then(Value::as_u64).unwrap_or(0))
                .cloned();

            let video_url = best_video
                .get("baseUrl")
                .and_then(Value::as_str)
                .or_else(|| best_video.get("base_url").and_then(Value::as_str))
                .ok_or("bilibili: video url missing")?
                .to_string();
            let audio_url = best_audio
                .as_ref()
                .and_then(|a| {
                    a.get("baseUrl")
                        .and_then(Value::as_str)
                        .or_else(|| a.get("base_url").and_then(Value::as_str))
                })
                .map(|s| s.to_string());

            let tmp_video = dest.with_extension("video.m4s");
            let tmp_audio = dest.with_extension("audio.m4s");
            let v_headers = cdn_headers(&video_url);
            let v_headers_ref: Vec<(&str, &str)> =
                v_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

            // Video (50% of progress), audio (next 30%), merge (last 20%).
            let video_progress = Arc::new(Mutex::new(Box::new({
                let on_progress = on_progress.clone();
                move |p: u8| {
                    let mut f = on_progress.lock().unwrap();
                    (f)(p / 2);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            self.http.download_file(
                &video_url,
                &tmp_video,
                &v_headers_ref,
                &self.cancel,
                video_progress,
            )?;

            if let Some(aurl) = audio_url {
                let a_headers = cdn_headers(&aurl);
                let a_headers_ref: Vec<(&str, &str)> =
                    a_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
                let audio_progress = Arc::new(Mutex::new(Box::new({
                    let on_progress = on_progress.clone();
                    move |p: u8| {
                        let mut f = on_progress.lock().unwrap();
                        (f)(50 + p * 3 / 10);
                    }
                }) as Box<dyn FnMut(u8) + Send>));
                self.http.download_file(
                    &aurl,
                    &tmp_audio,
                    &a_headers_ref,
                    &self.cancel,
                    audio_progress,
                )?;
            }

            report(80);
            self.merge_av(&tmp_video, tmp_audio.exists().then_some(tmp_audio.clone()), dest)?;
            report(100);
            return Ok(());
        }

        // Legacy non-DASH response (`durl`): yt-dlp treats a single segment as
        // one plain http format; multi-segment FLV becomes a multi-video that
        // does not fit this pipeline, so it is rejected with a clear message.
        let durl = play_info
            .get("durl")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if durl.len() > 1 {
            return Err(format!(
                "bilibili: legacy video is split into {} segments (old FLV format), which this app cannot download",
                durl.len()
            ));
        }
        let url = durl
            .first()
            .and_then(|f| {
                f.get("url")
                    .and_then(Value::as_str)
                    .or_else(|| f.get("baseUrl").and_then(Value::as_str))
            })
            .ok_or("bilibili: no dash or durl formats in playurl")?;
        let headers = cdn_headers(url);
        let headers_ref: Vec<(&str, &str)> =
            headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.http.download_file(
            url,
            dest,
            &headers_ref,
            &self.cancel,
            on_progress.clone(),
        )?;
        report(100);
        Ok(())
    }

    /// ffmpeg `-c copy` merge with `+faststart`, the same flags yt-dlp's
    /// FFmpegMergerPP uses for mp4 output. Removes the inputs afterwards.
    fn merge_av(&self, video: &Path, audio: Option<PathBuf>, dest: &Path) -> Result<(), String> {
        let mut cmd = std::process::Command::new(&self.ffmpeg);
        cmd.arg("-y").arg("-i").arg(video);
        if let Some(a) = &audio {
            cmd.arg("-i").arg(a);
        }
        cmd.arg("-c")
            .arg("copy")
            .arg("-movflags")
            .arg("+faststart")
            .arg(dest);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("ffmpeg: {e}"))?;
        let _ = std::fs::remove_file(video);
        if let Some(a) = audio {
            let _ = std::fs::remove_file(a);
        }
        if !out.status.success() {
            let _ = std::fs::remove_file(dest);
            return Err(format!(
                "ffmpeg merge failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(())
    }

    /// Concatenate several already-downloaded videos into one file with the
    /// same stream-copy merge `merge_av` uses: ffmpeg's concat demuxer with
    /// `-c copy -movflags +faststart` (no re-encode; the parts of one
    /// anthology share codecs/parameters). Inputs and the temp list file are
    /// removed on success. A single input is moved onto `dest` directly.
    pub fn merge_videos(&self, files: &[PathBuf], dest: &Path) -> Result<(), String> {
        if files.is_empty() {
            return Err("merge: no input files".into());
        }
        if files.len() == 1 {
            std::fs::rename(&files[0], dest).map_err(|e| format!("move part -> dest: {e}"))?;
            return Ok(());
        }
        let list = dest.with_extension("concat.txt");
        let mut text = String::new();
        for f in files {
            // concat-demuxer quoting: wrap in single quotes, escape inner ones
            let quoted = f.display().to_string().replace('\'', "'\\''");
            text.push_str(&format!("file '{quoted}'\n"));
        }
        std::fs::write(&list, text).map_err(|e| format!("write concat list: {e}"))?;
        let mut cmd = std::process::Command::new(&self.ffmpeg);
        cmd.arg("-y")
            .arg("-f")
            .arg("concat")
            .arg("-safe")
            .arg("0")
            .arg("-i")
            .arg(&list)
            .arg("-c")
            .arg("copy")
            .arg("-movflags")
            .arg("+faststart")
            .arg(dest);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("ffmpeg: {e}"))?;
        let _ = std::fs::remove_file(&list);
        if !out.status.success() {
            let _ = std::fs::remove_file(dest);
            return Err(format!(
                "ffmpeg concat failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        for f in files {
            let _ = std::fs::remove_file(f);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // YouTube
    // -----------------------------------------------------------------------

    fn youtube_probe_titles(&self, url: &str) -> Result<Vec<String>, String> {
        let video_id = youtube_video_id(url).ok_or("could not parse YouTube video id")?;
        let pr = self.youtube_player_response(&video_id)?;
        let title = pr
            .pointer("/videoDetails/title")
            .and_then(Value::as_str)
            .unwrap_or("video")
            .to_string();
        Ok(vec![title])
    }

    fn youtube_download(
        &self,
        url: &str,
        out_dir: &Path,
        quality: u32,
    ) -> Result<Vec<PathBuf>, String> {
        self.log(format!("[downloader] downloading {url}"));
        let video_id = youtube_video_id(url).ok_or("could not parse YouTube video id")?;
        let pr = self.youtube_player_response(&video_id)?;
        let title = pr
            .pointer("/videoDetails/title")
            .and_then(Value::as_str)
            .unwrap_or("video")
            .to_string();
        let filename = format!("{} [{}].mp4", sanitize_filename(&title), video_id);
        let dest = out_dir.join(filename);

        let streaming = pr
            .get("streamingData")
            .ok_or("youtube: no streamingData in player response")?;
        let adaptive = streaming
            .get("adaptiveFormats")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let muxed = streaming
            .get("formats")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        // Format selection follows yt-dlp's default `bv*+ba/b`: separate best
        // video + best audio first, a progressive (muxed) format only as a
        // last resort. DRM and OTF streams are skipped like yt-dlp does.
        let usable = |f: &Value| -> bool {
            let has_drm = f
                .get("drmFamilies")
                .and_then(Value::as_array)
                .map(|a| !a.is_empty())
                .unwrap_or(false);
            !has_drm
                && f.get("type").and_then(Value::as_str) != Some("FORMAT_STREAM_TYPE_OTF")
        };
        let video_key = |v: &Value| {
            (
                v.get("height").and_then(Value::as_u64).unwrap_or(0),
                v.get("bandwidth").and_then(Value::as_u64).unwrap_or(0),
            )
        };
        let videos: Vec<Value> = adaptive
            .iter()
            .filter(|f| {
                usable(f)
                    && f.get("mimeType")
                        .and_then(Value::as_str)
                        .map(|m| m.starts_with("video/"))
                        .unwrap_or(false)
            })
            .cloned()
            .collect();
        let audios: Vec<Value> = adaptive
            .iter()
            .filter(|f| {
                usable(f)
                    && f.get("mimeType")
                        .and_then(Value::as_str)
                        .map(|m| m.starts_with("audio/"))
                        .unwrap_or(false)
            })
            .cloned()
            .collect();

        let best_video = videos
            .iter()
            .filter(|v| video_key(v).0 > 0 && video_key(v).0 <= quality as u64)
            .max_by_key(|v| video_key(v))
            .or_else(|| videos.iter().max_by_key(|v| video_key(v)))
            .cloned();
        let best_audio = audios
            .iter()
            .max_by_key(|a| a.get("bitrate").and_then(Value::as_u64).unwrap_or(0))
            .cloned();

        if let (Some(best_video), Some(best_audio)) = (best_video, best_audio) {
            let video_url = youtube_format_url(&best_video)?;
            let audio_url = youtube_format_url(&best_audio)?;

            let tmp_video = dest.with_extension("video.m4v");
            let tmp_audio = dest.with_extension("audio.m4a");
            let mut v_headers: Vec<(&'static str, String)> = Vec::new();
            if let Some(pair) = self.cookie_pair(&video_url) {
                v_headers.push(pair);
            }
            let v_headers_ref: Vec<(&str, &str)> =
                v_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let video_progress = Arc::new(Mutex::new(Box::new({
                let on_progress = self.on_progress.clone();
                move |p: u8| {
                    let mut f = on_progress.lock().unwrap();
                    (f)(p / 2);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            self.http.download_file(
                &video_url,
                &tmp_video,
                &v_headers_ref,
                &self.cancel,
                video_progress,
            )?;
            let mut a_headers: Vec<(&'static str, String)> = Vec::new();
            if let Some(pair) = self.cookie_pair(&audio_url) {
                a_headers.push(pair);
            }
            let a_headers_ref: Vec<(&str, &str)> =
                a_headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let audio_progress = Arc::new(Mutex::new(Box::new({
                let on_progress = self.on_progress.clone();
                move |p: u8| {
                    let mut f = on_progress.lock().unwrap();
                    (f)(50 + p * 3 / 10);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            self.http.download_file(
                &audio_url,
                &tmp_audio,
                &a_headers_ref,
                &self.cancel,
                audio_progress,
            )?;
            self.progress(80);
            self.merge_av(&tmp_video, Some(tmp_audio), &dest)?;
            self.progress(100);
            return Ok(vec![dest]);
        }

        // Fallback: best progressive (muxed) format at the requested quality.
        let progressive_key = |f: &Value| f.get("height").and_then(Value::as_u64).unwrap_or(0);
        let progressive = muxed
            .iter()
            .filter(|f| {
                usable(f)
                    && f.get("audioQuality").is_some()
                    && progressive_key(f) > 0
                    && progressive_key(f) <= quality as u64
            })
            .max_by_key(|f| progressive_key(f))
            .or_else(|| muxed.iter().filter(|f| usable(f)).max_by_key(|f| progressive_key(f)))
            .ok_or("youtube: no suitable video format")?;
        let fmt_url = youtube_format_url(progressive)?;
        let mut headers: Vec<(&'static str, String)> = Vec::new();
        if let Some(pair) = self.cookie_pair(&fmt_url) {
            headers.push(pair);
        }
        let headers_ref: Vec<(&str, &str)> =
            headers.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.http.download_file(
            &fmt_url,
            &dest,
            &headers_ref,
            &self.cancel,
            self.on_progress.clone(),
        )?;
        self.progress(100);
        Ok(vec![dest])
    }

    /// YouTube Innertube player response. Mirrors yt-dlp's default client list
    /// `('visionos', 'web')`: visionOS is tried first (JS-less, no PO token),
    /// then the web client with cookies + SAPISIDHASH authorization when
    /// available. Client identification headers and the visitor id from a
    /// previous response are sent like `generate_api_headers` does.
    fn youtube_player_response(&self, video_id: &str) -> Result<Value, String> {
        struct YtClient {
            name: &'static str,
            context: Value,
            client_name_id: u32,
            version: &'static str,
            /// yt-dlp sends the client's own User-Agent when its context has one.
            user_agent: Option<&'static str>,
            /// yt-dlp SUPPORTS_COOKIES: only these clients get cookies + auth.
            supports_cookies: bool,
        }
        let clients = [
            YtClient {
                name: "visionos",
                context: serde_json::json!({
                    "clientName": "VISIONOS",
                    "clientVersion": "1.02",
                    "deviceMake": "Apple",
                    "deviceModel": "RealityDevice17,1",
                    "userAgent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15",
                    "osName": "visionOS",
                    "osVersion": "26.5.23O471",
                    "hl": "en",
                    "timeZone": "UTC",
                    "utcOffsetMinutes": 0,
                }),
                client_name_id: 101,
                version: "1.02",
                user_agent: Some("Mozilla/5.0 (Macintosh; Intel Mac OS X 15_7_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.0 Safari/605.1.15"),
                supports_cookies: false,
            },
            YtClient {
                name: "web",
                context: serde_json::json!({
                    "clientName": "WEB",
                    "clientVersion": "2.20260708.00.00",
                    "hl": "en",
                    "timeZone": "UTC",
                    "utcOffsetMinutes": 0,
                }),
                client_name_id: 1,
                version: "2.20260708.00.00",
                user_agent: None,
                supports_cookies: true,
            },
        ];
        let mut last_err = String::new();
        let mut visitor_data: Option<String> = None;
        for client in &clients {
            let mut headers: Vec<(&str, String)> = vec![
                ("Origin", "https://www.youtube.com".into()),
                ("Referer", "https://www.youtube.com/".into()),
                ("X-YouTube-Client-Name", client.client_name_id.to_string()),
                ("X-YouTube-Client-Version", client.version.into()),
            ];
            if let Some(ua) = client.user_agent {
                headers.push(("User-Agent", ua.into()));
            }
            if let Some(vd) = &visitor_data {
                headers.push(("X-Goog-Visitor-Id", vd.clone()));
            }
            if client.supports_cookies {
                if let Some(pair) = self.cookie_pair("https://www.youtube.com/") {
                    headers.push(pair);
                }
                if let Some(auth) = self.sid_authorization() {
                    headers.push(("Authorization", auth));
                    headers.push(("X-Origin", "https://www.youtube.com".into()));
                }
            }
            let headers_ref: Vec<(&str, &str)> =
                headers.iter().map(|(k, v)| (*k, v.as_str())).collect();

            let body = serde_json::json!({
                "videoId": video_id,
                "context": { "client": client.context },
            });
            match self.http.post_json(
                "https://www.youtube.com/youtubei/v1/player?prettyPrint=false",
                &body,
                &headers_ref,
            ) {
                Ok(pr) => {
                    if visitor_data.is_none() {
                        visitor_data = pr
                            .pointer("/responseContext/visitorData")
                            .and_then(Value::as_str)
                            .map(str::to_string);
                    }
                    let status = pr
                        .pointer("/playabilityStatus/status")
                        .and_then(Value::as_str)
                        .unwrap_or("ERROR");
                    if status == "OK" {
                        return Ok(pr);
                    }
                    let reason = pr
                        .pointer("/playabilityStatus/reason")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown");
                    last_err = format!("{} client: playability {status} ({reason})", client.name);
                }
                Err(e) => last_err = format!("{} client: {e}", client.name),
            }
        }
        Err(format!("youtube player response failed: {last_err}"))
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Range-chunk size for media downloads (yt-dlp uses 10 MiB chunks too).
const DL_CHUNK_SIZE: u64 = 10 * 1024 * 1024;
/// Per-chunk retry attempts (yt-dlp's default --retries is higher; 3 keeps
/// the UI wait bounded).
const DL_MAX_ATTEMPTS: u32 = 3;

/// Stable `.part` file path for a media URL, so interrupted downloads can be
/// resumed (yt-dlp resumes `.part` files the same way). Keyed by the URL
/// because a re-signed URL may point at different content (other quality).
fn part_path(url: &str) -> PathBuf {
    // FNV-1a 64
    let mut h: u64 = 0xcbf29ce484222325;
    for b in url.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    std::env::temp_dir().join(format!("liveneko_dl_{h:016x}.part"))
}

/// Move a finished `.part` file onto its destination. The part file lives in
/// the temp dir while `dest` lives in the data dir; when the two sit on
/// different drives (data dir moved off the system drive) `rename` fails with
/// ERROR_NOT_SAME_DEVICE (os error 17), so fall back to copy+delete.
fn finalize_part(part: &Path, dest: &Path) -> Result<(), String> {
    if std::fs::rename(part, dest).is_ok() {
        return Ok(());
    }
    std::fs::copy(part, dest).map_err(|e| format!("move part -> dest: {e}"))?;
    std::fs::remove_file(part).map_err(|e| format!("move part -> dest: {e}"))
}

/// Apply a `--playlist-items`-style page selection to an anthology's page
/// list: keeps only the requested pages **in playlist order** (yt-dlp yields
/// entries in playlist order regardless of the order requested) and returns
/// the requested pages that do not exist so the caller can log them.
fn filter_pages_by_parts(
    pages: &[(u32, u64, String)],
    want: &[u32],
) -> (Vec<(u32, u64, String)>, Vec<u32>) {
    let keep: Vec<(u32, u64, String)> = pages
        .iter()
        .filter(|(p, _, _)| want.contains(p))
        .cloned()
        .collect();
    let missing: Vec<u32> = want
        .iter()
        .filter(|w| !pages.iter().any(|(p, _, _)| p == *w))
        .copied()
        .collect();
    (keep, missing)
}

/// Host portion of a URL (no scheme, no path).
fn host_of(url: &str) -> &str {
    let rest = url.split_once("://").map(|(_, r)| r).unwrap_or(url);
    rest.split(['/', '?', '#']).next().unwrap_or(rest)
}

/// Case-insensitive header lookup in a raw header block; returns the value.
fn strip_header<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) {
        Some(v.trim())
    } else {
        None
    }
}

/// yt-dlp `_make_sid_authorization`: sha1("<timestamp> <sid> <origin>").
fn sha1_hex(data: &str) -> String {
    use sha1::{Digest, Sha1};
    let mut hasher = Sha1::new();
    hasher.update(data.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Resolve a YouTube format's media URL. Formats without a plain `url` carry
/// `signatureCipher`, which needs the player's JavaScript to decrypt — a
/// limitation shared with yt-dlp running without a JS runtime.
fn youtube_format_url(format: &Value) -> Result<String, String> {
    if let Some(u) = format.get("url").and_then(Value::as_str) {
        return Ok(u.to_string());
    }
    if format.get("signatureCipher").is_some() {
        return Err(
            "youtube: the selected client returned encrypted stream URLs (signatureCipher) and no direct link; try again later or enable browser cookies in Settings".into(),
        );
    }
    Err("youtube: format url missing".into())
}

/// Resolved Bilibili video metadata (from the view API or the HTML page).
struct BiliMeta {
    bvid: String,
    title: String,
    /// (page number, cid, part title)
    pages: Vec<(u32, u64, String)>,
}

/// Extract the Bilibili video id from a URL. Returns `(id, is_bvid)`.
fn bilibili_id_from_url(url: &str) -> Option<(String, bool)> {
    if let Some(rest) = query_param(url, "bvid")
        && !rest.is_empty()
    {
        return Some((rest, true));
    }
    if let Some(pos) = url.find("/video/") {
        let rest = &url[pos + 7..];
        let id: String = rest
            .chars()
            .take_while(|c| !matches!(c, '?' | '#' | '/'))
            .collect();
        if !id.is_empty() {
            let lower = id.to_lowercase();
            if lower.starts_with("bv") {
                return Some((id, true));
            }
            if lower.starts_with("av") {
                return Some((id, false));
            }
        }
    }
    None
}

/// Extract a balanced JSON object that starts after `marker`.
fn extract_json_object(text: &str, marker: &str) -> Option<Value> {
    let start = text.find(marker)? + marker.len();
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    let mut json_start = None;
    for (i, c) in text[start..].char_indices() {
        let idx = start + i;
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => {
                if depth == 0 {
                    json_start = Some(idx);
                }
                depth += 1;
            }
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    let s = json_start?;
                    let json = &text[s..=idx];
                    return serde_json::from_str(json).ok();
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract a YouTube video ID from a URL.
fn youtube_video_id(url: &str) -> Option<String> {
    // youtu.be/<id>
    if let Some(rest) = url.strip_prefix("https://youtu.be/") {
        return Some(
            rest.split(['?', '&', '/'])
                .next()?
                .to_string(),
        );
    }
    // watch?v=<id>
    if url.contains("youtube.com/watch") {
        return query_param(url, "v");
    }
    // shorts/<id>, embed/<id>, live/<id>
    for pat in [
        "youtube.com/shorts/",
        "youtube.com/embed/",
        "youtube.com/live/",
    ] {
        if let Some(pos) = url.find(pat) {
            let rest = &url[pos + pat.len()..];
            return Some(
                rest.split(['?', '&', '/'])
                    .next()?
                    .to_string(),
            );
        }
    }
    None
}

fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_md5() {
        assert_eq!(md5::hex_digest(b""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(
            md5::hex_digest(b"hello"),
            "5d41402abc4b2a76b9719d911017c592"
        );
    }

    #[test]
    fn test_extract_json() {
        let html = r#"<script>window.__INITIAL_STATE__={"a":1,"b":{"c":2}};</script>"#;
        let v = extract_json_object(html, "window.__INITIAL_STATE__=").unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"]["c"], 2);
    }

    #[test]
    fn test_youtube_id() {
        assert_eq!(
            youtube_video_id("https://www.youtube.com/watch?v=abc123"),
            Some("abc123".into())
        );
        assert_eq!(
            youtube_video_id("https://youtu.be/abc123?t=10"),
            Some("abc123".into())
        );
        assert_eq!(
            youtube_video_id("https://www.youtube.com/shorts/xyz789"),
            Some("xyz789".into())
        );
    }

    #[test]
    fn test_bilibili_id() {
        assert_eq!(
            bilibili_id_from_url("https://www.bilibili.com/video/BV1gKdQBiELm"),
            Some(("BV1gKdQBiELm".into(), true))
        );
        assert_eq!(
            bilibili_id_from_url("https://www.bilibili.com/video/BV1gKdQBiELm?p=2"),
            Some(("BV1gKdQBiELm".into(), true))
        );
        assert_eq!(
            bilibili_id_from_url("https://www.bilibili.com/festival/bh3-7th?bvid=BV1tr4y1f7p2&"),
            Some(("BV1tr4y1f7p2".into(), true))
        );
        assert_eq!(
            bilibili_id_from_url("https://www.bilibili.com/video/av170001"),
            Some(("av170001".into(), false))
        );
        assert_eq!(
            bilibili_id_from_url("https://www.youtube.com/watch?v=x"),
            None
        );
    }

    // Live network tests (no auth). Run with: cargo test --lib downloader::tests::live -- --ignored --nocapture
    fn live_dl() -> Downloader {
        Downloader::new(
            Arc::new(AtomicBool::new(false)),
            Arc::new(Mutex::new(
                Box::new(|p: u8| eprintln!("  progress {p}%")) as Box<dyn FnMut(u8) + Send>
            )),
            Arc::new(Mutex::new(
                Box::new(|l: String| eprintln!("  {l}")) as Box<dyn FnMut(String) + Send>
            )),
            None,
        )
    }

    #[test]
    fn test_part_path_is_stable_and_url_keyed() {
        let a = part_path("https://example.com/a?x=1");
        let b = part_path("https://example.com/a?x=1");
        let c = part_path("https://example.com/a?x=2");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.to_string_lossy().ends_with(".part"));
    }

    #[test]
    fn test_host_of() {
        assert_eq!(host_of("https://api.bilibili.com/x?y=1"), "api.bilibili.com");
        assert_eq!(host_of("https://rr1---sn-x.googlevideo.com/videoplayback?n=a"), "rr1---sn-x.googlevideo.com");
    }

    #[test]
    fn test_filter_pages_by_parts() {
        let pages = |ps: &[(u32, &str)]| -> Vec<(u32, u64, String)> {
            ps.iter().map(|(p, t)| (*p, 100 + *p as u64, t.to_string())).collect()
        };
        let all = pages(&[(1, "a"), (2, "b"), (3, "c"), (4, "d")]);

        // subset selection comes back in playlist order, not request order
        let (keep, missing) = filter_pages_by_parts(&all, &[3, 1]);
        assert_eq!(keep, pages(&[(1, "a"), (3, "c")]));
        assert!(missing.is_empty());

        // requested pages that do not exist are reported, the rest kept
        let (keep, missing) = filter_pages_by_parts(&all, &[2, 9]);
        assert_eq!(keep, pages(&[(2, "b")]));
        assert_eq!(missing, vec![9]);

        // duplicates in the request must not duplicate downloads
        let (keep, missing) = filter_pages_by_parts(&all, &[2, 2]);
        assert_eq!(keep, pages(&[(2, "b")]));
        assert!(missing.is_empty());
    }

    #[test]
    fn test_sha1_hex() {
        // sha1("abc") from FIPS 180-1 test vectors
        assert_eq!(sha1_hex("abc"), "a9993e364706816aba3e25717850c26c9cd0d89d");
    }

    #[test]
    fn test_sid_authorization() {
        use crate::cookies::Cookie;
        let mk = |name: &str, value: &str| Cookie {
            name: name.into(),
            value: value.into(),
            domain: ".youtube.com".into(),
            expires: None,
            secure: true,
        };
        let silent = |_: u8| {};
        let silent_log = |_: String| {};
        let new_dl = |cookies: Option<Arc<Vec<Cookie>>>| {
            Downloader::new(
                Arc::new(AtomicBool::new(false)),
                Arc::new(Mutex::new(
                    Box::new(silent) as Box<dyn FnMut(u8) + Send>
                )),
                Arc::new(Mutex::new(
                    Box::new(silent_log) as Box<dyn FnMut(String) + Send>
                )),
                cookies,
            )
        };
        // no cookies at all -> no auth header
        assert!(new_dl(None).sid_authorization().is_none());
        // cookies not matching youtube -> no auth header
        let off_domain = Cookie {
            name: "SAPISID".into(),
            value: "x".into(),
            domain: ".example.org".into(),
            expires: None,
            secure: false,
        };
        assert!(new_dl(Some(Arc::new(vec![off_domain]))).sid_authorization().is_none());
        // SAPISID + 3PAPISID -> SAPISIDHASH and SAPISID3PHASH, no 1PHASH
        let auth = new_dl(Some(Arc::new(vec![
            mk("SAPISID", "testSapisid123"),
            mk("__Secure-3PAPISID", "third"),
        ])))
        .sid_authorization()
        .unwrap();
        assert_eq!(auth.matches("SAPISIDHASH").count(), 1);
        assert!(!auth.contains("SAPISID1PHASH"));
        assert!(auth.contains("SAPISID3PHASH"));
    }

    #[test]
    #[ignore]
    fn live_bilibili_probe() {
        let d = live_dl();
        let url = std::env::var("BILI_TEST_URL")
            .unwrap_or_else(|_| "https://www.bilibili.com/video/BV1E7uU6tEPA".into());
        let titles = d.probe_titles(&url).unwrap();
        eprintln!("bilibili titles: {titles:?}");
        assert!(!titles.is_empty());
    }

    #[test]
    #[ignore]
    fn live_bilibili_download() {
        let d = live_dl();
        let url = std::env::var("BILI_TEST_URL")
            .unwrap_or_else(|_| "https://www.bilibili.com/video/BV1E7uU6tEPA".into());
        let dir = std::env::temp_dir().join("liveneko_bili_test");
        let files = d.download(&url, &dir, 720, None).unwrap();
        eprintln!("bilibili files: {files:?}");
        assert!(!files.is_empty());
        assert!(files[0].exists());
    }

    #[test]
    #[ignore]
    fn live_youtube_probe() {
        let d = live_dl();
        let titles = d
            .probe_titles("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            .unwrap();
        eprintln!("youtube titles: {titles:?}");
        assert!(!titles.is_empty());
    }
}
