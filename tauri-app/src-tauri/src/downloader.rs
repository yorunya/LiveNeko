//! In-process, unauthenticated video downloader for Bilibili and YouTube.

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

    /// Download a URL to a local file, reporting progress (0..100) via `on_progress`.
    fn download_file(
        &self,
        url: &str,
        dest: &Path,
        extra_headers: &[(&str, &str)],
        cancel: &Arc<AtomicBool>,
        on_progress: Arc<Mutex<Box<dyn FnMut(u8) + Send>>>,
    ) -> Result<(), String> {
        let mut cmd = std::process::Command::new("curl");
        cmd.arg("-sSL")
            .arg("--compressed")
            .arg("--connect-timeout")
            .arg("15")
            .arg("--max-time")
            .arg("600")
            .arg("-o")
            .arg(dest);
        for (k, v) in &self.default_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        for (k, v) in extra_headers {
            cmd.arg("-H").arg(format!("{k}: {v}"));
        }
        cmd.arg(url);
        crate::pipeline::hide_console(&mut cmd);
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("curl: {e}"))?;
        let stderr = child.stderr.take().expect("stderr piped");
        // curl writes a simple progress meter to stderr when not silent; we
        // read it in a thread so we can parse the percentage.
        let cancel2 = cancel.clone();
        let on_progress2 = on_progress.clone();
        let stderr_handle = std::thread::spawn(move || {
            use std::io::{BufRead, BufReader};
            let reader = BufReader::new(stderr);
            let mut last = 0u8;
            for line in reader.lines() {
                if cancel2.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok(l) = line {
                    // curl progress: "  % Total    % Received ..." or "  3.5%"
                    for token in l.split_whitespace() {
                        if let Some(pct) = token.strip_suffix('%') {
                            if let Ok(f) = pct.parse::<f32>() {
                                let p = f as u8;
                                if p > last {
                                    last = p;
                                    let mut cb = on_progress2.lock().unwrap();
                                    (cb)(p);
                                }
                            }
                        }
                    }
                }
            }
        });
        let status = child.wait().map_err(|e| format!("curl wait: {e}"))?;
        let _ = stderr_handle.join();
        if cancel.load(Ordering::SeqCst) {
            let _ = std::fs::remove_file(dest);
            return Err("cancelled".into());
        }
        if !status.success() {
            let _ = std::fs::remove_file(dest);
            return Err(format!("curl download failed: {status}"));
        }
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

        for chunk in msg.chunks_exact(64) {
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
    let q = url.splitn(2, '?').nth(1)?;
    for pair in q.split('&') {
        let mut it = pair.splitn(2, '=');
        if it.next() == Some(key) {
            return it.next().map(|v| urlencoding_decode(v));
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
    on_progress: Arc<Mutex<Box<dyn FnMut(u8) + Send>>>,
    /// Log callback.
    on_log: Arc<Mutex<Box<dyn FnMut(String) + Send>>>,
}

impl Downloader {
    pub fn new(
        cancel: Arc<AtomicBool>,
        on_progress: Arc<Mutex<Box<dyn FnMut(u8) + Send>>>,
        on_log: Arc<Mutex<Box<dyn FnMut(String) + Send>>>,
    ) -> Self {
        Self {
            http: HttpClient::new(),
            ffmpeg: "ffmpeg".into(),
            cancel,
            on_progress,
            on_log,
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

    /// Download a single video (or all parts of an anthology) into `out_dir`.
    /// `quality` is the maximum height (360/480/720/1080).
    /// Returns the list of downloaded media files in order.
    pub fn download(
        &self,
        url: &str,
        out_dir: &Path,
        quality: u32,
    ) -> Result<Vec<PathBuf>, String> {
        std::fs::create_dir_all(out_dir).map_err(|e| format!("create dir: {e}"))?;
        if url.contains("bilibili.com") {
            self.bilibili_download(url, out_dir, quality)
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
    ) -> Result<Vec<PathBuf>, String> {
        self.log(format!("[downloader] downloading {url}"));
        let meta = self.bilibili_meta(url)?;

        // If a specific ?p= is requested, keep only that page.
        let part_id = query_param(url, "p").and_then(|v| v.parse::<u32>().ok());
        let selected: Vec<(u32, u64, String)> = if let Some(pid) = part_id {
            meta.pages
                .iter()
                .filter(|(p, _, _)| *p == pid)
                .cloned()
                .collect()
        } else {
            meta.pages.clone()
        };
        if selected.is_empty() {
            return Err(format!("no video page p{} found", part_id.unwrap_or(0)));
        }
        let multiple = selected.len() > 1;
        self.log(format!("URL yields {} video(s)", selected.len()));

        let mut files = Vec::new();
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
            let dest = out_dir.join(filename);
            self.bilibili_download_playinfo(&play_info, &dest, url, quality)?;
            files.push(dest);
        }
        Ok(files)
    }

    /// Resolve a Bilibili URL into bvid/title/pages. Prefers the unsigned
    /// `x/web-interface/view` API, which keeps working when the HTML page is
    /// risk-blocked (HTTP 412 / code -352); falls back to scraping
    /// `__INITIAL_STATE__` from the page (also covers festival pages).
    fn bilibili_meta(&self, url: &str) -> Result<BiliMeta, String> {
        if let Some((id, is_bvid)) = bilibili_id_from_url(url) {
            let q = if is_bvid {
                format!("bvid={}", urlencode(&id))
            } else {
                format!("aid={}", urlencode(id.trim_start_matches("av")))
            };
            if let Ok(v) = self.http.get_json(
                &format!("https://api.bilibili.com/x/web-interface/view?{q}"),
                &[("Referer", "https://www.bilibili.com/")],
            ) {
                let code = v.get("code").and_then(Value::as_i64).unwrap_or(-1);
                if code == 0 {
                    if let Some(d) = v.get("data") {
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
        }

        // Fallback: scrape the HTML page (festival pages etc.).
        let html = self
            .http
            .get_text(url, &[("Referer", "https://www.bilibili.com/")])?;
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
            if let Some((key, ts)) = guard.as_ref() {
                if ts.elapsed() < Duration::from_secs(30) {
                    return Ok(key.clone());
                }
            }
        }
        let nav = self.http.get_json(
            "https://api.bilibili.com/x/web-interface/nav",
            &[("Referer", "https://www.bilibili.com/")],
        )?;
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

    /// Fetch the playurl (DASH formats) for one cid, WBI-signed and carrying the danmaku fingerprint params, exactly as yt-dlp's `_download_playinfo` does for an unauthenticated session (`try_look=1`, `fnval=4048`).
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
            ("try_look".to_string(), "1".to_string()),
            ("wts".to_string(), wts.to_string()),
        ]
        .into_iter()
        .collect();
        // yt-dlp only drops `try_look` when logged in; unauthenticated keeps it.
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
        let raw = self
            .http
            .get_json(&url, &[("Referer", "https://www.bilibili.com/")])?;
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
    ) -> Result<(), String> {
        let dash = play_info
            .get("dash")
            .ok_or("bilibili: no dash formats in playurl")?;
        let videos = dash
            .get("video")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let audios = dash
            .get("audio")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

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
            .ok_or("bilibili: video url missing")?;
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
        let headers = [("Referer", referer)];

        // Video (50% of progress), audio (next 30%), merge (last 20%).
        let video_progress = Arc::new(Mutex::new(Box::new({
            let on_progress = self.on_progress.clone();
            move |p: u8| {
                let mut f = on_progress.lock().unwrap();
                (f)(p / 2);
            }
        }) as Box<dyn FnMut(u8) + Send>));
        self.http.download_file(
            video_url,
            &tmp_video,
            &headers,
            &self.cancel,
            video_progress,
        )?;

        if let Some(aurl) = audio_url {
            let audio_progress = Arc::new(Mutex::new(Box::new({
                let on_progress = self.on_progress.clone();
                move |p: u8| {
                    let mut f = on_progress.lock().unwrap();
                    (f)(50 + p * 3 / 10);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            self.http
                .download_file(&aurl, &tmp_audio, &headers, &self.cancel, audio_progress)?;
        }

        self.progress(80);
        // Merge with ffmpeg
        let mut cmd = std::process::Command::new(&self.ffmpeg);
        cmd.arg("-y").arg("-i").arg(&tmp_video);
        if tmp_audio.exists() {
            cmd.arg("-i").arg(&tmp_audio);
        }
        cmd.arg("-c")
            .arg("copy")
            .arg("-movflags")
            .arg("+faststart")
            .arg(dest);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("ffmpeg: {e}"))?;
        let _ = std::fs::remove_file(&tmp_video);
        let _ = std::fs::remove_file(&tmp_audio);
        if !out.status.success() {
            let _ = std::fs::remove_file(dest);
            return Err(format!(
                "ffmpeg merge failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        self.progress(100);
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
        let mut formats = Vec::new();
        if let Some(f) = streaming.get("formats").and_then(Value::as_array) {
            formats.extend(f.iter().cloned());
        }
        if let Some(f) = streaming.get("adaptiveFormats").and_then(Value::as_array) {
            formats.extend(f.iter().cloned());
        }

        // Prefer progressive (has both audio+video) at requested quality.
        let progressive = formats
            .iter()
            .filter(|f| {
                f.get("audioQuality").is_some()
                    && f.get("height")
                        .and_then(Value::as_u64)
                        .map(|h| h <= quality as u64)
                        .unwrap_or(false)
            })
            .max_by_key(|f| f.get("height").and_then(Value::as_u64).unwrap_or(0));

        if let Some(fmt) = progressive {
            let url = fmt
                .get("url")
                .and_then(Value::as_str)
                .ok_or("youtube: format url missing")?;
            let progress = Arc::new(Mutex::new(Box::new({
                let on_progress = self.on_progress.clone();
                move |p: u8| {
                    let mut f = on_progress.lock().unwrap();
                    (f)(p);
                }
            }) as Box<dyn FnMut(u8) + Send>));
            self.http
                .download_file(url, &dest, &[], &self.cancel, progress)?;
            return Ok(vec![dest]);
        }

        // Fallback: best video + best audio (adaptive), then merge.
        let videos: Vec<Value> = formats
            .iter()
            .filter(|f| {
                f.get("mimeType")
                    .and_then(Value::as_str)
                    .map(|m| m.starts_with("video/"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
        let audios: Vec<Value> = formats
            .iter()
            .filter(|f| {
                f.get("mimeType")
                    .and_then(Value::as_str)
                    .map(|m| m.starts_with("audio/"))
                    .unwrap_or(false)
            })
            .cloned()
            .collect();
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
            .ok_or("youtube: no suitable video format")?;
        let best_audio = audios
            .iter()
            .max_by_key(|a| a.get("bitrate").and_then(Value::as_u64).unwrap_or(0))
            .cloned()
            .ok_or("youtube: no audio format")?;

        let video_url = best_video
            .get("url")
            .and_then(Value::as_str)
            .ok_or("youtube: video url missing")?;
        let audio_url = best_audio
            .get("url")
            .and_then(Value::as_str)
            .ok_or("youtube: audio url missing")?;

        let tmp_video = dest.with_extension("video.m4v");
        let tmp_audio = dest.with_extension("audio.m4a");
        let video_progress = Arc::new(Mutex::new(Box::new({
            let on_progress = self.on_progress.clone();
            move |p: u8| {
                let mut f = on_progress.lock().unwrap();
                (f)(p / 2);
            }
        }) as Box<dyn FnMut(u8) + Send>));
        self.http
            .download_file(video_url, &tmp_video, &[], &self.cancel, video_progress)?;
        let audio_progress = Arc::new(Mutex::new(Box::new({
            let on_progress = self.on_progress.clone();
            move |p: u8| {
                let mut f = on_progress.lock().unwrap();
                (f)(50 + p * 3 / 10);
            }
        }) as Box<dyn FnMut(u8) + Send>));
        self.http
            .download_file(audio_url, &tmp_audio, &[], &self.cancel, audio_progress)?;
        self.progress(80);

        let mut cmd = std::process::Command::new(&self.ffmpeg);
        cmd.arg("-y")
            .arg("-i")
            .arg(&tmp_video)
            .arg("-i")
            .arg(&tmp_audio)
            .arg("-c")
            .arg("copy")
            .arg("-movflags")
            .arg("+faststart")
            .arg(&dest);
        crate::pipeline::hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("ffmpeg: {e}"))?;
        let _ = std::fs::remove_file(&tmp_video);
        let _ = std::fs::remove_file(&tmp_audio);
        if !out.status.success() {
            let _ = std::fs::remove_file(&dest);
            return Err(format!(
                "ffmpeg merge failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        self.progress(100);
        Ok(vec![dest])
    }

    /// YouTube Innertube player response (no login, no PO token).
    /// Mirrors `YoutubeIE._extract_player_responses` with `visionos` + `web` fallback.
    fn youtube_player_response(&self, video_id: &str) -> Result<Value, String> {
        // Try the lightweight clients first (they do not require a PO token).
        let clients = [
            (
                "visionos",
                serde_json::json!({
                    "clientName": "VISIONOS",
                    "clientVersion": "1.02",
                    "deviceMake": "Apple",
                    "deviceModel": "RealityDevice17,1",
                    "osName": "visionOS",
                    "osVersion": "26.5.23O471",
                    "hl": "en",
                    "timeZone": "UTC",
                    "utcOffsetMinutes": 0,
                }),
            ),
            (
                "web",
                serde_json::json!({
                    "clientName": "WEB",
                    "clientVersion": "2.20260708.00.00",
                    "hl": "en",
                    "timeZone": "UTC",
                    "utcOffsetMinutes": 0,
                }),
            ),
        ];
        let mut last_err = String::new();
        for (name, client) in &clients {
            let body = serde_json::json!({
                "videoId": video_id,
                "context": { "client": client },
            });
            match self.http.post_json(
                &format!("https://www.youtube.com/youtubei/v1/player?prettyPrint=false"),
                &body,
                &[
                    ("Origin", "https://www.youtube.com"),
                    ("Referer", "https://www.youtube.com/"),
                ],
            ) {
                Ok(pr) => {
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
                    last_err = format!("{name} client: playability {status} ({reason})");
                }
                Err(e) => last_err = format!("{name} client: {e}"),
            }
        }
        Err(format!("youtube player response failed: {last_err}"))
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Resolved Bilibili video metadata (from the view API or the HTML page).
struct BiliMeta {
    bvid: String,
    title: String,
    /// (page number, cid, part title)
    pages: Vec<(u32, u64, String)>,
}

/// Extract the Bilibili video id from a URL. Returns `(id, is_bvid)`.
fn bilibili_id_from_url(url: &str) -> Option<(String, bool)> {
    if let Some(rest) = query_param(url, "bvid") {
        if !rest.is_empty() {
            return Some((rest, true));
        }
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
            rest.split(|c| c == '?' || c == '&' || c == '/')
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
                rest.split(|c| c == '?' || c == '&' || c == '/')
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
        )
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
        let files = d.download(&url, &dir, 720).unwrap();
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
