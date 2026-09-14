use crate::assets::Assets;
use crate::config::AppConfig;
use crate::model_ipc::{ModelServer, log_line};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};

/// Prevent a console window from flashing up when spawning subprocesses from a GUI app.
#[cfg(windows)]
pub(crate) fn hide_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x08000000);
}

#[cfg(not(windows))]
pub(crate) fn hide_console(_cmd: &mut Command) {}

/// Resolve once the cancel flag is set (polled so it can race an in-flight async request via `tokio::select!`).
async fn cancel_signal(cancel: &Arc<AtomicBool>) {
    loop {
        if cancel.load(Ordering::SeqCst) {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ItemStatus {
    Queued,
    Running,
    Done,
    Error,
    Cancelled,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct QueueItem {
    pub id: String,
    pub title: String,
    pub url: Option<String>,
    pub local_path: Option<String>,
    pub status: ItemStatus,
    /// Per-stage progress 0..100, index 0..3 (stages 1..4).
    pub stage_progress: Vec<u8>,
    /// 1-based part currently being processed (1 when a single video).
    pub current_part: u32,
    /// Total parts for a multi-part URL (1 otherwise).
    pub total_parts: u32,
    pub error: Option<String>,
    /// Selected anthology pages (1-based, yt-dlp --playlist-items
    /// equivalent): None = all parts; Some with >1 page = the user chose to
    /// merge those parts into one video before analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<Vec<u32>>,
}

impl QueueItem {
    pub fn from_url(id: String, title: String, url: String) -> Self {
        Self {
            id,
            title,
            url: Some(url),
            local_path: None,
            status: ItemStatus::Queued,
            stage_progress: vec![0; 4],
            current_part: 1,
            total_parts: 1,
            error: None,
            parts: None,
        }
    }
    pub fn from_file(id: String, path: String) -> Self {
        let title = Path::new(&path)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| path.clone());
        Self {
            id,
            title,
            url: None,
            local_path: Some(path),
            status: ItemStatus::Queued,
            stage_progress: vec![0; 4],
            current_part: 1,
            total_parts: 1,
            error: None,
            parts: None,
        }
    }
}

#[derive(Clone)]
pub struct PipelineHandle {
    pub cancel: Arc<AtomicBool>,
    pub child: Arc<Mutex<Option<Child>>>,
    /// PIDs of resident model server subprocesses so stop can kill them.
    pub model_pids: Arc<Mutex<Vec<u32>>>,
    /// PIDs of concurrent ffmpeg subprocesses (audio extract/resample + video decode) so stop can kill them.
    pub ffmpeg_pids: Arc<Mutex<Vec<u32>>>,
    /// Optional log file that all pipeline log lines are appended to.
    pub log_file: Arc<Mutex<Option<std::fs::File>>>,
}

impl Default for PipelineHandle {
    fn default() -> Self {
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            child: Arc::new(Mutex::new(None)),
            model_pids: Arc::new(Mutex::new(Vec::new())),
            ffmpeg_pids: Arc::new(Mutex::new(Vec::new())),
            log_file: Arc::new(Mutex::new(None)),
        }
    }
}

pub struct Runner {
    pub app: AppHandle,
    pub assets: Assets,
    pub handle: PipelineHandle,
    /// Root for user data (results/, work/, spk/). Resolved from the configurable
    /// data dir; defaults to the app-data dir.
    pub data_root: PathBuf,
    current_stage: u8,
    /// 1-based part currently being processed (1 for single videos).
    current_part: u32,
    /// Total parts for a multi-part URL (1 otherwise).
    total_parts: u32,
    /// Resident audio model server (VAD/ASR/SPK), launched once per pipeline.
    pub audio_server: Option<ModelServer>,
    /// Resident visual model server (VideoNeko ViT), launched once per pipeline.
    pub visual_server: Option<ModelServer>,
    /// Visual model input size (height, width), read from the model config.
    pub visual_size: Option<(u32, u32)>,
    /// Optional browser cookie import for downloads (yt-dlp
    /// --cookies-from-browser equivalent): "" | "firefox" | "chrome" | "edge".
    cookie_browser: String,
    /// Cookies loaded once per run from the selected browser; None until the
    /// first load (the value is Some(empty) when the browser yielded nothing).
    browser_cookies: Arc<Mutex<Option<Arc<Vec<crate::cookies::Cookie>>>>>,
}

/// Serialize the audio worker configuration (ASR backend, VAD/SPK model
/// directories and the optional speaker reference) for `audio_server.py`.
fn build_audio_config(
    config: &AppConfig,
    refs: Option<(&Path, &str, &[crate::silero_vad::Segment])>,
) -> serde_json::Value {
    let asr = if config.asr_is_api() {
        serde_json::json!({
            "backend": "qwen3-api",
            "baseUrl": config.qwen3_base_url.trim(),
            "apiKey": config.qwen3_api_key.trim(),
            "model": config.qwen3_model.trim(),
            "language": config.qwen3_language.trim(),
        })
    } else {
        serde_json::json!({
            "backend": "local",
            "type": config.asr_type,
            "dir": config.asr_model_dir.trim(),
            "language": config.asr_language.trim(),
        })
    };
    let spk = if config.spk_enabled && !config.spk_model_dir.trim().is_empty() {
        serde_json::json!({ "dir": config.spk_model_dir.trim() })
    } else {
        serde_json::Value::Null
    };
    let refs_json = match refs {
        Some((wav, name, segments)) => serde_json::json!({
            "file": wav.display().to_string(),
            "name": name,
            // Speech segments of the reference wav, pre-computed by the native
            // Silero VAD — the worker no longer runs a VAD model itself.
            "segments": segments
                .iter()
                .map(|s| [s.start_ms, s.end_ms])
                .collect::<Vec<_>>(),
        }),
        None => serde_json::Value::Null,
    };
    serde_json::json!({
        "asr": asr,
        "spk": spk,
        "ref": refs_json,
        // Max simultaneously processed ASR batches in the worker (1 = sequential).
        "asrConcurrency": config.asr_concurrency,
    })
}

impl Runner {
    pub fn new(
        app: AppHandle,
        assets: Assets,
        handle: PipelineHandle,
        cookie_browser: String,
        data_root: PathBuf,
    ) -> Self {
        Self {
            app,
            assets,
            handle,
            data_root,
            current_stage: 0,
            current_part: 1,
            total_parts: 1,
            audio_server: None,
            visual_server: None,
            visual_size: None,
            cookie_browser,
            browser_cookies: Arc::new(Mutex::new(None)),
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.handle.cancel.load(Ordering::SeqCst)
    }

    /// Launch the resident model servers (models load once here, reused for
    /// every queued video). Returns an error if a required server fails.
    pub fn start_model_servers(&mut self, config: &AppConfig) -> Result<(), String> {
        let cancel = self.handle.cancel.clone();
        let pids = self.handle.model_pids.clone();
        let python = crate::commands::PYTHON_CMD.to_string();

        // Build the audio worker configuration (ASR backend + SPK path) and
        // hand it to the resident worker as a JSON file. VAD is native Rust
        // (bundled Silero ONNX, see silero_vad.rs); the user provides the
        // FunASR ASR/SPK models. Resolve the speaker reference only when a
        // SPK model is actually in use; without one, speaker identification
        // stays disabled even if a speaker name/reference was saved earlier.
        let spk_active = config.spk_enabled && !config.spk_model_dir.trim().is_empty();
        let refs = if spk_active {
            match self.speaker_reference(config)? {
                Some((wav, name)) => {
                    let mut vad = crate::silero_vad::SileroVad::new(&self.assets.silero_model)?;
                    let segments = crate::silero_vad::detect_wav_segments(
                        &mut vad,
                        &wav,
                        &self.handle.cancel,
                    )?;
                    if segments.is_empty() {
                        return Err(format!(
                            "speaker reference {} contains no detectable speech — choose a clearer clip in Settings",
                            wav.display()
                        ));
                    }
                    self.emit_log(
                        "pipeline",
                        format!(
                            "[model] reference voiceprint: {} segment(s) from {}",
                            segments.len(),
                            wav.display()
                        ),
                    );
                    Some((wav, name, segments))
                }
                None => None,
            }
        } else {
            None
        };
        let audio_config = build_audio_config(
            config,
            refs.as_ref()
                .map(|(w, n, segs)| (w.as_path(), n.as_str(), segs.as_slice())),
        );
        let config_dir = self.data_root.join("work");
        let _ = std::fs::create_dir_all(&config_dir);
        let config_path = config_dir.join("audio_server.json");
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&audio_config).map_err(|e| e.to_string())?,
        )
        .map_err(|e| format!("write audio model config: {e}"))?;

        match &refs {
            Some((_, speaker_name, _)) => {
                self.emit_log(
                    "pipeline",
                    format!("[model] speaker identification: {speaker_name}"),
                );
            }
            None => {
                self.emit_log(
                    "pipeline",
                    "[model] no speaker configured: transcripts will not tag a specific speaker"
                        .to_string(),
                );
            }
        }
        if config.asr_is_api() {
            self.emit_log(
                "pipeline",
                format!("[model] ASR backend: Qwen3-ASR API ({})", config.qwen3_model),
            );
        } else {
            self.emit_log(
                "pipeline",
                format!(
                    "[model] ASR model: {} ({})",
                    config.asr_type, config.asr_model_dir
                ),
            );
        }
        if !config.spk_enabled || config.spk_model_dir.trim().is_empty() {
            self.emit_log(
                "pipeline",
                "[model] no SPK model configured: speaker identification is disabled".to_string(),
            );
        }
        let audio_args = vec!["--config".to_string(), config_path.display().to_string()];

        let audio_script = self.assets.scripts_dir.join("audio_server.py");
        self.emit_log(
            "pipeline",
            "[model] loading audio models (ASR/SPK); VAD is the bundled native Silero (CPU)...".to_string(),
        );
        let audio = ModelServer::spawn(
            &self.app,
            &python,
            &audio_script,
            &audio_args,
            cancel,
            pids,
            self.handle.log_file.clone(),
        )?;
        self.emit_log("pipeline", "[model] audio server ready".to_string());
        self.audio_server = Some(audio);

        if !config.videoneko_model_dir.is_empty() {
            let cancel2 = self.handle.cancel.clone();
            let pids2 = self.handle.model_pids.clone();
            let visual_script = self.assets.scripts_dir.join("visual_server.py");
            let visual_args = vec![
                "--model-dir".to_string(),
                config.videoneko_model_dir.clone(),
            ];
            self.emit_log(
                "pipeline",
                "[model] loading visual model (VideoNeko)...".to_string(),
            );
            let visual = ModelServer::spawn(
                &self.app,
                &python,
                &visual_script,
                &visual_args,
                cancel2,
                pids2,
                self.handle.log_file.clone(),
            )?;
            self.visual_size = Some(read_model_image_size(Path::new(
                &config.videoneko_model_dir,
            ))?);
            self.emit_log("pipeline", "[model] visual server ready".to_string());
            self.visual_server = Some(visual);
        }

        Ok(())
    }

    /// Shut down the resident model servers (send shutdown + kill).
    pub fn stop_model_servers(&mut self) {
        if let Some(mut s) = self.audio_server.take() {
            s.shutdown();
        }
        if let Some(mut s) = self.visual_server.take() {
            s.shutdown();
        }
    }

    /// Resolve the configured speaker reference: `(reference WAV path, display name)` when a speaker is configured and its imported WAV exists, `None` when no speaker is configured. Errors when a speaker is configured but the reference WAV went missing (the user should re-save it in Settings).
    fn speaker_reference(&self, config: &AppConfig) -> Result<Option<(PathBuf, String)>, String> {
        let name = config.speaker_name.trim();
        if name.is_empty() {
            return Ok(None);
        }
        let dir = self.data_root.join("spk");
        let wav = dir.join(config.speaker_ref.trim());
        if config.speaker_ref.trim().is_empty() || !wav.is_file() {
            return Err(format!(
                "speaker \"{name}\" is configured but its reference WAV is missing — open Settings and choose it again"
            ));
        }
        Ok(Some((wav, name.to_string())))
    }

    /// Run, per part, in parallel: audio thread: ffmpeg audio decode with the optional noise-reduction filters (video -> filtered_16k_wav, progress 0..20) -> native Silero VAD (filtered_16k_wav -> utterance segments, CPU) -> Python ASR/SPK -> raw utterances visual thread: ffmpeg GPU decode (video -> frames_raw RGB blob) -> predict (frames_raw -> raw per-second labels) The model workers only run inference; Rust extracts, filters, VADs, and writes the per-part asr.txt/visual.txt files from the returned raw results.
    pub fn run_audio_visual(
        &mut self,
        item_id: &str,
        filtered_wav: &Path,
        video: &Path,
        frames_raw: &Path,
        asr_txt: &Path,
        visual_txt: &Path,
        nr_filter: Option<&str>,
    ) -> Result<(), String> {
        let process_req = serde_json::json!({
            "cmd": "process",
            "id": item_id,
            "input": filtered_wav.display().to_string(),
        });
        let visual_req = serde_json::json!({
            "cmd": "predict",
            "id": item_id,
            "input": frames_raw.display().to_string(),
        });

        let (v_h, v_w) = self.visual_size.ok_or("visual model size not set")?;

        let app = self.app.clone();
        let id = item_id.to_string();
        let cancel = self.handle.cancel.clone();
        let ffmpeg_pids = self.handle.ffmpeg_pids.clone();
        let log_file = self.handle.log_file.clone();
        let part = self.current_part;
        let total_parts = self.total_parts;
        let silero_model = self.assets.silero_model.clone();
        let duration_secs = self.video_duration(video).ok();

        let audio_server = self
            .audio_server
            .as_mut()
            .ok_or("audio model server is not running")?;
        let visual_server = self
            .visual_server
            .as_mut()
            .ok_or("visual model server is not running")?;
        // Run the audio chain (decode+NR -> VAD -> ASR) and the visual chain (decode -> predict) concurrently.
        let (audio_res, visual_res) = std::thread::scope(|s| {
            let a_app = app.clone();
            let a_id = id.clone();
            let a_pids = ffmpeg_pids.clone();
            let a_log = log_file.clone();
            let a_cancel = cancel.clone();
            let a_duration = duration_secs;
            let a_nr_filter = nr_filter.map(|f| f.to_string());
            let a_silero = silero_model.clone();
            let mut process_req = process_req;
            let a = s.spawn(move || {
                // Stage 2: ffmpeg audio decode (+ optional noise-reduction
                // filters) into the 16 kHz mono WAV, progress 0..20.
                {
                    let app2 = a_app.clone();
                    let id2 = a_id.clone();
                    // Emit only when the percentage changes (0..20 => <=21 events): each event is a full JSON round-trip to the webview and flooding it crashed the app.
                    let last = Arc::new(Mutex::new(None::<u8>));
                    let progress_ctx = (app2, id2, last);
                    let mut on_progress = move |out_time_us: u64| {
                        let Some(dur) = a_duration else {
                            return;
                        };
                        if dur == 0 {
                            return;
                        }
                        let p = (((out_time_us as f64 / 1_000_000.0) / dur as f64) * 20.0) as u8;
                        let p = p.min(20);
                        let (app2, id2, last) = &progress_ctx;
                        let mut last = last.lock().unwrap();
                        if *last == Some(p) {
                            return;
                        }
                        *last = Some(p);
                        let _ = app2.emit(
                            "pipeline://stage",
                            serde_json::json!({
                                "itemId": id2,
                                "stage": 2,
                                "progress": p,
                                "part": part,
                                "totalParts": total_parts,
                            }),
                        );
                    };
                    run_ffmpeg(
                        &a_app,
                        &a_pids,
                        &a_log,
                        &a_cancel,
                        &a_id,
                        &ffmpeg_extract_args(video, filtered_wav, a_nr_filter.as_deref()),
                        Some(&mut on_progress),
                    )?;
                }
                // Stage 2: native Silero VAD on the extracted audio (CPU; a few
                // ms of model load and well under a second of inference per
                // minute of audio, so it needs no progress range of its own).
                let mut vad = crate::silero_vad::SileroVad::new(&a_silero)?;
                let segments = crate::silero_vad::detect_wav_segments(&mut vad, filtered_wav, &a_cancel)?;
                crate::model_ipc::log_line(
                    &a_app,
                    &a_log,
                    &a_id,
                    &format!("[audio] VAD: {} utterance(s)", segments.len()),
                );
                process_req["segments"] = serde_json::json!(
                    segments
                        .iter()
                        .map(|s| [s.start_ms, s.end_ms])
                        .collect::<Vec<_>>()
                );
                // Stage 2: ASR/SPK (progress 20..100, emitted by audio_server).
                audio_server.request(&a_app, &a_id, 2, part, total_parts, process_req)
            });
            let v_pids = ffmpeg_pids.clone();
            let v_log = log_file.clone();
            let v = s.spawn(move || {
                if cancel.load(Ordering::SeqCst) {
                    return Err("cancelled".to_string());
                }
                // Stage 3: hardware-decode video -> 1 fps RGB frames (ffmpeg).
                let _ = app.emit(
                    "pipeline://stage",
                    serde_json::json!({
                        "itemId": id, "stage": 3, "progress": 0,
                        "part": part, "totalParts": total_parts,
                    }),
                );
                run_ffmpeg(
                    &app,
                    &v_pids,
                    &v_log,
                    &cancel,
                    &id,
                    &ffmpeg_decode_args(video, frames_raw, v_h, v_w),
                    None,
                )?;
                let _ = app.emit(
                    "pipeline://stage",
                    serde_json::json!({
                        "itemId": id, "stage": 3, "progress": 20,
                        "part": part, "totalParts": total_parts,
                    }),
                );
                // Stage 3: classify the decoded frames (progress 20..100).
                visual_server.request(&app, &id, 3, part, total_parts, visual_req)
            });
            let ar = a
                .join()
                .unwrap_or_else(|_| Err("audio thread panicked".to_string()));
            let vr = v
                .join()
                .unwrap_or_else(|_| Err("visual thread panicked".to_string()));
            (ar, vr)
        });

        let audio_ev = audio_res?;
        let visual_ev = visual_res?;

        let a_ok = audio_ev
            .get("ok")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        if !a_ok {
            let e = audio_ev
                .get("error")
                .and_then(|x| x.as_str())
                .unwrap_or("audio failed");
            return Err(format!("transcription failed: {e}"));
        }
        let v_ok = visual_ev
            .get("ok")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        if !v_ok {
            let e = visual_ev
                .get("error")
                .and_then(|x| x.as_str())
                .unwrap_or("visual failed");
            return Err(format!("visual processing failed: {e}"));
        }

        // Post-process the raw results in Rust and write the per-part files.
        let utterances: Vec<serde_json::Value> = audio_ev
            .get("utterances")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        std::fs::write(asr_txt, format_asr(&utterances)).map_err(|e| format!("write asr: {e}"))?;

        let preds: Vec<String> = visual_ev
            .get("preds")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();
        std::fs::write(visual_txt, format_visual(&preds))
            .map_err(|e| format!("write visual: {e}"))?;

        Ok(())
    }

    fn emit_log(&self, item_id: &str, line: String) {
        log_line(&self.app, &self.handle.log_file, item_id, &line);
    }

    fn emit_stage(&self, item_id: &str, stage: u8, progress: u8) {
        let _ = self.app.emit(
            "pipeline://stage",
            serde_json::json!({
                "itemId": item_id,
                "stage": stage,
                "progress": progress,
                "part": self.current_part,
                "totalParts": self.total_parts,
            }),
        );
    }

    /// Load the configured browser's cookies once per run (cached). Enabled by
    /// the "cookies from browser" setting; errors hard like yt-dlp's
    /// CookieLoadError when the selected browser cannot be read.
    fn load_cookies_once(&self) -> Result<Option<Arc<Vec<crate::cookies::Cookie>>>, String> {
        if self.cookie_browser.is_empty() {
            return Ok(None);
        }
        let mut guard = self.browser_cookies.lock().unwrap();
        if let Some(cached) = guard.as_ref() {
            return Ok(Some(cached.clone()));
        }
        let log_ref = &self;
        let cookies = crate::cookies::load_browser_cookies(&self.cookie_browser, &|msg| {
            log_ref.emit_log("downloader", format!("[cookies] {msg}"));
        })?;
        self.emit_log(
            "downloader",
            format!(
                "[cookies] using cookies from {} ({} cookie(s))",
                self.cookie_browser,
                cookies.len()
            ),
        );
        let arc = Arc::new(cookies);
        *guard = Some(arc.clone());
        Ok(Some(arc))
    }

    /// Build a downloader for this item, wiring progress and logs to the UI.
    fn make_downloader(
        &self,
        item_id: &str,
        cookies: Option<Arc<Vec<crate::cookies::Cookie>>>,
    ) -> crate::downloader::Downloader {
        let cancel = self.handle.cancel.clone();
        let stage = self.current_stage;
        let app = self.app.clone();
        let item_id_owned = item_id.to_string();
        let on_progress = Arc::new(Mutex::new(Box::new(move |p: u8| {
            let _ = app.emit(
                "pipeline://stage",
                serde_json::json!({
                    "itemId": item_id_owned,
                    "stage": stage,
                    "progress": p.min(100),
                    "part": 1,
                    "totalParts": 1,
                }),
            );
        }) as Box<dyn FnMut(u8) + Send>));
        let app2 = self.app.clone();
        let item_id2 = item_id.to_string();
        let log_file2 = self.handle.log_file.clone();
        let on_log = Arc::new(Mutex::new(Box::new(move |line: String| {
            log_line(&app2, &log_file2, &item_id2, &line);
        }) as Box<dyn FnMut(String) + Send>));
        crate::downloader::Downloader::new(cancel, on_progress, on_log, cookies)
    }

    fn run_ytdlp(
        &self,
        item_id: &str,
        url: &str,
        out_dir: &Path,
        quality: u32,
        parts: Option<&[u32]>,
    ) -> Result<bool, String> {
        self.emit_log(item_id, format!("[downloader] downloading {url}"));
        let cookies = self.load_cookies_once()?;
        let dl = self.make_downloader(item_id, cookies);
        let files = dl.download(url, out_dir, quality, parts)?;
        self.emit_log(
            item_id,
            format!("[downloader] download complete ({} file(s))", files.len()),
        );
        Ok(true)
    }

    /// List the video titles a URL yields, one per line (used to detect multi-video / multi-part pages).
    fn ytdlp_list_titles(&self, item_id: &str, url: &str) -> Result<Vec<String>, String> {
        self.emit_log(item_id, format!("[downloader] listing titles: {url}"));
        let cookies = self.load_cookies_once()?;
        let dl = self.make_downloader(item_id, cookies);
        let titles = dl.probe_titles(url)?;
        Ok(titles
            .into_iter()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect())
    }

    /// Download the selected videos (or ALL videos when `parts` is None) from
    /// a URL into out_dir, preserving playlist order in the filenames and
    /// merging each part's video/audio streams into a single file.
    fn ytdlp_download_playlist(
        &self,
        item_id: &str,
        url: &str,
        out_dir: &Path,
        quality: u32,
        parts: Option<&[u32]>,
    ) -> Result<bool, String> {
        let what = match parts {
            Some(ps) => format!("{} selected video(s)", ps.len()),
            None => "all videos".to_string(),
        };
        self.emit_log(item_id, format!("[downloader] downloading {what}: {url}"));
        let cookies = self.load_cookies_once()?;
        let dl = self.make_downloader(item_id, cookies);
        let files = dl.download(url, out_dir, quality, parts)?;
        self.emit_log(
            item_id,
            format!("[downloader] download complete ({} file(s))", files.len()),
        );
        Ok(true)
    }

    /// Concatenate downloaded multi-part files into one video with the
    /// downloader's stream-copy merge (ffmpeg concat demuxer).
    fn merge_videos(&self, item_id: &str, files: &[PathBuf], dest: &Path) -> Result<(), String> {
        self.emit_log(
            item_id,
            format!(
                "[downloader] merging {} part(s) into {}",
                files.len(),
                dest.display()
            ),
        );
        let cookies = self.load_cookies_once().ok().flatten();
        let dl = self.make_downloader(item_id, cookies);
        dl.merge_videos(files, dest)
    }

    /// Duration of a media file in whole seconds (via ffprobe).
    fn video_duration(&self, video: &Path) -> Result<u64, String> {
        let mut cmd = std::process::Command::new("ffprobe");
        cmd.args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            &video.display().to_string(),
        ]);
        hide_console(&mut cmd);
        let out = cmd.output().map_err(|e| format!("ffprobe: {e}"))?;
        if !out.status.success() {
            return Err(format!("ffprobe failed for {}", video.display()));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        text.trim()
            .parse::<f64>()
            .map(|s| s as u64)
            .map_err(|e| format!("bad duration '{text}': {e}"))
    }

    /// Run the stage-5 summarization for one video. Reusable so a stored asr.txt + visual.txt pair can be re-summarized with a different engine.
    /// The OpenAI-compatible request is made in-process via openai-rust2.
    pub fn summarize(
        &self,
        config: &AppConfig,
        item_id: &str,
        title: &str,
        visual_txt: &Path,
        asr_txt: &Path,
        output_md: &Path,
    ) -> Result<(), String> {
        // Custom prompt (user-edited in Settings) takes precedence over the bundled prompt.md.
        let prompt = if config.custom_prompt.trim().is_empty() {
            std::fs::read_to_string(&self.assets.prompt_md)
                .map_err(|e| format!("read prompt: {e}"))?
        } else {
            config.custom_prompt.clone()
        };
        let visual =
            std::fs::read_to_string(visual_txt).map_err(|e| format!("read visual: {e}"))?;
        let asr = std::fs::read_to_string(asr_txt).map_err(|e| format!("read asr: {e}"))?;

        self.emit_log(
            item_id,
            format!("[llm] engine={} title={title}", config.engine),
        );
        let (base_url, api_key, model, max_tokens, temperature, top_p, thinking) =
            match config.engine.as_str() {
                "ollama" => (
                    config.ollama_base_url.trim_end_matches('/').to_string(),
                    "ollama".to_string(),
                    config.ollama_model.clone(),
                    config.api_max_tokens,
                    config.api_temperature,
                    config.api_top_p,
                    config.ollama_thinking,
                ),
                "llamacpp" => (
                    config.llamacpp_base_url.trim_end_matches('/').to_string(),
                    "llamacpp".to_string(),
                    config.llamacpp_model.clone(),
                    config.api_max_tokens,
                    config.api_temperature,
                    config.api_top_p,
                    config.llamacpp_thinking,
                ),
                _ => (
                    config.api_base_url.trim_end_matches('/').to_string(),
                    config.api_key.clone(),
                    config.api_model.clone(),
                    config.api_max_tokens,
                    config.api_temperature,
                    config.api_top_p,
                    config.api_thinking,
                ),
            };
        if base_url.is_empty() || model.is_empty() {
            return Err(format!("{} engine is not configured", config.engine));
        }

        // openai-rust2 is async; we run on a small current-thread runtime here.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| format!("runtime: {e}"))?;
        let is_ollama = config.engine == "ollama";
        let cancel = self.handle.cancel.clone();
        let user_content =
            format!("video title: {title}\n\nvisual.txt:\n{visual}\n\nasr.txt:\n{asr}");
        let result = rt.block_on(async move {
            tokio::select! {
                res = llm_request(
                    &base_url, &api_key, &model, max_tokens, temperature, top_p,
                    thinking, is_ollama, &prompt, &user_content,
                ) => res,
                _ = cancel_signal(&cancel) => Err("cancelled".to_string()),
            }
        });
        let (content, reasoning) = result?;
        // Fold the separate reasoning_content (DeepSeek-style thinking) in as well.
        let mut combined = content;
        if !reasoning.trim().is_empty() {
            combined = format!("<think>{reasoning}</think>\n{combined}");
        }
        let (thinking, summary) = split_think_summary(&combined);
        if summary.trim().is_empty() {
            self.emit_log(item_id, "[llm] warning: empty response".to_string());
        }

        std::fs::create_dir_all(output_md.parent().unwrap_or_else(|| Path::new(".")))
            .map_err(|e| e.to_string())?;
        std::fs::write(output_md, summary).map_err(|e| format!("write summary: {e}"))?;
        if !thinking.trim().is_empty() {
            let think_path = output_md.parent().unwrap().join("thinking.txt");
            std::fs::write(&think_path, thinking).map_err(|e| format!("write thinking: {e}"))?;
        }
        self.emit_log(
            item_id,
            format!("Summary written to {}", output_md.display()),
        );
        Ok(())
    }
}

/// Build the chat completions URL from an OpenAI-compatible base URL. The base may or may not include a `/v1` (or full path) suffix, so normalize:
/// - "https://api.deepseek.com"          -> "https://api.deepseek.com/v1/chat/completions"
/// - "http://localhost:11434/v1"         -> "http://localhost:11434/v1/chat/completions"
/// - "https://host:8080/v1/completions"  -> "https://host:8080/v1/chat/completions"
pub fn chat_completions_url(base: &str) -> String {
    let mut b = base.trim_end_matches('/').to_string();
    // strip a trailing "/completions" or "/chat/completions" segment
    if b.ends_with("/chat/completions") {
        b.truncate(b.len() - "/chat/completions".len());
    } else if b.ends_with("/completions") {
        b.truncate(b.len() - "/completions".len());
    }
    if b.ends_with("/v1") {
        format!("{b}/chat/completions")
    } else {
        format!("{b}/v1/chat/completions")
    }
}

/// Build the Ollama native chat URL. Ollama serves /api/chat on its root
/// (e.g. http://localhost:11434/api/chat); the configured base may include a trailing "/v1" from the OpenAI-compat convention, which is stripped.
fn ollama_chat_url(base: &str) -> String {
    let mut b = base.trim_end_matches('/').to_string();
    if b.ends_with("/v1") {
        b.truncate(b.len() - "/v1".len());
    }
    format!("{}/api/chat", b.trim_end_matches('/'))
}

/// Perform the actual LLM summarization request (Ollama native or an OpenAI-compatible endpoint). Returns (content, reasoning).
async fn llm_request(
    base_url: &str,
    api_key: &str,
    model: &str,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    thinking: bool,
    is_ollama: bool,
    prompt: &str,
    user_content: &str,
) -> Result<(String, String), String> {
    let client = openai_rust2::Client::shared_client();
    let messages = serde_json::json!([
        { "role": "system", "content": prompt },
        { "role": "user", "content": user_content },
    ]);

    if is_ollama {
        // Ollama: use the NATIVE /api/chat endpoint. The OpenAI-compat /v1/chat/completions path cannot disable thinking, so qwen3.x reasoning models fill `reasoning` and leave `content` empty.
        let url = ollama_chat_url(base_url);
        let body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "think": thinking,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        });
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Ollama request failed: {e}"))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| format!("Ollama read body failed: {e}"))?;
        if !status.is_success() {
            return Err(format!("Ollama API error (status {status}): {text}"));
        }
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| format!("Ollama parse response failed: {e}"))?;
        let content = json["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        // native /api/chat names it "thinking"; some servers use "reasoning"
        let reasoning = json["message"]["thinking"]
            .as_str()
            .or_else(|| json["message"]["reasoning"].as_str())
            .unwrap_or("")
            .to_string();
        return Ok((content, reasoning));
    }

    let mut body = serde_json::json!({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "stream": false,
        "max_tokens": max_tokens,
    });
    // Reasoning models (DeepSeek etc.) return their answer in a separate `reasoning_content` field and may leave `content` empty, so when thinking is disabled we force it off; otherwise leave it enabled and read the response as raw JSON to keep both fields.
    if !thinking {
        body["thinking"] = serde_json::json!({ "type": "disabled" });
    }
    let url = chat_completions_url(base_url);
    let resp = client
        .post(&url)
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("LLM request failed: {e}"))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| format!("LLM read body failed: {e}"))?;
    if !status.is_success() {
        return Err(format!("LLM API error (status {status}): {text}"));
    }
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("LLM parse response failed: {e}"))?;
    let msg = &json["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("").to_string();
    Ok((content, reasoning))
}

/// Split the model output into (thinking, summary). Handles explicit <think>...</think> blocks and prose reasoning before the first timestamped entry ("[HH:MM:SS - HH:MM:SS]").
fn split_think_summary(text: &str) -> (String, String) {
    let mut thinking = String::new();
    let mut summary = text.trim().to_string();
    let start = text.find("<think>");
    let end = text.find("</think>");
    if let (Some(s), Some(e)) = (start, end)
        && e > s
    {
        thinking = text[s + "<think>".len()..e].trim().to_string();
        summary = (text[..s].to_string() + &text[e + "</think>".len()..])
            .trim()
            .to_string();
    }
    if thinking.is_empty() {
        // reasoning often appears as prose before the first timestamped line
        if let Some(idx) = summary
            .lines()
            .position(|l| l.trim_start().starts_with("[") && l.contains("]"))
            && idx > 0
        {
            let all: Vec<&str> = summary.lines().collect();
            thinking = all[..idx].join("\n").trim().to_string();
            summary = all[idx..].join("\n").trim().to_string();
        }
    }
    (thinking, summary)
}

/// Download the videos a URL yields (all of them, or the selected anthology
/// pages) into `dl_dir` (the results/<title>/ dir) and return the ordered
/// media files. When more than one part was selected (merge choice), the parts
/// are concatenated into a single <title>.mp4 up front and only that file is
/// returned. Otherwise multi-part pages are NOT merged — each part is returned
/// in p0N order for separate analysis; the final asr/visual are merged with
/// timestamp offsets by the caller.
fn download_from_url(
    runner: &Runner,
    item_id: &str,
    url: &str,
    dl_dir: &Path,
    quality: u32,
    title: &str,
    parts: Option<&[u32]>,
) -> Result<Vec<PathBuf>, String> {
    // Probe how many videos the URL yields ("title*1 p01 title*2" per part).
    let titles = runner.ytdlp_list_titles(item_id, url)?;
    runner.emit_log(item_id, format!("URL yields {} video(s)", titles.len()));

    // Effective video count: the selection when the user picked pages (empty
    // selection = all), else everything the URL yields.
    let count = parts
        .filter(|p| !p.is_empty())
        .map_or(titles.len(), <[u32]>::len);
    if count <= 1 {
        // single video: plain download
        runner.run_ytdlp(item_id, url, dl_dir, quality, parts)?;
    } else {
        // multi-part / playlist: download the (selected) parts (no merge)
        runner.ytdlp_download_playlist(item_id, url, dl_dir, quality, parts)?;
    }

    // collect media files, sorted by name so p01 < p02 < ... (playlist_index prefix is zero-padded in the -o template)
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dl_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            let is_media = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|ext| {
                    matches!(
                        ext.to_lowercase().as_str(),
                        "mp4" | "mkv" | "webm" | "mov" | "avi" | "flv"
                    )
                });
            if p.is_file() && is_media {
                files.push(p);
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("yt-dlp finished but no media file found in download dir".to_string());
    }

    let merged_name = if title.trim().is_empty() {
        "video".to_string()
    } else {
        sanitize_filename(title)
    };
    if files.len() == 1 {
        // rename the single file to a clean <title>.mp4 inside the title dir
        let src = &files[0];
        let dest = dl_dir.join(format!("{merged_name}.mp4"));
        if src != &dest {
            let _ = std::fs::copy(src, &dest);
            if !dest.exists() {
                let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("mp4");
                let dest2 = dl_dir.join(format!("{merged_name}.{ext}"));
                let _ = std::fs::copy(src, &dest2);
                if dest2.exists() {
                    return Ok(vec![dest2]);
                }
            }
        }
        if dest.exists() {
            return Ok(vec![dest]);
        }
        return Ok(vec![src.clone()]);
    }

    // user chose to merge the selected parts: concatenate them in playlist
    // order into one file and analyze that single video
    if parts.is_some_and(|p| p.len() > 1) {
        let dest = dl_dir.join(format!("{merged_name}.mp4"));
        runner.merge_videos(item_id, &files, &dest)?;
        return Ok(vec![dest]);
    }

    // multi-part: keep every part in order (p01 < p02 < ...)
    runner.emit_log(
        item_id,
        format!("{} parts kept unmerged, in order", files.len()),
    );
    Ok(files)
}

/// Reduce a raw video title string to the shared prefix: drop any leading "NNN_" playlist index, cut at the first " p0N " marker and strip a trailing "[video id]".
pub fn simplify_title_str(raw: &str) -> String {
    // Strip only a leading "NNN_" playlist index (digits immediately followed by an underscore), not arbitrary leading digits, so titles are preserved.
    let trimmed = {
        let b = raw.as_bytes();
        let mut n = 0;
        while n < b.len() && b[n].is_ascii_digit() {
            n += 1;
        }
        if n > 0 && n < b.len() && b[n] == b'_' {
            &raw[n + 1..]
        } else {
            raw
        }
    }
    .trim_start_matches('_');
    let trimmed = if let Some(pos) = trimmed.find(" p0") {
        &trimmed[..pos]
    } else {
        trimmed
    }
    .trim_end();
    if let Some(open) = trimmed.rfind(" [")
        && trimmed.ends_with(']')
        && trimmed[open..].len() > 2
    {
        return trimmed[..open].trim_end().to_string();
    }
    trimmed.to_string()
}

/// Probe the video title(s) a URL yields. Used to show the real title in the queue immediately when a URL is added.
/// Titles are fetched in-process via `crate::downloader`. `cookie_browser`
/// optionally enables reading that browser's cookies for the probe.
pub fn probe_ytdlp_titles(url: &str, cookie_browser: &str) -> Result<Vec<String>, String> {
    let cancel = Arc::new(AtomicBool::new(false));
    let on_progress = Arc::new(Mutex::new(Box::new(|_: u8| {}) as Box<dyn FnMut(u8) + Send>));
    let on_log = Arc::new(Mutex::new(
        Box::new(|_: String| {}) as Box<dyn FnMut(String) + Send>
    ));
    let cookies = if cookie_browser.trim().is_empty() {
        None
    } else {
        Some(Arc::new(crate::cookies::load_browser_cookies(
            cookie_browser,
            &|_: &str| {},
        )?))
    };
    let dl = crate::downloader::Downloader::new(cancel, on_progress, on_log, cookies);
    let titles = dl.probe_titles(url)?;
    Ok(titles
        .into_iter()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
}

/// Structured probe for the multi-part picker: the URL's main title plus its
/// parts as `(1-based page, part title)` in playlist order.
pub fn probe_url_parts(
    url: &str,
    cookie_browser: &str,
) -> Result<(String, Vec<(u32, String)>), String> {
    let cancel = Arc::new(AtomicBool::new(false));
    let on_progress = Arc::new(Mutex::new(Box::new(|_: u8| {}) as Box<dyn FnMut(u8) + Send>));
    let on_log = Arc::new(Mutex::new(
        Box::new(|_: String| {}) as Box<dyn FnMut(String) + Send>
    ));
    let cookies = if cookie_browser.trim().is_empty() {
        None
    } else {
        Some(Arc::new(crate::cookies::load_browser_cookies(
            cookie_browser,
            &|_: &str| {},
        )?))
    };
    let dl = crate::downloader::Downloader::new(cancel, on_progress, on_log, cookies);
    dl.probe_parts(url)
}

/// Make a string safe to use as a Windows file name.
fn sanitize_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '_',
            c => c,
        })
        .collect()
}

fn parse_hms(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return None;
    }
    let h: u64 = parts[0].trim().parse().ok()?;
    let m: u64 = parts[1].trim().parse().ok()?;
    let sec: u64 = parts[2].trim().parse().ok()?;
    Some(h * 3600 + m * 60 + sec)
}

fn fmt_hms(total: u64) -> String {
    format!(
        "{:02}:{:02}:{:02}",
        total / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

/// Minimum utterance duration (seconds) kept in asr.txt (was UTT_MIN_S in the old Python audio_server).
const UTT_MIN_S: f64 = 1.5;
/// Majority-vote smoothing window (seconds) for visual predictions (was SMOOTH_WINDOW in the old Python visual_server).
const SMOOTH_WINDOW: usize = 15;

/// Build the ffmpeg noise-reduction audio filter chain from the config, e.g.
/// `highpass=f=80,lowpass=f=14000,afftdn=nr=6:nf=-50`. `None` when noise
/// reduction is disabled (plain 16 kHz decode).
pub fn build_nr_filter(config: &AppConfig) -> Option<String> {
    if !config.nr_enabled {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if config.highpass_hz > 0 {
        parts.push(format!("highpass=f={}", config.highpass_hz));
    }
    if config.lowpass_hz > 0 {
        parts.push(format!("lowpass=f={}", config.lowpass_hz));
    }
    parts.push(format!(
        "afftdn=nr={}:nf={}",
        config.afftdn_nr, config.afftdn_nf
    ));
    Some(parts.join(","))
}

/// ffmpeg args to decode the audio of `video` into a 16 kHz mono PCM WAV,
/// optionally applying the noise-reduction filter chain, and reporting
/// machine-readable progress on stdout (`-progress pipe:1`).
fn ffmpeg_extract_args(video: &Path, output: &Path, nr_filter: Option<&str>) -> Vec<String> {
    let mut args = vec![
        "-y".to_string(),
        "-i".to_string(),
        video.display().to_string(),
        "-vn".to_string(),
    ];
    if let Some(filter) = nr_filter {
        args.push("-af".to_string());
        args.push(filter.to_string());
    }
    args.extend([
        "-acodec".to_string(),
        "pcm_s16le".to_string(),
        "-ar".to_string(),
        "16000".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        output.display().to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
        "-progress".to_string(),
        "pipe:1".to_string(),
    ]);
    args
}

/// ffmpeg args to hardware-decode `video`, sample 1 fps, scale to `width`x`height` and write a single raw RGB24 blob to `output`.
fn ffmpeg_decode_args(video: &Path, output: &Path, height: u32, width: u32) -> Vec<String> {
    vec![
        "-y".to_string(),
        "-hwaccel".to_string(),
        "auto".to_string(),
        "-i".to_string(),
        video.display().to_string(),
        "-vf".to_string(),
        format!("fps=1,scale={width}:{height}"),
        "-f".to_string(),
        "rawvideo".to_string(),
        "-pix_fmt".to_string(),
        "rgb24".to_string(),
        output.display().to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
    ]
}

/// Read the (height, width) input size the fine-tuned ViT expects, from its `preprocessor_config.json` (handles both `shortest_edge` and height/width).
fn read_model_image_size(model_dir: &Path) -> Result<(u32, u32), String> {
    let cfg = model_dir.join("preprocessor_config.json");
    let text =
        std::fs::read_to_string(&cfg).map_err(|e| format!("read preprocessor config: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse preprocessor config: {e}"))?;
    let size = json
        .get("size")
        .ok_or_else(|| "preprocessor config has no 'size'".to_string())?;
    if let Some(se) = size.get("shortest_edge").and_then(|v| v.as_u64()) {
        return Ok((se as u32, se as u32));
    }
    let h = size.get("height").and_then(|v| v.as_u64()).unwrap_or(224) as u32;
    let w = size.get("width").and_then(|v| v.as_u64()).unwrap_or(224) as u32;
    Ok((h, w))
}

/// Run an ffmpeg subprocess with the given args, forwarding stderr to the UI. Registers the child PID so `stop_pipeline` can kill it; safe to call from multiple threads concurrently (the audio and visual threads each run their own ffmpeg). `on_progress` (when given) receives ffmpeg's `out_time` in microseconds from `-progress pipe:1` output — only pass it for commands that include that flag.
fn run_ffmpeg(
    app: &AppHandle,
    pids: &Arc<Mutex<Vec<u32>>>,
    log_file: &Arc<Mutex<Option<std::fs::File>>>,
    cancel: &Arc<AtomicBool>,
    item_id: &str,
    args: &[String],
    on_progress: Option<&mut (dyn FnMut(u64) + Send)>,
) -> Result<(), String> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(args);
    hide_console(&mut cmd);
    cmd.stderr(Stdio::piped());
    if on_progress.is_some() {
        cmd.stdout(Stdio::piped());
    } else {
        cmd.stdout(Stdio::null());
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn ffmpeg: {e}"))?;
    let stderr = child.stderr.take().expect("stderr piped");
    let stdout = child.stdout.take();
    let pid = child.id();
    pids.lock().unwrap().push(pid);

    std::thread::scope(|s| -> Result<(), String> {
        let app2 = app.clone();
        let id2 = item_id.to_string();
        let log_file2 = log_file.clone();
        s.spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for l in reader.lines().map_while(Result::ok) {
                log_line(&app2, &log_file2, &id2, &l);
            }
        });
        if let (Some(out), Some(cb)) = (stdout, on_progress) {
            s.spawn(move || {
                use std::io::BufRead;
                let reader = std::io::BufReader::new(out);
                for line in reader.lines().map_while(Result::ok) {
                    // `-progress pipe:1` emits key=value lines; both out_time
                    // keys carry microseconds (out_time_ms is a misnomer).
                    if let Some(us) = line
                        .strip_prefix("out_time_ms=")
                        .or_else(|| line.strip_prefix("out_time_us="))
                        .and_then(|v| v.trim().parse::<u64>().ok())
                    {
                        cb(us);
                    }
                }
            });
        }
        let status = child.wait().map_err(|e| format!("wait ffmpeg: {e}"))?;
        pids.lock().unwrap().retain(|p| *p != pid);
        if cancel.load(Ordering::SeqCst) {
            return Err("cancelled".to_string());
        }
        if !status.success() {
            return Err(format!("ffmpeg failed (exit {:?})", status.code()));
        }
        Ok(())
    })
}

fn sensevoice_regex() -> &'static regex::Regex {
    use std::sync::OnceLock;
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    RE.get_or_init(|| {
        regex::Regex::new(r"<\|[^|]+\|><\|([^|]+)\|><\|([^|]+)\|><\|[^|]+\|>([^<]*)")
            .expect("valid sensevoice regex")
    })
}

/// Parse a SenseVoice tagged utterance into (emotion, text). Mirrors the old Python `parse_utterance` in audio_server.py.
fn parse_sensevoice(raw: &str) -> (String, String) {
    let re = sensevoice_regex();
    let mut emotion = String::new();
    let mut parts: Vec<String> = Vec::new();
    let mut any = false;
    for caps in re.captures_iter(raw) {
        if !any {
            emotion = caps[1].replace("EMO_", "");
            any = true;
        }
        let event = &caps[2];
        let text = caps[3].trim();
        if text.is_empty() {
            parts.push(format!("[{event}]"));
        } else {
            parts.push(text.to_string());
        }
    }
    if !any {
        return ("UNKNOWN".to_string(), raw.trim().to_string());
    }
    (emotion, parts.join(" "))
}

/// Majority label of a window, ties broken by first occurrence.
fn majority(window: &[String]) -> String {
    use std::collections::HashMap;
    let mut counts: HashMap<&str, usize> = HashMap::new();
    let mut first: HashMap<&str, usize> = HashMap::new();
    for (i, s) in window.iter().enumerate() {
        *counts.entry(s.as_str()).or_insert(0) += 1;
        first.entry(s.as_str()).or_insert(i);
    }
    let mut best: Option<(&str, usize, usize)> = None;
    for (s, c) in counts.iter() {
        let fi = first[*s];
        let better = match best {
            None => true,
            Some((_, bc, bfi)) => *c > bc || (*c == bc && fi < bfi),
        };
        if better {
            best = Some((*s, *c, fi));
        }
    }
    best.map(|(s, _, _)| s.to_string()).unwrap_or_default()
}

/// Sliding-window majority smoothing (moves the old Python `smooth` to Rust).
fn smooth(preds: &[String], window: usize) -> Vec<String> {
    if window <= 1 {
        return preds.to_vec();
    }
    let half = window / 2;
    (0..preds.len())
        .map(|i| {
            let lo = i.saturating_sub(half);
            let hi = (i + half + 1).min(preds.len());
            majority(&preds[lo..hi])
        })
        .collect()
}

/// Group consecutive equal predictions into inclusive (start, end) intervals.
fn to_intervals(preds: &[String]) -> Vec<(usize, usize, &str)> {
    let mut out = Vec::new();
    if preds.is_empty() {
        return out;
    }
    let mut start = 0;
    for i in 1..preds.len() {
        if preds[i] != preds[start] {
            out.push((start, i - 1, preds[start].as_str()));
            start = i;
        }
    }
    out.push((start, preds.len() - 1, preds[start].as_str()));
    out
}

/// Format the raw utterances returned by the audio server into asr.txt content: drop too-short utterances, parse SenseVoice tags, sort, and emit "[HH:MM:SS-HH:MM:SS] [speaker] [emotion] text" lines.
fn format_asr(utterances: &[serde_json::Value]) -> String {
    let mut rows: Vec<(u64, u64, String, String, String)> = Vec::new();
    for u in utterances {
        let start_ms = u.get(0).and_then(|v| v.as_u64()).unwrap_or(0);
        let end_ms = u.get(1).and_then(|v| v.as_u64()).unwrap_or(0);
        let speaker = u
            .get(2)
            .and_then(|v| v.as_str())
            .unwrap_or("other")
            .to_string();
        let raw = u.get(3).and_then(|v| v.as_str()).unwrap_or("");
        if (end_ms.saturating_sub(start_ms)) as f64 / 1000.0 < UTT_MIN_S {
            continue;
        }
        let (emotion, text) = parse_sensevoice(raw);
        if text.is_empty() {
            continue;
        }
        rows.push((start_ms, end_ms, speaker, emotion, text));
    }
    rows.sort_by_key(|r| r.0);
    let mut out = String::new();
    for (start_ms, end_ms, speaker, emotion, text) in rows {
        out.push_str(&format!(
            "[{}-{}] [{}] [{}] {}\n",
            fmt_hms(start_ms / 1000),
            fmt_hms(end_ms.div_ceil(1000)),
            speaker,
            emotion,
            text,
        ));
    }
    out
}

/// Format raw per-second predictions into visual.txt content: smooth, merge into intervals, and emit "[HH:MM:SS-HH:MM:SS] label" lines.
fn format_visual(preds: &[String]) -> String {
    let smoothed = smooth(preds, SMOOTH_WINDOW);
    let mut out = String::new();
    for (start, end, label) in to_intervals(&smoothed) {
        out.push_str(&format!(
            "[{}-{}] {}\n",
            fmt_hms(start as u64),
            fmt_hms((end + 1) as u64),
            label,
        ));
    }
    out
}

/// Shift every "[hh:mm:ss-hh:mm:ss]" timestamp prefix in `content` by `offset` seconds (used to realign per-part asr/visual lines onto the full video timeline). Lines without a timestamp prefix are passed through.
fn offset_timestamps(content: &str, offset: u64) -> String {
    if offset == 0 {
        return content.to_string();
    }
    content
        .lines()
        .map(|line| {
            let t = line.trim_start();
            if let Some(rest) = t.strip_prefix('[')
                && let Some(end) = rest.find(']')
                && let Some((a, b)) = rest[..end].split_once('-')
                && let (Some(ta), Some(tb)) = (parse_hms(a.trim()), parse_hms(b.trim()))
            {
                return format!(
                    "[{}-{}]{}",
                    fmt_hms(ta + offset),
                    fmt_hms(tb + offset),
                    &rest[end + 1..]
                );
            }
            line.to_string()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Merge per-part asr/visual text (one entry per part, each with that part's  duration) into a single stream whose timestamps match the full video. Returns (merged_asr, merged_visual).
fn merge_part_outputs(parts: &[(PathBuf, PathBuf, u64)]) -> Result<(String, String), String> {
    let mut asr = String::new();
    let mut visual = String::new();
    let mut offset: u64 = 0;
    for (asr_path, visual_path, dur) in parts {
        let a = std::fs::read_to_string(asr_path).map_err(|e| format!("read asr part: {e}"))?;
        let v =
            std::fs::read_to_string(visual_path).map_err(|e| format!("read visual part: {e}"))?;
        asr.push_str(&offset_timestamps(&a, offset));
        if !asr.ends_with('\n') {
            asr.push('\n');
        }
        visual.push_str(&offset_timestamps(&v, offset));
        if !visual.ends_with('\n') {
            visual.push('\n');
        }
        offset += dur;
    }
    Ok((asr, visual))
}

/// Process one queue item through all 5 stages. Returns () on success, Err on failure (cancellation surfaces as the special "cancelled").
pub fn run_item(
    runner: &mut Runner,
    config: &AppConfig,
    item: &QueueItem,
    work_dir: &Path,
) -> Result<(), String> {
    let id = item.id.clone();

    // ---------- Stage 1: Video Input ----------
    runner.current_stage = 1;
    runner.emit_stage(&id, 1, 0);

    // Resolve the video title first, then create results/<title>/ to hold all of this video's outputs (and, for URL downloads, the video files).
    let results_dir = runner.data_root.join("results");
    std::fs::create_dir_all(&results_dir).map_err(|e| e.to_string())?;

    let (videos, video_title, title_dir) = if let Some(url) = &item.url {
        // Probe the titles up front so we know the dir name before downloading.
        let titles = runner.ytdlp_list_titles(&id, url)?;
        // Title for the results dir: a single selected part keeps its
        // "main p0N part" title (unsimplified) so separate per-part tasks do
        // not collide in results/<main title>/; everything else uses the
        // simplified main title.
        let title = match item.parts.as_deref() {
            Some([k]) => titles
                .get((*k as usize).saturating_sub(1))
                .cloned()
                .filter(|t| !t.trim().is_empty())
                .unwrap_or_else(|| {
                    titles
                        .first()
                        .map(|t| simplify_title_str(t))
                        .unwrap_or_else(|| "video".to_string())
                }),
            _ => titles
                .first()
                .map(|t| simplify_title_str(t))
                .filter(|t| !t.trim().is_empty())
                .unwrap_or_else(|| "video".to_string()),
        };
        let dir = sanitize_filename(&title);
        let tdir = results_dir.join(&dir);
        std::fs::create_dir_all(&tdir).map_err(|e| format!("create title dir: {e}"))?;
        let paths = download_from_url(
            runner,
            &id,
            url,
            &tdir,
            config.download_quality,
            &title,
            item.parts.as_deref(),
        )?;
        (paths, title, tdir)
    } else if let Some(p) = &item.local_path {
        let pb = PathBuf::from(p);
        if !pb.exists() {
            return Err(format!("local file not found: {p}"));
        }
        let title = pb
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| item.title.clone());
        let dir = sanitize_filename(&title);
        let tdir = results_dir.join(&dir);
        std::fs::create_dir_all(&tdir).map_err(|e| format!("create title dir: {e}"))?;
        (vec![pb], title, tdir)
    } else {
        return Err("no url or local path".to_string());
    };
    runner.current_part = 1;
    runner.total_parts = videos.len() as u32;
    runner.emit_log(
        &id,
        format!("Video(s): {} (title: {video_title})", videos.len()),
    );
    runner.emit_log(&id, format!("Results dir: {}", title_dir.display()));
    runner.emit_stage(&id, 1, 100);

    // ---------- Stages 2-3: ASR + visual, per part ----------
    // Multi-part inputs are analysed one by one (in p0N order) without merging the video files; afterwards the per-part asr/visual are merged and their timestamps realigned onto the full video timeline.
    let mut part_outputs: Vec<(PathBuf, PathBuf, u64)> = Vec::new();
    let n_parts = videos.len();
    for (idx, video) in videos.iter().enumerate() {
        runner.current_part = (idx + 1) as u32;
        let part_tag = if n_parts > 1 {
            format!("_p{}", idx + 1)
        } else {
            String::new()
        };
        runner.emit_log(&id, format!("[part {}] {}", idx + 1, video.display()));

        // Stages 2+3 run concurrently: the audio thread decodes the audio (with optional noise-reduction filters) and runs VAD+ASR; the visual thread decodes and predicts in parallel.
        runner.current_stage = 2;
        runner.emit_stage(&id, 2, 0);
        runner.emit_stage(&id, 3, 0);
        let filtered_wav = work_dir.join(format!("filtered{part_tag}.wav"));
        let frames_raw = work_dir.join(format!("frames{part_tag}.raw"));
        let asr_part = work_dir.join(format!("asr{part_tag}.txt"));
        let visual_part = work_dir.join(format!("visual{part_tag}.txt"));
        let nr_filter = build_nr_filter(config);
        if let Some(f) = &nr_filter {
            runner.emit_log(&id, format!("[audio] noise reduction: -af \"{f}\""));
        } else {
            runner.emit_log(&id, "[audio] noise reduction: disabled".to_string());
        }
        runner.run_audio_visual(
            &id,
            &filtered_wav,
            video,
            &frames_raw,
            &asr_part,
            &visual_part,
            nr_filter.as_deref(),
        )?;
        runner.emit_log(&id, format!("Filtered audio: {}", filtered_wav.display()));
        runner.emit_stage(&id, 2, 100);
        runner.emit_stage(&id, 3, 100);

        let dur = runner.video_duration(video)?;
        part_outputs.push((asr_part, visual_part, dur));
    }

    // Merge per-part asr/visual, realigning timestamps to the full video.
    let (merged_asr, merged_visual) = merge_part_outputs(&part_outputs)?;
    let asr_txt = work_dir.join("asr.txt");
    let visual_txt = work_dir.join("visual.txt");
    std::fs::write(&asr_txt, &merged_asr).map_err(|e| format!("write asr: {e}"))?;
    std::fs::write(&visual_txt, &merged_visual).map_err(|e| format!("write visual: {e}"))?;
    runner.emit_log(
        &id,
        "Merged per-part asr/visual with realigned timestamps".to_string(),
    );

    // ---------- Stage 4: Summarization ----------
    runner.current_stage = 4;
    runner.emit_stage(&id, 4, 0);
    let summary_md = work_dir.join("summary.md");
    if let Err(summary_err) = runner.summarize(
        config,
        &id,
        &video_title,
        &visual_txt,
        &asr_txt,
        &summary_md,
    ) {
        // Persist the ASR/visual transcripts and a summary.md noting the failure (or cancellation) so the result still shows up and can be re-summarized.
        let _ = std::fs::copy(&asr_txt, title_dir.join("asr.txt"));
        let _ = std::fs::copy(&visual_txt, title_dir.join("visual.txt"));
        let reason = if summary_err == "cancelled" {
            "# Summary canceled\n\nThe request to the LLM engine was canceled.\n\nThe ASR and visual transcripts were saved.\n".to_string()
        } else {
            format!(
                "# Summary failed\n\nReason: {summary_err}\n\nThe ASR and visual transcripts were saved.\nUse **Re-summarize** to try again.\n"
            )
        };
        let _ = std::fs::write(title_dir.join("summary.md"), reason);
        if summary_err == "cancelled" {
            runner.emit_log(&id, "Summary canceled by user".to_string());
        } else {
            runner.emit_log(
                &id,
                format!("Summary failed: {summary_err} (asr + visual saved for re-summary)"),
            );
        }
        // ASR + visual are saved; the per-item work dir is no longer needed.
        let _ = std::fs::remove_dir_all(work_dir);
        return Err(summary_err);
    }
    runner.emit_stage(&id, 4, 100);

    // Copy outputs into the results/<title>/ dir (fixed names: summary.md, asr.txt, visual.txt, thinking.txt).
    std::fs::copy(&summary_md, title_dir.join("summary.md"))
        .map_err(|e| format!("copy summary: {e}"))?;
    let _ = std::fs::copy(&asr_txt, title_dir.join("asr.txt"));
    let _ = std::fs::copy(&visual_txt, title_dir.join("visual.txt"));
    // companion thinking file (may not exist when the model didn't reason)
    let _ = std::fs::copy(
        work_dir.join("thinking.txt"),
        title_dir.join("thinking.txt"),
    );
    runner.emit_log(
        &id,
        format!("Saved summary + asr + visual in {}", title_dir.display()),
    );
    // Clear the per-item work dir (raw/filtered/16 kHz wavs, per-part text and the intermediate summary) now that results are persisted.
    let _ = std::fs::remove_dir_all(work_dir);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nr_filter_matches_expected_chain() {
        let mut cfg = AppConfig::new();
        assert_eq!(
            build_nr_filter(&cfg).as_deref(),
            Some("highpass=f=80,lowpass=f=14000,afftdn=nr=6:nf=-50")
        );
        cfg.nr_enabled = false;
        assert_eq!(build_nr_filter(&cfg), None);
    }

    #[test]
    fn nr_filter_skips_disabled_bandpass_parts() {
        let mut cfg = AppConfig::new();
        cfg.highpass_hz = 0;
        cfg.lowpass_hz = 0;
        cfg.afftdn_nr = 12.5;
        cfg.afftdn_nf = -45;
        assert_eq!(build_nr_filter(&cfg).as_deref(), Some("afftdn=nr=12.5:nf=-45"));
    }

    #[test]
    fn normalize_clamps_nr_params() {
        let mut cfg = AppConfig::new();
        cfg.highpass_hz = 50000;
        cfg.lowpass_hz = 1;
        cfg.afftdn_nr = 200.0;
        cfg.afftdn_nf = 0;
        cfg.normalize(std::path::Path::new("/tmp/liveneko-test"));
        assert_eq!(cfg.highpass_hz, 20000);
        assert_eq!(cfg.lowpass_hz, 0, "lowpass below highpass is dropped");
        assert_eq!(cfg.afftdn_nr, 97.0);
        assert_eq!(cfg.afftdn_nf, -20);
    }
}
