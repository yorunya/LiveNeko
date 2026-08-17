use crate::assets::Assets;
use crate::config::AppConfig;
use crate::model_ipc::{log_line, ModelServer};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

/// Prevent a console window from flashing up when spawning subprocesses from a GUI app.
#[cfg(windows)]
fn hide_console(cmd: &mut Command) {
    use std::os::windows::process::CommandExt;
    cmd.creation_flags(0x08000000);
}

#[cfg(not(windows))]
fn hide_console(_cmd: &mut Command) {}

/// Decode bytes captured from a child process stdout. Native console apps on Windows (e.g. yt-dlp) emit non-ASCII text using the system ANSI codepage (GBK on zh-CN) instead of UTF-8, which breaks UTF-8 line decoding. Try UTF-8 first, then fall back to GBK.
fn decode_console_text(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            let (text, _, _) = encoding_rs::GBK.decode(bytes);
            text.into_owned()
        }
    }
}

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
}

impl Runner {
    pub fn new(app: AppHandle, assets: Assets, handle: PipelineHandle) -> Self {
        Self {
            app,
            assets,
            handle,
            current_stage: 0,
            current_part: 1,
            total_parts: 1,
            audio_server: None,
            visual_server: None,
            visual_size: None,
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
        let python = config.python_cmd.clone();

        // Prepare 16 kHz speaker references. Rust owns every ffmpeg subprocess, so the audio worker never shells out to ffmpeg.
        let ref_media = self.assets.spk_refs();
        if ref_media.is_empty() {
            return Err("no speaker reference media found".to_string());
        }
        let app_data_dir = self.app.path().app_data_dir().map_err(|e| e.to_string())?;
        let refs_dir = app_data_dir.join("work").join("refs_16k");
        let _ = std::fs::remove_dir_all(&refs_dir);
        std::fs::create_dir_all(&refs_dir).map_err(|e| e.to_string())?;
        for (i, media) in ref_media.iter().enumerate() {
            let out = refs_dir.join(format!("ref_{i:03}.wav"));
            if !self.run_process(
                "pipeline",
                Path::new("ffmpeg"),
                &ffmpeg_resample_args(media, &out, 16000),
                None,
            )? {
                return Err(format!(
                    "failed to resample speaker reference {}",
                    media.display()
                ));
            }
        }

        let audio_script = self.assets.scripts_dir.join("audio_server.py");
        let audio_args = vec![
            "--model-dir".to_string(),
            self.assets.audio_model_dir.display().to_string(),
            "--ref-dir".to_string(),
            refs_dir.display().to_string(),
            "--filter-model-dir".to_string(),
            self.assets.filter_model_dir.display().to_string(),
        ];
        self.emit_log(
            "pipeline",
            "[model] loading audio models (filter/VAD/ASR/SPK)...".to_string(),
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

    /// Run, per part, in parallel: audio thread: ffmpeg extract 48 kHz (video -> raw_wav) -> process (raw_wav -> denoise + downsample + VAD/ASR/SPK -> raw utterances) visual thread: ffmpeg GPU decode (video -> frames_raw RGB blob) -> predict (frames_raw -> raw per-second labels)
    /// The model workers only run inference; Rust extracts and writes the per-part asr.txt/visual.txt files from the returned raw results.
    pub fn run_audio_visual(
        &mut self,
        item_id: &str,
        raw_wav: &Path,
        video: &Path,
        frames_raw: &Path,
        asr_txt: &Path,
        visual_txt: &Path,
    ) -> Result<(), String> {
        let process_req = serde_json::json!({
            "cmd": "process",
            "id": item_id,
            "input": raw_wav.display().to_string(),
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

        let audio_server = self
            .audio_server
            .as_mut()
            .ok_or("audio model server is not running")?;
        let visual_server = self
            .visual_server
            .as_mut()
            .ok_or("visual model server is not running")?;

        // Run the audio chain (extract -> process) and the visual chain (decode -> predict) concurrently.
        let (audio_res, visual_res) = std::thread::scope(|s| {
            let a_app = app.clone();
            let a_id = id.clone();
            let a_pids = ffmpeg_pids.clone();
            let a_log = log_file.clone();
            let a_cancel = cancel.clone();
            let a = s.spawn(move || {
                // Stage 2: extract raw 48 kHz audio.
                run_ffmpeg(
                    &a_app,
                    &a_pids,
                    &a_log,
                    &a_cancel,
                    &a_id,
                    &ffmpeg_extract_args(video, raw_wav),
                )?;
                // Stage 2: denoise + VAD/ASR/SPK (progress 0..20 denoise, 20..100 ASR).
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

    /// Spawn a process with piped stdout/stderr, forward both to the UI as log lines, parse "PROGRESS <n>" lines into stage progress, kill on cancellation, and wait for completion.
    fn run_process(
        &self,
        item_id: &str,
        program: &Path,
        args: &[String],
        cwd: Option<&Path>,
    ) -> Result<bool, String> {
        let mut cmd = Command::new(program);
        cmd.args(args);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        hide_console(&mut cmd);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        // Keep the child visible to the stop command.
        let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        *self.handle.child.lock().unwrap() = Some(child);

        let stderr_app = self.app.clone();
        let item_id3 = item_id.to_string();
        let log_file = self.handle.log_file.clone();
        let stderr_handle = std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line {
                    log_line(&stderr_app, &log_file, &item_id3, &l);
                }
            }
        });

        // Read stdout in this thread so we can map PROGRESS lines to the UI.
        use std::io::BufRead;
        let reader = std::io::BufReader::new(stdout);
        for line in reader.lines() {
            if let Ok(l) = line {
                let trimmed = l.trim();
                if let Some(rest) = trimmed.strip_prefix("PROGRESS ") {
                    if let Ok(p) = rest.parse::<u8>() {
                        self.emit_stage(item_id, self.current_stage, p.min(100));
                    }
                } else if !trimmed.is_empty() {
                    self.emit_log(item_id, l);
                }
            }
        }

        // Take the child back out of the shared handle to wait on it.
        let status = self
            .handle
            .child
            .lock()
            .unwrap()
            .take()
            .map(|mut c| c.wait())
            .unwrap_or_else(|| Err(std::io::Error::new(std::io::ErrorKind::Other, "no child")));
        let status = status.map_err(|e| format!("wait failed: {e}"))?;
        let _ = stderr_handle.join();

        let cancelled = self.is_cancelled();
        if cancelled {
            return Err("cancelled".to_string());
        }
        Ok(status.success())
    }

    /// Like run_process, but returns the collected stdout lines (for probing commands like `yt-dlp --print`).
    fn run_process_capture(
        &self,
        item_id: &str,
        program: &Path,
        args: &[String],
        cwd: Option<&Path>,
    ) -> Result<Vec<String>, String> {
        let mut cmd = Command::new(program);
        cmd.args(args);
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        hide_console(&mut cmd);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        *self.handle.child.lock().unwrap() = Some(child);

        let stderr_app = self.app.clone();
        let item_id3 = item_id.to_string();
        let log_file = self.handle.log_file.clone();
        let stderr_handle = std::thread::spawn(move || {
            use std::io::BufRead;
            let reader = std::io::BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(l) = line {
                    log_line(&stderr_app, &log_file, &item_id3, &l);
                }
            }
        });

        use std::io::BufRead;
        let mut reader = std::io::BufReader::new(stdout);
        let mut lines = Vec::new();
        let mut raw = Vec::new();
        loop {
            raw.clear();
            let n = reader
                .read_until(b'\n', &mut raw)
                .map_err(|e| format!("read stdout: {e}"))?;
            if n == 0 {
                break;
            }
            let line = decode_console_text(&raw);
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("PROGRESS ") {
                if let Ok(p) = rest.parse::<u8>() {
                    self.emit_stage(item_id, self.current_stage, p.min(100));
                }
            } else if !trimmed.is_empty() {
                lines.push(line);
            }
        }

        let status = self
            .handle
            .child
            .lock()
            .unwrap()
            .take()
            .map(|mut c| c.wait())
            .unwrap_or_else(|| Err(std::io::Error::new(std::io::ErrorKind::Other, "no child")));
        let status = status.map_err(|e| format!("wait failed: {e}"))?;
        let _ = stderr_handle.join();

        let cancelled = self.is_cancelled();
        if cancelled {
            return Err("cancelled".to_string());
        }
        if !status.success() {
            return Err(format!(
                "{} exited with code {:?}",
                program.display(),
                status.code()
            ));
        }
        Ok(lines)
    }

    fn run_ytdlp(
        &self,
        item_id: &str,
        url: &str,
        out_dir: &Path,
        quality: u32,
    ) -> Result<bool, String> {
        self.emit_log(item_id, format!("[yt-dlp] downloading {url}"));
        let mut args = vec![
            "--merge-output-format".to_string(),
            "mp4".to_string(),
            "-f".to_string(),
            quality_format(quality),
        ];
        args.push("-P".to_string());
        args.push(out_dir.display().to_string());
        args.push("-o".to_string());
        args.push("%(title).100B [%(id)s].%(ext)s".to_string());
        args.push(url.to_string());
        let r = self.run_process(item_id, &self.assets.yt_dlp_exe, &args, None);
        match &r {
            Ok(true) => self.emit_log(item_id, "[yt-dlp] download complete".to_string()),
            Ok(false) => self.emit_log(item_id, "[yt-dlp] download failed".to_string()),
            Err(_) => {}
        }
        r
    }

    /// List the video titles a URL yields, one per line (used to detect multi-video / multi-part pages). Runs `yt-dlp --print %(title)s`.
    fn ytdlp_list_titles(&self, item_id: &str, url: &str) -> Result<Vec<String>, String> {
        self.emit_log(item_id, format!("[yt-dlp] listing titles: {url}"));
        let args = vec![
            "--print".to_string(),
            "%(title)s".to_string(),
            url.to_string(),
        ];
        let lines = self.run_process_capture(item_id, &self.assets.yt_dlp_exe, &args, None)?;
        Ok(lines
            .iter()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect())
    }

    /// Download ALL videos from a URL (`--yes-playlist`) into out_dir, preserving playlist order in the filenames and merging each part's video/audio streams into a single file.
    fn ytdlp_download_playlist(
        &self,
        item_id: &str,
        url: &str,
        out_dir: &Path,
        quality: u32,
    ) -> Result<bool, String> {
        self.emit_log(item_id, format!("[yt-dlp] downloading all videos: {url}"));
        let mut args = vec![
            "--yes-playlist".to_string(),
            "--merge-output-format".to_string(),
            "mp4".to_string(),
            "-f".to_string(),
            quality_format(quality),
        ];
        args.push("-P".to_string());
        args.push(out_dir.display().to_string());
        args.push("-o".to_string());
        args.push("%(playlist_index)03d_%(title).100B [%(id)s].%(ext)s".to_string());
        args.push(url.to_string());
        let r = self.run_process(item_id, &self.assets.yt_dlp_exe, &args, None);
        match &r {
            Ok(true) => self.emit_log(item_id, "[yt-dlp] download complete".to_string()),
            Ok(false) => self.emit_log(item_id, "[yt-dlp] download failed".to_string()),
            Err(_) => {}
        }
        r
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
    if let (Some(s), Some(e)) = (start, end) {
        if e > s {
            thinking = text[s + "<think>".len()..e].trim().to_string();
            summary = (text[..s].to_string() + &text[e + "</think>".len()..])
                .trim()
                .to_string();
        }
    }
    if thinking.is_empty() {
        // reasoning often appears as prose before the first timestamped line
        if let Some(idx) = summary
            .lines()
            .position(|l| l.trim_start().starts_with("[") && l.contains("]"))
        {
            if idx > 0 {
                let all: Vec<&str> = summary.lines().collect();
                thinking = all[..idx].join("\n").trim().to_string();
                summary = all[idx..].join("\n").trim().to_string();
            }
        }
    }
    (thinking, summary)
}

/// Download all videos from a URL directly into `dl_dir` (the results/<title>/ dir) and return the ordered media files. Multi-part pages are NOT merged here —each part is returned in p0N order for separate analysis; the final asr/visual are merged with timestamp offsets by the caller.
fn download_from_url(
    runner: &Runner,
    item_id: &str,
    url: &str,
    dl_dir: &Path,
    quality: u32,
    title: &str,
) -> Result<Vec<PathBuf>, String> {
    // Probe how many videos the URL yields ("title*1 p01 title*2" per part).
    let titles = runner.ytdlp_list_titles(item_id, url)?;
    runner.emit_log(item_id, format!("URL yields {} video(s)", titles.len()));

    if titles.len() <= 1 {
        // single video: plain download
        runner.run_ytdlp(item_id, url, dl_dir, quality)?;
    } else {
        // multi-part / playlist: download all parts (no merge)
        runner.ytdlp_download_playlist(item_id, url, dl_dir, quality)?;
    }

    // collect media files, sorted by name so p01 < p02 < ... (playlist_index prefix is zero-padded in the -o template)
    let mut files: Vec<PathBuf> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dl_dir) {
        for entry in rd.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    if matches!(
                        ext.to_lowercase().as_str(),
                        "mp4" | "mkv" | "webm" | "mov" | "avi" | "flv"
                    ) {
                        files.push(p);
                    }
                }
            }
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("yt-dlp finished but no media file found in download dir".to_string());
    }

    if files.len() == 1 {
        // rename the single file to a clean <title>.mp4 inside the title dir
        let src = &files[0];
        let merged_name = if title.trim().is_empty() {
            "video".to_string()
        } else {
            sanitize_filename(title)
        };
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
    if let Some(open) = trimmed.rfind(" [") {
        if trimmed.ends_with(']') && trimmed[open..].len() > 2 {
            return trimmed[..open].trim_end().to_string();
        }
    }
    trimmed.to_string()
}

/// Probe the video title(s) a URL yields with `yt-dlp --print %(title)s`. Used to show the real title in the queue immediately when a URL is added.
pub fn probe_ytdlp_titles(yt_dlp_exe: &Path, url: &str) -> Result<Vec<String>, String> {
    let mut cmd = std::process::Command::new(yt_dlp_exe);
    cmd.args(["--print", "%(title)s", url]);
    hide_console(&mut cmd);
    let out = cmd.output().map_err(|e| format!("run yt-dlp: {e}"))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr);
        return Err(format!("yt-dlp failed: {}", msg.trim()));
    }
    let text = decode_console_text(&out.stdout);
    Ok(text
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect())
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

/// The yt-dlp `-f` selector for the requested download quality.
fn quality_format(quality: u32) -> String {
    let q = match quality {
        360 => 360,
        480 => 480,
        1080 => 1080,
        _ => 720,
    };
    format!("bestvideo[height<={q}]+bestaudio/best[height<={q}]")
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

/// ffmpeg args to resample `input` to `sr` Hz mono PCM into `output`.
fn ffmpeg_resample_args(input: &Path, output: &Path, sr: u32) -> Vec<String> {
    vec![
        "-y".to_string(),
        "-i".to_string(),
        input.display().to_string(),
        "-vn".to_string(),
        "-acodec".to_string(),
        "pcm_s16le".to_string(),
        "-ar".to_string(),
        sr.to_string(),
        "-ac".to_string(),
        "1".to_string(),
        output.display().to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
    ]
}

/// ffmpeg args to extract 48 kHz mono PCM audio from `video` into `output`.
fn ffmpeg_extract_args(video: &Path, output: &Path) -> Vec<String> {
    vec![
        "-y".to_string(),
        "-i".to_string(),
        video.display().to_string(),
        "-vn".to_string(),
        "-acodec".to_string(),
        "pcm_s16le".to_string(),
        "-ar".to_string(),
        "48000".to_string(),
        "-ac".to_string(),
        "1".to_string(),
        output.display().to_string(),
        "-loglevel".to_string(),
        "error".to_string(),
    ]
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

/// Run an ffmpeg subprocess with the given args, forwarding stderr to the UI. Registers the child PID so `stop_pipeline` can kill it; safe to call from multiple threads concurrently (the audio and visual threads each run their own ffmpeg).
fn run_ffmpeg(
    app: &AppHandle,
    pids: &Arc<Mutex<Vec<u32>>>,
    log_file: &Arc<Mutex<Option<std::fs::File>>>,
    cancel: &Arc<AtomicBool>,
    item_id: &str,
    args: &[String],
) -> Result<(), String> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(args);
    hide_console(&mut cmd);
    cmd.stdout(Stdio::null()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn ffmpeg: {e}"))?;
    let stderr = child.stderr.take().expect("stderr piped");
    let pid = child.id();
    pids.lock().unwrap().push(pid);

    let app2 = app.clone();
    let id2 = item_id.to_string();
    let log_file2 = log_file.clone();
    let stderr_thread = std::thread::spawn(move || {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(stderr);
        for line in reader.lines() {
            if let Ok(l) = line {
                log_line(&app2, &log_file2, &id2, &l);
            }
        }
    });

    let status = child.wait().map_err(|e| format!("wait ffmpeg: {e}"))?;
    pids.lock().unwrap().retain(|p| *p != pid);
    let _ = stderr_thread.join();
    if cancel.load(Ordering::SeqCst) {
        return Err("cancelled".to_string());
    }
    if !status.success() {
        return Err(format!("ffmpeg failed (exit {:?})", status.code()));
    }
    Ok(())
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
            fmt_hms((end_ms + 999) / 1000),
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
            if let Some(rest) = t.strip_prefix('[') {
                if let Some(end) = rest.find(']') {
                    let inner = &rest[..end];
                    if let Some((a, b)) = inner.split_once('-') {
                        if let (Some(ta), Some(tb)) = (parse_hms(a.trim()), parse_hms(b.trim())) {
                            return format!(
                                "[{}-{}]{}",
                                fmt_hms(ta + offset),
                                fmt_hms(tb + offset),
                                &rest[end + 1..]
                            );
                        }
                    }
                }
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
    let results_dir = runner
        .app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("results");
    std::fs::create_dir_all(&results_dir).map_err(|e| e.to_string())?;

    let (videos, video_title, title_dir) = if let Some(url) = &item.url {
        // Probe the titles up front so we know the dir name before downloading.
        let titles = runner.ytdlp_list_titles(&id, url)?;
        let title = titles
            .first()
            .map(|t| simplify_title_str(t))
            .filter(|t| !t.trim().is_empty())
            .unwrap_or_else(|| "video".to_string());
        let dir = sanitize_filename(&title);
        let tdir = results_dir.join(&dir);
        std::fs::create_dir_all(&tdir).map_err(|e| format!("create title dir: {e}"))?;
        let paths = download_from_url(runner, &id, url, &tdir, config.download_quality, &title)?;
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

        // Stages 2+3 run concurrently: the audio thread extracts raw 48 kHz audio and runs denoise+ASR; the visual thread decodes and predicts in parallel, so the ffmpeg extract overlaps the visual work.
        runner.current_stage = 2;
        runner.emit_stage(&id, 2, 0);
        runner.emit_stage(&id, 3, 0);
        let raw_wav = work_dir.join(format!("raw{part_tag}.wav"));
        let frames_raw = work_dir.join(format!("frames{part_tag}.raw"));
        let asr_part = work_dir.join(format!("asr{part_tag}.txt"));
        let visual_part = work_dir.join(format!("visual{part_tag}.txt"));
        runner.run_audio_visual(&id, &raw_wav, video, &frames_raw, &asr_part, &visual_part)?;
        runner.emit_log(&id, format!("Raw audio: {}", raw_wav.display()));
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
