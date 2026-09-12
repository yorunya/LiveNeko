use crate::assets::Assets;
use crate::config::AppConfig;
use crate::model_ipc::log_line;
use crate::pipeline::{self, hide_console, ItemStatus, PipelineHandle, QueueItem, Runner};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use tauri::{AppHandle, Emitter, Manager, State};
use uuid::Uuid;

pub struct AppState {
    pub config: Mutex<AppConfig>,
    pub queue: Mutex<Vec<QueueItem>>,
    pub pipeline: PipelineHandle,
    pub running: AtomicBool,
    /// Legacy-result migration is idempotent; run it once per process.
    pub migrated: AtomicBool,
    pub app_data_dir: PathBuf,
    /// OS appearance ("light" | "dark"), detected once at startup
    pub os_theme: String,
    pub os_accent: Option<String>,
}

impl AppState {
    pub fn new(app_data_dir: PathBuf, os_theme: String, os_accent: Option<String>) -> Self {
        let mut config = AppConfig::load(&app_data_dir);
        let data_root = config.data_root(&app_data_dir);
        if config.funasr_models_dir.trim().is_empty() {
            config.funasr_models_dir = data_root
                .join("funasr-models")
                .to_string_lossy()
                .to_string();
        }
        Self {
            config: Mutex::new(config),
            queue: Mutex::new(Vec::new()),
            pipeline: PipelineHandle::default(),
            running: AtomicBool::new(false),
            migrated: AtomicBool::new(false),
            app_data_dir,
            os_theme,
            os_accent,
        }
    }

    /// Effective root for user data (results/, work/, funasr-models/, spk/).
    pub fn data_root(&self) -> PathBuf {
        self.config
            .lock()
            .unwrap()
            .data_root(&self.app_data_dir)
    }
}

fn emit_app(app: &AppHandle, event: &str, payload: serde_json::Value) {
    let _ = app.emit(event, payload);
}

fn ensure_assets(app: &AppHandle, cfg: &AppConfig) -> Result<Assets, String> {
    let assets = Assets::resolve(app);
    if !assets.silero_model.exists() {
        return Err(format!(
            "Silero VAD model missing at {}",
            assets.silero_model.display()
        ));
    }
    if !assets.model_tools_script().exists() {
        return Err(format!(
            "model helper script missing at {}",
            assets.model_tools_script().display()
        ));
    }
    let report = validate_model_config_impl(&assets, cfg)?;
    if report.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let errors = report
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .unwrap_or_default();
        return Err(if errors.is_empty() {
            "FunASR model configuration is incomplete — open Settings and configure the ASR model"
                .to_string()
        } else {
            format!("FunASR model configuration invalid: {errors}")
        });
    }
    Ok(assets)
}

/// Run the Python model checker (`scripts/model_tools.py check`) for the
/// current config and return its parsed JSON report. This only stats files and
/// reads configs — it never imports torch/funasr, so it is fast and safe to
/// call before every run.
pub(crate) fn validate_model_config_impl(
    assets: &Assets,
    cfg: &AppConfig,
) -> Result<serde_json::Value, String> {
    let script = assets.model_tools_script();
    if !script.exists() {
        return Err(format!("model helper script missing at {}", script.display()));
    }
    let mut checks = serde_json::Map::new();
    if cfg.asr_is_api() {
        checks.insert(
            "asr".to_string(),
            serde_json::json!({
                "kind": "api", "enabled": true,
                "baseUrl": cfg.qwen3_base_url.trim(),
                "apiKey": cfg.qwen3_api_key.trim(),
                "model": cfg.qwen3_model.trim(),
            }),
        );
    } else {
        checks.insert(
            "asr".to_string(),
            serde_json::json!({
                "kind": "asr", "enabled": true,
                "type": cfg.asr_type, "dir": cfg.asr_model_dir.trim(),
            }),
        );
    }
    // No "vad" slot: VAD is the bundled native Silero model (no user config).
    if cfg.spk_enabled && !cfg.spk_model_dir.trim().is_empty() {
        checks.insert(
            "spk".to_string(),
            serde_json::json!({
                "kind": "spk", "enabled": true,
                "type": "cam++", "dir": cfg.spk_model_dir.trim(),
            }),
        );
    } else {
        checks.insert(
            "spk".to_string(),
            serde_json::json!({ "kind": "spk", "enabled": false }),
        );
    }

    let check_path = std::env::temp_dir().join(format!(
        "liveneko_model_check_{}.json",
        std::process::id()
    ));
    std::fs::write(
        &check_path,
        serde_json::to_string(&checks).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("write model check: {e}"))?;
    let out = run_capture(
        PYTHON_CMD,
        &[
            "-u",
            script.to_str().unwrap_or(""),
            "check",
            "--config",
            check_path.to_str().unwrap_or(""),
        ],
    );
    let _ = std::fs::remove_file(&check_path);
    let out = out?;
    serde_json::from_str(&out).map_err(|e| format!("invalid model check output: {e}"))
}

// Commands
#[tauri::command]
pub fn check_environment(
    app: AppHandle,
    state: State<'_, AppState>,
    force: Option<bool>,
) -> Result<serde_json::Value, String> {
    // Environment check runs only once per machine unless forced (Re-check button).
    let env_report_path = state.app_data_dir.join("env_report.json");
    let mut cfg = state.config.lock().unwrap().clone();
    if !force.unwrap_or(false)
        && cfg.env_checked
        && env_report_path.exists()
        && let Ok(text) = std::fs::read_to_string(&env_report_path)
        && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text)
    {
        return Ok(parsed);
    }

    let report = run_environment_check(&app);
    let _ = std::fs::write(&env_report_path, report.to_string());
    cfg.env_checked = true;
    let _ = cfg.save(&state.app_data_dir);
    *state.config.lock().unwrap() = cfg;
    Ok(report)
}

/// Interpreter used to run all Python worker scripts (must be on PATH).
pub(crate) const PYTHON_CMD: &str = "python";

fn run_environment_check(app: &AppHandle) -> serde_json::Value {
    let assets = Assets::resolve(app);

    let python_check = run_capture(PYTHON_CMD, &["--version"]);
    let python_ok = python_check.is_ok();
    let python_version = python_check.unwrap_or_default();
    let ffmpeg_ok = run_capture("ffmpeg", &["-version"]).is_ok();

    // Python libraries check
    let mut libs = serde_json::json!({});
    let mut download_libs = serde_json::json!({});
    let mut cuda = false;
    let mut python_libs_ok = false;
    if python_ok {
        let script = assets.scripts_dir.join("env_check.py");
        if script.exists()
            && let Ok(out) =
                run_capture_cwd(PYTHON_CMD, &["-u", script.to_str().unwrap()], None)
            && let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&out)
        {
            cuda = parsed
                .get("cuda")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            // env_check.py emits {"cuda": .., "libraries": {..}}; the frontend expects the flat per-library object under "libraries", so unwrap it here.
            libs = parsed.get("libraries").cloned().unwrap_or_default();
            download_libs = parsed
                .get("downloadLibraries")
                .cloned()
                .unwrap_or_default();
            python_libs_ok = true;
        }
    }

    serde_json::json!({
        "python": { "command": PYTHON_CMD, "ok": python_ok, "version": python_version },
        "ffmpeg": ffmpeg_ok,
        "cuda": cuda,
        "pythonLibraries": python_libs_ok,
        "libraries": libs,
        "downloadLibraries": download_libs,
        "assets": {
            "promptMd": assets.prompt_md.exists(),
            "scripts": assets.scripts_dir.exists(),
        },
    })
}

fn run_capture(program: &str, args: &[&str]) -> Result<String, String> {
    run_capture_cwd(program, args, None)
}

fn run_capture_cwd(
    program: &str,
    args: &[&str],
    cwd: Option<&std::path::Path>,
) -> Result<String, String> {
    use std::process::Command;
    let mut cmd = Command::new(program);
    cmd.args(args);
    if let Some(d) = cwd {
        cmd.current_dir(d);
    }
    // prevent a console window flashing when running python from the GUI
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let out = cmd.output().map_err(|e| format!("{e}"))?;
    if !out.status.success() {
        let msg = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(if msg.is_empty() {
            format!("exit code {:?}", out.status.code())
        } else {
            msg
        });
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

#[tauri::command]
pub fn get_config(state: State<'_, AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

/// OS appearance captured once at startup.
#[derive(serde::Serialize)]
pub struct OsThemeInfo {
    /// "light" | "dark"
    pub theme: String,
    pub accent: Option<String>,
}

#[tauri::command]
pub fn get_os_theme(state: State<'_, AppState>) -> OsThemeInfo {
    OsThemeInfo {
        theme: state.os_theme.clone(),
        accent: state.os_accent.clone(),
    }
}

#[tauri::command]
pub fn save_config(
    state: State<'_, AppState>,
    mut config: AppConfig,
    speaker_wav: Option<String>,
) -> Result<(), String> {
    // preserve backend-managed fields that the settings UI does not send
    let existing = state.config.lock().unwrap().clone();
    config.env_checked = existing.env_checked;
    config.normalize(&state.app_data_dir);

    // Data directory: when the root changes, relocate the existing user data so
    // nothing is left behind in the old location.
    let old_root = existing.data_root(&state.app_data_dir);
    let new_root = config.data_root(&state.app_data_dir);
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    if old_root != new_root {
        if state.running.load(Ordering::SeqCst) {
            return Err(
                "the data directory cannot be changed while the pipeline is running".to_string(),
            );
        }
        validate_data_root(&new_root, &old_root)?;
        std::fs::create_dir_all(&new_root)
            .map_err(|e| format!("create data directory {}: {e}", new_root.display()))?;
        rebase_funasr_paths(&mut config, &old_root, &new_root);
        moved = move_data_dirs(&old_root, &new_root)?;
    }

    // Speaker reference handling: the WAV is imported (16 kHz-checked, converted
    // when needed) into <dataRoot>/spk/ at save time, so the file persists with
    // the settings. If anything after the move fails, put the data back.
    match apply_speaker_and_save(&state, &mut config, &new_root, speaker_wav.as_deref()) {
        Ok(()) => {
            *state.config.lock().unwrap() = config;
            Ok(())
        }
        Err(e) => {
            rollback_moves(&moved);
            Err(e)
        }
    }
}

/// Persist the speaker reference (importing a freshly picked WAV when given)
/// and write config.json. Does not touch `AppState`.
fn apply_speaker_and_save(
    state: &AppState,
    config: &mut AppConfig,
    root: &Path,
    speaker_wav: Option<&str>,
) -> Result<(), String> {
    if config.speaker_name.is_empty() {
        // speaker identification off — drop any stored reference
        let _ = std::fs::remove_dir_all(root.join("spk"));
    } else if speaker_wav.is_some_and(|s| !s.trim().is_empty()) {
        config.speaker_ref = import_speaker_wav(root, speaker_wav.unwrap(), &config.speaker_name)?;
    } else {
        // keep the previously imported reference; it must still exist on disk
        let wav = root.join("spk").join(config.speaker_ref.trim());
        if config.speaker_ref.trim().is_empty() || !wav.exists() {
            return Err(format!(
                "speaker \"{}\" needs a reference WAV file — choose one in Settings",
                config.speaker_name
            ));
        }
    }
    config.save(&state.app_data_dir)
}

/// Reject a data directory that cannot safely become the new data root.
fn validate_data_root(new_root: &Path, old_root: &Path) -> Result<(), String> {
    if !new_root.is_absolute() {
        return Err("data directory must be an absolute path".to_string());
    }
    if new_root.exists() && !new_root.is_dir() {
        return Err(format!(
            "data directory is a file, not a folder: {}",
            new_root.display()
        ));
    }
    // Moving the root into itself would recurse.
    if new_root != old_root && new_root.starts_with(old_root) {
        return Err(format!(
            "data directory cannot be inside {}",
            old_root.display()
        ));
    }
    Ok(())
}

/// Point the model paths at the new root when they referenced the old default
/// model store (an externally-chosen store is left alone). `speaker_ref` is a
/// bare filename, so it follows the spk root automatically.
fn rebase_funasr_paths(config: &mut AppConfig, old_root: &Path, new_root: &Path) {
    let old_store = old_root.join("funasr-models");
    let new_store = new_root.join("funasr-models");
    if config.funasr_models_dir.trim().is_empty() {
        config.funasr_models_dir = new_store.to_string_lossy().to_string();
    } else if Path::new(config.funasr_models_dir.trim()) == old_store {
        config.funasr_models_dir = new_store.to_string_lossy().to_string();
    }
    for dir in [&mut config.asr_model_dir, &mut config.spk_model_dir] {
        let trimmed = dir.trim().to_string();
        if let Ok(rel) = Path::new(&trimmed).strip_prefix(&old_store) {
            *dir = new_store.join(rel).to_string_lossy().to_string();
        }
    }
}

/// Move the user-data subdirectories into `new_root`, returning the pairs that
/// were moved so a later failure can be rolled back. Refuses to merge into a
/// destination that already holds data.
fn move_data_dirs(old_root: &Path, new_root: &Path) -> Result<Vec<(PathBuf, PathBuf)>, String> {
    const SUBDIRS: [&str; 4] = ["results", "work", "funasr-models", "spk"];
    let mut moved: Vec<(PathBuf, PathBuf)> = Vec::new();
    for name in SUBDIRS {
        let src = old_root.join(name);
        if !src.exists() {
            continue;
        }
        let dst = new_root.join(name);
        if dst.exists() {
            let non_empty = std::fs::read_dir(&dst)
                .map(|mut d| d.next().is_some())
                .unwrap_or(true);
            if non_empty {
                rollback_moves(&moved);
                return Err(format!(
                    "{} already contains data — choose an empty folder",
                    dst.display()
                ));
            }
            let _ = std::fs::remove_dir(&dst);
        }
        if let Err(e) = move_dir(&src, &dst) {
            rollback_moves(&moved);
            return Err(format!(
                "move {} -> {}: {e}",
                src.display(),
                dst.display()
            ));
        }
        moved.push((src, dst));
    }
    Ok(moved)
}

/// Rename a directory, falling back to copy+delete when it spans volumes.
fn move_dir(src: &Path, dst: &Path) -> Result<(), String> {
    if std::fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    copy_dir_recursive(src, dst).map_err(|e| e.to_string())?;
    std::fs::remove_dir_all(src).map_err(|e| e.to_string())
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &to)?;
        } else {
            std::fs::copy(entry.path(), &to)?;
        }
    }
    Ok(())
}

/// Best-effort undo of `move_data_dirs` (a cross-volume rollback falls back to
/// leaving the files where they are).
fn rollback_moves(moved: &[(PathBuf, PathBuf)]) {
    for (src, dst) in moved.iter().rev() {
        let _ = std::fs::rename(dst, src);
    }
}

/// Directory holding the imported speaker reference WAV(s).
fn speaker_dir(root: &Path) -> std::path::PathBuf {
    root.join("spk")
}

/// Filesystem-safe stem derived from the speaker display name.
fn sanitize_speaker_stem(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    let trimmed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let stem = if trimmed.is_empty() {
        "speaker".to_string()
    } else {
        trimmed
    };
    stem.chars().take(60).collect()
}

/// Folder name for a downloaded model, taken from the model name — the last
/// segment of the hub repo id, with the org prefix dropped:
/// `FunAudioLLM/SenseVoiceSmall` -> `SenseVoiceSmall`.
fn model_folder_name(model_id: &str) -> String {
    // Model name = the last non-empty path segment of the repo id.
    let leaf = model_id
        .trim()
        .rsplit(|c| c == '/' || c == '\\')
        .find(|s| !s.is_empty())
        .unwrap_or("");
    let mut out = String::new();
    for c in leaf.chars() {
        match c {
            ':' | '*' | '?' | '"' | '<' | '>' | '|' => out.push('_'),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    // Windows forbids trailing spaces/dots in a path component.
    while out.ends_with(' ') || out.ends_with('.') {
        out.pop();
    }
    if out.is_empty() || out == "." || out == ".." {
        "model".to_string()
    } else {
        out
    }
}

/// Read the sample rate of `src` by parsing ffmpeg's stream info output
/// ("... Audio: pcm_s16le, 16000 Hz, 1 channels ..."). ffmpeg prints that info to stderr and exits non-zero when no output file is given, which is fine for a header probe.
fn probe_wav_sample_rate(src: &std::path::Path) -> Result<u32, String> {
    use std::process::Command;
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-hide_banner", "-i"]).arg(src);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // no console window flash
    }
    let out = cmd.output().map_err(|e| format!("run ffmpeg: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() {
        let Some(audio) = line.find("Audio:") else {
            continue;
        };
        let rest = &line[audio..];
        let Some(hz) = rest.find("Hz") else { continue };
        let digits: String = rest[..hz]
            .chars()
            .rev()
            .skip_while(|c| c.is_whitespace())
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        if let Ok(rate) = digits.parse::<u32>() {
            return Ok(rate);
        }
    }
    Err(format!(
        "could not read the audio stream info of {} — is it a valid media file?",
        src.display()
    ))
}

/// Import the user-provided reference WAV into `<root>/spk/`: verify with
/// ffmpeg, convert to 16 kHz mono when needed, and keep exactly one reference
/// file. Returns the stored filename (recorded in the config).
fn import_speaker_wav(root: &Path, src: &str, name: &str) -> Result<String, String> {
    if run_capture("ffmpeg", &["-version"]).is_err() {
        return Err(
            "ffmpeg is required to import a speaker reference WAV but was not found".to_string(),
        );
    }
    let src_path = std::path::Path::new(src);
    if !src_path.exists() {
        return Err(format!("reference WAV not found: {src}"));
    }
    let dir = speaker_dir(root);
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;

    let file_name = format!("{}.wav", sanitize_speaker_stem(name));
    let target = dir.join(&file_name);

    let rate = probe_wav_sample_rate(src_path)?;
    if rate == 16000 {
        // already 16 kHz — plain copy preserves the original audio bit-for-bit
        std::fs::copy(src_path, &target)
            .map_err(|e| format!("copy {}: {e}", src_path.display()))?;
    } else {
        // resample to 16 kHz mono so the spk model can consume it directly
        run_capture(
            "ffmpeg",
            &[
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                src,
                "-ar",
                "16000",
                "-ac",
                "1",
                "-vn",
                target.to_str().ok_or("reference path is not valid UTF-8")?,
            ],
        )
        .map_err(|e| format!("convert to 16 kHz: {e}"))?;
    }

    // keep exactly one reference file: remove leftovers from earlier speakers
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            let is_wav = p
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("wav"));
            if is_wav && p != target {
                let _ = std::fs::remove_file(p);
            }
        }
    }
    Ok(file_name)
}

/// Return the current effective summary prompt: the user's custom prompt if set, otherwise the bundled default prompt.md.
#[tauri::command]
pub fn get_prompt(app: AppHandle, state: State<'_, AppState>) -> Result<String, String> {
    let cfg = state.config.lock().unwrap().clone();
    if !cfg.custom_prompt.trim().is_empty() {
        return Ok(cfg.custom_prompt.clone());
    }
    let assets = crate::assets::Assets::resolve(&app);
    std::fs::read_to_string(&assets.prompt_md).map_err(|e| format!("read prompt: {e}"))
}

/// Restore the bundled default prompt (clears the custom one).
#[tauri::command]
pub fn reset_prompt(state: State<'_, AppState>) -> Result<(), String> {
    let mut cfg = state.config.lock().unwrap().clone();
    cfg.custom_prompt = String::new();
    cfg.save(&state.app_data_dir)?;
    *state.config.lock().unwrap() = cfg;
    Ok(())
}

/// First-launch setup status so the frontend can drive the guided wizard.
#[tauri::command]
pub fn get_setup_status(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let cfg = state.config.lock().unwrap().clone();
    let model_ok =
        !cfg.videoneko_model_dir.is_empty() && std::fs::metadata(&cfg.videoneko_model_dir).is_ok();
    let needs_funasr = !cfg.funasr_configured();
    let needs_spk = cfg.spk_enabled && cfg.spk_model_dir.trim().is_empty();
    Ok(serde_json::json!({
        "firstLaunch": cfg.videoneko_model_dir.is_empty()
            && cfg.ollama_model.is_empty()
            && cfg.llamacpp_model.is_empty()
            && cfg.api_model.is_empty(),
        "needsVideoneko": !model_ok,
        "needsFunasr": needs_funasr,
        "needsSpk": needs_spk,
        "dataDir": cfg.data_root(&state.app_data_dir).to_string_lossy(),
        "videonekoModelDir": cfg.videoneko_model_dir,
        "funasrModelsDir": cfg.funasr_models_dir,
        "asrType": cfg.asr_type,
        "asrModelDir": cfg.asr_model_dir,
        "spkModelDir": cfg.spk_model_dir,
    }))
}

#[tauri::command]
pub fn get_queue(state: State<'_, AppState>) -> Vec<QueueItem> {
    state.queue.lock().unwrap().clone()
}

#[tauri::command]
pub async fn add_url(
    state: State<'_, AppState>,
    url: String,
    title: Option<String>,
) -> Result<QueueItem, String> {
    if !url.starts_with("http") {
        return Err("URL must start with http:// or https://".to_string());
    }
    // Use the caller-provided title, otherwise probe the real video title in-process so the queue shows it instead of a placeholder.
    let title = match title {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            let cookie_browser = state.config.lock().unwrap().cookie_browser.clone();
            match crate::pipeline::probe_ytdlp_titles(&url, &cookie_browser) {
                Ok(titles) => titles
                    .first()
                    .map(|t| crate::pipeline::simplify_title_str(t))
                    .unwrap_or_else(|| "Bilibili video".to_string()),
                Err(_) => "Bilibili video".to_string(),
            }
        }
    };
    let id = Uuid::new_v4().simple().to_string();
    let item = QueueItem::from_url(id, title, url);
    state.queue.lock().unwrap().push(item.clone());
    Ok(item)
}

#[tauri::command]
pub fn add_local_file(state: State<'_, AppState>, path: String) -> Result<QueueItem, String> {
    let pb = PathBuf::from(&path);
    if !pb.exists() {
        return Err(format!("file not found: {path}"));
    }
    let id = Uuid::new_v4().simple().to_string();
    let item = QueueItem::from_file(id, path);
    state.queue.lock().unwrap().push(item.clone());
    Ok(item)
}

#[tauri::command]
pub fn remove_item(state: State<'_, AppState>, id: String) -> Result<(), String> {
    let mut q = state.queue.lock().unwrap();
    q.retain(|i| i.id != id);
    Ok(())
}

#[tauri::command]
pub fn clear_queue(state: State<'_, AppState>) {
    state.queue.lock().unwrap().clear();
}

#[tauri::command]
pub fn is_running(state: State<'_, AppState>) -> bool {
    state.running.load(Ordering::SeqCst)
}

#[tauri::command]
pub fn start_pipeline(app: AppHandle, state: State<'_, AppState>) -> Result<(), String> {
    if state.running.swap(true, Ordering::SeqCst) {
        return Err("pipeline already running".to_string());
    }
    let cfg = state.config.lock().unwrap().clone();
    let assets = ensure_assets(&app, &cfg)?;
    let items: Vec<QueueItem> = state
        .queue
        .lock()
        .unwrap()
        .iter()
        .filter(|i| matches!(i.status, ItemStatus::Queued))
        .cloned()
        .collect();
    if items.is_empty() {
        state.running.store(false, Ordering::SeqCst);
        return Err("queue is empty".to_string());
    }
    // reset cancellation for this run
    state.pipeline.cancel.store(false, Ordering::SeqCst);

    // Open the run log file (work/pipeline.log);
    let data_root = cfg.data_root(&state.app_data_dir);
    let work_root = data_root.join("work");
    let _ = std::fs::create_dir_all(&work_root);
    if let Ok(f) = std::fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(work_root.join("pipeline.log"))
    {
        *state.pipeline.log_file.lock().unwrap() = Some(f);
    }

    let app2 = app.clone();
    let handle = state.pipeline.clone();

    std::thread::spawn(move || {
        let mut runner = Runner::new(
            app2.clone(),
            assets.clone(),
            handle.clone(),
            cfg.cookie_browser.clone(),
            data_root,
        );
        // launch resident model servers ONCE (models load here, reused for all queued videos); they stay alive until the queue is done.
        if let Err(e) = runner.start_model_servers(&cfg) {
            log_line(
                &app2,
                &handle.log_file,
                "pipeline",
                &format!("[model] failed to start servers: {e}"),
            );
            log_line(
                &app2,
                &handle.log_file,
                "pipeline",
                "[model] please check the environment check results in Settings",
            );
            app2.state::<AppState>()
                .running
                .store(false, Ordering::SeqCst);
            emit_app(&app2, "pipeline://finished", serde_json::json!({}));
            return;
        }
        for item in items {
            if runner.is_cancelled() {
                break;
            }
            // mark running
            {
                let app_state = app2.state::<AppState>();
                let mut q = app_state.queue.lock().unwrap();
                if let Some(it) = q.iter_mut().find(|i| i.id == item.id) {
                    it.status = ItemStatus::Running;
                }
            }
            emit_app(
                &app2,
                "pipeline://start",
                serde_json::json!({ "itemId": item.id }),
            );

            let work_dir = work_root.join(&item.id);
            let _ = std::fs::create_dir_all(&work_dir);

            let result = pipeline::run_item(&mut runner, &cfg, &item, &work_dir);

            {
                let app_state = app2.state::<AppState>();
                let mut q = app_state.queue.lock().unwrap();
                if let Some(it) = q.iter_mut().find(|i| i.id == item.id) {
                    match &result {
                        Ok(()) => it.status = ItemStatus::Done,
                        Err(e) if e == "cancelled" => it.status = ItemStatus::Cancelled,
                        Err(e) => {
                            it.status = ItemStatus::Error;
                            it.error = Some(e.clone());
                        }
                    }
                }
            }
            match &result {
                Ok(()) => emit_app(
                    &app2,
                    "pipeline://done",
                    serde_json::json!({ "itemId": item.id, "ok": true }),
                ),
                Err(e) if e == "cancelled" => emit_app(
                    &app2,
                    "pipeline://done",
                    serde_json::json!({ "itemId": item.id, "ok": false, "cancelled": true }),
                ),
                Err(e) => emit_app(
                    &app2,
                    "pipeline://done",
                    serde_json::json!({ "itemId": item.id, "ok": false, "error": e }),
                ),
            }
        }
        runner.stop_model_servers();
        app2.state::<AppState>()
            .running
            .store(false, Ordering::SeqCst);
        emit_app(&app2, "pipeline://finished", serde_json::json!({}));
    });
    Ok(())
}

#[tauri::command]
pub fn stop_pipeline(state: State<'_, AppState>) {
    state.pipeline.cancel.store(true, Ordering::SeqCst);
    if let Some(mut child) = state.pipeline.child.lock().unwrap().take() {
        let _ = child.kill();
    }
    // kill resident model servers and any running ffmpeg subprocesses
    let mut pids: Vec<u32> = state.pipeline.model_pids.lock().unwrap().clone();
    pids.extend(state.pipeline.ffmpeg_pids.lock().unwrap().iter().copied());
    for pid in pids {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let _ = std::process::Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .creation_flags(0x08000000)
                .status();
        }
        #[cfg(not(windows))]
        {
            let _ = std::process::Command::new("kill")
                .arg("-9")
                .arg(pid.to_string())
                .status();
        }
    }
}

/// Convert any such leftovers into `results/<stem>/<name>` directories.
fn migrate_legacy_results(state: &AppState) {
    let data_root = state.data_root();
    let target = data_root.join("results");
    std::fs::create_dir_all(&target).ok();

    // Scan both the config root and the (possibly moved) data root, so legacy
    // flat files are found regardless of where the data dir points.
    let mut flat_dirs = vec![
        state.app_data_dir.join("work").join("results"),
        state.app_data_dir.join("results"),
        data_root.join("work").join("results"),
        target.clone(),
    ];
    for legacy in flat_dirs.drain(..) {
        if !legacy.exists() {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(&legacy) {
            let names: Vec<String> = rd
                .flatten()
                .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                .collect();
            for name in names {
                if name.ends_with(".summary.md") {
                    let stem = name.trim_end_matches(".summary.md");
                    migrate_flat_one(&legacy, &target, stem);
                }
            }
        }
    }
}

fn migrate_flat_one(legacy: &std::path::Path, target: &std::path::Path, stem: &str) {
    let dir = target.join(stem);
    std::fs::create_dir_all(&dir).ok();
    let map = [
        (format!("{stem}.summary.md"), "summary.md"),
        (format!("{stem}.asr.txt"), "asr.txt"),
        (format!("{stem}.visual.txt"), "visual.txt"),
        (format!("{stem}.thinking.txt"), "thinking.txt"),
    ];
    for (src_name, dst_name) in map {
        let src = legacy.join(&src_name);
        let dst = dir.join(dst_name);
        if src.exists() && !dst.exists() {
            let _ = std::fs::copy(&src, &dst);
        }
    }
}

#[tauri::command]
pub fn list_results(state: State<'_, AppState>) -> Result<Vec<serde_json::Value>, String> {
    if !state.migrated.swap(true, Ordering::SeqCst) {
        migrate_legacy_results(state.inner());
    }
    let results_dir = state.data_root().join("results");
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&results_dir) {
        for entry in rd.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let summary_file = dir.join("summary.md");
            let stem = dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            // A single metadata() call doubles as the existence check and yields both mtime and size, avoiding a separate `exists()` stat.
            let meta = match std::fs::metadata(&summary_file) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let modified = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
            out.push(serde_json::json!({
                "stem": stem,
                "modified": modified,
                "size": meta.len(),
            }));
        }
    }
    out.sort_by_key(|v| -v["modified"].as_i64().unwrap_or(0));
    Ok(out)
}

#[tauri::command]
pub fn read_result(state: State<'_, AppState>, stem: String) -> Result<serde_json::Value, String> {
    let results_dir = state.data_root().join("results").join(&stem);
    let summary = std::fs::read_to_string(results_dir.join("summary.md")).unwrap_or_default();
    let thinking = std::fs::read_to_string(results_dir.join("thinking.txt")).unwrap_or_default();
    let asr = std::fs::read_to_string(results_dir.join("asr.txt")).unwrap_or_default();
    let visual = std::fs::read_to_string(results_dir.join("visual.txt")).unwrap_or_default();
    Ok(serde_json::json!({
        "stem": stem,
        "summary": summary,
        "thinking": thinking,
        "asr": asr,
        "visual": visual,
    }))
}

/// Search all summaries. Supports regular expressions. Returns matching stems with a content snippet.
#[tauri::command]
pub fn search_results(
    state: State<'_, AppState>,
    query: String,
    use_regex: bool,
) -> Result<Vec<serde_json::Value>, String> {
    let results_dir = state.data_root().join("results");
    let mut out = Vec::new();
    let q = query.trim();
    if q.is_empty() {
        return Ok(out);
    }
    let pattern: Option<regex::Regex> = if use_regex {
        Some(regex::Regex::new(q).map_err(|e| format!("invalid regex: {e}"))?)
    } else {
        None
    };
    // Lowercase the query once; it is reused for every result below.
    let q_lower = q.to_lowercase();
    let matches_query = |text: &str| -> bool {
        match &pattern {
            Some(re) => re.is_match(text),
            None => text.to_lowercase().contains(&q_lower),
        }
    };

    if let Ok(rd) = std::fs::read_dir(&results_dir) {
        for entry in rd.flatten() {
            let dir = entry.path();
            if !dir.is_dir() {
                continue;
            }
            let summary_file = dir.join("summary.md");
            let stem = dir
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let summary = std::fs::read_to_string(&summary_file).unwrap_or_default();
            if !matches_query(&stem) && !matches_query(&summary) {
                continue;
            }
            // Build a snippet around the first match.
            let snippet = make_snippet(&stem, &summary, &q_lower, &pattern);
            out.push(serde_json::json!({
                "stem": stem,
                "snippet": snippet,
            }));
        }
    }
    out.sort_by_key(|v| v["stem"].as_str().unwrap_or("").to_string());
    Ok(out)
}

/// Map a byte index inside the lowercased copy back onto the original string. `to_lowercase` may change byte length for some Unicode chars, so the index is translated char-by-char to keep it a valid char boundary.
fn lower_to_orig_index(orig: &str, lower: &str, idx: usize) -> usize {
    let n = lower[..idx.min(lower.len())].chars().count();
    orig.char_indices()
        .nth(n)
        .map(|(i, _)| i)
        .unwrap_or(orig.len())
}

/// Snap a byte index down/up to the nearest UTF-8 char boundary so that `&s[lo..hi]` can never panic.
fn prev_char_boundary(s: &str, mut i: usize) -> usize {
    if i > s.len() {
        i = s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}
fn next_char_boundary(s: &str, mut i: usize) -> usize {
    if i > s.len() {
        return s.len();
    }
    while i < s.len() && !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

fn make_snippet(
    stem: &str,
    summary: &str,
    q_lower: &str,
    pattern: &Option<regex::Regex>,
) -> String {
    // Prefer a match in the summary content; fall back to the title.
    let haystack = if summary.is_empty() { stem } else { summary };
    let find_match = |text: &str| -> Option<(usize, usize)> {
        match pattern {
            Some(re) => re.find(text).map(|m| (m.start(), m.end())),
            None => {
                let lower = text.to_lowercase();
                lower.find(q_lower).map(|i| {
                    let start = lower_to_orig_index(text, &lower, i);
                    let n_chars =
                        lower[..i.min(lower.len())].chars().count() + q_lower.chars().count();
                    let end = text
                        .char_indices()
                        .nth(n_chars)
                        .map(|(i, _)| i)
                        .unwrap_or(text.len());
                    (start, end)
                })
            }
        }
    };
    if let Some((start, end)) = find_match(haystack) {
        let lo = prev_char_boundary(haystack, start.saturating_sub(60));
        let hi = next_char_boundary(haystack, end + 60);
        let mut snip = haystack[lo..hi].to_string();
        if lo > 0 {
            snip.insert(0, '…');
        }
        if hi < haystack.len() {
            snip.push('…');
        }
        return snip;
    }
    // match was in the title
    let mut snip = stem.chars().take(120).collect::<String>();
    if stem.chars().count() > 120 {
        snip.push('…');
    }
    snip
}

/// Re-run only the summarization stage for an existing result, reusing its stored asr.txt + visual.txt. Lets the user switch the LLM engine regenerate the summary without reprocessing the video.
#[tauri::command]
pub fn re_summarize(
    app: AppHandle,
    state: State<'_, AppState>,
    stem: String,
) -> Result<(), String> {
    let assets = crate::assets::Assets::resolve(&app);
    let result_dir = state.data_root().join("results").join(&stem);
    let asr_path = result_dir.join("asr.txt");
    let visual_path = result_dir.join("visual.txt");
    if !asr_path.exists() {
        return Err(format!("stored asr.txt not found for '{stem}'"));
    }
    if !visual_path.exists() {
        return Err(format!("stored visual.txt not found for '{stem}'"));
    }
    let config = state.config.lock().unwrap().clone();
    let data_root = config.data_root(&state.app_data_dir);
    let runner = Runner::new(
        app.clone(),
        assets,
        state.pipeline.clone(),
        config.cookie_browser.clone(),
        data_root.clone(),
    );
    // mark as running so the UI hides the Start button / shows state
    state.pipeline.cancel.store(false, Ordering::SeqCst);
    let app2 = app.clone();
    let stem2 = stem.clone();
    // write into a per-stem work dir, then move outputs into results/<stem> so a companion thinking.txt doesn't collide across concurrent re-summaries
    let work_dir = data_root.join("work").join(format!("resummarize-{stem}"));
    std::fs::create_dir_all(&work_dir).map_err(|e| e.to_string())?;
    let work_dir2 = work_dir.clone();
    std::thread::spawn(move || {
        emit_app(
            &app2,
            "pipeline://start",
            serde_json::json!({ "itemId": stem2 }),
        );
        let output = work_dir.join("summary.md");
        let result = runner.summarize(&config, &stem2, &stem2, &visual_path, &asr_path, &output);
        let res = result.and_then(|_| {
            std::fs::copy(&output, result_dir.join("summary.md")).map_err(|e| e.to_string())?;
            let _ = std::fs::copy(
                work_dir.join("thinking.txt"),
                result_dir.join("thinking.txt"),
            );
            Ok(())
        });
        // On failure/cancellation, keep the latest reason visible in the result while preserving the stored asr.txt/visual.txt for another retry.
        if let Err(e) = &res {
            let reason = if e == "cancelled" {
                "# Summary canceled\n\nThe request to the LLM engine was canceled.\n\nThe ASR and visual transcripts were saved.\n".to_string()
            } else {
                format!(
                    "# Summary failed\n\nReason: {e}\n\nThe ASR and visual transcripts were saved.\nUse **Re-summarize** to try again.\n"
                )
            };
            let _ = std::fs::write(result_dir.join("summary.md"), reason);
        }
        match res {
            Ok(()) => emit_app(
                &app2,
                "pipeline://done",
                serde_json::json!({ "itemId": stem2, "ok": true }),
            ),
            Err(e) => emit_app(
                &app2,
                "pipeline://done",
                serde_json::json!({ "itemId": stem2, "ok": false, "error": e }),
            ),
        }
        let _ = std::fs::remove_dir_all(&work_dir2);
        emit_app(&app2, "pipeline://finished", serde_json::json!({}));
    });
    Ok(())
}

#[tauri::command]
pub fn export_result(state: State<'_, AppState>, stem: String, dest: String) -> Result<(), String> {
    let results_dir = state.data_root().join("results").join(&stem);
    let src = results_dir.join("summary.md");
    std::fs::copy(&src, PathBuf::from(&dest)).map_err(|e| format!("copy: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn delete_result(state: State<'_, AppState>, stem: String) -> Result<(), String> {
    let dir = state.data_root().join("results").join(&stem);
    let _ = std::fs::remove_dir_all(&dir);
    Ok(())
}

#[tauri::command]
pub async fn test_api_connection(
    base_url: String,
    api_key: String,
    model: String,
) -> Result<serde_json::Value, String> {
    use crate::pipeline::chat_completions_url;
    let client = openai_rust2::Client::shared_client();
    let url = chat_completions_url(&base_url);
    let body = serde_json::json!({
        "model": model,
        "messages": [{ "role": "user", "content": "ping" }],
        "max_tokens": 8,
        "stream": false,
        "thinking": { "type": "disabled" },
    });
    let resp = client
        .post(&url)
        .bearer_auth(&api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if status.is_success() {
        let json: serde_json::Value = serde_json::from_str(&text).unwrap_or_default();
        let reply = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();
        Ok(serde_json::json!({ "ok": true, "reply": reply }))
    } else {
        Err(format!("HTTP {status}: {text}"))
    }
}

/// Validate the FunASR model configuration (paths, file contents and model
/// type compatibility) without loading any model. Returns the same report the
/// pipeline uses to gate a run.
#[tauri::command]
pub async fn validate_model_config(
    app: AppHandle,
    mut config: AppConfig,
) -> Result<serde_json::Value, String> {
    let app_data_dir = app
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."));
    config.normalize(&app_data_dir);
    tauri::async_runtime::spawn_blocking(move || {
        let assets = Assets::resolve(&app);
        validate_model_config_impl(&assets, &config)
    })
    .await
    .map_err(|e| format!("validate task failed: {e}"))?
}

/// Download a model snapshot from Hugging Face / ModelScope into its own
/// directory named after the model (`<base>/<asr|spk>/<model_folder_name>`).
/// Progress is emitted as `model://progress` events with
/// `{ "kind": "asr"|"spk", "progress": 0..100 }`.
#[tauri::command]
pub async fn download_model(
    app: AppHandle,
    state: State<'_, AppState>,
    kind: String,
    source: String,
    model_id: String,
    dest: Option<String>,
) -> Result<serde_json::Value, String> {
    let kind = kind.trim().to_lowercase();
    if !matches!(kind.as_str(), "asr" | "spk") {
        return Err("model kind must be 'asr' or 'spk'".to_string());
    }
    let source = source.trim().to_lowercase();
    if !matches!(source.as_str(), "huggingface" | "hf" | "modelscope" | "ms") {
        return Err("model source must be 'huggingface' or 'modelscope'".to_string());
    }
    let model_id = model_id.trim().to_string();
    if model_id.is_empty() {
        return Err("model id is required".to_string());
    }
    let dest = dest
        .map(|d| d.trim().to_string())
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| {
            let base = state.config.lock().unwrap().funasr_models_dir.clone();
            let base = if base.trim().is_empty() {
                state
                    .data_root()
                    .join("funasr-models")
                    .to_string_lossy()
                    .to_string()
            } else {
                base
            };
            std::path::Path::new(&base)
                .join(&kind)
                .join(model_folder_name(&model_id))
                .to_string_lossy()
                .to_string()
        });
    let assets = Assets::resolve(&app);
    let script = assets.model_tools_script();
    if !script.exists() {
        return Err(format!("model helper script missing at {}", script.display()));
    }
    let app2 = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        run_download_blocking(&app2, &script, &kind, &source, &model_id, &dest)
    })
    .await
    .map_err(|e| format!("download task failed: {e}"))?
}

/// Run the Python downloader, forwarding progress events while collecting the
/// final JSON result.
fn run_download_blocking(
    app: &AppHandle,
    script: &std::path::Path,
    kind: &str,
    source: &str,
    model_id: &str,
    dest: &str,
) -> Result<serde_json::Value, String> {
    use std::io::{BufRead, BufReader};

    let mut cmd = Command::new(PYTHON_CMD);
    cmd.arg("-u").arg(script).args([
        "download",
        "--source",
        source,
        "--model-id",
        model_id,
        "--dest",
        dest,
    ]);
    hide_console(&mut cmd);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("spawn downloader: {e}"))?;
    let stdout = child.stdout.take().ok_or("no stdout on downloader")?;
    let stderr = child.stderr.take().ok_or("no stderr on downloader")?;
    let err_thread = std::thread::spawn(move || {
        let mut text = String::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            text.push_str(&line);
            text.push('\n');
        }
        text
    });
    let mut result: Option<serde_json::Value> = None;
    for line in BufReader::new(stdout).lines().map_while(Result::ok) {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            if let Some(progress) = value.get("progress").and_then(|v| v.as_u64()) {
                let _ = app.emit(
                    "model://progress",
                    serde_json::json!({ "kind": kind, "progress": progress }),
                );
            } else if value.get("ok").is_some() {
                result = Some(value);
            }
        }
    }
    let status = child.wait().map_err(|e| format!("downloader wait: {e}"))?;
    let stderr_text = err_thread.join().unwrap_or_default();
    let result = result.ok_or_else(|| {
        format!(
            "downloader produced no result (exit {status:?}): {}",
            stderr_text.trim()
        )
    })?;
    if result.get("ok").and_then(|v| v.as_bool()) != Some(true) {
        let message = result
            .get("errors")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join("; ")
            })
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| stderr_text.trim().to_string());
        return Err(message);
    }
    Ok(result)
}

/// Probe the configured Qwen3-ASR (OpenAI-compatible) endpoint.
#[tauri::command]
pub async fn test_asr_connection(
    app: AppHandle,
    base_url: String,
    api_key: String,
    model: String,
    language: Option<String>,
) -> Result<serde_json::Value, String> {
    let assets = Assets::resolve(&app);
    let script = assets.model_tools_script();
    if !script.exists() {
        return Err(format!("model helper script missing at {}", script.display()));
    }
    tauri::async_runtime::spawn_blocking(move || {
        let out = run_capture(
            PYTHON_CMD,
            &[
                "-u",
                script.to_str().unwrap_or(""),
                "check-api",
                "--base-url",
                &base_url,
                "--api-key",
                &api_key,
                "--model",
                &model,
                "--language",
                language.as_deref().unwrap_or(""),
            ],
        )?;
        let value: serde_json::Value =
            serde_json::from_str(&out).map_err(|e| format!("invalid API check output: {e}"))?;
        if value.get("ok").and_then(|v| v.as_bool()) != Some(true) {
            let message = value
                .get("errors")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                })
                .unwrap_or_else(|| "connection failed".to_string());
            return Err(message);
        }
        Ok(value)
    })
    .await
    .map_err(|e| format!("API check task failed: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_data_root() {
        let base = std::env::temp_dir().join("liveneko_validate_data_root");
        let old = base.join("old");
        let nested = old.join("sub");
        let other = base.join("new");
        assert!(validate_data_root(Path::new("relative/dir"), &old).is_err());
        assert!(validate_data_root(&nested, &old).is_err());
        assert!(validate_data_root(&other, &old).is_ok());
        // Reverting to the current root (the default) is allowed.
        assert!(validate_data_root(&old, &old).is_ok());
    }

    #[test]
    fn test_rebase_funasr_paths() {
        let base = std::env::temp_dir().join("liveneko_rebase");
        let old_root = base.join("old");
        let new_root = base.join("new");
        let old_store = old_root.join("funasr-models");
        let new_store = new_root.join("funasr-models");
        let external = base.join("external").join("campplus");

        let mut cfg = AppConfig::new();
        cfg.funasr_models_dir = old_store.to_string_lossy().to_string();
        cfg.asr_model_dir = old_store
            .join("asr")
            .join("SenseVoiceSmall")
            .to_string_lossy()
            .to_string();
        cfg.spk_model_dir = external.to_string_lossy().to_string();
        rebase_funasr_paths(&mut cfg, &old_root, &new_root);
        assert_eq!(cfg.funasr_models_dir, new_store.to_string_lossy().to_string());
        assert_eq!(
            cfg.asr_model_dir,
            new_store
                .join("asr")
                .join("SenseVoiceSmall")
                .to_string_lossy()
                .to_string()
        );
        assert_eq!(
            cfg.spk_model_dir,
            external.to_string_lossy().to_string(),
            "an external model store is left untouched"
        );

        let mut blank = AppConfig::new();
        blank.funasr_models_dir = String::new();
        rebase_funasr_paths(&mut blank, &old_root, &new_root);
        assert_eq!(blank.funasr_models_dir, new_store.to_string_lossy().to_string());
    }

    #[test]
    fn test_move_data_dirs() {
        let base = std::env::temp_dir().join(format!("liveneko_move_{}", std::process::id()));
        let old = base.join("old");
        let new = base.join("new");
        let _ = std::fs::remove_dir_all(&base);
        let files = [
            "results/video/summary.md",
            "work/pipeline.log",
            "funasr-models/asr/SenseVoiceSmall/model.pt",
            "spk/taffy.wav",
        ];
        for file in files {
            let p = old.join(file);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"x").unwrap();
        }
        let moved = move_data_dirs(&old, &new).unwrap();
        assert_eq!(moved.len(), 4);
        for file in files {
            assert!(new.join(file).is_file(), "missing after move: {file}");
            assert!(!old.join(file).exists(), "not moved: {file}");
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_move_data_dirs_rejects_non_empty_destination() {
        let base = std::env::temp_dir().join(format!("liveneko_move_reject_{}", std::process::id()));
        let old = base.join("old");
        let new = base.join("new");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(old.join("results")).unwrap();
        std::fs::write(old.join("results/a.md"), b"x").unwrap();
        std::fs::create_dir_all(new.join("results")).unwrap();
        std::fs::write(new.join("results/existing.md"), b"y").unwrap();
        assert!(move_data_dirs(&old, &new).is_err());
        // the source is untouched after a refusal
        assert!(old.join("results/a.md").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_move_data_dirs_is_noop_when_sources_missing() {
        let base = std::env::temp_dir().join(format!("liveneko_move_none_{}", std::process::id()));
        let old = base.join("old");
        let new = base.join("new");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&old).unwrap();
        assert!(move_data_dirs(&old, &new).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn test_sanitize_speaker_stem() {
        assert_eq!(sanitize_speaker_stem("taffy"), "taffy");
        assert_eq!(sanitize_speaker_stem("  taffy  cat  "), "taffy cat");
        assert_eq!(
            sanitize_speaker_stem("a/b\\c:d*e?f\"g<h>i|j"),
            "a b c d e f g h i j"
        );
        assert_eq!(sanitize_speaker_stem("///"), "speaker");
        assert_eq!(sanitize_speaker_stem(""), "speaker");
    }

    #[test]
    fn test_model_folder_name() {
        // Only the model name is used; the org prefix is dropped.
        assert_eq!(
            model_folder_name("FunAudioLLM/SenseVoiceSmall"),
            "SenseVoiceSmall"
        );
        assert_eq!(model_folder_name("funasr/campplus"), "campplus");
        assert_eq!(
            model_folder_name("iic/speech_campplus_sv_zh-cn_16k-common"),
            "speech_campplus_sv_zh-cn_16k-common"
        );
        // Bare ids and multi-segment / trailing-separator ids.
        assert_eq!(model_folder_name("Model"), "Model");
        assert_eq!(model_folder_name("a/b/Model"), "Model");
        assert_eq!(model_folder_name("  plain  "), "plain");
        assert_eq!(model_folder_name("name."), "name");
        assert_eq!(model_folder_name(""), "model");
        assert_eq!(model_folder_name(".."), "model");
    }

    #[test]
    fn test_probe_wav_sample_rate() {
        if std::process::Command::new("ffmpeg")
            .args(["-version"])
            .output()
            .is_err()
        {
            // ffmpeg unavailable in this environment; nothing to verify against
            return;
        }
        let dir = std::env::temp_dir().join("liveneko_probe_test");
        std::fs::create_dir_all(&dir).unwrap();
        for rate in [16000u32, 48000] {
            let wav = dir.join(format!("rate{rate}.wav"));
            let status = std::process::Command::new("ffmpeg")
                .args([
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=440:duration=1",
                    "-ar",
                    &rate.to_string(),
                    "-ac",
                    "1",
                    wav.to_str().unwrap(),
                ])
                .status()
                .unwrap();
            assert!(status.success(), "ffmpeg failed to generate the test wav");
            assert_eq!(probe_wav_sample_rate(&wav).unwrap(), rate);
            let _ = std::fs::remove_file(&wav);
        }
        let _ = std::fs::remove_dir(&dir);
    }
}
