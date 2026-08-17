use crate::assets::Assets;
use crate::config::AppConfig;
use crate::model_ipc::log_line;
use crate::pipeline::{self, ItemStatus, PipelineHandle, QueueItem, Runner};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
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
}

impl AppState {
    pub fn new(app_data_dir: PathBuf) -> Self {
        Self {
            config: Mutex::new(AppConfig::load(&app_data_dir)),
            queue: Mutex::new(Vec::new()),
            pipeline: PipelineHandle::default(),
            running: AtomicBool::new(false),
            migrated: AtomicBool::new(false),
            app_data_dir,
        }
    }
}

fn emit_app(app: &AppHandle, event: &str, payload: serde_json::Value) {
    let _ = app.emit(event, payload);
}

fn ensure_assets(app: &AppHandle) -> Result<Assets, String> {
    let assets = Assets::resolve(app);
    if !assets.yt_dlp_exe.exists() {
        return Err(format!(
            "yt-dlp.exe not found at {}",
            assets.yt_dlp_exe.display()
        ));
    }
    if !assets.audio_models_present() {
        return Err(format!(
            "bundled audio models missing under {}",
            assets.audio_model_dir.display()
        ));
    }
    if !assets.filter_model_present() {
        return Err(format!(
            "DeepFilterNet model missing under {}",
            assets.filter_model_dir.display()
        ));
    }
    if assets.spk_refs().is_empty() {
        return Err(format!(
            "no speaker reference media found in {}",
            assets.spk_dir.display()
        ));
    }
    Ok(assets)
}

// Commands
#[tauri::command]
pub fn check_environment(
    app: AppHandle,
    state: State<'_, AppState>,
    force: Option<bool>,
) -> Result<serde_json::Value, String> {
    // Environment check runs only once per machine unless forced (Re-check button). Persist the result so the second launch onwards skips the slow python/asset probing.
    let env_report_path = state.app_data_dir.join("env_report.json");
    let mut cfg = state.config.lock().unwrap().clone();
    if !force.unwrap_or(false) && cfg.env_checked && env_report_path.exists() {
        if let Ok(text) = std::fs::read_to_string(&env_report_path) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
                return Ok(parsed);
            }
        }
    }

    let report = run_environment_check(&app, &cfg);
    let _ = std::fs::write(&env_report_path, report.to_string());
    cfg.env_checked = true;
    let _ = cfg.save(&state.app_data_dir);
    *state.config.lock().unwrap() = cfg;
    Ok(report)
}

fn run_environment_check(app: &AppHandle, cfg: &AppConfig) -> serde_json::Value {
    let assets = Assets::resolve(app);

    // Python check
    let python_check = run_capture(&cfg.python_cmd, &["--version"]);
    let python_ok = python_check.is_ok();
    let python_version = python_check.unwrap_or_default();

    // ffmpeg check
    let ffmpeg_ok = run_capture("ffmpeg", &["-version"]).is_ok();

    // Python libraries check
    let mut libs = serde_json::json!({});
    let mut cuda = false;
    let mut python_libs_ok = false;
    if python_ok {
        let script = assets.scripts_dir.join("env_check.py");
        if script.exists() {
            if let Ok(out) =
                run_capture_cwd(&cfg.python_cmd, &["-u", script.to_str().unwrap()], None)
            {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&out) {
                    cuda = parsed
                        .get("cuda")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    // env_check.py emits {"cuda": .., "libraries": {..}}; the frontend expects the flat per-library object under "libraries", so unwrap it here.
                    libs = parsed.get("libraries").cloned().unwrap_or_default();
                    python_libs_ok = true;
                }
            }
        }
    }

    serde_json::json!({
        "python": { "command": cfg.python_cmd, "ok": python_ok, "version": python_version },
        "ffmpeg": ffmpeg_ok,
        "cuda": cuda,
        "pythonLibraries": python_libs_ok,
        "libraries": libs,
        "assets": {
            "ytDlp": assets.yt_dlp_exe.exists(),
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

#[tauri::command]
pub fn save_config(state: State<'_, AppState>, mut config: AppConfig) -> Result<(), String> {
    // preserve backend-managed fields that the settings UI does not send
    let existing = state.config.lock().unwrap().clone();
    config.env_checked = existing.env_checked;
    if config.python_cmd.trim().is_empty() {
        config.python_cmd = existing.python_cmd.clone();
    }
    config.normalize();
    config.save(&state.app_data_dir)?;
    *state.config.lock().unwrap() = config;
    Ok(())
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
    Ok(serde_json::json!({
        "firstLaunch": cfg.videoneko_model_dir.is_empty()
            && cfg.ollama_model.is_empty()
            && cfg.llamacpp_model.is_empty()
            && cfg.api_model.is_empty(),
        "needsVideoneko": !model_ok,
        "videonekoModelDir": cfg.videoneko_model_dir,
    }))
}

#[tauri::command]
pub fn get_queue(state: State<'_, AppState>) -> Vec<QueueItem> {
    state.queue.lock().unwrap().clone()
}

#[tauri::command]
pub async fn add_url(
    app: AppHandle,
    state: State<'_, AppState>,
    url: String,
    title: Option<String>,
) -> Result<QueueItem, String> {
    if !url.starts_with("http") {
        return Err("URL must start with http:// or https://".to_string());
    }
    // Use the caller-provided title, otherwise probe the real video title with yt-dlp so the queue shows it instead of a placeholder.
    let title = match title {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            let assets = crate::assets::Assets::resolve(&app);
            match crate::pipeline::probe_ytdlp_titles(&assets.yt_dlp_exe, &url) {
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
    let assets = ensure_assets(&app)?;
    let cfg = state.config.lock().unwrap().clone();
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

    // Open the run log file (work/pipeline.log); all pipeline log lines are
    // appended to it alongside being emitted to the UI.
    let work_root = state.app_data_dir.join("work");
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
    let work_root = work_root;

    std::thread::spawn(move || {
        let mut runner = Runner::new(app2.clone(), assets.clone(), handle.clone());
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
    let target = state.app_data_dir.join("results");
    std::fs::create_dir_all(&target).ok();

    let mut flat_dirs = vec![
        state.app_data_dir.join("work").join("results"),
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
    let results_dir = state.app_data_dir.join("results");
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
    let results_dir = state.app_data_dir.join("results").join(&stem);
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
    let results_dir = state.app_data_dir.join("results");
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
            snip.insert_str(0, "…");
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
    let result_dir = state.app_data_dir.join("results").join(&stem);
    let asr_path = result_dir.join("asr.txt");
    let visual_path = result_dir.join("visual.txt");
    if !asr_path.exists() {
        return Err(format!("stored asr.txt not found for '{stem}'"));
    }
    if !visual_path.exists() {
        return Err(format!("stored visual.txt not found for '{stem}'"));
    }
    let config = state.config.lock().unwrap().clone();
    let runner = Runner::new(app.clone(), assets, state.pipeline.clone());
    // mark as running so the UI hides the Start button / shows state
    state.pipeline.cancel.store(false, Ordering::SeqCst);
    let app2 = app.clone();
    let stem2 = stem.clone();
    // write into a per-stem work dir, then move outputs into results/<stem> so a companion thinking.txt doesn't collide across concurrent re-summaries
    let work_dir = state
        .app_data_dir
        .join("work")
        .join(format!("resummarize-{stem}"));
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
    let results_dir = state.app_data_dir.join("results").join(&stem);
    let src = results_dir.join("summary.md");
    std::fs::copy(&src, PathBuf::from(&dest)).map_err(|e| format!("copy: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn delete_result(state: State<'_, AppState>, stem: String) -> Result<(), String> {
    let dir = state.app_data_dir.join("results").join(&stem);
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
