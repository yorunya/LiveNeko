mod assets;
mod commands;
mod config;
mod cookies;
mod downloader;
mod model_ipc;
mod os_theme;
mod pipeline;
mod silero_vad;

use std::path::PathBuf;
use tauri::Manager;

/// Log every panic to %APPDATA%\com.liveneko.desktop\panics.log
fn install_panic_log(app_data_dir: &std::path::Path) {
    let log_path = app_data_dir.join("panics.log");
    std::panic::set_hook(Box::new(move |info| {
        let payload = info
            .payload()
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| info.payload().downcast_ref::<String>().map(|s| s.as_str()))
            .unwrap_or("<unknown payload>");
        let location = info
            .location()
            .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()))
            .unwrap_or_else(|| "<unknown>".to_string());
        let thread = std::thread::current();
        let thread_name = thread.name().unwrap_or("<unnamed>");
        let backtrace = std::backtrace::Backtrace::force_capture();
        let entry = format!(
            "[{}] thread '{thread_name}' panicked at {location}:\n{payload}\nbacktrace:\n{backtrace}\n---\n",
            chrono_timestamp(),
        );
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
        {
            let _ = f.write_all(entry.as_bytes());
        }
        // Keep the default stderr behavior too.
        eprintln!("thread '{thread_name}' panicked at {location}:\n{payload}");
    }));
}

fn chrono_timestamp() -> String {
    // Minimal RFC3339-ish timestamp without pulling in a time crate.
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let secs_today = secs % 86400;
    let h = (secs_today / 3600) as u32;
    let m = (secs_today % 3600 / 60) as u32;
    let s = (secs_today % 60) as u32;
    let (mut y, mut d) = (1970i64, secs / 86400);
    // days -> y/m/d (UTC)
    loop {
        let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
        let days = if leap { 366 } else { 365 };
        if d >= days {
            d -= days;
            y += 1;
        } else {
            break;
        }
    }
    const DIM: [u64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let mut month = 1u32;
    for (i, &dm) in DIM.iter().enumerate() {
        let dm = if i == 1 && leap { 29 } else { dm };
        if d >= dm {
            d -= dm;
            month += 1;
        } else {
            break;
        }
    }
    format!("{y:04}-{month:02}-{:02}T{h:02}:{m:02}:{s:02}Z", d + 1)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            std::fs::create_dir_all(&app_data_dir).ok();
            install_panic_log(&app_data_dir);
            // Detect the OS theme + accent color exactly once, here at startup,
            // via the official Tauri API and platform methods (see os_theme.rs).
            // No theme-change listener is registered anywhere; the app keeps
            // these initial values for its whole lifetime.
            let os_theme = os_theme::detect_theme(app).to_string();
            let os_accent = os_theme::detect_accent_color();
            app.manage(commands::AppState::new(app_data_dir, os_theme, os_accent));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::check_environment,
            commands::get_config,
            commands::get_os_theme,
            commands::save_config,
            commands::validate_model_config,
            commands::download_model,
            commands::test_asr_connection,
            commands::get_prompt,
            commands::reset_prompt,
            commands::get_setup_status,
            commands::get_queue,
            commands::add_url,
            commands::list_url_videos,
            commands::add_local_file,
            commands::remove_item,
            commands::clear_queue,
            commands::is_running,
            commands::start_pipeline,
            commands::stop_pipeline,
            commands::list_results,
            commands::read_result,
            commands::search_results,
            commands::re_summarize,
            commands::export_result,
            commands::delete_result,
            commands::test_api_connection,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
