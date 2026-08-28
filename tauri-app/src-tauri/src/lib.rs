mod assets;
mod commands;
mod config;
mod df_denoise;
mod downloader;
mod model_ipc;
mod os_theme;
mod pipeline;

use std::path::PathBuf;
use tauri::Manager;

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            std::fs::create_dir_all(&app_data_dir).ok();
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
            commands::get_prompt,
            commands::reset_prompt,
            commands::get_setup_status,
            commands::get_queue,
            commands::add_url,
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
