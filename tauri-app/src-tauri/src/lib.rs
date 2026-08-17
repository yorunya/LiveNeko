mod assets;
mod commands;
mod config;
mod model_ipc;
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
            app.manage(commands::AppState::new(app_data_dir));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::check_environment,
            commands::get_config,
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
