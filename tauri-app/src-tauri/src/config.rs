use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct AppConfig {
    /// Python interpreter used to run the worker scripts ("python", "py", full path...).
    pub python_cmd: String,
    /// Path to the user's Visual model directory.
    pub videoneko_model_dir: String,
    /// Summarization engine: "api" | "ollama" | "llamacpp"
    pub engine: String,
    /// User-customized summary prompt. Empty means use the bundled prompt.md.
    pub custom_prompt: String,
    /// Download quality for yt-dlp: 360 | 480 | 720 | 1080 (default 720).
    pub download_quality: u32,
    /// Whether the environment check has already been run (persisted across launches).
    pub env_checked: bool,
    /// UI language: "" (system default) | "en" | "zh".
    pub language: String,
    // OpenAI-compatible API
    pub api_base_url: String,
    pub api_key: String,
    pub api_model: String,
    pub api_max_tokens: u32,
    pub api_temperature: f32,
    pub api_top_p: f32,
    /// Enable thinking/reasoning for the API engine (default off).
    pub api_thinking: bool,
    // Ollama (OpenAI-compatible /v1 endpoint)
    pub ollama_base_url: String,
    pub ollama_model: String,
    /// Enable thinking/reasoning for the Ollama engine (default off).
    pub ollama_thinking: bool,
    // llama.cpp server (OpenAI-compatible /v1 endpoint)
    pub llamacpp_base_url: String,
    pub llamacpp_model: String,
    /// Enable thinking/reasoning for the llama.cpp engine (default off).
    pub llamacpp_thinking: bool,
}

impl AppConfig {
    pub fn new() -> Self {
        Self {
            python_cmd: "python".to_string(),
            videoneko_model_dir: String::new(),
            engine: "api".to_string(),
            custom_prompt: String::new(),
            download_quality: 720,
            env_checked: false,
            language: String::new(),
            api_base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            api_model: "gpt-4o".to_string(),
            api_max_tokens: 8192,
            api_temperature: 0.4,
            api_top_p: 1.0,
            api_thinking: false,
            ollama_base_url: "http://localhost:11434/v1".to_string(),
            ollama_model: String::new(),
            ollama_thinking: false,
            llamacpp_base_url: "http://localhost:8080/v1".to_string(),
            llamacpp_model: String::new(),
            llamacpp_thinking: false,
        }
    }

    pub fn config_file_path(app_data_dir: &std::path::Path) -> PathBuf {
        app_data_dir.join("config.json")
    }

    /// Normalize/migrate field values so they always hold valid defaults
    /// (covers configs saved before new fields were added).
    pub fn normalize(&mut self) {
        if self.engine.is_empty() || !matches!(self.engine.as_str(), "api" | "ollama" | "llamacpp")
        {
            self.engine = "api".to_string();
        }
        if !matches!(self.language.as_str(), "" | "en" | "zh") {
            self.language = String::new();
        }
        if self.api_temperature == 0.0 {
            self.api_temperature = 0.4;
        }
        if self.api_top_p == 0.0 {
            self.api_top_p = 1.0;
        }
        if !matches!(self.download_quality, 360 | 480 | 720 | 1080) {
            self.download_quality = 720;
        }
        if self.ollama_base_url.is_empty() {
            self.ollama_base_url = "http://localhost:11434/v1".to_string();
        }
        if self.llamacpp_base_url.is_empty() {
            self.llamacpp_base_url = "http://localhost:8080/v1".to_string();
        }
    }

    pub fn load(app_data_dir: &std::path::Path) -> Self {
        let path = Self::config_file_path(app_data_dir);
        let mut cfg = if let Ok(text) = std::fs::read_to_string(&path) {
            serde_json::from_str(&text).unwrap_or_else(|_| Self::new())
        } else {
            Self::new()
        };
        cfg.normalize();
        cfg
    }

    pub fn save(&self, app_data_dir: &std::path::Path) -> Result<(), String> {
        std::fs::create_dir_all(app_data_dir).map_err(|e| e.to_string())?;
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(Self::config_file_path(app_data_dir), text).map_err(|e| e.to_string())
    }
}
