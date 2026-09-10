use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct AppConfig {
    /// Path to the user's Visual model directory.
    pub videoneko_model_dir: String,
    /// Summarization engine: "api" | "ollama" | "llamacpp"
    pub engine: String,
    /// User-customized summary prompt. Empty means use the bundled prompt.md.
    pub custom_prompt: String,
    /// Download quality for yt-dlp: 360 | 480 | 720 | 1080 (default 720).
    pub download_quality: u32,
    /// Optional browser cookie import for downloads (yt-dlp
    /// --cookies-from-browser equivalent): "" | "firefox" | "chrome" | "edge".
    pub cookie_browser: String,
    /// Whether the environment check has already been run (persisted across launches).
    pub env_checked: bool,
    /// Speaker display name used in transcripts (e.g. "taffy"). Empty = no speaker identification.
    pub speaker_name: String,
    /// Reference WAV filename (16 kHz mono) stored under <app_data>/spk/. Empty = none.
    pub speaker_ref: String,
    /// UI language: "" (system default) | "en" | "zh".
    pub language: String,

    // ---- FunASR audio models (user-provided; never bundled) ----
    /// Base directory used for models downloaded from Hugging Face / ModelScope.
    pub funasr_models_dir: String,
    /// ASR backend: "sensevoice-small" | "fun-asr-nano" | "paraformer-zh-streaming" | "qwen3-api"
    pub asr_type: String,
    /// ASR model source for local backends: "local" | "huggingface" | "modelscope"
    pub asr_source: String,
    /// ASR repository id on the selected hub (used for downloads / display).
    pub asr_model_id: String,
    /// Local ASR model directory (either user-picked or a completed download).
    pub asr_model_dir: String,
    /// Optional ASR language hint ("" = model default; en/zh/auto/...).
    pub asr_language: String,
    /// VAD model source: "local" | "huggingface" | "modelscope" (fsmn-vad compatible).
    pub vad_source: String,
    pub vad_model_id: String,
    pub vad_model_dir: String,
    /// SPK model is optional: when disabled (or missing) speaker identification
    /// is disabled and every utterance is labelled "other".
    pub spk_enabled: bool,
    /// SPK model source: "local" | "huggingface" | "modelscope" (cam++ compatible).
    pub spk_source: String,
    pub spk_model_id: String,
    pub spk_model_dir: String,
    // Qwen3-ASR online API (used when asrType == "qwen3-api")
    pub qwen3_base_url: String,
    pub qwen3_api_key: String,
    pub qwen3_model: String,
    pub qwen3_language: String,

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

pub const ASR_TYPES: [&str; 4] = [
    "sensevoice-small",
    "fun-asr-nano",
    "paraformer-zh-streaming",
    "qwen3-api",
];

/// Default hub repository id for a model type + source. `source` is
/// "huggingface" or "modelscope"; anything else falls back to Hugging Face ids.
pub fn default_model_id(kind: &str, source: &str) -> String {
    let hf = source != "modelscope";
    let id = match kind {
        "sensevoice-small" => {
            if hf { "FunAudioLLM/SenseVoiceSmall" } else { "iic/SenseVoiceSmall" }
        }
        "fun-asr-nano" => "FunAudioLLM/Fun-ASR-Nano-2512",
        "paraformer-zh-streaming" => {
            if hf {
                "funasr/paraformer-zh-streaming"
            } else {
                "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"
            }
        }
        "fsmn-vad" => {
            if hf {
                "funasr/fsmn-vad"
            } else {
                "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            }
        }
        "cam++" => {
            if hf { "funasr/campplus" } else { "iic/speech_campplus_sv_zh-cn_16k-common" }
        }
        _ => "",
    };
    id.to_string()
}

fn valid_source(value: &str) -> bool {
    matches!(value, "local" | "huggingface" | "modelscope")
}

impl AppConfig {
    pub fn new() -> Self {
        Self {
            videoneko_model_dir: String::new(),
            engine: "api".to_string(),
            custom_prompt: String::new(),
            download_quality: 720,
            cookie_browser: String::new(),
            env_checked: false,
            speaker_name: String::new(),
            speaker_ref: String::new(),
            language: String::new(),
            funasr_models_dir: String::new(),
            asr_type: "sensevoice-small".to_string(),
            asr_source: "huggingface".to_string(),
            asr_model_id: default_model_id("sensevoice-small", "huggingface"),
            asr_model_dir: String::new(),
            asr_language: String::new(),
            vad_source: "huggingface".to_string(),
            vad_model_id: default_model_id("fsmn-vad", "huggingface"),
            vad_model_dir: String::new(),
            spk_enabled: false,
            spk_source: "huggingface".to_string(),
            spk_model_id: default_model_id("cam++", "huggingface"),
            spk_model_dir: String::new(),
            qwen3_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            qwen3_api_key: String::new(),
            qwen3_model: "qwen3-asr-flash".to_string(),
            qwen3_language: String::new(),
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

    /// True when the online Qwen3-ASR backend is selected.
    pub fn asr_is_api(&self) -> bool {
        self.asr_type == "qwen3-api"
    }

    /// True when the required FunASR fields are configured (no filesystem or
    /// network validation — see `validate_model_config` for that).
    pub fn funasr_configured(&self) -> bool {
        let asr_ok = if self.asr_is_api() {
            !self.qwen3_base_url.trim().is_empty() && !self.qwen3_model.trim().is_empty()
        } else {
            !self.asr_model_dir.trim().is_empty()
        };
        asr_ok && !self.vad_model_dir.trim().is_empty()
    }

    /// Normalize/migrate field values so they always hold valid defaults
    pub fn normalize(&mut self) {
        if self.engine.is_empty() || !matches!(self.engine.as_str(), "api" | "ollama" | "llamacpp")
        {
            self.engine = "api".to_string();
        }
        if !matches!(self.language.as_str(), "" | "en" | "zh") {
            self.language = String::new();
        }
        // Speaker: a name without a reference means no speaker identification;
        // the name is a free-form display label, so only trim and cap its length.
        self.speaker_name = self.speaker_name.trim().to_string();
        if self.speaker_name.chars().count() > 60 {
            self.speaker_name = self.speaker_name.chars().take(60).collect();
        }
        if self.speaker_name.is_empty() {
            self.speaker_ref.clear();
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
        if !matches!(self.cookie_browser.as_str(), "" | "firefox" | "chrome" | "edge") {
            self.cookie_browser.clear();
        }
        if self.ollama_base_url.is_empty() {
            self.ollama_base_url = "http://localhost:11434/v1".to_string();
        }
        if self.llamacpp_base_url.is_empty() {
            self.llamacpp_base_url = "http://localhost:8080/v1".to_string();
        }

        // FunASR audio models
        if !ASR_TYPES.contains(&self.asr_type.as_str()) {
            self.asr_type = "sensevoice-small".to_string();
        }
        if !valid_source(&self.asr_source) {
            self.asr_source = "huggingface".to_string();
        }
        if !valid_source(&self.vad_source) {
            self.vad_source = "huggingface".to_string();
        }
        if !valid_source(&self.spk_source) {
            self.spk_source = "huggingface".to_string();
        }
        if self.asr_model_id.trim().is_empty() {
            self.asr_model_id = default_model_id(&self.asr_type, &self.asr_source);
        }
        if self.vad_model_id.trim().is_empty() {
            self.vad_model_id = default_model_id("fsmn-vad", &self.vad_source);
        }
        if self.spk_model_id.trim().is_empty() {
            self.spk_model_id = default_model_id("cam++", &self.spk_source);
        }
        self.asr_model_dir = self.asr_model_dir.trim().to_string();
        self.vad_model_dir = self.vad_model_dir.trim().to_string();
        self.spk_model_dir = self.spk_model_dir.trim().to_string();
        self.asr_language = self.asr_language.trim().to_string();
        self.funasr_models_dir = self.funasr_models_dir.trim().to_string();
        if self.qwen3_base_url.trim().is_empty() {
            self.qwen3_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string();
        }
        if self.qwen3_model.trim().is_empty() {
            self.qwen3_model = "qwen3-asr-flash".to_string();
        }
        // Without a SPK model directory there is nothing to load; keep the
        // switch off so the pipeline never tries.
        if self.spk_model_dir.is_empty() {
            self.spk_enabled = false;
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
