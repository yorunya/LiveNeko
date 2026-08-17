use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Resolved locations of all bundled assets. During development these fall back to the repo root; in production they resolve from the resource directory.
#[derive(Clone, Debug)]
pub struct Assets {
    pub yt_dlp_exe: PathBuf,
    pub prompt_md: PathBuf,
    pub scripts_dir: PathBuf,
    pub audio_model_dir: PathBuf,
    pub filter_model_dir: PathBuf,
    pub spk_dir: PathBuf,
}

impl Assets {
    /// Resolve each asset independently: prefer the resource dir (production), falling back to the repo root for that specific asset when the resource dir copy is missing/stale.
    pub fn resolve(app: &AppHandle) -> Self {
        let resource = app
            .path()
            .resource_dir()
            .unwrap_or_else(|_| PathBuf::from("."));
        let repo_root = std::env::var("CARGO_MANIFEST_DIR")
            .ok()
            .and_then(|manifest| {
                Path::new(&manifest)
                    .parent()
                    .and_then(|p| p.parent())
                    .map(|p| p.to_path_buf())
            })
            .unwrap_or_default();

        let pick = |rel: &str| -> PathBuf {
            let res = resource.join(rel);
            if res.exists() {
                return res;
            }
            if !repo_root.as_os_str().is_empty() {
                let rr = repo_root.join(rel);
                if rr.exists() {
                    return rr;
                }
            }
            res
        };

        Self {
            yt_dlp_exe: pick("yt-dlp.exe"),
            prompt_md: pick("prompt.md"),
            scripts_dir: pick("scripts"),
            audio_model_dir: pick("model"),
            filter_model_dir: pick("model/DeepFilterNet3"),
            spk_dir: pick("spk"),
        }
    }

    pub fn audio_models_present(&self) -> bool {
        self.audio_model_dir.join("SenseVoiceSmall").exists()
            && self.audio_model_dir.join("fsmn-vad").exists()
            && self.audio_model_dir.join("cam++").exists()
    }

    pub fn filter_model_present(&self) -> bool {
        self.filter_model_dir.join("config.ini").exists()
    }

    pub fn spk_refs(&self) -> Vec<PathBuf> {
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&self.spk_dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                let ext = p
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_lowercase();
                if matches!(
                    ext.as_str(),
                    "mp4" | "mkv" | "mov" | "webm" | "avi" | "wav" | "mp3" | "m4a" | "flac" | "ogg"
                ) {
                    out.push(p);
                }
            }
        }
        out
    }
}
