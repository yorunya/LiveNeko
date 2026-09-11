use std::path::{Path, PathBuf};
use tauri::{AppHandle, Manager};

/// Resolved locations of all bundled assets. During development these fall back to the repo tree; in production they resolve from the resource directory. FunASR ASR/SPK models are NOT bundled — the user provides them (see `AppConfig`); the Silero VAD ONNX IS bundled.
#[derive(Clone, Debug)]
pub struct Assets {
    pub prompt_md: PathBuf,
    pub scripts_dir: PathBuf,
    /// Bundled Silero VAD ONNX model (see `crate::silero_vad`).
    pub silero_model: PathBuf,
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

        // The Silero VAD ONNX ships inside the app bundle (src-tauri/models/ in
        // the repo tree, model/ in the resource dir).
        let silero_model = {
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
            let candidates = [
                resource.join("model/silero_vad.onnx"),
                PathBuf::from(manifest_dir).join("models/silero_vad.onnx"),
            ];
            candidates
                .iter()
                .find(|p| p.exists())
                .cloned()
                .unwrap_or_else(|| candidates[0].clone())
        };

        Self {
            prompt_md: pick("prompt.md"),
            scripts_dir: pick("scripts"),
            silero_model,
        }
    }

    /// Path to the Python model helper used for validation and downloads.
    pub fn model_tools_script(&self) -> PathBuf {
        self.scripts_dir.join("model_tools.py")
    }
}
