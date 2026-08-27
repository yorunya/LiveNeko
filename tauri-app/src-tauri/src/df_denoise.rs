use df::tract::{DfParams, DfTract, RuntimeParams};
use df::transforms::resample;
use df::wav_utils::{ReadWav, write_wav_arr2};
use ndarray::{Array2, ArrayD, Axis};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const TARGET_SR: usize = 16000;
const ATTEN_LIM_DB: f32 = 18.0;

/// Owned DeepFilterNet denoiser backed by the Rust `df` crate.
///
/// Loads the ONNX model once and can process multiple input files. The model
/// is not `Sync`, so the caller must provide mutable access during inference.
pub struct Denoiser {
    model: DfTract,
}

impl Denoiser {
    /// Load a DeepFilterNet ONNX model from a gzipped tar archive.
    pub fn new(model_tar: &Path) -> Result<Self, String> {
        let dfp = DfParams::new(model_tar.to_path_buf())
            .map_err(|e| format!("load df model from {}: {e}", model_tar.display()))?;
        let rp = RuntimeParams::default_with_ch(1).with_atten_lim(ATTEN_LIM_DB);
        let model = DfTract::new(dfp, &rp)
            .map_err(|e| format!("init df tract model from {}: {e}", model_tar.display()))?;
        Ok(Self { model })
    }

    /// Denoise a 48 kHz (or any sample rate) mono WAV and write the enhanced
    /// audio to `output_wav` at 16 kHz.
    ///
    /// Progress is reported as 0..20 so the caller can reserve the 20..100
    /// range for downstream ASR. Returns `Err("cancelled")` if `cancel` is set.
    pub fn process_file<P: AsRef<Path>, Q: AsRef<Path>>(
        &mut self,
        input_wav: P,
        output_wav: Q,
        cancel: &Arc<AtomicBool>,
        progress: &dyn Fn(u8),
    ) -> Result<(), String> {
        let input_wav = input_wav.as_ref();
        let output_wav = output_wav.as_ref();
        let model_sr = self.model.sr;
        let hop = self.model.hop_size;

        let reader = ReadWav::new(input_wav.to_str().unwrap_or(""))
            .map_err(|e| format!("read input wav {}: {e}", input_wav.display()))?;
        let input_sr = reader.sr;
        let mut noisy = reader
            .samples_arr2()
            .map_err(|e| format!("read samples from {}: {e}", input_wav.display()))?;

        // Resample to the model's expected sample rate (48 kHz for DFN3).
        if input_sr != model_sr {
            noisy = resample(noisy.view(), input_sr, model_sr, None)
                .map_err(|e| format!("resample {input_sr} -> {model_sr}: {e}"))?;
        }

        let total_samples = noisy.len_of(Axis(1));
        let mut enh: Array2<f32> = ArrayD::default(noisy.shape())
            .into_dimensionality()
            .map_err(|e| format!("alloc enh buffer: {e}"))?;
        let mut processed = 0usize;

        for (ns_f, enh_f) in noisy
            .view()
            .axis_chunks_iter(Axis(1), hop)
            .zip(enh.view_mut().axis_chunks_iter_mut(Axis(1), hop))
        {
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled".to_string());
            }
            if ns_f.len_of(Axis(1)) < hop {
                // Leave the trailing partial hop unprocessed, matching the
                // upstream `deep-filter` CLI behavior.
                break;
            }
            self.model
                .process(ns_f, enh_f)
                .map_err(|e| format!("df process failed: {e}"))?;
            processed += hop.min(total_samples.saturating_sub(processed));
            let p = ((processed as f64 / total_samples.max(1) as f64) * 20.0) as u8;
            progress(p.min(20));
        }

        // Downsample the enhanced audio to 16 kHz for VAD/ASR/SPK.
        if cancel.load(Ordering::SeqCst) {
            return Err("cancelled".to_string());
        }
        let enh_16k = resample(enh.view(), model_sr, TARGET_SR, None)
            .map_err(|e| format!("resample {model_sr} -> {TARGET_SR}: {e}"))?;

        write_wav_arr2(
            output_wav.to_str().unwrap_or(""),
            enh_16k.view(),
            TARGET_SR as u32,
        )
        .map_err(|e| format!("write filtered wav {}: {e}", output_wav.display()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn deepfilternet_model_loads() {
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let tar = manifest
            .join("../../DeepFilterNet/models/DeepFilterNet3_onnx.tar.gz")
            .canonicalize()
            .expect("model tar not found");
        let _denoiser = Denoiser::new(&tar).expect("failed to load DeepFilterNet ONNX model");
    }
}
