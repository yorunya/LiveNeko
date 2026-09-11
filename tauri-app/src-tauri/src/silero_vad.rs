//! Native voice-activity detection with Silero VAD (ONNX, CPU-only), replacing
//! the FunASR `fsmn-vad` Python model the app previously required as a
//! user-provided download.
//!
//! The model is the ~2.3 MB `silero_vad.onnx` bundled with the app and run
//! through the `ort` crate (ONNX Runtime, CPU execution provider). Inputs are
//! `[x (1×(64+512) f32), state (2×1×128 f32), sr (1×i64)]`, outputs
//! `[output (1×1 speech probability), stateN]`, one 512-sample (32 ms) window
//! at a time; 64 samples of context prefix the window and the recurrent
//! `state` tensor is carried across windows. The segmentation state machine
//! mirrors silero-vad's canonical `utils_vad.get_speech_timestamps`.
//!
//! # Behavior compared to fsmn-vad (measured, see README)
//!
//! * Segmentation granularity: fsmn-vad closes an utterance after ~800 ms of
//!   trailing silence (FunASR `max_end_silence_time`); silero uses an explicit
//!   `MIN_SILENCE_MS` (300 ms here, tuned so both models produced the same
//!   segment count on a 3-minute denoised stream recording).
//! * Near-silence: fsmn-vad emits segments on very low-level residual audio
//!   (post-denoise RMS ≈ 0.002); ASR then transcribes them as empty `。`
//!   lines. Silero's probability threshold rejects that residue, so
//!   transcripts are cleaner, at the cost of skipping faint far-field speech
//!   (audible but ≈20× quieter than normal speech).
//! * Timestamps: fsmn-vad quantizes to 10 ms frames, silero to 32 ms windows;
//!   both are reported as integer milliseconds, so the IPC protocol and the
//!   `[hh:mm:ss-hh:mm:ss]` output format are unchanged.

use ndarray::{Array1, Array2, ArrayD};
use ort::session::Session;
use ort::value::Value;
use std::mem::take;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// Speech probability threshold (silero default).
const THRESHOLD: f32 = 0.5;
/// A triggered segment closes after this much silence (fsmn-vad's FunASR
/// default of 800 ms is noticeably coarser; 300 ms produced identical segment
/// counts on reference material).
const MIN_SILENCE_MS: i64 = 300;
/// Shorter candidate segments are discarded (silero default).
const MIN_SPEECH_MS: i64 = 250;
/// Segments longer than this are split at the last silence gap, like
/// fsmn-vad's `max_single_segment_time` (60 s) — also keeps any single ASR
/// chunk bounded.
const MAX_SPEECH_MS: i64 = 60_000;
/// Extend every reported segment by this much on each side (silero default).
const SPEECH_PAD_MS: i64 = 30;
/// 512 new samples per window at 16 kHz (32 ms), prefixed by 64 context
/// samples (silero v5/v6 canonical 576-sample input).
const WINDOW_SAMPLES: usize = 512;
const CONTEXT_SAMPLES: usize = 64;
const SAMPLE_RATE: usize = 16000;
/// Probability hysteresis below the threshold before a window counts as
/// silence (silero reference behavior).
const HYSTERESIS: f32 = 0.15;
/// When tracking the last gap for max-speech splits, only gaps longer than
/// this count (silero's `min_silence_samples_at_max_speech`).
const SILENCE_AT_MAX_MS: i64 = 98;

/// A detected utterance, in milliseconds (start-inclusive, end-exclusive,
/// matching the app's `[hh:mm:ss-hh:mm:ss]` span syntax).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
}

/// Native Silero VAD. Not `Sync`; create one per thread (model load is a few
/// milliseconds, so a per-file instance is fine).
pub struct SileroVad {
    session: Session,
    state: ArrayD<f32>,
    context: Array1<f32>,
    sample_rate: Array1<i64>,
}

impl SileroVad {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        let session = Session::builder()
            .map_err(|e| format!("ORT session builder: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("load silero_vad.onnx from {}: {e}", model_path.display()))?;
        Ok(Self {
            session,
            state: ArrayD::zeros(vec![2, 1, 128]),
            context: Array1::zeros(CONTEXT_SAMPLES),
            sample_rate: Array1::from_vec(vec![SAMPLE_RATE as i64]),
        })
    }

    fn reset(&mut self) {
        self.state = ArrayD::zeros(vec![2, 1, 128]);
        self.context = Array1::zeros(CONTEXT_SAMPLES);
    }

    /// Speech probability for one 512-sample window, advancing the recurrent
    /// state and the rolling context.
    fn window_probability(&mut self, window: &[f32]) -> Result<f32, String> {
        let mut input = Vec::with_capacity(CONTEXT_SAMPLES + window.len());
        input.extend_from_slice(self.context.as_slice().unwrap());
        input.extend_from_slice(window);

        let frame = Array2::from_shape_vec([1, input.len()], input)
            .map_err(|e| format!("shape vad frame: {e}"))?;
        let frame_value = Value::from_array(frame).map_err(|e| e.to_string())?;
        let state_value = Value::from_array(take(&mut self.state)).map_err(|e| e.to_string())?;
        let sr_value = Value::from_array(self.sample_rate.clone()).map_err(|e| e.to_string())?;
        let res = self
            .session
            .run([
                (&frame_value).into(),
                (&state_value).into(),
                (&sr_value).into(),
            ])
            .map_err(|e| format!("silero inference: {e}"))?;

        let (shape, state_data) = res["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?;
        let shape: Vec<usize> = shape.as_ref().iter().map(|&d| d as usize).collect();
        self.state = ArrayD::from_shape_vec(shape, state_data.to_vec())
            .map_err(|e| format!("reshape vad state: {e}"))?;
        self.context = Array1::from_vec(window[window.len() - CONTEXT_SAMPLES..].to_vec());

        let prob = *res["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?
            .1
            .first()
            .unwrap();
        Ok(prob)
    }

    /// Detect speech in normalized (`-1.0..1.0`) mono 16 kHz samples.
    pub fn detect(&mut self, samples: &[f32], cancel: &AtomicBool) -> Result<Vec<Segment>, String> {
        self.reset();
        let mut segments: Vec<Segment> = Vec::new();
        let mut triggered = false;
        let mut start_ms = 0i64;
        let mut temp_end_ms = 0i64;
        let mut prev_end_ms = 0i64;
        let mut next_start_ms = 0i64;
        let mut current_ms = 0i64;
        let window_ms = (WINDOW_SAMPLES * 1000 / SAMPLE_RATE) as i64;

        for chunk in samples.chunks(WINDOW_SAMPLES) {
            if chunk.len() < WINDOW_SAMPLES {
                break; // silero needs full windows; drop the trailing partial one
            }
            if cancel.load(Ordering::SeqCst) {
                return Err("cancelled".to_string());
            }
            let prob = self.window_probability(chunk)?;
            current_ms += window_ms;

            if prob > THRESHOLD {
                if temp_end_ms != 0 {
                    temp_end_ms = 0;
                    if next_start_ms < prev_end_ms {
                        next_start_ms = current_ms - window_ms;
                    }
                }
                if !triggered {
                    triggered = true;
                    start_ms = current_ms - window_ms;
                }
                continue;
            }

            if triggered && (current_ms - start_ms) > MAX_SPEECH_MS {
                // Split long speech at the last eligible silence gap.
                if prev_end_ms > 0 {
                    segments.push(Segment { start_ms, end_ms: prev_end_ms });
                    if next_start_ms < prev_end_ms {
                        triggered = false;
                    } else {
                        start_ms = next_start_ms;
                    }
                } else {
                    segments.push(Segment { start_ms, end_ms: current_ms });
                    triggered = false;
                }
                prev_end_ms = 0;
                next_start_ms = 0;
                temp_end_ms = 0;
                continue;
            }

            if triggered && prob < THRESHOLD - HYSTERESIS {
                if temp_end_ms == 0 {
                    temp_end_ms = current_ms;
                }
                if current_ms - temp_end_ms > SILENCE_AT_MAX_MS {
                    prev_end_ms = temp_end_ms;
                }
                if current_ms - temp_end_ms >= MIN_SILENCE_MS {
                    // Close the segment; drop candidates shorter than
                    // MIN_SPEECH (silero's get_speech_timestamps behavior).
                    let end_ms = temp_end_ms;
                    if end_ms - start_ms > MIN_SPEECH_MS {
                        segments.push(Segment { start_ms, end_ms });
                    }
                    prev_end_ms = 0;
                    next_start_ms = 0;
                    temp_end_ms = 0;
                    triggered = false;
                }
            }
        }

        // Trailing speech: close the last open segment at the end of the
        // audio (min-speech is enforced here too so a stray trailing blip
        // cannot become a junk ASR line).
        if triggered && current_ms - start_ms > MIN_SPEECH_MS {
            segments.push(Segment { start_ms, end_ms: current_ms });
        }

        // Apply the speech padding and clamp to the audio bounds.
        let total_ms = (samples.len() as i64) * 1000 / SAMPLE_RATE as i64;
        Ok(segments
            .into_iter()
            .map(|s| Segment {
                start_ms: (s.start_ms - SPEECH_PAD_MS).max(0),
                end_ms: (s.end_ms + SPEECH_PAD_MS).min(total_ms),
            })
            .filter(|s| s.end_ms > s.start_ms)
            .collect())
    }
}

/// Read a mono 16 kHz PCM WAV (the ffmpeg extraction output format) as
/// normalized samples and detect speech in it.
pub fn detect_wav_segments(
    vad: &mut SileroVad,
    wav: &Path,
    cancel: &AtomicBool,
) -> Result<Vec<Segment>, String> {
    let mut reader = hound::WavReader::new(
        std::fs::File::open(wav).map_err(|e| format!("open wav {}: {e}", wav.display()))?,
    )
    .map_err(|e| format!("read wav header {}: {e}", wav.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE as u32 {
        return Err(format!(
            "unexpected sample rate {} in {} (expected {SAMPLE_RATE})",
            spec.sample_rate,
            wav.display()
        ));
    }
    if spec.channels != 1 {
        return Err(format!(
            "unexpected {}-channel wav {} (expected mono)",
            spec.channels,
            wav.display()
        ));
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| Ok(s? as f32 / 32767.0))
            .collect::<Result<_, _>>()
            .map_err(|e: hound::Error| format!("decode wav {}: {e}", wav.display()))?,
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .map_err(|e: hound::Error| format!("decode wav {}: {e}", wav.display()))?,
    };
    vad.detect(&samples, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/silero_vad.onnx")
    }

    #[test]
    fn silence_and_empty_input_yield_no_segments() {
        let cancel = AtomicBool::new(false);
        let mut vad = SileroVad::new(&model()).expect("load silero model");
        let silence = vec![0f32; SAMPLE_RATE * 3];
        let segs = vad.detect(&silence, &cancel).expect("detect");
        assert!(segs.is_empty(), "expected no segments in silence, got {segs:?}");
        let segs = vad.detect(&[], &cancel).expect("detect");
        assert!(segs.is_empty());
    }
}
