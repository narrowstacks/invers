//! Application state management
//!
//! Central state for the GUI application, managing loaded images,
//! current settings, and processing state.

use invers_core::{BaseEstimation, FilmPreset, ScanProfile};
use std::path::PathBuf;

/// Main application state
#[derive(Debug, Clone)]
pub struct AppState {
    /// Currently loaded image path
    pub current_image: Option<PathBuf>,

    /// Current film preset
    pub film_preset: Option<FilmPreset>,

    /// Current scan profile
    pub scan_profile: Option<ScanProfile>,

    /// Current base estimation
    pub base_estimation: Option<BaseEstimation>,

    /// Selected ROI for base estimation (x, y, width, height)
    pub roi: Option<(u32, u32, u32, u32)>,

    /// Preview settings
    pub preview: PreviewSettings,

    /// Batch queue
    pub batch_queue: Vec<BatchItem>,

    /// Export settings
    pub export: ExportSettings,
}

/// Preview display settings
#[derive(Debug, Clone)]
pub struct PreviewSettings {
    /// Current zoom level (1.0 = 100%)
    pub zoom: f32,

    /// Pan offset (x, y)
    pub pan_offset: (f32, f32),

    /// Show histogram overlay
    pub show_histogram: bool,

    /// Show pixel probe
    pub show_pixel_probe: bool,

    /// Preview resolution scale (0.0-1.0)
    pub preview_scale: f32,
}

/// Export configuration
#[derive(Debug, Clone)]
pub struct ExportSettings {
    /// Output directory
    pub output_dir: PathBuf,

    /// Output format
    pub format: String,

    /// Working colorspace
    pub colorspace: String,

    /// Bit depth policy
    pub bit_depth_policy: String,
}

/// Item in the batch processing queue
#[derive(Debug, Clone)]
pub struct BatchItem {
    /// Input file path
    pub input: PathBuf,

    /// Processing status
    pub status: BatchStatus,

    /// Progress (0.0-1.0)
    pub progress: f32,
}

/// Status of a batch item
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchStatus {
    Pending,
    Processing,
    Completed,
    Failed,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            current_image: None,
            film_preset: None,
            scan_profile: None,
            base_estimation: None,
            roi: None,
            preview: PreviewSettings::default(),
            batch_queue: Vec::new(),
            export: ExportSettings::default(),
        }
    }
}

impl Default for PreviewSettings {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            pan_offset: (0.0, 0.0),
            show_histogram: true,
            show_pixel_probe: false,
            preview_scale: 0.25, // 25% for real-time preview
        }
    }
}

impl Default for ExportSettings {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            format: "tiff16".to_string(),
            colorspace: "linear-rec2020".to_string(),
            bit_depth_policy: "match-input".to_string(),
        }
    }
}
