//! Data models for Positize
//!
//! Core data structures for film presets, scan profiles, and processing options.

use serde::{Deserialize, Serialize};

/// Film preset containing film-specific conversion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilmPreset {
    /// Name of the film (e.g., "Kodak Portra 400")
    pub name: String,

    /// Per-channel base offsets (R, G, B)
    pub base_offsets: [f32; 3],

    /// 3x3 color correction matrix
    pub color_matrix: [[f32; 3]; 3],

    /// Tone curve parameters
    pub tone_curve: ToneCurveParams,

    /// Optional notes or description
    pub notes: Option<String>,
}

/// Tone curve parameters for positive conversion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToneCurveParams {
    /// Curve type (e.g., "neutral", "s-curve", "linear")
    pub curve_type: String,

    /// Curve strength/intensity (0.0 - 1.0)
    pub strength: f32,

    /// Additional curve-specific parameters
    #[serde(default)]
    pub params: std::collections::HashMap<String, f32>,
}

/// Scan profile defining capture source characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScanProfile {
    /// Name of the scan profile
    pub name: String,

    /// Capture source type (e.g., "dslr", "mirrorless", "flatbed")
    pub source_type: String,

    /// Typical white level for this source
    pub white_level: Option<f32>,

    /// Typical black level for this source
    pub black_level: Option<f32>,

    /// Demosaic hints (for RAW sources)
    pub demosaic_hints: Option<DemosaicHints>,

    /// White balance hints
    pub white_balance_hints: Option<WhiteBalanceHints>,
}

/// Demosaic processing hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemosaicHints {
    /// Demosaic algorithm preference
    pub algorithm: String,

    /// Quality vs speed preference (0.0 = fast, 1.0 = quality)
    pub quality: f32,
}

/// White balance processing hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhiteBalanceHints {
    /// Auto white balance preference
    pub auto: bool,

    /// Manual color temperature (if not auto)
    pub temperature: Option<f32>,

    /// Tint adjustment
    pub tint: Option<f32>,
}

/// Film base estimation from ROI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseEstimation {
    /// Region of interest (x, y, width, height)
    pub roi: Option<(u32, u32, u32, u32)>,

    /// Per-channel median values (R, G, B)
    pub medians: [f32; 3],

    /// Per-channel noise statistics (standard deviation)
    pub noise_stats: Option<[f32; 3]>,

    /// Whether this was auto-estimated or manual
    pub auto_estimated: bool,
}

/// Conversion options for the processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertOptions {
    /// Input file path(s)
    pub input_paths: Vec<std::path::PathBuf>,

    /// Output directory
    pub output_dir: std::path::PathBuf,

    /// Output format ("tiff" or "dng")
    pub output_format: OutputFormat,

    /// Working colorspace
    pub working_colorspace: String,

    /// Bit depth policy
    pub bit_depth_policy: BitDepthPolicy,

    /// Film preset to apply
    pub film_preset: Option<FilmPreset>,

    /// Scan profile to use
    pub scan_profile: Option<ScanProfile>,

    /// Base estimation (if pre-computed)
    pub base_estimation: Option<BaseEstimation>,

    /// Number of parallel threads for batch processing
    pub num_threads: Option<usize>,

    /// Skip tone curve application
    pub skip_tone_curve: bool,

    /// Skip color matrix correction
    pub skip_color_matrix: bool,

    /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
    pub exposure_compensation: f32,

    /// Enable debug output
    pub debug: bool,
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// 16-bit linear TIFF
    Tiff16,

    /// Linear DNG
    LinearDng,
}

/// Bit depth handling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitDepthPolicy {
    /// Match input bit depth when possible
    MatchInput,

    /// Always use 16-bit output
    Force16Bit,

    /// Preserve maximum precision
    MaxPrecision,
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            curve_type: "neutral".to_string(),
            strength: 0.5,
            params: std::collections::HashMap::new(),
        }
    }
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Tiff16
    }
}

impl Default for BitDepthPolicy {
    fn default() -> Self {
        Self::MatchInput
    }
}
