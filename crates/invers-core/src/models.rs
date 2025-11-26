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
    /// Curve type (e.g., "neutral", "s-curve", "linear", "asymmetric")
    pub curve_type: String,

    /// Overall curve strength/intensity (0.0 - 1.0)
    /// For "asymmetric" curve type, this is a blend factor with linear
    pub strength: f32,

    /// Shadow (toe) curve strength (0.0 - 1.0)
    /// Higher values lift shadows more aggressively
    #[serde(default = "default_toe_strength")]
    pub toe_strength: f32,

    /// Highlight (shoulder) curve strength (0.0 - 1.0)
    /// Higher values compress highlights more aggressively
    #[serde(default = "default_shoulder_strength")]
    pub shoulder_strength: f32,

    /// Where shadows transition to midtones (0.0 - 0.5)
    /// Default 0.25 means toe region extends to 25% brightness
    #[serde(default = "default_toe_length")]
    pub toe_length: f32,

    /// Where midtones transition to highlights (0.5 - 1.0)
    /// Default 0.75 means shoulder region starts at 75% brightness
    #[serde(default = "default_shoulder_start")]
    pub shoulder_start: f32,

    /// Additional curve-specific parameters
    #[serde(default)]
    pub params: std::collections::HashMap<String, f32>,
}

fn default_toe_strength() -> f32 {
    0.4
}

fn default_shoulder_strength() -> f32 {
    0.3
}

fn default_toe_length() -> f32 {
    0.25
}

fn default_shoulder_start() -> f32 {
    0.75
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

    // New auto-adjustment options
    /// Enable auto-levels (histogram stretching per channel)
    #[serde(default = "default_true")]
    pub enable_auto_levels: bool,

    /// Clipping percentage for auto-levels (0.0-10.0)
    #[serde(default = "default_clip_percent")]
    pub auto_levels_clip_percent: f32,

    /// Enable auto-color (neutralize color casts)
    /// Note: Usually unnecessary when auto-levels is enabled
    #[serde(default = "default_false")]
    pub enable_auto_color: bool,

    /// Auto-color correction strength (0.0-1.0)
    #[serde(default = "default_auto_color_strength")]
    pub auto_color_strength: f32,

    /// Minimum multiplier auto-color will apply to any channel
    #[serde(default = "default_auto_color_min_gain")]
    pub auto_color_min_gain: f32,

    /// Maximum multiplier auto-color will apply to any channel
    #[serde(default = "default_auto_color_max_gain")]
    pub auto_color_max_gain: f32,

    /// Base estimation brightest pixel percentage (1.0-30.0)
    #[serde(default = "default_base_brightest_percent")]
    pub base_brightest_percent: f32,

    /// Base estimation sampling mode
    #[serde(default)]
    pub base_sampling_mode: BaseSamplingMode,

    /// Inversion mode (linear or logarithmic)
    #[serde(default)]
    pub inversion_mode: InversionMode,

    /// Shadow lift mode
    #[serde(default)]
    pub shadow_lift_mode: ShadowLiftMode,

    /// Shadow lift target black point (0.0-0.1)
    #[serde(default = "default_shadow_lift_value")]
    pub shadow_lift_value: f32,

    /// Highlight compression factor (0.0-1.0, 1.0 = no compression)
    #[serde(default = "default_one")]
    pub highlight_compression: f32,

    /// Enable automatic exposure normalization based on scene median
    #[serde(default = "default_true")]
    pub enable_auto_exposure: bool,

    /// Target median luminance for auto exposure (0.0-1.0)
    #[serde(default = "default_auto_exposure_target")]
    pub auto_exposure_target_median: f32,

    /// Strength of auto exposure adjustment (0.0-1.0)
    #[serde(default = "default_auto_exposure_strength")]
    pub auto_exposure_strength: f32,

    /// Minimum gain applied by auto exposure (prevents over-darkening)
    #[serde(default = "default_auto_exposure_min_gain")]
    pub auto_exposure_min_gain: f32,

    /// Maximum gain applied by auto exposure (prevents over-brightening)
    #[serde(default = "default_auto_exposure_max_gain")]
    pub auto_exposure_max_gain: f32,
}

// Default value functions for serde
fn default_true() -> bool {
    true
}

fn default_false() -> bool {
    false
}

fn default_clip_percent() -> f32 {
    0.25
}

fn default_auto_color_strength() -> f32 {
    0.6
}

fn default_auto_color_min_gain() -> f32 {
    0.7
}

fn default_auto_color_max_gain() -> f32 {
    1.3
}

fn default_base_brightest_percent() -> f32 {
    5.0
}

fn default_shadow_lift_value() -> f32 {
    0.02
}

fn default_one() -> f32 {
    1.0
}

fn default_auto_exposure_target() -> f32 {
    0.25
}

fn default_auto_exposure_strength() -> f32 {
    1.0
}

fn default_auto_exposure_min_gain() -> f32 {
    0.6
}

fn default_auto_exposure_max_gain() -> f32 {
    1.4
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    /// 16-bit linear TIFF
    #[default]
    Tiff16,

    /// Linear DNG
    LinearDng,
}

/// Bit depth handling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BitDepthPolicy {
    /// Match input bit depth when possible
    #[default]
    MatchInput,

    /// Always use 16-bit output
    Force16Bit,

    /// Preserve maximum precision
    MaxPrecision,
}

/// Base estimation sampling mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BaseSamplingMode {
    /// Use median of brightest pixels (default, robust)
    #[default]
    Median,

    /// Use mean of brightest pixels (more sensitive to maximum)
    Mean,

    /// Use maximum values (most aggressive)
    Max,
}

/// Method for automatic base estimation region selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BaseEstimationMethod {
    /// Sample discrete border regions (top, bottom, left, right) individually
    #[default]
    Regions,

    /// Sample outer N% of entire image as continuous border frame
    Border,
}

/// Inversion mode for negative-to-positive conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InversionMode {
    /// Linear inversion: (base - negative) / base
    #[default]
    Linear,

    /// Logarithmic inversion: 10^(log10(base) - log10(negative))
    Logarithmic,
}

/// Shadow lift mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShadowLiftMode {
    /// Fixed lift value
    Fixed,

    /// Percentile-based adaptive lift (e.g., lift 1st percentile to target)
    #[default]
    Percentile,

    /// No shadow lift
    None,
}

impl Default for ToneCurveParams {
    fn default() -> Self {
        Self {
            curve_type: "neutral".to_string(),
            strength: 0.5,
            toe_strength: default_toe_strength(),
            shoulder_strength: default_shoulder_strength(),
            toe_length: default_toe_length(),
            shoulder_start: default_shoulder_start(),
            params: std::collections::HashMap::new(),
        }
    }
}
