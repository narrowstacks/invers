//! Conversion options for the processing pipeline.

mod defaults;
mod density;
mod enums;
mod impls;

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::models::base_estimation::{BaseEstimationMethod, BaseSamplingMode};
use crate::models::cb::CbOptions;
use crate::models::preset::{FilmPreset, ToneCurveParams};
use crate::models::scan_profile::ScanProfile;

// Re-export default functions for use in serde attributes
pub(crate) use defaults::{
    default_auto_color_max_divergence, default_auto_color_max_gain, default_auto_color_min_gain,
    default_auto_color_strength, default_auto_exposure_max_gain, default_auto_exposure_min_gain,
    default_auto_exposure_strength, default_auto_exposure_target, default_base_brightest_percent,
    default_clip_percent, default_false, default_one, default_shadow_lift_value, default_true,
    default_wb_strength,
};

// Re-export public types
pub use density::{DensityBalance, DensityBalanceSource, NeutralPointSample};
pub use enums::{
    AutoLevelsMode, AutoWbMode, BitDepthPolicy, InversionMode, OutputFormat, PipelineMode,
    ShadowLiftMode,
};

/// Conversion options for the processing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertOptions {
    /// Input file path(s)
    pub input_paths: Vec<PathBuf>,

    /// Output directory
    pub output_dir: PathBuf,

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
    pub base_estimation: Option<crate::models::base_estimation::BaseEstimation>,

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

    /// Preserve shadow/highlight headroom (don't stretch to full 0-1 range)
    /// When true, output range is approximately 0.005-0.98
    #[serde(default = "default_false")]
    pub preserve_headroom: bool,

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

    /// Maximum allowed divergence between channel gains in auto-color (0.0-1.0)
    /// Limits how much channels can be adjusted relative to each other.
    /// Lower values preserve more scene color character, higher values allow more correction.
    /// Default 0.15 (15%) prevents aggressive neutralization of warm/cool scenes.
    #[serde(default = "default_auto_color_max_divergence")]
    pub auto_color_max_divergence: f32,

    /// Base estimation brightest pixel percentage (1.0-30.0)
    #[serde(default = "default_base_brightest_percent")]
    pub base_brightest_percent: f32,

    /// Base estimation sampling mode
    #[serde(default)]
    pub base_sampling_mode: BaseSamplingMode,

    /// Base estimation method (regions, border, or histogram)
    #[serde(default)]
    pub base_estimation_method: BaseEstimationMethod,

    /// Auto-levels histogram stretching mode
    #[serde(default)]
    pub auto_levels_mode: AutoLevelsMode,

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

    /// Disable all clipping operations to preserve full dynamic range
    /// When enabled, auto-levels will normalize without clipping,
    /// auto-color gains will be limited to prevent exceeding 1.0,
    /// and no clamping will be applied to output values
    #[serde(default = "default_false")]
    pub no_clip: bool,

    /// Enable auto white balance correction
    #[serde(default = "default_false")]
    pub enable_auto_wb: bool,

    /// Auto white balance strength (0.0-1.0, default 1.0)
    #[serde(default = "default_wb_strength")]
    pub auto_wb_strength: f32,

    /// Auto white balance mode
    /// - GrayPixel: Find gray pixels, fallback to highlights/average
    /// - Average: Use average of all pixels (curves-based "AUTO AVG" mode)
    #[serde(default)]
    pub auto_wb_mode: AutoWbMode,

    /// Use GPU acceleration (requires "gpu" feature)
    #[serde(default = "default_true")]
    pub use_gpu: bool,

    // ============================================================
    // Research Pipeline Options
    // ============================================================
    /// Pipeline mode: Legacy (default) or Research
    ///
    /// The Research pipeline implements density balance BEFORE inversion,
    /// which eliminates color crossover between shadows and highlights.
    #[serde(default)]
    pub pipeline_mode: PipelineMode,

    /// Density balance parameters for the research pipeline.
    ///
    /// If not specified, will be auto-calculated from neutral point sampling
    /// or use default values [1.05, 1.0, 0.90].
    #[serde(default)]
    pub density_balance: Option<DensityBalance>,

    /// Neutral point sample for auto-calculating density balance.
    ///
    /// If provided with an ROI, those pixels will be sampled to calculate
    /// density balance exponents. If None, auto-detection will attempt to
    /// find neutral gray areas in the image.
    #[serde(default)]
    pub neutral_point: Option<NeutralPointSample>,

    /// Override density balance red exponent (R^db_r).
    /// Only used if density_balance is not explicitly set.
    /// Typical range: 0.8-1.3, default 1.05
    #[serde(default)]
    pub density_balance_red: Option<f32>,

    /// Override density balance blue exponent (B^db_b).
    /// Only used if density_balance is not explicitly set.
    /// Typical range: 0.7-1.1, default 0.90
    #[serde(default)]
    pub density_balance_blue: Option<f32>,

    /// Override tone curve parameters.
    /// If set, this takes precedence over film_preset tone curve.
    /// Allows specifying tone curve type (e.g., "log", "cinematic") from CLI.
    #[serde(default)]
    pub tone_curve_override: Option<ToneCurveParams>,

    // ============================================================
    // Curves-Based Pipeline Options
    // ============================================================
    /// Curves-based pipeline options.
    ///
    /// Only used when pipeline_mode is set to CbStyle.
    /// Contains all the curve-based processing parameters.
    #[serde(default)]
    pub cb_options: Option<CbOptions>,
}
