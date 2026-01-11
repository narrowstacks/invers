//! Conversion options for the processing pipeline.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use super::base_estimation::{BaseEstimationMethod, BaseSamplingMode};
use super::cb::CbOptions;
use super::preset::{FilmPreset, ToneCurveParams};
use super::scan_profile::ScanProfile;

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
    pub base_estimation: Option<super::base_estimation::BaseEstimation>,

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

// Default value functions for serde
pub(crate) fn default_true() -> bool {
    true
}

pub(crate) fn default_false() -> bool {
    false
}

pub(crate) fn default_clip_percent() -> f32 {
    0.25
}

pub(crate) fn default_auto_color_strength() -> f32 {
    0.6
}

pub(crate) fn default_wb_strength() -> f32 {
    1.0
}

pub(crate) fn default_auto_color_min_gain() -> f32 {
    0.7
}

pub(crate) fn default_auto_color_max_gain() -> f32 {
    1.3
}

pub(crate) fn default_auto_color_max_divergence() -> f32 {
    0.15
}

pub(crate) fn default_base_brightest_percent() -> f32 {
    5.0
}

pub(crate) fn default_shadow_lift_value() -> f32 {
    0.02
}

pub(crate) fn default_one() -> f32 {
    1.0
}

pub(crate) fn default_auto_exposure_target() -> f32 {
    0.25
}

pub(crate) fn default_auto_exposure_strength() -> f32 {
    1.0
}

pub(crate) fn default_auto_exposure_min_gain() -> f32 {
    0.6
}

pub(crate) fn default_auto_exposure_max_gain() -> f32 {
    1.4
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            output_dir: PathBuf::from("."),
            output_format: OutputFormat::default(),
            working_colorspace: "linear-rec2020".to_string(),
            bit_depth_policy: BitDepthPolicy::default(),
            film_preset: None,
            scan_profile: None,
            base_estimation: None,
            num_threads: None,
            skip_tone_curve: false,
            skip_color_matrix: false,
            exposure_compensation: 1.0,
            debug: false,
            enable_auto_levels: default_true(),
            auto_levels_clip_percent: default_clip_percent(),
            preserve_headroom: default_false(),
            enable_auto_color: default_false(),
            auto_color_strength: default_auto_color_strength(),
            auto_color_min_gain: default_auto_color_min_gain(),
            auto_color_max_gain: default_auto_color_max_gain(),
            auto_color_max_divergence: default_auto_color_max_divergence(),
            base_brightest_percent: default_base_brightest_percent(),
            base_sampling_mode: BaseSamplingMode::default(),
            base_estimation_method: BaseEstimationMethod::default(),
            auto_levels_mode: AutoLevelsMode::default(),
            inversion_mode: InversionMode::default(),
            shadow_lift_mode: ShadowLiftMode::default(),
            shadow_lift_value: default_shadow_lift_value(),
            highlight_compression: default_one(),
            enable_auto_exposure: default_true(),
            auto_exposure_target_median: default_auto_exposure_target(),
            auto_exposure_strength: default_auto_exposure_strength(),
            auto_exposure_min_gain: default_auto_exposure_min_gain(),
            auto_exposure_max_gain: default_auto_exposure_max_gain(),
            no_clip: default_false(),
            enable_auto_wb: default_false(),
            auto_wb_strength: default_wb_strength(),
            auto_wb_mode: AutoWbMode::default(),
            use_gpu: default_true(),
            pipeline_mode: PipelineMode::default(),
            density_balance: None,
            neutral_point: None,
            density_balance_red: None,
            density_balance_blue: None,
            tone_curve_override: None,
            cb_options: None,
        }
    }
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

/// Auto-levels histogram stretching mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutoLevelsMode {
    /// Stretch each channel independently (can shift colors)
    #[default]
    PerChannel,

    /// Use same stretch factor for all channels (preserves color relationships)
    /// Also known as PreserveSaturation mode
    Unified,

    /// Saturation-aware: reduces stretch for channels that would clip heavily
    SaturationAware,
}

/// Auto white balance mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutoWbMode {
    /// Gray pixel detection - find pixels with similar R/G/B values
    /// Falls back to highlights, then to average
    #[default]
    GrayPixel,

    /// Average/Gray World - assume average of all pixels should be neutral
    Average,

    /// Percentile-based (Robust White Patch) - use high percentile as white reference
    /// More robust than max RGB, preserves more color character than gray world
    /// This is likely closest to the curves-based "AUTO AVG" behavior
    Percentile,
}

/// Inversion mode for negative-to-positive conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InversionMode {
    /// Linear inversion: (base - negative) / base
    Linear,

    /// Logarithmic inversion: 10^(log10(base) - log10(negative))
    Logarithmic,

    /// Divide-blend inversion:
    /// 1. Divide: pixel / base (per channel)
    /// 2. Apply gamma 2.2 (convert from linear to gamma-encoded)
    /// 3. Invert: 1.0 - result
    ///
    /// This mode mimics Photoshop's Divide blend mode workflow.
    DivideBlend,

    /// Orange mask-aware inversion for color negative film.
    ///
    /// This mode properly accounts for the orange mask present in color negative
    /// film. The mask exists because real-world dyes have impurities:
    /// - Magenta dye absorbs some blue light (not just green)
    /// - Cyan dye absorbs some green light (not just red)
    ///
    /// Film manufacturers add colored dye couplers to compensate, creating
    /// the characteristic orange mask. Simple inversion of this mask produces
    /// a blue cast in shadows.
    ///
    /// This mode:
    /// 1. Performs standard inversion: 1.0 - (pixel / base)
    /// 2. Calculates per-channel shadow floor based on mask characteristics
    /// 3. Applies shadow correction to neutralize the blue cast
    /// 4. Automatically skips color matrix (no longer needed)
    #[default]
    MaskAware,

    /// Simple B&W inversion for grayscale or monochrome images.
    ///
    /// This mode is optimized for black and white film:
    /// 1. Simple inversion: 1.0 - (pixel / base)
    /// 2. Sets black point slightly below the film base (with headroom)
    /// 3. Skips color-specific operations (color matrix, auto-color, etc.)
    ///
    /// The headroom parameter (default 5%) preserves shadow detail by not
    /// clipping the film base completely to black.
    BlackAndWhite,
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

/// Pipeline mode selection
///
/// Controls which processing pipeline is used for negative-to-positive conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PipelineMode {
    /// Legacy pipeline with existing inversion modes and post-processing stages.
    /// This is the original Invers pipeline, kept for backward compatibility.
    #[default]
    Legacy,

    /// Research-based pipeline implementing densitometry principles.
    ///
    /// Key difference from Legacy: applies **density balance BEFORE inversion**
    /// using per-channel power functions to align characteristic curves.
    ///
    /// Pipeline stages:
    /// 1. Film base white balance (divide by base to normalize orange mask)
    /// 2. Density balance (per-channel power: R^db_r, G^1.0, B^db_b)
    /// 3. Reciprocal inversion (positive = k / negative)
    /// 4. Auto-levels (histogram normalization)
    /// 5. Tone curve
    /// 6. Export
    ///
    /// This approach eliminates color crossover between shadows and highlights
    /// by aligning the RGB characteristic curves before inversion.
    Research,

    /// Curves-based pipeline inspired by Negative Lab Pro algorithms.
    ///
    /// This pipeline implements a curve-based approach with multiple white balance
    /// methods and tonality modes. Key features:
    ///
    /// Pipeline stages:
    /// 1. Histogram analysis to find white/black points per channel
    /// 2. Film base normalization
    /// 3. Inversion via per-channel tone curves
    /// 4. White balance application (5 methods: linear, gamma-weighted)
    /// 5. Exposure/brightness/contrast via sigmoid curves
    /// 6. Shadow/highlight toning
    /// 7. Final curve application
    ///
    /// Supports two processing orders:
    /// - `colorFirst`: Apply WB before tone adjustments
    /// - `tonesFirst`: Apply tones before WB (preserves more color character)
    CbStyle,
}

/// Source of density balance values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DensityBalanceSource {
    /// Auto-calculated from neutral point sampling
    #[default]
    Auto,

    /// Manually specified values
    Manual,

    /// Default values (R=1.05, G=1.0, B=0.90)
    Default,
}

/// Density balance parameters for the research pipeline.
///
/// Per-channel power functions applied BEFORE inversion to align
/// the characteristic curves of each RGB emulsion layer.
///
/// Each film layer has a slightly different gamma (e.g., R=0.63, G=0.71, B=0.73).
/// Without density balance, this causes "color crossover" where shadows shift
/// toward one color cast while highlights shift toward another.
///
/// The density balance exponents correct this:
/// - R_balanced = R^db_r (typically 1.0-1.1)
/// - G_balanced = G^db_g (always 1.0, reference channel)
/// - B_balanced = B^db_b (typically 0.85-0.95)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityBalance {
    /// Per-channel exponents [R, G, B].
    /// G is always 1.0 (reference channel).
    /// R typically 1.0-1.1, B typically 0.85-0.95.
    pub exponents: [f32; 3],

    /// Source of the density balance values
    pub source: DensityBalanceSource,
}

impl Default for DensityBalance {
    fn default() -> Self {
        Self {
            // Typical starting values from research.md
            exponents: [1.05, 1.0, 0.90],
            source: DensityBalanceSource::Default,
        }
    }
}

impl DensityBalance {
    /// Create density balance from manual exponent values
    pub fn manual(red_exp: f32, blue_exp: f32) -> Self {
        Self {
            exponents: [red_exp, 1.0, blue_exp],
            source: DensityBalanceSource::Manual,
        }
    }

    /// Calculate density balance from a neutral point sample.
    ///
    /// Given RGB values from a known neutral gray area, calculates
    /// the exponents needed to make all channels equal after the
    /// power transformation.
    ///
    /// Algorithm:
    /// - G is reference (exponent = 1.0)
    /// - R^db_r = G => db_r = ln(G) / ln(R)
    /// - B^db_b = G => db_b = ln(G) / ln(B)
    pub fn from_neutral_point(neutral_rgb: [f32; 3]) -> Self {
        let [r, g, b] = neutral_rgb;

        // Avoid log(0) and ensure reasonable inputs
        let r = r.max(0.001);
        let g = g.max(0.001);
        let b = b.max(0.001);

        // G is reference (exponent = 1.0)
        // R^db_r = G => db_r = ln(G) / ln(R)
        // B^db_b = G => db_b = ln(G) / ln(B)
        let db_r = if (r - g).abs() > 0.001 {
            (g.ln() / r.ln()).clamp(0.8, 1.3)
        } else {
            1.0 // R ≈ G, no correction needed
        };

        let db_b = if (b - g).abs() > 0.001 {
            (g.ln() / b.ln()).clamp(0.7, 1.1)
        } else {
            1.0 // B ≈ G, no correction needed
        };

        Self {
            exponents: [db_r, 1.0, db_b],
            source: DensityBalanceSource::Auto,
        }
    }
}

/// Neutral point sample for density balance calculation.
///
/// Used to auto-calculate density balance exponents by sampling
/// a known neutral gray area in the image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralPointSample {
    /// Region of interest for neutral sampling (x, y, width, height).
    /// If None, auto-detection will search for neutral areas.
    pub roi: Option<(u32, u32, u32, u32)>,

    /// Sampled RGB values from neutral point (after film base normalization)
    pub neutral_rgb: [f32; 3],

    /// Whether this was auto-detected or manually specified
    pub auto_detected: bool,
}

impl Default for NeutralPointSample {
    fn default() -> Self {
        Self {
            roi: None,
            neutral_rgb: [0.5, 0.5, 0.5],
            auto_detected: true,
        }
    }
}
