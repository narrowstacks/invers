//! Curves-based pipeline types.
//!
//! Types for the curves-based processing pipeline, which uses curve-based
//! inversion with multiple white balance methods.

use serde::{Deserialize, Serialize};

// ============================================================
// Curves-Based Pipeline Types
// ============================================================

/// White balance application method for curves-based pipeline.
///
/// These methods determine how per-channel WB offsets are applied
/// across the tonal range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbWbMethod {
    /// Linear fixed: WB offset varies linearly with protection at shadow/highlight extremes.
    /// Shadows (< 0.1): offset scales with value/0.1
    /// Midtones (0.1-0.8): full offset applied
    /// Highlights (> 0.8): offset blends toward preserving highlights
    #[default]
    LinearFixed,

    /// Linear dynamic: Simple additive offset (value + offset).
    /// Most aggressive color shift, uniform across all tones.
    LinearDynamic,

    /// Shadow-weighted: Applies WB via power function (value^(1/gamma)).
    /// Stronger effect in shadows, preserves highlights.
    /// Based on gamma balance calculation.
    ShadowWeighted,

    /// Highlight-weighted: Inverse power function (1 - (1-value)^gamma).
    /// Stronger effect in highlights, preserves shadows.
    HighlightWeighted,

    /// Midtone-weighted: Average of shadow and highlight weighted.
    /// Balanced effect across tonal range.
    MidtoneWeighted,
}

/// White balance tonality mode for curves-based pipeline.
///
/// Determines how temperature/tint adjustments are distributed
/// across RGB channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbWbTonality {
    /// Neutral density: Balanced distribution across all channels.
    /// R: -temp/2 - tint/2
    /// G: -temp/2 + tint/2
    /// B: +temp/2 - tint/2
    #[default]
    NeutralDensity,

    /// Subtract density: Direct subtraction model.
    /// R: -temp - tint
    /// G: -temp
    /// B: -tint
    SubtractDensity,

    /// Temperature/tint density: Simplified temp/tint model.
    /// R: -tint/2
    /// G: 0
    /// B: temp - tint/2
    TempTintDensity,
}

/// Processing layer order for curves-based pipeline.
///
/// Controls whether color/WB adjustments are applied before or after
/// tonal adjustments (brightness, contrast, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbLayerOrder {
    /// Apply color/WB adjustments first, then tonal adjustments.
    /// More aggressive color correction.
    #[default]
    ColorFirst,

    /// Apply tonal adjustments first, then color/WB.
    /// Preserves more of the original color character.
    TonesFirst,
}

/// Curves-based pipeline options.
///
/// Configuration for the curves-based processing pipeline, which uses
/// curve-based inversion with multiple white balance methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CbOptions {
    // ---- Preset Selections ----
    /// Tone profile preset (determines default tone settings)
    #[serde(default)]
    pub tone_profile: CbToneProfile,

    /// Enhanced profile / LUT selection
    #[serde(default)]
    pub enhanced_profile: CbEnhancedProfile,

    /// Color model selection
    #[serde(default)]
    pub color_model: CbColorModel,

    /// Source type (camera scan, vuescan, etc.)
    #[serde(default)]
    pub source_type: CbSourceType,

    /// Film character preset
    #[serde(default)]
    pub film_character: CbFilmCharacter,

    /// Engine preset version
    #[serde(default)]
    pub engine_preset: CbEnginePreset,

    // ---- White Balance Settings ----
    /// White balance preset selection (Auto AVG, Auto-Neutral, Kodak, Fuji, etc.)
    #[serde(default)]
    pub wb_preset: CbWbPreset,

    /// White balance temperature adjustment (-100 to +100)
    /// Negative = cooler (more blue), Positive = warmer (more red/yellow)
    #[serde(default)]
    pub wb_temp: f32,

    /// White balance tint adjustment (-100 to +100)
    /// Negative = more green, Positive = more magenta
    #[serde(default)]
    pub wb_tint: f32,

    /// White balance application method
    #[serde(default)]
    pub wb_method: CbWbMethod,

    /// White balance tonality mode (how temp/tint map to RGB)
    #[serde(default)]
    pub wb_tonality: CbWbTonality,

    // ---- Tone Settings ----
    /// Brightness adjustment (-100 to +100)
    #[serde(default)]
    pub brightness: f32,

    /// Exposure adjustment (-100 to +100)
    #[serde(default)]
    pub exposure: f32,

    /// Contrast adjustment (-100 to +100)
    /// Applied via sigmoid curve (tanh-based)
    #[serde(default)]
    pub contrast: f32,

    /// Highlights adjustment (-100 to +100)
    #[serde(default)]
    pub highlights: f32,

    /// Shadows adjustment (-100 to +100)
    #[serde(default)]
    pub shadows: f32,

    /// Whites adjustment (-100 to +100)
    #[serde(default)]
    pub whites: f32,

    /// Blacks adjustment (-100 to +100)
    #[serde(default)]
    pub blacks: f32,

    /// Gamma adjustment (default 1.0, range 0.5-2.0)
    #[serde(default = "default_gamma")]
    pub gamma: f32,

    // ---- Color Settings ----
    /// Cyan channel offset (for color balance)
    #[serde(default)]
    pub cyan: f32,

    /// Magenta/Tint channel offset
    #[serde(default)]
    pub tint: f32,

    /// Yellow/Temperature channel offset
    #[serde(default)]
    pub temp: f32,

    // ---- Processing Order ----
    /// Layer order: color first or tones first
    #[serde(default)]
    pub layer_order: CbLayerOrder,

    // ---- Clipping Settings ----
    /// Soft shadows: extend curve slightly below 0 for smoother rolloff
    #[serde(default)]
    pub soft_shadows: bool,

    /// Soft highlights: extend curve slightly above 255 for smoother rolloff
    #[serde(default)]
    pub soft_highlights: bool,

    /// Clip white point softening (0-20)
    #[serde(default)]
    pub soft_high: f32,

    /// Clip black point softening (0-20)
    #[serde(default)]
    pub soft_low: f32,

    // ---- Shadow/Highlight Toning ----
    /// Shadow cyan toning (-100 to +100)
    #[serde(default)]
    pub shadow_cyan: f32,

    /// Shadow tint toning (-100 to +100)
    #[serde(default)]
    pub shadow_tint: f32,

    /// Shadow temperature toning (-100 to +100)
    #[serde(default)]
    pub shadow_temp: f32,

    /// Shadow toning range (1-10, higher = wider range)
    #[serde(default = "default_shadow_range")]
    pub shadow_range: f32,

    /// Highlight cyan toning (-100 to +100)
    #[serde(default)]
    pub highlight_cyan: f32,

    /// Highlight tint toning (-100 to +100)
    #[serde(default)]
    pub highlight_tint: f32,

    /// Highlight temperature toning (-100 to +100)
    #[serde(default)]
    pub highlight_temp: f32,

    /// Highlight toning range (1-10, higher = wider range)
    #[serde(default = "default_highlight_range")]
    pub highlight_range: f32,

    // ---- Histogram Thresholds ----
    /// Black point threshold for histogram analysis (0.0-1.0)
    /// Percentage of pixels to clip at black
    #[serde(default = "default_cb_black_threshold")]
    pub black_threshold: f32,

    /// White point threshold for histogram analysis (0.0-1.0)
    /// Percentage of pixels to clip at white
    #[serde(default = "default_cb_white_threshold")]
    pub white_threshold: f32,

    // ---- Enhanced Profile Settings ----
    /// Enhanced profile strength (0-200, default 100)
    #[serde(default = "default_enhanced_profile_strength")]
    pub enhanced_profile_strength: f32,

    /// Post-processing saturation (1-10, default 5)
    #[serde(default = "default_post_saturation")]
    pub post_saturation: f32,
}

fn default_gamma() -> f32 {
    1.0
}

fn default_enhanced_profile_strength() -> f32 {
    100.0
}

fn default_post_saturation() -> f32 {
    5.0
}

fn default_shadow_range() -> f32 {
    5.0
}

fn default_highlight_range() -> f32 {
    5.0
}

fn default_cb_black_threshold() -> f32 {
    0.0
}

fn default_cb_white_threshold() -> f32 {
    0.0
}

impl Default for CbOptions {
    fn default() -> Self {
        Self {
            // Preset selections
            tone_profile: CbToneProfile::default(),
            enhanced_profile: CbEnhancedProfile::default(),
            color_model: CbColorModel::default(),
            source_type: CbSourceType::default(),
            film_character: CbFilmCharacter::default(),
            engine_preset: CbEnginePreset::default(),
            // WB settings
            wb_preset: CbWbPreset::default(),
            wb_temp: 0.0,
            wb_tint: 0.0,
            wb_method: CbWbMethod::default(),
            wb_tonality: CbWbTonality::default(),
            // Tone settings
            brightness: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            gamma: default_gamma(),
            // Color settings
            cyan: 0.0,
            tint: 0.0,
            temp: 0.0,
            // Processing order
            layer_order: CbLayerOrder::default(),
            // Clipping settings
            soft_shadows: false,
            soft_highlights: false,
            soft_high: 0.0,
            soft_low: 0.0,
            // Shadow/highlight toning
            shadow_cyan: 0.0,
            shadow_tint: 0.0,
            shadow_temp: 0.0,
            shadow_range: default_shadow_range(),
            highlight_cyan: 0.0,
            highlight_tint: 0.0,
            highlight_temp: 0.0,
            highlight_range: default_highlight_range(),
            // Histogram thresholds
            black_threshold: default_cb_black_threshold(),
            white_threshold: default_cb_white_threshold(),
            // Enhanced profile settings
            enhanced_profile_strength: default_enhanced_profile_strength(),
            post_saturation: default_post_saturation(),
        }
    }
}

impl CbOptions {
    /// Create CbOptions with defaults from selected presets.
    ///
    /// This applies the default values from the tone profile, color model,
    /// and engine preset to create a fully configured CbOptions.
    pub fn from_presets(
        tone_profile: CbToneProfile,
        enhanced_profile: CbEnhancedProfile,
        color_model: CbColorModel,
        film_character: CbFilmCharacter,
        engine_preset: CbEnginePreset,
    ) -> Self {
        let tone_defaults = tone_profile.defaults();
        let color_defaults = color_model.defaults();
        let engine_settings = engine_preset.settings();

        Self {
            // Preset selections
            tone_profile,
            enhanced_profile,
            color_model,
            source_type: CbSourceType::default(),
            film_character,
            engine_preset,
            // WB settings from engine
            wb_preset: CbWbPreset::default(),
            wb_temp: 0.0,
            wb_tint: 0.0,
            wb_method: engine_settings.wb_method,
            wb_tonality: engine_settings.wb_tonality,
            // Tone settings from profile
            brightness: tone_defaults.brightness,
            exposure: 0.0,
            contrast: tone_defaults.contrast,
            highlights: tone_defaults.highlights,
            shadows: tone_defaults.shadows,
            whites: tone_defaults.whites,
            blacks: tone_defaults.blacks,
            gamma: tone_defaults.gamma,
            // Color settings
            cyan: 0.0,
            tint: 0.0,
            temp: 0.0,
            // Processing order from engine
            layer_order: engine_settings.layer_order,
            // Clipping settings from profile
            soft_shadows: false,
            soft_highlights: false,
            soft_high: tone_defaults.soft_high,
            soft_low: tone_defaults.soft_low,
            // Shadow/highlight toning
            shadow_cyan: 0.0,
            shadow_tint: 0.0,
            shadow_temp: 0.0,
            shadow_range: default_shadow_range(),
            highlight_cyan: 0.0,
            highlight_tint: 0.0,
            highlight_temp: 0.0,
            highlight_range: default_highlight_range(),
            // Histogram thresholds from color model
            black_threshold: color_defaults.black_threshold,
            white_threshold: color_defaults.white_threshold,
            // Enhanced profile settings
            enhanced_profile_strength: default_enhanced_profile_strength(),
            post_saturation: default_post_saturation(),
        }
    }

    /// Create CbOptions matching curves-based default settings for color negatives.
    pub fn default_negative() -> Self {
        Self::from_presets(
            CbToneProfile::Standard,
            CbEnhancedProfile::Frontier,
            CbColorModel::Basic,
            CbFilmCharacter::Kodak,
            CbEnginePreset::V3_1,
        )
    }

    /// Create CbOptions matching curves-based default settings for B&W negatives.
    pub fn default_mono() -> Self {
        Self::from_presets(
            CbToneProfile::LinearGamma,
            CbEnhancedProfile::Frontier,
            CbColorModel::Basic,
            CbFilmCharacter::Kodak,
            CbEnginePreset::V3_1,
        )
    }

    /// Create CbOptions matching curves-based default settings for positive film.
    pub fn default_positive() -> Self {
        Self::from_presets(
            CbToneProfile::Linear,
            CbEnhancedProfile::None,
            CbColorModel::Basic,
            CbFilmCharacter::Kodak,
            CbEnginePreset::V3_1,
        )
    }
}

/// Per-channel histogram analysis results (curves-based).
///
/// Contains white point, black point, and mean for each channel
/// after histogram analysis.
#[derive(Debug, Clone, Default)]
pub struct CbChannelOrigins {
    /// White point (0-255 scale, brightest significant value)
    pub white_point: f32,

    /// Black point (0-255 scale, darkest significant value)
    pub black_point: f32,

    /// Mean point (0.0-1.0 scale)
    pub mean_point: f32,
}

/// Complete histogram analysis for all RGB channels.
#[derive(Debug, Clone, Default)]
pub struct CbHistogramAnalysis {
    /// Red channel analysis
    pub red: CbChannelOrigins,

    /// Green channel analysis
    pub green: CbChannelOrigins,

    /// Blue channel analysis
    pub blue: CbChannelOrigins,
}

// ============================================================
// CB Preset Types
// ============================================================

/// CB tone profile presets.
///
/// Each tone profile provides default values for brightness, contrast,
/// shadows, highlights, blacks, whites, gamma, and softness parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbToneProfile {
    /// Standard tone profile with balanced contrast
    #[default]
    Standard,

    /// Linear profile with no automatic adjustments
    Linear,

    /// Linear with gamma correction (0.66)
    LinearGamma,

    /// Linear with extended flat highlights
    LinearFlat,

    /// Linear with deep shadows
    LinearDeep,

    /// Logarithmic profile for wide dynamic range
    Logarithmic,

    /// Logarithmic with richer shadows and added contrast
    LogarithmicRich,

    /// Logarithmic with flatter highlight/shadow rolloff
    LogarithmicFlat,

    /// Soft tones overall (lifted blacks, crushed whites)
    AllSoft,

    /// Hard tones overall (increased contrast)
    AllHard,

    /// Hard highlights with standard shadows
    HighlightHard,

    /// Soft highlights with lifted shadows
    HighlightSoft,

    /// Hard shadows with standard highlights
    ShadowHard,

    /// Soft shadows with compressed highlights
    ShadowSoft,

    /// Auto-tone profile (brightness/contrast adjusted automatically)
    AutoTone,
}

impl CbToneProfile {
    /// Get the default parameter values for this tone profile
    pub fn defaults(&self) -> CbToneProfileParams {
        match self {
            Self::Standard => CbToneProfileParams {
                brightness: 0.0,
                blacks: 2.0,
                whites: -2.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 10.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::Linear => CbToneProfileParams {
                brightness: 0.0,
                blacks: 0.0,
                whites: 0.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: -3.0,
                soft_high: -3.0,
                auto_tone: false,
            },
            Self::LinearGamma => CbToneProfileParams {
                brightness: 0.0,
                blacks: 0.0,
                whites: 0.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 0.66,
                contrast: 0.0,
                soft_low: -3.0,
                soft_high: -3.0,
                auto_tone: false,
            },
            Self::LinearFlat => CbToneProfileParams {
                brightness: 0.0,
                blacks: 0.0,
                whites: 0.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: -10.0,
                soft_high: -15.0,
                auto_tone: false,
            },
            Self::LinearDeep => CbToneProfileParams {
                brightness: 0.0,
                blacks: 0.0,
                whites: 0.0,
                shadows: -12.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: -3.0,
                soft_high: -3.0,
                auto_tone: false,
            },
            Self::Logarithmic => CbToneProfileParams {
                brightness: 0.0,
                blacks: -10.0,
                whites: 0.0,
                shadows: -10.0,
                highlights: -25.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: -3.0,
                soft_high: -3.0,
                auto_tone: false,
            },
            Self::LogarithmicRich => CbToneProfileParams {
                brightness: 0.0,
                blacks: -20.0,
                whites: 0.0,
                shadows: -10.0,
                highlights: -25.0,
                gamma: 1.0,
                contrast: 5.0,
                soft_low: -3.0,
                soft_high: -3.0,
                auto_tone: false,
            },
            Self::LogarithmicFlat => CbToneProfileParams {
                brightness: 0.0,
                blacks: -10.0,
                whites: 0.0,
                shadows: -10.0,
                highlights: -25.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: -6.0,
                soft_high: -9.0,
                auto_tone: false,
            },
            Self::AllSoft => CbToneProfileParams {
                brightness: 0.0,
                blacks: 10.0,
                whites: -10.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::AllHard => CbToneProfileParams {
                brightness: 0.0,
                blacks: 2.0,
                whites: -2.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 25.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::HighlightHard => CbToneProfileParams {
                brightness: 0.0,
                blacks: 2.0,
                whites: -2.0,
                shadows: 0.0,
                highlights: 10.0,
                gamma: 1.0,
                contrast: 10.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::HighlightSoft => CbToneProfileParams {
                brightness: 0.0,
                blacks: 2.0,
                whites: -10.0,
                shadows: -10.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::ShadowHard => CbToneProfileParams {
                brightness: 0.0,
                blacks: 2.0,
                whites: -2.0,
                shadows: -10.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 10.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::ShadowSoft => CbToneProfileParams {
                brightness: 0.0,
                blacks: 9.0,
                whites: -2.0,
                shadows: 0.0,
                highlights: 10.0,
                gamma: 1.0,
                contrast: 0.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
            Self::AutoTone => CbToneProfileParams {
                brightness: 0.0,
                blacks: 5.0,
                whites: -1.0,
                shadows: 0.0,
                highlights: 0.0,
                gamma: 1.0,
                contrast: 10.0,
                soft_low: 0.0,
                soft_high: 0.0,
                auto_tone: true,
            },
        }
    }
}

/// Default parameter values for a tone profile.
#[derive(Debug, Clone)]
pub struct CbToneProfileParams {
    pub brightness: f32,
    pub blacks: f32,
    pub whites: f32,
    pub shadows: f32,
    pub highlights: f32,
    pub gamma: f32,
    pub contrast: f32,
    pub soft_low: f32,
    pub soft_high: f32,
    pub auto_tone: bool,
}

/// CB enhanced profile (LUT) selection.
///
/// These profiles apply color grading similar to different
/// film lab minilabs and scanners.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbEnhancedProfile {
    /// No enhanced profile / LUT
    #[default]
    None,

    /// Natural profile - subtle color grading
    Natural,

    /// Frontier profile - Fuji Frontier minilab look
    Frontier,

    /// Crystal profile - Crystal Archive paper look
    Crystal,

    /// Pakon profile - Pakon scanner look
    Pakon,
}

/// CB color model for different scan sources.
///
/// Each color model provides color adjustments optimized
/// for different scanning workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbColorModel {
    /// No color model adjustments
    None,

    /// Basic color model with subtle blue hue shift
    #[default]
    Basic,

    /// Frontier-style color with red/blue hue adjustments
    Frontier,

    /// Noritsu-style color
    Noritsu,

    /// Black and white conversion
    BlackAndWhite,
}

impl CbColorModel {
    /// Get the default color adjustments for this color model
    pub fn defaults(&self) -> CbColorModelParams {
        match self {
            Self::None => CbColorModelParams {
                red_hue: 0,
                red_saturation: 0,
                green_hue: 0,
                green_saturation: 0,
                blue_hue: 0,
                blue_saturation: 0,
                black_threshold: default_cb_black_threshold(),
                white_threshold: default_cb_white_threshold(),
                convert_to_grayscale: false,
            },
            Self::Basic => CbColorModelParams {
                red_hue: 0,
                red_saturation: 0,
                green_hue: 0,
                green_saturation: 0,
                blue_hue: -10,
                blue_saturation: 0,
                black_threshold: default_cb_black_threshold(),
                white_threshold: default_cb_white_threshold(),
                convert_to_grayscale: false,
            },
            Self::Frontier => CbColorModelParams {
                red_hue: 15,
                red_saturation: -10,
                green_hue: 0,
                green_saturation: 0,
                blue_hue: -15,
                blue_saturation: 0,
                black_threshold: default_cb_black_threshold(),
                white_threshold: default_cb_white_threshold(),
                convert_to_grayscale: false,
            },
            Self::Noritsu => CbColorModelParams {
                red_hue: 15,
                red_saturation: -4,
                green_hue: 0,
                green_saturation: 0,
                blue_hue: -10,
                blue_saturation: 0,
                black_threshold: default_cb_black_threshold(),
                white_threshold: default_cb_white_threshold(),
                convert_to_grayscale: false,
            },
            Self::BlackAndWhite => CbColorModelParams {
                red_hue: 0,
                red_saturation: 0,
                green_hue: 0,
                green_saturation: 0,
                blue_hue: 0,
                blue_saturation: 0,
                black_threshold: default_cb_black_threshold(),
                white_threshold: default_cb_white_threshold(),
                convert_to_grayscale: true,
            },
        }
    }
}

/// Color model parameters for CB processing.
#[derive(Debug, Clone)]
pub struct CbColorModelParams {
    pub red_hue: i32,
    pub red_saturation: i32,
    pub green_hue: i32,
    pub green_saturation: i32,
    pub blue_hue: i32,
    pub blue_saturation: i32,
    pub black_threshold: f32,
    pub white_threshold: f32,
    pub convert_to_grayscale: bool,
}

/// CB source type for different scanning methods.
///
/// Different sources require different color handling
/// due to varying gamma, color space, and other characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbSourceType {
    /// Camera scan (DSLR/mirrorless digitization)
    #[default]
    CameraScan,

    /// VueScan RAW output
    VuescanRaw,

    /// Epson scanner output
    EpsonScan,

    /// Linear/raw scanner output
    LinearScan,
}

/// CB white balance preset selection.
///
/// These map to the WB selection dropdown in the curves-based UI.
/// Some presets use auto-analyzed neutral points, while others
/// use fixed film character adjustments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbWbPreset {
    /// No white balance adjustment
    None,

    /// Auto color - simple average-based WB (grayworld)
    #[default]
    AutoColor,

    /// Auto-Neutral - uses analyzed neutral point (most balanced)
    AutoNeutral,

    /// Auto-Warm - uses analyzed warm point (shifted warmer)
    AutoWarm,

    /// Auto-Cool - uses analyzed cool point (shifted cooler)
    AutoCool,

    /// Auto-Mix - combination of neutral points
    AutoMix,

    /// Standard/Generic color preset
    Standard,

    /// Kodak film character
    Kodak,

    /// Fuji film character
    Fuji,

    /// Cinestill 800T (tungsten balanced)
    CineT,

    /// Cinestill 50D (daylight balanced)
    CineD,

    /// Custom - uses manual wb_temp and wb_tint values
    Custom,
}

/// CB film character presets.
///
/// Film character affects the white balance multipliers
/// used during conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbFilmCharacter {
    /// No film character adjustments
    None,

    /// Generic color negative
    GenericColor,

    /// Kodak Portra style
    #[default]
    Kodak,

    /// Fuji style
    Fuji,

    /// Cinestill 50D (daylight balanced)
    Cinestill50D,

    /// Cinestill 800T (tungsten balanced)
    Cinestill800T,
}

impl CbFilmCharacter {
    /// Get the RGB multipliers for this film character
    pub fn multipliers(&self) -> [f32; 3] {
        match self {
            Self::None => [1.0, 1.0, 1.0],
            Self::GenericColor | Self::Kodak | Self::Fuji | Self::Cinestill50D => {
                [1.233, 1.167, 1.0]
            }
            Self::Cinestill800T => [1.233, 1.167, 1.0],
        }
    }

    /// Get the cyan/magenta/yellow adjustments for this film character
    pub fn cmy_adjustments(&self) -> (i32, i32, i32) {
        match self {
            Self::None => (0, 0, 0),
            Self::GenericColor | Self::Cinestill50D => (0, 10, 17),
            Self::Kodak => (0, 7, 17),
            Self::Fuji => (0, 4, 17),
            Self::Cinestill800T => (0, 7, 24),
        }
    }
}

/// CB engine preset versions.
///
/// Different engine versions use different processing approaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbEnginePreset {
    /// Version 3.1 - Latest with multi-pass correction and shadow-weighted WB
    #[default]
    V3_1,

    /// Version 3.0 - Single pass with shadow-weighted WB
    V3_0,

    /// Version 2.3 - Linear fixed WB method
    V2_3,

    /// Version 2.2 - Precise curve points
    V2_2,

    /// Version 2.1 - Smooth curves, tones first
    V2_1,
}

impl CbEnginePreset {
    /// Get the engine settings for this preset
    pub fn settings(&self) -> CbEngineSettings {
        match self {
            Self::V3_1 => CbEngineSettings {
                wb_method: CbWbMethod::ShadowWeighted,
                wb_tonality: CbWbTonality::TempTintDensity, // "addDensity" in CB
                layer_order: CbLayerOrder::ColorFirst,
                multi_pass: true,
            },
            Self::V3_0 => CbEngineSettings {
                wb_method: CbWbMethod::ShadowWeighted,
                wb_tonality: CbWbTonality::TempTintDensity,
                layer_order: CbLayerOrder::ColorFirst,
                multi_pass: false,
            },
            Self::V2_3 => CbEngineSettings {
                wb_method: CbWbMethod::LinearFixed,
                wb_tonality: CbWbTonality::TempTintDensity,
                layer_order: CbLayerOrder::ColorFirst,
                multi_pass: false,
            },
            Self::V2_2 => CbEngineSettings {
                wb_method: CbWbMethod::LinearFixed,
                wb_tonality: CbWbTonality::TempTintDensity,
                layer_order: CbLayerOrder::ColorFirst,
                multi_pass: false,
            },
            Self::V2_1 => CbEngineSettings {
                wb_method: CbWbMethod::LinearFixed,
                wb_tonality: CbWbTonality::NeutralDensity,
                layer_order: CbLayerOrder::TonesFirst,
                multi_pass: false,
            },
        }
    }
}

/// Engine settings for CB processing.
#[derive(Debug, Clone)]
pub struct CbEngineSettings {
    pub wb_method: CbWbMethod,
    pub wb_tonality: CbWbTonality,
    pub layer_order: CbLayerOrder,
    pub multi_pass: bool,
}
