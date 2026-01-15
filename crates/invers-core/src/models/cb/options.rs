//! CbOptions struct for curves-based pipeline configuration.

use serde::{Deserialize, Serialize};

use super::color_model::{CbColorModel, CbEnhancedProfile};
use super::enums::{CbLayerOrder, CbWbMethod, CbWbTonality};
use super::presets::{CbEnginePreset, CbFilmCharacter, CbSourceType, CbWbPreset};
use super::tone_profile::CbToneProfile;

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

pub fn default_cb_black_threshold() -> f32 {
    0.0
}

pub fn default_cb_white_threshold() -> f32 {
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
