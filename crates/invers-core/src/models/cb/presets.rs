//! Preset types for curves-based pipeline.

use serde::{Deserialize, Serialize};

use super::enums::{CbLayerOrder, CbWbMethod, CbWbTonality};

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
