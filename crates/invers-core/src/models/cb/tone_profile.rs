//! Tone profile types for curves-based pipeline.

use serde::{Deserialize, Serialize};

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
