//! Film preset types for conversion parameters.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
    pub params: HashMap<String, f32>,
}

pub(crate) fn default_toe_strength() -> f32 {
    0.4
}

pub(crate) fn default_shoulder_strength() -> f32 {
    0.0 // Preserve highlights - no shoulder compression by default
}

pub(crate) fn default_toe_length() -> f32 {
    0.25
}

pub(crate) fn default_shoulder_start() -> f32 {
    0.75
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
            params: HashMap::new(),
        }
    }
}
