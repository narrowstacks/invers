//! Color model types for curves-based pipeline.

use serde::{Deserialize, Serialize};

use super::options::{default_cb_black_threshold, default_cb_white_threshold};

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
