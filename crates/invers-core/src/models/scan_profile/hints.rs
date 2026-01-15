//! Demosaic and white balance hints for RAW processing.

use serde::{Deserialize, Serialize};

/// Demosaic processing hints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemosaicHints {
    /// Demosaic algorithm preference
    pub algorithm: String,

    /// Quality vs speed preference (0.0 = fast, 1.0 = quality)
    pub quality: f32,
}

impl Default for DemosaicHints {
    fn default() -> Self {
        Self {
            algorithm: "ahd".to_string(),
            quality: 0.5,
        }
    }
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

impl Default for WhiteBalanceHints {
    fn default() -> Self {
        Self {
            auto: true,
            temperature: None,
            tint: None,
        }
    }
}
