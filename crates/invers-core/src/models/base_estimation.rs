//! Base estimation types for film base detection.

use serde::{Deserialize, Serialize};

use super::scan_profile::MaskProfile;

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

    /// Auto-detected mask profile based on base color ratios
    /// This is calculated from medians and used by MaskAware inversion mode
    #[serde(default)]
    pub mask_profile: Option<MaskProfile>,
}

impl Default for BaseEstimation {
    fn default() -> Self {
        Self {
            roi: None,
            medians: [0.5, 0.5, 0.5],
            noise_stats: None,
            auto_estimated: true,
            mask_profile: None,
        }
    }
}

/// Method for automatic base estimation region selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BaseEstimationMethod {
    /// Sample discrete border regions (top, bottom, left, right) individually
    /// Evaluates 5 candidate regions and picks the best one based on brightness
    #[default]
    Regions,

    /// Sample outer N% of entire image as continuous border frame
    Border,

    /// Use brightness histogram peak to estimate base color
    /// Finds the mode in the upper brightness range (0.30-0.90)
    Histogram,
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
