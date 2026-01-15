//! Automatic levels adjustment functions
//!
//! Provides histogram-based auto-levels with various modes and gamma correction.

mod analysis;
mod auto_levels;
mod histogram;

#[cfg(test)]
mod tests;

// Re-export AutoLevelsMode from models for backward compatibility
pub use crate::models::AutoLevelsMode;

// Re-export from submodules
pub use analysis::{apply_levels_complete, measure_dark_mid_light, measure_dark_mid_light_rgb};
pub use auto_levels::{
    auto_levels, auto_levels_no_clip, auto_levels_with_gamma, auto_levels_with_mode,
    auto_levels_with_target_midpoint,
};
pub use histogram::{
    find_histogram_ends, find_histogram_ends_rgb, otsu_threshold, otsu_threshold_rgb,
    HistogramEnds, OtsuResult,
};

/// Per-channel levels parameters for complete levels control
#[derive(Debug, Clone, Copy)]
pub struct LevelsParams {
    /// Input black point (0.0-1.0)
    pub input_black: f32,
    /// Input white point (0.0-1.0)
    pub input_white: f32,
    /// Gamma value (1.0 = no change, <1.0 = lighten, >1.0 = darken)
    pub gamma: f32,
    /// Output black point (0.0-1.0)
    pub output_black: f32,
    /// Output white point (0.0-1.0)
    pub output_white: f32,
}

impl Default for LevelsParams {
    fn default() -> Self {
        Self {
            input_black: 0.0,
            input_white: 1.0,
            gamma: 1.0,
            output_black: 0.0,
            output_white: 1.0,
        }
    }
}

impl LevelsParams {
    /// Create from just input range (useful for auto-levels results)
    pub fn from_input_range(black: f32, white: f32) -> Self {
        Self {
            input_black: black,
            input_white: white,
            ..Default::default()
        }
    }

    /// Calculate gamma to bring midpoint to target
    /// Formula: gamma = log(target / max) / log(0.5)
    pub fn gamma_for_midpoint(target_mid: f32) -> f32 {
        if target_mid <= 0.0 || target_mid >= 1.0 {
            return 1.0;
        }
        (target_mid.ln() / 0.5_f32.ln()).clamp(0.1, 10.0)
    }
}
