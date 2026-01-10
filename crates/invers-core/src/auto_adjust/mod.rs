//! Automatic adjustment functions for image processing
//!
//! Provides auto-levels, auto-color, auto white balance, and other automatic
//! corrections similar to Photoshop's automatic adjustment tools.

mod color;
mod exposure;
mod levels;
mod parallel;
mod white_balance;

/// Minimum number of pixels to trigger parallel processing
pub(crate) const PARALLEL_THRESHOLD: usize = 30_000;

// Re-export white balance functions
pub use white_balance::{
    apply_white_balance_from_temperature, auto_white_balance, auto_white_balance_avg,
    auto_white_balance_no_clip, auto_white_balance_percentile, kelvin_to_rgb_multipliers,
};

// Re-export levels functions and types
pub use levels::{
    apply_levels_complete, auto_levels, auto_levels_no_clip, auto_levels_with_gamma,
    auto_levels_with_mode, auto_levels_with_target_midpoint, find_histogram_ends,
    find_histogram_ends_rgb, measure_dark_mid_light, measure_dark_mid_light_rgb, otsu_threshold,
    otsu_threshold_rgb, AutoLevelsMode, HistogramEnds, LevelsParams, OtsuResult,
};

// Re-export color functions
pub use color::{auto_color, auto_color_no_clip, limit_channel_divergence};

// Re-export exposure functions
pub use exposure::{
    adaptive_shadow_lift, auto_exposure, auto_exposure_no_clip, compress_highlights,
};
