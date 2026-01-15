//! White balance adjustment functions
//!
//! Provides automatic and manual white balance correction including
//! temperature-based adjustment and various auto white balance algorithms.
//!
//! # Modules
//! - `auto_wb`: Auto white balance algorithms (gray pixel, average, percentile)
//! - `compute`: Compute-only functions for GPU pipelines
//! - `kelvin`: Temperature-based white balance conversion

mod auto_wb;
mod compute;
mod kelvin;

#[cfg(test)]
mod tests;

// Re-export public API

// Auto white balance functions
pub use auto_wb::{
    auto_white_balance, auto_white_balance_avg, auto_white_balance_no_clip,
    auto_white_balance_percentile,
};

// Compute-only functions (for GPU optimization)
pub use compute::{
    compute_wb_multipliers_avg, compute_wb_multipliers_gray_pixel,
    compute_wb_multipliers_percentile,
};

// Temperature/Kelvin-based functions
pub use kelvin::{apply_white_balance_from_temperature, kelvin_to_rgb_multipliers};

// Internal types shared between modules
pub(crate) use auto_wb::WbStats;

// Constants for subsampled data thresholds
pub(crate) const SUBSAMPLED_GRAY_THRESHOLD: usize = 16;
pub(crate) const SUBSAMPLED_HIGHLIGHT_THRESHOLD: usize = 2;
