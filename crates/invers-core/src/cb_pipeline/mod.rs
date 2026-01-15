//! Curves-based processing pipeline
//!
//! Implements a curve-based negative-to-positive conversion pipeline inspired by
//! Negative Lab Pro's algorithms. This pipeline uses per-channel histogram analysis,
//! multiple white balance methods, and sigmoid-based tone adjustments.
//!
//! Key features:
//! - Per-channel histogram analysis to find white/black points
//! - 5 different white balance application methods
//! - 3 white balance tonality modes
//! - Sigmoid contrast via tanh/atanh
//! - Shadow/highlight toning
//! - Configurable processing order (color-first vs tones-first)

mod histogram;
mod layers;
mod process;
mod white_balance;

#[cfg(test)]
mod tests;

// Re-export public items
pub use histogram::analyze_histogram;
pub use process::process_image_cb;
pub use white_balance::{
    analyze_wb_points, calculate_gamma_balance, calculate_linear_balance, calculate_wb_gamma,
    calculate_wb_offsets, calculate_wb_preset_offsets, AnalyzedWbPoints,
};

// ============================================================
// Math Utilities
// ============================================================

/// Compute atanh (inverse hyperbolic tangent)
/// atanh(x) = 0.5 * ln((1+x)/(1-x))
#[inline]
fn atanh(x: f32) -> f32 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

/// Compute log base b of x: log_b(x) = ln(x) / ln(b)
#[inline]
pub(crate) fn logb(base: f32, x: f32) -> f32 {
    if base <= 0.0 || base == 1.0 || x <= 0.0 {
        return 1.0;
    }
    x.ln() / base.ln()
}

/// Clamp value to 0.0-1.0 range
#[inline]
pub(crate) fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

// ============================================================
// Sigmoid Contrast Functions (tanh-based)
// ============================================================

/// Core sigmoid function using tanh
/// sig(contrast, midpoint, x) = tanh(0.5 * contrast * (x - midpoint))
#[inline]
fn sigmoid_core(contrast: f32, midpoint: f32, x: f32) -> f32 {
    (0.5 * contrast * (x - midpoint)).tanh()
}

/// Apply sigmoidal contrast
/// Normalizes the tanh output to 0-1 range
pub(crate) fn apply_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = if range != 0.0 { range } else { 1.0 };
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);
    let sig_x = sigmoid_core(contrast, midpoint, x / r);

    ((sig_x - sig_0) / (sig_1 - sig_0)) * r
}

/// Inverse sigmoidal contrast
pub(crate) fn apply_inverse_sigmoid(contrast: f32, midpoint: f32, x: f32, range: f32) -> f32 {
    let r = if range != 0.0 { range } else { 1.0 };
    let sig_0 = sigmoid_core(contrast, midpoint, 0.0);
    let sig_1 = sigmoid_core(contrast, midpoint, 1.0);

    // Calculate the argument for atanh
    let mut arg = (sig_1 - sig_0) * x / r + sig_0;

    // Clamp to valid atanh range (-1, 1)
    arg = arg.clamp(-0.9999, 0.9999);

    (midpoint + 2.0 / contrast * atanh(arg)) * r
}
