//! WGSL shader sources embedded at compile time.

/// Container for all shader source code.
pub struct Shaders;

impl Shaders {
    /// Inversion operations (linear, log, divide-blend, mask-aware).
    pub const INVERSION: &'static str = include_str!("inversion.wgsl");

    /// Tone curve operations (S-curve, asymmetric).
    pub const TONE_CURVE: &'static str = include_str!("tone_curve.wgsl");

    /// Color matrix multiplication and gain application.
    pub const COLOR_MATRIX: &'static str = include_str!("color_matrix.wgsl");

    /// Histogram accumulation using atomics.
    pub const HISTOGRAM: &'static str = include_str!("histogram.wgsl");

    /// Colorspace conversions (RGB↔HSL, HSL adjustments).
    pub const COLOR_CONVERT: &'static str = include_str!("color_convert.wgsl");

    /// Utility operations (clamp, exposure, shadow lift, highlight compress).
    pub const UTILITY: &'static str = include_str!("utility.wgsl");
}
