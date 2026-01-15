//! Color management and transformations
//!
//! Provides colorspace conversions (RGB <-> HSL, RGB <-> LAB), ICC profile handling,
//! and working colorspace transformations.

mod conversions;
mod hsl;
mod lab;

#[cfg(test)]
mod tests;

// Re-export primary types
pub use hsl::Hsl;
pub use lab::Lab;

// Re-export colorspace enum and transform
pub use conversions::{transform_colorspace, Colorspace};

// Re-export HSL functions
pub use hsl::{hsl_array_to_rgb, hsl_to_rgb, rgb_array_to_hsl, rgb_to_hsl};

// Re-export LAB functions
pub use lab::{
    lab_array_to_rgb, lab_to_rgb, lab_to_rgb_with_colorspace, rgb_array_to_lab, rgb_to_lab,
    rgb_to_lab_with_colorspace,
};

// Re-export HSL adjustments
pub use hsl::apply_hsl_adjustments;
