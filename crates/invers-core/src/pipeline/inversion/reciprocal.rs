//! Reciprocal inversion for the research pipeline
//!
//! This module contains the simple reciprocal inversion used after
//! density-balance normalization in the research pipeline.

use crate::models::ConvertOptions;

/// Apply inversion to convert normalized negative to positive.
///
/// After dividing by base and applying density balance, values are normalized:
/// - Film base (brightest) -> 1.0
/// - Scene shadows (near base brightness) -> close to 1.0
/// - Scene highlights (darkest) -> much less than 1.0
///
/// The inversion maps this to positive space:
/// - Film base (1.0) -> 0.0 (black, representing scene shadows)
/// - Scene shadows (~0.9) -> small positive value (dark)
/// - Scene highlights (~0.3) -> large positive value (bright)
///
/// Formula: positive = 1.0 - normalized_negative
/// This is equivalent to (base - pixel) / base after the normalization step.
pub fn apply_reciprocal_inversion(
    data: &mut [f32],
    options: &ConvertOptions,
) -> Result<(), String> {
    for pixel in data.chunks_exact_mut(3) {
        // Inversion: positive = 1.0 - normalized
        // This correctly maps film base (1.0) to black and scene highlights (<1.0) to bright
        pixel[0] = 1.0 - pixel[0];
        pixel[1] = 1.0 - pixel[1];
        pixel[2] = 1.0 - pixel[2];
    }

    if options.debug {
        let stats = crate::pipeline::compute_stats(data);
        eprintln!(
            "[RESEARCH] After inversion - min: {:.6}, max: {:.6}, mean: {:.6}",
            stats.0, stats.1, stats.2
        );
    }

    Ok(())
}
