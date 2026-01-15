//! Tone curve implementations
//!
//! This module contains the core curve algorithms:
//! - S-curves for natural film-like contrast
//! - Log-based curves for cinematic shadow lift
//! - Asymmetric curves for film-like toe/shoulder response

use crate::models::ToneCurveParams;
use rayon::prelude::*;

use super::{clamp_to_working_range, PARALLEL_THRESHOLD};

/// Apply S-curve tone mapping for natural film-like contrast
/// Strength: 0.0 = no curve (linear), 1.0 = maximum curve
///
/// Uses parallel processing for large images (>100k values)
pub fn apply_s_curve(data: &mut [f32], strength: f32) {
    // Clamp strength to valid range
    let strength = strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        // Effectively linear, no transformation needed
        return;
    }

    if data.len() >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for value in chunk.iter_mut() {
                *value = apply_s_curve_point(*value, strength);
            }
        });
    } else {
        // Sequential for small images
        for value in data.iter_mut() {
            *value = apply_s_curve_point(*value, strength);
        }
    }
}

/// Apply S-curve transformation to a single value
/// Uses a modified smoothstep function with adjustable contrast
pub fn apply_s_curve_point(x: f32, strength: f32) -> f32 {
    // Clamp input to valid range
    let x = x.clamp(0.0, 1.0);

    // Blend between linear and S-curve based on strength
    // S-curve uses a smoothstep-like function: 3x^2 - 2x^3
    // For more contrast, we can use higher-order polynomials

    // Calculate S-curve value using smoothstep
    let s_value = if x < 0.5 {
        // Shadow region: lift shadows slightly
        let t = x * 2.0;
        let smooth = t * t * (3.0 - 2.0 * t);
        smooth * 0.5
    } else {
        // Highlight region: compress highlights slightly
        let t = (x - 0.5) * 2.0;
        let smooth = t * t * (3.0 - 2.0 * t);
        0.5 + smooth * 0.5
    };

    // Apply contrast adjustment based on strength
    // Stronger strength = more pronounced S-curve
    let contrast_factor = 1.0 + strength * 0.5;
    let adjusted = (s_value - 0.5) * contrast_factor + 0.5;

    // Blend between original linear value and S-curve
    let result = x * (1.0 - strength) + adjusted * strength;

    clamp_to_working_range(result)
}

/// Apply log-based tone curve similar to cinematic "CINEMATIC - Log" profile
///
/// This curve uses a logarithmic function that:
/// - Lifts shadows significantly (log approaches 0 slowly)
/// - Keeps midtones relatively stable
/// - Softly compresses highlights
///
/// Formula: output = log(1 + input * k) / log(1 + k)
/// where k controls the curve shape (higher k = more aggressive shadow lift)
pub fn apply_log_curve(data: &mut [f32], strength: f32) {
    let strength = strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        return; // Effectively linear
    }

    // k controls the log curve shape
    // k=10 gives moderate lift, k=50 gives aggressive lift
    // We blend based on strength: k ranges from 5 (subtle) to 30 (strong)
    let k = 5.0 + strength * 25.0;
    let log_denom = (1.0 + k).ln();

    if data.len() >= PARALLEL_THRESHOLD {
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for value in chunk.iter_mut() {
                *value = apply_log_curve_point(*value, strength, k, log_denom);
            }
        });
    } else {
        for value in data.iter_mut() {
            *value = apply_log_curve_point(*value, strength, k, log_denom);
        }
    }
}

/// Apply log curve transformation to a single value
fn apply_log_curve_point(x: f32, strength: f32, k: f32, log_denom: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);

    // Log curve: log(1 + x*k) / log(1 + k)
    // This maps 0->0 and 1->1 but lifts everything in between
    let log_value = (1.0 + x * k).ln() / log_denom;

    // Blend between linear and log curve
    let result = x * (1.0 - strength) + log_value * strength;

    clamp_to_working_range(result)
}

/// Apply asymmetric film-like tone curve
///
/// This curve has three distinct regions:
/// - Toe (shadows): Lifts shadows using a gamma-like curve
/// - Mid (linear): Passes through unchanged for natural midtones
/// - Shoulder (highlights): Compresses highlights using soft-clip
///
/// The result is more film-like than symmetric S-curves because real film
/// has different response characteristics in shadows vs highlights.
///
/// Uses parallel processing for large images (>100k values)
pub fn apply_asymmetric_curve(data: &mut [f32], params: &ToneCurveParams) {
    let strength = params.strength.clamp(0.0, 1.0);

    if strength < 0.01 {
        return; // Effectively linear
    }

    // Clamp parameters to valid ranges
    let toe_strength = params.toe_strength.clamp(0.0, 1.0);
    let shoulder_strength = params.shoulder_strength.clamp(0.0, 1.0);
    let toe_length = params.toe_length.clamp(0.05, 0.45);
    let shoulder_start = params.shoulder_start.clamp(0.55, 0.95);

    if data.len() >= PARALLEL_THRESHOLD {
        // Parallel processing for large images
        const CHUNK_SIZE: usize = 256 * 3;
        data.par_chunks_mut(CHUNK_SIZE).for_each(|chunk| {
            for value in chunk.iter_mut() {
                let curved = apply_asymmetric_curve_point(
                    *value,
                    toe_strength,
                    shoulder_strength,
                    toe_length,
                    shoulder_start,
                );
                *value = clamp_to_working_range(*value * (1.0 - strength) + curved * strength);
            }
        });
    } else {
        // Sequential for small images
        for value in data.iter_mut() {
            let curved = apply_asymmetric_curve_point(
                *value,
                toe_strength,
                shoulder_strength,
                toe_length,
                shoulder_start,
            );
            *value = clamp_to_working_range(*value * (1.0 - strength) + curved * strength);
        }
    }
}

/// Apply asymmetric curve transformation to a single value
///
/// Implements a piecewise curve:
/// - x < toe_length: Toe region with shadow lift (gamma < 1)
/// - toe_length <= x <= shoulder_start: Linear passthrough
/// - x > shoulder_start: Shoulder region with highlight compression
pub(crate) fn apply_asymmetric_curve_point(
    x: f32,
    toe_strength: f32,
    shoulder_strength: f32,
    toe_length: f32,
    shoulder_start: f32,
) -> f32 {
    let x = x.clamp(0.0, 1.0);

    if x < toe_length {
        // Toe region: lift shadows
        // Use power function: output = toe_length * (x / toe_length)^(1/gamma)
        // where gamma > 1 for shadow lift (we use 1/(1 + toe_strength))
        let gamma = 1.0 / (1.0 + toe_strength * 1.5);
        let normalized = x / toe_length;
        let lifted = normalized.powf(gamma);

        // Scale back to toe_length and apply smooth transition
        // The output at toe_length should equal toe_length for continuity
        toe_length * lifted
    } else if x > shoulder_start {
        // Shoulder region: compress highlights
        // Use soft-clip: output = shoulder_start + (1 - shoulder_start) * (1 - (1 - t)^gamma)
        // where t = (x - shoulder_start) / (1 - shoulder_start)
        let gamma = 1.0 + shoulder_strength * 2.0;
        let range = 1.0 - shoulder_start;
        let normalized = (x - shoulder_start) / range;
        let compressed = 1.0 - (1.0 - normalized).powf(gamma);

        // Scale back to remaining range
        shoulder_start + range * compressed
    } else {
        // Linear mid region: pass through unchanged
        x
    }
}
