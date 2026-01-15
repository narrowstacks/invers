//! Main tone curve application function
//!
//! This module contains the primary dispatcher that routes to the appropriate
//! curve algorithm based on the curve type parameter.

use crate::models::ToneCurveParams;

use super::curves::{apply_asymmetric_curve, apply_log_curve, apply_s_curve};

/// Apply tone curve based on the specified curve type
///
/// Supported curve types:
/// - "linear": No transformation
/// - "asymmetric": Film-like curve with separate toe/shoulder controls
/// - "log" | "cinematic": Log-based curve for cinematic shadow lift
/// - "neutral" | "s-curve": Standard S-curve for natural contrast
/// - Unknown types fall back to S-curve
pub fn apply_tone_curve(data: &mut [f32], curve_params: &ToneCurveParams) {
    match curve_params.curve_type.as_str() {
        "linear" => {
            // No transformation needed
        }
        "asymmetric" => {
            // Apply asymmetric film-like curve with separate toe/shoulder controls
            apply_asymmetric_curve(data, curve_params);
        }
        "log" | "cinematic" => {
            // Apply log-based curve similar to cinematic "CINEMATIC - Log" profile
            // Lifts shadows, compresses highlights, cinematic look
            apply_log_curve(data, curve_params.strength);
        }
        "neutral" | "s-curve" => {
            // Apply S-curve with configurable strength
            apply_s_curve(data, curve_params.strength);
        }
        _ => {
            // Unknown curve type, apply neutral S-curve
            apply_s_curve(data, curve_params.strength);
        }
    }
}
