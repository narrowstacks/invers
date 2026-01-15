//! HSL adjustments for scanner-specific color correction.

use serde::{Deserialize, Serialize};

/// 8-color HSL adjustments (Camera Raw style)
///
/// Adjusts Hue, Saturation, and Luminance for 8 color ranges:
/// - R (Reds): Hue ~0-30, 330-360
/// - O (Oranges): Hue ~30-60
/// - Y (Yellows): Hue ~60-90
/// - G (Greens): Hue ~90-150
/// - A (Aquas/Cyans): Hue ~150-210
/// - B (Blues): Hue ~210-270
/// - P (Purples): Hue ~270-300
/// - M (Magentas): Hue ~300-330
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HslAdjustments {
    /// Hue shifts for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub hue: [f32; 8],

    /// Saturation adjustments for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub saturation: [f32; 8],

    /// Luminance adjustments for each of the 8 color ranges (-100 to +100)
    /// Order: [R, O, Y, G, A, B, P, M]
    #[serde(default)]
    pub luminance: [f32; 8],
}

impl Default for HslAdjustments {
    fn default() -> Self {
        Self {
            hue: [0.0; 8],
            saturation: [0.0; 8],
            luminance: [0.0; 8],
        }
    }
}

impl HslAdjustments {
    /// Check if any adjustments are non-zero
    pub fn has_adjustments(&self) -> bool {
        self.hue.iter().any(|&v| v.abs() > 0.001)
            || self.saturation.iter().any(|&v| v.abs() > 0.001)
            || self.luminance.iter().any(|&v| v.abs() > 0.001)
    }

    /// Color range indices
    pub const REDS: usize = 0;
    pub const ORANGES: usize = 1;
    pub const YELLOWS: usize = 2;
    pub const GREENS: usize = 3;
    pub const AQUAS: usize = 4;
    pub const BLUES: usize = 5;
    pub const PURPLES: usize = 6;
    pub const MAGENTAS: usize = 7;
}
