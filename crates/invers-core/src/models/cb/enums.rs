//! Core enums for curves-based pipeline white balance and layer order.

use serde::{Deserialize, Serialize};

/// White balance application method for curves-based pipeline.
///
/// These methods determine how per-channel WB offsets are applied
/// across the tonal range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbWbMethod {
    /// Linear fixed: WB offset varies linearly with protection at shadow/highlight extremes.
    /// Shadows (< 0.1): offset scales with value/0.1
    /// Midtones (0.1-0.8): full offset applied
    /// Highlights (> 0.8): offset blends toward preserving highlights
    #[default]
    LinearFixed,

    /// Linear dynamic: Simple additive offset (value + offset).
    /// Most aggressive color shift, uniform across all tones.
    LinearDynamic,

    /// Shadow-weighted: Applies WB via power function (value^(1/gamma)).
    /// Stronger effect in shadows, preserves highlights.
    /// Based on gamma balance calculation.
    ShadowWeighted,

    /// Highlight-weighted: Inverse power function (1 - (1-value)^gamma).
    /// Stronger effect in highlights, preserves shadows.
    HighlightWeighted,

    /// Midtone-weighted: Average of shadow and highlight weighted.
    /// Balanced effect across tonal range.
    MidtoneWeighted,
}

/// White balance tonality mode for curves-based pipeline.
///
/// Determines how temperature/tint adjustments are distributed
/// across RGB channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbWbTonality {
    /// Neutral density: Balanced distribution across all channels.
    /// R: -temp/2 - tint/2
    /// G: -temp/2 + tint/2
    /// B: +temp/2 - tint/2
    #[default]
    NeutralDensity,

    /// Subtract density: Direct subtraction model.
    /// R: -temp - tint
    /// G: -temp
    /// B: -tint
    SubtractDensity,

    /// Temperature/tint density: Simplified temp/tint model.
    /// R: -tint/2
    /// G: 0
    /// B: temp - tint/2
    TempTintDensity,
}

/// Processing layer order for curves-based pipeline.
///
/// Controls whether color/WB adjustments are applied before or after
/// tonal adjustments (brightness, contrast, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CbLayerOrder {
    /// Apply color/WB adjustments first, then tonal adjustments.
    /// More aggressive color correction.
    #[default]
    ColorFirst,

    /// Apply tonal adjustments first, then color/WB.
    /// Preserves more of the original color character.
    TonesFirst,
}
