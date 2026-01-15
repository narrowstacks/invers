//! Enums for conversion options.

use serde::{Deserialize, Serialize};

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    /// 16-bit linear TIFF
    #[default]
    Tiff16,

    /// Linear DNG
    LinearDng,
}

/// Bit depth handling policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BitDepthPolicy {
    /// Match input bit depth when possible
    #[default]
    MatchInput,

    /// Always use 16-bit output
    Force16Bit,

    /// Preserve maximum precision
    MaxPrecision,
}

/// Auto-levels histogram stretching mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutoLevelsMode {
    /// Stretch each channel independently (can shift colors)
    #[default]
    PerChannel,

    /// Use same stretch factor for all channels (preserves color relationships)
    /// Also known as PreserveSaturation mode
    Unified,

    /// Saturation-aware: reduces stretch for channels that would clip heavily
    SaturationAware,
}

/// Auto white balance mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AutoWbMode {
    /// Gray pixel detection - find pixels with similar R/G/B values
    /// Falls back to highlights, then to average
    #[default]
    GrayPixel,

    /// Average/Gray World - assume average of all pixels should be neutral
    Average,

    /// Percentile-based (Robust White Patch) - use high percentile as white reference
    /// More robust than max RGB, preserves more color character than gray world
    /// This is likely closest to the curves-based "AUTO AVG" behavior
    Percentile,
}

/// Inversion mode for negative-to-positive conversion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InversionMode {
    /// Linear inversion: (base - negative) / base
    Linear,

    /// Logarithmic inversion: 10^(log10(base) - log10(negative))
    Logarithmic,

    /// Divide-blend inversion:
    /// 1. Divide: pixel / base (per channel)
    /// 2. Apply gamma 2.2 (convert from linear to gamma-encoded)
    /// 3. Invert: 1.0 - result
    ///
    /// This mode mimics Photoshop's Divide blend mode workflow.
    DivideBlend,

    /// Orange mask-aware inversion for color negative film.
    ///
    /// This mode properly accounts for the orange mask present in color negative
    /// film. The mask exists because real-world dyes have impurities:
    /// - Magenta dye absorbs some blue light (not just green)
    /// - Cyan dye absorbs some green light (not just red)
    ///
    /// Film manufacturers add colored dye couplers to compensate, creating
    /// the characteristic orange mask. Simple inversion of this mask produces
    /// a blue cast in shadows.
    ///
    /// This mode:
    /// 1. Performs standard inversion: 1.0 - (pixel / base)
    /// 2. Calculates per-channel shadow floor based on mask characteristics
    /// 3. Applies shadow correction to neutralize the blue cast
    /// 4. Automatically skips color matrix (no longer needed)
    #[default]
    MaskAware,

    /// Simple B&W inversion for grayscale or monochrome images.
    ///
    /// This mode is optimized for black and white film:
    /// 1. Simple inversion: 1.0 - (pixel / base)
    /// 2. Sets black point slightly below the film base (with headroom)
    /// 3. Skips color-specific operations (color matrix, auto-color, etc.)
    ///
    /// The headroom parameter (default 5%) preserves shadow detail by not
    /// clipping the film base completely to black.
    BlackAndWhite,
}

/// Shadow lift mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShadowLiftMode {
    /// Fixed lift value
    Fixed,

    /// Percentile-based adaptive lift (e.g., lift 1st percentile to target)
    #[default]
    Percentile,

    /// No shadow lift
    None,
}

/// Pipeline mode selection
///
/// Controls which processing pipeline is used for negative-to-positive conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PipelineMode {
    /// Legacy pipeline with existing inversion modes and post-processing stages.
    /// This is the original Invers pipeline, kept for backward compatibility.
    #[default]
    Legacy,

    /// Research-based pipeline implementing densitometry principles.
    ///
    /// Key difference from Legacy: applies **density balance BEFORE inversion**
    /// using per-channel power functions to align characteristic curves.
    ///
    /// Pipeline stages:
    /// 1. Film base white balance (divide by base to normalize orange mask)
    /// 2. Density balance (per-channel power: R^db_r, G^1.0, B^db_b)
    /// 3. Reciprocal inversion (positive = k / negative)
    /// 4. Auto-levels (histogram normalization)
    /// 5. Tone curve
    /// 6. Export
    ///
    /// This approach eliminates color crossover between shadows and highlights
    /// by aligning the RGB characteristic curves before inversion.
    Research,

    /// Curves-based pipeline inspired by Negative Lab Pro algorithms.
    ///
    /// This pipeline implements a curve-based approach with multiple white balance
    /// methods and tonality modes. Key features:
    ///
    /// Pipeline stages:
    /// 1. Histogram analysis to find white/black points per channel
    /// 2. Film base normalization
    /// 3. Inversion via per-channel tone curves
    /// 4. White balance application (5 methods: linear, gamma-weighted)
    /// 5. Exposure/brightness/contrast via sigmoid curves
    /// 6. Shadow/highlight toning
    /// 7. Final curve application
    ///
    /// Supports two processing orders:
    /// - `colorFirst`: Apply WB before tone adjustments
    /// - `tonesFirst`: Apply tones before WB (preserves more color character)
    CbStyle,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inversion_mode_default() {
        let mode = InversionMode::default();
        assert_eq!(mode, InversionMode::MaskAware);
    }

    #[test]
    fn test_pipeline_mode_default() {
        let mode = PipelineMode::default();
        assert_eq!(mode, PipelineMode::Legacy);
    }

    #[test]
    fn test_shadow_lift_mode_default() {
        let mode = ShadowLiftMode::default();
        assert_eq!(mode, ShadowLiftMode::Percentile);
    }

    #[test]
    fn test_auto_levels_mode_default() {
        let mode = AutoLevelsMode::default();
        assert_eq!(mode, AutoLevelsMode::PerChannel);
    }

    #[test]
    fn test_auto_wb_mode_default() {
        let mode = AutoWbMode::default();
        assert_eq!(mode, AutoWbMode::GrayPixel);
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert_eq!(format, OutputFormat::Tiff16);
    }

    #[test]
    fn test_bit_depth_policy_default() {
        let policy = BitDepthPolicy::default();
        assert_eq!(policy, BitDepthPolicy::MatchInput);
    }
}
