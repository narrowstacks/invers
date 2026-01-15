//! Histogram analysis types for curves-based pipeline.

/// Per-channel histogram analysis results (curves-based).
///
/// Contains white point, black point, and mean for each channel
/// after histogram analysis.
#[derive(Debug, Clone, Default)]
pub struct CbChannelOrigins {
    /// White point (0-255 scale, brightest significant value)
    pub white_point: f32,

    /// Black point (0-255 scale, darkest significant value)
    pub black_point: f32,

    /// Mean point (0.0-1.0 scale)
    pub mean_point: f32,
}

/// Complete histogram analysis for all RGB channels.
#[derive(Debug, Clone, Default)]
pub struct CbHistogramAnalysis {
    /// Red channel analysis
    pub red: CbChannelOrigins,

    /// Green channel analysis
    pub green: CbChannelOrigins,

    /// Blue channel analysis
    pub blue: CbChannelOrigins,
}
