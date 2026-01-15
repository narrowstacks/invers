//! White balance types for CLI interface.

/// White balance preset for CLI interface.
///
/// This provides a unified, user-friendly interface for white balance settings,
/// consolidating the various auto_wb flags into a single enum.
#[derive(Clone, Debug, Default, clap::ValueEnum)]
pub enum WhiteBalance {
    /// Auto white balance using average/gray world assumption (default)
    #[default]
    Auto,
    /// No white balance adjustment
    None,
    /// Neutral/gray world assumption with full strength
    Neutral,
    /// Warmer tones (reduced blue, slight red boost)
    Warm,
    /// Cooler tones (reduced red, slight blue boost)
    Cool,
}

/// White balance settings derived from the unified WhiteBalance preset.
#[derive(Clone, Debug)]
pub struct WhiteBalanceSettings {
    /// Whether auto white balance is enabled
    pub enabled: bool,
    /// Strength of the white balance correction (0.0-1.0)
    pub strength: f32,
    /// The auto WB mode to use
    pub mode: &'static str,
    /// Color temperature bias (positive = warmer, negative = cooler)
    pub temperature_bias: f32,
}

impl WhiteBalance {
    /// Convert the unified WhiteBalance preset to internal settings.
    ///
    /// Returns settings for:
    /// - `Auto`: Enable auto WB with strength 0.5, average mode
    /// - `None`: Disable auto WB
    /// - `Neutral`: Enable auto WB with "gray" mode for neutral tones
    /// - `Warm`: Enable auto WB with warm bias (reduce blue)
    /// - `Cool`: Enable auto WB with cool bias (reduce red)
    pub fn to_settings(&self) -> WhiteBalanceSettings {
        match self {
            WhiteBalance::Auto => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: 0.0,
            },
            WhiteBalance::None => WhiteBalanceSettings {
                enabled: false,
                strength: 0.0,
                mode: "avg",
                temperature_bias: 0.0,
            },
            WhiteBalance::Neutral => WhiteBalanceSettings {
                enabled: true,
                strength: 1.0,
                mode: "gray",
                temperature_bias: 0.0,
            },
            WhiteBalance::Warm => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: 500.0, // Positive = warmer (shift toward yellow/red)
            },
            WhiteBalance::Cool => WhiteBalanceSettings {
                enabled: true,
                strength: 0.5,
                mode: "avg",
                temperature_bias: -500.0, // Negative = cooler (shift toward blue)
            },
        }
    }
}
