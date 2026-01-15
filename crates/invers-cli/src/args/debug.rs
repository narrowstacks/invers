//! Debug-only argument structs for CLI commands.

#[cfg(debug_assertions)]
use clap::Args;

/// Debug-only arguments (only available in debug builds).
/// These flags are hidden from default help output.
#[cfg(debug_assertions)]
#[derive(Args, Clone, Debug, Default)]
pub struct DebugArgs {
    /// [DEBUG] Skip tone curve application
    #[arg(long, hide = true)]
    pub no_tonecurve: bool,

    /// [DEBUG] Skip color matrix correction
    #[arg(long, hide = true)]
    pub no_colormatrix: bool,

    /// [DEBUG] Inversion mode override: "mask-aware", "linear", "log", "divide-blend", or "bw"
    #[arg(long, value_name = "MODE", hide = true)]
    pub inversion: Option<String>,

    /// [DEBUG] Enable automatic white balance correction
    #[arg(long, hide = true)]
    pub auto_wb: bool,

    /// [DEBUG] Strength of auto white balance correction (0.0-1.0)
    #[arg(long, value_name = "FLOAT", default_value = "1.0", hide = true)]
    pub auto_wb_strength: f32,

    /// [DEBUG] Auto white balance mode: "gray", "avg", or "pct" (percentile-based)
    /// "pct" uses 98th percentile (robust white patch) - robust for varied scenes
    #[arg(long, value_name = "MODE", default_value = "gray", hide = true)]
    pub auto_wb_mode: String,

    /// [DEBUG] Tone curve type: "neutral", "log", "cinematic", "linear", "asymmetric"
    /// "log"/"cinematic" provides cinematic log-style tone profile
    #[arg(long, value_name = "TYPE", hide = true)]
    pub tone_curve: Option<String>,

    /// [DEBUG] Enable debug output (detailed pipeline parameters)
    #[arg(long, hide = true)]
    pub debug: bool,
}

/// Placeholder for release builds - provides default values.
#[cfg(not(debug_assertions))]
#[derive(Clone, Debug, Default)]
pub struct DebugArgs {
    pub no_tonecurve: bool,
    pub no_colormatrix: bool,
    pub inversion: Option<String>,
    pub auto_wb: bool,
    pub auto_wb_strength: f32,
    pub auto_wb_mode: String,
    pub tone_curve: Option<String>,
    pub debug: bool,
}

#[cfg(not(debug_assertions))]
impl DebugArgs {
    pub fn release_defaults() -> Self {
        Self {
            no_tonecurve: false,
            no_colormatrix: false,
            inversion: None,
            auto_wb: true,
            auto_wb_strength: 1.0,
            auto_wb_mode: "avg".to_string(),
            tone_curve: None,
            debug: false,
        }
    }
}
