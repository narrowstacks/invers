//! Processing parameter types for CLI commands.

use super::WhiteBalance;

/// Parameters for processing a single image.
/// Used by both convert and batch commands to avoid duplication.
#[derive(Clone)]
pub struct ProcessingParams {
    // Basic options
    pub export: String,
    pub exposure: f32,
    pub cpu_only: bool,
    pub silent: bool,
    pub verbose: bool,
    pub debug: bool,

    // White balance (user-facing unified interface)
    pub white_balance: WhiteBalance,

    // Pipeline options
    pub pipeline: String,
    pub db_red: Option<f32>,
    pub db_blue: Option<f32>,
    pub neutral_roi: Option<String>,

    // CB options
    pub cb_tone: Option<String>,
    pub cb_lut: Option<String>,
    pub cb_color: Option<String>,
    pub cb_film: Option<String>,
    pub cb_wb: Option<String>,

    // Debug options (always present, release builds use defaults)
    pub no_tonecurve: bool,
    pub no_colormatrix: bool,
    pub inversion: Option<String>,
    pub auto_wb: bool,
    pub auto_wb_strength: f32,
    pub auto_wb_mode: String,
    pub tone_curve: Option<String>,
}

impl Default for ProcessingParams {
    fn default() -> Self {
        Self {
            export: "tiff16".to_string(),
            exposure: 1.0,
            cpu_only: false,
            silent: false,
            verbose: false,
            debug: false,
            white_balance: WhiteBalance::Auto,
            pipeline: "legacy".to_string(),
            db_red: None,
            db_blue: None,
            neutral_roi: None,
            cb_tone: None,
            cb_lut: None,
            cb_color: None,
            cb_film: None,
            cb_wb: None,
            no_tonecurve: false,
            no_colormatrix: false,
            inversion: None,
            auto_wb: true,
            auto_wb_strength: 1.0,
            auto_wb_mode: "avg".to_string(),
            tone_curve: None,
        }
    }
}
