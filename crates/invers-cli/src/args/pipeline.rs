//! Pipeline argument structs for CLI commands.

use clap::Args;

/// Common pipeline arguments shared between Convert and Batch commands.
/// These are advanced/research options hidden from default help output.
#[derive(Args, Clone, Debug)]
pub struct PipelineArgs {
    /// Pipeline mode: "legacy" (default), "research", or "cb"
    /// Research pipeline uses density balance BEFORE inversion for better color accuracy
    /// CB pipeline uses curve-based processing for film-like rendering
    #[arg(long, value_name = "MODE", default_value = "legacy", hide = true)]
    pub pipeline: String,

    /// [RESEARCH] Density balance red exponent (R^db_r)
    /// Typical range: 0.8-1.3, default 1.05
    #[arg(long, value_name = "FLOAT", hide = true)]
    pub db_red: Option<f32>,

    /// [RESEARCH] Density balance blue exponent (B^db_b)
    /// Typical range: 0.7-1.1, default 0.90
    #[arg(long, value_name = "FLOAT", hide = true)]
    pub db_blue: Option<f32>,

    /// [RESEARCH] Neutral point ROI for auto-calculating density balance (x,y,width,height)
    /// Sample a known gray area to auto-compute density balance
    #[arg(long, value_name = "X,Y,W,H", hide = true)]
    pub neutral_roi: Option<String>,

    /// [CB] Tone profile preset (standard, logarithmic, log-rich, log-flat, linear,
    /// linear-gamma, linear-flat, linear-deep, all-soft, all-hard, highlight-hard,
    /// highlight-soft, shadow-hard, shadow-soft, auto)
    #[arg(long, value_name = "PROFILE", hide = true)]
    pub cb_tone: Option<String>,

    /// [CB] Enhanced profile/LUT (none, natural, frontier, crystal, pakon)
    #[arg(long, value_name = "PROFILE", hide = true)]
    pub cb_lut: Option<String>,

    /// [CB] Color model (none, basic, frontier, noritsu, bw)
    #[arg(long, value_name = "MODEL", hide = true)]
    pub cb_color: Option<String>,

    /// [CB] Film character (none, generic, kodak, fuji, cinestill-50d, cinestill-800t)
    #[arg(long, value_name = "CHARACTER", hide = true)]
    pub cb_film: Option<String>,

    /// [CB] White balance preset (none, auto, neutral, warm, cool, mix, standard, kodak, fuji, cine-t, cine-d)
    #[arg(long, value_name = "PRESET", hide = true)]
    pub cb_wb: Option<String>,
}
