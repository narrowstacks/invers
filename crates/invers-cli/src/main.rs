use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod commands;

use commands::{
    cmd_analyze, cmd_batch, cmd_convert, cmd_init, cmd_preset_create, cmd_preset_list,
    cmd_preset_show,
};

#[cfg(debug_assertions)]
use commands::{cmd_diagnose, cmd_test_params};

#[derive(Parser)]
#[command(name = "invers")]
#[command(version, about = "Film negative to positive converter", long_about = None)]
struct Cli {
    /// Show the path to the config file being used (if any) and exit
    #[arg(long)]
    config_path: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

// =============================================================================
// Shared argument structs to eliminate duplication between Convert and Batch
// =============================================================================

/// Common pipeline arguments shared between Convert and Batch commands
#[derive(Args, Clone, Debug)]
pub struct PipelineArgs {
    /// Pipeline mode: "legacy" (default), "research", or "cb"
    /// Research pipeline uses density balance BEFORE inversion for better color accuracy
    /// CB pipeline uses curve-based processing for film-like rendering
    #[arg(long, value_name = "MODE", default_value = "legacy")]
    pub pipeline: String,

    /// [RESEARCH] Density balance red exponent (R^db_r)
    /// Typical range: 0.8-1.3, default 1.05
    #[arg(long, value_name = "FLOAT")]
    pub db_red: Option<f32>,

    /// [RESEARCH] Density balance blue exponent (B^db_b)
    /// Typical range: 0.7-1.1, default 0.90
    #[arg(long, value_name = "FLOAT")]
    pub db_blue: Option<f32>,

    /// [RESEARCH] Neutral point ROI for auto-calculating density balance (x,y,width,height)
    /// Sample a known gray area to auto-compute density balance
    #[arg(long, value_name = "X,Y,W,H")]
    pub neutral_roi: Option<String>,

    /// [CB] Tone profile preset (standard, logarithmic, log-rich, log-flat, linear,
    /// linear-gamma, linear-flat, linear-deep, all-soft, all-hard, highlight-hard,
    /// highlight-soft, shadow-hard, shadow-soft, auto)
    #[arg(long, value_name = "PROFILE")]
    pub cb_tone: Option<String>,

    /// [CB] Enhanced profile/LUT (none, natural, frontier, crystal, pakon)
    #[arg(long, value_name = "PROFILE")]
    pub cb_lut: Option<String>,

    /// [CB] Color model (none, basic, frontier, noritsu, bw)
    #[arg(long, value_name = "MODEL")]
    pub cb_color: Option<String>,

    /// [CB] Film character (none, generic, kodak, fuji, cinestill-50d, cinestill-800t)
    #[arg(long, value_name = "CHARACTER")]
    pub cb_film: Option<String>,

    /// [CB] White balance preset (none, auto, neutral, warm, cool, mix, standard, kodak, fuji, cine-t, cine-d)
    #[arg(long, value_name = "PRESET")]
    pub cb_wb: Option<String>,
}

/// Debug-only arguments (only available in debug builds)
#[cfg(debug_assertions)]
#[derive(Args, Clone, Debug, Default)]
pub struct DebugArgs {
    /// [DEBUG] Skip tone curve application
    #[arg(long)]
    pub no_tonecurve: bool,

    /// [DEBUG] Skip color matrix correction
    #[arg(long)]
    pub no_colormatrix: bool,

    /// [DEBUG] Inversion mode override: "mask-aware", "linear", "log", "divide-blend", or "bw"
    #[arg(long, value_name = "MODE")]
    pub inversion: Option<String>,

    /// [DEBUG] Enable automatic white balance correction
    #[arg(long)]
    pub auto_wb: bool,

    /// [DEBUG] Strength of auto white balance correction (0.0-1.0)
    #[arg(long, value_name = "FLOAT", default_value = "1.0")]
    pub auto_wb_strength: f32,

    /// [DEBUG] Auto white balance mode: "gray", "avg", or "pct" (percentile-based)
    /// "pct" uses 98th percentile (robust white patch) - robust for varied scenes
    #[arg(long, value_name = "MODE", default_value = "gray")]
    pub auto_wb_mode: String,

    /// [DEBUG] Tone curve type: "neutral", "log", "cinematic", "linear", "asymmetric"
    /// "log"/"cinematic" provides cinematic log-style tone profile
    #[arg(long, value_name = "TYPE")]
    pub tone_curve: Option<String>,

    /// [DEBUG] Enable debug output (detailed pipeline parameters)
    #[arg(long)]
    pub debug: bool,
}

/// Placeholder for release builds - provides default values
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

#[derive(Subcommand)]
enum Commands {
    /// Convert negative image(s) to positive
    Convert {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output directory or file path
        #[arg(short, long, value_name = "PATH")]
        out: Option<PathBuf>,

        /// Film preset file
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// Scan profile file
        #[arg(short, long, value_name = "FILE")]
        scan_profile: Option<PathBuf>,

        /// Export format (tiff16 or dng)
        #[arg(long, value_name = "FORMAT", default_value = "tiff16")]
        export: String,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Use 'invers analyze' to determine these values, then reuse across a roll
        #[arg(long, value_name = "R,G,B")]
        base: Option<String>,

        /// Suppress non-essential output (timing, progress messages)
        #[arg(long)]
        silent: bool,

        /// Force CPU-only processing (GPU is used by default when available)
        #[arg(long)]
        cpu: bool,

        /// Enable verbose output (config loading, processing details)
        #[arg(short, long)]
        verbose: bool,

        /// Pipeline and processing options
        #[command(flatten)]
        pipeline_args: PipelineArgs,

        /// Debug options (only available in debug builds)
        #[cfg(debug_assertions)]
        #[command(flatten)]
        debug_args: DebugArgs,
    },

    /// Analyze image and estimate film base color
    ///
    /// Use this command to inspect an image and determine base RGB values
    /// that can be reused across multiple frames from the same roll of film.
    Analyze {
        /// Input file to analyze
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Base estimation method: "regions" (default) or "border"
        #[arg(long, value_name = "METHOD", default_value = "regions")]
        base_method: String,

        /// Border percentage for "border" base method (1-25%, default: 5)
        #[arg(long, value_name = "PERCENT", default_value = "5.0")]
        border_percent: f32,

        /// Output as JSON (machine-readable)
        #[arg(long)]
        json: bool,

        /// Save analysis to file
        #[arg(short, long, value_name = "FILE")]
        save: Option<PathBuf>,

        /// Show detailed analysis output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Batch process multiple files with shared settings
    ///
    /// By default, assumes all images are from the same roll of film and shares
    /// the base color estimation from the first image across all files.
    /// Use --per-image to estimate base for each file independently.
    Batch {
        /// Input files or directories
        #[arg(value_name = "INPUTS")]
        inputs: Vec<PathBuf>,

        /// Recursively search directories for images
        #[arg(short = 'r', long)]
        recursive: bool,

        /// Base estimation file (JSON from 'analyze --save')
        /// Takes priority over first-image estimation
        #[arg(long, value_name = "FILE")]
        base_from: Option<PathBuf>,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Use 'invers analyze' to determine these values
        #[arg(long, value_name = "R,G,B")]
        base: Option<String>,

        /// Estimate base per-image instead of sharing from first image
        /// By default, batch assumes all images are from the same roll
        #[arg(long)]
        per_image: bool,

        /// Film preset file
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// Scan profile file
        #[arg(short, long, value_name = "FILE")]
        scan_profile: Option<PathBuf>,

        /// Export format (tiff16 or dng)
        #[arg(long, value_name = "FORMAT", default_value = "tiff16")]
        export: String,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Output directory
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Number of parallel threads
        #[arg(short = 'j', long, value_name = "N")]
        threads: Option<usize>,

        /// Suppress non-essential output (timing, progress messages)
        #[arg(long)]
        silent: bool,

        /// Enable verbose output (base estimation details, config loading)
        #[arg(short, long)]
        verbose: bool,

        /// Force CPU-only processing (GPU is used by default when available)
        #[arg(long)]
        cpu: bool,

        /// Pipeline and processing options
        #[command(flatten)]
        pipeline_args: PipelineArgs,

        /// Debug options (only available in debug builds)
        #[cfg(debug_assertions)]
        #[command(flatten)]
        debug_args: DebugArgs,
    },

    /// Manage film presets
    Preset {
        #[command(subcommand)]
        action: PresetAction,
    },

    /// Initialize user configuration directory with default presets
    ///
    /// Copies default configuration and preset files to ~/invers/.
    /// Safe to run multiple times - won't overwrite existing files.
    Init {
        /// Force overwrite of existing files
        #[arg(long)]
        force: bool,
    },

    // === Debug-only commands (only available in debug builds) ===
    /// [DEBUG] Diagnose and compare our conversion with third-party software
    #[cfg(debug_assertions)]
    Diagnose {
        /// Original negative image
        #[arg(value_name = "ORIGINAL")]
        original: PathBuf,

        /// Third-party converted image to compare against
        #[arg(value_name = "THIRD_PARTY")]
        third_party: PathBuf,

        /// Film preset file (optional)
        #[arg(short, long, value_name = "FILE")]
        preset: Option<PathBuf>,

        /// ROI for base estimation (x,y,width,height)
        #[arg(long, value_name = "X,Y,W,H")]
        roi: Option<String>,

        /// Output directory for diagnostic images
        #[arg(short, long, value_name = "DIR")]
        out: Option<PathBuf>,

        /// Skip tone curve application
        #[arg(long)]
        no_tonecurve: bool,

        /// Skip color matrix correction
        #[arg(long)]
        no_colormatrix: bool,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(long, value_name = "FLOAT", default_value = "1.0")]
        exposure: f32,

        /// Enable debug output
        #[arg(long)]
        debug: bool,
    },

    /// [DEBUG] Test and optimize parameters against reference conversion
    #[cfg(debug_assertions)]
    TestParams {
        /// Original negative image
        #[arg(value_name = "ORIGINAL")]
        original: PathBuf,

        /// Reference conversion to match
        #[arg(value_name = "REFERENCE")]
        reference: PathBuf,

        /// Run grid search over parameter space
        #[arg(long)]
        grid: bool,

        /// Use parallel grid search (much faster for large grids)
        #[arg(long)]
        parallel: bool,

        /// Use adaptive search (coarse->fine refinement, most efficient)
        #[arg(long)]
        adaptive: bool,

        /// Target score for adaptive search (stops when reached)
        #[arg(long, value_name = "FLOAT", default_value = "0.05")]
        target_score: f32,

        /// Maximum refinement iterations for adaptive search
        #[arg(long, value_name = "N", default_value = "5")]
        max_iterations: usize,

        /// Number of top results to show
        #[arg(long, value_name = "N", default_value = "5")]
        top: usize,

        /// Output JSON file for results
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Save test conversion output for visual comparison
        #[arg(long, value_name = "FILE")]
        save_output: Option<PathBuf>,

        /// Test specific clip percent value (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        clip_percent: Option<f32>,

        /// Test specific tone curve strength (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        tone_strength: Option<f32>,

        /// Test specific exposure compensation (skip grid search)
        #[arg(long, value_name = "FLOAT")]
        exposure: Option<f32>,
    },
}

#[derive(Subcommand)]
enum PresetAction {
    /// List available presets
    List {
        /// Directory to list presets from
        #[arg(short, long, value_name = "DIR")]
        dir: Option<PathBuf>,
    },

    /// Show details of a preset
    Show {
        /// Preset name or file path
        preset: String,
    },

    /// Create a new preset template
    Create {
        /// Output file path
        output: PathBuf,

        /// Preset name
        #[arg(short, long)]
        name: String,
    },
}

fn main() {
    let cli = Cli::parse();

    // Handle --config-path flag
    if cli.config_path {
        let handle = invers_core::config::pipeline_config_handle();
        match &handle.source {
            Some(path) => println!("{}", path.display()),
            None => println!("No config file found (using built-in defaults)"),
        }
        return;
    }

    // Require a subcommand if --config-path wasn't used
    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            eprintln!("Error: A subcommand is required. Use --help for usage.");
            std::process::exit(1);
        }
    };

    let result = match command {
        Commands::Convert {
            input,
            out,
            preset,
            scan_profile,
            export,
            exposure,
            base,
            silent,
            cpu,
            verbose,
            pipeline_args,
            #[cfg(debug_assertions)]
            debug_args,
        } => {
            // In release builds, use default values for debug-only options
            #[cfg(not(debug_assertions))]
            let debug_args = DebugArgs::release_defaults();

            cmd_convert(
                input,
                out,
                preset,
                scan_profile,
                export,
                debug_args.no_tonecurve,
                debug_args.no_colormatrix,
                exposure,
                debug_args.inversion,
                base,
                silent,
                cpu,
                debug_args.auto_wb,
                debug_args.auto_wb_strength,
                debug_args.auto_wb_mode,
                debug_args.tone_curve,
                verbose,
                debug_args.debug,
                pipeline_args.pipeline,
                pipeline_args.db_red,
                pipeline_args.db_blue,
                pipeline_args.neutral_roi,
                pipeline_args.cb_tone,
                pipeline_args.cb_lut,
                pipeline_args.cb_color,
                pipeline_args.cb_film,
                pipeline_args.cb_wb,
            )
        }

        Commands::Analyze {
            input,
            roi,
            base_method,
            border_percent,
            json,
            save,
            verbose,
        } => cmd_analyze(input, roi, base_method, border_percent, json, save, verbose),

        Commands::Batch {
            inputs,
            recursive,
            base_from,
            base,
            per_image,
            preset,
            scan_profile,
            export,
            exposure,
            out,
            threads,
            silent,
            verbose,
            cpu,
            pipeline_args,
            #[cfg(debug_assertions)]
            debug_args,
        } => {
            // In release builds, use default values for debug-only options
            #[cfg(not(debug_assertions))]
            let debug_args = DebugArgs::release_defaults();

            cmd_batch(
                inputs,
                recursive,
                base_from,
                base,
                per_image,
                preset,
                scan_profile,
                export,
                exposure,
                out,
                threads,
                silent,
                verbose,
                cpu,
                debug_args.no_tonecurve,
                debug_args.no_colormatrix,
                debug_args.inversion,
                debug_args.auto_wb,
                debug_args.auto_wb_strength,
                debug_args.auto_wb_mode,
                debug_args.tone_curve,
                debug_args.debug,
                pipeline_args.pipeline,
                pipeline_args.db_red,
                pipeline_args.db_blue,
                pipeline_args.neutral_roi,
                pipeline_args.cb_tone,
                pipeline_args.cb_lut,
                pipeline_args.cb_color,
                pipeline_args.cb_film,
                pipeline_args.cb_wb,
            )
        }

        Commands::Preset { action } => match action {
            PresetAction::List { dir } => cmd_preset_list(dir),
            PresetAction::Show { preset } => cmd_preset_show(preset),
            PresetAction::Create { output, name } => cmd_preset_create(output, name),
        },

        Commands::Init { force } => cmd_init(force),

        #[cfg(debug_assertions)]
        Commands::Diagnose {
            original,
            third_party,
            preset,
            roi,
            out,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        } => cmd_diagnose(
            original,
            third_party,
            preset,
            roi,
            out,
            no_tonecurve,
            no_colormatrix,
            exposure,
            debug,
        ),

        #[cfg(debug_assertions)]
        Commands::TestParams {
            original,
            reference,
            grid,
            parallel,
            adaptive,
            target_score,
            max_iterations,
            top,
            output,
            save_output,
            clip_percent,
            tone_strength,
            exposure,
        } => cmd_test_params(
            original,
            reference,
            grid,
            parallel,
            adaptive,
            target_score,
            max_iterations,
            top,
            output,
            save_output,
            clip_percent,
            tone_strength,
            exposure,
        ),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
