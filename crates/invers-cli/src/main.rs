//! Invers CLI - Film negative to positive converter
//!
//! This is the command-line interface for the Invers film conversion tool.
//! It provides commands for:
//!
//! - `convert`: Convert a single negative image to positive
//! - `batch`: Process multiple images with shared settings
//! - `analyze`: Analyze an image to estimate film base color
//! - `init`: Initialize user configuration directory
//!
//! Debug-only commands (available in debug builds):
//! - `diagnose`: Compare conversion against third-party software
//! - `test-params`: Optimize parameters against a reference

use clap::{CommandFactory, Parser, Subcommand};
use clap_complete::{generate, Shell};
use std::io;
use std::path::PathBuf;

mod commands;

use commands::{cmd_analyze, cmd_batch, cmd_convert, cmd_init};
use invers_cli::args::{DebugArgs, PipelineArgs};
use invers_cli::WhiteBalance;

#[cfg(debug_assertions)]
use commands::{cmd_diagnose, cmd_test_params};

#[derive(Parser)]
#[command(name = "invers")]
#[command(version, about = "Film negative to positive converter", long_about = None)]
#[command(
    after_help = "Use --help with a subcommand for more options. Advanced/research flags are hidden by default."
)]
struct Cli {
    /// Show the path to the config file being used (if any) and exit
    #[arg(long)]
    config_path: bool,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert negative image(s) to positive
    #[command(
        after_help = "Advanced/research flags are available but hidden. Use --pipeline, --cb-*, --db-* for experimental features."
    )]
    Convert {
        /// Input file or directory
        #[arg(value_name = "INPUT")]
        input: PathBuf,

        /// Output directory or file path
        #[arg(short, long, value_name = "PATH", help_heading = "Output Options")]
        out: Option<PathBuf>,

        /// Export format (tiff16 or dng)
        #[arg(
            long,
            value_name = "FORMAT",
            default_value = "tiff16",
            help_heading = "Output Options"
        )]
        export: String,

        /// White balance preset: auto (default), none, neutral, warm, cool
        #[arg(long, short = 'w', value_enum, default_value_t = WhiteBalance::Auto, help_heading = "Processing Options")]
        white_balance: WhiteBalance,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(
            long,
            value_name = "FLOAT",
            default_value = "1.0",
            help_heading = "Processing Options"
        )]
        exposure: f32,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Use 'invers analyze' to determine these values, then reuse across a roll
        #[arg(long, value_name = "R,G,B", help_heading = "Processing Options")]
        base: Option<String>,

        /// Force black and white conversion mode
        #[arg(long, help_heading = "Processing Options")]
        bw: bool,

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
    #[command(
        after_help = "Advanced/research flags are available but hidden. Use --pipeline, --cb-*, --db-* for experimental features."
    )]
    Batch {
        /// Input files or directories
        #[arg(value_name = "INPUTS")]
        inputs: Vec<PathBuf>,

        /// Recursively search directories for images
        #[arg(short = 'r', long)]
        recursive: bool,

        /// Base estimation file (JSON from 'analyze --save')
        /// Takes priority over first-image estimation
        #[arg(long, value_name = "FILE", help_heading = "Base Estimation")]
        base_from: Option<PathBuf>,

        /// Manual base RGB values (comma-separated: R,G,B)
        /// Use 'invers analyze' to determine these values
        #[arg(long, value_name = "R,G,B", help_heading = "Base Estimation")]
        base: Option<String>,

        /// Estimate base per-image instead of sharing from first image
        /// By default, batch assumes all images are from the same roll
        #[arg(long, help_heading = "Base Estimation")]
        per_image: bool,

        /// Export format (tiff16 or dng)
        #[arg(
            long,
            value_name = "FORMAT",
            default_value = "tiff16",
            help_heading = "Output Options"
        )]
        export: String,

        /// Output directory
        #[arg(short, long, value_name = "DIR", help_heading = "Output Options")]
        out: Option<PathBuf>,

        /// White balance preset: auto (default), none, neutral, warm, cool
        #[arg(long, short = 'w', value_enum, default_value_t = WhiteBalance::Auto, help_heading = "Processing Options")]
        white_balance: WhiteBalance,

        /// Exposure compensation multiplier (1.0 = no change, >1.0 = brighter)
        #[arg(
            long,
            value_name = "FLOAT",
            default_value = "1.0",
            help_heading = "Processing Options"
        )]
        exposure: f32,

        /// Force black and white conversion mode
        #[arg(long, help_heading = "Processing Options")]
        bw: bool,

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

        /// Dry run: list files that would be processed without actually processing
        #[arg(long)]
        dry_run: bool,

        /// Pipeline and processing options
        #[command(flatten)]
        pipeline_args: PipelineArgs,

        /// Debug options (only available in debug builds)
        #[cfg(debug_assertions)]
        #[command(flatten)]
        debug_args: DebugArgs,
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

    /// Generate shell completions for bash, zsh, fish, or powershell
    ///
    /// To install completions, pipe the output to the appropriate file:
    ///   bash: invers completions bash > ~/.bash_completion.d/invers
    ///   zsh:  invers completions zsh > ~/.zfunc/_invers
    ///   fish: invers completions fish > ~/.config/fish/completions/invers.fish
    Completions {
        /// Shell to generate completions for
        #[arg(value_enum)]
        shell: Shell,
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
            export,
            white_balance,
            exposure,
            base,
            silent,
            cpu,
            verbose,
            bw,
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
                export,
                white_balance,
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
                bw,
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
            export,
            white_balance,
            exposure,
            out,
            threads,
            silent,
            verbose,
            cpu,
            dry_run,
            bw,
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
                export,
                white_balance,
                exposure,
                out,
                threads,
                silent,
                verbose,
                cpu,
                dry_run,
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
                bw,
            )
        }

        Commands::Init { force } => cmd_init(force),

        Commands::Completions { shell } => {
            let mut cmd = Cli::command();
            generate(shell, &mut cmd, "invers", &mut io::stdout());
            Ok(())
        }

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
