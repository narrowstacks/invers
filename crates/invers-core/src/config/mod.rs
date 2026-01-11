//! Pipeline configuration management.
//!
//! This module provides configuration loading, global verbose flag management,
//! and the main pipeline configuration types.

mod defaults;
#[cfg(debug_assertions)]
mod testing;

// Re-export public types
pub use defaults::PipelineDefaults;
#[cfg(not(debug_assertions))]
pub use defaults::{SimplePipelineConfig, SimplePipelineDefaults};
#[cfg(debug_assertions)]
pub use testing::{ParameterTestDefaults, TestingConfig, TestingGridValues};

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};

use serde::Deserialize;

// Global verbose flag for controlling debug output
static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Set the global verbose flag. When true, debug messages will be printed.
pub fn set_verbose(verbose: bool) {
    VERBOSE.store(verbose, Ordering::SeqCst);
}

/// Check if verbose mode is enabled.
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::SeqCst)
}

/// Print a message to stderr only if verbose mode is enabled.
#[macro_export]
macro_rules! verbose_println {
    ($($arg:tt)*) => {
        if $crate::config::is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// Canonical list of candidate config file names we search for on disk.
const CONFIG_FILENAMES: &[&str] = &["pipeline.yml", "pipeline.yaml", "pipeline_defaults.yml"];

/// Public handle that stores the loaded configuration, its source path, and warnings.
pub struct PipelineConfigHandle {
    pub config: PipelineConfig,
    pub source: Option<PathBuf>,
    pub warnings: Vec<String>,
}

impl PipelineConfigHandle {
    fn with_config(config: PipelineConfig, source: Option<PathBuf>, warnings: Vec<String>) -> Self {
        Self {
            config,
            source,
            warnings,
        }
    }
}

/// Complete configuration file structure.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct PipelineConfig {
    pub defaults: PipelineDefaults,
    #[cfg(debug_assertions)]
    pub testing: TestingConfig,
}

impl PipelineConfig {
    #[allow(dead_code)] // Only used in debug builds
    fn sanitize(mut self) -> Self {
        self.defaults.sanitize();
        #[cfg(debug_assertions)]
        {
            if let Some(ref mut defaults) = self.testing.parameter_test_defaults {
                defaults.sanitize();
            }
            if let Some(ref mut grid) = self.testing.default_grid {
                grid.sanitize_with(TestingGridValues::default_grid());
            }
            if let Some(ref mut grid) = self.testing.minimal_grid {
                grid.sanitize_with(TestingGridValues::minimal_grid());
            }
            if let Some(ref mut grid) = self.testing.comprehensive_grid {
                grid.sanitize_with(TestingGridValues::comprehensive_grid());
            }
        }
        self
    }
}

/// Load configuration from disk, optionally forcing a specific path.
/// In release builds, only loads simplified config (6 essential options).
/// In debug builds, loads full config including testing parameters.
pub fn load_pipeline_config(custom_path: Option<&Path>) -> PipelineConfigHandle {
    let mut warnings = Vec::new();
    let candidates = get_config_candidates(custom_path);

    for candidate in candidates {
        if !candidate.exists() || !candidate.is_file() {
            continue;
        }

        match fs::read_to_string(&candidate) {
            Ok(contents) => {
                // In debug builds, load full config
                #[cfg(debug_assertions)]
                {
                    match serde_yaml::from_str::<PipelineConfig>(&contents) {
                        Ok(config) => {
                            let sanitized = config.sanitize();
                            let source = fs::canonicalize(&candidate).unwrap_or(candidate);
                            return PipelineConfigHandle::with_config(
                                sanitized,
                                Some(source),
                                warnings,
                            );
                        }
                        Err(err) => warnings.push(format!(
                            "Failed to parse pipeline config {}: {}",
                            candidate.display(),
                            err
                        )),
                    }
                }

                // In release builds, load simplified config only
                #[cfg(not(debug_assertions))]
                {
                    match serde_yaml::from_str::<SimplePipelineConfig>(&contents) {
                        Ok(simple_config) => {
                            let full_defaults = simple_config.defaults.to_full_defaults();
                            let config = PipelineConfig {
                                defaults: full_defaults,
                            };
                            let source = fs::canonicalize(&candidate).unwrap_or(candidate);
                            return PipelineConfigHandle::with_config(
                                config,
                                Some(source),
                                warnings,
                            );
                        }
                        Err(err) => warnings.push(format!(
                            "Failed to parse pipeline config {}: {}",
                            candidate.display(),
                            err
                        )),
                    }
                }
            }
            Err(err) => warnings.push(format!(
                "Failed to read pipeline config {}: {}",
                candidate.display(),
                err
            )),
        }
    }

    warnings.push("No pipeline config found; using built-in defaults.".to_string());
    PipelineConfigHandle::with_config(PipelineConfig::default(), None, warnings)
}

/// Get list of config file candidates to try
fn get_config_candidates(custom_path: Option<&Path>) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    if let Some(path) = custom_path {
        candidates.push(path.to_path_buf());
    }

    if let Ok(env_path) = std::env::var("INVERS_CONFIG") {
        candidates.push(PathBuf::from(env_path));
    }

    if let Ok(cwd) = std::env::current_dir() {
        for name in CONFIG_FILENAMES {
            candidates.push(cwd.join("config").join(name));
            candidates.push(cwd.join(name));
        }
    }

    if let Some(home_dir) = dirs::home_dir() {
        for name in CONFIG_FILENAMES {
            candidates.push(home_dir.join("invers").join(name));
        }
    }

    candidates
}

static PIPELINE_CONFIG_HANDLE: OnceLock<PipelineConfigHandle> = OnceLock::new();
static PRINT_CONFIG_ONCE: Once = Once::new();

/// Access the global pipeline configuration (loaded once per process).
pub fn pipeline_config_handle() -> &'static PipelineConfigHandle {
    PIPELINE_CONFIG_HANDLE.get_or_init(|| load_pipeline_config(None))
}

/// Print config source and warnings the first time it is requested (only in verbose mode).
pub fn log_config_usage() {
    PRINT_CONFIG_ONCE.call_once(|| {
        if !is_verbose() {
            return;
        }
        let handle = pipeline_config_handle();
        if let Some(source) = &handle.source {
            eprintln!("[invers] Loaded pipeline config from {}", source.display());
        } else {
            eprintln!("[invers] Using built-in pipeline defaults");
        }

        for warning in &handle.warnings {
            eprintln!("[invers] Config warning: {}", warning);
        }
    });
}
