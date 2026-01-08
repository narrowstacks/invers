use crate::models::{BaseSamplingMode, InversionMode, ShadowLiftMode};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Once, OnceLock};

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

/// Simplified configuration for release builds.
/// Only contains the essential options users typically need to adjust.
#[cfg(not(debug_assertions))]
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct SimplePipelineConfig {
    pub defaults: SimplePipelineDefaults,
}

/// Essential pipeline defaults for release builds.
/// For advanced tuning, use a debug build.
#[cfg(not(debug_assertions))]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SimplePipelineDefaults {
    /// Method for inverting negative to positive (MaskAware, Linear, etc.)
    pub inversion_mode: InversionMode,
    /// Enable automatic level adjustment
    pub enable_auto_levels: bool,
    /// Preserve shadow/highlight headroom
    pub preserve_headroom: bool,
    /// Manual exposure compensation (1.0 = no change)
    pub exposure_compensation: f32,
    /// Enable automatic color correction
    pub enable_auto_color: bool,
    /// Skip tone curve application
    pub skip_tone_curve: bool,
    /// Enable automatic exposure adjustment to normalize brightness
    pub enable_auto_exposure: bool,
}

#[cfg(not(debug_assertions))]
impl Default for SimplePipelineDefaults {
    fn default() -> Self {
        Self {
            inversion_mode: InversionMode::MaskAware,
            enable_auto_levels: true,
            preserve_headroom: true,
            exposure_compensation: 1.0,
            enable_auto_color: true,
            skip_tone_curve: false,
            enable_auto_exposure: true,
        }
    }
}

#[cfg(not(debug_assertions))]
impl SimplePipelineDefaults {
    /// Convert to full PipelineDefaults, filling in non-configurable options with hardcoded values
    pub fn to_full_defaults(&self) -> PipelineDefaults {
        PipelineDefaults {
            inversion_mode: self.inversion_mode,
            enable_auto_levels: self.enable_auto_levels,
            preserve_headroom: self.preserve_headroom,
            exposure_compensation: self.exposure_compensation,
            enable_auto_color: self.enable_auto_color,
            skip_tone_curve: self.skip_tone_curve,
            // Hardcoded values for release builds (advanced tuning requires debug build)
            // Data-preserving defaults for maximum editing latitude
            auto_levels_clip_percent: 0.0, // No highlight clipping
            auto_color_strength: 1.0,
            auto_color_min_gain: 0.7,
            auto_color_max_gain: 1.3,
            base_brightest_percent: 5.0,
            base_sampling_mode: BaseSamplingMode::Median,
            shadow_lift_mode: ShadowLiftMode::Percentile,
            shadow_lift_value: 0.02,
            highlight_compression: 1.0, // No highlight compression
            enable_auto_exposure: self.enable_auto_exposure,
            auto_exposure_target_median: 0.63, // Optimized for typical film negatives
            auto_exposure_strength: 1.0,
            auto_exposure_min_gain: 0.3,
            auto_exposure_max_gain: 5.0, // Allow significant boost for underexposed negatives
            skip_color_matrix: false,
        }
    }
}

/// Default pipeline parameter values.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PipelineDefaults {
    pub enable_auto_levels: bool,
    pub auto_levels_clip_percent: f32,
    pub preserve_headroom: bool,
    pub enable_auto_color: bool,
    pub auto_color_strength: f32,
    pub auto_color_min_gain: f32,
    pub auto_color_max_gain: f32,
    pub base_brightest_percent: f32,
    pub base_sampling_mode: BaseSamplingMode,
    pub inversion_mode: InversionMode,
    pub shadow_lift_mode: ShadowLiftMode,
    pub shadow_lift_value: f32,
    pub highlight_compression: f32,
    pub enable_auto_exposure: bool,
    pub auto_exposure_target_median: f32,
    pub auto_exposure_strength: f32,
    pub auto_exposure_min_gain: f32,
    pub auto_exposure_max_gain: f32,
    pub exposure_compensation: f32,
    pub skip_tone_curve: bool,
    pub skip_color_matrix: bool,
}

impl PipelineDefaults {
    #[allow(dead_code)] // Only used in debug builds
    pub(crate) fn sanitize(&mut self) {
        self.auto_levels_clip_percent = self.auto_levels_clip_percent.clamp(0.0, 10.0);
        self.auto_color_min_gain = self.auto_color_min_gain.max(0.1);
        self.auto_color_max_gain = self.auto_color_max_gain.max(self.auto_color_min_gain);
        self.base_brightest_percent = self.base_brightest_percent.clamp(1.0, 30.0);
        self.shadow_lift_value = self.shadow_lift_value.clamp(0.0, 0.1);
        self.highlight_compression = self.highlight_compression.clamp(0.0, 1.0);
        self.auto_exposure_target_median = self.auto_exposure_target_median.clamp(0.01, 0.9);
        self.auto_exposure_strength = self.auto_exposure_strength.clamp(0.0, 1.0);
        self.auto_exposure_min_gain = self.auto_exposure_min_gain.max(0.01);
        self.auto_exposure_max_gain = self
            .auto_exposure_max_gain
            .max(self.auto_exposure_min_gain + f32::EPSILON);
        self.exposure_compensation = self.exposure_compensation.max(0.01);
    }
}

impl Default for PipelineDefaults {
    fn default() -> Self {
        Self {
            enable_auto_levels: true,
            auto_levels_clip_percent: 0.25,
            preserve_headroom: true,
            enable_auto_color: true,
            auto_color_strength: 0.6,
            auto_color_min_gain: 0.7,
            auto_color_max_gain: 1.3,
            base_brightest_percent: 5.0,
            base_sampling_mode: BaseSamplingMode::Median,
            inversion_mode: InversionMode::MaskAware,
            shadow_lift_mode: ShadowLiftMode::Percentile,
            shadow_lift_value: 0.02,
            highlight_compression: 1.0,
            enable_auto_exposure: true,
            auto_exposure_target_median: 0.25,
            auto_exposure_strength: 1.0,
            auto_exposure_min_gain: 0.6,
            auto_exposure_max_gain: 1.4,
            exposure_compensation: 1.0,
            skip_tone_curve: true,
            skip_color_matrix: false,
        }
    }
}

/// Testing-related configuration overrides (debug builds only).
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TestingConfig {
    pub parameter_test_defaults: Option<ParameterTestDefaults>,
    pub default_grid: Option<TestingGridValues>,
    pub minimal_grid: Option<TestingGridValues>,
    pub comprehensive_grid: Option<TestingGridValues>,
}

/// Defaults for a single parameter test run (debug builds only).
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ParameterTestDefaults {
    pub enable_auto_levels: bool,
    pub clip_percent: f32,
    pub enable_auto_color: bool,
    pub auto_color_strength: f32,
    pub auto_color_min_gain: f32,
    pub auto_color_max_gain: f32,
    pub base_brightest_percent: f32,
    pub base_sampling_mode: BaseSamplingMode,
    pub inversion_mode: InversionMode,
    pub shadow_lift_mode: ShadowLiftMode,
    pub shadow_lift_value: f32,
    pub highlight_compression: f32,
    pub tone_curve_strength: f32,
    pub skip_tone_curve: bool,
    pub exposure_compensation: f32,
    pub enable_auto_exposure: bool,
    pub auto_exposure_target_median: f32,
    pub auto_exposure_strength: f32,
    pub auto_exposure_min_gain: f32,
    pub auto_exposure_max_gain: f32,
}

#[cfg(debug_assertions)]
impl ParameterTestDefaults {
    pub(crate) fn sanitize(&mut self) {
        self.clip_percent = self.clip_percent.clamp(0.0, 10.0);
        self.auto_color_min_gain = self.auto_color_min_gain.max(0.1);
        self.auto_color_max_gain = self.auto_color_max_gain.max(self.auto_color_min_gain);
        self.base_brightest_percent = self.base_brightest_percent.clamp(1.0, 30.0);
        self.shadow_lift_value = self.shadow_lift_value.clamp(0.0, 0.1);
        self.highlight_compression = self.highlight_compression.clamp(0.0, 1.0);
        self.tone_curve_strength = self.tone_curve_strength.clamp(0.0, 1.0);
        self.exposure_compensation = self.exposure_compensation.max(0.01);
        self.auto_exposure_target_median = self.auto_exposure_target_median.clamp(0.01, 0.9);
        self.auto_exposure_strength = self.auto_exposure_strength.clamp(0.0, 1.0);
        self.auto_exposure_min_gain = self.auto_exposure_min_gain.max(0.01);
        self.auto_exposure_max_gain = self
            .auto_exposure_max_gain
            .max(self.auto_exposure_min_gain + f32::EPSILON);
    }
}

#[cfg(debug_assertions)]
impl Default for ParameterTestDefaults {
    fn default() -> Self {
        Self {
            enable_auto_levels: true,
            clip_percent: 0.25,
            enable_auto_color: true,
            auto_color_strength: 0.6,
            auto_color_min_gain: 0.7,
            auto_color_max_gain: 1.3,
            base_brightest_percent: 5.0,
            base_sampling_mode: BaseSamplingMode::Median,
            inversion_mode: InversionMode::MaskAware,
            shadow_lift_mode: ShadowLiftMode::Percentile,
            shadow_lift_value: 0.02,
            highlight_compression: 1.0,
            tone_curve_strength: 0.5,
            skip_tone_curve: false,
            exposure_compensation: 1.0,
            enable_auto_exposure: true,
            auto_exposure_target_median: 0.25,
            auto_exposure_strength: 1.0,
            auto_exposure_min_gain: 0.6,
            auto_exposure_max_gain: 1.4,
        }
    }
}

/// Configurable grid of parameter values for batch testing (debug builds only).
#[cfg(debug_assertions)]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TestingGridValues {
    pub auto_levels: Vec<bool>,
    pub clip_percent: Vec<f32>,
    pub auto_color: Vec<bool>,
    pub auto_color_strength: Vec<f32>,
    pub auto_color_min_gain: Vec<f32>,
    pub auto_color_max_gain: Vec<f32>,
    pub base_brightest_percent: Vec<f32>,
    pub base_sampling_mode: Vec<BaseSamplingMode>,
    pub inversion_mode: Vec<InversionMode>,
    pub shadow_lift_mode: Vec<ShadowLiftMode>,
    pub shadow_lift_value: Vec<f32>,
    pub tone_curve_strength: Vec<f32>,
    pub exposure_compensation: Vec<f32>,
}

#[cfg(debug_assertions)]
impl TestingGridValues {
    pub(crate) fn sanitize_with(&mut self, defaults: TestingGridValues) {
        if self.auto_levels.is_empty() {
            self.auto_levels = defaults.auto_levels;
        }
        if self.clip_percent.is_empty() {
            self.clip_percent = defaults.clip_percent;
        }
        if self.auto_color.is_empty() {
            self.auto_color = defaults.auto_color;
        }
        if self.auto_color_strength.is_empty() {
            self.auto_color_strength = defaults.auto_color_strength;
        }
        if self.auto_color_min_gain.is_empty() {
            self.auto_color_min_gain = defaults.auto_color_min_gain;
        }
        if self.auto_color_max_gain.is_empty() {
            self.auto_color_max_gain = defaults.auto_color_max_gain;
        }
        if self.base_brightest_percent.is_empty() {
            self.base_brightest_percent = defaults.base_brightest_percent;
        }
        if self.base_sampling_mode.is_empty() {
            self.base_sampling_mode = defaults.base_sampling_mode;
        }
        if self.inversion_mode.is_empty() {
            self.inversion_mode = defaults.inversion_mode;
        }
        if self.shadow_lift_mode.is_empty() {
            self.shadow_lift_mode = defaults.shadow_lift_mode;
        }
        if self.shadow_lift_value.is_empty() {
            self.shadow_lift_value = defaults.shadow_lift_value;
        }
        if self.tone_curve_strength.is_empty() {
            self.tone_curve_strength = defaults.tone_curve_strength;
        }
        if self.exposure_compensation.is_empty() {
            self.exposure_compensation = defaults.exposure_compensation;
        }

        self.clip_percent
            .iter_mut()
            .for_each(|v| *v = v.clamp(0.0, 10.0));
        self.auto_color_min_gain
            .iter_mut()
            .for_each(|v| *v = v.max(0.1));
        self.auto_color_max_gain.iter_mut().for_each(|v| {
            *v = v.max(0.1);
        });
        let min_max = self
            .auto_color_min_gain
            .iter()
            .cloned()
            .fold(0.1_f32, |acc, value| acc.max(value));
        self.auto_color_max_gain
            .iter_mut()
            .for_each(|v| *v = v.max(min_max));
        self.base_brightest_percent
            .iter_mut()
            .for_each(|v| *v = v.clamp(1.0, 30.0));
        self.shadow_lift_value
            .iter_mut()
            .for_each(|v| *v = v.clamp(0.0, 0.1));
        self.tone_curve_strength
            .iter_mut()
            .for_each(|v| *v = v.clamp(0.0, 1.0));
        self.exposure_compensation
            .iter_mut()
            .for_each(|v| *v = v.max(0.01));
    }

    pub(crate) fn default_grid() -> Self {
        Self {
            auto_levels: vec![true],
            clip_percent: vec![0.25, 0.5, 1.0],
            auto_color: vec![true, false],
            auto_color_strength: vec![0.6, 0.8],
            auto_color_min_gain: vec![0.7, 0.8],
            auto_color_max_gain: vec![1.2, 1.3],
            base_brightest_percent: vec![5.0, 10.0, 15.0],
            base_sampling_mode: vec![BaseSamplingMode::Median],
            inversion_mode: vec![InversionMode::MaskAware],
            shadow_lift_mode: vec![ShadowLiftMode::Percentile],
            shadow_lift_value: vec![0.02],
            tone_curve_strength: vec![0.4, 0.5, 0.6, 0.7],
            exposure_compensation: vec![1.0],
        }
    }

    pub(crate) fn minimal_grid() -> Self {
        Self {
            auto_levels: vec![true],
            clip_percent: vec![0.25],
            auto_color: vec![true, false],
            auto_color_strength: vec![0.6, 0.8],
            auto_color_min_gain: vec![0.7],
            auto_color_max_gain: vec![1.3],
            base_brightest_percent: vec![5.0, 10.0],
            base_sampling_mode: vec![BaseSamplingMode::Median],
            inversion_mode: vec![InversionMode::MaskAware],
            shadow_lift_mode: vec![ShadowLiftMode::Percentile],
            shadow_lift_value: vec![0.02],
            tone_curve_strength: vec![0.4, 0.5, 0.6],
            exposure_compensation: vec![1.0],
        }
    }

    pub(crate) fn comprehensive_grid() -> Self {
        Self {
            auto_levels: vec![true],
            clip_percent: vec![0.2, 0.4, 0.6, 1.0, 2.0, 5.0],
            auto_color: vec![true, false],
            auto_color_strength: vec![0.5, 0.6, 0.8, 1.0],
            auto_color_min_gain: vec![0.65, 0.7, 0.75],
            auto_color_max_gain: vec![1.1, 1.2, 1.3, 1.4],
            base_brightest_percent: vec![5.0, 10.0, 15.0, 20.0],
            base_sampling_mode: vec![BaseSamplingMode::Median, BaseSamplingMode::Mean],
            inversion_mode: vec![InversionMode::MaskAware],
            shadow_lift_mode: vec![ShadowLiftMode::Percentile],
            shadow_lift_value: vec![0.015, 0.02, 0.03],
            tone_curve_strength: vec![0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            exposure_compensation: vec![0.9, 1.0, 1.05],
        }
    }
}

#[cfg(debug_assertions)]
impl Default for TestingGridValues {
    fn default() -> Self {
        Self::default_grid()
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
                            return PipelineConfigHandle::with_config(config, Some(source), warnings);
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
