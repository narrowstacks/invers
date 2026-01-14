//! Default pipeline parameter values and their validation/sanitization.

use crate::models::{BaseSamplingMode, InversionMode, ShadowLiftMode};
use serde::Deserialize;

/// Simplified configuration for release builds.
/// Only contains the essential options users typically need to adjust.
#[cfg(not(debug_assertions))]
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct SimplePipelineConfig {
    pub defaults: SimplePipelineDefaults,
}

/// Essential pipeline defaults for release builds.
/// Advanced options can be overridden via config, while testing remains debug-only.
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
    /// Optional overrides for advanced tuning in release builds
    pub auto_levels_clip_percent: Option<f32>,
    pub auto_color_strength: Option<f32>,
    pub auto_color_min_gain: Option<f32>,
    pub auto_color_max_gain: Option<f32>,
    pub auto_color_max_divergence: Option<f32>,
    pub base_brightest_percent: Option<f32>,
    pub base_sampling_mode: Option<BaseSamplingMode>,
    pub shadow_lift_mode: Option<ShadowLiftMode>,
    pub shadow_lift_value: Option<f32>,
    pub highlight_compression: Option<f32>,
    pub auto_exposure_target_median: Option<f32>,
    pub auto_exposure_strength: Option<f32>,
    pub auto_exposure_min_gain: Option<f32>,
    pub auto_exposure_max_gain: Option<f32>,
    pub skip_color_matrix: Option<bool>,
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
            skip_tone_curve: true,
            enable_auto_exposure: true,
            auto_levels_clip_percent: None,
            auto_color_strength: None,
            auto_color_min_gain: None,
            auto_color_max_gain: None,
            auto_color_max_divergence: None,
            base_brightest_percent: None,
            base_sampling_mode: None,
            shadow_lift_mode: None,
            shadow_lift_value: None,
            highlight_compression: None,
            auto_exposure_target_median: None,
            auto_exposure_strength: None,
            auto_exposure_min_gain: None,
            auto_exposure_max_gain: None,
            skip_color_matrix: None,
        }
    }
}

#[cfg(not(debug_assertions))]
impl SimplePipelineDefaults {
    /// Convert to full PipelineDefaults, filling in non-configurable options with hardcoded values
    pub fn to_full_defaults(&self) -> PipelineDefaults {
        let mut defaults = PipelineDefaults {
            inversion_mode: self.inversion_mode,
            enable_auto_levels: self.enable_auto_levels,
            preserve_headroom: self.preserve_headroom,
            exposure_compensation: self.exposure_compensation,
            enable_auto_color: self.enable_auto_color,
            skip_tone_curve: self.skip_tone_curve,
            // Release defaults for advanced options (config overrides may replace these)
            // Data-preserving defaults for maximum editing latitude
            auto_levels_clip_percent: 0.0,   // No highlight clipping
            auto_color_strength: 1.0,        // Full strength for blue cast correction
            auto_color_min_gain: 0.6,        // Allow stronger reduction for blue cast
            auto_color_max_gain: 1.4,        // Allow stronger boost for cyan cast
            auto_color_max_divergence: 0.25, // 25% max divergence for better cast removal
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
        };

        // Apply any config overrides so pipeline_defaults.yml always wins.
        if let Some(value) = self.auto_levels_clip_percent {
            defaults.auto_levels_clip_percent = value;
        }
        if let Some(value) = self.auto_color_strength {
            defaults.auto_color_strength = value;
        }
        if let Some(value) = self.auto_color_min_gain {
            defaults.auto_color_min_gain = value;
        }
        if let Some(value) = self.auto_color_max_gain {
            defaults.auto_color_max_gain = value;
        }
        if let Some(value) = self.auto_color_max_divergence {
            defaults.auto_color_max_divergence = value;
        }
        if let Some(value) = self.base_brightest_percent {
            defaults.base_brightest_percent = value;
        }
        if let Some(value) = self.base_sampling_mode {
            defaults.base_sampling_mode = value;
        }
        if let Some(value) = self.shadow_lift_mode {
            defaults.shadow_lift_mode = value;
        }
        if let Some(value) = self.shadow_lift_value {
            defaults.shadow_lift_value = value;
        }
        if let Some(value) = self.highlight_compression {
            defaults.highlight_compression = value;
        }
        if let Some(value) = self.auto_exposure_target_median {
            defaults.auto_exposure_target_median = value;
        }
        if let Some(value) = self.auto_exposure_strength {
            defaults.auto_exposure_strength = value;
        }
        if let Some(value) = self.auto_exposure_min_gain {
            defaults.auto_exposure_min_gain = value;
        }
        if let Some(value) = self.auto_exposure_max_gain {
            defaults.auto_exposure_max_gain = value;
        }
        if let Some(value) = self.skip_color_matrix {
            defaults.skip_color_matrix = value;
        }

        defaults
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
    pub auto_color_max_divergence: f32,
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
        self.auto_color_max_divergence = self.auto_color_max_divergence.clamp(0.0, 1.0);
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
            auto_color_max_divergence: 0.15,
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
