//! Testing-related configuration types (debug builds only).

use crate::models::{BaseSamplingMode, InversionMode, ShadowLiftMode};
use serde::Deserialize;

/// Testing-related configuration overrides (debug builds only).
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TestingConfig {
    pub parameter_test_defaults: Option<ParameterTestDefaults>,
    pub default_grid: Option<TestingGridValues>,
    pub minimal_grid: Option<TestingGridValues>,
    pub comprehensive_grid: Option<TestingGridValues>,
}

/// Defaults for a single parameter test run (debug builds only).
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ParameterTestDefaults {
    pub enable_auto_levels: bool,
    pub clip_percent: f32,
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
    pub tone_curve_strength: f32,
    pub skip_tone_curve: bool,
    pub exposure_compensation: f32,
    pub enable_auto_exposure: bool,
    pub auto_exposure_target_median: f32,
    pub auto_exposure_strength: f32,
    pub auto_exposure_min_gain: f32,
    pub auto_exposure_max_gain: f32,
}

impl ParameterTestDefaults {
    pub(crate) fn sanitize(&mut self) {
        self.clip_percent = self.clip_percent.clamp(0.0, 10.0);
        self.auto_color_min_gain = self.auto_color_min_gain.max(0.1);
        self.auto_color_max_gain = self.auto_color_max_gain.max(self.auto_color_min_gain);
        self.auto_color_max_divergence = self.auto_color_max_divergence.clamp(0.0, 1.0);
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

impl Default for ParameterTestDefaults {
    fn default() -> Self {
        Self {
            enable_auto_levels: true,
            clip_percent: 0.25,
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
            tone_curve_strength: 0.5,
            skip_tone_curve: true,
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
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TestingGridValues {
    pub auto_levels: Vec<bool>,
    pub clip_percent: Vec<f32>,
    pub auto_color: Vec<bool>,
    pub auto_color_strength: Vec<f32>,
    pub auto_color_min_gain: Vec<f32>,
    pub auto_color_max_gain: Vec<f32>,
    pub auto_color_max_divergence: Vec<f32>,
    pub base_brightest_percent: Vec<f32>,
    pub base_sampling_mode: Vec<BaseSamplingMode>,
    pub inversion_mode: Vec<InversionMode>,
    pub shadow_lift_mode: Vec<ShadowLiftMode>,
    pub shadow_lift_value: Vec<f32>,
    pub tone_curve_strength: Vec<f32>,
    pub exposure_compensation: Vec<f32>,
}

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
        if self.auto_color_max_divergence.is_empty() {
            self.auto_color_max_divergence = defaults.auto_color_max_divergence;
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
        self.auto_color_max_divergence
            .iter_mut()
            .for_each(|v| *v = v.clamp(0.0, 1.0));
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
            auto_color_max_divergence: vec![0.15],
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
            auto_color_max_divergence: vec![0.15],
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
            auto_color_max_divergence: vec![0.1, 0.15, 0.2],
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

impl Default for TestingGridValues {
    fn default() -> Self {
        Self::default_grid()
    }
}
