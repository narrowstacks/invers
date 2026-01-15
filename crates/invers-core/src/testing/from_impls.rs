//! From trait implementations for config types
//!
//! Converts configuration types to testing types.

use crate::config;
use crate::testing::types::{ParameterGrid, ParameterTest};

impl From<config::ParameterTestDefaults> for ParameterTest {
    fn from(defaults: config::ParameterTestDefaults) -> Self {
        Self {
            enable_auto_levels: defaults.enable_auto_levels,
            clip_percent: defaults.clip_percent,
            enable_auto_color: defaults.enable_auto_color,
            auto_color_strength: defaults.auto_color_strength,
            auto_color_min_gain: defaults.auto_color_min_gain,
            auto_color_max_gain: defaults.auto_color_max_gain,
            auto_color_max_divergence: defaults.auto_color_max_divergence,
            base_brightest_percent: defaults.base_brightest_percent,
            base_sampling_mode: defaults.base_sampling_mode,
            inversion_mode: defaults.inversion_mode,
            shadow_lift_mode: defaults.shadow_lift_mode,
            shadow_lift_value: defaults.shadow_lift_value,
            highlight_compression: defaults.highlight_compression,
            enable_auto_exposure: defaults.enable_auto_exposure,
            auto_exposure_target_median: defaults.auto_exposure_target_median,
            auto_exposure_strength: defaults.auto_exposure_strength,
            auto_exposure_min_gain: defaults.auto_exposure_min_gain,
            auto_exposure_max_gain: defaults.auto_exposure_max_gain,
            tone_curve_strength: defaults.tone_curve_strength,
            skip_tone_curve: defaults.skip_tone_curve,
            exposure_compensation: defaults.exposure_compensation,
        }
    }
}

impl From<config::TestingGridValues> for ParameterGrid {
    fn from(values: config::TestingGridValues) -> Self {
        Self {
            auto_levels: values.auto_levels,
            clip_percent: values.clip_percent,
            auto_color: values.auto_color,
            auto_color_strength: values.auto_color_strength,
            auto_color_min_gain: values.auto_color_min_gain,
            auto_color_max_gain: values.auto_color_max_gain,
            auto_color_max_divergence: values.auto_color_max_divergence,
            base_brightest_percent: values.base_brightest_percent,
            base_sampling_mode: values.base_sampling_mode,
            inversion_mode: values.inversion_mode,
            shadow_lift_mode: values.shadow_lift_mode,
            shadow_lift_value: values.shadow_lift_value,
            tone_curve_strength: values.tone_curve_strength,
            exposure_compensation: values.exposure_compensation,
        }
    }
}
