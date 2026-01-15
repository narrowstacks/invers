//! Trait implementations for ConvertOptions.

use std::path::PathBuf;

use super::defaults::{
    default_auto_color_max_divergence, default_auto_color_max_gain, default_auto_color_min_gain,
    default_auto_color_strength, default_auto_exposure_max_gain, default_auto_exposure_min_gain,
    default_auto_exposure_strength, default_auto_exposure_target, default_base_brightest_percent,
    default_clip_percent, default_false, default_one, default_shadow_lift_value, default_true,
    default_wb_strength,
};
use super::enums::{
    AutoLevelsMode, AutoWbMode, BitDepthPolicy, InversionMode, OutputFormat, PipelineMode,
    ShadowLiftMode,
};
use super::ConvertOptions;
use crate::models::base_estimation::{BaseEstimationMethod, BaseSamplingMode};

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            input_paths: Vec::new(),
            output_dir: PathBuf::from("."),
            output_format: OutputFormat::default(),
            working_colorspace: "linear-rec2020".to_string(),
            bit_depth_policy: BitDepthPolicy::default(),
            film_preset: None,
            scan_profile: None,
            base_estimation: None,
            num_threads: None,
            skip_tone_curve: false,
            skip_color_matrix: false,
            exposure_compensation: 1.0,
            debug: false,
            enable_auto_levels: default_true(),
            auto_levels_clip_percent: default_clip_percent(),
            preserve_headroom: default_false(),
            enable_auto_color: default_false(),
            auto_color_strength: default_auto_color_strength(),
            auto_color_min_gain: default_auto_color_min_gain(),
            auto_color_max_gain: default_auto_color_max_gain(),
            auto_color_max_divergence: default_auto_color_max_divergence(),
            base_brightest_percent: default_base_brightest_percent(),
            base_sampling_mode: BaseSamplingMode::default(),
            base_estimation_method: BaseEstimationMethod::default(),
            auto_levels_mode: AutoLevelsMode::default(),
            inversion_mode: InversionMode::default(),
            shadow_lift_mode: ShadowLiftMode::default(),
            shadow_lift_value: default_shadow_lift_value(),
            highlight_compression: default_one(),
            enable_auto_exposure: default_true(),
            auto_exposure_target_median: default_auto_exposure_target(),
            auto_exposure_strength: default_auto_exposure_strength(),
            auto_exposure_min_gain: default_auto_exposure_min_gain(),
            auto_exposure_max_gain: default_auto_exposure_max_gain(),
            no_clip: default_false(),
            enable_auto_wb: default_false(),
            auto_wb_strength: default_wb_strength(),
            auto_wb_mode: AutoWbMode::default(),
            use_gpu: default_true(),
            pipeline_mode: PipelineMode::default(),
            density_balance: None,
            neutral_point: None,
            density_balance_red: None,
            density_balance_blue: None,
            tone_curve_override: None,
            cb_options: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();

        // Check key defaults
        assert!(options.enable_auto_levels);
        assert!(!options.enable_auto_color);
        assert!((options.auto_levels_clip_percent - 0.25).abs() < 0.001);
        assert!((options.exposure_compensation - 1.0).abs() < 0.001);
        assert!(!options.debug);
        assert!(!options.skip_tone_curve);
        assert!(!options.skip_color_matrix);
        assert_eq!(options.inversion_mode, InversionMode::MaskAware);
        assert_eq!(options.pipeline_mode, PipelineMode::Legacy);
    }

    #[test]
    fn test_convert_options_shadow_lift_defaults() {
        let options = ConvertOptions::default();

        assert!((options.shadow_lift_value - 0.02).abs() < 0.001);
        assert_eq!(options.shadow_lift_mode, ShadowLiftMode::Percentile);
    }

    #[test]
    fn test_convert_options_auto_exposure_defaults() {
        let options = ConvertOptions::default();

        assert!(options.enable_auto_exposure);
        assert!((options.auto_exposure_target_median - 0.25).abs() < 0.01);
        assert!((options.auto_exposure_strength - 1.0).abs() < 0.001);
        assert!((options.auto_exposure_min_gain - 0.6).abs() < 0.01);
        assert!((options.auto_exposure_max_gain - 1.4).abs() < 0.01);
    }
}
