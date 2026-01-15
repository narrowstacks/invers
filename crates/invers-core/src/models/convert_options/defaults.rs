//! Default value functions for serde.

/// Default true value for serde
pub fn default_true() -> bool {
    true
}

/// Default false value for serde
pub fn default_false() -> bool {
    false
}

/// Default clip percent for auto-levels (0.25%)
pub fn default_clip_percent() -> f32 {
    0.25
}

/// Default auto-color strength (0.6)
pub fn default_auto_color_strength() -> f32 {
    0.6
}

/// Default white balance strength (1.0)
pub fn default_wb_strength() -> f32 {
    1.0
}

/// Default minimum auto-color gain (0.7)
pub fn default_auto_color_min_gain() -> f32 {
    0.7
}

/// Default maximum auto-color gain (1.3)
pub fn default_auto_color_max_gain() -> f32 {
    1.3
}

/// Default maximum auto-color divergence (0.15 = 15%)
pub fn default_auto_color_max_divergence() -> f32 {
    0.15
}

/// Default base brightest percent for base estimation (5.0%)
pub fn default_base_brightest_percent() -> f32 {
    5.0
}

/// Default shadow lift value (0.02)
pub fn default_shadow_lift_value() -> f32 {
    0.02
}

/// Default value of 1.0 for multipliers
pub fn default_one() -> f32 {
    1.0
}

/// Default auto-exposure target median (0.25)
pub fn default_auto_exposure_target() -> f32 {
    0.25
}

/// Default auto-exposure strength (1.0)
pub fn default_auto_exposure_strength() -> f32 {
    1.0
}

/// Default minimum auto-exposure gain (0.6)
pub fn default_auto_exposure_min_gain() -> f32 {
    0.6
}

/// Default maximum auto-exposure gain (1.4)
pub fn default_auto_exposure_max_gain() -> f32 {
    1.4
}
