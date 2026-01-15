//! Tests for scan profile types.

use super::*;

// ========================================================================
// HslAdjustments Tests
// ========================================================================

#[test]
fn test_hsl_adjustments_default() {
    let adj = HslAdjustments::default();

    // All values should be 0.0
    for &h in &adj.hue {
        assert!((h - 0.0).abs() < 0.001);
    }
    for &s in &adj.saturation {
        assert!((s - 0.0).abs() < 0.001);
    }
    for &l in &adj.luminance {
        assert!((l - 0.0).abs() < 0.001);
    }
}

#[test]
fn test_hsl_adjustments_has_adjustments_false() {
    let adj = HslAdjustments::default();
    assert!(!adj.has_adjustments());
}

#[test]
fn test_hsl_adjustments_has_adjustments_hue() {
    let mut adj = HslAdjustments::default();
    adj.hue[HslAdjustments::REDS] = 10.0;

    assert!(adj.has_adjustments());
}

#[test]
fn test_hsl_adjustments_has_adjustments_saturation() {
    let mut adj = HslAdjustments::default();
    adj.saturation[HslAdjustments::BLUES] = -20.0;

    assert!(adj.has_adjustments());
}

#[test]
fn test_hsl_adjustments_has_adjustments_luminance() {
    let mut adj = HslAdjustments::default();
    adj.luminance[HslAdjustments::GREENS] = 15.0;

    assert!(adj.has_adjustments());
}

#[test]
fn test_hsl_adjustments_color_indices() {
    // Verify color indices are correct
    assert_eq!(HslAdjustments::REDS, 0);
    assert_eq!(HslAdjustments::ORANGES, 1);
    assert_eq!(HslAdjustments::YELLOWS, 2);
    assert_eq!(HslAdjustments::GREENS, 3);
    assert_eq!(HslAdjustments::AQUAS, 4);
    assert_eq!(HslAdjustments::BLUES, 5);
    assert_eq!(HslAdjustments::PURPLES, 6);
    assert_eq!(HslAdjustments::MAGENTAS, 7);
}

// ========================================================================
// MaskProfile Tests
// ========================================================================

#[test]
fn test_mask_profile_default() {
    let profile = MaskProfile::default();

    assert!((profile.magenta_impurity - 0.50).abs() < 0.01);
    assert!((profile.cyan_impurity - 0.30).abs() < 0.01);
    assert!((profile.correction_strength - 1.0).abs() < 0.01);
}

#[test]
fn test_mask_profile_from_base_medians_typical_orange() {
    // Typical orange mask: R=0.8, G=0.6, B=0.35
    let profile = MaskProfile::from_base_medians(&[0.8, 0.6, 0.35]);

    // Should have standard impurity values
    assert!((profile.magenta_impurity - 0.50).abs() < 0.01);
    assert!((profile.cyan_impurity - 0.30).abs() < 0.01);

    // Should have some correction strength (not zero)
    assert!(
        profile.correction_strength > 0.0,
        "Orange mask should have correction strength: {}",
        profile.correction_strength
    );
    assert!(
        profile.correction_strength <= 0.7,
        "Correction strength should be clamped: {}",
        profile.correction_strength
    );
}

#[test]
fn test_mask_profile_from_base_medians_neutral() {
    // Neutral gray base: R=G=B
    let profile = MaskProfile::from_base_medians(&[0.5, 0.5, 0.5]);

    // Correction strength should be low/zero for neutral base
    assert!(
        profile.correction_strength < 0.1,
        "Neutral base should have low correction: {}",
        profile.correction_strength
    );
}

#[test]
fn test_mask_profile_from_base_medians_handles_near_zero() {
    // Very small values should not cause issues
    let profile = MaskProfile::from_base_medians(&[0.0001, 0.0001, 0.0001]);

    // Should be finite
    assert!(profile.correction_strength.is_finite());
    assert!(profile.magenta_impurity.is_finite());
    assert!(profile.cyan_impurity.is_finite());
}

#[test]
fn test_mask_profile_calculate_shadow_floors_default() {
    let profile = MaskProfile::default();
    let (red_floor, green_floor, blue_floor) = profile.calculate_shadow_floors();

    // Red floor is always 0
    assert!((red_floor - 0.0).abs() < 0.001);

    // Green floor: 1.0 * (0.30 / 1.30) ≈ 0.231
    assert!(
        (green_floor - 0.231).abs() < 0.01,
        "Green floor expected ~0.231, got {}",
        green_floor
    );

    // Blue floor: 1.0 * (0.50 / 1.50) ≈ 0.333
    assert!(
        (blue_floor - 0.333).abs() < 0.01,
        "Blue floor expected ~0.333, got {}",
        blue_floor
    );
}

#[test]
fn test_mask_profile_calculate_shadow_floors_zero_strength() {
    let profile = MaskProfile {
        magenta_impurity: 0.5,
        cyan_impurity: 0.3,
        correction_strength: 0.0,
    };
    let (red_floor, green_floor, blue_floor) = profile.calculate_shadow_floors();

    // All floors should be 0 with zero correction strength
    assert!((red_floor - 0.0).abs() < 0.001);
    assert!((green_floor - 0.0).abs() < 0.001);
    assert!((blue_floor - 0.0).abs() < 0.001);
}

#[test]
fn test_mask_profile_calculate_shadow_floors_partial_strength() {
    let profile = MaskProfile {
        magenta_impurity: 0.5,
        cyan_impurity: 0.3,
        correction_strength: 0.5, // 50% strength
    };
    let (red_floor, green_floor, blue_floor) = profile.calculate_shadow_floors();

    // Red floor is always 0
    assert!((red_floor - 0.0).abs() < 0.001);

    // Green floor: 0.5 * (0.30 / 1.30) ≈ 0.115
    assert!(
        (green_floor - 0.115).abs() < 0.02,
        "Green floor expected ~0.115, got {}",
        green_floor
    );

    // Blue floor: 0.5 * (0.50 / 1.50) ≈ 0.167
    assert!(
        (blue_floor - 0.167).abs() < 0.02,
        "Blue floor expected ~0.167, got {}",
        blue_floor
    );
}

// ========================================================================
// DemosaicHints Tests
// ========================================================================

#[test]
fn test_demosaic_hints_default() {
    let hints = DemosaicHints::default();

    assert_eq!(hints.algorithm, "ahd");
    assert!((hints.quality - 0.5).abs() < 0.01);
}

// ========================================================================
// WhiteBalanceHints Tests
// ========================================================================

#[test]
fn test_white_balance_hints_default() {
    let hints = WhiteBalanceHints::default();

    assert!(hints.auto);
    assert!(hints.temperature.is_none());
    assert!(hints.tint.is_none());
}
