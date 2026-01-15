//! Tests for inversion algorithms

use super::*;
use crate::models::{BaseEstimation, ConvertOptions, InversionMode, MaskProfile, ShadowLiftMode};

/// Helper to create default ConvertOptions for testing
fn test_options() -> ConvertOptions {
    ConvertOptions {
        debug: false,
        shadow_lift_mode: ShadowLiftMode::None,
        highlight_compression: 1.0,
        ..Default::default()
    }
}

/// Helper to create a BaseEstimation with given medians
fn test_base(medians: [f32; 3]) -> BaseEstimation {
    BaseEstimation {
        roi: None,
        medians,
        noise_stats: None,
        auto_estimated: true,
        mask_profile: None,
    }
}

// ========================================================================
// Linear Inversion Tests
// ========================================================================

#[test]
fn test_linear_inversion_basic() {
    // Linear inversion: positive = (base - negative) / base
    let mut data = vec![
        0.3, 0.4, 0.5, // First pixel - some negative values below base
        0.6, 0.7, 0.8, // Second pixel - closer to base (should be darker after inversion)
    ];
    let base = test_base([0.8, 0.9, 1.0]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // For linear: (base - negative) / base
    // First pixel R: (0.8 - 0.3) / 0.8 = 0.625
    assert!(
        (data[0] - 0.625).abs() < 0.001,
        "Expected R=0.625, got {}",
        data[0]
    );
    // First pixel G: (0.9 - 0.4) / 0.9 = 0.555...
    assert!(
        (data[1] - 0.5555).abs() < 0.01,
        "Expected G≈0.556, got {}",
        data[1]
    );
    // First pixel B: (1.0 - 0.5) / 1.0 = 0.5
    assert!(
        (data[2] - 0.5).abs() < 0.001,
        "Expected B=0.5, got {}",
        data[2]
    );
}

#[test]
fn test_linear_inversion_at_base() {
    // Pixels at base level should become 0 (black)
    let mut data = vec![0.8, 0.9, 1.0];
    let base = test_base([0.8, 0.9, 1.0]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // At base: (base - base) / base = 0
    assert!(
        data[0].abs() < 0.001,
        "Expected R=0 at base, got {}",
        data[0]
    );
    assert!(
        data[1].abs() < 0.001,
        "Expected G=0 at base, got {}",
        data[1]
    );
    assert!(
        data[2].abs() < 0.001,
        "Expected B=0 at base, got {}",
        data[2]
    );
}

#[test]
fn test_linear_inversion_at_zero() {
    // Dark negative pixels (0) should become bright (1.0)
    let mut data = vec![0.0, 0.0, 0.0];
    let base = test_base([0.8, 0.9, 1.0]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // At zero: (base - 0) / base = 1.0
    assert!(
        (data[0] - 1.0).abs() < 0.001,
        "Expected R=1 at zero, got {}",
        data[0]
    );
    assert!(
        (data[1] - 1.0).abs() < 0.001,
        "Expected G=1 at zero, got {}",
        data[1]
    );
    assert!(
        (data[2] - 1.0).abs() < 0.001,
        "Expected B=1 at zero, got {}",
        data[2]
    );
}

// ========================================================================
// Logarithmic Inversion Tests
// ========================================================================

#[test]
fn test_logarithmic_inversion_basic() {
    // Logarithmic: exp(ln(base) - ln(neg)), clamped
    let mut data = vec![0.4, 0.5, 0.6];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Logarithmic;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // Log inversion increases darker pixels more
    // R: exp(ln(0.8) - ln(0.4)) = exp(ln(2)) = 2.0, but clamped by LN_MAX
    assert!(data[0] > 1.0, "Logarithmic should amplify dark values");
    // G: exp(ln(0.8) - ln(0.5)) = 1.6
    assert!(
        (data[1] - 1.6).abs() < 0.1,
        "Expected G≈1.6, got {}",
        data[1]
    );
}

#[test]
fn test_logarithmic_inversion_handles_small_values() {
    // Small values should be clamped to avoid division by zero
    let mut data = vec![0.00001, 0.00001, 0.00001];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Logarithmic;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // Values should be finite and clamped by the exp(LN_MAX) limit (~10)
    for &val in &data {
        assert!(val.is_finite(), "Value should be finite, got {}", val);
        assert!(val <= 10.1, "Value should be clamped, got {}", val);
    }
}

// ========================================================================
// DivideBlend Inversion Tests
// ========================================================================

#[test]
fn test_divide_blend_inversion() {
    // DivideBlend: 1.0 - (pixel / base)^(1/2.2)
    let mut data = vec![0.4, 0.5, 0.6];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::DivideBlend;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // R: 1.0 - (0.4/0.8)^(1/2.2) = 1.0 - 0.5^0.4545 = 1.0 - 0.73 ≈ 0.27
    assert!(
        (data[0] - 0.27).abs() < 0.05,
        "Expected R≈0.27, got {}",
        data[0]
    );
}

#[test]
fn test_divide_blend_at_base() {
    // At base level: 1.0 - (base/base)^gamma = 1.0 - 1.0 = 0.0
    let mut data = vec![0.8, 0.8, 0.8];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::DivideBlend;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    for &val in &data {
        assert!(val.abs() < 0.001, "At base should be 0, got {}", val);
    }
}

// ========================================================================
// MaskAware Inversion Tests
// ========================================================================

#[test]
fn test_mask_aware_inversion_basic() {
    // MaskAware: 1.0 - (pixel/base), then shadow floor correction
    let mut data = vec![0.4, 0.5, 0.6];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::MaskAware;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // Basic inversion: 1.0 - (pixel/base)
    // R: 1.0 - 0.5 = 0.5 (no shadow correction for R)
    // Shadow corrections depend on mask_profile
}

#[test]
fn test_mask_aware_with_custom_mask_profile() {
    let mut data = vec![0.4, 0.5, 0.6];
    let mut base = test_base([0.8, 0.8, 0.8]);
    base.mask_profile = Some(MaskProfile {
        magenta_impurity: 0.1,
        cyan_impurity: 0.05,
        correction_strength: 1.0,
    });
    let mut options = test_options();
    options.inversion_mode = InversionMode::MaskAware;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // Results should be valid (shadow correction applied)
    for &val in &data {
        assert!(val.is_finite(), "Value should be finite");
    }
}

// ========================================================================
// BlackAndWhite Inversion Tests
// ========================================================================

#[test]
fn test_bw_inversion_basic() {
    // B&W uses average base and applies headroom
    let mut data = vec![0.3, 0.3, 0.3, 0.6, 0.6, 0.6];
    let base = test_base([0.9, 0.9, 0.9]); // Average = 0.9
    let mut options = test_options();
    options.inversion_mode = InversionMode::BlackAndWhite;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // B&W inversion: ((base - pixel) * scale).clamp(0, 1)
    // With 5% headroom: black_point = 0.9 * 0.95 = 0.855
    // scale = 1.0 / 0.855 ≈ 1.17
    // First pixel: (0.9 - 0.3) * 1.17 ≈ 0.7
    assert!(
        data[0] > 0.5 && data[0] < 1.0,
        "Dark negative should be bright, got {}",
        data[0]
    );
    // Second pixel closer to base should be darker
    assert!(
        data[3] < data[0],
        "Closer to base should be darker after inversion"
    );
}

#[test]
fn test_bw_inversion_clamps_to_valid_range() {
    let mut data = vec![0.0, 0.0, 0.0, 0.95, 0.95, 0.95];
    let base = test_base([0.9, 0.9, 0.9]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::BlackAndWhite;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // All values should be in [0, 1]
    for &val in &data {
        assert!(
            (0.0..=1.0).contains(&val),
            "Value should be in [0,1], got {}",
            val
        );
    }
}

// ========================================================================
// Shadow Lift Tests
// ========================================================================

#[test]
fn test_shadow_lift_fixed() {
    let mut data = vec![-0.1, 0.0, 0.1];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;
    options.shadow_lift_mode = ShadowLiftMode::Fixed;
    options.shadow_lift_value = 0.05;

    let _ = invert_negative(&mut data, &base, 3, &options);

    // Fixed lift should add shadow_lift_value + offset to bring min >= 0
    let min_val = data.iter().cloned().fold(f32::MAX, f32::min);
    assert!(
        min_val >= 0.0,
        "Min should be lifted to >= 0, got {}",
        min_val
    );
}

#[test]
fn test_shadow_lift_none() {
    let mut data = vec![0.3, 0.4, 0.5];
    let base = test_base([0.8, 0.8, 0.8]);
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;
    options.shadow_lift_mode = ShadowLiftMode::None;

    let _ = invert_negative(&mut data, &base, 3, &options);

    // With ShadowLiftMode::None, negative values are clamped to 0
    for &val in &data {
        assert!(val >= 0.0, "All values should be >= 0 with None mode");
    }
}

// ========================================================================
// Reciprocal Inversion Tests
// ========================================================================

#[test]
fn test_reciprocal_inversion() {
    let mut data = vec![0.3, 0.5, 0.7, 0.8, 0.9, 1.0];
    let options = test_options();

    let result = apply_reciprocal_inversion(&mut data, &options);
    assert!(result.is_ok());

    // Reciprocal: 1.0 - normalized
    assert!(
        (data[0] - 0.7).abs() < 0.001,
        "1.0 - 0.3 = 0.7, got {}",
        data[0]
    );
    assert!(
        (data[1] - 0.5).abs() < 0.001,
        "1.0 - 0.5 = 0.5, got {}",
        data[1]
    );
    assert!(
        (data[2] - 0.3).abs() < 0.001,
        "1.0 - 0.7 = 0.3, got {}",
        data[2]
    );
    // Film base (1.0) should become black (0.0)
    assert!(
        (data[5] - 0.0).abs() < 0.001,
        "1.0 - 1.0 = 0.0, got {}",
        data[5]
    );
}

// ========================================================================
// Error Handling Tests
// ========================================================================

#[test]
fn test_inversion_wrong_channel_count() {
    let mut data = vec![0.3, 0.4, 0.5];
    let base = test_base([0.8, 0.8, 0.8]);
    let options = test_options();

    // Pass wrong channel count
    let result = invert_negative(&mut data, &base, 4, &options);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Expected 3 channels"));
}

#[test]
fn test_inversion_data_not_divisible_by_3() {
    let mut data = vec![0.3, 0.4]; // Not divisible by 3
    let base = test_base([0.8, 0.8, 0.8]);
    let options = test_options();

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not divisible by 3"));
}

#[test]
fn test_inversion_empty_data() {
    let mut data: Vec<f32> = vec![];
    let base = test_base([0.8, 0.8, 0.8]);
    let options = test_options();

    // Empty data should still work (nothing to process)
    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());
}

// ========================================================================
// Numerical Stability Tests
// ========================================================================

#[test]
fn test_inversion_base_near_zero_clamped() {
    // Very small base values should be clamped to avoid division issues
    let mut data = vec![0.3, 0.4, 0.5];
    let base = test_base([0.00001, 0.00001, 0.00001]); // Very small base
    let mut options = test_options();
    options.inversion_mode = InversionMode::Linear;

    let result = invert_negative(&mut data, &base, 3, &options);
    assert!(result.is_ok());

    // Values should be finite (not NaN or Inf)
    for &val in &data {
        assert!(val.is_finite(), "Value should be finite, got {}", val);
    }
}

#[test]
fn test_all_inversion_modes_produce_finite_values() {
    let test_data = vec![0.2, 0.4, 0.6, 0.5, 0.7, 0.9];
    let base = test_base([0.8, 0.9, 1.0]);

    let modes = [
        InversionMode::Linear,
        InversionMode::Logarithmic,
        InversionMode::DivideBlend,
        InversionMode::MaskAware,
        InversionMode::BlackAndWhite,
    ];

    for mode in &modes {
        let mut data = test_data.clone();
        let mut options = test_options();
        options.inversion_mode = *mode;

        let result = invert_negative(&mut data, &base, 3, &options);
        assert!(result.is_ok(), "Mode {:?} should succeed", mode);

        for (i, &val) in data.iter().enumerate() {
            assert!(
                val.is_finite(),
                "Mode {:?} produced non-finite value at index {}: {}",
                mode,
                i,
                val
            );
        }
    }
}
