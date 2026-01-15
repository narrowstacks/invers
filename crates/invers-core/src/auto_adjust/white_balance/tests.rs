//! Tests for white balance functions

use super::*;

#[test]
fn test_kelvin_to_rgb_multipliers() {
    // Test D65 (6500K) - should be close to neutral
    let d65 = kelvin_to_rgb_multipliers(6500.0);
    assert!((d65[0] - 1.0).abs() < 0.1, "D65 red should be near 1.0");
    assert_eq!(d65[1], 1.0, "Green is always 1.0");
    assert!((d65[2] - 1.0).abs() < 0.1, "D65 blue should be near 1.0");

    // Test warm temperature (3200K) - tungsten
    let warm = kelvin_to_rgb_multipliers(3200.0);
    assert!(warm[0] < 1.0, "Warm temp should reduce red");
    assert!(warm[2] > 1.0, "Warm temp should boost blue");

    // Test cool temperature (10000K)
    let cool = kelvin_to_rgb_multipliers(10000.0);
    assert!(cool[0] > 1.0, "Cool temp should boost red");
    assert!(cool[2] < 1.0, "Cool temp should reduce blue");
}

#[test]
fn test_auto_white_balance_neutral_image() {
    // Test with already neutral image
    let mut data = vec![0.5f32; 300]; // 100 neutral gray pixels
    let multipliers = auto_white_balance(&mut data, 3, 1.0);

    // Should return near-neutral multipliers
    assert!(
        (multipliers[0] - 1.0).abs() < 0.01,
        "Neutral image should have r_mult ~1.0"
    );
    assert!(
        (multipliers[1] - 1.0).abs() < 0.01,
        "Neutral image should have g_mult ~1.0"
    );
    assert!(
        (multipliers[2] - 1.0).abs() < 0.01,
        "Neutral image should have b_mult ~1.0"
    );
}

#[test]
fn test_auto_white_balance_strength() {
    // Test strength parameter
    let mut data1 = vec![0.6, 0.5, 0.4]; // Single pixel with color cast
    let mut data2 = data1.clone();

    let mult_full = auto_white_balance(&mut data1, 3, 1.0);
    let mult_half = auto_white_balance(&mut data2, 3, 0.5);

    // Half strength should produce multipliers closer to 1.0
    assert!(
        (mult_half[0] - 1.0).abs() < (mult_full[0] - 1.0).abs(),
        "Half strength should be less aggressive"
    );
}
