//! Tests for CB pipeline math utilities and core functions

use super::*;

#[test]
fn test_sigmoid_contrast() {
    // Test that sigmoid at midpoint returns midpoint
    let result = apply_sigmoid(2.0, 0.5, 0.5, 1.0);
    assert!((result - 0.5).abs() < 0.01);

    // Test that sigmoid increases contrast
    let low = apply_sigmoid(2.0, 0.5, 0.3, 1.0);
    let high = apply_sigmoid(2.0, 0.5, 0.7, 1.0);
    assert!(low < 0.3);
    assert!(high > 0.7);
}

#[test]
fn test_inverse_sigmoid() {
    // Test that inverse sigmoid reverses sigmoid
    let original = 0.3;
    let sigmoid_result = apply_sigmoid(2.0, 0.5, original, 1.0);
    let recovered = apply_inverse_sigmoid(2.0, 0.5, sigmoid_result, 1.0);
    assert!((recovered - original).abs() < 0.01);
}

#[test]
fn test_clamp01() {
    assert_eq!(clamp01(-0.5), 0.0);
    assert_eq!(clamp01(0.5), 0.5);
    assert_eq!(clamp01(1.5), 1.0);
}

#[test]
fn test_logb() {
    // log_2(8) = 3
    assert!((logb(2.0, 8.0) - 3.0).abs() < 0.001);
    // log_10(100) = 2
    assert!((logb(10.0, 100.0) - 2.0).abs() < 0.001);
    // Edge cases
    assert_eq!(logb(0.0, 10.0), 1.0);
    assert_eq!(logb(1.0, 10.0), 1.0);
    assert_eq!(logb(10.0, 0.0), 1.0);
}

#[test]
fn test_compute_stats() {
    let data = vec![0.0, 0.5, 1.0, 0.25, 0.75];
    let (min, max, mean) = process::compute_stats(&data);
    assert_eq!(min, 0.0);
    assert_eq!(max, 1.0);
    assert!((mean - 0.5).abs() < 0.001);
}

#[test]
fn test_compute_stats_empty() {
    let data: Vec<f32> = vec![];
    let (min, max, mean) = process::compute_stats(&data);
    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);
    assert_eq!(mean, 0.0);
}
