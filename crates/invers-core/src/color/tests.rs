//! Tests for color conversion functions

use super::*;

#[test]
fn test_rgb_hsl_roundtrip() {
    let test_cases = [
        (1.0, 0.0, 0.0), // Red
        (0.0, 1.0, 0.0), // Green
        (0.0, 0.0, 1.0), // Blue
        (1.0, 1.0, 1.0), // White
        (0.0, 0.0, 0.0), // Black
        (0.5, 0.5, 0.5), // Gray
        (1.0, 0.5, 0.0), // Orange
        (0.5, 0.0, 0.5), // Purple
    ];

    for (r, g, b) in test_cases {
        let hsl = rgb_to_hsl(r, g, b);
        let (r2, g2, b2) = hsl_to_rgb(hsl);

        assert!(
            (r - r2).abs() < 1e-5,
            "R mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            r,
            r2
        );
        assert!(
            (g - g2).abs() < 1e-5,
            "G mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            g,
            g2
        );
        assert!(
            (b - b2).abs() < 1e-5,
            "B mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            b,
            b2
        );
    }
}

#[test]
fn test_hsl_values() {
    // Red should be H=0, S=1, L=0.5
    let hsl = rgb_to_hsl(1.0, 0.0, 0.0);
    assert!((hsl.h - 0.0).abs() < 1e-5);
    assert!((hsl.s - 1.0).abs() < 1e-5);
    assert!((hsl.l - 0.5).abs() < 1e-5);

    // Green should be H=120, S=1, L=0.5
    let hsl = rgb_to_hsl(0.0, 1.0, 0.0);
    assert!((hsl.h - 120.0).abs() < 1e-5);
    assert!((hsl.s - 1.0).abs() < 1e-5);

    // Blue should be H=240, S=1, L=0.5
    let hsl = rgb_to_hsl(0.0, 0.0, 1.0);
    assert!((hsl.h - 240.0).abs() < 1e-5);
    assert!((hsl.s - 1.0).abs() < 1e-5);
}

#[test]
fn test_rgb_lab_roundtrip() {
    let test_cases = [
        (1.0, 0.0, 0.0), // Red
        (0.0, 1.0, 0.0), // Green
        (0.0, 0.0, 1.0), // Blue
        (1.0, 1.0, 1.0), // White
        (0.5, 0.5, 0.5), // Gray
        (0.8, 0.4, 0.2), // Orange-ish
    ];

    for (r, g, b) in test_cases {
        let lab = rgb_to_lab(r, g, b);
        let (r2, g2, b2) = lab_to_rgb(lab);

        // LAB roundtrip may have slightly more error due to matrix operations
        assert!(
            (r - r2).abs() < 1e-4,
            "R mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            r,
            r2
        );
        assert!(
            (g - g2).abs() < 1e-4,
            "G mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            g,
            g2
        );
        assert!(
            (b - b2).abs() < 1e-4,
            "B mismatch for ({}, {}, {}): {} vs {}",
            r,
            g,
            b,
            b,
            b2
        );
    }
}

#[test]
fn test_lab_values() {
    // White should be L=100, a=0, b=0
    let lab = rgb_to_lab(1.0, 1.0, 1.0);
    assert!((lab.l - 100.0).abs() < 0.1);
    assert!(lab.a.abs() < 0.1);
    assert!(lab.b.abs() < 0.1);

    // Black should be L=0, a=0, b=0
    let lab = rgb_to_lab(0.0, 0.0, 0.0);
    assert!(lab.l.abs() < 0.1);
    assert!(lab.a.abs() < 0.1);
    assert!(lab.b.abs() < 0.1);

    // Gray should have a=0, b=0
    let lab = rgb_to_lab(0.5, 0.5, 0.5);
    assert!(lab.a.abs() < 0.1);
    assert!(lab.b.abs() < 0.1);
}
