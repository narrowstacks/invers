//! Density balance types for the research pipeline.

use serde::{Deserialize, Serialize};

/// Source of density balance values
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DensityBalanceSource {
    /// Auto-calculated from neutral point sampling
    #[default]
    Auto,

    /// Manually specified values
    Manual,

    /// Default values (R=1.05, G=1.0, B=0.90)
    Default,
}

/// Density balance parameters for the research pipeline.
///
/// Per-channel power functions applied BEFORE inversion to align
/// the characteristic curves of each RGB emulsion layer.
///
/// Each film layer has a slightly different gamma (e.g., R=0.63, G=0.71, B=0.73).
/// Without density balance, this causes "color crossover" where shadows shift
/// toward one color cast while highlights shift toward another.
///
/// The density balance exponents correct this:
/// - R_balanced = R^db_r (typically 1.0-1.1)
/// - G_balanced = G^db_g (always 1.0, reference channel)
/// - B_balanced = B^db_b (typically 0.85-0.95)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DensityBalance {
    /// Per-channel exponents [R, G, B].
    /// G is always 1.0 (reference channel).
    /// R typically 1.0-1.1, B typically 0.85-0.95.
    pub exponents: [f32; 3],

    /// Source of the density balance values
    pub source: DensityBalanceSource,
}

impl Default for DensityBalance {
    fn default() -> Self {
        Self {
            // Typical starting values from research.md
            exponents: [1.05, 1.0, 0.90],
            source: DensityBalanceSource::Default,
        }
    }
}

impl DensityBalance {
    /// Create density balance from manual exponent values
    pub fn manual(red_exp: f32, blue_exp: f32) -> Self {
        Self {
            exponents: [red_exp, 1.0, blue_exp],
            source: DensityBalanceSource::Manual,
        }
    }

    /// Calculate density balance from a neutral point sample.
    ///
    /// Given RGB values from a known neutral gray area, calculates
    /// the exponents needed to make all channels equal after the
    /// power transformation.
    ///
    /// Algorithm:
    /// - G is reference (exponent = 1.0)
    /// - R^db_r = G => db_r = ln(G) / ln(R)
    /// - B^db_b = G => db_b = ln(G) / ln(B)
    pub fn from_neutral_point(neutral_rgb: [f32; 3]) -> Self {
        let [r, g, b] = neutral_rgb;

        // Avoid log(0) and ensure reasonable inputs
        let r = r.max(0.001);
        let g = g.max(0.001);
        let b = b.max(0.001);

        // G is reference (exponent = 1.0)
        // R^db_r = G => db_r = ln(G) / ln(R)
        // B^db_b = G => db_b = ln(G) / ln(B)
        let db_r = if (r - g).abs() > 0.001 {
            (g.ln() / r.ln()).clamp(0.8, 1.3)
        } else {
            1.0 // R ≈ G, no correction needed
        };

        let db_b = if (b - g).abs() > 0.001 {
            (g.ln() / b.ln()).clamp(0.7, 1.1)
        } else {
            1.0 // B ≈ G, no correction needed
        };

        Self {
            exponents: [db_r, 1.0, db_b],
            source: DensityBalanceSource::Auto,
        }
    }
}

/// Neutral point sample for density balance calculation.
///
/// Used to auto-calculate density balance exponents by sampling
/// a known neutral gray area in the image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutralPointSample {
    /// Region of interest for neutral sampling (x, y, width, height).
    /// If None, auto-detection will search for neutral areas.
    pub roi: Option<(u32, u32, u32, u32)>,

    /// Sampled RGB values from neutral point (after film base normalization)
    pub neutral_rgb: [f32; 3],

    /// Whether this was auto-detected or manually specified
    pub auto_detected: bool,
}

impl Default for NeutralPointSample {
    fn default() -> Self {
        Self {
            roi: None,
            neutral_rgb: [0.5, 0.5, 0.5],
            auto_detected: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_density_balance_default() {
        let db = DensityBalance::default();

        assert!((db.exponents[0] - 1.05).abs() < 0.001);
        assert!((db.exponents[1] - 1.0).abs() < 0.001);
        assert!((db.exponents[2] - 0.90).abs() < 0.001);
        assert_eq!(db.source, DensityBalanceSource::Default);
    }

    #[test]
    fn test_density_balance_manual() {
        let db = DensityBalance::manual(1.1, 0.85);

        assert!((db.exponents[0] - 1.1).abs() < 0.001);
        assert!((db.exponents[1] - 1.0).abs() < 0.001);
        assert!((db.exponents[2] - 0.85).abs() < 0.001);
        assert_eq!(db.source, DensityBalanceSource::Manual);
    }

    #[test]
    fn test_density_balance_from_neutral_point_equal() {
        // Equal RGB = no correction needed
        let db = DensityBalance::from_neutral_point([0.5, 0.5, 0.5]);

        // With equal channels, exponents should be close to 1.0
        assert!(
            (db.exponents[0] - 1.0).abs() < 0.01,
            "R exp should be ~1.0 for equal channels"
        );
        assert!((db.exponents[1] - 1.0).abs() < 0.001, "G exp should be 1.0");
        assert!(
            (db.exponents[2] - 1.0).abs() < 0.01,
            "B exp should be ~1.0 for equal channels"
        );
        assert_eq!(db.source, DensityBalanceSource::Auto);
    }

    #[test]
    fn test_density_balance_from_neutral_point_typical() {
        // Typical orange mask values: R=0.7, G=0.5, B=0.35
        // This should produce exponents to correct for the orange cast
        let db = DensityBalance::from_neutral_point([0.7, 0.5, 0.35]);

        // R > G, so R exponent should be > 1.0 to compress R
        assert!(
            db.exponents[0] > 1.0,
            "R exp should be > 1.0 when R > G: {}",
            db.exponents[0]
        );
        // B < G, so B exponent should be < 1.0 to expand B
        assert!(
            db.exponents[2] < 1.0,
            "B exp should be < 1.0 when B < G: {}",
            db.exponents[2]
        );
        // G always 1.0
        assert!((db.exponents[1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_density_balance_from_neutral_point_clamped() {
        // Very extreme values should be clamped
        let db = DensityBalance::from_neutral_point([0.99, 0.01, 0.001]);

        // Red should be clamped to max (1.3)
        assert!(
            db.exponents[0] <= 1.3,
            "R exp should be clamped: {}",
            db.exponents[0]
        );
        assert!(
            db.exponents[0] >= 0.8,
            "R exp should be clamped: {}",
            db.exponents[0]
        );

        // Blue should be clamped to range
        assert!(
            db.exponents[2] <= 1.1,
            "B exp should be clamped: {}",
            db.exponents[2]
        );
        assert!(
            db.exponents[2] >= 0.7,
            "B exp should be clamped: {}",
            db.exponents[2]
        );
    }

    #[test]
    fn test_density_balance_from_neutral_point_handles_near_zero() {
        // Very small values should not cause issues
        let db = DensityBalance::from_neutral_point([0.0001, 0.0001, 0.0001]);

        // All exponents should be finite
        for exp in db.exponents {
            assert!(exp.is_finite(), "Exponent should be finite: {}", exp);
        }
    }

    #[test]
    fn test_neutral_point_sample_default() {
        let nps = NeutralPointSample::default();

        assert!(nps.roi.is_none());
        assert!((nps.neutral_rgb[0] - 0.5).abs() < 0.001);
        assert!((nps.neutral_rgb[1] - 0.5).abs() < 0.001);
        assert!((nps.neutral_rgb[2] - 0.5).abs() < 0.001);
        assert!(nps.auto_detected);
    }

    #[test]
    fn test_density_balance_source_default() {
        let source = DensityBalanceSource::default();
        assert_eq!(source, DensityBalanceSource::Auto);
    }
}
