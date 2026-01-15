//! Histogram utilities for image analysis
//!
//! Provides functions for finding histogram ends, Otsu's thresholding, and other
//! histogram-based analysis methods.

/// Result of histogram ends detection
#[derive(Debug, Clone, Copy)]
pub struct HistogramEnds {
    /// First non-zero bin index (0.0-1.0 normalized)
    pub low: f32,
    /// Last non-zero bin index (0.0-1.0 normalized)
    pub high: f32,
    /// Number of non-empty bins
    pub non_empty_bins: usize,
}

/// Find first and last non-zero histogram bins
///
/// Finds the actual data range without percentile clipping.
/// Useful for detecting true black/white points in an image.
///
/// Returns normalized values (0.0-1.0) for the first and last non-empty bins.
pub fn find_histogram_ends(data: &[f32]) -> HistogramEnds {
    const NUM_BUCKETS: usize = 65536;
    let mut histogram = vec![0u32; NUM_BUCKETS];

    // Build histogram
    for &value in data {
        let bucket =
            ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        histogram[bucket] += 1;
    }

    // Find first non-zero bin
    let mut low_bucket = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            low_bucket = i;
            break;
        }
    }

    // Find last non-zero bin
    let mut high_bucket = NUM_BUCKETS - 1;
    for (i, &count) in histogram.iter().enumerate().rev() {
        if count > 0 {
            high_bucket = i;
            break;
        }
    }

    // Count non-empty bins
    let non_empty_bins = histogram.iter().filter(|&&c| c > 0).count();

    HistogramEnds {
        low: low_bucket as f32 / (NUM_BUCKETS - 1) as f32,
        high: high_bucket as f32 / (NUM_BUCKETS - 1) as f32,
        non_empty_bins,
    }
}

/// Find histogram ends per channel for RGB data
///
/// Returns [R_ends, G_ends, B_ends]
pub fn find_histogram_ends_rgb(data: &[f32]) -> [HistogramEnds; 3] {
    const NUM_BUCKETS: usize = 65536;
    let mut r_hist = vec![0u32; NUM_BUCKETS];
    let mut g_hist = vec![0u32; NUM_BUCKETS];
    let mut b_hist = vec![0u32; NUM_BUCKETS];

    // Build per-channel histograms
    for pixel in data.chunks_exact(3) {
        let r_bucket =
            ((pixel[0].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let g_bucket =
            ((pixel[1].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        let b_bucket =
            ((pixel[2].clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        r_hist[r_bucket] += 1;
        g_hist[g_bucket] += 1;
        b_hist[b_bucket] += 1;
    }

    [
        histogram_ends_from_array(&r_hist),
        histogram_ends_from_array(&g_hist),
        histogram_ends_from_array(&b_hist),
    ]
}

/// Helper to extract ends from a pre-built histogram
fn histogram_ends_from_array(histogram: &[u32]) -> HistogramEnds {
    let num_buckets = histogram.len();

    let mut low_bucket = 0usize;
    for (i, &count) in histogram.iter().enumerate() {
        if count > 0 {
            low_bucket = i;
            break;
        }
    }

    let mut high_bucket = num_buckets - 1;
    for (i, &count) in histogram.iter().enumerate().rev() {
        if count > 0 {
            high_bucket = i;
            break;
        }
    }

    let non_empty_bins = histogram.iter().filter(|&&c| c > 0).count();

    HistogramEnds {
        low: low_bucket as f32 / (num_buckets - 1) as f32,
        high: high_bucket as f32 / (num_buckets - 1) as f32,
        non_empty_bins,
    }
}

/// Result of Otsu's threshold computation
#[derive(Debug, Clone, Copy)]
pub struct OtsuResult {
    /// Optimal threshold value (0.0-1.0)
    pub threshold: f32,
    /// Inter-class variance at the threshold
    pub variance: f32,
    /// Ratio of pixels below threshold
    pub below_ratio: f32,
}

/// Otsu's method for automatic thresholding
///
/// Finds the optimal threshold that separates an image into two classes
/// (e.g., film base vs image content) by maximizing inter-class variance.
///
/// This is useful for automatically detecting the boundary between
/// unexposed film base and actual image content.
///
/// Returns the optimal threshold (0.0-1.0) and the inter-class variance.
pub fn otsu_threshold(data: &[f32]) -> OtsuResult {
    const NUM_BUCKETS: usize = 256; // 256 bins is standard for Otsu

    // Build histogram
    let mut histogram = vec![0u32; NUM_BUCKETS];
    for &value in data {
        let bucket =
            ((value.clamp(0.0, 1.0) * (NUM_BUCKETS - 1) as f32) as usize).min(NUM_BUCKETS - 1);
        histogram[bucket] += 1;
    }

    let total = data.len() as f32;
    if total < 2.0 {
        return OtsuResult {
            threshold: 0.5,
            variance: 0.0,
            below_ratio: 0.5,
        };
    }

    // Calculate normalized histogram (probabilities)
    let probabilities: Vec<f32> = histogram.iter().map(|&c| c as f32 / total).collect();

    // Calculate cumulative sums and cumulative means
    let mut cum_sum = vec![0.0f32; NUM_BUCKETS];
    let mut cum_mean = vec![0.0f32; NUM_BUCKETS];

    cum_sum[0] = probabilities[0];
    cum_mean[0] = 0.0; // bucket 0 * p[0]

    for i in 1..NUM_BUCKETS {
        cum_sum[i] = cum_sum[i - 1] + probabilities[i];
        cum_mean[i] = cum_mean[i - 1] + (i as f32) * probabilities[i];
    }

    let global_mean = cum_mean[NUM_BUCKETS - 1];

    // Find threshold that maximizes inter-class variance
    let mut max_variance = 0.0f32;
    let mut optimal_threshold = 0usize;

    for t in 0..NUM_BUCKETS - 1 {
        let w0 = cum_sum[t]; // Weight of class 0 (below threshold)
        let w1 = 1.0 - w0; // Weight of class 1 (above threshold)

        if w0 < 1e-10 || w1 < 1e-10 {
            continue;
        }

        let mu0 = cum_mean[t] / w0; // Mean of class 0
        let mu1 = (global_mean - cum_mean[t]) / w1; // Mean of class 1

        // Inter-class variance
        let variance = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);

        if variance > max_variance {
            max_variance = variance;
            optimal_threshold = t;
        }
    }

    OtsuResult {
        threshold: optimal_threshold as f32 / (NUM_BUCKETS - 1) as f32,
        variance: max_variance,
        below_ratio: cum_sum[optimal_threshold],
    }
}

/// Otsu threshold for RGB data using luminance
///
/// Computes Otsu threshold on the luminance channel (Rec.709 weights)
pub fn otsu_threshold_rgb(data: &[f32]) -> OtsuResult {
    // Extract luminance values
    let luminances: Vec<f32> = data
        .chunks_exact(3)
        .map(|rgb| 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])
        .collect();

    otsu_threshold(&luminances)
}
