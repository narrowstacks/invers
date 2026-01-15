//! Statistical analysis functions for diagnostic comparisons

/// Statistics for a single channel
#[derive(Debug, Clone)]
pub struct ChannelStats {
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub percentiles: Vec<(u8, f32)>, // (percentile, value)
}

/// Histogram data for a channel (256 bins)
#[derive(Debug, Clone)]
pub struct Histogram {
    pub bins: Vec<u32>,
    pub bin_edges: Vec<f32>,
}

/// Compute comprehensive statistics for an image in a single pass
pub fn compute_statistics(data: &[f32], channels: u8) -> [ChannelStats; 3] {
    if channels != 3 {
        panic!("Only 3-channel RGB images supported");
    }

    let num_pixels = data.len() / 3;

    // Pre-allocate with exact capacity to avoid reallocation
    let mut channel_data: [Vec<f32>; 3] = [
        Vec::with_capacity(num_pixels),
        Vec::with_capacity(num_pixels),
        Vec::with_capacity(num_pixels),
    ];

    // Single pass to separate channels
    for pixel in data.chunks_exact(3) {
        channel_data[0].push(pixel[0]);
        channel_data[1].push(pixel[1]);
        channel_data[2].push(pixel[2]);
    }

    // Compute stats for each channel
    [
        compute_channel_stats(&channel_data[0]),
        compute_channel_stats(&channel_data[1]),
        compute_channel_stats(&channel_data[2]),
    ]
}

/// Compute statistics for a single channel using efficient algorithms
fn compute_channel_stats(data: &[f32]) -> ChannelStats {
    if data.is_empty() {
        return ChannelStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            percentiles: vec![],
        };
    }

    // Create a sorted copy for percentile calculations
    let mut sorted = data.to_vec();

    // Single pass for min, max, and sum
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0;

    for &val in data {
        min = min.min(val);
        max = max.max(val);
        sum += val;
    }

    let mean = sum / data.len() as f32;

    // Use partial sort for median - only sort what we need
    let mid = sorted.len() / 2;
    sorted.select_nth_unstable_by(mid, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    let median = sorted[mid];

    // Compute standard deviation in a single pass
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
    let std_dev = variance.sqrt();

    // Compute percentiles efficiently using partial sorts
    let percentile_values = vec![1, 5, 25, 50, 75, 95, 99];
    let mut percentiles = Vec::with_capacity(percentile_values.len());

    for p in percentile_values {
        let idx = ((p as f32 / 100.0) * (sorted.len() - 1) as f32).round() as usize;
        // Use select_nth_unstable for each percentile
        sorted.select_nth_unstable_by(idx, |a, b| {
            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
        });
        percentiles.push((p, sorted[idx]));
    }

    ChannelStats {
        min,
        max,
        mean,
        median,
        std_dev,
        percentiles,
    }
}

/// Generate histogram for an image with pre-allocated buffers
pub fn compute_histograms(data: &[f32], channels: u8, num_bins: usize) -> [Histogram; 3] {
    if channels != 3 {
        panic!("Only 3-channel RGB images supported");
    }

    let num_pixels = data.len() / 3;

    // Pre-allocate with exact capacity
    let mut channel_data: [Vec<f32>; 3] = [
        Vec::with_capacity(num_pixels),
        Vec::with_capacity(num_pixels),
        Vec::with_capacity(num_pixels),
    ];

    // Single pass to separate channels
    for pixel in data.chunks_exact(3) {
        channel_data[0].push(pixel[0]);
        channel_data[1].push(pixel[1]);
        channel_data[2].push(pixel[2]);
    }

    [
        compute_channel_histogram(&channel_data[0], num_bins),
        compute_channel_histogram(&channel_data[1], num_bins),
        compute_channel_histogram(&channel_data[2], num_bins),
    ]
}

/// Generate histogram for a single channel
fn compute_channel_histogram(data: &[f32], num_bins: usize) -> Histogram {
    let mut bins = vec![0u32; num_bins];
    let bin_edges: Vec<f32> = (0..=num_bins).map(|i| i as f32 / num_bins as f32).collect();

    for &value in data {
        let clamped = value.clamp(0.0, 1.0);
        let bin_idx = ((clamped * (num_bins - 1) as f32) as usize).min(num_bins - 1);
        bins[bin_idx] += 1;
    }

    Histogram { bins, bin_edges }
}
