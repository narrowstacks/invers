//! Diagnostic tools for comparing image conversions
//!
//! Provides comprehensive comparison between our conversion pipeline
//! and third-party software conversions.

mod compare;
mod output;
mod stats;

pub use compare::{
    compare_conversions, create_difference_map, extract_sample_patches, RgbSamplePatches,
};
pub use output::{print_report, save_diagnostic_images};
pub use stats::{compute_histograms, compute_statistics, ChannelStats, Histogram};

/// Complete diagnostic comparison result
#[derive(Debug)]
pub struct DiagnosticReport {
    pub our_stats: [ChannelStats; 3],
    pub third_party_stats: [ChannelStats; 3],
    pub difference_stats: [ChannelStats; 3],
    pub our_histograms: [Histogram; 3],
    pub third_party_histograms: [Histogram; 3],
    pub color_shift: [f32; 3], // Average offset per channel
    pub exposure_ratio: f32,   // Overall brightness ratio (third_party / ours)
}

impl DiagnosticReport {
    /// Create a new diagnostic report from computed statistics
    pub fn new(
        our_stats: [ChannelStats; 3],
        third_party_stats: [ChannelStats; 3],
        difference_stats: [ChannelStats; 3],
        our_histograms: [Histogram; 3],
        third_party_histograms: [Histogram; 3],
        color_shift: [f32; 3],
        exposure_ratio: f32,
    ) -> Self {
        Self {
            our_stats,
            third_party_stats,
            difference_stats,
            our_histograms,
            third_party_histograms,
            color_shift,
            exposure_ratio,
        }
    }
}
