"""Utilities for reference comparison testing."""

from .imagemagick import (
    ImageMetrics,
    ChannelStats,
    HistogramData,
    get_image_info,
    compute_comparison_metrics,
    compute_channel_stats,
    compute_histogram,
    create_diff_image,
    create_side_by_side,
    create_histogram_comparison,
    normalize_image,
)

__all__ = [
    "ImageMetrics",
    "ChannelStats",
    "HistogramData",
    "get_image_info",
    "compute_comparison_metrics",
    "compute_channel_stats",
    "compute_histogram",
    "create_diff_image",
    "create_side_by_side",
    "create_histogram_comparison",
    "normalize_image",
]
