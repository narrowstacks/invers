#!/usr/bin/env python3
"""
ImageMagick CLI wrapper utilities for image comparison metrics.

This module provides functions to compute various image quality metrics
using ImageMagick's compare and identify commands.
"""

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ChannelStats:
    """Statistics for a single color channel."""
    mean: float = 0.0
    std_dev: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    kurtosis: float = 0.0
    skewness: float = 0.0
    entropy: float = 0.0


@dataclass
class HistogramData:
    """Histogram data for an image."""
    red: List[int] = field(default_factory=list)
    green: List[int] = field(default_factory=list)
    blue: List[int] = field(default_factory=list)
    luminance: List[int] = field(default_factory=list)


@dataclass
class ImageMetrics:
    """Comprehensive image comparison metrics."""
    # Error metrics
    rmse: float = 0.0           # Root Mean Square Error (0-1, lower is better)
    psnr: float = 0.0           # Peak Signal-to-Noise Ratio in dB (higher is better)
    mae: float = 0.0            # Mean Absolute Error (0-1, lower is better)
    mse: float = 0.0            # Mean Square Error (0-1, lower is better)

    # Structural metrics
    ssim: float = 0.0           # Structural Similarity Index (0-1, higher is better)
    ncc: float = 0.0            # Normalized Cross-Correlation (-1 to 1, 1 is best)

    # Per-channel differences
    channel_rmse: Dict[str, float] = field(default_factory=dict)
    channel_mae: Dict[str, float] = field(default_factory=dict)
    channel_bias: Dict[str, float] = field(default_factory=dict)  # Mean difference

    # Tonal metrics
    mean_luminance_diff: float = 0.0
    contrast_diff: float = 0.0
    histogram_correlation: float = 0.0

    # Color metrics (approximate)
    saturation_diff: float = 0.0
    hue_shift: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "rmse": self.rmse,
            "psnr": self.psnr,
            "mae": self.mae,
            "mse": self.mse,
            "ssim": self.ssim,
            "ncc": self.ncc,
            "channel_rmse": self.channel_rmse,
            "channel_mae": self.channel_mae,
            "channel_bias": self.channel_bias,
            "mean_luminance_diff": self.mean_luminance_diff,
            "contrast_diff": self.contrast_diff,
            "histogram_correlation": self.histogram_correlation,
            "saturation_diff": self.saturation_diff,
            "hue_shift": self.hue_shift,
        }


def run_magick(args: List[str], timeout: int = 120) -> Tuple[str, str, int]:
    """Run an ImageMagick command and return stdout, stderr, returncode."""
    cmd = ["magick"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", 1
    except FileNotFoundError:
        return "", "ImageMagick not found. Install with: brew install imagemagick", 1


def get_image_info(image_path: Path) -> Dict:
    """Get basic image information using ImageMagick identify."""
    stdout, stderr, rc = run_magick([
        "identify",
        "-format", "%w %h %z %m %[colorspace]",
        str(image_path)
    ])

    if rc != 0:
        raise RuntimeError(f"Failed to get image info: {stderr}")

    parts = stdout.strip().split()
    return {
        "width": int(parts[0]),
        "height": int(parts[1]),
        "depth": int(parts[2]),
        "format": parts[3],
        "colorspace": parts[4] if len(parts) > 4 else "unknown",
    }


def compute_channel_stats(image_path: Path) -> Dict[str, ChannelStats]:
    """Compute per-channel statistics using ImageMagick."""
    stats = {}

    # Get verbose statistics
    stdout, stderr, rc = run_magick([
        "identify",
        "-verbose",
        str(image_path)
    ])

    if rc != 0:
        raise RuntimeError(f"Failed to compute stats: {stderr}")

    # Parse channel statistics from verbose output
    current_channel = None
    channel_data = {}

    for line in stdout.split('\n'):
        line = line.strip()

        # Detect channel sections
        if line.startswith('Channel statistics:'):
            continue
        elif line in ['Red:', 'Green:', 'Blue:', 'Alpha:', 'Gray:', 'Overall:']:
            current_channel = line.rstrip(':').lower()
            channel_data[current_channel] = ChannelStats()
        elif current_channel and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()

            # Extract numeric value (handle formats like "0.5 (128)")
            match = re.match(r'([0-9.e+-]+)', value)
            if match:
                try:
                    num_val = float(match.group(1))
                    cs = channel_data[current_channel]

                    if key == 'mean':
                        cs.mean = num_val
                    elif key == 'standard deviation':
                        cs.std_dev = num_val
                    elif key in ['min', 'minimum']:
                        cs.min_val = num_val
                    elif key in ['max', 'maximum']:
                        cs.max_val = num_val
                    elif key == 'kurtosis':
                        cs.kurtosis = num_val
                    elif key == 'skewness':
                        cs.skewness = num_val
                    elif key == 'entropy':
                        cs.entropy = num_val
                except ValueError:
                    pass

    return channel_data


def compute_histogram(image_path: Path, bins: int = 256) -> HistogramData:
    """Compute histogram data for an image."""
    hist = HistogramData()

    # Get histogram using ImageMagick
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Generate histogram text file
        stdout, stderr, rc = run_magick([
            str(image_path),
            "-format", "%c",
            "-depth", "8",
            "histogram:info:" + tmp_path
        ])

        if rc != 0:
            # Fallback: try simpler histogram approach
            stdout, stderr, rc = run_magick([
                str(image_path),
                "-depth", "8",
                "-format", "%c",
                "histogram:info:-"
            ])
            if rc != 0:
                return hist
            hist_text = stdout
        else:
            with open(tmp_path, 'r') as f:
                hist_text = f.read()

        # Initialize bins
        hist.red = [0] * bins
        hist.green = [0] * bins
        hist.blue = [0] * bins
        hist.luminance = [0] * bins

        # Parse histogram output
        # Format: "count: (r,g,b) #hex srgb(r,g,b)"
        for line in hist_text.split('\n'):
            match = re.search(r'(\d+):\s*\(\s*(\d+),\s*(\d+),\s*(\d+)\)', line)
            if match:
                count = int(match.group(1))
                r = int(match.group(2))
                g = int(match.group(3))
                b = int(match.group(4))

                if 0 <= r < bins:
                    hist.red[r] += count
                if 0 <= g < bins:
                    hist.green[g] += count
                if 0 <= b < bins:
                    hist.blue[b] += count

                # Compute luminance bin
                lum = int(0.299 * r + 0.587 * g + 0.114 * b)
                if 0 <= lum < bins:
                    hist.luminance[lum] += count

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return hist


def compute_comparison_metrics(
    image1_path: Path,
    image2_path: Path,
    resize: Optional[int] = None
) -> ImageMetrics:
    """
    Compute comprehensive comparison metrics between two images.

    Args:
        image1_path: Path to first image (typically invers output)
        image2_path: Path to second image (reference)
        resize: Optional max dimension for faster processing

    Returns:
        ImageMetrics with all computed metrics
    """
    metrics = ImageMetrics()

    # Prepare images (resize if needed)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        img1 = tmp / "img1.miff"
        img2 = tmp / "img2.miff"

        # Convert to common format and optionally resize
        resize_args = ["-resize", f"{resize}x{resize}>"] if resize else []

        for src, dst in [(image1_path, img1), (image2_path, img2)]:
            stdout, stderr, rc = run_magick([
                str(src),
                *resize_args,
                "-colorspace", "sRGB",
                "-depth", "16",
                str(dst)
            ])
            if rc != 0:
                raise RuntimeError(f"Failed to prepare image {src}: {stderr}")

        # Compute RMSE
        stdout, stderr, rc = run_magick([
            "compare",
            "-metric", "RMSE",
            str(img1), str(img2),
            "null:"
        ])
        output = stdout + stderr
        match = re.search(r'([0-9.e+-]+)\s*\(([0-9.e+-]+)\)', output)
        if match:
            metrics.rmse = float(match.group(2))  # Normalized value
            metrics.mse = metrics.rmse ** 2

        # Compute PSNR
        stdout, stderr, rc = run_magick([
            "compare",
            "-metric", "PSNR",
            str(img1), str(img2),
            "null:"
        ])
        output = stdout + stderr
        match = re.search(r'([0-9.e+-]+)', output)
        if match:
            try:
                metrics.psnr = float(match.group(1))
                if metrics.psnr == float('inf'):
                    metrics.psnr = 100.0  # Cap at 100 dB for identical images
            except ValueError:
                metrics.psnr = 0.0

        # Compute MAE
        stdout, stderr, rc = run_magick([
            "compare",
            "-metric", "MAE",
            str(img1), str(img2),
            "null:"
        ])
        output = stdout + stderr
        match = re.search(r'([0-9.e+-]+)\s*\(([0-9.e+-]+)\)', output)
        if match:
            metrics.mae = float(match.group(2))

        # Compute SSIM (if available in ImageMagick build)
        stdout, stderr, rc = run_magick([
            "compare",
            "-metric", "SSIM",
            str(img1), str(img2),
            "null:"
        ])
        output = stdout + stderr
        match = re.search(r'([0-9.e+-]+)', output)
        if match:
            try:
                metrics.ssim = float(match.group(1))
            except ValueError:
                metrics.ssim = 0.0

        # Compute NCC (Normalized Cross-Correlation)
        stdout, stderr, rc = run_magick([
            "compare",
            "-metric", "NCC",
            str(img1), str(img2),
            "null:"
        ])
        output = stdout + stderr
        match = re.search(r'([0-9.e+-]+)', output)
        if match:
            try:
                metrics.ncc = float(match.group(1))
            except ValueError:
                metrics.ncc = 0.0

        # Compute per-channel metrics
        for channel in ['red', 'green', 'blue']:
            ch_img1 = tmp / f"ch1_{channel}.miff"
            ch_img2 = tmp / f"ch2_{channel}.miff"

            # Extract channel
            for src, dst in [(img1, ch_img1), (img2, ch_img2)]:
                run_magick([
                    str(src),
                    "-channel", channel.capitalize(),
                    "-separate",
                    str(dst)
                ])

            # Channel RMSE
            stdout, stderr, rc = run_magick([
                "compare",
                "-metric", "RMSE",
                str(ch_img1), str(ch_img2),
                "null:"
            ])
            output = stdout + stderr
            match = re.search(r'([0-9.e+-]+)\s*\(([0-9.e+-]+)\)', output)
            if match:
                metrics.channel_rmse[channel] = float(match.group(2))

            # Channel MAE
            stdout, stderr, rc = run_magick([
                "compare",
                "-metric", "MAE",
                str(ch_img1), str(ch_img2),
                "null:"
            ])
            output = stdout + stderr
            match = re.search(r'([0-9.e+-]+)\s*\(([0-9.e+-]+)\)', output)
            if match:
                metrics.channel_mae[channel] = float(match.group(2))

        # Compute channel bias (mean difference)
        stats1 = compute_channel_stats(img1)
        stats2 = compute_channel_stats(img2)

        for channel in ['red', 'green', 'blue']:
            if channel in stats1 and channel in stats2:
                metrics.channel_bias[channel] = stats1[channel].mean - stats2[channel].mean

        # Compute luminance difference
        if 'overall' in stats1 and 'overall' in stats2:
            metrics.mean_luminance_diff = stats1['overall'].mean - stats2['overall'].mean
            metrics.contrast_diff = stats1['overall'].std_dev - stats2['overall'].std_dev

        # Compute histogram correlation
        hist1 = compute_histogram(img1)
        hist2 = compute_histogram(img2)
        metrics.histogram_correlation = _compute_histogram_correlation(
            hist1.luminance, hist2.luminance
        )

        # Compute approximate saturation and hue differences
        sat_diff, hue_diff = _compute_color_differences(img1, img2)
        metrics.saturation_diff = sat_diff
        metrics.hue_shift = hue_diff

    return metrics


def _compute_histogram_correlation(hist1: List[int], hist2: List[int]) -> float:
    """Compute Pearson correlation between two histograms."""
    if not hist1 or not hist2 or len(hist1) != len(hist2):
        return 0.0

    n = len(hist1)
    sum1 = sum(hist1)
    sum2 = sum(hist2)

    if sum1 == 0 or sum2 == 0:
        return 0.0

    # Normalize
    h1 = [x / sum1 for x in hist1]
    h2 = [x / sum2 for x in hist2]

    mean1 = sum(h1) / n
    mean2 = sum(h2) / n

    # Compute correlation
    num = sum((h1[i] - mean1) * (h2[i] - mean2) for i in range(n))
    den1 = sum((h1[i] - mean1) ** 2 for i in range(n)) ** 0.5
    den2 = sum((h2[i] - mean2) ** 2 for i in range(n)) ** 0.5

    if den1 == 0 or den2 == 0:
        return 0.0

    return num / (den1 * den2)


def _compute_color_differences(img1: Path, img2: Path) -> Tuple[float, float]:
    """Compute approximate saturation and hue differences."""
    sat_diff = 0.0
    hue_diff = 0.0

    # Convert to HSL and compare
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        hsl1 = tmp / "hsl1.txt"
        hsl2 = tmp / "hsl2.txt"

        # Get mean HSL values
        for src, dst in [(img1, hsl1), (img2, hsl2)]:
            stdout, stderr, rc = run_magick([
                str(src),
                "-colorspace", "HSL",
                "-format", "%[fx:mean.r] %[fx:mean.g] %[fx:mean.b]",
                "info:"
            ])
            if rc == 0:
                with open(dst, 'w') as f:
                    f.write(stdout)

        try:
            with open(hsl1) as f:
                h1, s1, l1 = map(float, f.read().strip().split())
            with open(hsl2) as f:
                h2, s2, l2 = map(float, f.read().strip().split())

            sat_diff = s1 - s2

            # Hue is circular, compute shortest angular distance
            hue_diff = h1 - h2
            if hue_diff > 0.5:
                hue_diff -= 1.0
            elif hue_diff < -0.5:
                hue_diff += 1.0
            hue_diff *= 360  # Convert to degrees

        except (ValueError, FileNotFoundError):
            pass

    return sat_diff, hue_diff


def create_diff_image(
    image1_path: Path,
    image2_path: Path,
    output_path: Path,
    mode: str = "absolute"
) -> None:
    """
    Create a difference visualization between two images.

    Args:
        image1_path: First image path
        image2_path: Second image path
        output_path: Output difference image path
        mode: "absolute" for abs diff, "highlight" for red/blue overlay
    """
    if mode == "absolute":
        # Absolute difference with enhancement
        run_magick([
            str(image1_path),
            str(image2_path),
            "-compose", "difference",
            "-composite",
            "-auto-level",
            "-evaluate", "multiply", "5",
            str(output_path)
        ])
    elif mode == "highlight":
        # Red/cyan highlight of differences
        run_magick([
            "(",
            str(image1_path),
            str(image2_path),
            "-compose", "difference",
            "-composite",
            "-threshold", "5%",
            ")",
            str(image1_path),
            "-compose", "multiply",
            "-composite",
            str(output_path)
        ])
    else:
        # Default ImageMagick compare visualization
        run_magick([
            "compare",
            str(image1_path),
            str(image2_path),
            str(output_path)
        ])


def create_side_by_side(
    images: List[Tuple[Path, str]],
    output_path: Path,
    thumbnail_width: int = 800,
    include_labels: bool = True
) -> None:
    """
    Create a side-by-side comparison image with labels.

    Args:
        images: List of (image_path, label) tuples
        output_path: Output image path
        thumbnail_width: Width of each thumbnail
        include_labels: Whether to add text labels
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        processed = []

        for i, (img_path, label) in enumerate(images):
            proc_path = tmp / f"proc_{i}.png"

            if include_labels:
                run_magick([
                    str(img_path),
                    "-resize", f"{thumbnail_width}x",
                    "-gravity", "South",
                    "-background", "white",
                    "-splice", "0x40",
                    "-gravity", "South",
                    "-font", "Helvetica",
                    "-pointsize", "16",
                    "-annotate", "+0+10", label,
                    str(proc_path)
                ])
            else:
                run_magick([
                    str(img_path),
                    "-resize", f"{thumbnail_width}x",
                    str(proc_path)
                ])

            processed.append(proc_path)

        # Combine horizontally
        run_magick([
            *[str(p) for p in processed],
            "+append",
            str(output_path)
        ])


def create_histogram_comparison(
    image1_path: Path,
    image2_path: Path,
    output_path: Path,
    label1: str = "Image 1",
    label2: str = "Image 2"
) -> None:
    """
    Create a histogram comparison visualization.

    Args:
        image1_path: First image path
        image2_path: Second image path
        output_path: Output histogram comparison path
        label1: Label for first image
        label2: Label for second image
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        hist1 = tmp / "hist1.png"
        hist2 = tmp / "hist2.png"

        # Generate histograms
        for src, dst, label in [(image1_path, hist1, label1), (image2_path, hist2, label2)]:
            run_magick([
                str(src),
                "-define", "histogram:unique-colors=false",
                "-write", "histogram:" + str(dst),
                "-delete", "0",
                str(dst)
            ])

            # Add label
            run_magick([
                str(dst),
                "-gravity", "North",
                "-background", "white",
                "-splice", "0x30",
                "-font", "Helvetica",
                "-pointsize", "14",
                "-annotate", "+0+5", label,
                str(dst)
            ])

        # Combine vertically
        run_magick([
            str(hist1),
            str(hist2),
            "-append",
            str(output_path)
        ])


def normalize_image(
    image_path: Path,
    output_path: Path,
    method: str = "stretch"
) -> None:
    """
    Normalize an image for fair comparison.

    Args:
        image_path: Input image path
        output_path: Output normalized image path
        method: "stretch" for auto-levels, "normalize" for full normalize
    """
    if method == "stretch":
        run_magick([
            str(image_path),
            "-auto-level",
            str(output_path)
        ])
    elif method == "normalize":
        run_magick([
            str(image_path),
            "-normalize",
            str(output_path)
        ])
    else:
        # Just copy
        run_magick([
            str(image_path),
            str(output_path)
        ])


if __name__ == "__main__":
    # Simple test
    import sys
    if len(sys.argv) >= 3:
        img1 = Path(sys.argv[1])
        img2 = Path(sys.argv[2])

        print(f"Comparing {img1} vs {img2}")
        metrics = compute_comparison_metrics(img1, img2)

        print(f"\nMetrics:")
        print(f"  RMSE: {metrics.rmse:.6f}")
        print(f"  PSNR: {metrics.psnr:.2f} dB")
        print(f"  SSIM: {metrics.ssim:.6f}")
        print(f"  MAE:  {metrics.mae:.6f}")
        print(f"  NCC:  {metrics.ncc:.6f}")
        print(f"\nChannel RMSE:")
        for ch, val in metrics.channel_rmse.items():
            print(f"  {ch}: {val:.6f}")
        print(f"\nChannel Bias:")
        for ch, val in metrics.channel_bias.items():
            print(f"  {ch}: {val:+.6f}")
    else:
        print("Usage: python imagemagick.py <image1> <image2>")
