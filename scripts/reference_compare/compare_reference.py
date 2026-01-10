#!/usr/bin/env python3
"""
Compare invers conversion results against reference images.

This script runs invers with specified settings on a negative image,
then computes comprehensive metrics comparing the output to a reference
positive image from professional software (e.g., Grain2Pixel, commercial plugins).

Usage:
    python compare_reference.py negative.tif reference.tif [options]

Examples:
    # Basic comparison with default settings
    python compare_reference.py scan.tif reference.tif

    # Compare with specific inversion mode
    python compare_reference.py scan.tif reference.tif --inversion-mode mask-aware

    # Test multiple inversion modes
    python compare_reference.py scan.tif reference.tif --test-all-modes

    # With manual base values
    python compare_reference.py scan.tif reference.tif --base 0.736,0.537,0.357
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.imagemagick import (
    ImageMetrics,
    compute_comparison_metrics,
    compute_channel_stats,
    create_diff_image,
    create_side_by_side,
    create_histogram_comparison,
    get_image_info,
)


@dataclass
class ConversionConfig:
    """Configuration for a single invers conversion run."""
    label: str
    inversion_mode: Optional[str] = None
    base_rgb: Optional[Tuple[float, float, float]] = None
    exposure: float = 1.0
    auto_wb: bool = False
    no_tonecurve: bool = False
    no_colormatrix: bool = False
    no_auto_levels: bool = False
    extra_args: Optional[List[str]] = None


@dataclass
class ComparisonResult:
    """Result of a single comparison run."""
    config: ConversionConfig
    output_path: Path
    metrics: ImageMetrics
    processing_time: float
    detected_base: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "label": self.config.label,
                "inversion_mode": self.config.inversion_mode,
                "base_rgb": self.config.base_rgb,
                "exposure": self.config.exposure,
                "auto_wb": self.config.auto_wb,
                "no_tonecurve": self.config.no_tonecurve,
                "no_colormatrix": self.config.no_colormatrix,
                "no_auto_levels": self.config.no_auto_levels,
            },
            "output_path": str(self.output_path),
            "metrics": self.metrics.to_dict(),
            "processing_time": self.processing_time,
            "detected_base": self.detected_base,
        }


def find_invers_binary() -> Path:
    """Find the invers binary, preferring release build."""
    # Check environment variable first
    if "INVERS_BIN" in os.environ:
        path = Path(os.environ["INVERS_BIN"])
        if path.exists():
            return path

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent

    release_path = repo_root / "target" / "release" / "invers"
    debug_path = repo_root / "target" / "debug" / "invers"

    if release_path.exists():
        return release_path
    elif debug_path.exists():
        print("Warning: Using debug build (slower)", file=sys.stderr)
        return debug_path
    else:
        # Try system PATH
        result = subprocess.run(["which", "invers"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
        raise FileNotFoundError(
            "Could not find invers binary. Run 'cargo build --release' first."
        )


def parse_base_from_output(output: str) -> Optional[Tuple[float, float, float]]:
    """Extract base RGB values from invers verbose output."""
    import re
    match = re.search(r'Base \(RGB\): \[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', output)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    return None


def parse_time_from_output(output: str) -> float:
    """Extract processing time from invers output."""
    import re
    match = re.search(r'\(([0-9.]+)s\)', output)
    if match:
        return float(match.group(1))
    return 0.0


def run_invers(
    invers_path: Path,
    input_path: Path,
    output_path: Path,
    config: ConversionConfig,
) -> Tuple[str, float, Optional[Tuple[float, float, float]]]:
    """
    Run invers with specified configuration.

    Returns:
        Tuple of (combined output, processing time, detected base RGB)
    """
    cmd = [
        str(invers_path),
        "convert",
        str(input_path),
        "--verbose",
        "-o", str(output_path),
    ]

    if config.inversion_mode:
        cmd.extend(["--inversion-mode", config.inversion_mode])

    if config.base_rgb:
        base_str = f"{config.base_rgb[0]},{config.base_rgb[1]},{config.base_rgb[2]}"
        cmd.extend(["--base", base_str])

    if config.exposure != 1.0:
        cmd.extend(["--exposure", str(config.exposure)])

    if config.auto_wb:
        cmd.append("--auto-wb")

    if config.no_tonecurve:
        cmd.append("--no-tonecurve")

    if config.no_colormatrix:
        cmd.append("--no-colormatrix")

    if config.no_auto_levels:
        cmd.append("--no-auto-levels")

    if config.extra_args:
        cmd.extend(config.extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    combined_output = result.stdout + result.stderr

    processing_time = parse_time_from_output(combined_output)
    detected_base = parse_base_from_output(combined_output)

    if result.returncode != 0:
        print(f"Warning: invers returned non-zero exit code", file=sys.stderr)
        print(f"Output: {combined_output}", file=sys.stderr)

    return combined_output, processing_time, detected_base


def run_comparison(
    invers_path: Path,
    negative_path: Path,
    reference_path: Path,
    config: ConversionConfig,
    output_dir: Path,
    resize: Optional[int] = None,
) -> ComparisonResult:
    """
    Run a single comparison: convert with invers, compare to reference.

    Args:
        invers_path: Path to invers binary
        negative_path: Input negative image
        reference_path: Reference positive image
        config: Conversion configuration
        output_dir: Directory for output files
        resize: Optional resize dimension for faster metric computation

    Returns:
        ComparisonResult with metrics and paths
    """
    # Generate output filename
    safe_label = config.label.replace(" ", "_").replace("/", "-")
    output_path = output_dir / f"{negative_path.stem}_{safe_label}.tif"

    # Run invers conversion
    output, processing_time, detected_base = run_invers(
        invers_path, negative_path, output_path, config
    )

    if not output_path.exists():
        raise RuntimeError(f"Conversion failed, output not created: {output_path}")

    # Compute comparison metrics
    metrics = compute_comparison_metrics(output_path, reference_path, resize=resize)

    return ComparisonResult(
        config=config,
        output_path=output_path,
        metrics=metrics,
        processing_time=processing_time,
        detected_base=detected_base,
    )


def print_metrics_table(results: List[ComparisonResult]) -> None:
    """Print a formatted table of metrics for all results."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Comparison Results")

        table.add_column("Config", style="cyan")
        table.add_column("RMSE", justify="right")
        table.add_column("PSNR", justify="right")
        table.add_column("SSIM", justify="right")
        table.add_column("MAE", justify="right")
        table.add_column("Time", justify="right")

        for r in results:
            # Color-code SSIM
            ssim_str = f"{r.metrics.ssim:.4f}"
            if r.metrics.ssim > 0.95:
                ssim_str = f"[green]{ssim_str}[/green]"
            elif r.metrics.ssim > 0.85:
                ssim_str = f"[yellow]{ssim_str}[/yellow]"
            else:
                ssim_str = f"[red]{ssim_str}[/red]"

            table.add_row(
                r.config.label,
                f"{r.metrics.rmse:.4f}",
                f"{r.metrics.psnr:.1f} dB",
                ssim_str,
                f"{r.metrics.mae:.4f}",
                f"{r.processing_time:.2f}s",
            )

        console.print(table)

    except ImportError:
        # Fallback to simple table
        print("\n" + "=" * 80)
        print(f"{'Config':<25} {'RMSE':>10} {'PSNR':>12} {'SSIM':>10} {'MAE':>10} {'Time':>8}")
        print("-" * 80)
        for r in results:
            print(f"{r.config.label:<25} {r.metrics.rmse:>10.4f} {r.metrics.psnr:>10.1f} dB "
                  f"{r.metrics.ssim:>10.4f} {r.metrics.mae:>10.4f} {r.processing_time:>7.2f}s")
        print("=" * 80)


def print_channel_analysis(results: List[ComparisonResult]) -> None:
    """Print per-channel analysis for the best result."""
    if not results:
        return

    # Find best result by SSIM
    best = max(results, key=lambda r: r.metrics.ssim)

    print(f"\nChannel Analysis (best result: {best.config.label}):")
    print("-" * 50)

    print(f"{'Channel':<10} {'RMSE':>12} {'MAE':>12} {'Bias':>12}")
    for ch in ['red', 'green', 'blue']:
        rmse = best.metrics.channel_rmse.get(ch, 0)
        mae = best.metrics.channel_mae.get(ch, 0)
        bias = best.metrics.channel_bias.get(ch, 0)
        print(f"{ch.capitalize():<10} {rmse:>12.4f} {mae:>12.4f} {bias:>+12.4f}")

    print(f"\nTonal Metrics:")
    print(f"  Mean Luminance Diff: {best.metrics.mean_luminance_diff:+.4f}")
    print(f"  Contrast Diff:       {best.metrics.contrast_diff:+.4f}")
    print(f"  Histogram Corr:      {best.metrics.histogram_correlation:.4f}")

    print(f"\nColor Metrics:")
    print(f"  Saturation Diff:     {best.metrics.saturation_diff:+.4f}")
    print(f"  Hue Shift:           {best.metrics.hue_shift:+.1f} degrees")


def generate_visual_comparison(
    results: List[ComparisonResult],
    reference_path: Path,
    output_dir: Path,
    thumbnail_width: int = 600,
) -> Path:
    """Generate visual comparison image."""
    images = [(reference_path, "Reference")]
    for r in results:
        images.append((r.output_path, r.config.label))

    output_path = output_dir / "visual_comparison.png"
    create_side_by_side(images, output_path, thumbnail_width=thumbnail_width)
    return output_path


def generate_diff_images(
    results: List[ComparisonResult],
    reference_path: Path,
    output_dir: Path,
) -> List[Path]:
    """Generate difference images for each result."""
    diff_paths = []
    for r in results:
        safe_label = r.config.label.replace(" ", "_").replace("/", "-")
        diff_path = output_dir / f"diff_{safe_label}.png"
        create_diff_image(r.output_path, reference_path, diff_path)
        diff_paths.append(diff_path)
    return diff_paths


def get_default_configs() -> List[ConversionConfig]:
    """Get default configurations to test."""
    return [
        ConversionConfig(label="default"),
        ConversionConfig(label="mask-aware", inversion_mode="mask-aware"),
        ConversionConfig(label="linear", inversion_mode="linear"),
        ConversionConfig(label="log", inversion_mode="log"),
        ConversionConfig(label="divide", inversion_mode="divide"),
    ]


def get_all_mode_configs() -> List[ConversionConfig]:
    """Get configurations for all inversion modes with various settings."""
    configs = []

    modes = ["mask-aware", "linear", "log", "divide"]

    for mode in modes:
        # Base config
        configs.append(ConversionConfig(label=mode, inversion_mode=mode))

        # With auto-wb
        configs.append(ConversionConfig(
            label=f"{mode}+wb",
            inversion_mode=mode,
            auto_wb=True
        ))

        # Without tone curve
        configs.append(ConversionConfig(
            label=f"{mode}-tc",
            inversion_mode=mode,
            no_tonecurve=True
        ))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Compare invers conversion against reference images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("negative", type=Path, help="Input negative image")
    parser.add_argument("reference", type=Path, help="Reference positive image")

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: same as negative)",
    )

    parser.add_argument(
        "--inversion-mode", "-m",
        type=str,
        choices=["mask-aware", "linear", "log", "divide", "bw"],
        help="Inversion mode to test",
    )

    parser.add_argument(
        "--base", "-b",
        type=str,
        help="Manual base RGB values as R,G,B (e.g., 0.736,0.537,0.357)",
    )

    parser.add_argument(
        "--exposure", "-e",
        type=float,
        default=1.0,
        help="Exposure compensation (default: 1.0)",
    )

    parser.add_argument(
        "--auto-wb",
        action="store_true",
        help="Enable automatic white balance",
    )

    parser.add_argument(
        "--no-tonecurve",
        action="store_true",
        help="Disable tone curve",
    )

    parser.add_argument(
        "--no-colormatrix",
        action="store_true",
        help="Disable color matrix",
    )

    parser.add_argument(
        "--test-all-modes",
        action="store_true",
        help="Test all inversion modes",
    )

    parser.add_argument(
        "--test-comprehensive",
        action="store_true",
        help="Test all modes with various settings combinations",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize images to max dimension for faster metrics (e.g., 2000)",
    )

    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Skip generating visual comparison images",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output JSON results file",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.negative.exists():
        print(f"Error: Negative file not found: {args.negative}", file=sys.stderr)
        sys.exit(1)

    if not args.reference.exists():
        print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    # Set up paths
    invers_path = find_invers_binary()
    output_dir = args.output_dir or args.negative.parent / "comparison_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Negative:  {args.negative}")
    print(f"Reference: {args.reference}")
    print(f"Output:    {output_dir}")
    print(f"invers:    {invers_path}")
    print()

    # Build configurations to test
    configs = []

    if args.test_comprehensive:
        configs = get_all_mode_configs()
    elif args.test_all_modes:
        configs = get_default_configs()
    else:
        # Single configuration from arguments
        base_rgb = None
        if args.base:
            try:
                parts = [float(x.strip()) for x in args.base.split(",")]
                if len(parts) != 3:
                    raise ValueError("Expected 3 values")
                base_rgb = tuple(parts)
            except ValueError as e:
                print(f"Error: Invalid base format: {e}", file=sys.stderr)
                sys.exit(1)

        label = args.inversion_mode or "default"
        if args.auto_wb:
            label += "+wb"
        if args.no_tonecurve:
            label += "-tc"
        if base_rgb:
            label += "+base"

        configs.append(ConversionConfig(
            label=label,
            inversion_mode=args.inversion_mode,
            base_rgb=base_rgb,
            exposure=args.exposure,
            auto_wb=args.auto_wb,
            no_tonecurve=args.no_tonecurve,
            no_colormatrix=args.no_colormatrix,
        ))

    # Run comparisons
    results = []
    total = len(configs)

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{total}] Testing: {config.label}...", end=" ", flush=True)
        try:
            result = run_comparison(
                invers_path=invers_path,
                negative_path=args.negative,
                reference_path=args.reference,
                config=config,
                output_dir=output_dir,
                resize=args.resize,
            )
            results.append(result)
            print(f"SSIM={result.metrics.ssim:.4f}, PSNR={result.metrics.psnr:.1f}dB")
        except Exception as e:
            print(f"FAILED: {e}")

    if not results:
        print("Error: No successful comparisons", file=sys.stderr)
        sys.exit(1)

    # Print results
    print_metrics_table(results)
    print_channel_analysis(results)

    # Find best result
    best = max(results, key=lambda r: r.metrics.ssim)
    print(f"\nBest result: {best.config.label} (SSIM={best.metrics.ssim:.4f})")

    # Generate visualizations
    if not args.no_visuals:
        print("\nGenerating visualizations...")

        vis_path = generate_visual_comparison(
            results, args.reference, output_dir
        )
        print(f"  Visual comparison: {vis_path}")

        diff_paths = generate_diff_images(results, args.reference, output_dir)
        print(f"  Difference images: {len(diff_paths)} files")

        # Histogram comparison for best result
        hist_path = output_dir / "histogram_comparison.png"
        create_histogram_comparison(
            best.output_path, args.reference, hist_path,
            label1=f"invers ({best.config.label})",
            label2="Reference"
        )
        print(f"  Histogram comparison: {hist_path}")

    # Save JSON results
    json_path = args.json or output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "negative": str(args.negative),
        "reference": str(args.reference),
        "results": [r.to_dict() for r in results],
        "best_config": best.config.label,
        "best_ssim": best.metrics.ssim,
    }

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
