#!/usr/bin/env python3
"""
Compare film base detection results across different settings.

This script runs invers with various base detection settings and creates
side-by-side comparison images for visual evaluation.

Usage:
    python compare_base_detection.py input.tif [--manual-base R,G,B] [--output-dir DIR]

Examples:
    # Basic comparison (auto vs current algorithm)
    python compare_base_detection.py city-001.tif

    # Compare with a known good manual base value
    python compare_base_detection.py city-001.tif --manual-base 0.736,0.537,0.357

    # Specify output directory
    python compare_base_detection.py city-001.tif --output-dir ./comparisons
"""

import argparse
import subprocess
import tempfile
import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ConversionResult:
    """Result of a single conversion run."""
    label: str
    output_path: Path
    base_rgb: Tuple[float, float, float]
    gb_ratio: float
    processing_time: float


def find_invers_binary() -> Path:
    """Find the invers binary, preferring release build."""
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

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
    # Look for "Base (RGB): [R, G, B]" pattern
    match = re.search(r'Base \(RGB\): \[([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)\]', output)
    if match:
        return (float(match.group(1)), float(match.group(2)), float(match.group(3)))
    return None


def parse_time_from_output(output: str) -> float:
    """Extract processing time from invers output."""
    match = re.search(r'\(([0-9.]+)s\)', output)
    if match:
        return float(match.group(1))
    return 0.0


def run_invers(
    invers_path: Path,
    input_path: Path,
    output_path: Path,
    manual_base: Optional[Tuple[float, float, float]] = None,
    extra_args: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Run invers and return (stdout, stderr)."""
    cmd = [
        str(invers_path),
        "convert",
        str(input_path),
        "--auto-wb",
        "--no-tonecurve",
        "--verbose",
        "-o", str(output_path),
    ]

    if manual_base:
        base_str = f"{manual_base[0]},{manual_base[1]},{manual_base[2]}"
        cmd.extend(["--base", base_str])

    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout + result.stderr, result.stderr


def convert_with_settings(
    invers_path: Path,
    input_path: Path,
    output_dir: Path,
    label: str,
    manual_base: Optional[Tuple[float, float, float]] = None,
) -> ConversionResult:
    """Run a single conversion and return results."""
    output_path = output_dir / f"{input_path.stem}_{label}.tif"

    output, _ = run_invers(invers_path, input_path, output_path, manual_base)

    base_rgb = parse_base_from_output(output)
    if base_rgb is None:
        if manual_base:
            base_rgb = manual_base
        else:
            base_rgb = (0.0, 0.0, 0.0)

    gb_ratio = base_rgb[1] / base_rgb[2] if base_rgb[2] > 0 else 0.0
    processing_time = parse_time_from_output(output)

    return ConversionResult(
        label=label,
        output_path=output_path,
        base_rgb=base_rgb,
        gb_ratio=gb_ratio,
        processing_time=processing_time,
    )


def create_comparison_image(
    results: List[ConversionResult],
    output_path: Path,
    crop_region: Optional[str] = None,
    thumbnail_width: int = 800,
) -> None:
    """Create a side-by-side comparison image using ImageMagick."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cropped_images = []

        for i, result in enumerate(results):
            cropped_path = tmp / f"crop_{i}.png"

            # Build ImageMagick command
            cmd = ["magick", str(result.output_path)]

            if crop_region:
                cmd.extend(["-crop", crop_region])

            # Resize and add label
            label = f"{result.label}\\nBase: [{result.base_rgb[0]:.3f}, {result.base_rgb[1]:.3f}, {result.base_rgb[2]:.3f}]\\nG/B: {result.gb_ratio:.3f}"

            cmd.extend([
                "-resize", f"{thumbnail_width}x",
                "-font", "Helvetica",
                "-pointsize", "18",
                "-gravity", "South",
                "-background", "white",
                "-splice", "0x60",
                "-gravity", "South",
                "-annotate", "+0+5", label,
                str(cropped_path),
            ])

            subprocess.run(cmd, check=True, capture_output=True)
            cropped_images.append(cropped_path)

        # Combine images horizontally
        combine_cmd = ["magick"] + [str(p) for p in cropped_images] + [
            "+append",
            str(output_path),
        ]
        subprocess.run(combine_cmd, check=True, capture_output=True)


def detect_crop_region(input_path: Path) -> str:
    """Detect a reasonable center crop region based on image dimensions."""
    result = subprocess.run(
        ["magick", "identify", "-format", "%w %h", str(input_path)],
        capture_output=True, text=True, check=True
    )
    width, height = map(int, result.stdout.strip().split())

    # Crop center 30% of image
    crop_w = int(width * 0.3)
    crop_h = int(height * 0.3)
    crop_x = (width - crop_w) // 2
    crop_y = (height - crop_h) // 2

    return f"{crop_w}x{crop_h}+{crop_x}+{crop_y}"


def main():
    parser = argparse.ArgumentParser(
        description="Compare film base detection results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Input negative TIFF file")
    parser.add_argument(
        "--manual-base", "-m",
        type=str,
        help="Manual base values as R,G,B (e.g., 0.736,0.537,0.357)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop region as WxH+X+Y (default: auto-detect center)",
    )
    parser.add_argument(
        "--thumbnail-width", "-w",
        type=int,
        default=800,
        help="Width of each thumbnail in comparison (default: 800)",
    )
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Use full image instead of cropping",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Set up paths
    invers_path = find_invers_binary()
    output_dir = args.output_dir or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse manual base if provided
    manual_base = None
    if args.manual_base:
        try:
            parts = [float(x.strip()) for x in args.manual_base.split(",")]
            if len(parts) != 3:
                raise ValueError("Expected 3 values")
            manual_base = tuple(parts)
        except ValueError as e:
            print(f"Error: Invalid manual base format: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine crop region
    crop_region = None
    if not args.full_image:
        crop_region = args.crop or detect_crop_region(args.input)
        print(f"Using crop region: {crop_region}")

    print(f"Input: {args.input}")
    print(f"Using invers: {invers_path}")
    print()

    # Run conversions
    results = []

    # Auto detection (current algorithm)
    print("Running auto base detection...")
    result = convert_with_settings(
        invers_path, args.input, output_dir, "auto"
    )
    results.append(result)
    print(f"  Base: [{result.base_rgb[0]:.4f}, {result.base_rgb[1]:.4f}, {result.base_rgb[2]:.4f}]")
    print(f"  G/B ratio: {result.gb_ratio:.3f}")
    print(f"  Time: {result.processing_time:.2f}s")
    print()

    # Manual base (if provided)
    if manual_base:
        print(f"Running with manual base {manual_base}...")
        result = convert_with_settings(
            invers_path, args.input, output_dir, "manual", manual_base
        )
        results.append(result)
        print(f"  G/B ratio: {result.gb_ratio:.3f}")
        print(f"  Time: {result.processing_time:.2f}s")
        print()

    # Create comparison image
    comparison_path = output_dir / f"{args.input.stem}_comparison.png"
    print(f"Creating comparison image: {comparison_path}")

    try:
        create_comparison_image(
            results,
            comparison_path,
            crop_region=crop_region,
            thumbnail_width=args.thumbnail_width,
        )
        print(f"Done! Comparison saved to: {comparison_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating comparison image: {e}", file=sys.stderr)
        print("Make sure ImageMagick is installed.", file=sys.stderr)
        sys.exit(1)

    # Print summary table
    print()
    print("=" * 70)
    print(f"{'Label':<12} {'R':>8} {'G':>8} {'B':>8} {'G/B':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r.label:<12} {r.base_rgb[0]:>8.4f} {r.base_rgb[1]:>8.4f} {r.base_rgb[2]:>8.4f} {r.gb_ratio:>8.3f} {r.processing_time:>7.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
