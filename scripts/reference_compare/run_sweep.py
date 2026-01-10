#!/usr/bin/env python3
"""
Automated parameter sweep for optimizing invers settings.

This script systematically tests different parameter combinations
to find optimal settings that best match reference images.

Usage:
    python run_sweep.py negative.tif reference.tif [options]

Examples:
    # Basic sweep with default parameters
    python run_sweep.py scan.tif reference.tif

    # Quick sweep (fewer combinations)
    python run_sweep.py scan.tif reference.tif --quick

    # Focused sweep on exposure parameters
    python run_sweep.py scan.tif reference.tif --focus exposure

    # Custom parameter grid
    python run_sweep.py scan.tif reference.tif --config sweep_config.yml
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.imagemagick import compute_comparison_metrics, ImageMetrics


@dataclass
class SweepConfig:
    """Configuration for parameter sweep."""
    # Inversion modes to test
    inversion_modes: List[str]

    # Exposure values to test
    exposure_values: List[float]

    # Auto-WB options
    auto_wb_options: List[bool]

    # Tone curve options
    no_tonecurve_options: List[bool]

    # Color matrix options
    no_colormatrix_options: List[bool]

    # Auto-levels options
    no_auto_levels_options: List[bool]

    # Maximum combinations to test
    max_iterations: int

    # Target metric for optimization
    target_metric: str  # "ssim", "psnr", "rmse", "composite"

    @staticmethod
    def default() -> "SweepConfig":
        """Get default sweep configuration."""
        return SweepConfig(
            inversion_modes=["mask-aware", "linear", "log", "divide"],
            exposure_values=[0.9, 1.0, 1.1],
            auto_wb_options=[False, True],
            no_tonecurve_options=[False, True],
            no_colormatrix_options=[False],
            no_auto_levels_options=[False],
            max_iterations=100,
            target_metric="ssim",
        )

    @staticmethod
    def quick() -> "SweepConfig":
        """Get quick sweep configuration with fewer combinations."""
        return SweepConfig(
            inversion_modes=["mask-aware", "linear"],
            exposure_values=[1.0],
            auto_wb_options=[False, True],
            no_tonecurve_options=[False, True],
            no_colormatrix_options=[False],
            no_auto_levels_options=[False],
            max_iterations=20,
            target_metric="ssim",
        )

    @staticmethod
    def exposure_focused() -> "SweepConfig":
        """Get exposure-focused sweep configuration."""
        return SweepConfig(
            inversion_modes=["mask-aware"],
            exposure_values=[0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            auto_wb_options=[True],
            no_tonecurve_options=[False],
            no_colormatrix_options=[False],
            no_auto_levels_options=[False],
            max_iterations=50,
            target_metric="ssim",
        )

    @staticmethod
    def color_focused() -> "SweepConfig":
        """Get color-focused sweep configuration."""
        return SweepConfig(
            inversion_modes=["mask-aware", "linear", "divide"],
            exposure_values=[1.0],
            auto_wb_options=[False, True],
            no_tonecurve_options=[False, True],
            no_colormatrix_options=[False, True],
            no_auto_levels_options=[False],
            max_iterations=50,
            target_metric="ssim",
        )

    @staticmethod
    def from_yaml(path: Path) -> "SweepConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return SweepConfig(
            inversion_modes=data.get('inversion_modes', ["mask-aware"]),
            exposure_values=data.get('exposure_values', [1.0]),
            auto_wb_options=data.get('auto_wb_options', [False]),
            no_tonecurve_options=data.get('no_tonecurve_options', [False]),
            no_colormatrix_options=data.get('no_colormatrix_options', [False]),
            no_auto_levels_options=data.get('no_auto_levels_options', [False]),
            max_iterations=data.get('max_iterations', 100),
            target_metric=data.get('target_metric', 'ssim'),
        )

    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        return (
            len(self.inversion_modes) *
            len(self.exposure_values) *
            len(self.auto_wb_options) *
            len(self.no_tonecurve_options) *
            len(self.no_colormatrix_options) *
            len(self.no_auto_levels_options)
        )


@dataclass
class ParameterSet:
    """A single set of parameters to test."""
    inversion_mode: str
    exposure: float
    auto_wb: bool
    no_tonecurve: bool
    no_colormatrix: bool
    no_auto_levels: bool

    def to_label(self) -> str:
        """Generate a descriptive label."""
        parts = [self.inversion_mode]
        if self.exposure != 1.0:
            parts.append(f"e{self.exposure:.1f}")
        if self.auto_wb:
            parts.append("wb")
        if self.no_tonecurve:
            parts.append("notc")
        if self.no_colormatrix:
            parts.append("nocm")
        if self.no_auto_levels:
            parts.append("noal")
        return "_".join(parts)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "inversion_mode": self.inversion_mode,
            "exposure": self.exposure,
            "auto_wb": self.auto_wb,
            "no_tonecurve": self.no_tonecurve,
            "no_colormatrix": self.no_colormatrix,
            "no_auto_levels": self.no_auto_levels,
        }


@dataclass
class SweepResult:
    """Result of a single parameter combination test."""
    params: ParameterSet
    metrics: ImageMetrics
    score: float
    processing_time: float
    output_path: Optional[Path] = None


def generate_parameter_combinations(config: SweepConfig) -> Iterator[ParameterSet]:
    """Generate all parameter combinations from config."""
    for combo in itertools.product(
        config.inversion_modes,
        config.exposure_values,
        config.auto_wb_options,
        config.no_tonecurve_options,
        config.no_colormatrix_options,
        config.no_auto_levels_options,
    ):
        yield ParameterSet(
            inversion_mode=combo[0],
            exposure=combo[1],
            auto_wb=combo[2],
            no_tonecurve=combo[3],
            no_colormatrix=combo[4],
            no_auto_levels=combo[5],
        )


def find_invers_binary() -> Path:
    """Find the invers binary."""
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
        return debug_path
    else:
        result = subprocess.run(["which", "invers"], capture_output=True, text=True)
        if result.returncode == 0:
            return Path(result.stdout.strip())
        raise FileNotFoundError("Could not find invers binary")


def run_conversion(
    invers_path: Path,
    input_path: Path,
    output_path: Path,
    params: ParameterSet,
) -> float:
    """Run invers with specified parameters, return processing time."""
    cmd = [
        str(invers_path),
        "convert",
        str(input_path),
        "-o", str(output_path),
        "--inversion-mode", params.inversion_mode,
        "--exposure", str(params.exposure),
    ]

    if params.auto_wb:
        cmd.append("--auto-wb")
    if params.no_tonecurve:
        cmd.append("--no-tonecurve")
    if params.no_colormatrix:
        cmd.append("--no-colormatrix")
    if params.no_auto_levels:
        cmd.append("--no-auto-levels")

    import time
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start

    if result.returncode != 0:
        raise RuntimeError(f"Conversion failed: {result.stderr}")

    return elapsed


def compute_score(metrics: ImageMetrics, target_metric: str) -> float:
    """Compute optimization score from metrics."""
    if target_metric == "ssim":
        return metrics.ssim
    elif target_metric == "psnr":
        # Normalize PSNR to 0-1 range (assuming 0-50 dB range)
        return min(metrics.psnr / 50.0, 1.0)
    elif target_metric == "rmse":
        # Invert RMSE (lower is better)
        return 1.0 - metrics.rmse
    elif target_metric == "composite":
        # Weighted composite score
        return (
            0.5 * metrics.ssim +
            0.3 * min(metrics.psnr / 50.0, 1.0) +
            0.2 * (1.0 - metrics.rmse)
        )
    else:
        return metrics.ssim


def run_sweep(
    invers_path: Path,
    negative_path: Path,
    reference_path: Path,
    config: SweepConfig,
    output_dir: Path,
    resize: Optional[int] = None,
    keep_outputs: bool = False,
    progress_callback=None,
) -> List[SweepResult]:
    """
    Run parameter sweep.

    Args:
        invers_path: Path to invers binary
        negative_path: Input negative image
        reference_path: Reference positive image
        config: Sweep configuration
        output_dir: Directory for outputs
        resize: Optional resize dimension for faster metrics
        keep_outputs: Whether to keep converted images
        progress_callback: Optional callback(current, total, result)

    Returns:
        List of SweepResult sorted by score (best first)
    """
    results = []
    combinations = list(generate_parameter_combinations(config))

    # Limit to max_iterations
    if len(combinations) > config.max_iterations:
        # Sample evenly
        step = len(combinations) // config.max_iterations
        combinations = combinations[::step][:config.max_iterations]

    total = len(combinations)

    for i, params in enumerate(combinations):
        try:
            # Create output path
            if keep_outputs:
                output_path = output_dir / f"{negative_path.stem}_{params.to_label()}.tif"
            else:
                # Use temp file
                tmp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                output_path = Path(tmp.name)

            # Run conversion
            processing_time = run_conversion(
                invers_path, negative_path, output_path, params
            )

            # Compute metrics
            metrics = compute_comparison_metrics(output_path, reference_path, resize=resize)

            # Compute score
            score = compute_score(metrics, config.target_metric)

            result = SweepResult(
                params=params,
                metrics=metrics,
                score=score,
                processing_time=processing_time,
                output_path=output_path if keep_outputs else None,
            )
            results.append(result)

            # Clean up temp file
            if not keep_outputs:
                output_path.unlink(missing_ok=True)

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total, result)

        except Exception as e:
            print(f"Warning: Failed for {params.to_label()}: {e}", file=sys.stderr)

    # Sort by score (best first)
    results.sort(key=lambda r: -r.score)

    return results


def print_results(results: List[SweepResult], top_n: int = 10) -> None:
    """Print sweep results."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title=f"Top {min(top_n, len(results))} Parameter Combinations")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Parameters", style="green")
        table.add_column("Score", justify="right")
        table.add_column("SSIM", justify="right")
        table.add_column("PSNR", justify="right")
        table.add_column("RMSE", justify="right")

        for i, r in enumerate(results[:top_n], 1):
            # Color score
            score_str = f"{r.score:.4f}"
            if r.score > 0.95:
                score_str = f"[green]{score_str}[/green]"
            elif r.score > 0.85:
                score_str = f"[yellow]{score_str}[/yellow]"

            table.add_row(
                str(i),
                r.params.to_label(),
                score_str,
                f"{r.metrics.ssim:.4f}",
                f"{r.metrics.psnr:.1f} dB",
                f"{r.metrics.rmse:.4f}",
            )

        console.print(table)

    except ImportError:
        print(f"\nTop {min(top_n, len(results))} Parameter Combinations:")
        print("=" * 80)
        print(f"{'Rank':<5} {'Parameters':<35} {'Score':>8} {'SSIM':>8} {'PSNR':>10} {'RMSE':>8}")
        print("-" * 80)

        for i, r in enumerate(results[:top_n], 1):
            print(f"{i:<5} {r.params.to_label():<35} {r.score:>8.4f} "
                  f"{r.metrics.ssim:>8.4f} {r.metrics.psnr:>8.1f} dB {r.metrics.rmse:>8.4f}")

        print("=" * 80)


def save_results(results: List[SweepResult], output_path: Path) -> None:
    """Save sweep results to JSON."""
    data = {
        "timestamp": datetime.now().isoformat(),
        "total_tested": len(results),
        "best_score": results[0].score if results else 0,
        "best_params": results[0].params.to_dict() if results else {},
        "results": [
            {
                "rank": i + 1,
                "params": r.params.to_dict(),
                "label": r.params.to_label(),
                "score": r.score,
                "metrics": r.metrics.to_dict(),
                "processing_time": r.processing_time,
            }
            for i, r in enumerate(results)
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def generate_optimal_command(result: SweepResult, negative_path: Path) -> str:
    """Generate the optimal invers command."""
    cmd_parts = ["invers", "convert", str(negative_path)]

    cmd_parts.extend(["--inversion-mode", result.params.inversion_mode])

    if result.params.exposure != 1.0:
        cmd_parts.extend(["--exposure", str(result.params.exposure)])

    if result.params.auto_wb:
        cmd_parts.append("--auto-wb")

    if result.params.no_tonecurve:
        cmd_parts.append("--no-tonecurve")

    if result.params.no_colormatrix:
        cmd_parts.append("--no-colormatrix")

    if result.params.no_auto_levels:
        cmd_parts.append("--no-auto-levels")

    return " ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Automated parameter sweep for optimizing invers settings",
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
        "--config", "-c",
        type=Path,
        default=None,
        help="YAML configuration file for sweep parameters",
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick sweep with fewer combinations",
    )

    parser.add_argument(
        "--focus",
        type=str,
        choices=["exposure", "color", "all"],
        default="all",
        help="Focus sweep on specific aspect",
    )

    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=None,
        help="Maximum number of combinations to test",
    )

    parser.add_argument(
        "--target-metric", "-t",
        type=str,
        choices=["ssim", "psnr", "rmse", "composite"],
        default="ssim",
        help="Metric to optimize for (default: ssim)",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=2000,
        help="Resize images to max dimension for faster metrics (default: 2000)",
    )

    parser.add_argument(
        "--keep-outputs",
        action="store_true",
        help="Keep all converted images",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output results as JSON",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to display (default: 10)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.negative.exists():
        print(f"Error: Negative file not found: {args.negative}", file=sys.stderr)
        sys.exit(1)

    if not args.reference.exists():
        print(f"Error: Reference file not found: {args.reference}", file=sys.stderr)
        sys.exit(1)

    # Determine config
    if args.config:
        config = SweepConfig.from_yaml(args.config)
    elif args.quick:
        config = SweepConfig.quick()
    elif args.focus == "exposure":
        config = SweepConfig.exposure_focused()
    elif args.focus == "color":
        config = SweepConfig.color_focused()
    else:
        config = SweepConfig.default()

    # Override from command line
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    config.target_metric = args.target_metric

    # Setup
    invers_path = find_invers_binary()
    output_dir = args.output_dir or args.negative.parent / "sweep_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Negative:   {args.negative}")
    print(f"Reference:  {args.reference}")
    print(f"Output:     {output_dir}")
    print(f"invers:     {invers_path}")
    print(f"Target:     {config.target_metric}")
    print(f"Total combinations: {config.total_combinations()}")
    print(f"Max iterations: {config.max_iterations}")
    print()

    # Progress display
    try:
        from tqdm import tqdm
        pbar = tqdm(total=min(config.total_combinations(), config.max_iterations),
                   desc="Testing", unit="combo")

        def progress_callback(current, total, result):
            pbar.update(1)
            pbar.set_postfix({"best": f"{result.score:.3f}"})
    except ImportError:
        pbar = None

        def progress_callback(current, total, result):
            print(f"[{current}/{total}] {result.params.to_label()} -> score={result.score:.4f}")

    # Run sweep
    results = run_sweep(
        invers_path=invers_path,
        negative_path=args.negative,
        reference_path=args.reference,
        config=config,
        output_dir=output_dir,
        resize=args.resize,
        keep_outputs=args.keep_outputs,
        progress_callback=progress_callback,
    )

    if pbar:
        pbar.close()

    if not results:
        print("Error: No successful results", file=sys.stderr)
        sys.exit(1)

    # Print results
    print()
    print_results(results, top_n=args.top_n)

    # Best result details
    best = results[0]
    print(f"\nBest Parameters: {best.params.to_label()}")
    print(f"  Score: {best.score:.4f}")
    print(f"  SSIM:  {best.metrics.ssim:.4f}")
    print(f"  PSNR:  {best.metrics.psnr:.1f} dB")
    print(f"  RMSE:  {best.metrics.rmse:.4f}")

    print(f"\nOptimal command:")
    print(f"  {generate_optimal_command(best, args.negative)}")

    # Save results
    json_path = args.json or output_dir / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, json_path)
    print(f"\nResults saved to: {json_path}")


if __name__ == "__main__":
    main()
