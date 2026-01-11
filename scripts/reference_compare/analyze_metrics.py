#!/usr/bin/env python3
"""
Analyze metrics from comparison results and generate reports.

This script aggregates metrics from multiple comparison JSON files
and generates a comprehensive analysis report with insights.

Usage:
    python analyze_metrics.py comparison_*.json --report analysis.md

Examples:
    # Analyze all comparison results in a directory
    python analyze_metrics.py results/*.json

    # Generate markdown report
    python analyze_metrics.py results/*.json --report report.md

    # Show only summary statistics
    python analyze_metrics.py results/*.json --summary
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple comparisons."""
    config_label: str
    count: int
    ssim_mean: float
    ssim_std: float
    ssim_min: float
    ssim_max: float
    psnr_mean: float
    psnr_std: float
    rmse_mean: float
    rmse_std: float
    mae_mean: float
    channel_bias_mean: Dict[str, float]
    luminance_diff_mean: float
    contrast_diff_mean: float


def load_comparison_results(json_paths: List[Path]) -> List[Dict]:
    """Load comparison results from JSON files."""
    all_results = []

    for path in json_paths:
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            # Handle both single result and batch result formats
            if 'results' in data:
                for result in data['results']:
                    result['source_file'] = str(path)
                    result['negative'] = data.get('negative', 'unknown')
                    result['reference'] = data.get('reference', 'unknown')
                    all_results.append(result)
            else:
                data['source_file'] = str(path)
                all_results.append(data)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {path}: {e}", file=sys.stderr)

    return all_results


def aggregate_by_config(results: List[Dict]) -> Dict[str, AggregatedMetrics]:
    """Aggregate metrics by configuration label."""
    grouped = defaultdict(list)

    for r in results:
        label = r.get('config', {}).get('label', 'unknown')
        metrics = r.get('metrics', {})
        grouped[label].append(metrics)

    aggregated = {}
    for label, metrics_list in grouped.items():
        if not metrics_list:
            continue

        ssim_vals = [m.get('ssim', 0) for m in metrics_list]
        psnr_vals = [m.get('psnr', 0) for m in metrics_list]
        rmse_vals = [m.get('rmse', 0) for m in metrics_list]
        mae_vals = [m.get('mae', 0) for m in metrics_list]

        # Channel biases
        channel_biases = defaultdict(list)
        for m in metrics_list:
            for ch, bias in m.get('channel_bias', {}).items():
                channel_biases[ch].append(bias)

        channel_bias_mean = {
            ch: np.mean(vals) for ch, vals in channel_biases.items()
        }

        lum_diffs = [m.get('mean_luminance_diff', 0) for m in metrics_list]
        contrast_diffs = [m.get('contrast_diff', 0) for m in metrics_list]

        aggregated[label] = AggregatedMetrics(
            config_label=label,
            count=len(metrics_list),
            ssim_mean=np.mean(ssim_vals),
            ssim_std=np.std(ssim_vals),
            ssim_min=np.min(ssim_vals),
            ssim_max=np.max(ssim_vals),
            psnr_mean=np.mean(psnr_vals),
            psnr_std=np.std(psnr_vals),
            rmse_mean=np.mean(rmse_vals),
            rmse_std=np.std(rmse_vals),
            mae_mean=np.mean(mae_vals),
            channel_bias_mean=channel_bias_mean,
            luminance_diff_mean=np.mean(lum_diffs),
            contrast_diff_mean=np.mean(contrast_diffs),
        )

    return aggregated


def rank_configurations(aggregated: Dict[str, AggregatedMetrics]) -> List[Tuple[str, float]]:
    """Rank configurations by weighted score."""
    rankings = []

    for label, agg in aggregated.items():
        # Weighted score (higher is better)
        # SSIM (0-1): weight 0.5
        # PSNR (typically 20-50): weight 0.3, normalized
        # RMSE (0-1, lower is better): weight 0.2
        score = (
            0.5 * agg.ssim_mean +
            0.3 * min(agg.psnr_mean / 50.0, 1.0) +
            0.2 * (1.0 - agg.rmse_mean)
        )
        rankings.append((label, score))

    return sorted(rankings, key=lambda x: x[1], reverse=True)


def identify_issues(aggregated: Dict[str, AggregatedMetrics]) -> List[Dict]:
    """Identify potential issues from the metrics."""
    issues = []

    for label, agg in aggregated.items():
        # Low SSIM indicates structural differences
        if agg.ssim_mean < 0.85:
            issues.append({
                "config": label,
                "severity": "high" if agg.ssim_mean < 0.7 else "medium",
                "type": "structural",
                "description": f"Low structural similarity (SSIM={agg.ssim_mean:.3f}). "
                              f"Images may have significant tonal or color differences.",
            })

        # Large channel bias indicates color cast
        for ch, bias in agg.channel_bias_mean.items():
            if abs(bias) > 0.05:
                issues.append({
                    "config": label,
                    "severity": "medium" if abs(bias) < 0.1 else "high",
                    "type": "color_cast",
                    "channel": ch,
                    "description": f"{ch.capitalize()} channel bias of {bias:+.3f}. "
                                  f"Consider adjusting color matrix or white balance.",
                })

        # Large luminance difference indicates exposure issues
        if abs(agg.luminance_diff_mean) > 0.05:
            direction = "brighter" if agg.luminance_diff_mean > 0 else "darker"
            issues.append({
                "config": label,
                "severity": "medium",
                "type": "exposure",
                "description": f"Output is {direction} than reference (diff={agg.luminance_diff_mean:+.3f}). "
                              f"Consider adjusting exposure compensation.",
            })

        # Contrast difference
        if abs(agg.contrast_diff_mean) > 0.03:
            direction = "higher" if agg.contrast_diff_mean > 0 else "lower"
            issues.append({
                "config": label,
                "severity": "low",
                "type": "contrast",
                "description": f"Contrast is {direction} than reference (diff={agg.contrast_diff_mean:+.3f}). "
                              f"Consider adjusting tone curve or shadow/highlight settings.",
            })

        # High variance indicates inconsistent results
        if agg.ssim_std > 0.05 and agg.count > 1:
            issues.append({
                "config": label,
                "severity": "low",
                "type": "inconsistent",
                "description": f"High variance in results (SSIM std={agg.ssim_std:.3f}). "
                              f"Results may be sensitive to input characteristics.",
            })

    return issues


def generate_markdown_report(
    results: List[Dict],
    aggregated: Dict[str, AggregatedMetrics],
    rankings: List[Tuple[str, float]],
    issues: List[Dict],
    output_path: Path,
) -> None:
    """Generate a comprehensive markdown report."""
    lines = []

    # Header
    lines.append("# Reference Comparison Analysis Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\nTotal comparisons: {len(results)}")
    lines.append(f"Configurations tested: {len(aggregated)}")

    # Executive Summary
    lines.append("\n## Executive Summary")
    if rankings:
        best_label, best_score = rankings[0]
        best = aggregated[best_label]
        lines.append(f"\n**Best Configuration:** `{best_label}`")
        lines.append(f"- Mean SSIM: {best.ssim_mean:.4f}")
        lines.append(f"- Mean PSNR: {best.psnr_mean:.1f} dB")
        lines.append(f"- Mean RMSE: {best.rmse_mean:.4f}")

    # Overall assessment
    if rankings:
        best_ssim = aggregated[rankings[0][0]].ssim_mean
        if best_ssim > 0.95:
            lines.append("\n**Assessment:** Excellent match to reference. "
                        "invers is producing results comparable to reference software.")
        elif best_ssim > 0.85:
            lines.append("\n**Assessment:** Good match with some differences. "
                        "Review the identified issues below for improvement suggestions.")
        else:
            lines.append("\n**Assessment:** Significant differences detected. "
                        "Review issues and consider parameter adjustments or manual base values.")

    # Rankings Table
    lines.append("\n## Configuration Rankings")
    lines.append("\n| Rank | Configuration | Score | SSIM | PSNR | RMSE |")
    lines.append("|------|--------------|-------|------|------|------|")

    for i, (label, score) in enumerate(rankings, 1):
        agg = aggregated[label]
        lines.append(f"| {i} | `{label}` | {score:.3f} | "
                    f"{agg.ssim_mean:.4f} | {agg.psnr_mean:.1f} dB | {agg.rmse_mean:.4f} |")

    # Detailed Metrics
    lines.append("\n## Detailed Metrics by Configuration")

    for label, agg in sorted(aggregated.items(), key=lambda x: -x[1].ssim_mean):
        lines.append(f"\n### {label}")
        lines.append(f"\nSamples: {agg.count}")

        lines.append("\n**Quality Metrics:**")
        lines.append(f"- SSIM: {agg.ssim_mean:.4f} (std: {agg.ssim_std:.4f}, "
                    f"range: {agg.ssim_min:.4f} - {agg.ssim_max:.4f})")
        lines.append(f"- PSNR: {agg.psnr_mean:.1f} dB (std: {agg.psnr_std:.1f})")
        lines.append(f"- RMSE: {agg.rmse_mean:.4f} (std: {agg.rmse_std:.4f})")
        lines.append(f"- MAE: {agg.mae_mean:.4f}")

        lines.append("\n**Channel Biases:**")
        for ch in ['red', 'green', 'blue']:
            bias = agg.channel_bias_mean.get(ch, 0)
            indicator = "+" if bias > 0.01 else ("-" if bias < -0.01 else "~")
            lines.append(f"- {ch.capitalize()}: {bias:+.4f} [{indicator}]")

        lines.append("\n**Tonal Characteristics:**")
        lines.append(f"- Luminance diff: {agg.luminance_diff_mean:+.4f}")
        lines.append(f"- Contrast diff: {agg.contrast_diff_mean:+.4f}")

    # Issues and Recommendations
    if issues:
        lines.append("\n## Identified Issues")

        # Group by severity
        for severity in ['high', 'medium', 'low']:
            severity_issues = [i for i in issues if i['severity'] == severity]
            if severity_issues:
                emoji = {"high": "!", "medium": "?", "low": "i"}[severity]
                lines.append(f"\n### {severity.capitalize()} Priority")

                for issue in severity_issues:
                    lines.append(f"\n**[{emoji}] {issue['config']}** - {issue['type']}")
                    lines.append(f"\n{issue['description']}")

    # Recommendations
    lines.append("\n## Recommendations")

    if issues:
        # Generate specific recommendations based on issues
        has_color_cast = any(i['type'] == 'color_cast' for i in issues)
        has_exposure = any(i['type'] == 'exposure' for i in issues)
        has_contrast = any(i['type'] == 'contrast' for i in issues)

        if has_color_cast:
            lines.append("\n### Color Correction")
            lines.append("- Try enabling `--auto-wb` if not already enabled")
            lines.append("- Consider adjusting the color matrix in the film preset")
            lines.append("- Manual base values may help if auto-detection is inaccurate")

        if has_exposure:
            lines.append("\n### Exposure")
            lines.append("- Adjust `--exposure` parameter (< 1.0 for darker, > 1.0 for brighter)")
            lines.append("- Check auto-exposure settings if enabled")
            lines.append("- Verify base estimation is sampling appropriate film border area")

        if has_contrast:
            lines.append("\n### Contrast")
            lines.append("- Try `--no-tonecurve` to disable tone curve modifications")
            lines.append("- Adjust shadow_lift_value in config for shadow behavior")
            lines.append("- Modify highlight_compression for highlight handling")

    else:
        lines.append("\nNo significant issues identified. Results are well-matched to reference.")

    # Footer
    lines.append("\n---")
    lines.append(f"\n*Report generated by invers reference comparison framework*")

    # Write report
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def print_summary(
    aggregated: Dict[str, AggregatedMetrics],
    rankings: List[Tuple[str, float]],
) -> None:
    """Print summary to console."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Configuration Rankings")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Configuration", style="green")
        table.add_column("Score", justify="right")
        table.add_column("SSIM", justify="right")
        table.add_column("PSNR", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("Samples", justify="right")

        for i, (label, score) in enumerate(rankings, 1):
            agg = aggregated[label]

            # Color SSIM
            ssim_str = f"{agg.ssim_mean:.4f}"
            if agg.ssim_mean > 0.95:
                ssim_str = f"[green]{ssim_str}[/green]"
            elif agg.ssim_mean > 0.85:
                ssim_str = f"[yellow]{ssim_str}[/yellow]"
            else:
                ssim_str = f"[red]{ssim_str}[/red]"

            table.add_row(
                str(i),
                label,
                f"{score:.3f}",
                ssim_str,
                f"{agg.psnr_mean:.1f} dB",
                f"{agg.rmse_mean:.4f}",
                str(agg.count),
            )

        console.print(table)

    except ImportError:
        # Fallback
        print("\nConfiguration Rankings:")
        print("=" * 70)
        print(f"{'Rank':<5} {'Configuration':<20} {'Score':>8} {'SSIM':>8} {'PSNR':>10} {'RMSE':>8}")
        print("-" * 70)

        for i, (label, score) in enumerate(rankings, 1):
            agg = aggregated[label]
            print(f"{i:<5} {label:<20} {score:>8.3f} {agg.ssim_mean:>8.4f} "
                  f"{agg.psnr_mean:>8.1f} dB {agg.rmse_mean:>8.4f}")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze metrics from comparison results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "json_files",
        nargs="+",
        type=str,
        help="JSON result files (supports glob patterns)",
    )

    parser.add_argument(
        "--report", "-r",
        type=Path,
        default=None,
        help="Output markdown report path",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show only summary statistics",
    )

    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Output aggregated results as JSON",
    )

    args = parser.parse_args()

    # Expand glob patterns
    json_paths = []
    for pattern in args.json_files:
        if '*' in pattern or '?' in pattern:
            json_paths.extend(Path(p) for p in glob.glob(pattern))
        else:
            json_paths.append(Path(pattern))

    if not json_paths:
        print("Error: No JSON files found", file=sys.stderr)
        sys.exit(1)

    # Load results
    print(f"Loading {len(json_paths)} result file(s)...")
    results = load_comparison_results(json_paths)

    if not results:
        print("Error: No valid results found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} comparison result(s)")

    # Aggregate
    aggregated = aggregate_by_config(results)
    rankings = rank_configurations(aggregated)
    issues = identify_issues(aggregated)

    # Print summary
    print_summary(aggregated, rankings)

    if not args.summary and issues:
        print(f"\nIdentified {len(issues)} potential issue(s)")
        for issue in issues[:5]:  # Show first 5
            severity_icon = {"high": "[!]", "medium": "[?]", "low": "[i]"}[issue['severity']]
            print(f"  {severity_icon} {issue['config']}: {issue['type']}")

    # Generate report
    if args.report:
        generate_markdown_report(results, aggregated, rankings, issues, args.report)
        print(f"\nReport saved to: {args.report}")

    # JSON output
    if args.json_out:
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "total_comparisons": len(results),
            "rankings": [{"label": l, "score": s} for l, s in rankings],
            "aggregated": {
                label: {
                    "count": agg.count,
                    "ssim_mean": agg.ssim_mean,
                    "ssim_std": agg.ssim_std,
                    "psnr_mean": agg.psnr_mean,
                    "rmse_mean": agg.rmse_mean,
                    "channel_bias_mean": agg.channel_bias_mean,
                }
                for label, agg in aggregated.items()
            },
            "issues": issues,
        }

        with open(args.json_out, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON output saved to: {args.json_out}")


if __name__ == "__main__":
    main()
