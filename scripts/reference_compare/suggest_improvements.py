#!/usr/bin/env python3
"""
Suggest pipeline improvements based on comparison metrics.

This script analyzes comparison results and generates specific
parameter recommendations to improve color, exposure, and tonal
matching with reference images.

Usage:
    python suggest_improvements.py results_dir/ [options]

Examples:
    # Analyze results and suggest improvements
    python suggest_improvements.py ./comparison_results/

    # Focus on color improvements
    python suggest_improvements.py ./comparison_results/ --focus color

    # Generate suggested config file
    python suggest_improvements.py ./comparison_results/ --output-config suggested.yml
"""

import argparse
import glob
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml


@dataclass
class ParameterSuggestion:
    """A suggested parameter change."""
    parameter: str
    current_value: Optional[str]
    suggested_value: str
    confidence: float  # 0-1
    reason: str
    impact: str  # "color", "exposure", "contrast", "structure"


@dataclass
class ImprovementPlan:
    """Complete improvement plan with prioritized suggestions."""
    suggestions: List[ParameterSuggestion] = field(default_factory=list)
    expected_ssim_gain: float = 0.0
    primary_issues: List[str] = field(default_factory=list)
    secondary_issues: List[str] = field(default_factory=list)


def load_results(results_dir: Path) -> List[Dict]:
    """Load all comparison results from directory."""
    results = []

    for json_path in results_dir.glob("*.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            if 'results' in data:
                for r in data['results']:
                    r['source'] = str(json_path)
                results.extend(data['results'])
            elif 'metrics' in data:
                data['source'] = str(json_path)
                results.append(data)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping {json_path}: {e}", file=sys.stderr)

    return results


def analyze_color_issues(results: List[Dict]) -> Dict:
    """Analyze color-related issues from results."""
    analysis = {
        "has_issues": False,
        "channel_biases": {"red": [], "green": [], "blue": []},
        "saturation_diffs": [],
        "hue_shifts": [],
        "dominant_bias": None,
        "color_cast_direction": None,
    }

    for r in results:
        metrics = r.get('metrics', {})

        for ch in ['red', 'green', 'blue']:
            bias = metrics.get('channel_bias', {}).get(ch, 0)
            analysis['channel_biases'][ch].append(bias)

        analysis['saturation_diffs'].append(metrics.get('saturation_diff', 0))
        analysis['hue_shifts'].append(metrics.get('hue_shift', 0))

    # Compute averages
    for ch in ['red', 'green', 'blue']:
        vals = analysis['channel_biases'][ch]
        if vals:
            analysis['channel_biases'][ch] = np.mean(vals)
        else:
            analysis['channel_biases'][ch] = 0

    analysis['saturation_diffs'] = np.mean(analysis['saturation_diffs']) if analysis['saturation_diffs'] else 0
    analysis['hue_shifts'] = np.mean(analysis['hue_shifts']) if analysis['hue_shifts'] else 0

    # Determine if there are issues
    biases = analysis['channel_biases']
    max_bias = max(abs(biases['red']), abs(biases['green']), abs(biases['blue']))

    if max_bias > 0.02:
        analysis['has_issues'] = True

        # Find dominant bias
        if abs(biases['red']) >= abs(biases['green']) and abs(biases['red']) >= abs(biases['blue']):
            analysis['dominant_bias'] = 'red'
            analysis['color_cast_direction'] = 'cyan' if biases['red'] > 0 else 'red'
        elif abs(biases['green']) >= abs(biases['blue']):
            analysis['dominant_bias'] = 'green'
            analysis['color_cast_direction'] = 'magenta' if biases['green'] > 0 else 'green'
        else:
            analysis['dominant_bias'] = 'blue'
            analysis['color_cast_direction'] = 'yellow' if biases['blue'] > 0 else 'blue'

    return analysis


def analyze_exposure_issues(results: List[Dict]) -> Dict:
    """Analyze exposure-related issues from results."""
    analysis = {
        "has_issues": False,
        "luminance_diffs": [],
        "avg_luminance_diff": 0,
        "direction": None,
        "suggested_compensation": 1.0,
    }

    for r in results:
        metrics = r.get('metrics', {})
        lum_diff = metrics.get('mean_luminance_diff', 0)
        analysis['luminance_diffs'].append(lum_diff)

    if analysis['luminance_diffs']:
        avg = np.mean(analysis['luminance_diffs'])
        analysis['avg_luminance_diff'] = avg

        if abs(avg) > 0.03:
            analysis['has_issues'] = True
            analysis['direction'] = 'bright' if avg > 0 else 'dark'

            # Calculate suggested compensation
            # Positive diff = output brighter than reference = need less exposure
            # Negative diff = output darker than reference = need more exposure
            # Scale: ~0.1 diff -> ~0.2 exposure adjustment
            adjustment = -avg * 2.0
            analysis['suggested_compensation'] = max(0.5, min(2.0, 1.0 + adjustment))

    return analysis


def analyze_contrast_issues(results: List[Dict]) -> Dict:
    """Analyze contrast-related issues from results."""
    analysis = {
        "has_issues": False,
        "contrast_diffs": [],
        "avg_contrast_diff": 0,
        "direction": None,
        "shadow_issue": False,
        "highlight_issue": False,
    }

    for r in results:
        metrics = r.get('metrics', {})
        contrast_diff = metrics.get('contrast_diff', 0)
        analysis['contrast_diffs'].append(contrast_diff)

    if analysis['contrast_diffs']:
        avg = np.mean(analysis['contrast_diffs'])
        analysis['avg_contrast_diff'] = avg

        if abs(avg) > 0.02:
            analysis['has_issues'] = True
            analysis['direction'] = 'high' if avg > 0 else 'low'

    return analysis


def analyze_structural_issues(results: List[Dict]) -> Dict:
    """Analyze structural similarity issues."""
    analysis = {
        "has_issues": False,
        "ssim_values": [],
        "avg_ssim": 0,
        "min_ssim": 1.0,
        "histogram_correlations": [],
        "avg_hist_corr": 0,
    }

    for r in results:
        metrics = r.get('metrics', {})
        analysis['ssim_values'].append(metrics.get('ssim', 0))
        analysis['histogram_correlations'].append(metrics.get('histogram_correlation', 0))

    if analysis['ssim_values']:
        analysis['avg_ssim'] = np.mean(analysis['ssim_values'])
        analysis['min_ssim'] = np.min(analysis['ssim_values'])

        if analysis['avg_ssim'] < 0.90:
            analysis['has_issues'] = True

    if analysis['histogram_correlations']:
        analysis['avg_hist_corr'] = np.mean(analysis['histogram_correlations'])

    return analysis


def get_best_config(results: List[Dict]) -> Optional[Dict]:
    """Get the best performing configuration from results."""
    if not results:
        return None

    best = max(results, key=lambda r: r.get('metrics', {}).get('ssim', 0))
    return best.get('config', {})


def generate_suggestions(
    color_analysis: Dict,
    exposure_analysis: Dict,
    contrast_analysis: Dict,
    structural_analysis: Dict,
    best_config: Optional[Dict],
) -> ImprovementPlan:
    """Generate improvement suggestions based on analysis."""
    plan = ImprovementPlan()

    # Color suggestions
    if color_analysis['has_issues']:
        plan.primary_issues.append(f"Color cast toward {color_analysis['color_cast_direction']}")

        biases = color_analysis['channel_biases']

        # Suggest auto-wb if not already enabled
        if best_config and not best_config.get('auto_wb', False):
            plan.suggestions.append(ParameterSuggestion(
                parameter="auto_wb",
                current_value="false",
                suggested_value="true",
                confidence=0.8,
                reason=f"Channel biases detected (R:{biases['red']:+.3f}, G:{biases['green']:+.3f}, B:{biases['blue']:+.3f})",
                impact="color",
            ))

        # Suggest color matrix adjustments
        if abs(biases['red']) > 0.05 or abs(biases['green']) > 0.05 or abs(biases['blue']) > 0.05:
            # Calculate corrective matrix hint
            correction = f"R:{1.0 - biases['red']:.2f}, G:{1.0 - biases['green']:.2f}, B:{1.0 - biases['blue']:.2f}"
            plan.suggestions.append(ParameterSuggestion(
                parameter="color_matrix_adjustment",
                current_value="identity",
                suggested_value=correction,
                confidence=0.6,
                reason="Significant channel biases suggest color matrix needs adjustment",
                impact="color",
            ))

        # Saturation adjustment
        sat_diff = color_analysis['saturation_diffs']
        if abs(sat_diff) > 0.05:
            direction = "increase" if sat_diff < 0 else "decrease"
            plan.suggestions.append(ParameterSuggestion(
                parameter="saturation",
                current_value="1.0",
                suggested_value=f"{1.0 - sat_diff:.2f}",
                confidence=0.5,
                reason=f"Output saturation differs by {sat_diff:+.3f}",
                impact="color",
            ))

    # Exposure suggestions
    if exposure_analysis['has_issues']:
        direction = exposure_analysis['direction']
        plan.primary_issues.append(f"Exposure too {direction}")

        plan.suggestions.append(ParameterSuggestion(
            parameter="exposure_compensation",
            current_value="1.0",
            suggested_value=f"{exposure_analysis['suggested_compensation']:.2f}",
            confidence=0.85,
            reason=f"Luminance difference of {exposure_analysis['avg_luminance_diff']:+.3f}",
            impact="exposure",
        ))

        # If output is too bright, might need to adjust auto-exposure
        if direction == 'bright':
            plan.suggestions.append(ParameterSuggestion(
                parameter="auto_exposure_target_median",
                current_value="0.63",
                suggested_value="0.55",
                confidence=0.5,
                reason="Output consistently brighter than reference",
                impact="exposure",
            ))

    # Contrast suggestions
    if contrast_analysis['has_issues']:
        direction = contrast_analysis['direction']
        plan.secondary_issues.append(f"Contrast too {direction}")

        if direction == 'high':
            plan.suggestions.append(ParameterSuggestion(
                parameter="tone_curve",
                current_value="enabled",
                suggested_value="disabled (--no-tonecurve)",
                confidence=0.6,
                reason="Output has higher contrast than reference",
                impact="contrast",
            ))

            plan.suggestions.append(ParameterSuggestion(
                parameter="shadow_lift_value",
                current_value="0.02",
                suggested_value="0.03",
                confidence=0.5,
                reason="Lifting shadows may reduce perceived contrast",
                impact="contrast",
            ))
        else:
            plan.suggestions.append(ParameterSuggestion(
                parameter="highlight_compression",
                current_value="1.0",
                suggested_value="0.9",
                confidence=0.5,
                reason="Output has lower contrast than reference",
                impact="contrast",
            ))

    # Structural suggestions
    if structural_analysis['has_issues']:
        if structural_analysis['avg_ssim'] < 0.80:
            plan.primary_issues.append("Significant structural differences")

            # Try different inversion mode
            current_mode = best_config.get('inversion_mode', 'mask-aware') if best_config else 'mask-aware'
            suggested_modes = ['mask-aware', 'linear', 'log', 'divide']
            suggested_modes = [m for m in suggested_modes if m != current_mode]

            plan.suggestions.append(ParameterSuggestion(
                parameter="inversion_mode",
                current_value=current_mode,
                suggested_value=f"try: {', '.join(suggested_modes[:2])}",
                confidence=0.7,
                reason=f"Low SSIM ({structural_analysis['avg_ssim']:.3f}) suggests inversion mode may not be optimal",
                impact="structure",
            ))

            # Suggest manual base if auto isn't working
            plan.suggestions.append(ParameterSuggestion(
                parameter="base_rgb",
                current_value="auto-detected",
                suggested_value="manual (use analyze command to determine)",
                confidence=0.6,
                reason="Auto base detection may not be optimal for this negative",
                impact="structure",
            ))

    # Calculate expected improvement
    if plan.suggestions:
        # Rough estimate based on number and confidence of suggestions
        total_confidence = sum(s.confidence for s in plan.suggestions)
        num_suggestions = len(plan.suggestions)
        current_ssim = structural_analysis['avg_ssim']

        # Estimate potential SSIM gain
        max_possible_gain = 1.0 - current_ssim
        estimated_gain = max_possible_gain * min(0.5, total_confidence / (num_suggestions * 2))
        plan.expected_ssim_gain = estimated_gain

    return plan


def print_improvement_plan(plan: ImprovementPlan) -> None:
    """Print the improvement plan to console."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # Issues panel
        if plan.primary_issues or plan.secondary_issues:
            issues_text = ""
            if plan.primary_issues:
                issues_text += "[bold red]Primary Issues:[/bold red]\n"
                for issue in plan.primary_issues:
                    issues_text += f"  - {issue}\n"
            if plan.secondary_issues:
                issues_text += "[yellow]Secondary Issues:[/yellow]\n"
                for issue in plan.secondary_issues:
                    issues_text += f"  - {issue}\n"

            console.print(Panel(issues_text.strip(), title="Identified Issues"))

        # Suggestions table
        if plan.suggestions:
            table = Table(title="Suggested Parameter Changes")
            table.add_column("Parameter", style="cyan")
            table.add_column("Current", style="dim")
            table.add_column("Suggested", style="green")
            table.add_column("Confidence")
            table.add_column("Impact")

            for s in sorted(plan.suggestions, key=lambda x: -x.confidence):
                conf_bar = "[green]" + "=" * int(s.confidence * 10) + "[/green]" + \
                          "[dim]" + "-" * (10 - int(s.confidence * 10)) + "[/dim]"

                table.add_row(
                    s.parameter,
                    s.current_value or "default",
                    s.suggested_value,
                    conf_bar,
                    s.impact,
                )

            console.print(table)

            if plan.expected_ssim_gain > 0:
                console.print(f"\n[bold]Expected SSIM improvement:[/bold] +{plan.expected_ssim_gain:.3f}")

    except ImportError:
        # Fallback
        print("\n" + "=" * 70)
        print("IDENTIFIED ISSUES")
        print("-" * 70)

        if plan.primary_issues:
            print("Primary:")
            for issue in plan.primary_issues:
                print(f"  [!] {issue}")

        if plan.secondary_issues:
            print("Secondary:")
            for issue in plan.secondary_issues:
                print(f"  [?] {issue}")

        print("\n" + "=" * 70)
        print("SUGGESTED PARAMETER CHANGES")
        print("-" * 70)
        print(f"{'Parameter':<25} {'Current':<15} {'Suggested':<20} {'Conf':>6}")
        print("-" * 70)

        for s in sorted(plan.suggestions, key=lambda x: -x.confidence):
            print(f"{s.parameter:<25} {(s.current_value or 'default'):<15} "
                  f"{s.suggested_value:<20} {s.confidence:>5.0%}")
            print(f"  Reason: {s.reason}")

        print("=" * 70)

        if plan.expected_ssim_gain > 0:
            print(f"\nExpected SSIM improvement: +{plan.expected_ssim_gain:.3f}")


def generate_config_file(plan: ImprovementPlan, output_path: Path) -> None:
    """Generate a suggested configuration file."""
    config = {
        "# Suggested configuration based on reference comparison analysis": None,
        "# Generated by suggest_improvements.py": None,
        "defaults": {}
    }

    for s in plan.suggestions:
        if s.parameter == "exposure_compensation":
            try:
                config["defaults"]["exposure_compensation"] = float(s.suggested_value)
            except ValueError:
                pass

        elif s.parameter == "auto_wb":
            config["defaults"]["enable_auto_wb"] = s.suggested_value == "true"

        elif s.parameter == "tone_curve":
            config["defaults"]["skip_tone_curve"] = "disabled" in s.suggested_value

        elif s.parameter == "shadow_lift_value":
            try:
                config["defaults"]["shadow_lift_value"] = float(s.suggested_value)
            except ValueError:
                pass

        elif s.parameter == "highlight_compression":
            try:
                config["defaults"]["highlight_compression"] = float(s.suggested_value)
            except ValueError:
                pass

        elif s.parameter == "auto_exposure_target_median":
            try:
                config["defaults"]["auto_exposure_target_median"] = float(s.suggested_value)
            except ValueError:
                pass

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Suggest pipeline improvements based on comparison metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing comparison JSON results",
    )

    parser.add_argument(
        "--focus",
        type=str,
        choices=["color", "exposure", "contrast", "all"],
        default="all",
        help="Focus analysis on specific aspect (default: all)",
    )

    parser.add_argument(
        "--output-config", "-c",
        type=Path,
        default=None,
        help="Output suggested configuration as YAML file",
    )

    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output suggestions as JSON",
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed analysis",
    )

    args = parser.parse_args()

    # Validate input
    if not args.results_dir.exists():
        print(f"Error: Results directory not found: {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    # Load results
    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        print("Error: No valid results found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} comparison result(s)")

    # Run analysis
    print("\nAnalyzing metrics...")

    color_analysis = analyze_color_issues(results)
    exposure_analysis = analyze_exposure_issues(results)
    contrast_analysis = analyze_contrast_issues(results)
    structural_analysis = analyze_structural_issues(results)
    best_config = get_best_config(results)

    if args.verbose:
        print(f"\nColor Analysis:")
        print(f"  Has issues: {color_analysis['has_issues']}")
        print(f"  Channel biases: R={color_analysis['channel_biases']['red']:+.3f}, "
              f"G={color_analysis['channel_biases']['green']:+.3f}, "
              f"B={color_analysis['channel_biases']['blue']:+.3f}")

        print(f"\nExposure Analysis:")
        print(f"  Has issues: {exposure_analysis['has_issues']}")
        print(f"  Avg luminance diff: {exposure_analysis['avg_luminance_diff']:+.3f}")

        print(f"\nContrast Analysis:")
        print(f"  Has issues: {contrast_analysis['has_issues']}")
        print(f"  Avg contrast diff: {contrast_analysis['avg_contrast_diff']:+.3f}")

        print(f"\nStructural Analysis:")
        print(f"  Has issues: {structural_analysis['has_issues']}")
        print(f"  Avg SSIM: {structural_analysis['avg_ssim']:.4f}")

    # Generate suggestions
    plan = generate_suggestions(
        color_analysis,
        exposure_analysis,
        contrast_analysis,
        structural_analysis,
        best_config,
    )

    # Print results
    print_improvement_plan(plan)

    # Generate config file
    if args.output_config:
        generate_config_file(plan, args.output_config)
        print(f"\nSuggested config saved to: {args.output_config}")

    # JSON output
    if args.json:
        json_data = {
            "analysis": {
                "color": color_analysis,
                "exposure": exposure_analysis,
                "contrast": contrast_analysis,
                "structural": structural_analysis,
            },
            "suggestions": [
                {
                    "parameter": s.parameter,
                    "current_value": s.current_value,
                    "suggested_value": s.suggested_value,
                    "confidence": s.confidence,
                    "reason": s.reason,
                    "impact": s.impact,
                }
                for s in plan.suggestions
            ],
            "expected_ssim_gain": plan.expected_ssim_gain,
            "primary_issues": plan.primary_issues,
            "secondary_issues": plan.secondary_issues,
        }

        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON output saved to: {args.json}")


if __name__ == "__main__":
    main()
