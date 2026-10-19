"""Incentive Surface & Reward Scope Visualization for FRAME-AR.

Implements:
1. MetricScopeContract: bounds, population quantiles, saturation index, perturbation sensitivity,
   and degeneracy risk ratio.
2. 2D SVG contour slice generator for publication/dashboard reporting.
3. ANSI terminal heatmap generator for live terminal observability.
4. METRIC_INCENTIVE_SURFACE telemetry streaming.
"""

from dataclasses import dataclass, field
import json
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from frame.schemas.hypothesis import TargetDirection
from frame.utils.canonical_json import canonical_json_dumps


# ==============================================================================
# 1. MetricScopeContract
# ==============================================================================

@dataclass
class MetricScopeContract:
    metric_name: str
    target_direction: TargetDirection
    theoretical_min: float
    theoretical_max: float
    population_quantiles: Dict[str, float]  # p0, p25, p50, p75, p90, p100
    saturation_index: float  # Fraction of upper values within 1% of theoretical max
    perturbation_sensitivity: Dict[str, float]  # input, parameter, label sensitivities
    degeneracy_risk_ratio: float  # (M_null_max - M_worst) / (p90 - p10)
    is_saturated: bool
    is_high_degeneracy_risk: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        metric_name: str,
        observed_scores: List[float],
        theoretical_min: float = 0.0,
        theoretical_max: float = 1.0,
        null_max_score: float = 0.5,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
        perturbation_sensitivity: Optional[Dict[str, float]] = None,
    ) -> "MetricScopeContract":
        if not observed_scores:
            raise ValueError("observed_scores must contain at least one observation")
        scores = np.asarray(observed_scores, dtype=np.float64)

        p0 = float(np.percentile(scores, 0))
        p25 = float(np.percentile(scores, 25))
        p50 = float(np.percentile(scores, 50))
        p75 = float(np.percentile(scores, 75))
        p90 = float(np.percentile(scores, 90))
        p100 = float(np.percentile(scores, 100))
        p10 = float(np.percentile(scores, 10))

        quantiles = {
            "p0": p0,
            "p10": p10,
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "p90": p90,
            "p100": p100,
        }

        # Saturation index: fraction within 1% of theoretical max (for MAXIMIZE)
        # or within 1% of theoretical min (for MINIMIZE)
        val_range = max(1e-12, theoretical_max - theoretical_min)
        if target_direction == TargetDirection.MAXIMIZE or target_direction == "maximize":
            sat_threshold = theoretical_max - 0.01 * val_range
            saturation_index = float(np.mean(scores >= sat_threshold))
        else:
            sat_threshold = theoretical_min + 0.01 * val_range
            saturation_index = float(np.mean(scores <= sat_threshold))

        is_saturated = saturation_index > 0.50

        # Degeneracy Risk Ratio (DRR): (M_null_max - M_worst) / (p90 - p10)
        p90_p10_spread = max(1e-12, p90 - p10)
        is_max = (target_direction == TargetDirection.MAXIMIZE or target_direction == "maximize")
        if is_max:
            worst_score = p0
            excess = null_max_score - worst_score
        else:
            worst_score = p100
            excess = worst_score - null_max_score

        drr = float(max(0.0, excess) / p90_p10_spread)
        is_high_degeneracy_risk = drr > 0.50

        default_sens = {
            "input_noise": 0.05,
            "param_noise": 0.08,
            "label_noise": 0.12,
        }
        sens = perturbation_sensitivity or default_sens

        return cls(
            metric_name=metric_name,
            target_direction=target_direction,
            theoretical_min=theoretical_min,
            theoretical_max=theoretical_max,
            population_quantiles=quantiles,
            saturation_index=saturation_index,
            perturbation_sensitivity=sens,
            degeneracy_risk_ratio=drr,
            is_saturated=is_saturated,
            is_high_degeneracy_risk=is_high_degeneracy_risk,
        )


# ==============================================================================
# 2. 2D SVG Contour Slice Generator
# ==============================================================================

def _colormap_viridis(val: float) -> str:
    """Simple 3-stop colormap from dark blue (0.0) -> teal (0.5) -> bright yellow (1.0)."""
    val = max(0.0, min(1.0, val))
    if val < 0.5:
        t = val * 2.0
        r = int(20 * (1 - t) + 30 * t)
        g = int(30 * (1 - t) + 160 * t)
        b = int(120 * (1 - t) + 150 * t)
    else:
        t = (val - 0.5) * 2.0
        r = int(30 * (1 - t) + 245 * t)
        g = int(160 * (1 - t) + 225 * t)
        b = int(150 * (1 - t) + 40 * t)
    return f"rgb({r},{g},{b})"


def generate_svg_contour_slice(
    x_param: str,
    y_param: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    grid_fn: Callable[[float, float], float],
    title: str = "Metric Incentive Surface",
    grid_size: int = 15,
    svg_width: int = 600,
    svg_height: int = 500,
) -> str:
    """Generates a standalone, valid SVG contour heatmap across a 2D parameter slice."""
    grid_size = max(2, int(grid_size))
    margin_left = 70
    margin_right = 90
    margin_top = 50
    margin_bottom = 60

    plot_w = svg_width - margin_left - margin_right
    plot_h = svg_height - margin_top - margin_bottom

    x_vals = np.linspace(x_range[0], x_range[1], grid_size)
    y_vals = np.linspace(y_range[0], y_range[1], grid_size)

    # Evaluate grid values
    grid_values: List[List[float]] = []
    min_v = math.inf
    max_v = -math.inf

    for j, y in enumerate(y_vals):
        row = []
        for i, x in enumerate(x_vals):
            v = float(grid_fn(float(x), float(y)))
            row.append(v)
            if v < min_v:
                min_v = v
            if v > max_v:
                max_v = v
        grid_values.append(row)

    v_range = max(1e-12, max_v - min_v)
    is_flat = (max_v - min_v) < 1e-12

    cell_w = plot_w / grid_size
    cell_h = plot_h / grid_size

    # Build SVG XML elements
    elements = []
    elements.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_width} {svg_height}" width="{svg_width}" height="{svg_height}">')
    elements.append('  <style>')
    elements.append('    .title { font-family: sans-serif; font-size: 16px; font-weight: bold; fill: #222; text-anchor: middle; }')
    elements.append('    .axis-label { font-family: sans-serif; font-size: 12px; fill: #444; }')
    elements.append('    .tick-label { font-family: sans-serif; font-size: 10px; fill: #666; }')
    elements.append('  </style>')

    # Background
    elements.append(f'  <rect width="{svg_width}" height="{svg_height}" fill="#fcfcfc" />')

    # Title
    elements.append(f'  <text x="{svg_width / 2}" y="30" class="title">{title}</text>')

    # Grid Cells (inverting y for SVG coordinate system)
    for j in range(grid_size):
        for i in range(grid_size):
            val = grid_values[j][i]
            norm_val = 0.5 if is_flat else (val - min_v) / v_range
            color = _colormap_viridis(norm_val)
            cx = margin_left + i * cell_w
            cy = margin_top + (grid_size - 1 - j) * cell_h
            elements.append(f'  <rect x="{cx:.1f}" y="{cy:.1f}" width="{cell_w + 0.5:.1f}" height="{cell_h + 0.5:.1f}" fill="{color}" stroke="none" />')

    # Border around plot
    elements.append(f'  <rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#333" stroke-width="1.5" />')

    # X-axis label and ticks
    elements.append(f'  <text x="{margin_left + plot_w / 2}" y="{svg_height - 15}" class="axis-label" text-anchor="middle">{x_param}</text>')
    elements.append(f'  <text x="{margin_left}" y="{margin_top + plot_h + 15}" class="tick-label" text-anchor="middle">{x_range[0]:.2f}</text>')
    elements.append(f'  <text x="{margin_left + plot_w}" y="{margin_top + plot_h + 15}" class="tick-label" text-anchor="middle">{x_range[1]:.2f}</text>')

    # Y-axis label and ticks
    elements.append(f'  <text x="20" y="{margin_top + plot_h / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 20 {margin_top + plot_h / 2})">{y_param}</text>')
    elements.append(f'  <text x="{margin_left - 10}" y="{margin_top + plot_h}" class="tick-label" text-anchor="end">{y_range[0]:.2f}</text>')
    elements.append(f'  <text x="{margin_left - 10}" y="{margin_top + 10}" class="tick-label" text-anchor="end">{y_range[1]:.2f}</text>')

    # Colorbar Legend on right
    bar_x = margin_left + plot_w + 20
    bar_w = 15
    bar_h = plot_h
    legend_steps = 30
    step_h = bar_h / legend_steps
    for s in range(legend_steps):
        s_val = s / (legend_steps - 1)
        c = _colormap_viridis(s_val)
        sy = margin_top + (legend_steps - 1 - s) * step_h
        elements.append(f'  <rect x="{bar_x}" y="{sy:.1f}" width="{bar_w}" height="{step_h + 0.5:.1f}" fill="{c}" />')
    elements.append(f'  <rect x="{bar_x}" y="{margin_top}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#333" stroke-width="1" />')
    elements.append(f'  <text x="{bar_x + bar_w + 5}" y="{margin_top + 10}" class="tick-label">{max_v:.2f}</text>')
    elements.append(f'  <text x="{bar_x + bar_w + 5}" y="{margin_top + bar_h}" class="tick-label">{min_v:.2f}</text>')

    elements.append('</svg>')
    return '\n'.join(elements)


# ==============================================================================
# 3. ANSI Terminal Heatmap Generator
# ==============================================================================

def generate_ansi_terminal_heatmap(
    x_param: str,
    y_param: str,
    x_values: List[float],
    y_values: List[float],
    grid_data: List[List[float]],
    title: str = "Metric Incentive Surface",
) -> str:
    """Renders a 2D matrix into colored ANSI terminal block characters."""
    lines: List[str] = []
    lines.append(f"\033[1m=== {title} ===\033[0m")
    lines.append(f"Y-axis: {y_param} | X-axis: {x_param}\n")

    if not x_values or not y_values or not grid_data:
        lines.append("[Empty surface data]")
        return "\n".join(lines)

    flat = [v for row in grid_data for v in row]
    min_v = min(flat) if flat else 0.0
    max_v = max(flat) if flat else 1.0
    span = max(1e-12, max_v - min_v)

    # Invert rows so highest Y is at top
    for j in reversed(range(len(y_values))):
        y_val = y_values[j]
        row_str = f"{y_val:6.2f} | "
        for i in range(len(x_values)):
            val = grid_data[j][i]
            norm = 0.5 if (max_v - min_v) < 1e-12 else (val - min_v) / span
            # Color calculation: 0 = blue, 0.5 = green, 1.0 = red
            if norm < 0.5:
                r = int(50 * norm * 2)
                g = int(200 * norm * 2)
                b = int(220 * (1 - norm * 2))
            else:
                t = (norm - 0.5) * 2
                r = int(50 * (1 - t) + 240 * t)
                g = int(200 * (1 - t) + 100 * t)
                b = int(50 * (1 - t))
            row_str += f"\033[48;2;{r};{g};{b}m  \033[0m"
        lines.append(row_str)

    # Bottom axis line
    x_axis_prefix = "       +-" + "--" * len(x_values)
    lines.append(x_axis_prefix)
    if len(x_values) > 1:
        x_first = f"{x_values[0]:.2f}"
        x_last = f"{x_values[-1]:.2f}"
        total_width = len(x_values) * 2
        pad = max(1, total_width - len(x_first) - len(x_last))
        x_ticks = "         " + x_first + " " * pad + x_last
    else:
        x_ticks = f"         {x_values[0]:.2f}"
    lines.append(x_ticks)
    lines.append(f"Min: {min_v:.3f} | Max: {max_v:.3f}")
    return "\n".join(lines)


# ==============================================================================
# 4. Telemetry Streamer
# ==============================================================================

def stream_metric_incentive_telemetry(
    contract: MetricScopeContract,
    slice_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats a METRIC_INCENTIVE_SURFACE JSON telemetry event."""
    event = {
        "event_type": "METRIC_INCENTIVE_SURFACE",
        "metric_name": contract.metric_name,
        "target_direction": contract.target_direction.value if hasattr(contract.target_direction, "value") else str(contract.target_direction),
        "theoretical_min": contract.theoretical_min,
        "theoretical_max": contract.theoretical_max,
        "population_quantiles": contract.population_quantiles,
        "saturation_index": contract.saturation_index,
        "is_saturated": contract.is_saturated,
        "perturbation_sensitivity": contract.perturbation_sensitivity,
        "degeneracy_risk_ratio": contract.degeneracy_risk_ratio,
        "is_high_degeneracy_risk": contract.is_high_degeneracy_risk,
        "slice_data": slice_data or {},
    }
    return canonical_json_dumps(event)
