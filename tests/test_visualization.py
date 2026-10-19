"""Unit Tests for Incentive Surface & Reward Scope Visualization (visualization.py)."""

import json
import math
import pytest

from frame.feedback.visualization import (
    MetricScopeContract,
    generate_svg_contour_slice,
    generate_ansi_terminal_heatmap,
    stream_metric_incentive_telemetry,
)
from frame.schemas.hypothesis import TargetDirection


# ==============================================================================
# 1. MetricScopeContract Tests
# ==============================================================================

def test_metric_scope_contract_normal_and_saturated():
    """Verifies bounds, quantiles, and saturation detection."""
    # 1. Normal well-distributed accuracy metric [0.50 to 0.85]
    normal_scores = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    contract_normal = MetricScopeContract.create(
        metric_name="accuracy",
        observed_scores=normal_scores,
        theoretical_min=0.0,
        theoretical_max=1.0,
        null_max_score=0.20,
        target_direction=TargetDirection.MAXIMIZE,
    )

    assert contract_normal.is_saturated is False
    assert contract_normal.saturation_index == 0.0
    assert contract_normal.population_quantiles["p50"] == 0.675
    assert contract_normal.degeneracy_risk_ratio < 0.50
    assert contract_normal.is_high_degeneracy_risk is False

    # 2. Saturated metric: majority of scores piled up at 0.995 (within 1% of 1.0)
    saturated_scores = [0.995] * 8 + [0.80, 0.85]
    contract_sat = MetricScopeContract.create(
        metric_name="saturated_acc",
        observed_scores=saturated_scores,
        theoretical_min=0.0,
        theoretical_max=1.0,
        null_max_score=0.10,
        target_direction=TargetDirection.MAXIMIZE,
    )
    assert contract_sat.is_saturated is True
    assert contract_sat.saturation_index >= 0.80


def test_metric_scope_degeneracy_risk_trigger():
    """Verifies that high degeneracy risk ratio flags reward hacking vulnerability."""
    # If best null model (e.g. 0.80) is near or above the true population spread
    scores = [0.70, 0.72, 0.73, 0.74, 0.75]
    contract_risky = MetricScopeContract.create(
        metric_name="fragile_metric",
        observed_scores=scores,
        theoretical_min=0.0,
        theoretical_max=1.0,
        null_max_score=0.76,  # Null model easily matches or beats candidate scores
        target_direction=TargetDirection.MAXIMIZE,
    )
    assert contract_risky.is_high_degeneracy_risk is True
    assert contract_risky.degeneracy_risk_ratio > 0.50


# ==============================================================================
# 2. 2D SVG Contour Slice Generator Tests
# ==============================================================================

def test_generate_svg_contour_slice():
    """Verifies that SVG contour slice outputs well-formed SVG XML with tags and labels."""
    def test_surface(lr, hidden_dim):
        return math.sin(lr) * math.cos(hidden_dim / 100.0)

    svg_content = generate_svg_contour_slice(
        x_param="Learning Rate",
        y_param="Hidden Dim",
        x_range=(0.0001, 0.01),
        y_range=(32.0, 512.0),
        grid_fn=test_surface,
        title="Accuracy Incentive Surface",
        grid_size=10,
    )

    assert isinstance(svg_content, str)
    assert svg_content.startswith("<svg")
    assert svg_content.endswith("</svg>")
    assert "Learning Rate" in svg_content
    assert "Hidden Dim" in svg_content
    assert "Accuracy Incentive Surface" in svg_content
    assert "<rect" in svg_content
    assert "rgb(" in svg_content


# ==============================================================================
# 3. ANSI Terminal Heatmap Tests
# ==============================================================================

def test_generate_ansi_terminal_heatmap():
    """Verifies that ANSI terminal heatmap renders formatted rows with color escapes."""
    x_vals = [0.1, 0.2, 0.3]
    y_vals = [10.0, 20.0, 30.0]
    grid_data = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]

    heatmap = generate_ansi_terminal_heatmap(
        x_param="LR",
        y_param="BatchSize",
        x_values=x_vals,
        y_values=y_vals,
        grid_data=grid_data,
        title="Terminal Surface Slice",
    )

    assert isinstance(heatmap, str)
    assert "Terminal Surface Slice" in heatmap
    assert "Y-axis: BatchSize | X-axis: LR" in heatmap
    assert "\033[" in heatmap  # Contains ANSI color escapes


# ==============================================================================
# 4. Telemetry Streamer Tests
# ==============================================================================

def test_stream_metric_incentive_telemetry():
    """Verifies JSON telemetry streaming structure and valid serialization."""
    contract = MetricScopeContract.create(
        metric_name="cross_entropy",
        observed_scores=[1.5, 1.2, 1.0, 0.8],
        theoretical_min=0.0,
        theoretical_max=5.0,
        null_max_score=2.0,
        target_direction=TargetDirection.MINIMIZE,
    )

    payload = stream_metric_incentive_telemetry(contract, slice_data={"resolution": "10x10"})
    parsed = json.loads(payload)

    assert parsed["event_type"] == "METRIC_INCENTIVE_SURFACE"
    assert parsed["metric_name"] == "cross_entropy"
    assert parsed["target_direction"] == "minimize"
    assert "population_quantiles" in parsed
    assert parsed["slice_data"]["resolution"] == "10x10"


def test_metric_scope_contract_empty_scores_raises():
    """Verifies that empty observed_scores raises ValueError."""
    with pytest.raises(ValueError, match="observed_scores must contain at least one observation"):
        MetricScopeContract.create(
            metric_name="empty_metric",
            observed_scores=[],
        )


def test_generate_svg_contour_slice_flat_surface():
    """Verifies that flat constant surfaces render valid SVG without NaNs."""
    svg = generate_svg_contour_slice(
        x_param="Param A",
        y_param="Param B",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        grid_fn=lambda x, y: 5.0,
        title="Flat Surface",
        grid_size=4,
    )
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "nan" not in svg.lower()


def test_generate_ansi_terminal_heatmap_small_x():
    """Verifies that heatmaps with 2 x values format ticks cleanly without smashed strings."""
    heatmap = generate_ansi_terminal_heatmap(
        x_param="X",
        y_param="Y",
        x_values=[0.1, 0.9],
        y_values=[1.0, 2.0],
        grid_data=[[0.5, 0.5], [0.5, 0.5]],
        title="Two-column Heatmap",
    )
    assert "Two-column Heatmap" in heatmap
    assert "0.10" in heatmap
    assert "0.90" in heatmap
