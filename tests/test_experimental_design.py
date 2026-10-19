"""Unit Tests for Explicit Experimental Design Stage (experimental_design.py)."""

import math
import pytest
import torch
import torch.nn as nn

from frame.brain.experimental_design import (
    calculate_required_sample_size,
    TwoStageScreeningController,
    QuadBaselineGenerator,
    verify_method_code_alignment,
)
from frame.schemas.hypothesis import TargetDirection


# ==============================================================================
# 1. Dynamic Power Analysis Tests
# ==============================================================================

def test_dynamic_power_analysis_sample_sizing():
    """Verifies that non-central Student's t calculation matches calibrated benchmarks."""
    # Under alpha_1 = 0.010937, mu_D = 0.02, rope_eps = 0.01:
    alpha = 0.010937
    mu_d = 0.02
    rope_eps = 0.01

    # At sigma_D = 0.005 -> k* = 6, Power ≈ 89.2%
    res_005 = calculate_required_sample_size(
        alpha=alpha,
        effect_size_delta=mu_d,
        rope_epsilon=rope_eps,
        sigma_d=0.005,
    )
    assert res_005.required_k == 6
    assert abs(res_005.achieved_power - 0.892) < 0.01

    # At sigma_D = 0.002 -> k* = 3, Power ≈ 80.7%
    res_002 = calculate_required_sample_size(
        alpha=alpha,
        effect_size_delta=mu_d,
        rope_epsilon=rope_eps,
        sigma_d=0.002,
    )
    assert res_002.required_k == 3
    assert abs(res_002.achieved_power - 0.807) < 0.01

    # At sigma_D = 0.010 -> k* = 13
    res_010 = calculate_required_sample_size(
        alpha=alpha,
        effect_size_delta=mu_d,
        rope_epsilon=rope_eps,
        sigma_d=0.010,
    )
    assert res_010.required_k == 13
    assert res_010.achieved_power >= 0.80

    # At sigma_D = 0.020 -> k* = 42
    res_020 = calculate_required_sample_size(
        alpha=alpha,
        effect_size_delta=mu_d,
        rope_epsilon=rope_eps,
        sigma_d=0.020,
    )
    assert res_020.required_k == 42
    assert res_020.achieved_power >= 0.80


# ==============================================================================
# 2. Two-Stage Group Sequential Screening Tests
# ==============================================================================

def test_two_stage_futility_screening():
    """Verifies that Stage 1 futility filter aborts negative branches and passes positive ones."""
    controller = TwoStageScreeningController(rope_epsilon=0.01)

    # 1. Futile branch: candidate scores are lower or equal to baseline
    base_scores = {"seed_1": 0.85, "seed_2": 0.84, "seed_3": 0.86}
    cand_futile = {"seed_1": 0.83, "seed_2": 0.84, "seed_3": 0.85}
    decision_futile = controller.evaluate_stage1_futility(base_scores, cand_futile)

    assert decision_futile.proceed is False
    assert decision_futile.status == "FUTILITY_PRUNED"
    assert decision_futile.mean_difference < 0.0

    # 2. Promising branch: candidate shows positive directional gain
    cand_promising = {"seed_1": 0.88, "seed_2": 0.87, "seed_3": 0.89}
    decision_promising = controller.evaluate_stage1_futility(base_scores, cand_promising)

    assert decision_promising.proceed is True
    assert decision_promising.status == "ADVANCE_TO_STAGE_2"
    assert decision_promising.mean_difference > 0.0


# ==============================================================================
# 3. Quad-Baseline Control Protocol Tests
# ==============================================================================

def test_quad_baseline_generation_and_evaluation():
    """Verifies generation of 4 controls and dominance assertion."""
    base_kwargs = {"hidden_dim": 64, "layers": 2}
    configs = QuadBaselineGenerator.generate_configs(base_kwargs, candidate_flops_ratio=1.44, candidate_params_ratio=1.44)

    assert "C_FLOP" in configs
    assert "C_param" in configs
    assert "C_ablate" in configs
    assert "C_neg" in configs
    assert configs["C_FLOP"].model_kwargs["hidden_dim"] > 64

    # 1. Dominant candidate
    control_scores = {
        "C_FLOP": 0.82,
        "C_param": 0.81,
        "C_ablate": 0.79,
        "C_neg": 0.75,
    }
    cand_score = 0.85
    res_dominant = QuadBaselineGenerator.evaluate_quad_baselines(
        cand_score, control_scores, min_margin=0.01
    )
    assert res_dominant.passed_all_controls is True
    assert res_dominant.verdict == "PASSED_QUAD_BASELINES"

    # 2. Subordinate candidate (loses to matched-FLOP baseline)
    cand_score_weak = 0.815
    res_weak = QuadBaselineGenerator.evaluate_quad_baselines(
        cand_score_weak, control_scores, min_margin=0.01
    )
    assert res_weak.passed_all_controls is False
    assert res_weak.verdict == "FAILED_CONTROL_DOMINANCE"


# ==============================================================================
# 4. Method-to-Implementation Alignment Verification Tests
# ==============================================================================

def test_method_code_alignment_verification():
    """Verifies AST diff and active gradient flow audit."""
    base_code = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
"""

    cand_code_valid = """
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.gate = nn.Linear(5, 5)
    def forward(self, x):
        h = self.fc(x)
        return h * torch.sigmoid(self.gate(h))
"""

    # Model matching valid candidate code
    class ValidCandModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            self.gate = nn.Linear(5, 5)
        def forward(self, x):
            h = self.fc(x)
            return h * torch.sigmoid(self.gate(h))

    valid_model = ValidCandModel()
    sample_x = torch.randn(4, 10)
    loss_f = lambda out: out.sum()

    res_valid = verify_method_code_alignment(
        baseline_code=base_code,
        candidate_code=cand_code_valid,
        candidate_model=valid_model,
        new_param_names=["gate"],
        sample_input=sample_x,
        loss_fn=loss_f,
    )
    assert res_valid.passed is True
    assert res_valid.verdict == "ALIGNED_IMPLEMENTATION"
    assert res_valid.active_gradient_flow is True
    assert res_valid.min_gradient_norm > 1e-8

    # Dead parameter candidate (added parameter disconnected from graph)
    class DeadParamModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 5)
            self.gate = nn.Linear(5, 5)  # never used in forward pass
        def forward(self, x):
            return self.fc(x)

    dead_model = DeadParamModel()
    res_dead = verify_method_code_alignment(
        baseline_code=base_code,
        candidate_code=cand_code_valid,
        candidate_model=dead_model,
        new_param_names=["gate"],
        sample_input=sample_x,
        loss_fn=loss_f,
    )
    assert res_dead.passed is False
    assert res_dead.verdict == "METHOD_CODE_MISALIGNMENT"
    assert res_dead.active_gradient_flow is False


def test_two_stage_stage2_confirmatory_expansion():
    """Verifies that Stage 2 confirmatory expansion evaluates paired difference ROPE against LORD threshold."""
    controller = TwoStageScreeningController(rope_epsilon=0.01)

    # 1. Confirmed candidate across 6 seeds with large paired gain
    base_scores = {f"seed_{i}": 0.80 for i in range(1, 7)}
    cand_scores = {f"seed_{i}": 0.85 for i in range(1, 7)}

    decision_confirmed = controller.evaluate_stage2_confirmatory(
        baseline_scores=base_scores,
        candidate_scores=cand_scores,
        alpha_override=0.010937,
    )
    assert decision_confirmed.proceed is True
    assert decision_confirmed.status == "CONFIRMED_CANDIDATE"
    assert decision_confirmed.sample_size_evaluated == 6
    assert decision_confirmed.mean_difference == pytest.approx(0.05, abs=1e-5)
    assert decision_confirmed.details["p_value_rope"] < 0.001
    assert "allocated_alpha_t" in decision_confirmed.details
    assert decision_confirmed.details["allocated_alpha_t"] > 0
    assert decision_confirmed.details["allocated_alpha_t"] == pytest.approx(0.010937)
    assert decision_confirmed.details["alpha_t"] == decision_confirmed.details["allocated_alpha_t"]

    # 2. Confirmatory testing with LORDController
    from frame.verification.statistics import LORDController
    lord = LORDController(alpha=0.05)
    decision_lord = controller.evaluate_stage2_confirmatory(
        baseline_scores=base_scores,
        candidate_scores=cand_scores,
        lord_controller=lord,
    )
    assert decision_lord.proceed is True
    assert "allocated_alpha_t" in decision_lord.details
    assert decision_lord.details["allocated_alpha_t"] > 0
    assert decision_lord.details["allocated_alpha_t"] == pytest.approx(0.0109372, abs=1e-5)

    # 3. Rejected candidate: gains only 0.005, which does not exceed rope_epsilon=0.01
    cand_weak = {f"seed_{i}": 0.805 for i in range(1, 7)}
    decision_rejected = controller.evaluate_stage2_confirmatory(
        baseline_scores=base_scores,
        candidate_scores=cand_weak,
        alpha_override=0.010937,
    )
    assert decision_rejected.proceed is False
    assert decision_rejected.status == "REJECTED_NULL"
    assert decision_rejected.details["allocated_alpha_t"] > 0


def test_power_analysis_minimization():
    """Verifies sample size calculation for minimization metrics."""
    res_min = calculate_required_sample_size(
        alpha=0.010937,
        target_power=0.80,
        effect_size_delta=0.02,
        rope_epsilon=0.01,
        sigma_d=0.005,
        target_direction=TargetDirection.MINIMIZE,
    )
    assert res_min.required_k == 6
    assert res_min.achieved_power >= 0.80


def test_method_alignment_rejects_syntax_error():
    """Verifies that candidate code with syntax errors is rejected by alignment verification."""
    base_code = "class Net(nn.Module):\n    pass"
    cand_broken = "class Net(nn.Module):\n    def forward(syntax error"

    res = verify_method_code_alignment(
        baseline_code=base_code,
        candidate_code=cand_broken,
    )
    assert res.passed is False
    assert res.verdict == "METHOD_CODE_MISALIGNMENT"
    assert res.ast_diff_detected is False
