"""Explicit Experimental Design Stage for FRAME-AR.

Implements:
1. Dynamic non-central Student's t power analysis for k* sample sizing under LORD alpha_t.
2. Two-stage group sequential screening (Stage 1 k=3 futility screening, Stage 2 confirmatory expansion).
3. Quad-baseline control protocol (C_FLOP, C_param, C_ablate, C_neg).
4. Method-to-implementation alignment verification (structural AST diff + active gradient flow check).
"""

import ast
from dataclasses import dataclass, field
import difflib
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
import torch
import torch.nn as nn

from frame.schemas.hypothesis import TargetDirection


# ==============================================================================
# 1. Dynamic Power Analysis & Sample Sizing
# ==============================================================================

@dataclass
class PowerAnalysisResult:
    required_k: int
    achieved_power: float
    target_power: float
    alpha: float
    effect_size_delta: float
    rope_epsilon: float
    sigma_d: float
    non_centrality_param: float
    critical_t: float


def calculate_required_sample_size(
    alpha: float = 0.010937,
    target_power: float = 0.80,
    effect_size_delta: float = 0.02,
    rope_epsilon: float = 0.01,
    sigma_d: float = 0.005,
    max_k: int = 120,
    target_direction: TargetDirection = TargetDirection.MAXIMIZE,
) -> PowerAnalysisResult:
    """Calculates required sample size k* using non-central Student's t-distribution.
    
    Tests H0: mu_D <= rope_epsilon vs H1: mu_D > rope_epsilon, where true mean difference
    is effect_size_delta.
    """
    is_min = (target_direction == TargetDirection.MINIMIZE or target_direction == "minimize")
    directional_delta = -effect_size_delta if is_min and effect_size_delta < 0 else effect_size_delta
    net_effect = directional_delta - rope_epsilon
    if net_effect <= 0:
        raise ValueError(
            f"Directional effect size delta ({effect_size_delta}) must exceed rope_epsilon ({rope_epsilon})"
        )

    best_k = max_k
    best_power = 0.0
    best_ncp = 0.0
    best_tcrit = 0.0

    for k in range(3, max_k + 1):
        df = k - 1
        ncp = (net_effect * math.sqrt(k)) / sigma_d
        tcrit = float(stats.t.ppf(1.0 - alpha, df))
        # Power is the survival function (1 - CDF) of non-central t at critical t
        power = float(stats.nct.sf(tcrit, df, ncp))

        if power >= target_power:
            best_k = k
            best_power = power
            best_ncp = ncp
            best_tcrit = tcrit
            break
        else:
            best_k = k
            best_power = power
            best_ncp = ncp
            best_tcrit = tcrit

    return PowerAnalysisResult(
        required_k=best_k,
        achieved_power=best_power,
        target_power=target_power,
        alpha=alpha,
        effect_size_delta=effect_size_delta,
        rope_epsilon=rope_epsilon,
        sigma_d=sigma_d,
        non_centrality_param=best_ncp,
        critical_t=best_tcrit,
    )


# ==============================================================================
# 2. Two-Stage Group Sequential Screening
# ==============================================================================

class SequentialStage(str):
    STAGE_1_FUTILITY = "stage_1_futility"
    STAGE_2_CONFIRMATORY = "stage_2_confirmatory"


@dataclass
class ScreeningDecision:
    stage: str
    proceed: bool
    status: str  # FUTILITY_PRUNED | ADVANCE_TO_STAGE_2 | CONFIRMED_CANDIDATE | REJECTED_NULL
    sample_size_evaluated: int
    mean_difference: float
    details: Dict[str, Any] = field(default_factory=dict)


class TwoStageScreeningController:
    """Manages two-stage group sequential screening:
    Stage 1: k=3 futility filter. Aborts immediately if mean difference <= 0.
    Stage 2: Expansion to k* seeds for confirmatory testing under LORD alpha_t.
    """

    def __init__(self, rope_epsilon: float = 0.01, target_direction: TargetDirection = TargetDirection.MAXIMIZE):
        self.rope_epsilon = rope_epsilon
        self.target_direction = target_direction

    def evaluate_stage1_futility(
        self,
        baseline_scores: Dict[str, float],
        candidate_scores: Dict[str, float],
    ) -> ScreeningDecision:
        """Stage 1: Checks futility at k=3 paired seeds.
        
        Aborts if directional difference mean(D) <= 0.
        """
        common_seeds = sorted(set(baseline_scores.keys()).intersection(candidate_scores.keys()))
        if len(common_seeds) < 3:
            raise ValueError(f"Stage 1 requires at least 3 paired seeds, got {len(common_seeds)}")

        diffs = []
        is_max = (self.target_direction == TargetDirection.MAXIMIZE or self.target_direction == "maximize")
        for s in common_seeds[:3]:
            b = baseline_scores[s]
            c = candidate_scores[s]
            d = (c - b) if is_max else (b - c)
            diffs.append(d)

        mean_diff = float(np.mean(diffs))
        if mean_diff <= 0.0:
            return ScreeningDecision(
                stage=SequentialStage.STAGE_1_FUTILITY,
                proceed=False,
                status="FUTILITY_PRUNED",
                sample_size_evaluated=3,
                mean_difference=mean_diff,
                details={
                    "reason": "Candidate difference <= 0 in initial 3 seeds. Pruning early.",
                    "allocated_alpha_t": 0.0,
                    "alpha_t": 0.0,
                },
            )

        return ScreeningDecision(
            stage=SequentialStage.STAGE_1_FUTILITY,
            proceed=True,
            status="ADVANCE_TO_STAGE_2",
            sample_size_evaluated=3,
            mean_difference=mean_diff,
            details={
                "reason": "Candidate passed Stage 1 futility filter with positive mean difference.",
                "allocated_alpha_t": 0.0,
                "alpha_t": 0.0,
            },
        )

    def evaluate_stage2_confirmatory(
        self,
        baseline_scores: Dict[str, float],
        candidate_scores: Dict[str, float],
        lord_controller: Optional[Any] = None,
        alpha_override: Optional[float] = None,
        rope_epsilon_override: Optional[float] = None,
        edge_id: Optional[str] = None,
        source_hypothesis_id: Optional[str] = None,
        target_hypothesis_id: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> ScreeningDecision:
        """Stage 2: Evaluates confirmatory expansion across paired seeds against LORD threshold.
        
        Tests H0: mu_D <= rope_epsilon vs H1: mu_D > rope_epsilon.
        Supports per-edge pre-registered rope_epsilon via rope_epsilon_override.
        """
        common_seeds = sorted(set(baseline_scores.keys()).intersection(candidate_scores.keys()))
        k = len(common_seeds)
        if k < 3:
            raise ValueError(f"Stage 2 confirmatory screening requires at least 3 paired seeds, got {k}")

        is_max = (self.target_direction == TargetDirection.MAXIMIZE or self.target_direction == "maximize")
        diffs = []
        for s in common_seeds:
            b = baseline_scores[s]
            c = candidate_scores[s]
            d = (c - b) if is_max else (b - c)
            diffs.append(d)

        diffs_arr = np.asarray(diffs, dtype=np.float64)
        mean_diff = float(np.mean(diffs_arr))
        s_d = float(np.std(diffs_arr, ddof=1)) if k > 1 else 0.0
        se = s_d / math.sqrt(k) if s_d > 1e-15 else 1e-15

        effective_rope = float(rope_epsilon_override) if rope_epsilon_override is not None else float(self.rope_epsilon)
        t_rope = (mean_diff - effective_rope) / se
        p_rope = float(stats.t.sf(t_rope, df=k - 1))

        if alpha_override is not None:
            alpha_t = max(float(alpha_override), 1e-15)
            is_confirmed = (p_rope <= alpha_t)
        elif lord_controller is not None:
            is_confirmed, alpha_t_val = lord_controller.record_result(p_rope)
            alpha_t = max(float(alpha_t_val), 1e-15)
        else:
            alpha_t = 0.010937  # Default LORD step 1 alpha
            is_confirmed = (p_rope <= alpha_t)

        status = "CONFIRMED_CANDIDATE" if is_confirmed else "REJECTED_NULL"
        return ScreeningDecision(
            stage=SequentialStage.STAGE_2_CONFIRMATORY,
            proceed=is_confirmed,
            status=status,
            sample_size_evaluated=k,
            mean_difference=mean_diff,
            details={
                "p_value_rope": p_rope,
                "t_statistic": t_rope,
                "allocated_alpha_t": alpha_t,
                "alpha_t": alpha_t,
                "rope_epsilon": effective_rope,
                "sample_std": s_d,
                "edge_id": edge_id,
                "source_hypothesis_id": source_hypothesis_id,
                "target_hypothesis_id": target_hypothesis_id,
                "edge_type": edge_type or "atomic_transition",
            },
        )



# ==============================================================================
# 3. Quad-Baseline Control Protocol
# ==============================================================================

@dataclass
class BaselineConfig:
    name: str
    control_type: str  # C_FLOP | C_PARAM | C_ABLATE | C_NEG
    description: str
    model_kwargs: Dict[str, Any]


@dataclass
class QuadBaselineResult:
    candidate_score: float
    baseline_scores: Dict[str, float]  # { "C_FLOP": ..., "C_PARAM": ..., ... }
    margins: Dict[str, float]
    passed_all_controls: bool
    verdict: str  # PASSED_QUAD_BASELINES | FAILED_CONTROL_DOMINANCE
    details: Dict[str, Any] = field(default_factory=dict)


class QuadBaselineGenerator:
    """Generates and evaluates 4 controls:
    1. C_FLOP: Matched-FLOP baseline
    2. C_param: Matched-parameter baseline
    3. C_ablate: Core mutation ablation
    4. C_neg: Inverted/scrambled control
    """

    @staticmethod
    def generate_configs(
        base_kwargs: Dict[str, Any],
        candidate_flops_ratio: float = 1.25,
        candidate_params_ratio: float = 1.20,
    ) -> Dict[str, BaselineConfig]:
        """Synthesizes control configurations given scaling ratios."""
        h_dim = base_kwargs.get("hidden_dim", 128)

        # C_FLOP: scale width to match FLOPs (FLOPs ~ width^2)
        flop_h_dim = int(round(h_dim * math.sqrt(candidate_flops_ratio)))
        c_flop_kwargs = dict(base_kwargs, hidden_dim=flop_h_dim)

        # C_param: scale width to match params
        param_h_dim = int(round(h_dim * math.sqrt(candidate_params_ratio)))
        c_param_kwargs = dict(base_kwargs, hidden_dim=param_h_dim)

        # C_ablate: revert mutation flag
        c_ablate_kwargs = dict(base_kwargs, use_mutation=False)

        # C_neg: invert or scramble mechanism
        c_neg_kwargs = dict(base_kwargs, scramble_mutation=True)

        return {
            "C_FLOP": BaselineConfig(
                name="C_FLOP",
                control_type="C_FLOP",
                description="Matched-FLOP baseline widening hidden dimension",
                model_kwargs=c_flop_kwargs,
            ),
            "C_param": BaselineConfig(
                name="C_param",
                control_type="C_PARAM",
                description="Matched-parameter baseline widening hidden dimension",
                model_kwargs=c_param_kwargs,
            ),
            "C_ablate": BaselineConfig(
                name="C_ablate",
                control_type="C_ABLATE",
                description="Ablation baseline removing proposed mutation",
                model_kwargs=c_ablate_kwargs,
            ),
            "C_neg": BaselineConfig(
                name="C_neg",
                control_type="C_NEG",
                description="Negative control with scrambled or inverted mutation",
                model_kwargs=c_neg_kwargs,
            ),
        }

    @staticmethod
    def evaluate_quad_baselines(
        candidate_score: float,
        control_scores: Dict[str, float],
        min_margin: float = 0.005,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
    ) -> QuadBaselineResult:
        """Asserts that candidate outperforms all 4 baselines by at least min_margin."""
        is_max = (target_direction == TargetDirection.MAXIMIZE or target_direction == "maximize")
        margins = {}
        all_passed = True

        for name, score in control_scores.items():
            margin = (candidate_score - score) if is_max else (score - candidate_score)
            margins[name] = float(margin)
            if margin < min_margin:
                all_passed = False

        verdict = "PASSED_QUAD_BASELINES" if all_passed else "FAILED_CONTROL_DOMINANCE"
        return QuadBaselineResult(
            candidate_score=candidate_score,
            baseline_scores=control_scores,
            margins=margins,
            passed_all_controls=all_passed,
            verdict=verdict,
            details={"min_margin": min_margin},
        )


# ==============================================================================
# 4. Method-to-Implementation Alignment Verification
# ==============================================================================

@dataclass
class AlignmentVerificationResult:
    passed: bool
    alignment_score: float
    ast_diff_detected: bool
    active_gradient_flow: bool
    min_gradient_norm: float
    verdict: str  # ALIGNED_IMPLEMENTATION | METHOD_CODE_MISALIGNMENT
    details: Dict[str, Any] = field(default_factory=dict)


def verify_method_code_alignment(
    baseline_code: str,
    candidate_code: str,
    candidate_model: Optional[nn.Module] = None,
    new_param_names: Optional[List[str]] = None,
    sample_input: Optional[torch.Tensor] = None,
    loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    min_alignment_score: float = 0.85,
    min_grad_norm_thresh: float = 1e-8,
) -> AlignmentVerificationResult:
    """Verifies that the candidate implementation substantively matches the proposed method.
    
    1. Structural AST Diff check: confirms genuine AST change (not just comments or whitespace).
    2. Active Gradient Flow check: verifies that newly introduced parameters have requires_grad=True,
       receive non-zero gradients (||grad|| > 1e-8), and directly impact the loss.
    """
    # 1. Structural AST Diff
    ast_error = None
    try:
        base_ast = ast.parse(baseline_code)
        cand_ast = ast.parse(candidate_code)
        base_dump = ast.dump(base_ast, annotate_fields=False)
        cand_dump = ast.dump(cand_ast, annotate_fields=False)
        ast_diff_detected = (base_dump != cand_dump)
        similarity = difflib.SequenceMatcher(None, base_dump, cand_dump).ratio()
        # AST modification score: 1.0 if changed meaningfully
        ast_score = 1.0 if (ast_diff_detected and similarity < 0.999) else 0.0
    except SyntaxError as e:
        ast_diff_detected = False
        ast_score = 0.0
        ast_error = f"SyntaxError in code: {e}"
    except Exception as e:
        ast_diff_detected = (baseline_code.strip() != candidate_code.strip())
        ast_score = 0.0 if not ast_diff_detected else 0.2
        ast_error = str(e)

    # 2. Active Gradient Flow Audit
    active_gradient_flow = True
    min_grad_norm = math.inf
    grad_details: Dict[str, Any] = {}

    if candidate_model is not None and sample_input is not None and loss_fn is not None:
        candidate_model.train()
        candidate_model.zero_grad()

        if isinstance(sample_input, (tuple, list)):
            output = candidate_model(*sample_input)
        else:
            output = candidate_model(sample_input)
        loss = loss_fn(output)
        loss.backward()

        params_to_check = []
        if new_param_names:
            for name, param in candidate_model.named_parameters():
                if any(np_name in name for np_name in new_param_names):
                    params_to_check.append((name, param))
        else:
            params_to_check = list(candidate_model.named_parameters())

        if not params_to_check:
            active_gradient_flow = False
            min_grad_norm = 0.0
        else:
            for name, param in params_to_check:
                if not param.requires_grad:
                    active_gradient_flow = False
                    grad_details[name] = "requires_grad is False"
                    min_grad_norm = 0.0
                    break

                if param.grad is None:
                    active_gradient_flow = False
                    grad_details[name] = "grad is None"
                    min_grad_norm = 0.0
                    break

                gnorm = float(param.grad.norm().item())
                if gnorm < min_grad_norm:
                    min_grad_norm = gnorm

                if gnorm < min_grad_norm_thresh:
                    active_gradient_flow = False
                    grad_details[name] = f"gradient norm {gnorm:.2e} < {min_grad_norm_thresh:.2e}"
                    break
    elif candidate_model is not None:
        # If no input/loss provided, simply verify parameter flags
        for name, param in candidate_model.named_parameters():
            if not param.requires_grad:
                active_gradient_flow = False
                break
        min_grad_norm = 1.0 if active_gradient_flow else 0.0

    if min_grad_norm == math.inf:
        min_grad_norm = 1.0

    # Combined alignment score
    grad_score = 1.0 if active_gradient_flow else 0.0
    alignment_score = 0.4 * ast_score + 0.6 * grad_score

    passed = (alignment_score >= min_alignment_score) and ast_diff_detected and active_gradient_flow
    verdict = "ALIGNED_IMPLEMENTATION" if passed else "METHOD_CODE_MISALIGNMENT"

    return AlignmentVerificationResult(
        passed=passed,
        alignment_score=alignment_score,
        ast_diff_detected=ast_diff_detected,
        active_gradient_flow=active_gradient_flow,
        min_gradient_norm=min_grad_norm,
        verdict=verdict,
        details={
            "ast_score": ast_score,
            "grad_score": grad_score,
            "grad_details": grad_details,
        },
    )
