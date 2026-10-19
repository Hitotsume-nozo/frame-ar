"""Targeted Analytical Performance Alignment Score (TAPAScore) for FRAME-AR.

Provides a rigorous numerical scalar score in [0, 100] (or normalized [0, 1])
formulated over symbolic proofs, recombination grammar, two-stage statistical screening
beyond ROPE, adversarial null model robustness margins, and prediction variance preservation:

    TAPAScore = Phi(R_proof, R_legality, R_stat, Delta_robust, VarRatio)

Also provides Modality B (Open-Ended / Sparse Autoencoders) formulation with hard
gates on dead feature ratio (< 1%) and downstream loss recovery (R >= 80%).
"""

import math
from typing import Optional, Tuple


def compute_tapa_score(
    r_proof: float,
    r_legality: float,
    r_stat: float,
    delta_robust: float,
    var_ratio: float,
    epsilon_robust: float = 0.005,
    scale_to_100: bool = True,
    weights: Optional[Tuple[float, float, float]] = None,
) -> float:
    """Computes the Targeted Analytical Performance Alignment Score (TAPAScore).

    Formula:
        TAPAScore = R_proof * R_legality * [w_stat * tanh(max(0, R_stat)) +
                                           w_robust * eta_robust +
                                           w_var * eta_var] * (100.0 if scale_to_100 else 1.0)

    Args:
        r_proof: Proof validity indicator from L0-L2 math proofs (1.0 = pass, 0.0 = fail).
        r_legality: Recombination grammar compliance indicator (1.0 = legal, 0.0 = illegal).
        r_stat: Statistical discovery strength: max(0, D_bar - eps_ROPE) * ln(1 + alpha_t / max(p_rope, 1e-9)).
        delta_robust: Minimum performance margin over adversarial null models (min_m (L_null,m - L_cand)).
        var_ratio: Prediction variance preservation ratio (s_y_hat / s_y).
        epsilon_robust: Pre-registered robustness margin threshold (default 0.005).
        scale_to_100: If True, scale result to [0, 100]; if False, bounded in [0, 1].
        weights: Optional tuple (w_stat, w_robust, w_var), defaults to (0.50, 0.25, 0.25).

    Returns:
        float scalar in [0.0, 100.0] (or [0.0, 1.0]).
    """
    # Hard epistemic gates: any mathematical pole or grammar illegality zeroes out score immediately
    if r_proof <= 0.0 or r_legality <= 0.0:
        return 0.0

    w_stat, w_robust, w_var = weights if weights is not None else (0.50, 0.25, 0.25)
    total_w = w_stat + w_robust + w_var
    if total_w <= 0:
        return 0.0
    w_stat /= total_w
    w_robust /= total_w
    w_var /= total_w

    # 1. Statistical alignment component (bounded via tanh)
    eta_stat = float(math.tanh(max(0.0, float(r_stat))))

    # 2. Robustness component: clipped relative margin over epsilon_robust
    eps_r = max(float(epsilon_robust), 1e-9)
    eta_robust = float(max(0.0, min(1.0, float(delta_robust) / eps_r)))

    # 3. Variance preservation component:
    # Floor at 0.50 (prevent collapsing to constant mean), optimal at 1.0, penalize explosion > 2.0
    vr = float(var_ratio)
    if vr < 0.50:
        eta_var = 0.0
    elif vr <= 1.0:
        eta_var = (vr - 0.50) / 0.50
    elif vr <= 2.0:
        eta_var = max(0.0, 1.0 - 0.5 * (vr - 1.0))
    else:
        eta_var = max(0.0, 0.5 - 0.5 * (vr - 2.0))

    phi = w_stat * eta_stat + w_robust * eta_robust + w_var * eta_var
    phi = max(0.0, min(1.0, phi))

    multiplier = 100.0 if scale_to_100 else 1.0
    return round(float(r_proof * r_legality * phi * multiplier), 4)


def compute_tapa_score_modality_b(
    r_proof: float,
    r_legality: float,
    l0_sparsity: float,
    l0_target: float,
    loss_recovery: float,
    dead_feature_ratio: float,
    r_stat: float = 0.0,
    dead_threshold: float = 0.01,
    recovery_floor: float = 0.80,
    scale_to_100: bool = True,
) -> float:
    """Computes TAPAScore for Modality B (Open-Ended / Sparse Autoencoders).

    Hard Gates:
        - r_proof > 0 and r_legality > 0
        - dead_feature_ratio < dead_threshold (default 0.01 = 1%)
        - loss_recovery >= recovery_floor (default 0.80 = 80%)

    Continuous Pillars:
        - Downstream loss recovery R in [recovery_floor, 1.0] (weight 0.45)
        - L0 sparsity alignment exp(-|L0 - L0_target| / L0_target) (weight 0.35)
        - Statistical discovery strength tanh(max(0, r_stat)) (weight 0.20)
    """
    if r_proof <= 0.0 or r_legality <= 0.0:
        return 0.0

    if dead_feature_ratio >= dead_threshold:
        return 0.0

    if loss_recovery < recovery_floor:
        return 0.0

    # Downstream loss recovery alignment
    eta_recovery = max(0.0, min(1.0, (loss_recovery - recovery_floor) / (1.0 - recovery_floor)))

    # Sparsity alignment
    target = max(1e-6, float(l0_target))
    delta_sparsity = abs(float(l0_sparsity) - target) / target
    eta_sparsity = float(math.exp(-delta_sparsity))

    # Statistical alignment
    eta_stat = float(math.tanh(max(0.0, float(r_stat))))

    phi = 0.45 * eta_recovery + 0.35 * eta_sparsity + 0.20 * eta_stat
    phi = max(0.0, min(1.0, phi))

    multiplier = 100.0 if scale_to_100 else 1.0
    return round(float(phi * multiplier), 4)


__all__ = [
    "compute_tapa_score",
    "compute_tapa_score_modality_b",
]
