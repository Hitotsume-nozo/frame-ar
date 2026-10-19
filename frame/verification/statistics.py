"""Mathematical Execution Verification Engine for FRAME-AR.

Implements:
1. Common Random Numbers (CRN) Paired Seed Testing
2. Region of Practical Equivalence (ROPE) Superiority Testing
3. LORD 3 (LORD++) Online Streaming False Discovery Rate (FDR) Controller
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats

DEFAULT_LORD_STATE_PATH: str = os.path.join(".frame", "lord_state.json")

from frame.schemas.hypothesis import TargetDirection
from frame.schemas.verification import (
    StatisticalVerificationReport,
    PairedSeedMetrics,
)

# Riemann Zeta function evaluated at s=1.6: scipy.special.zeta(1.6) ≈ 2.28576595
ZETA_1_6 = 2.28576595


def gamma_lord(step: int) -> float:
    """Computes decaying weight gamma_t = 1 / (zeta(1.6) * t^1.6)."""
    if step <= 0:
        return 0.0
    return 1.0 / (ZETA_1_6 * (step ** 1.6))


class LORDController:
    """LORD 3 (LORD++) Online Streaming FDR Controller.
    
    Guarantees FDR <= alpha at any arbitrary stopping time T across asynchronous
    experimental streams.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.w0 = alpha / 2.0  # Initial alpha wealth (e.g. 0.025)
        self.current_step = 0
        self.discovery_indices: List[int] = []  # tau_1, tau_2, ... (1-indexed)

    def get_next_alpha(self) -> float:
        """Calculates alpha_t allocated for step t = current_step + 1."""
        t = self.current_step + 1
        
        # Base decay term
        alpha_t = self.w0 * gamma_lord(t)
        
        # Wealth rejuvenation from discoveries
        if len(self.discovery_indices) >= 1:
            tau_1 = self.discovery_indices[0]
            if tau_1 < t:
                alpha_t += (self.alpha - self.w0) * gamma_lord(t - tau_1)
        
        if len(self.discovery_indices) >= 2:
            for tau_j in self.discovery_indices[1:]:
                if tau_j < t:
                    alpha_t += self.alpha * gamma_lord(t - tau_j)
                    
        return alpha_t

    def record_result(self, p_value: float) -> Tuple[bool, float]:
        """Records the p-value for the current step, updates wealth, and advances step counter.
        
        Returns:
            Tuple[is_discovery, allocated_alpha_t]
        """
        self.current_step += 1
        t = self.current_step
        
        # Compute allocated alpha for this step
        alpha_t = self.w0 * gamma_lord(t)
        if len(self.discovery_indices) >= 1:
            tau_1 = self.discovery_indices[0]
            if tau_1 < t:
                alpha_t += (self.alpha - self.w0) * gamma_lord(t - tau_1)
        if len(self.discovery_indices) >= 2:
            for tau_j in self.discovery_indices[1:]:
                if tau_j < t:
                    alpha_t += self.alpha * gamma_lord(t - tau_j)

        is_discovery = (p_value <= alpha_t)
        if is_discovery:
            self.discovery_indices.append(t)
            
        return is_discovery, alpha_t

    def to_dict(self) -> Dict[str, Any]:
        """Serializes controller state to dict."""
        return {
            "alpha": self.alpha,
            "w0": self.w0,
            "current_step": self.current_step,
            "discovery_indices": self.discovery_indices,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LORDController":
        """Reconstructs controller from dict."""
        c = cls(alpha=data.get("alpha", 0.05))
        c.w0 = data.get("w0", c.alpha / 2.0)
        c.current_step = data.get("current_step", 0)
        c.discovery_indices = data.get("discovery_indices", [])
        return c

    def save_to_file(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Persists LORD controller state to disk."""
        target = Path(filepath or DEFAULT_LORD_STATE_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return str(target)

    @classmethod
    def load_from_file(cls, filepath: Optional[Union[str, Path]] = None) -> "LORDController":
        """Loads LORD controller state from disk."""
        target = Path(filepath or DEFAULT_LORD_STATE_PATH)
        if not target.exists():
            return cls()
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)



def evaluate_paired_seed_rope(
    hypothesis_id: str,
    metric_name: str,
    baseline_scores: Dict[str, float],
    candidate_scores: Dict[str, float],
    rope_epsilon: float,
    lord_controller: LORDController,
    target_direction: TargetDirection = TargetDirection.MAXIMIZE,
    hypothesis_hash: Optional[str] = None,
    receipt_hash: Optional[str] = None,
    edge_id: Optional[str] = None,
    source_hypothesis_id: Optional[str] = None,
    target_hypothesis_id: Optional[str] = None,
    source_node_id: Optional[str] = None,
    target_node_id: Optional[str] = None,
    edge_type: str = "atomic_transition",
    tapa_score: Optional[float] = None,
) -> StatisticalVerificationReport:
    """Performs Paired Seed Analysis and tests directional superiority beyond ROPE.
    
    H0: mu_D <= epsilon_ROPE  vs  H1: mu_D > epsilon_ROPE
    where D_i is the directional improvement (positive indicates candidate is better).
    For MAXIMIZE (e.g. accuracy), D_i = Y_candidate,i - Y_baseline,i
    For MINIMIZE (e.g. loss, latency), D_i = Y_baseline,i - Y_candidate,i
    """
    common_seeds = sorted(set(baseline_scores.keys()).intersection(candidate_scores.keys()))
    if len(common_seeds) < 3:
        raise ValueError(f"At least 3 paired seeds required for verification, got {len(common_seeds)}")
        
    diffs = {}
    diff_values = []
    base_values = []
    cand_values = []
    
    for s in common_seeds:
        y_cand = candidate_scores[s]
        y_base = baseline_scores[s]
        if target_direction == TargetDirection.MINIMIZE or target_direction == "minimize":
            d = y_base - y_cand  # positive when candidate loss is lower than baseline
        else:
            d = y_cand - y_base  # positive when candidate accuracy is higher than baseline
        diffs[s] = d
        diff_values.append(d)
        base_values.append(y_base)
        cand_values.append(y_cand)
        
    k = len(diff_values)
    mean_d = float(np.mean(diff_values))
    std_d = float(np.std(diff_values, ddof=1))
    se = std_d / math.sqrt(k)
    
    # Standard paired t-statistic
    t_stat = (mean_d / se) if se > 1e-12 else (999.0 if mean_d > 0 else -999.0)
    
    # Directional ROPE test against epsilon
    if se > 1e-12:
        t_rope = (mean_d - rope_epsilon) / se
        # One-sided survival function (1 - CDF)
        p_rope = float(stats.t.sf(t_rope, df=k - 1))
    else:
        t_rope = 999.0 if mean_d > rope_epsilon else -999.0
        p_rope = 0.0 if mean_d > rope_epsilon else 1.0
        
    # Evaluate with online streaming FDR controller
    is_discovery, alpha_t = lord_controller.record_result(p_rope)
    
    verdict = "CONFIRMED_DISCOVERY" if is_discovery else "CONFIRMED_NULL"
    
    report = StatisticalVerificationReport(
        hypothesis_id=hypothesis_id,
        metric_name=metric_name,
        sample_size_n=k,
        mean_baseline=float(np.mean(base_values)),
        mean_candidate=float(np.mean(cand_values)),
        paired_metrics=PairedSeedMetrics(
            metric_name=metric_name,
            paired_differences=diffs,
            mean_difference=mean_d,
            standard_error=se,
            t_statistic=t_stat,
        ),
        rope_epsilon=rope_epsilon,
        rope_t_statistic=t_rope,
        p_value_rope=p_rope,
        lord_step=lord_controller.current_step,
        allocated_alpha_t=alpha_t,
        is_discovery=is_discovery,
        verdict=verdict,
        hypothesis_hash=hypothesis_hash,
        receipt_hash=receipt_hash,
        edge_id=edge_id,
        source_hypothesis_id=source_hypothesis_id,
        target_hypothesis_id=target_hypothesis_id or hypothesis_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        tapa_score=tapa_score,
    )
    report.report_hash = report.compute_hash()
    return report

