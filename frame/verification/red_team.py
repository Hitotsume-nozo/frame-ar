"""Adversarial Red-Team Degeneracy Suite and Causal Leakage Probe for FRAME-AR.

Implements:
1. 5-probe null degeneracy testing (zero, constant, random, identity, label-permuted).
2. Hard robustness margin gate: Delta_null > epsilon_robust.
3. Causal mask lookahead leakage probe asserting d(y_t) / d(x_{t+k}) == 0.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from frame.schemas.hypothesis import TargetDirection


@dataclass
class RedTeamDegeneracyResult:
    passed: bool
    candidate_score: float
    null_scores: Dict[str, float]
    max_null_score: float
    min_null_score: float
    margin: float
    epsilon_robust: float
    verdict: str  # PASSED_RED_TEAM | REJECTED_NULL_DEGENERACY
    execution_time_s: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalLeakageResult:
    passed: bool
    max_future_gradient: float
    leakage_detected: bool
    leaked_timesteps: List[Tuple[int, int]]  # List of (t, t+k) where leakage occurred
    verdict: str  # STRICTLY_CAUSAL | CAUSAL_LOOKAHEAD_LEAKAGE
    details: Dict[str, Any] = field(default_factory=dict)


class AdversarialRedTeamSuite:
    """Evaluates candidate models against 5 adversarial null models and probes causal masks."""

    def __init__(
        self,
        epsilon_robust: float = 0.02,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
    ):
        self.epsilon_robust = epsilon_robust
        self.target_direction = target_direction

    def evaluate_null_models(
        self,
        candidate_model: nn.Module,
        eval_data: List[Tuple[torch.Tensor, torch.Tensor]],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> RedTeamDegeneracyResult:
        """Evaluates candidate against 5 null models:
        1. Zero: f(x) = 0
        2. Constant: f(x) = mean(y)
        3. Random: f(x) ~ N(0, I)
        4. Identity: f(x) = x (or linear projection)
        5. Label-permuted: candidate evaluated on permuted targets
        """
        t0 = time.perf_counter()
        candidate_model.eval()
        device = next(candidate_model.parameters()).device if list(candidate_model.parameters()) else torch.device("cpu")

        # 1. Compute empirical mean of targets across dataset
        all_targets: List[torch.Tensor] = []
        candidate_batch_scores: List[float] = []

        with torch.inference_mode():
            for x_b, y_b in eval_data:
                x_b, y_b = x_b.to(device), y_b.to(device)
                all_targets.append(y_b)
                pred = candidate_model(x_b)
                candidate_batch_scores.append(float(metric_fn(pred, y_b)))

        candidate_score = float(np.mean(candidate_batch_scores)) if candidate_batch_scores else 0.0
        concat_targets = torch.cat(all_targets, dim=0)
        target_mean = concat_targets.float().mean(dim=0, keepdim=True)

        null_scores: Dict[str, float] = {}

        # 2. Evaluate 5 Null Models
        # (1) Zero Model
        zero_scores: List[float] = []
        for x_b, y_b in eval_data:
            x_b, y_b = x_b.to(device), y_b.to(device)
            # Create zeros matching target shape
            pred_zero = torch.zeros_like(y_b).float()
            zero_scores.append(float(metric_fn(pred_zero, y_b)))
        null_scores["zero_model"] = float(np.mean(zero_scores)) if zero_scores else 0.0

        # (2) Constant Model (mean of y)
        const_scores: List[float] = []
        for x_b, y_b in eval_data:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred_const = target_mean.expand(y_b.shape).to(device)
            const_scores.append(float(metric_fn(pred_const, y_b)))
        null_scores["constant_model"] = float(np.mean(const_scores)) if const_scores else 0.0

        # (3) Random Model N(0, I)
        rand_scores: List[float] = []
        for x_b, y_b in eval_data:
            x_b, y_b = x_b.to(device), y_b.to(device)
            pred_rand = torch.randn_like(y_b.float())
            rand_scores.append(float(metric_fn(pred_rand, y_b)))
        null_scores["random_model"] = float(np.mean(rand_scores)) if rand_scores else 0.0

        # (4) Identity Model f(x) = x (or linear projection if shapes mismatch)
        ident_scores: List[float] = []
        for x_b, y_b in eval_data:
            x_b, y_b = x_b.to(device), y_b.to(device)
            if x_b.shape == y_b.shape:
                pred_id = x_b.float()
            else:
                pred_id = torch.zeros_like(y_b).float()
                x_flat = x_b.view(x_b.shape[0], -1)
                y_flat = pred_id.view(y_b.shape[0], -1)
                common_feats = min(x_flat.shape[1], y_flat.shape[1])
                y_flat[:, :common_feats] = x_flat[:, :common_feats].float()
                pred_id = y_flat.view_as(y_b)
            ident_scores.append(float(metric_fn(pred_id, y_b)))
        null_scores["identity_model"] = float(np.mean(ident_scores)) if ident_scores else 0.0

        # (5) Label-Permuted Model
        # Use global target derangement so permutation is non-trivial even when batch size is 1
        perm_scores: List[float] = []
        total_samples = concat_targets.shape[0]
        if total_samples > 1:
            shift = max(1, total_samples // 2)
            permuted_global = torch.roll(concat_targets, shifts=shift, dims=0)
        else:
            permuted_global = concat_targets

        with torch.inference_mode():
            offset = 0
            for x_b, y_b in eval_data:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = candidate_model(x_b)
                bsz = y_b.shape[0]
                y_perm = permuted_global[offset:offset + bsz].to(device)
                offset += bsz
                perm_scores.append(float(metric_fn(pred, y_perm)))
        null_scores["label_permuted_model"] = float(np.mean(perm_scores)) if perm_scores else 0.0

        max_null = max(null_scores.values())
        min_null = min(null_scores.values())

        # Check robustness margin
        is_max = (self.target_direction == TargetDirection.MAXIMIZE or self.target_direction == "maximize")
        if is_max:
            margin = candidate_score - max_null
            passed = margin > self.epsilon_robust
        else:
            # For minimization, candidate score must be strictly lower than best null model
            margin = min_null - candidate_score
            passed = margin > self.epsilon_robust

        verdict = "PASSED_RED_TEAM" if passed else "REJECTED_NULL_DEGENERACY"
        elapsed = time.perf_counter() - t0

        return RedTeamDegeneracyResult(
            passed=passed,
            candidate_score=candidate_score,
            null_scores=null_scores,
            max_null_score=max_null,
            min_null_score=min_null,
            margin=margin,
            epsilon_robust=self.epsilon_robust,
            verdict=verdict,
            execution_time_s=elapsed,
        )

    @staticmethod
    def check_causal_lookahead_leakage(
        model: nn.Module,
        seq_len: int = 8,
        feature_dim: int = 16,
        tolerance: float = 1e-9,
        num_trials: int = 3,
    ) -> CausalLeakageResult:
        """Asserts that for autoregressive sequence models, future-token gradient
        d(y_t) / d(x_{t+k}) == 0 for all t and k >= 1.
        
        Args:
            model: PyTorch model accepting input [B, T, D] and producing output [B, T, D].
            seq_len: Length of sequence to probe.
            feature_dim: Dimension of features.
            tolerance: Maximum allowable gradient norm for future inputs.
            num_trials: Number of diverse input trials to prevent dead-neuron gradient masking.
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        leaked_timesteps: List[Tuple[int, int]] = []
        max_future_grad = 0.0

        for trial in range(num_trials):
            # Use distinct initialization per trial to avoid zero-gradient activation plateaus
            if trial == 0:
                x = torch.randn(1, seq_len, feature_dim, device=device, requires_grad=True)
            elif trial == 1:
                x = torch.empty(1, seq_len, feature_dim, device=device).uniform_(0.5, 2.0).requires_grad_()
            else:
                x = torch.empty(1, seq_len, feature_dim, device=device).uniform_(-2.0, -0.5).requires_grad_()

            for t in range(seq_len - 1):
                if x.grad is not None:
                    x.grad.zero_()
                model.zero_grad(set_to_none=True)
                
                output = model(x)
                # Scalar target: sum of elements at timestep t
                target = output[0, t].sum()
                target.backward()

                if x.grad is not None:
                    # Check gradients for timesteps t+1, t+2, ... seq_len-1
                    for k in range(t + 1, seq_len):
                        future_grad_norm = float(x.grad[0, k].norm().item())
                        if future_grad_norm > max_future_grad:
                            max_future_grad = future_grad_norm
                        if future_grad_norm > tolerance:
                            pair = (t, k)
                            if pair not in leaked_timesteps:
                                leaked_timesteps.append(pair)

        leakage_detected = len(leaked_timesteps) > 0 or max_future_grad > tolerance
        verdict = "CAUSAL_LOOKAHEAD_LEAKAGE" if leakage_detected else "STRICTLY_CAUSAL"

        return CausalLeakageResult(
            passed=not leakage_detected,
            max_future_gradient=max_future_grad,
            leakage_detected=leakage_detected,
            leaked_timesteps=leaked_timesteps,
            verdict=verdict,
            details={"seq_len": seq_len, "tolerance": tolerance, "num_trials": num_trials},
        )
