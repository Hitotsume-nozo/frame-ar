"""Unit Tests for Anti-Reward Hacking & Adversarial Defense Suite."""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from frame.schemas.hypothesis import (
    Hypothesis,
    TargetDirection,
    generate_sealed_seeds,
    create_seed_commitment,
    verify_seed_commitment,
)
from frame.verification.harness import DecoupledIsolatedVerifierHarness
from frame.verification.red_team import AdversarialRedTeamSuite


# ==============================================================================
# 1. Sealed vs Public Seed Protocol Tests
# ==============================================================================

def test_sealed_seed_hmac_generation_and_commitment():
    """Verifies that sealed seeds are cryptographically generated and verifiable."""
    verifier_key = "secret_verifier_key_2026"
    hyp_id = "hyp_attention_variant_12"

    seeds = generate_sealed_seeds(verifier_key, hyp_id, count=5)
    assert len(seeds) == 5
    assert all(0 <= s < 2**31 - 1 for s in seeds)

    # Determinism check
    seeds_again = generate_sealed_seeds(verifier_key, hyp_id, count=5)
    assert seeds == seeds_again

    # Different key produces different seeds
    seeds_diff_key = generate_sealed_seeds("different_key", hyp_id, count=5)
    assert seeds != seeds_diff_key

    # Commitment generation and verification
    salt = "salt_xyz_123"
    commitment = create_seed_commitment(seeds, salt)
    assert isinstance(commitment, str)
    assert len(commitment) == 64

    # Valid verification
    assert verify_seed_commitment(commitment, seeds, salt) is True

    # Tampered seeds detection
    tampered = list(seeds)
    tampered[0] += 1
    assert verify_seed_commitment(commitment, tampered, salt) is False


# ==============================================================================
# 2. Decoupled Isolated Verifier Harness Tests
# ==============================================================================

def test_decoupled_isolated_verifier_harness():
    """Verifies candidate evaluation on private holdout data with activation hashes."""
    # Synthetic holdout data: 5 batches of (x, y)
    holdout_data = []
    for _ in range(5):
        x = torch.randn(4, 8)
        y = (x.sum(dim=-1, keepdim=True) > 0).float()
        holdout_data.append((x, y))

    # Simple linear candidate model
    model = nn.Sequential(nn.Linear(8, 1), nn.Sigmoid())

    def accuracy_metric(preds, targets):
        binary_preds = (preds > 0.5).float()
        return float((binary_preds == targets).float().mean().item())

    harness = DecoupledIsolatedVerifierHarness(
        holdout_data=holdout_data,
        metric_fn=accuracy_metric,
        metric_name="accuracy",
        commitment_steps=(1, 3, 5),
    )

    result = harness.evaluate_candidate(
        hypothesis_id="hyp_eval_001",
        model=model,
        seed_list=[101, 102],
    )

    assert result.passed is True
    assert "seed_101" in result.scores
    assert "seed_102" in result.scores
    assert 0.0 <= result.mean_score <= 1.0
    # Activation hashes should be captured at steps 1, 3, 5
    assert 1 in result.activation_hashes
    assert 3 in result.activation_hashes
    assert 5 in result.activation_hashes
    assert len(result.compute_commitment_hash()) == 64


# ==============================================================================
# 3. Adversarial Red-Team Degeneracy Suite Tests
# ==============================================================================

def test_adversarial_red_team_passes_good_model():
    """Verifies that an informative model beats the 5 null models beyond epsilon."""
    # Synthetic regression dataset: y = 2 * x_0 + 0.5
    eval_data = []
    for _ in range(6):
        x = torch.randn(8, 4)
        y = 2.0 * x[:, :1] + 0.5
        eval_data.append((x, y))

    # Honest linear model with correct weights
    good_model = nn.Linear(4, 1)
    with torch.no_grad():
        good_model.weight.zero_()
        good_model.weight[0, 0] = 2.0
        good_model.bias.fill_(0.5)

    # Metric: Negative MSE (higher is better for MAXIMIZE)
    def neg_mse(preds, targets):
        return -float(((preds - targets) ** 2).mean().item())

    suite = AdversarialRedTeamSuite(epsilon_robust=0.1, target_direction=TargetDirection.MAXIMIZE)
    res = suite.evaluate_null_models(good_model, eval_data, neg_mse)

    assert res.passed is True
    assert res.verdict == "PASSED_RED_TEAM"
    assert res.margin > 0.1
    assert "zero_model" in res.null_scores
    assert "constant_model" in res.null_scores
    assert "random_model" in res.null_scores
    assert "identity_model" in res.null_scores
    assert "label_permuted_model" in res.null_scores


def test_adversarial_red_team_rejects_degenerate_model():
    """Verifies that a degenerate (all-zero) model is caught and rejected."""
    eval_data = []
    for _ in range(6):
        x = torch.randn(8, 4)
        y = torch.ones(8, 1) * 3.0  # constant target of 3.0
        eval_data.append((x, y))

    # Degenerate all-zero model
    zero_model = nn.Sequential(nn.Linear(4, 1, bias=False))
    with torch.no_grad():
        zero_model[0].weight.zero_()

    def neg_mse(preds, targets):
        return -float(((preds - targets) ** 2).mean().item())

    suite = AdversarialRedTeamSuite(epsilon_robust=0.05, target_direction=TargetDirection.MAXIMIZE)
    res = suite.evaluate_null_models(zero_model, eval_data, neg_mse)

    assert res.passed is False
    assert res.verdict == "REJECTED_NULL_DEGENERACY"


# ==============================================================================
# 4. Causal Mask Lookahead Leakage Probe Tests
# ==============================================================================

class CausalLinearSequenceModel(nn.Module):
    """Sequence model strictly enforcing causal lower-triangular weight masking."""

    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        self.weight = nn.Parameter(torch.randn(seq_len, seq_len))
        self.feature_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Apply lower-triangular causal mask to temporal mixing matrix
        causal_mask = torch.tril(torch.ones(self.seq_len, self.seq_len, device=x.device))
        masked_w = self.weight * causal_mask
        # x is [B, T, D]
        x_proj = self.feature_proj(x)
        # Mix along sequence dimension T
        out = torch.einsum("ij,bjd->bid", masked_w, x_proj)
        return out


class LeakySequenceModel(nn.Module):
    """Sequence model with lookahead leakage (unmasked future attention)."""

    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(seq_len, seq_len))
        self.feature_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # Dense unmasked temporal mixing -> leaks future tokens!
        x_proj = self.feature_proj(x)
        return torch.einsum("ij,bjd->bid", self.weight, x_proj)


def test_causal_leakage_probe_detects_causal_vs_leaky():
    """Verifies that the causal probe passes strictly causal models and rejects leaky ones."""
    seq_len = 6
    dim = 8

    # 1. Strictly causal model
    causal_model = CausalLinearSequenceModel(seq_len, dim)
    res_causal = AdversarialRedTeamSuite.check_causal_lookahead_leakage(
        causal_model, seq_len=seq_len, feature_dim=dim, tolerance=1e-7
    )
    assert res_causal.passed is True
    assert res_causal.verdict == "STRICTLY_CAUSAL"
    assert res_causal.leakage_detected is False
    assert len(res_causal.leaked_timesteps) == 0

    # 2. Leaky model
    leaky_model = LeakySequenceModel(seq_len, dim)
    res_leaky = AdversarialRedTeamSuite.check_causal_lookahead_leakage(
        leaky_model, seq_len=seq_len, feature_dim=dim, tolerance=1e-7
    )
    assert res_leaky.passed is False
    assert res_leaky.verdict == "CAUSAL_LOOKAHEAD_LEAKAGE"
    assert res_leaky.leakage_detected is True
    assert len(res_leaky.leaked_timesteps) > 0


def test_adversarial_red_team_batch_size_1_passes_good_model():
    """Verifies that an informative model on batch size 1 is NOT falsely rejected by label permutation."""
    eval_data_b1 = []
    for i in range(12):
        x = torch.randn(1, 4)
        y = 2.0 * x[:, :1] + 0.5
        eval_data_b1.append((x, y))

    good_model = nn.Linear(4, 1)
    with torch.no_grad():
        good_model.weight.zero_()
        good_model.weight[0, 0] = 2.0
        good_model.bias.fill_(0.5)

    def neg_mse(preds, targets):
        return -float(((preds - targets) ** 2).mean().item())

    suite = AdversarialRedTeamSuite(epsilon_robust=0.1, target_direction=TargetDirection.MAXIMIZE)
    res = suite.evaluate_null_models(good_model, eval_data_b1, neg_mse)

    assert res.passed is True
    assert res.verdict == "PASSED_RED_TEAM"
    assert res.margin > 0.1
    # Candidate score should beat label permuted model significantly
    assert res.candidate_score > res.null_scores["label_permuted_model"]


class LeakyNonlinearSequenceModel(nn.Module):
    """Sequence model with leaky future attention and non-linear threshold."""
    def __init__(self, seq_len: int, dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Linear(dim, dim)
        self.future_leak = nn.Parameter(torch.ones(seq_len, seq_len) * 0.5)

    def forward(self, x):
        h = self.proj(x)
        # Add future leakage
        out = torch.einsum("ij,bjd->bid", self.future_leak, h)
        return torch.relu(out + 1.0)


def test_causal_leakage_probe_multi_trial_catches_nonlinear_leakage():
    """Verifies that multi-trial causal probe detects leakage through non-linear activations."""
    seq_len = 6
    dim = 8
    model = LeakyNonlinearSequenceModel(seq_len, dim)
    res = AdversarialRedTeamSuite.check_causal_lookahead_leakage(
        model, seq_len=seq_len, feature_dim=dim, tolerance=1e-7, num_trials=3
    )
    assert res.passed is False
    assert res.leakage_detected is True
    assert res.verdict == "CAUSAL_LOOKAHEAD_LEAKAGE"
