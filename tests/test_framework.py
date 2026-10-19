"""End-to-End Verification Suite for FRAME-AR.

Verifies:
1. RFC 8785 Canonical JSON determinism
2. Pydantic v2 schemas and hash generation
3. PyTorch FlopCounterMode profiling and compute routing
4. LORD 3 streaming FDR math and wealth rejuvenation
5. Paired seed ROPE superiority statistical testing
6. Kaggle self-extracting payload generation
7. Manual execution capsule generation and micro-replay verification
"""

import math
import os
import tempfile
import torch
import torch.nn as nn
from scipy import stats

from frame.utils.canonical_json import canonical_json_dumps
from frame.schemas.hypothesis import Hypothesis, TargetDirection
from frame.schemas.compute import ExecutionTier
from frame.schemas.receipt import ExecutionReceipt, HardwareTelemetry
from frame.schemas.verification import StatisticalVerificationReport, PairedSeedMetrics
from frame.pot.tracer import profile_module_flops, route_compute_tier
from frame.verification.statistics import (
    LORDController,
    evaluate_paired_seed_rope,
    ZETA_1_6,
    gamma_lord,
)
from frame.runners.kaggle.bundler import create_self_extracting_payload
from frame.runners.manual.capsule import package_manual_capsule, verify_micro_replay


def test_canonical_json_determinism():
    """Verifies that dict insertion order does not alter canonical serialization."""
    d1 = {"z": 10, "a": "hello", "m": [1, 2, 3]}
    d2 = {"a": "hello", "m": [1, 2, 3], "z": 10}
    assert canonical_json_dumps(d1) == canonical_json_dumps(d2)
    assert canonical_json_dumps(d1) == '{"a":"hello","m":[1,2,3],"z":10}'


def test_hypothesis_schema_and_hash():
    """Verifies Hypothesis instantiation and non-circular hash calculation."""
    h = Hypothesis(
        hypothesis_id="hyp_001",
        title="Test Hypothesis",
        description="Testing non-circular hash generation",
        model_architecture="MLP",
        hyperparameters={"lr": 0.001, "hidden_dim": 128},
        metric_name="accuracy",
        baseline_code_snippet="model = Baseline()",
        candidate_code_snippet="model = Candidate()",
    )
    computed_hash = h.compute_hash()
    assert isinstance(computed_hash, str)
    assert len(computed_hash) == 64
    h.hypothesis_hash = computed_hash
    # Ensure assigning hypothesis_hash does not change recomputed hash
    assert h.compute_hash() == computed_hash


def test_pot_flop_counter_and_routing():
    """Verifies exact FlopCounterMode measurement and calibrated compute routing."""
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    sample_input = torch.randn(1, 64)
    fwd_flops, bwd_flops, params = profile_module_flops(model, sample_input)
    
    assert fwd_flops > 0
    assert bwd_flops > 0
    assert params == (64 * 128 + 128) + (128 * 10 + 10)
    
    # Test Tier 1 routing (small local workload)
    profile_small = route_compute_tier(fwd_flops, bwd_flops, params, seed_count=3, epochs=2, batches_per_epoch=5)
    assert profile_small.tier == ExecutionTier.TIER_1_LOCAL
    
    # Test Tier 2 routing (medium Kaggle workload)
    profile_medium = route_compute_tier(fwd_flops * 1000, bwd_flops * 1000, 5_000_000, seed_count=5, epochs=50, batches_per_epoch=1000)
    assert profile_medium.tier in (ExecutionTier.TIER_1_LOCAL, ExecutionTier.TIER_2_KAGGLE)
    
    # Test Tier 3 routing (massive compute workload exceeding Kaggle limits)
    profile_massive = route_compute_tier(10**11, 2 * 10**11, 70_000_000_000, seed_count=5, epochs=100, batches_per_epoch=10000)
    assert profile_massive.tier == ExecutionTier.TIER_3_MANUAL


def test_lord_3_streaming_fdr_math():
    """Verifies LORD 3 normalizing constant, step 1 alpha, and wealth rejuvenation."""
    # Check zeta(1.6) normalizing constant
    assert abs(ZETA_1_6 - 2.28576595) < 1e-6
    
    # Check step 1 alpha: W0 * gamma_1 = (0.05 / 2) / 2.28576595 ≈ 0.010937
    controller = LORDController(alpha=0.05)
    alpha_1 = controller.get_next_alpha()
    assert abs(alpha_1 - 0.0109372) < 1e-5
    
    # Record a non-discovery at step 1 (wealth should decay at step 2)
    is_disc, a1 = controller.record_result(p_value=0.25)
    assert not is_disc
    assert controller.current_step == 1
    alpha_2 = controller.get_next_alpha()
    assert alpha_2 < alpha_1  # Alpha decays without discovery
    
    # Record a confirmed discovery at step 2
    is_disc2, a2 = controller.record_result(p_value=0.0001)
    assert is_disc2
    assert len(controller.discovery_indices) == 1
    assert controller.discovery_indices[0] == 2
    
    # At step 3, alpha should be rejuvenated by the discovery at step 2
    alpha_3 = controller.get_next_alpha()
    assert alpha_3 > alpha_2


def test_paired_seed_rope_superiority():
    """Verifies that paired difference analysis correctly evaluates ROPE superiority."""
    controller = LORDController(alpha=0.05)
    
    # Synthetic candidate that significantly beats baseline beyond epsilon=0.01
    baseline = {"seed_1": 0.80, "seed_2": 0.81, "seed_3": 0.79, "seed_4": 0.82, "seed_5": 0.80}
    candidate = {"seed_1": 0.85, "seed_2": 0.86, "seed_3": 0.84, "seed_4": 0.87, "seed_5": 0.85}
    
    report = evaluate_paired_seed_rope(
        hypothesis_id="h_super",
        metric_name="accuracy",
        baseline_scores=baseline,
        candidate_scores=candidate,
        rope_epsilon=0.01,
        lord_controller=controller,
    )
    
    assert report.is_discovery is True
    assert report.verdict == "CONFIRMED_DISCOVERY"
    assert math.isclose(report.paired_metrics.mean_difference, 0.05, abs_tol=1e-6)
    assert report.p_value_rope < 0.001
    assert report.report_hash is not None


def test_paired_seed_rope_minimization():
    """Verifies that paired difference analysis correctly evaluates minimization (e.g. loss reduction)."""
    controller = LORDController(alpha=0.05)
    
    # Candidate achieves lower loss (reduction from ~2.0 to ~1.5)
    baseline_losses = {"seed_1": 2.05, "seed_2": 2.00, "seed_3": 1.95, "seed_4": 2.10, "seed_5": 2.00}
    candidate_losses = {"seed_1": 1.55, "seed_2": 1.50, "seed_3": 1.45, "seed_4": 1.60, "seed_5": 1.50}
    
    report = evaluate_paired_seed_rope(
        hypothesis_id="h_loss_drop",
        metric_name="cross_entropy_loss",
        baseline_scores=baseline_losses,
        candidate_scores=candidate_losses,
        rope_epsilon=0.05,
        lord_controller=controller,
        target_direction=TargetDirection.MINIMIZE,
    )
    
    assert report.is_discovery is True
    assert report.verdict == "CONFIRMED_DISCOVERY"
    assert math.isclose(report.paired_metrics.mean_difference, 0.50, abs_tol=1e-6)
    assert report.p_value_rope < 0.001



def test_kaggle_payload_bundler():
    """Verifies creation of in-memory self-extracting payload for Kaggle."""
    files = {
        "model.py": "def get_model(): return 'MyModel'\n",
        "config.json": '{"lr": 0.001}\n',
    }
    main_script = "import model\nprint(model.get_model())\n"
    bundle_script = create_self_extracting_payload(files, main_script)
    
    assert "PAYLOAD_B64" in bundle_script
    assert "base64.b64decode" in bundle_script
    assert "print(model.get_model())" in bundle_script


def test_manual_capsule_packaging_and_replay():
    """Verifies packaging of an air-gapped manual capsule and replay verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest = package_manual_capsule(
            output_dir=tmpdir,
            capsule_id="cap_001",
            hypothesis_id="hyp_001",
            code_content="print('Training heavy model...')",
            deterministic_seeds=[10, 20, 30],
        )
        assert os.path.exists(os.path.join(tmpdir, "manifest.json"))
        assert os.path.exists(os.path.join(tmpdir, "entrypoint.py"))
        assert os.path.exists(os.path.join(tmpdir, "run.sh"))
        assert os.path.exists(os.path.join(tmpdir, "Dockerfile"))
        
        # Verify micro-replay pass
        recorded = [2.50012, 1.80005, 1.20001]
        replayed_honest = [2.50014, 1.80004, 1.20002]
        res_honest = verify_micro_replay("hyp_001", recorded, replayed_honest, tolerance=1e-4)
        assert res_honest.passed is True
        assert res_honest.max_loss_delta <= 1e-4
        
        # Corrupted / fraudulent run
        replayed_corrupted = [2.50012, 1.20000, 0.50000]
        res_corrupted = verify_micro_replay("hyp_001", recorded, replayed_corrupted, tolerance=1e-4)
        assert res_corrupted.passed is False
        assert res_corrupted.max_loss_delta > 1e-4


def test_cryptographic_schema_linking_chain_of_custody():
    """Verifies that ExecutionReceipt and StatisticalVerificationReport preserve the cryptographic chain."""
    # 1. Hypothesis
    h = Hypothesis(
        hypothesis_id="hyp_chain_01",
        title="Chain of Custody Hypothesis",
        description="Verifying Merkle linking",
        model_architecture="Transformer",
        metric_name="accuracy",
        hyperparameters={"lr": 0.001},
        baseline_code_snippet="pass",
        candidate_code_snippet="pass",
    )
    hyp_hash = h.compute_hash()
    h.hypothesis_hash = hyp_hash

    # 2. Execution Receipt linking hypothesis_hash
    receipt = ExecutionReceipt(
        receipt_id="rcpt_001",
        hypothesis_id="hyp_chain_01",
        tier=ExecutionTier.TIER_1_LOCAL,
        baseline_seed_scores={"seed_1": 0.80, "seed_2": 0.81, "seed_3": 0.82},
        candidate_seed_scores={"seed_1": 0.85, "seed_2": 0.86, "seed_3": 0.87},
        replay_micro_step_losses=[1.0, 0.9, 0.8],
        replay_seed=42,
        wall_clock_seconds=12.5,
        hardware=HardwareTelemetry(
            device_name="NVIDIA GTX 1650",
            cuda_available=True,
            pytorch_version="2.2.0",
            peak_vram_mb=2048.0,
            cpu_threads=8,
        ),
        stdout_hash="stdout_hash_abc",
        code_hash="code_hash_123",
        hypothesis_hash=hyp_hash,
    )
    rcpt_hash = receipt.compute_hash()
    assert len(rcpt_hash) == 64
    receipt.receipt_hash = rcpt_hash
    # Assigning receipt_hash does not change hash
    assert receipt.compute_hash() == rcpt_hash
    assert receipt.hypothesis_hash == hyp_hash

    # 3. Statistical Verification Report linking both hypothesis_hash and receipt_hash
    lord = LORDController(alpha=0.05)
    report = evaluate_paired_seed_rope(
        hypothesis_id="hyp_chain_01",
        metric_name="accuracy",
        baseline_scores=receipt.baseline_seed_scores,
        candidate_scores=receipt.candidate_seed_scores,
        rope_epsilon=0.01,
        lord_controller=lord,
        hypothesis_hash=hyp_hash,
        receipt_hash=rcpt_hash,
    )
    assert report.hypothesis_hash == hyp_hash
    assert report.receipt_hash == rcpt_hash
    assert report.report_hash is not None
    assert len(report.report_hash) == 64
    # Ensure assigning report_hash did not change recomputed hash
    assert report.compute_hash() == report.report_hash


def test_statistics_type_hints_inspection():
    """Asserts that evaluate_paired_seed_rope type annotations resolve cleanly without NameError."""
    import typing
    import frame.verification.statistics as stats_mod
    hints = typing.get_type_hints(stats_mod.evaluate_paired_seed_rope)
    assert "hypothesis_hash" in hints
    assert "receipt_hash" in hints
    assert "return" in hints
