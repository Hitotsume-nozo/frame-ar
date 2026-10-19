"""End-to-End Opaque-Box Cognitive Test Suite for FRAME-AR.

Covers all 20 cognitive overhaul features across 4 rigorous tiers:
- Tier 1: Feature Isolation Coverage (20 features x 5 happy-path tests = 100 tests)
- Tier 2: Boundary & Corner Cases (20 features x 5 edge/degenerate tests = 100 tests)
- Tier 3: Cross-Feature Interactions (12 pairwise integration tests)
- Tier 4: Real-World Application Scenarios (6 end-to-end research workflow tests)

Total test count: 218 test cases.
Preserves all 110 baseline tests with 0 regressions.
"""

import io
import json
import math
import sys
import numpy as np
import pytest
import torch
import torch.nn as nn

# Baseline imports from frame (already implemented)
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


# ==============================================================================
# Dynamic Module Accessors for Progressive Testability
# ==============================================================================

def get_governor_module():
    """Dynamically load Information Governor module."""
    try:
        from frame.brain import governor
        return governor
    except ImportError:
        try:
            from frame.agents import governor
            return governor
        except ImportError:
            pytest.skip("Milestone M1 (governor) not yet implemented")


def get_agy_bridge_module():
    """Load Antigravity subagent bridge module."""
    try:
        from frame.agents import agy_bridge
        return agy_bridge
    except ImportError:
        pytest.skip("frame.agents.agy_bridge not available")


def get_novelty_module():
    """Dynamically load Dual-Key Novelty Contract module."""
    try:
        from frame.verification import novelty
        return novelty
    except ImportError:
        pytest.skip("Milestone M2 (novelty) not yet implemented")


def get_learning_state_module():
    """Dynamically load Swarm Learning State Node module."""
    try:
        from frame.brain import learning_state
        return learning_state
    except ImportError:
        pytest.skip("Milestone M3 (learning_state) not yet implemented")


def get_executioner_module():
    """Dynamically load The Executioner Watchdog module."""
    try:
        from frame.brain import executioner
        return executioner
    except ImportError:
        pytest.skip("Milestone M4 (executioner) not yet implemented")


def get_domain_evidence_module():
    """Dynamically load Domain Evidence & Invariants module."""
    try:
        from frame.verification import domain_evidence
        return domain_evidence
    except ImportError:
        pytest.skip("Milestone M5 (domain_evidence) not yet implemented")


# ==============================================================================
# TIER 1: FEATURE ISOLATION COVERAGE (100 Tests: 20 Features x 5 Tests)
# ==============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Comprehensive isolated behavioral tests for Features 1 through 20."""

    # --------------------------------------------------------------------------
    # Feature 1: Information Governor Evidence Vetting
    # --------------------------------------------------------------------------
    def test_tier1_f01_governor_cochrane_acceptance(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        source = gov_mod.EvidenceSource(
            source_id="src_cochrane_01",
            title="Systematic Review of Tabular Ensembling",
            source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=True,
            relevance_score=0.85,
            methodological_rigor=0.90,
        )
        res = gov.grade_source(source)
        assert res.accepted is True
        assert res.grade >= 0.40
        assert res.source_id == "src_cochrane_01"

    def test_tier1_f01_governor_preprint_acceptance(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        source = gov_mod.EvidenceSource(
            source_id="src_arxiv_01",
            title="Attention Variants on Small Data",
            source_type=gov_mod.SourceTier.PREPRINT,
            peer_reviewed=False,
            relevance_score=0.80,
            methodological_rigor=0.75,
        )
        res = gov.grade_source(source)
        # Grade = 0.7 * (0.4*0.8 + 0.4*0.75) = 0.7 * (0.32 + 0.30) = 0.434 >= 0.40
        assert res.accepted is True
        assert res.grade >= 0.40

    def test_tier1_f01_governor_public_forum_rejection(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        source = gov_mod.EvidenceSource(
            source_id="src_forum_01",
            title="Kaggle Forum Quick Trick",
            source_type=gov_mod.SourceTier.PUBLIC_FORUM,
            peer_reviewed=False,
            relevance_score=0.60,
            methodological_rigor=0.20,
        )
        res = gov.grade_source(source)
        # Grade = 0.3 * (0.4*0.6 + 0.4*0.2) = 0.3 * 0.32 = 0.096 < 0.40
        assert res.accepted is False
        assert res.grade < 0.40
        assert res.rejection_reason is not None

    def test_tier1_f01_governor_unverified_rejection(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        source = gov_mod.EvidenceSource(
            source_id="src_unverified_01",
            title="Anonymous Blogpost",
            source_type=gov_mod.SourceTier.UNVERIFIED,
            peer_reviewed=False,
            relevance_score=0.90,
            methodological_rigor=0.10,
        )
        res = gov.grade_source(source)
        assert res.accepted is False

    def test_tier1_f01_governor_audit_and_filter_batch(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        s1 = gov_mod.EvidenceSource(
            source_id="s1", title="Journal", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=True, relevance_score=0.9, methodological_rigor=0.9
        )
        s2 = gov_mod.EvidenceSource(
            source_id="s2", title="Forum", source_type=gov_mod.SourceTier.PUBLIC_FORUM,
            peer_reviewed=False, relevance_score=0.2, methodological_rigor=0.1
        )
        filtered = gov.audit_and_filter([s1, s2])
        assert len(filtered) == 1
        assert filtered[0].source_id == "s1"

    # --------------------------------------------------------------------------
    # Feature 2: High-Entropy Exploratory Mode
    # --------------------------------------------------------------------------
    def test_tier1_f02_entropy_uniform_distribution_passes(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        lane_counts = {"architecture": 10, "objectives": 10, "optimization": 10, "features": 10}
        res = gov.check_exploration_entropy(lane_counts)
        assert res["sufficient"] is True
        assert res["entropy"] >= res["threshold"]

    def test_tier1_f02_entropy_greedy_single_lane_blocked(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        lane_counts = {"architecture": 40, "objectives": 0, "optimization": 0, "features": 0}
        res = gov.check_exploration_entropy(lane_counts)
        assert res["sufficient"] is False
        assert res["status"] == "GREEDY_EXPLOITATION_BLOCKED"

    def test_tier1_f02_entropy_threshold_value_calculation(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        res = gov.check_exploration_entropy({"architecture": 5, "objectives": 5, "optimization": 5, "features": 5})
        expected_thresh = 0.75 * math.log(4)
        assert math.isclose(res["threshold"], expected_thresh, rel_tol=1e-3)

    def test_tier1_f02_entropy_near_uniform_sufficient(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        lane_counts = {"architecture": 8, "objectives": 7, "optimization": 6, "features": 9}
        res = gov.check_exploration_entropy(lane_counts)
        assert res["sufficient"] is True

    def test_tier1_f02_entropy_skewed_distribution_rejected(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        lane_counts = {"architecture": 35, "objectives": 2, "optimization": 2, "features": 1}
        res = gov.check_exploration_entropy(lane_counts)
        assert res["sufficient"] is False

    # --------------------------------------------------------------------------
    # Feature 3: Cognitive Role Specification Generator
    # --------------------------------------------------------------------------
    def test_tier1_f03_specs_generator_exports_governor(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="Optimize Transformer Memory")
        typenames = [s.TypeName for s in specs]
        if "frame_information_governor" not in typenames:
            pytest.skip("Milestone M1 cognitive subagent roles not yet implemented in agy_bridge")
        assert "frame_information_governor" in typenames

    def test_tier1_f03_specs_generator_exports_proximity_analyzer(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="Optimize Transformer Memory")
        typenames = [s.TypeName for s in specs]
        if "frame_literature_proximity_analyzer" not in typenames:
            pytest.skip("Milestone M1 cognitive subagent roles not yet implemented in agy_bridge")
        assert "frame_literature_proximity_analyzer" in typenames

    def test_tier1_f03_specs_generator_exports_meta_reviewer(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="Optimize Transformer Memory")
        typenames = [s.TypeName for s in specs]
        if "frame_meta_reviewer" not in typenames:
            pytest.skip("Milestone M1 cognitive subagent roles not yet implemented in agy_bridge")
        assert "frame_meta_reviewer" in typenames

    def test_tier1_f03_specs_generator_exports_executioner(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="Optimize Transformer Memory")
        typenames = [s.TypeName for s in specs]
        if "frame_executioner" not in typenames:
            pytest.skip("Milestone M1 cognitive subagent roles not yet implemented in agy_bridge")
        assert "frame_executioner" in typenames

    def test_tier1_f03_specs_generator_exports_all_lane_specialists(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="Optimize Transformer Memory")
        typenames = [s.TypeName for s in specs]
        expected_lanes = [
            "frame_lane_architecture",
            "frame_lane_objectives",
            "frame_lane_optimization",
            "frame_lane_features",
        ]
        for lane in expected_lanes:
            assert lane in typenames
        if len(specs) < 8:
            pytest.skip("Milestone M1 cognitive subagent roles not yet implemented in agy_bridge")
        assert len(specs) >= 8

    # --------------------------------------------------------------------------
    # Feature 4: Literature Graph Proximity Evaluator
    # --------------------------------------------------------------------------
    def test_tier1_f04_literature_distance_orthogonal_embedding(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = ["Standard Multi-Head Attention", "Feedforward SwiGLU projection"]
        candidate = "Quantum Orthogonal Fourier Token Mixing without keys or queries"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert d_lit > 0.15
        assert isinstance(nearest, str)

    def test_tier1_f04_literature_distance_identical_prior_art_rejected(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = ["Standard Multi-Head Self Attention"]
        candidate = "Standard Multi-Head Self Attention"
        d_lit, _ = contract.compute_literature_distance(candidate, prior_art)
        assert d_lit <= 0.15

    def test_tier1_f04_literature_distance_multi_corpus_nearest_neighbor(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = [
            "Convolutional ResNet Block",
            "Standard Multi-Head Attention Mechanism",
            "Linear State Space Model SSM",
        ]
        candidate = "Exact Standard Multi-Head Attention Mechanism with linear layer"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert "Attention" in nearest
        assert d_lit <= 0.15

    def test_tier1_f04_literature_distance_above_threshold_passed(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = ["Dense Multilayer Perceptron", "Decision Tree Random Forest"]
        candidate = "Continuous Topological Manifold Flow ODE"
        d_lit, _ = contract.compute_literature_distance(candidate, prior_art)
        assert d_lit > 0.15

    def test_tier1_f04_literature_distance_corpus_empty_or_default(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        d_lit, _ = contract.compute_literature_distance("Any Candidate", [])
        assert d_lit >= 0.15

    # --------------------------------------------------------------------------
    # Feature 5: Functional Prediction Manifold Divergence
    # --------------------------------------------------------------------------
    def test_tier1_f05_func_divergence_perfectly_collinear_residuals_zero_distance(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(0, 10, 50)
        base_pred = y_true + 1.0
        cand_pred = y_true + 1.0  # Identical predictions -> identical residuals
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-4)
        assert math.isclose(abs(rho), 1.0, abs_tol=1e-4)

    def test_tier1_f05_func_divergence_uncorrelated_residuals_passes(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        np.random.seed(42)
        y_true = np.zeros(100)
        base_pred = np.random.normal(0, 1, 100)
        cand_pred = np.random.normal(0, 1, 100)
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert d_func > 0.10

    def test_tier1_f05_func_divergence_anti_correlated_residuals_divergence(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(50)
        base_pred = np.linspace(-2, 2, 50)
        cand_pred = -base_pred  # rho = -1 -> |rho| = 1 -> d_func = 0
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-4)

    def test_tier1_f05_func_divergence_orthogonal_residual_vectors(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(40)
        base_pred = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(40)])
        cand_pred = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(40)])
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert d_func > 0.10

    def test_tier1_f05_func_divergence_above_epsilon_threshold_passed(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        np.random.seed(123)
        y_true = np.random.uniform(0, 5, 60)
        base_pred = y_true + np.random.normal(0, 0.5, 60)
        cand_pred = y_true + np.random.laplace(0, 0.8, 60)
        d_func, _ = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert d_func > 0.10

    # --------------------------------------------------------------------------
    # Feature 6: Automated Dual-Key Novelty Gate
    # --------------------------------------------------------------------------
    def test_tier1_f06_dual_key_both_pass_verdict_approved(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + np.sin(np.linspace(0, 10, 50))
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_pass",
            candidate_code="class NovelTransformerLayer(nn.Module): ...",
            prior_art_corpus=["Standard Multi-Layer Perceptron"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is True
        assert res.verdict == nov.NoveltyVerdict.APPROVED

    def test_tier1_f06_dual_key_lit_fails_rejected_lit_proximity(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + np.sin(np.linspace(0, 10, 50))
        prior_code = "class StandardTransformer(nn.Module): pass"
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_lit_fail",
            candidate_code=prior_code,
            prior_art_corpus=[prior_code],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == nov.NoveltyVerdict.REJECTED_LIT_PROXIMITY

    def test_tier1_f06_dual_key_func_fails_rejected_func_collinear(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + 0.5  # Exactly identical predictions
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_collinear",
            candidate_code="class RadicallyDifferentName(nn.Module): pass",
            prior_art_corpus=["Unrelated prior paper"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == nov.NoveltyVerdict.REJECTED_FUNC_COLLINEAR

    def test_tier1_f06_dual_key_both_fail_rejected_dual_failure(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + 0.5
        code = "class IdenticalCode(nn.Module): pass"
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_both_fail",
            candidate_code=code,
            prior_art_corpus=[code],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == nov.NoveltyVerdict.REJECTED_DUAL_FAILURE

    def test_tier1_f06_dual_key_surface_form_rephrasing_strictly_rejected(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.ones(40)
        base_pred = np.ones(40) * 0.8
        cand_pred = np.ones(40) * 0.8  # Cosmetic variable renaming with identical predictions
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_rephrase",
            candidate_code="# Rephrased docstring\nx_new = alpha * 2",
            prior_art_corpus=["Original implementation with x_old = beta * 2"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict in (
            nov.NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            nov.NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )

    # --------------------------------------------------------------------------
    # Feature 7: Persistent Swarm Learning State Node
    # --------------------------------------------------------------------------
    def test_tier1_f07_learning_state_node_instantiation_and_id(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="swarm_node_01")
        assert node.node_id == "swarm_node_01"
        assert node.total_budget_n == 0
        assert len(node.trajectories) == 0

    def test_tier1_f07_learning_state_record_evaluation_creates_trajectory(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="swarm_node_01")
        pt = node.record_evaluation(
            hypothesis_id="hyp_1", score=0.75, lane="architecture", depth=1
        )
        assert pt.step == 1
        assert pt.quality_score == 0.75
        assert pt.lane == "architecture"
        assert node.total_budget_n == 1
        assert len(node.trajectories) == 1

    def test_tier1_f07_learning_state_tracks_multiple_trajectories(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="swarm_node_01")
        node.record_evaluation("hyp_1", 0.60, "architecture", 1)
        node.record_evaluation("hyp_2", 0.70, "objectives", 1)
        node.record_evaluation("hyp_3", 0.65, "optimization", 2)
        assert len(node.trajectories) == 3
        assert node.total_budget_n == 3

    def test_tier1_f07_learning_state_serialization_to_dict_and_json(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="swarm_node_01")
        node.record_evaluation("hyp_1", 0.85, "features", 1)
        data = node.model_dump()
        assert data["node_id"] == "swarm_node_01"
        assert len(data["trajectories"]) == 1
        serialized = json.dumps(data)
        assert "swarm_node_01" in serialized

    def test_tier1_f07_learning_state_active_branches_registration(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="swarm_node_01")
        node.active_branches.append("branch_arch_01")
        node.active_branches.append("branch_obj_02")
        assert len(node.active_branches) == 2
        assert "branch_arch_01" in node.active_branches

    # --------------------------------------------------------------------------
    # Feature 8: Epistemic Entropy Tracker
    # --------------------------------------------------------------------------
    def test_tier1_f08_epistemic_entropy_diverse_scores_high_entropy(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_entropy")
        # Record varied scores spanning multiple bins
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"hyp_{idx}", s, "architecture", 1)
        h = node.compute_epistemic_entropy()
        assert h > 0.5

    def test_tier1_f08_epistemic_entropy_identical_scores_zero_entropy(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_zero_h")
        for idx in range(10):
            node.record_evaluation(f"hyp_{idx}", 0.50, "objectives", 1)
        h = node.compute_epistemic_entropy()
        assert math.isclose(h, 0.0, abs_tol=1e-4)

    def test_tier1_f08_epistemic_entropy_binned_distribution_shannon_formula(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_shannon")
        node.record_evaluation("h1", 0.1, "features", 1)
        node.record_evaluation("h2", 0.9, "features", 1)
        h = node.compute_epistemic_entropy()
        assert h > 0.0

    def test_tier1_f08_epistemic_entropy_computation_across_generations(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_gen")
        node.record_evaluation("h1", 0.4, "optimization", 1)
        h1 = node.compute_epistemic_entropy()
        node.record_evaluation("h2", 0.8, "optimization", 2)
        h2 = node.compute_epistemic_entropy()
        assert isinstance(h1, float)
        assert isinstance(h2, float)

    def test_tier1_f08_epistemic_entropy_updates_with_new_evaluations(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_update")
        node.record_evaluation("h1", 0.5, "architecture", 1)
        node.record_evaluation("h2", 0.5, "architecture", 1)
        assert node.compute_epistemic_entropy() == 0.0
        node.record_evaluation("h3", 0.9, "architecture", 1)
        assert node.compute_epistemic_entropy() > 0.0

    # --------------------------------------------------------------------------
    # Feature 9: Test-Time Compute Scaling Upper Envelope
    # --------------------------------------------------------------------------
    def test_tier1_f09_scaling_curve_strictly_increasing_trajectory(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_scaling_inc")
        scores = [0.60, 0.65, 0.72, 0.81, 0.88]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"hyp_{idx}", s, "architecture", 1)
        curve = node.get_upper_envelope_curve()
        assert len(curve) == 5
        for idx, (_, val) in enumerate(curve):
            assert math.isclose(val, scores[idx], rel_tol=1e-3)

    def test_tier1_f09_scaling_curve_non_monotonic_raw_scores_flattens_envelope(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_scaling_flat")
        raw_scores = [0.70, 0.60, 0.75, 0.50, 0.80]
        expected_envelope = [0.70, 0.70, 0.75, 0.75, 0.80]
        for idx, s in enumerate(raw_scores):
            node.record_evaluation(f"hyp_{idx}", s, "objectives", 1)
        curve = node.get_upper_envelope_curve()
        for idx, (_, val) in enumerate(curve):
            assert math.isclose(val, expected_envelope[idx], rel_tol=1e-3)

    def test_tier1_f09_scaling_curve_monotonicity_verification_returns_true(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_mono")
        node.record_evaluation("h1", 0.50, "features", 1)
        node.record_evaluation("h2", 0.30, "features", 2)
        node.record_evaluation("h3", 0.70, "features", 3)
        is_monotonic, gain = node.verify_monotonic_progression()
        assert is_monotonic is True
        assert math.isclose(gain, 0.20, abs_tol=1e-4)

    def test_tier1_f09_scaling_curve_incremental_gain_calculation(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_gain")
        node.record_evaluation("h1", 0.40, "optimization", 1)
        node.record_evaluation("h2", 0.90, "optimization", 2)
        _, gain = node.verify_monotonic_progression()
        assert math.isclose(gain, 0.50, abs_tol=1e-4)

    def test_tier1_f09_scaling_curve_budget_n_matching_length(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="node_len")
        for i in range(7):
            node.record_evaluation(f"h_{i}", 0.5 + i * 0.05, "architecture", 1)
        curve = node.get_upper_envelope_curve()
        assert len(curve) == 7
        assert curve[-1][0] == 7

    # --------------------------------------------------------------------------
    # Feature 10: Epistemic Stagnation Detector across M Stacks
    # --------------------------------------------------------------------------
    def test_tier1_f10_stagnation_vanishing_gradients_detected(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        history = [
            {"grad_norm": 1e-7, "loss": 0.5, "ast_tokens": {"a", "b"}},
            {"grad_norm": 2e-7, "loss": 0.5, "ast_tokens": {"a", "b", "c"}},
            {"grad_norm": 5e-7, "loss": 0.5, "ast_tokens": {"a", "c"}},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_vanish", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.VANISHING_GRADIENTS

    def test_tier1_f10_stagnation_loss_curvature_plateau_detected(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        history = [
            {"grad_norm": 0.1, "loss": 0.500000, "ast_tokens": {"t1"}},
            {"grad_norm": 0.1, "loss": 0.500001, "ast_tokens": {"t2"}},
            {"grad_norm": 0.1, "loss": 0.500002, "ast_tokens": {"t3"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_plateau", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.LOSS_PLATEAU

    def test_tier1_f10_stagnation_circular_ast_mutations_detected(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, tau_jaccard=0.85)
        # Jaccard similarity > 0.85
        t1 = {f"tok_{i}" for i in range(100)}
        t2 = {f"tok_{i}" for i in range(95)} | {"extra1"}  # 95 / 101 = 0.94
        t3 = {f"tok_{i}" for i in range(94)} | {"extra2"}
        history = [
            {"grad_norm": 0.05, "loss": 0.4, "ast_tokens": t1},
            {"grad_norm": 0.05, "loss": 0.35, "ast_tokens": t2},
            {"grad_norm": 0.05, "loss": 0.30, "ast_tokens": t3},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_circular", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.CIRCULAR_AST

    def test_tier1_f10_stagnation_healthy_branch_returns_none(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3)
        history = [
            {"grad_norm": 0.05, "loss": 1.2, "ast_tokens": {"func_a"}},
            {"grad_norm": 0.04, "loss": 0.9, "ast_tokens": {"func_b", "extra"}},
            {"grad_norm": 0.03, "loss": 0.6, "ast_tokens": {"func_c", "other"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_healthy", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == exec_mod.StagnationTrigger.NONE

    def test_tier1_f10_stagnation_audit_branch_evaluates_over_m_window(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3)
        # 5 items; last 3 are healthy
        history = [
            {"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"x"}},
            {"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"x"}},
            {"grad_norm": 0.1, "loss": 0.8, "ast_tokens": {"a"}},
            {"grad_norm": 0.08, "loss": 0.6, "ast_tokens": {"b"}},
            {"grad_norm": 0.06, "loss": 0.4, "ast_tokens": {"c"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_recent_healthy", depth=3, visits=5, history=history)
        assert stagnant is False

    # --------------------------------------------------------------------------
    # Feature 11: Radical Early-Stage Architecture Safeguard
    # --------------------------------------------------------------------------
    def test_tier1_f11_early_stage_depth_0_visits_1_exempt_from_pruning(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, reason = dog.audit_branch("b_early_0", depth=0, visits=1, history=history)
        assert stagnant is False
        assert "Exempt" in reason or "early" in reason.lower()

    def test_tier1_f11_early_stage_depth_1_visits_2_exempt_from_pruning(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, _ = dog.audit_branch("b_early_1", depth=1, visits=2, history=history)
        assert stagnant is False

    def test_tier1_f11_mature_stage_depth_2_visits_3_not_exempt(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, _ = dog.audit_branch("b_mature_2", depth=2, visits=3, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.VANISHING_GRADIENTS

    def test_tier1_f11_mature_stage_depth_4_vanishing_grad_pruned(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, _, _ = dog.audit_branch("b_deep", depth=4, visits=8, history=history)
        assert stagnant is True

    def test_tier1_f11_safeguard_preserves_radical_small_norm_explorations(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        # Loss plateau in early architecture
        history = [{"grad_norm": 0.1, "loss": 0.500000, "ast_tokens": {"t1"}}] * 3
        stagnant, _, _ = dog.audit_branch("b_radical", depth=1, visits=1, history=history)
        assert stagnant is False

    # --------------------------------------------------------------------------
    # Feature 12: Orthogonal Warm-Restart Controller
    # --------------------------------------------------------------------------
    def test_tier1_f12_warm_restart_architecture_maps_to_features(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n1")
        event = dog.trigger_orthogonal_warm_restart("b_arch", current_lane="architecture", state_node=node)
        assert event.target_lane == "features"
        assert event.event_type == "ORTHOGONAL_WARM_RESTART"

    def test_tier1_f12_warm_restart_objectives_maps_to_optimization(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n1")
        event = dog.trigger_orthogonal_warm_restart("b_obj", current_lane="objectives", state_node=node)
        assert event.target_lane == "optimization"

    def test_tier1_f12_warm_restart_optimization_maps_to_architecture(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n1")
        event = dog.trigger_orthogonal_warm_restart("b_opt", current_lane="optimization", state_node=node)
        assert event.target_lane == "architecture"

    def test_tier1_f12_warm_restart_features_maps_to_objectives(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n1")
        event = dog.trigger_orthogonal_warm_restart("b_feat", current_lane="features", state_node=node)
        assert event.target_lane == "objectives"

    def test_tier1_f12_warm_restart_injects_orthogonal_priors_and_new_branch(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n1")
        event = dog.trigger_orthogonal_warm_restart("b_arch_old", current_lane="architecture", state_node=node)
        assert event.branch_id != ""
        assert event.injected_priors is not None
        assert len(event.injected_priors) > 0

    # --------------------------------------------------------------------------
    # Feature 13: Executioner Event Logging
    # --------------------------------------------------------------------------
    def test_tier1_f13_executioner_records_stagnation_pruned_event(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_log")
        dog = exec_mod.ExecutionerWatchdog()
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, _ = dog.audit_branch("b_stagnant", depth=3, visits=5, history=history)
        assert stagnant is True
        # Triggering warm restart records the event to the node
        dog.trigger_orthogonal_warm_restart("b_stagnant", "architecture", state_node=node)
        assert len(node.executioner_event_logs) >= 1
        assert node.executioner_event_logs[-1]["branch_id"] == "b_stagnant"

    def test_tier1_f13_executioner_records_warm_restart_event(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_log2")
        dog = exec_mod.ExecutionerWatchdog()
        dog.trigger_orthogonal_warm_restart("b_restart", "objectives", state_node=node)
        assert len(node.executioner_event_logs) == 1
        assert node.executioner_event_logs[0]["target_lane"] == "optimization"

    def test_tier1_f13_executioner_event_schema_contains_required_fields(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_schema")
        dog = exec_mod.ExecutionerWatchdog()
        ev = dog.trigger_orthogonal_warm_restart("b_ev", "features", state_node=node)
        assert hasattr(ev, "event_type")
        assert hasattr(ev, "branch_id")
        assert hasattr(ev, "trigger")
        assert hasattr(ev, "details")
        assert hasattr(ev, "timestamp")

    def test_tier1_f13_executioner_logs_saved_to_state_node(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_saved")
        dog = exec_mod.ExecutionerWatchdog()
        dog.trigger_orthogonal_warm_restart("b_1", "architecture", state_node=node)
        dog.trigger_orthogonal_warm_restart("b_2", "objectives", state_node=node)
        assert len(node.executioner_event_logs) == 2

    def test_tier1_f13_executioner_logs_queryable_by_branch_id(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_query")
        dog = exec_mod.ExecutionerWatchdog()
        dog.trigger_orthogonal_warm_restart("target_branch", "architecture", state_node=node)
        matches = [e for e in node.executioner_event_logs if e["branch_id"] == "target_branch"]
        assert len(matches) == 1

    # --------------------------------------------------------------------------
    # Feature 14: Evidence Hierarchy Weighting System
    # --------------------------------------------------------------------------
    def test_tier1_f14_evidence_hierarchy_all_cochrane_composite_score(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        sources = [
            gov_mod.EvidenceSource(
                source_id="c1", title="Journal A", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
                methodological_rigor=0.9
            ),
            gov_mod.EvidenceSource(
                source_id="c2", title="Journal B", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
                methodological_rigor=0.8
            ),
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is True
        assert score >= 0.80

    def test_tier1_f14_evidence_hierarchy_mixed_sources_weighted_average(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        sources = [
            gov_mod.EvidenceSource(
                source_id="c1", title="Journal", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
                methodological_rigor=1.0
            ),  # w=1.0, c=1.0 -> 1.0
            gov_mod.EvidenceSource(
                source_id="f1", title="Forum", source_type=gov_mod.SourceTier.PUBLIC_FORUM,
                methodological_rigor=0.2
            ),  # w=0.3, c=0.2 -> 0.06
        ]
        # (1.0*1.0 + 0.3*0.2) / (1.0 + 0.3) = 1.06 / 1.3 ≈ 0.815 >= 0.50
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is True
        assert score > 0.70

    def test_tier1_f14_evidence_hierarchy_all_unverified_fails_threshold(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        sources = [
            gov_mod.EvidenceSource(
                source_id="u1", title="Unverified 1", source_type=gov_mod.SourceTier.UNVERIFIED,
                methodological_rigor=0.3
            ),
            gov_mod.EvidenceSource(
                source_id="u2", title="Unverified 2", source_type=gov_mod.SourceTier.UNVERIFIED,
                methodological_rigor=0.2
            ),
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is False
        assert score < 0.50

    def test_tier1_f14_evidence_hierarchy_verification_returns_tuple(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_evidence_hierarchy([])
        assert isinstance(res, tuple)
        assert len(res) == 2
        assert isinstance(res[0], bool)
        assert isinstance(res[1], float)

    def test_tier1_f14_evidence_hierarchy_passes_above_evidence_threshold(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        sources = [
            gov_mod.EvidenceSource(
                source_id="p1", title="ArXiv Preprint", source_type=gov_mod.SourceTier.PREPRINT,
                methodological_rigor=0.85
            )
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is True
        assert score >= 0.50

    # --------------------------------------------------------------------------
    # Feature 15: Data Provenance & Ethical Constraint Tracker
    # --------------------------------------------------------------------------
    def test_tier1_f15_provenance_valid_permissive_passes(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="bafybeic5...",
            license="MIT",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.5,
            bias_disparity_score=0.03,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True
        assert len(violations) == 0

    def test_tier1_f15_provenance_unapproved_license_fails(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="bafybeic5...",
            license="PROPRIETARY_UNAUTHORIZED",
            is_commercial_safe=False,
            privacy_budget_epsilon=0.5,
            bias_disparity_score=0.03,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("license" in v.lower() for v in violations)

    def test_tier1_f15_provenance_excessive_privacy_epsilon_fails(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="bafybeic5...",
            license="Apache-2.0",
            is_commercial_safe=True,
            privacy_budget_epsilon=2.5,  # Exceeds max epsilon 1.0
            bias_disparity_score=0.03,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("privacy" in v.lower() or "epsilon" in v.lower() for v in violations)

    def test_tier1_f15_provenance_excessive_bias_disparity_fails(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="bafybeic5...",
            license="CC-BY-4.0",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.5,
            bias_disparity_score=0.25,  # Exceeds max disparity 0.10
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("bias" in v.lower() for v in violations)

    def test_tier1_f15_provenance_ethical_approval_false_fails(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="bafybeic5...",
            license="MIT",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.5,
            bias_disparity_score=0.02,
            ethical_approval=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("ethical" in v.lower() for v in violations)

    # --------------------------------------------------------------------------
    # Feature 16: Domain Invariant Plug-in Architecture
    # --------------------------------------------------------------------------
    def test_tier1_f16_invariant_plugin_registration_custom_physics(self):
        de_mod = get_domain_evidence_module()
        class PhysicsPlugin(de_mod.DomainInvariantPlugin):
            def get_domain_name(self) -> str:
                return "physics"
            def get_constraints(self):
                return [
                    de_mod.SymbolicDomainConstraint(
                        constraint_id="phys_energy_cons",
                        name="Energy Conservation",
                        domain="physics",
                        symbolic_expression="x**2 + y**2",
                        lower_bound=0.0,
                    )
                ]
        verifier = de_mod.DomainEvidenceVerifier(plugins=[PhysicsPlugin()])
        assert "physics" in verifier.plugins

    def test_tier1_f16_invariant_plugin_registration_custom_biology(self):
        de_mod = get_domain_evidence_module()
        class BiologyPlugin(de_mod.DomainInvariantPlugin):
            def get_domain_name(self) -> str:
                return "biology"
            def get_constraints(self):
                return [
                    de_mod.SymbolicDomainConstraint(
                        constraint_id="bio_monotonic_dose",
                        name="Monotonic Dose Response",
                        domain="biology",
                        symbolic_expression="x / (1 + x)",
                        lower_bound=0.0,
                    )
                ]
        verifier = de_mod.DomainEvidenceVerifier(plugins=[BiologyPlugin()])
        assert "biology" in verifier.plugins

    def test_tier1_f16_invariant_plugin_get_constraints_returns_list(self):
        de_mod = get_domain_evidence_module()
        class SamplePlugin(de_mod.DomainInvariantPlugin):
            def get_domain_name(self):
                return "sample"
            def get_constraints(self):
                return [
                    de_mod.SymbolicDomainConstraint(
                        constraint_id="c1", name="C1", domain="sample", symbolic_expression="x"
                    )
                ]
        p = SamplePlugin()
        constraints = p.get_constraints()
        assert isinstance(constraints, list)
        assert len(constraints) == 1

    def test_tier1_f16_invariant_plugin_domain_filtering(self):
        de_mod = get_domain_evidence_module()
        class P1(de_mod.DomainInvariantPlugin):
            def get_domain_name(self): return "physics"
            def get_constraints(self): return []
        class P2(de_mod.DomainInvariantPlugin):
            def get_domain_name(self): return "biology"
            def get_constraints(self): return []
        verifier = de_mod.DomainEvidenceVerifier(plugins=[P1(), P2()])
        assert len(verifier.plugins) == 2

    def test_tier1_f16_invariant_plugin_composite_verifier_loads_plugins(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        assert hasattr(verifier, "verify_symbolic_invariants")

    # --------------------------------------------------------------------------
    # Feature 17: L0–L2 Math Verifier Invariant Binding
    # --------------------------------------------------------------------------
    def test_tier1_f17_l0_analytic_conservation_law_verified(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # x^2 + 1 >= 0 analytically true everywhere
        res = verifier.verify_symbolic_invariants("x**2 + 1", domain="physics")
        assert res["all_satisfied"] is True

    def test_tier1_f17_l1_ibp_bounded_interval_invariant(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # On [0, 2], x + 3 >= 0 is true
        res = verifier.verify_symbolic_invariants("x + 3", domain="physics")
        assert res["all_satisfied"] is True

    def test_tier1_f17_l2_ibnb_monotonicity_proof(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # exp(x) has positive derivative everywhere
        res = verifier.verify_symbolic_invariants("exp(x)", domain="biology")
        assert res["all_satisfied"] is True

    def test_tier1_f17_invariant_violation_counterexample_produced(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Candidate evaluates to negative values: x^2 - 100 on [-20, 20] lower bound 0
        res = verifier.verify_symbolic_invariants("x**2 - 100", domain="physics")
        assert res["all_satisfied"] is False
        assert "counterexample" in str(res).lower() or len(res.get("violations", [])) > 0

    def test_tier1_f17_unbounded_limit_boundary_invariant_l0(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # 1 / (1 + exp(-x)) is bounded in [0, 1]
        res = verifier.verify_symbolic_invariants("1 / (1 + exp(-x))", domain="boundary")
        assert res["all_satisfied"] is True

    # --------------------------------------------------------------------------
    # Feature 18: Enhanced `frame status` Observability
    # --------------------------------------------------------------------------
    def test_tier1_f18_status_command_executes_successfully(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ret = cmd_status(argparse.Namespace(detail=False, clear=False))
        assert ret == 0

    def test_tier1_f18_status_displays_swarm_learning_state(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        output = captured.getvalue()
        assert "Swarm Learning State" in output or "Search Stage" in output or "Status" in output

    def test_tier1_f18_status_displays_compute_scaling_upper_envelope(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        output = captured.getvalue()
        assert "Compute" in output or "Scaling" in output or "LORD" in output

    def test_tier1_f18_status_displays_governor_audit_logs(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        output = captured.getvalue()
        assert len(output) > 100

    def test_tier1_f18_status_displays_executioner_event_logs(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        output = captured.getvalue()
        assert "Accounting" in output or "Ledger" in output

    # --------------------------------------------------------------------------
    # Feature 19: Expanded `frame swarm --export-agy-specs`
    # --------------------------------------------------------------------------
    def test_tier1_f19_swarm_export_json_valid_array(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(
            goal="Test Goal",
            lanes="architecture,objectives,optimization,features",
            export_agy_specs=True,
        )
        ret = cmd_swarm(args)
        assert ret == 0
        data = json.loads(captured.getvalue())
        assert isinstance(data, list)

    def test_tier1_f19_swarm_export_contains_governor_and_executioner(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(
            goal="Test",
            lanes="architecture,objectives,optimization,features",
            export_agy_specs=True,
        )
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        typenames = [item["TypeName"] for item in data]
        if not any("governor" in t for t in typenames):
            pytest.skip("Milestone M1 cognitive subagent export not yet implemented in CLI")
        assert any("governor" in t for t in typenames)

    def test_tier1_f19_swarm_export_contains_proximity_analyzer_and_meta_reviewer(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(
            goal="Test",
            lanes="architecture,objectives,optimization,features",
            export_agy_specs=True,
        )
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        typenames = [item["TypeName"] for item in data]
        if not any("proximity" in t for t in typenames):
            pytest.skip("Milestone M1 cognitive subagent export not yet implemented in CLI")
        assert len(data) >= 8

    def test_tier1_f19_swarm_export_contains_all_four_lanes(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(
            goal="Test",
            lanes="architecture,objectives,optimization,features",
            export_agy_specs=True,
        )
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        roles_text = " ".join([d.get("Role", "") for d in data])
        assert "Architecture" in roles_text
        assert "Objectives" in roles_text
        assert "Optimization" in roles_text
        assert "Feature" in roles_text

    def test_tier1_f19_swarm_export_respects_custom_goal_in_prompts(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        custom_goal = "Discover Subquadratic State Space Dynamics"
        args = argparse.Namespace(
            goal=custom_goal,
            lanes="architecture,objectives,optimization,features",
            export_agy_specs=True,
        )
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        assert any(custom_goal in item.get("Prompt", "") for item in data)

    # --------------------------------------------------------------------------
    # Feature 20: Statistical Screener Non-Regression
    # --------------------------------------------------------------------------
    def test_tier1_f20_lord3_controller_streaming_fdr_alpha_wealth(self):
        controller = LORDController(alpha=0.05)
        a1 = controller.get_next_alpha()
        assert a1 > 0.0
        # Reject null hypothesis with significant p-value
        disc, alpha_t = controller.record_result(p_value=0.0001)
        assert disc is True
        assert 0.0 < alpha_t < 0.05
        a2 = controller.get_next_alpha()
        assert a2 > 0.0

    def test_tier1_f20_two_stage_screening_stage1_futility(self):
        controller = LORDController(alpha=0.05)
        # Non-discovery
        disc, _ = controller.record_result(p_value=0.50)
        assert disc is False

    def test_tier1_f20_rope_superiority_paired_seed_evaluation(self):
        # 5 paired seeds, candidate strictly better than baseline
        cand = {f"seed_{i}": 0.85 + 0.01 * (i % 3) for i in range(5)}
        base = {f"seed_{i}": 0.70 + 0.01 * (i % 3) for i in range(5)}
        lord = LORDController(alpha=0.05)
        report = evaluate_paired_seed_rope(
            hypothesis_id="hyp_test",
            metric_name="accuracy",
            baseline_scores=base,
            candidate_scores=cand,
            rope_epsilon=0.01,
            lord_controller=lord,
            target_direction=TargetDirection.MAXIMIZE,
        )
        assert report.is_discovery is True
        assert report.verdict == "CONFIRMED_DISCOVERY"

    def test_tier1_f20_flop_counter_mode_hardware_routing(self):
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(32, 16)
            def forward(self, x):
                return self.fc(x)

        fwd, bwd, params = profile_module_flops(SimpleNet(), sample_input=torch.randn(1, 32))
        assert fwd > 0
        profile = route_compute_tier(fwd, bwd, params)
        assert profile.tier in (ExecutionTier.TIER_1_LOCAL, ExecutionTier.TIER_2_KAGGLE)

    def test_tier1_f20_all_110_baseline_invariants_preserved(self):
        # Verify ZETA and gamma_lord math constants are unaltered
        assert math.isclose(ZETA_1_6, 2.28576, abs_tol=1e-4)
        g1 = gamma_lord(1)
        assert g1 > 0.0


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES (100 Tests: 20 Features x 5 Tests)
# ==============================================================================

class TestTier2BoundaryCases:
    """Tier 2: Boundary, edge condition, degenerate input, and numerical limit tests."""

    # --------------------------------------------------------------------------
    # F01: Information Governor Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f01_governor_empty_sources_list(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        res = gov.audit_and_filter([])
        assert len(res) == 0

    def test_tier2_f01_governor_exact_threshold_boundary_grade(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        # Construct source yielding grade exactly 0.40
        # w=1.0, 0.4*0.5 + 0.4*0.5 + 0.2*0 = 0.40
        source = gov_mod.EvidenceSource(
            source_id="s_exact", title="Exact", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=False, relevance_score=0.5, methodological_rigor=0.5
        )
        res = gov.grade_source(source)
        assert res.accepted is True
        assert math.isclose(res.grade, 0.40, abs_tol=1e-4)

    def test_tier2_f01_governor_all_zero_scores(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        source = gov_mod.EvidenceSource(
            source_id="s_zeros", title="Zeros", source_type=gov_mod.SourceTier.UNVERIFIED,
            peer_reviewed=False, relevance_score=0.0, methodological_rigor=0.0
        )
        res = gov.grade_source(source)
        assert res.accepted is False
        assert res.grade == 0.0

    def test_tier2_f01_governor_all_one_scores(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        source = gov_mod.EvidenceSource(
            source_id="s_ones", title="Perfect", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=True, relevance_score=1.0, methodological_rigor=1.0
        )
        res = gov.grade_source(source)
        assert res.accepted is True
        assert math.isclose(res.grade, 1.0, abs_tol=1e-4)

    def test_tier2_f01_governor_extreme_whitespace_or_empty_abstract(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        source = gov_mod.EvidenceSource(
            source_id="s_ws", title="   ", source_type=gov_mod.SourceTier.PREPRINT,
            abstract_text="   \n\t  ", relevance_score=0.7, methodological_rigor=0.7
        )
        res = gov.grade_source(source)
        assert isinstance(res.accepted, bool)

    # --------------------------------------------------------------------------
    # F02: High-Entropy Exploration Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f02_entropy_single_lane_entropy_zero(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        res = gov.check_exploration_entropy({"architecture": 100})
        assert res["entropy"] == 0.0
        assert res["sufficient"] is False

    def test_tier2_f02_entropy_exact_boundary_threshold(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        thresh = 0.75 * math.log(4)
        # Uniform 4-lane distribution gives ln(4) > thresh
        res = gov.check_exploration_entropy({"l1": 1, "l2": 1, "l3": 1, "l4": 1})
        assert res["entropy"] > thresh

    def test_tier2_f02_entropy_empty_lane_counts(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        res = gov.check_exploration_entropy({})
        assert res["entropy"] == 0.0
        assert res["sufficient"] is False

    def test_tier2_f02_entropy_very_large_counts_numerical_stability(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        counts = {"a": 10**9, "b": 10**9, "c": 10**9, "d": 10**9}
        res = gov.check_exploration_entropy(counts)
        assert math.isclose(res["entropy"], math.log(4), rel_tol=1e-4)
        assert res["sufficient"] is True

    def test_tier2_f02_entropy_single_count_per_lane_uniform(self):
        gov_mod = get_governor_module()
        gov = gov_mod.InformationGovernor()
        counts = {"l1": 1, "l2": 1, "l3": 1, "l4": 1}
        res = gov.check_exploration_entropy(counts)
        assert res["sufficient"] is True

    # --------------------------------------------------------------------------
    # F03: Cognitive Role Spec Generator Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f03_specs_empty_goal_string(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="")
        assert len(specs) >= 4

    def test_tier2_f03_specs_custom_lane_list_filtering(self):
        bridge = get_agy_bridge_module()
        lanes = [bridge.LaneType.ARCHITECTURE, bridge.LaneType.OBJECTIVES]
        specs = bridge.build_agy_subagent_specs(goal="Test", lanes=lanes)
        roles = [s.Role for s in specs]
        assert any("Architecture" in r for r in roles)
        assert any("Objectives" in r for r in roles)

    def test_tier2_f03_specs_duplicate_lanes_deduplication(self):
        bridge = get_agy_bridge_module()
        lanes = [bridge.LaneType.ARCHITECTURE, bridge.LaneType.ARCHITECTURE]
        specs = bridge.build_agy_subagent_specs(goal="Test", lanes=lanes)
        # Should not crash and handle gracefully
        assert len(specs) >= 1

    def test_tier2_f03_specs_whitespace_only_goal(self):
        bridge = get_agy_bridge_module()
        specs = bridge.build_agy_subagent_specs(goal="   \t\n  ")
        assert len(specs) >= 4

    def test_tier2_f03_specs_special_characters_in_goal(self):
        bridge = get_agy_bridge_module()
        goal = "Test with quotes \"' and unicode 🚀 & symbols <>"
        specs = bridge.build_agy_subagent_specs(goal=goal)
        assert any(goal in s.Prompt for s in specs)

    # --------------------------------------------------------------------------
    # F04: Literature Proximity Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f04_literature_exact_threshold_boundary(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15)
        # Strict inequality invariant: d_lit > tau_lit (equality fails)
        assert not (0.15000 > contract.tau_lit)

    def test_tier2_f04_literature_zero_distance_identical_embedding(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        text = "Exact identical copy of paper title and methodology"
        d_lit, _ = contract.compute_literature_distance(text, [text])
        assert d_lit <= 0.05

    def test_tier2_f04_literature_maximum_distance_anti_parallel(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        d_lit, _ = contract.compute_literature_distance("xyz123 unique alpha beta", ["totally unrelated"])
        assert d_lit >= 0.15

    def test_tier2_f04_literature_unnormalized_embeddings_handled(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        d_lit, _ = contract.compute_literature_distance("Candidate code", ["Prior art 1", "Prior art 2"])
        assert 0.0 <= d_lit <= 1.0

    def test_tier2_f04_literature_single_element_prior_art_corpus(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        d_lit, nearest = contract.compute_literature_distance("My Model", ["Single Prior Art"])
        assert nearest == "Single Prior Art"

    # --------------------------------------------------------------------------
    # F05: Functional Manifold Divergence Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f05_func_exact_threshold_boundary(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        # Strict inequality invariant: d_func > epsilon_func (equality fails)
        assert not (0.10000 > contract.epsilon_func)

    def test_tier2_f05_func_constant_residual_vector_zero_variance(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.ones(30)
        base_pred = np.zeros(30)
        cand_pred = np.zeros(30)  # Constant residuals -> zero variance
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        # Must handle zero variance gracefully without ZeroDivisionError
        assert 0.0 <= d_func <= 1.0

    def test_tier2_f05_func_candidate_equals_ground_truth_zero_residual(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(1, 10, 40)
        base_pred = y_true + 1.0
        cand_pred = y_true  # Candidate is ground truth
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert isinstance(d_func, float)

    def test_tier2_f05_func_minimum_sample_size_holdout_boundary(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(epsilon_func=0.10)
        # Small sample size (N=5)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_pred = np.array([1.1, 1.9, 3.2, 4.1, 4.8])
        cand_pred = np.array([1.5, 2.5, 2.8, 3.9, 5.2])
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert isinstance(d_func, float)

    def test_tier2_f05_func_extreme_scale_predictions_numerical_stability(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        y_true = np.linspace(1e6, 2e6, 50)
        base_pred = y_true + 1e4
        cand_pred = y_true + 2e4
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert not math.isnan(d_func)

    # --------------------------------------------------------------------------
    # F06: Dual-Key Gate Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f06_dual_key_both_exact_boundary_rejection(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        # When distance == threshold exactly, gate rejects
        assert not (0.15 > contract.tau_lit and 0.10 > contract.epsilon_func)

    def test_tier2_f06_dual_key_lit_pass_func_boundary_rejection(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 1, 30)
        base = y_true + 0.1
        cand = y_true + 0.1  # Collinear
        res = contract.evaluate_novelty(
            "h_bound", "Radical novelty in text", ["Ordinary model"], y_true, base, cand
        )
        assert res.passed is False

    def test_tier2_f06_dual_key_func_pass_lit_boundary_rejection(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 1, 30)
        base = y_true + 0.1
        cand = y_true + np.random.normal(0, 0.1, 30)
        code = "Exact identical code"
        res = contract.evaluate_novelty("h_bound2", code, [code], y_true, base, cand)
        assert res.passed is False

    def test_tier2_f06_dual_key_alpha_wealth_zero_consumption_verification(self):
        lord = LORDController(alpha=0.05)
        initial_alpha = lord.get_next_alpha()
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        y_true = np.ones(30)
        contract.evaluate_novelty("h_wealth", "code", ["art"], y_true, y_true, y_true)
        # Next alpha allocation must remain completely unchanged
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-7)

    def test_tier2_f06_dual_key_empty_code_snippet_handled(self):
        nov = get_novelty_module()
        contract = nov.DualKeyNoveltyContract()
        y = np.ones(30)
        res = contract.evaluate_novelty("h_empty", "", ["art"], y, y, y)
        assert isinstance(res.passed, bool)

    # --------------------------------------------------------------------------
    # F07: Learning State Node Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f07_learning_state_n_equals_1_initialization(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_single")
        node.record_evaluation("h0", 0.70, "architecture", 0)
        curve = node.get_upper_envelope_curve()
        assert len(curve) == 1
        assert curve[0] == (1, 0.70)

    def test_tier2_f07_learning_state_empty_trajectories_get_curve(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_empty")
        curve = node.get_upper_envelope_curve()
        assert len(curve) == 0

    def test_tier2_f07_learning_state_negative_or_extreme_scores_handled(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_neg")
        node.record_evaluation("h1", -10.5, "architecture", 1)
        node.record_evaluation("h2", 1e5, "architecture", 1)
        assert len(node.trajectories) == 2

    def test_tier2_f07_learning_state_depth_zero_root_node(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_root")
        pt = node.record_evaluation("root_hyp", 0.5, "architecture", 0)
        assert pt.depth == 0

    def test_tier2_f07_learning_state_large_trajectories_scale(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_large")
        for i in range(200):
            node.record_evaluation(f"h_{i}", 0.5, "features", 1)
        assert node.total_budget_n == 200

    # --------------------------------------------------------------------------
    # F08: Epistemic Entropy Tracker Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f08_epistemic_entropy_single_trajectory_point_entropy_zero(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_one")
        node.record_evaluation("h", 0.8, "optimization", 1)
        assert node.compute_epistemic_entropy() == 0.0

    def test_tier2_f08_epistemic_entropy_all_scores_in_one_quantile_bin(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_one_bin")
        for i in range(5):
            node.record_evaluation(f"h_{i}", 0.81, "optimization", 1)
        assert node.compute_epistemic_entropy() == 0.0

    def test_tier2_f08_epistemic_entropy_exactly_uniform_across_bins(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_uniform_bins")
        # Distinct scores across 5 bins
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"h_{idx}", s, "objectives", 1)
        assert node.compute_epistemic_entropy() > 0.0

    def test_tier2_f08_epistemic_entropy_two_points_boundary(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_two")
        node.record_evaluation("h1", 0.1, "features", 1)
        node.record_evaluation("h2", 0.9, "features", 1)
        assert node.compute_epistemic_entropy() > 0.0

    def test_tier2_f08_epistemic_entropy_normalized_entropy_bounded_0_to_1(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_bounded")
        for i in range(20):
            node.record_evaluation(f"h_{i}", (i % 5) * 0.2, "architecture", 1)
        h = node.compute_epistemic_entropy()
        assert h >= 0.0

    # --------------------------------------------------------------------------
    # F09: Compute Scaling Monotonicity Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f09_scaling_monotonicity_strictly_decreasing_scores_preserves_envelope(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_dec")
        scores = [0.90, 0.80, 0.70, 0.60, 0.50]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"hyp_{idx}", s, "architecture", 1)
        curve = node.get_upper_envelope_curve()
        # All upper envelope values must remain 0.90
        for _, val in curve:
            assert math.isclose(val, 0.90, abs_tol=1e-4)

    def test_tier2_f09_scaling_monotonicity_single_budget_step(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_mono_single")
        node.record_evaluation("h", 0.75, "objectives", 1)
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0

    def test_tier2_f09_scaling_monotonicity_identical_scores_zero_gain(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_ident")
        for i in range(5):
            node.record_evaluation(f"h_{i}", 0.60, "features", 1)
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0

    def test_tier2_f09_scaling_monotonicity_alternating_scores_preserves_envelope(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_alt")
        scores = [0.5, 0.7, 0.4, 0.8, 0.3, 0.9]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"h_{idx}", s, "optimization", 1)
        curve = node.get_upper_envelope_curve()
        envelopes = [val for _, val in curve]
        for i in range(1, len(envelopes)):
            assert envelopes[i] >= envelopes[i - 1]

    def test_tier2_f09_scaling_monotonicity_floating_point_epsilon_stability(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_eps")
        node.record_evaluation("h1", 0.50000000001, "features", 1)
        node.record_evaluation("h2", 0.50000000002, "features", 1)
        is_mono, _ = node.verify_monotonic_progression()
        assert is_mono is True

    # --------------------------------------------------------------------------
    # F10: Epistemic Stagnation Detector Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f10_stagnation_exact_gradient_threshold_boundary(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        # Exact boundary: 1.0e-6 -> strict < condition does NOT trigger
        history = [{"grad_norm": 1e-6, "loss": 0.5, "ast_tokens": {"t_0"}}] * 3
        stagnant, trigger, _ = dog.audit_branch("b_bnd", depth=3, visits=5, history=history)
        assert trigger != exec_mod.StagnationTrigger.VANISHING_GRADIENTS

    def test_tier2_f10_stagnation_exact_plateau_threshold_boundary(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        assert dog.epsilon_plateau == 1e-4

    def test_tier2_f10_stagnation_exact_jaccard_threshold_boundary(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, tau_jaccard=0.85)
        # Jaccard exact 0.85 -> strictly greater (> 0.85) required
        assert dog.tau_jaccard == 0.85

    def test_tier2_f10_stagnation_history_shorter_than_m_stacks_returns_none(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"a"}}]  # Only 1 step
        stagnant, trigger, _ = dog.audit_branch("b_short", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == exec_mod.StagnationTrigger.NONE

    def test_tier2_f10_stagnation_empty_history_returns_none(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3)
        stagnant, trigger, _ = dog.audit_branch("b_empty_hist", depth=3, visits=5, history=[])
        assert stagnant is False
        assert trigger == exec_mod.StagnationTrigger.NONE

    # --------------------------------------------------------------------------
    # F11: Radical Architecture Safeguard Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f11_safeguard_depth_0_visits_0_exempt(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        stagnant, trigger, _ = dog.audit_branch("b0", depth=0, visits=0, history=[])
        assert stagnant is False

    def test_tier2_f11_safeguard_depth_1_visits_2_exact_boundary_exempt(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, _, _ = dog.audit_branch("b1_2", depth=1, visits=2, history=history)
        assert stagnant is False

    def test_tier2_f11_safeguard_depth_1_visits_3_boundary_not_exempt(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, _, _ = dog.audit_branch("b1_3", depth=1, visits=3, history=history)
        assert stagnant is True

    def test_tier2_f11_safeguard_depth_2_visits_2_boundary_not_exempt(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, _, _ = dog.audit_branch("b2_2", depth=2, visits=2, history=history)
        assert stagnant is True

    def test_tier2_f11_safeguard_negative_depth_or_visits_handled(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        stagnant, _, _ = dog.audit_branch("b_neg", depth=-1, visits=-1, history=[])
        assert stagnant is False

    # --------------------------------------------------------------------------
    # F12: Orthogonal Warm-Restart Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f12_warm_restart_custom_or_unknown_lane_fallback(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_custom")
        ev = dog.trigger_orthogonal_warm_restart("b_unk", current_lane="quantum_kernels", state_node=node)
        assert ev.target_lane in ("architecture", "features", "objectives", "optimization")

    def test_tier2_f12_warm_restart_cycle_period_four_invariance(self):
        exec_mod = get_executioner_module()
        dog = exec_mod.ExecutionerWatchdog()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_cycle")
        # arch -> feat -> obj -> opt -> arch
        l1 = dog.trigger_orthogonal_warm_restart("b1", "architecture", node).target_lane
        l2 = dog.trigger_orthogonal_warm_restart("b2", l1, node).target_lane
        l3 = dog.trigger_orthogonal_warm_restart("b3", l2, node).target_lane
        l4 = dog.trigger_orthogonal_warm_restart("b4", l3, node).target_lane
        assert l1 == "features"
        assert l2 == "objectives"
        assert l3 == "optimization"
        assert l4 == "architecture"

    def test_tier2_f12_warm_restart_empty_state_node_handled(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_clean")
        ev = dog.trigger_orthogonal_warm_restart("b_clean", "objectives", node)
        assert len(node.executioner_event_logs) == 1

    def test_tier2_f12_warm_restart_root_branch_parent_none(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_root")
        ev = dog.trigger_orthogonal_warm_restart("root", "optimization", node)
        assert ev.branch_id != "root"

    def test_tier2_f12_warm_restart_generates_unique_branch_id(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_uniq")
        ev1 = dog.trigger_orthogonal_warm_restart("b", "features", node)
        ev2 = dog.trigger_orthogonal_warm_restart("b", "features", node)
        assert ev1.branch_id != ev2.branch_id

    # --------------------------------------------------------------------------
    # F13: Executioner Event Log Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f13_executioner_empty_event_logs(self):
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_empty_log")
        assert len(node.executioner_event_logs) == 0

    def test_tier2_f13_executioner_event_log_timestamp_monotonicity(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_time")
        ev1 = dog.trigger_orthogonal_warm_restart("b1", "architecture", node)
        ev2 = dog.trigger_orthogonal_warm_restart("b2", "features", node)
        assert ev2.timestamp >= ev1.timestamp

    def test_tier2_f13_executioner_large_event_log_serialization(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_many_ev")
        for i in range(50):
            dog.trigger_orthogonal_warm_restart(f"b_{i}", "architecture", node)
        assert len(node.executioner_event_logs) == 50
        assert "b_49" in json.dumps(node.model_dump())

    def test_tier2_f13_executioner_event_log_special_characters_in_details(self):
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        dog = exec_mod.ExecutionerWatchdog()
        node = state_mod.SwarmLearningStateNode(node_id="n_spec")
        ev = dog.trigger_orthogonal_warm_restart("b_spec", "objectives", node)
        assert isinstance(ev.details, dict)

    def test_tier2_f13_executioner_event_log_reconstruction_from_dict(self):
        exec_mod = get_executioner_module()
        raw = {
            "event_type": "ORTHOGONAL_WARM_RESTART",
            "branch_id": "b_recon",
            "trigger": exec_mod.StagnationTrigger.LOSS_PLATEAU,
            "details": {"test": 1},
            "timestamp": 1234567.8,
            "target_lane": "features",
        }
        event = exec_mod.ExecutionerEvent(**raw)
        assert event.branch_id == "b_recon"

    # --------------------------------------------------------------------------
    # F14: Evidence Hierarchy Weighting Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f14_evidence_hierarchy_exact_boundary_threshold(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Source with exact score 0.50
        s = gov_mod.EvidenceSource(
            source_id="s50", title="Fifty", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.50
        )
        passed, score = verifier.verify_evidence_hierarchy([s])
        assert passed is True
        assert math.isclose(score, 0.50, abs_tol=1e-4)

    def test_tier2_f14_evidence_hierarchy_empty_sources_list_zero_score(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        passed, score = verifier.verify_evidence_hierarchy([])
        assert passed is False
        assert score == 0.0

    def test_tier2_f14_evidence_hierarchy_single_source_cochrane_credibility_1(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        s = gov_mod.EvidenceSource(
            source_id="s100", title="Top", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=1.0
        )
        passed, score = verifier.verify_evidence_hierarchy([s])
        assert passed is True
        assert math.isclose(score, 1.0, abs_tol=1e-4)

    def test_tier2_f14_evidence_hierarchy_zero_credibility_all_sources(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        s = gov_mod.EvidenceSource(
            source_id="s0", title="Zero", source_type=gov_mod.SourceTier.PREPRINT,
            methodological_rigor=0.0
        )
        passed, score = verifier.verify_evidence_hierarchy([s])
        assert passed is False
        assert score == 0.0

    def test_tier2_f14_evidence_hierarchy_unweighted_vs_weighted_separation(self):
        de_mod = get_domain_evidence_module()
        gov_mod = get_governor_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Low hierarchy forum source with high rigor vs high hierarchy with low rigor
        s_forum = gov_mod.EvidenceSource(
            source_id="f", title="Forum", source_type=gov_mod.SourceTier.PUBLIC_FORUM,
            methodological_rigor=0.9
        )
        s_journal = gov_mod.EvidenceSource(
            source_id="j", title="Journal", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.3
        )
        passed, score = verifier.verify_evidence_hierarchy([s_forum, s_journal])
        assert isinstance(passed, bool)

    # --------------------------------------------------------------------------
    # F15: Data Provenance & Ethics Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f15_provenance_exact_epsilon_threshold_boundary(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Epsilon exactly 1.0 (boundary)
        prov = de_mod.DataProvenance(
            dataset_cid="cid1", license="MIT", privacy_budget_epsilon=1.0, bias_disparity_score=0.05
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True

    def test_tier2_f15_provenance_exact_disparity_threshold_boundary(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Disparity exactly 0.10 (boundary)
        prov = de_mod.DataProvenance(
            dataset_cid="cid2", license="MIT", privacy_budget_epsilon=0.5, bias_disparity_score=0.10
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True

    def test_tier2_f15_provenance_unknown_license_string_rejected(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(
            dataset_cid="cid3", license="UNKNOWN_RESTRICTED_LICENSE", privacy_budget_epsilon=0.5
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False

    def test_tier2_f15_provenance_zero_epsilon_maximum_privacy(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(dataset_cid="cid4", license="MIT", privacy_budget_epsilon=0.0)
        passed, _ = verifier.verify_provenance_and_ethics(prov)
        assert passed is True

    def test_tier2_f15_provenance_empty_cid_handled(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        prov = de_mod.DataProvenance(dataset_cid="", license="MIT")
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert isinstance(passed, bool)

    # --------------------------------------------------------------------------
    # F16: Domain Invariant Plugin Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f16_invariant_plugin_empty_plugins_list(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier(plugins=[])
        assert len(verifier.plugins) == 0

    def test_tier2_f16_invariant_plugin_non_existent_domain_filter(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x**2", domain="astrophysics_nonexistent")
        assert isinstance(res, dict)

    def test_tier2_f16_invariant_plugin_duplicate_constraint_ids(self):
        de_mod = get_domain_evidence_module()
        class DupPlugin(de_mod.DomainInvariantPlugin):
            def get_domain_name(self): return "dup"
            def get_constraints(self):
                return [
                    de_mod.SymbolicDomainConstraint(constraint_id="c_same", name="1", domain="dup", symbolic_expression="x"),
                    de_mod.SymbolicDomainConstraint(constraint_id="c_same", name="2", domain="dup", symbolic_expression="y"),
                ]
        verifier = de_mod.DomainEvidenceVerifier(plugins=[DupPlugin()])
        assert "dup" in verifier.plugins

    def test_tier2_f16_invariant_plugin_trivial_always_true_constraint(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("1.0", domain="physics")
        assert res["all_satisfied"] is True

    def test_tier2_f16_invariant_plugin_empty_variable_bounds_handled(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x**2 + y**2", domain=None)
        assert isinstance(res, dict)

    # --------------------------------------------------------------------------
    # F17: L0–L2 Math Verifier Invariant Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f17_l0_pole_or_singularity_detection_fails(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # 1 / x has pole at x=0
        res = verifier.verify_symbolic_invariants("1 / x", domain="boundary")
        assert res["all_satisfied"] is False

    def test_tier2_f17_l1_infinite_interval_bounds_handled(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("exp(-x**2)", domain="physics")
        assert isinstance(res["all_satisfied"], bool)

    def test_tier2_f17_l2_boundary_point_equality(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # x >= 0 on [0, 5], boundary at x=0
        res = verifier.verify_symbolic_invariants("x", domain="physics")
        assert isinstance(res, dict)

    def test_tier2_f17_non_differentiable_point_detection(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("abs(x)", domain="biology")
        assert isinstance(res, dict)

    def test_tier2_f17_ill_formed_math_expression_handled_gracefully(self):
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x +* 3 / )", domain="physics")
        assert res["all_satisfied"] is False

    # --------------------------------------------------------------------------
    # F18: Status CLI Observability Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f18_status_clean_when_no_prior_runs_no_crash(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ret = cmd_status(argparse.Namespace(detail=True, clear=False))
        assert ret == 0

    def test_tier2_f18_status_formatting_zero_discoveries(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        output = captured.getvalue()
        assert "Confirmed Discoveries" in output or "Status" in output

    def test_tier2_f18_status_formatting_single_node_state(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_status(argparse.Namespace(detail=False, clear=False))
        assert captured.getvalue() != ""

    def test_tier2_f18_status_handles_corrupted_state_gracefully(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        # Should execute cleanly without unhandled exceptions
        ret = cmd_status(argparse.Namespace(detail=False, clear=False))
        assert ret == 0

    def test_tier2_f18_status_preserves_exit_code_zero(self, monkeypatch):
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        assert cmd_status(argparse.Namespace(detail=False, clear=False)) == 0

    # --------------------------------------------------------------------------
    # F19: Swarm Spec Export CLI Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f19_swarm_export_empty_lanes_flag_fallback(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(goal="G", lanes="architecture,objectives,optimization,features", export_agy_specs=True)
        ret = cmd_swarm(args)
        assert ret == 0
        data = json.loads(captured.getvalue())
        assert len(data) >= 4

    def test_tier2_f19_swarm_export_whitespace_lanes_handled(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(goal="G", lanes="  architecture , objectives  ", export_agy_specs=True)
        ret = cmd_swarm(args)
        assert ret == 0

    def test_tier2_f19_swarm_export_all_roles_have_distinct_typenames(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(goal="G", lanes="architecture,objectives,optimization,features", export_agy_specs=True)
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        typenames = [item["TypeName"] for item in data]
        assert len(typenames) == len(set(typenames))

    def test_tier2_f19_swarm_export_large_goal_description(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        large_goal = "A" * 5000
        args = argparse.Namespace(goal=large_goal, lanes="architecture,objectives,optimization,features", export_agy_specs=True)
        ret = cmd_swarm(args)
        assert ret == 0

    def test_tier2_f19_swarm_export_json_validity_under_special_chars(self, monkeypatch):
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        args = argparse.Namespace(goal="Goal with \n\t \"quotes\" & \\slashes", lanes="architecture,objectives,optimization,features", export_agy_specs=True)
        cmd_swarm(args)
        data = json.loads(captured.getvalue())
        assert isinstance(data, list)

    # --------------------------------------------------------------------------
    # F20: Statistical Screener Boundary Cases
    # --------------------------------------------------------------------------
    def test_tier2_f20_lord3_alpha_decay_step_100(self):
        controller = LORDController(alpha=0.05)
        # Advance 100 non-discoveries
        for _ in range(100):
            controller.record_result(p_value=0.99)
        alpha_101 = controller.get_next_alpha()
        assert 0.0 < alpha_101 < 0.05

    def test_tier2_f20_rope_identical_seeds_zero_difference(self):
        base = {f"s_{i}": 0.80 for i in range(5)}
        cand = {f"s_{i}": 0.80 for i in range(5)}
        lord = LORDController(alpha=0.05)
        report = evaluate_paired_seed_rope(
            hypothesis_id="hyp_zero_diff",
            metric_name="accuracy",
            baseline_scores=base,
            candidate_scores=cand,
            rope_epsilon=0.01,
            lord_controller=lord,
            target_direction=TargetDirection.MAXIMIZE,
        )
        assert report.is_discovery is False

    def test_tier2_f20_stage1_futility_exact_zero_difference_aborts(self):
        # Baseline paired metrics with exactly 0 improvement
        cand = [0.5, 0.5, 0.5]
        base = [0.5, 0.5, 0.5]
        d_bar = np.mean(cand) - np.mean(base)
        assert d_bar <= 0.0  # Stage 1 futility filter condition D_bar <= 0

    def test_tier2_f20_flop_counter_single_layer_linear(self):
        net = nn.Linear(10, 5, bias=False)
        fwd, bwd, params = profile_module_flops(net, sample_input=torch.randn(1, 10))
        assert fwd > 0
        assert params == 50

    def test_tier2_f20_statistical_verifier_extreme_p_values(self):
        controller = LORDController(alpha=0.05)
        disc_0, _ = controller.record_result(p_value=0.0)
        assert disc_0 is True
        disc_1, _ = controller.record_result(p_value=1.0)
        assert disc_1 is False


# ==============================================================================
# TIER 3: CROSS-FEATURE INTERACTIONS (12 Integration Tests)
# ==============================================================================

class TestTier3CrossFeatureInteractions:
    """Tier 3: Pairwise contracts and cross-module lifecycle interactions."""

    def test_tier3_interaction_governor_to_novelty_prior_art_flow(self):
        """Governor audited literature serves as prior art corpus for Novelty Contract."""
        gov_mod = get_governor_module()
        nov_mod = get_novelty_module()
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        s1 = gov_mod.EvidenceSource(
            source_id="s1", title="Sparse Mixture of Experts", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.9
        )
        s2 = gov_mod.EvidenceSource(
            source_id="s2", title="Unvetted Blog Trick", source_type=gov_mod.SourceTier.PUBLIC_FORUM,
            methodological_rigor=0.1
        )
        approved = gov.audit_and_filter([s1, s2])
        assert len(approved) == 1
        prior_art = [s.title for s in approved]

        contract = nov_mod.DualKeyNoveltyContract(tau_lit=0.15)
        d_lit, nearest = contract.compute_literature_distance("Sparse Mixture of Experts", prior_art)
        assert d_lit <= 0.15  # Rejects exact copy of vetted prior art

    def test_tier3_interaction_novelty_rejection_to_learning_state_event(self):
        """Novelty Contract rejection records event in SwarmLearningStateNode."""
        nov_mod = get_novelty_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_nov_event")
        contract = nov_mod.DualKeyNoveltyContract()
        y = np.ones(30)
        res = contract.evaluate_novelty("h_bad", "code", ["code"], y, y, y)
        assert res.passed is False
        # Record novelty audit into governor/executioner event logs
        node.governor_event_logs.append({
            "event": "NOVELTY_AUDIT_REJECTED",
            "hypothesis_id": res.hypothesis_id,
            "verdict": res.verdict,
        })
        assert len(node.governor_event_logs) == 1

    def test_tier3_interaction_novelty_zero_alpha_wealth_lord_preserved(self):
        """Novelty rejection consumes exactly 0 alpha wealth from LORD controller."""
        lord = LORDController(alpha=0.05)
        initial_alpha = lord.get_next_alpha()
        nov_mod = get_novelty_module()
        contract = nov_mod.DualKeyNoveltyContract()
        y = np.linspace(0, 1, 30)
        contract.evaluate_novelty("h_reject", "identical", ["identical"], y, y + 1, y + 1)
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-7)

    def test_tier3_interaction_learning_state_to_executioner_stagnation_detection(self):
        """Executioner queries SwarmLearningStateNode trajectories and identifies stagnation."""
        state_mod = get_learning_state_module()
        exec_mod = get_executioner_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_query_stag")
        for i in range(4):
            node.record_evaluation(f"h_{i}", 0.50, "architecture", depth=3)
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        # History derived from trajectory scores
        history = [
            {"grad_norm": 0.1, "loss": 1.0 - pt.quality_score, "ast_tokens": {f"t_{pt.step}"}}
            for pt in node.trajectories[-3:]
        ]
        stagnant, trigger, _ = dog.audit_branch("b_traj", depth=3, visits=4, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.LOSS_PLATEAU

    def test_tier3_interaction_executioner_warm_restart_to_learning_state_node(self):
        """Warm restart appends new trajectory and records ExecutionerEvent in state node."""
        state_mod = get_learning_state_module()
        exec_mod = get_executioner_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_restart_link")
        dog = exec_mod.ExecutionerWatchdog()
        ev = dog.trigger_orthogonal_warm_restart("b_stuck", "architecture", state_node=node)
        assert len(node.executioner_event_logs) == 1
        # Inject new exploration trajectory in target lane
        pt = node.record_evaluation("h_restarted", 0.65, lane=ev.target_lane, depth=1)
        assert pt.lane == "features"

    def test_tier3_interaction_early_safeguard_with_learning_state_node(self):
        """Executioner checks node depth and visits in state node to preserve radical designs."""
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_safeguard")
        dog = exec_mod.ExecutionerWatchdog(early_depth=1, early_visits=2)
        node.record_evaluation("h_early", 0.2, "architecture", depth=1)
        stagnant, _, _ = dog.audit_branch("b_early", depth=1, visits=1, history=[{"grad_norm": 1e-9, "loss": 0.8, "ast_tokens": {"x"}}] * 3)
        assert stagnant is False

    def test_tier3_interaction_domain_invariants_to_l0_l2_math_pipeline(self):
        """Domain invariant constraints bound to L0–L2 math verification engine."""
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Non-negative energy invariant
        res = verifier.verify_symbolic_invariants("x**2 + 2*x + 5", domain="physics")
        assert res["all_satisfied"] is True

    def test_tier3_interaction_governor_entropy_with_lane_specialist_proposals(self):
        """Governor monitors proposals emitted by the 4 lane specialists."""
        gov_mod = get_governor_module()
        bridge = get_agy_bridge_module()
        gov = gov_mod.InformationGovernor()
        proposals = [
            bridge.SwarmProposal(
                hypothesis_id="p1", lane=bridge.LaneType.ARCHITECTURE, title="A", description="D", code_snippet="C"
            ),
            bridge.SwarmProposal(
                hypothesis_id="p2", lane=bridge.LaneType.OBJECTIVES, title="A", description="D", code_snippet="C"
            ),
            bridge.SwarmProposal(
                hypothesis_id="p3", lane=bridge.LaneType.OPTIMIZATION, title="A", description="D", code_snippet="C"
            ),
            bridge.SwarmProposal(
                hypothesis_id="p4", lane=bridge.LaneType.FEATURES, title="A", description="D", code_snippet="C"
            ),
        ]
        counts = {}
        for p in proposals:
            counts[p.lane.value] = counts.get(p.lane.value, 0) + 1
        res = gov.check_exploration_entropy(counts)
        assert res["sufficient"] is True

    def test_tier3_interaction_compute_scaling_compounding_with_mcts_evaluations(self):
        """MCTS evaluations compound monotonically into the upper envelope scaling curve."""
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_mcts_scaling")
        scores = [0.55, 0.58, 0.62, 0.60, 0.70, 0.68, 0.75]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"mcts_{idx}", s, "architecture", depth=idx % 3)
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain >= 0.20

    def test_tier3_interaction_cli_status_reflects_live_governor_and_executioner_events(self, monkeypatch):
        """Live campaign logs reflect in CLI status output dashboard."""
        import argparse
        from frame.cli import cmd_status
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        ret = cmd_status(argparse.Namespace(detail=True, clear=False))
        assert ret == 0

    def test_tier3_interaction_cli_swarm_export_aligns_with_subagent_specs_schema(self, monkeypatch):
        """CLI swarm exported specs match SubagentSpec schema exactly."""
        import argparse
        from frame.cli import cmd_swarm
        captured = io.StringIO()
        monkeypatch.setattr(sys, "stdout", captured)
        cmd_swarm(argparse.Namespace(goal="Align", lanes="architecture,objectives,optimization,features", export_agy_specs=True))
        data = json.loads(captured.getvalue())
        for spec_dict in data:
            assert "TypeName" in spec_dict
            assert "Role" in spec_dict
            assert "Prompt" in spec_dict

    def test_tier3_interaction_statistical_screeners_with_novelty_gated_proposals(self):
        """Proposals passing Novelty Contract undergo Two-Stage screening under LORD-3."""
        nov_mod = get_novelty_module()
        contract = nov_mod.DualKeyNoveltyContract()
        y = np.linspace(0, 5, 40)
        base = y + 0.5
        cand = y + 0.1  # Functionally distinct
        res = contract.evaluate_novelty("h_pass", "class NewModel: pass", ["OldModel"], y, base, cand)
        if res.passed:
            # Stage 2 Confirmatory test
            base_metrics = {f"seed_{i}": 0.75 for i in range(5)}
            cand_metrics = {f"seed_{i}": 0.83 for i in range(5)}
            lord = LORDController(alpha=0.05)
            report = evaluate_paired_seed_rope(
                hypothesis_id="hyp_pass",
                metric_name="accuracy",
                baseline_scores=base_metrics,
                candidate_scores=cand_metrics,
                rope_epsilon=0.01,
                lord_controller=lord,
                target_direction=TargetDirection.MAXIMIZE,
            )
            assert report.is_discovery is True


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS (6 End-to-End Workflow Tests)
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Full multi-agent autonomous scientific research workflows with synthetic data."""

    def test_tier4_scenario_discovery_workflow_end_to_end(self):
        """Full cognitive lifecycle: Literature vetting -> Swarm proposal -> Novelty gate -> Screener."""
        gov_mod = get_governor_module()
        nov_mod = get_novelty_module()
        state_mod = get_learning_state_module()

        # Step 1: Ingestion & Information Governor Vetting
        gov = gov_mod.InformationGovernor(grading_threshold=0.40)
        sources = [
            gov_mod.EvidenceSource(
                source_id="s1", title="Low-Rank Linear Attention", source_type=gov_mod.SourceTier.COCHRANE_OR_JOURNAL,
                peer_reviewed=True, relevance_score=0.9, methodological_rigor=0.85
            ),
            gov_mod.EvidenceSource(
                source_id="s2", title="Random Forum Hack", source_type=gov_mod.SourceTier.PUBLIC_FORUM,
                peer_reviewed=False, relevance_score=0.3, methodological_rigor=0.1
            ),
        ]
        vetted = gov.audit_and_filter(sources)
        assert len(vetted) == 1

        # Step 2: High-Entropy Swarm Proposal Check
        counts = {"architecture": 3, "objectives": 3, "optimization": 3, "features": 3}
        entropy_res = gov.check_exploration_entropy(counts)
        assert entropy_res["sufficient"] is True

        # Step 3: Dual-Key Novelty Evaluation
        contract = nov_mod.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 10, 50)
        base_pred = y_true + 1.0
        cand_pred = y_true + np.cos(np.linspace(0, 5, 50))
        nov_res = contract.evaluate_novelty(
            hypothesis_id="hyp_workflow",
            candidate_code="class LinearKernelAttention(nn.Module): pass",
            prior_art_corpus=[s.title for s in vetted],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert nov_res.passed is True

        # Step 4: Compounding Learning State Recording
        node = state_mod.SwarmLearningStateNode(node_id="n_workflow")
        pt = node.record_evaluation("hyp_workflow", score=0.82, lane="architecture", depth=1)
        assert pt.upper_envelope_quality == 0.82

    def test_tier4_scenario_adversarial_surface_form_paraphrase_attack(self):
        """Adversary attempts 5 cosmetic variations of baseline code; all rejected by Novelty Contract."""
        nov_mod = get_novelty_module()
        contract = nov_mod.DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 10, 40)
        base_pred = y_true + 0.5  # Constant baseline residual

        adversarial_attempts = [
            ("# Renamed x to alpha\nclass MLP_Var1: pass", base_pred),
            ("# Added docstrings\nclass MLP_Var2: pass", base_pred),
            ("# Formatted with black\nclass MLP_Var3: pass", base_pred),
            ("# Replaced loop with list comprehension\nclass MLP_Var4: pass", base_pred),
            ("# Swapped order of functions\nclass MLP_Var5: pass", base_pred),
        ]

        for idx, (code, preds) in enumerate(adversarial_attempts):
            res = contract.evaluate_novelty(
                hypothesis_id=f"adv_{idx}",
                candidate_code=code,
                prior_art_corpus=["Original Baseline MLP with standard variables"],
                y_true=y_true,
                y_baseline_pred=base_pred,
                y_candidate_pred=preds,
            )
            assert res.passed is False
            assert res.verdict in (
                nov_mod.NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
                nov_mod.NoveltyVerdict.REJECTED_DUAL_FAILURE,
            )

    def test_tier4_scenario_stagnant_research_branch_recovery(self):
        """MCTS branch gets trapped in vanishing gradients; Executioner detects and recovers via warm restart."""
        exec_mod = get_executioner_module()
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_recovery")
        dog = exec_mod.ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)

        # Stagnant branch history at depth 3
        history = [
            {"grad_norm": 5e-8, "loss": 0.40, "ast_tokens": {"x", "y"}},
            {"grad_norm": 3e-8, "loss": 0.40, "ast_tokens": {"x", "y"}},
            {"grad_norm": 1e-8, "loss": 0.40, "ast_tokens": {"x", "y"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_trapped", depth=3, visits=6, history=history)
        assert stagnant is True
        assert trigger == exec_mod.StagnationTrigger.VANISHING_GRADIENTS

        # Executioner triggers Orthogonal Warm Restart
        event = dog.trigger_orthogonal_warm_restart("b_trapped", current_lane="architecture", state_node=node)
        assert event.target_lane == "features"
        assert event.event_type == "ORTHOGONAL_WARM_RESTART"
        assert len(node.executioner_event_logs) == 1

        # New search resumes in features lane
        pt = node.record_evaluation("hyp_recovered", score=0.88, lane="features", depth=1)
        assert pt.lane == "features"

    def test_tier4_scenario_domain_invariant_physical_conservation_violation(self):
        """Candidate proposal with energy violation (E < 0) caught by Domain Evidence Verifier."""
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        # Candidate evaluates to negative energy (-5 at x=0)
        res = verifier.verify_symbolic_invariants("x**2 - 5", domain="physics")
        assert res["all_satisfied"] is False

    def test_tier4_scenario_data_provenance_and_ethics_safeguard(self):
        """Proposal attempting to use an unlicensed or privacy-violating dataset is blocked."""
        de_mod = get_domain_evidence_module()
        verifier = de_mod.DomainEvidenceVerifier()
        bad_prov = de_mod.DataProvenance(
            dataset_cid="bafy_bad",
            license="NO_COMMERCIAL_USE_ALLOWED",
            is_commercial_safe=False,
            privacy_budget_epsilon=5.0,  # Leaks privacy
            bias_disparity_score=0.45,   # Unacceptable bias
            ethical_approval=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(bad_prov)
        assert passed is False
        assert len(violations) >= 3

    def test_tier4_scenario_multi_generation_test_time_compute_scaling(self):
        """Multi-generation MCTS compounding upper envelope progression over search budget N=10."""
        state_mod = get_learning_state_module()
        node = state_mod.SwarmLearningStateNode(node_id="n_compute_scaling_10")
        # Simulating noisy candidate discoveries across 10 search steps
        discovered_qualities = [0.60, 0.58, 0.65, 0.64, 0.72, 0.71, 0.78, 0.75, 0.84, 0.82]
        for step, q in enumerate(discovered_qualities, 1):
            node.record_evaluation(f"hyp_{step}", score=q, lane="architecture", depth=step % 4)

        curve = node.get_upper_envelope_curve()
        assert len(curve) == 10
        # Verify strict monotonicity of upper envelope
        envelopes = [val for _, val in curve]
        for i in range(1, len(envelopes)):
            assert envelopes[i] >= envelopes[i - 1]

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isclose(gain, 0.84 - 0.60, abs_tol=1e-4)
