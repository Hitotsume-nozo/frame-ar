"""Tests for Pre-Flight FDR Edge-Indexing, Blend Edge, Event Appends, and TAPAScore.

Verifies:
1. Edge-indexed StatisticalVerificationReport schema (edge_id, source/target hypothesis IDs, edge_type, tapa_score).
2. Distinct per-edge rope_epsilon and LORD alpha_t allocations in TwoStageScreeningController and evaluate_paired_seed_rope.
3. The Blend Edge protocol: ensembled/blended candidates must be screened against the incumbent best single model.
4. SwarmLearningStateNode governor and executioner event appends across all audit hooks (evidence scoring, entropy invariant, literature distance, AST falsification, watchdog monotonicity, Upper Envelope updates, stagnation checks).
5. Numerical TAPAScore formulation for Modality A and Modality B.
"""

import math
import numpy as np
import pytest

from frame.schemas.verification import StatisticalVerificationReport, PairedSeedMetrics
from frame.schemas.hypothesis import TargetDirection
from frame.schemas.learning_state import SwarmLearningStateNode
from frame.schemas.executioner import StagnationTrigger
from frame.verification.statistics import LORDController, evaluate_paired_seed_rope
from frame.verification.tapa import compute_tapa_score, compute_tapa_score_modality_b
from frame.brain.experimental_design import TwoStageScreeningController
from frame.agents.governor import InformationGovernor, EvidenceSource, SourceTier
from frame.brain.executioner import ExecutionerWatchdog
from frame.verification.novelty import DualKeyNoveltyContract
from frame.brain.thinking import StructuralFalsificationMemory, SearchStage


class TestEdgeIndexedStatisticalVerificationReport:
    """Test suite for edge-indexed FDR reporting in StatisticalVerificationReport."""

    def test_edge_indexed_report_explicit_fields(self):
        report = StatisticalVerificationReport(
            hypothesis_id="hyp_cand_01",
            metric_name="rmse",
            sample_size_n=5,
            mean_baseline=0.55,
            mean_candidate=0.52,
            paired_metrics=PairedSeedMetrics(
                metric_name="rmse",
                paired_differences={"s1": 0.03, "s2": 0.03, "s3": 0.03},
                mean_difference=0.03,
                standard_error=0.001,
                t_statistic=30.0,
            ),
            rope_epsilon=0.005,
            rope_t_statistic=25.0,
            p_value_rope=1e-5,
            lord_step=1,
            allocated_alpha_t=0.010937,
            is_discovery=True,
            verdict="CONFIRMED_DISCOVERY",
            edge_id="edge_gen0_to_gen1",
            source_hypothesis_id="hyp_baseline_gen0",
            target_hypothesis_id="hyp_cand_01",
            source_node_id="node_root",
            target_node_id="node_gen1",
            edge_type="atomic_transition",
            tapa_score=92.5,
        )

        assert report.edge_id == "edge_gen0_to_gen1"
        assert report.source_hypothesis_id == "hyp_baseline_gen0"
        assert report.target_hypothesis_id == "hyp_cand_01"
        assert report.edge_type == "atomic_transition"
        assert report.tapa_score == 92.5
        assert len(report.compute_hash()) == 64

    def test_edge_indexed_report_default_population(self):
        """Omitting edge fields automatically derives consistent defaults."""
        report = StatisticalVerificationReport(
            hypothesis_id="hyp_cand_auto",
            metric_name="accuracy",
            sample_size_n=5,
            mean_baseline=0.80,
            mean_candidate=0.85,
            paired_metrics=PairedSeedMetrics(
                metric_name="accuracy",
                paired_differences={"s1": 0.05, "s2": 0.05, "s3": 0.05},
                mean_difference=0.05,
                standard_error=0.001,
                t_statistic=50.0,
            ),
            rope_epsilon=0.01,
            rope_t_statistic=40.0,
            p_value_rope=1e-6,
            lord_step=1,
            allocated_alpha_t=0.010937,
            is_discovery=True,
            verdict="CONFIRMED_DISCOVERY",
        )

        assert report.target_hypothesis_id == "hyp_cand_auto"
        assert report.edge_id == "edge_baseline_to_hyp_cand_auto"
        assert report.edge_type == "atomic_transition"

    def test_evaluate_paired_seed_rope_passes_edge_indices(self):
        lord = LORDController(alpha=0.05)
        base_scores = {"s1": 0.50, "s2": 0.52, "s3": 0.51, "s4": 0.53, "s5": 0.49}
        cand_scores = {"s1": 0.45, "s2": 0.46, "s3": 0.44, "s4": 0.47, "s5": 0.43}

        report = evaluate_paired_seed_rope(
            hypothesis_id="hyp_blend_stack",
            metric_name="rmse",
            baseline_scores=base_scores,
            candidate_scores=cand_scores,
            rope_epsilon=0.001,  # Stacking refinement margin
            lord_controller=lord,
            target_direction=TargetDirection.MINIMIZE,
            edge_id="edge_incumbent_to_blend",
            source_hypothesis_id="hyp_incumbent_best",
            target_hypothesis_id="hyp_blend_stack",
            edge_type="blend_ensemble",
            tapa_score=88.4,
        )

        assert report.edge_id == "edge_incumbent_to_blend"
        assert report.source_hypothesis_id == "hyp_incumbent_best"
        assert report.target_hypothesis_id == "hyp_blend_stack"
        assert report.edge_type == "blend_ensemble"
        assert report.rope_epsilon == 0.001
        assert report.tapa_score == 88.4
        assert report.is_discovery is True


class TestBlendEdgeVerificationProtocol:
    """Test suite ensuring blended/ensembled candidates must be verified against incumbent best single model."""

    def test_blend_edge_screening_against_incumbent_single_best(self):
        """Simulates Gen 1 (LGBM), Gen 2 (CatBoost), and Gen 4 (Stack).
        The blend MUST be tested against max(LGBM, CatBoost) with distinct alpha and eps_ROPE.
        """
        controller = TwoStageScreeningController(rope_epsilon=0.005, target_direction=TargetDirection.MINIMIZE)
        lord = LORDController(alpha=0.05)

        # Baseline (Gen 0)
        gen0_scores = {"s1": 0.60, "s2": 0.61, "s3": 0.59, "s4": 0.62, "s5": 0.58}
        # Best single model (e.g. LightGBM Gen 1)
        lgbm_scores = {"s1": 0.53, "s2": 0.54, "s3": 0.52, "s4": 0.55, "s5": 0.51}
        # Second single model (e.g. CatBoost Gen 2)
        cat_scores = {"s1": 0.54, "s2": 0.53, "s3": 0.53, "s4": 0.56, "s5": 0.52}

        # Step 1: Gen 1 vs Gen 0 passes
        s2_gen1 = controller.evaluate_stage2_confirmatory(
            baseline_scores=gen0_scores,
            candidate_scores=lgbm_scores,
            lord_controller=lord,
            rope_epsilon_override=0.005,
            edge_id="edge_gen0_to_gen1_lgbm",
            source_hypothesis_id="hyp_gen0",
            target_hypothesis_id="hyp_gen1_lgbm",
            edge_type="atomic_transition",
        )
        assert s2_gen1.proceed is True
        assert s2_gen1.details["edge_id"] == "edge_gen0_to_gen1_lgbm"
        alpha_step1 = s2_gen1.details["allocated_alpha_t"]

        # Incumbent single best is LGBM (mean 0.53)
        # Case A: A successful blend beating incumbent single best
        strong_blend_scores = {"s1": 0.51, "s2": 0.52, "s3": 0.50, "s4": 0.53, "s5": 0.49}
        s2_blend_pass = controller.evaluate_stage2_confirmatory(
            baseline_scores=lgbm_scores,  # Blend edge evaluates against incumbent single best
            candidate_scores=strong_blend_scores,
            lord_controller=lord,
            rope_epsilon_override=0.001,  # Pre-registered refinement margin for blends
            edge_id="edge_incumbent_to_blend",
            source_hypothesis_id="hyp_gen1_lgbm",
            target_hypothesis_id="hyp_blend_stack",
            edge_type="blend_ensemble",
        )
        assert s2_blend_pass.proceed is True
        assert s2_blend_pass.details["edge_type"] == "blend_ensemble"
        assert s2_blend_pass.details["rope_epsilon"] == 0.001
        assert s2_blend_pass.details["allocated_alpha_t"] != alpha_step1  # Distinct alpha consumed

        # Case B: A weak blend that beats Gen 0 but fails against incumbent single best
        weak_blend_scores = {"s1": 0.529, "s2": 0.539, "s3": 0.521, "s4": 0.549, "s5": 0.511}
        # Testing against Gen 0 would falsely look like discovery:
        s2_vs_gen0 = controller.evaluate_stage2_confirmatory(
            baseline_scores=gen0_scores,
            candidate_scores=weak_blend_scores,
            alpha_override=0.05,
            rope_epsilon_override=0.005,
        )
        assert s2_vs_gen0.proceed is True  # Falsely passes if compared to root

        # BUT testing the BLEND EDGE against incumbent single best correctly rejects:
        s2_blend_fail = controller.evaluate_stage2_confirmatory(
            baseline_scores=lgbm_scores,  # Mandatory blend edge
            candidate_scores=weak_blend_scores,
            lord_controller=lord,
            rope_epsilon_override=0.001,
            edge_id="edge_lgbm_to_weak_blend",
            source_hypothesis_id="hyp_gen1_lgbm",
            target_hypothesis_id="hyp_weak_blend",
            edge_type="blend_ensemble",
        )
        assert s2_blend_fail.proceed is False  # Correctly rejected: blend margin <= rope_epsilon


class TestLearningStateEventAppends:
    """Test suite for structured governor and executioner event appends."""

    def test_direct_event_append_helpers(self):
        node = SwarmLearningStateNode(node_id="test_node_events")

        gov_entry = node.append_governor_event("source_vetting", {"source_id": "src_1", "grade": 0.85})
        assert len(node.governor_event_logs) == 1
        assert gov_entry["event_type"] == "source_vetting"
        assert gov_entry["grade"] == 0.85

        exec_entry = node.log_executioner_event("watchdog_monotonicity", is_monotonic=True, net_gain=0.12)
        assert len(node.executioner_event_logs) == 1
        assert exec_entry["event_type"] == "watchdog_monotonicity"
        assert exec_entry["net_gain"] == 0.12

    def test_record_evaluation_and_watchdog_event_logging(self):
        node = SwarmLearningStateNode(node_id="test_eval_events")
        node.record_evaluation("hyp_01", 0.70, "features", depth=1, log_event=True)
        node.record_evaluation("hyp_02", 0.80, "architecture", depth=2, log_event=True)

        assert len(node.executioner_event_logs) == 2
        assert node.executioner_event_logs[0]["event_type"] == "upper_envelope_update"
        assert node.executioner_event_logs[0]["score"] == 0.70
        assert node.executioner_event_logs[1]["score"] == 0.80

        is_mono, gain = node.verify_monotonic_progression(log_event=True)
        assert is_mono is True
        assert math.isclose(gain, 0.10, abs_tol=1e-6)
        assert len(node.executioner_event_logs) == 3
        assert node.executioner_event_logs[2]["event_type"] == "watchdog_monotonicity"

    def test_information_governor_appends_to_learning_state(self):
        node = SwarmLearningStateNode(node_id="node_gov")
        gov = InformationGovernor()

        src = EvidenceSource(
            source_id="src_test",
            title="High-rigor tabular benchmark",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=True,
            relevance_score=0.9,
            methodological_rigor=0.9,
        )
        gov.grade_source(src, record=True, state_node=node)
        assert len(node.governor_event_logs) == 1
        assert node.governor_event_logs[0]["event_type"] == "source_vetting"
        assert node.governor_event_logs[0]["accepted"] is True

        gov.check_exploration_entropy(
            lane_counts={"architecture": 2, "features": 2, "objectives": 2, "optimization": 2},
            state_node=node,
        )
        assert len(node.governor_event_logs) == 2
        assert node.governor_event_logs[1]["event_type"] == "exploration_entropy_audit"

    def test_executioner_stagnation_check_appends_to_learning_state(self):
        node = SwarmLearningStateNode(node_id="node_exec")
        watchdog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)

        stagnant_history = [
            {"grad_norm": 1e-8, "loss": 0.50},
            {"grad_norm": 1e-8, "loss": 0.50},
            {"grad_norm": 1e-8, "loss": 0.50},
        ]
        is_stagnant, trigger, reason = watchdog.audit_branch(
            branch_id="branch_stagnant",
            depth=3,
            visits=5,
            history=stagnant_history,
            state_node=node,
        )
        assert is_stagnant is True
        assert trigger == StagnationTrigger.VANISHING_GRADIENTS
        assert len(node.executioner_event_logs) == 1
        assert node.executioner_event_logs[0]["event_type"] == "stagnation_check"
        assert node.executioner_event_logs[0]["is_stagnant"] is True

    def test_novelty_contract_and_ast_memory_appends(self):
        node = SwarmLearningStateNode(node_id="node_novelty")
        novelty = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_base = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        y_cand = np.array([2.0, 1.0, 4.0, 3.0, 4.5])

        novelty.evaluate_novelty(
            hypothesis_id="hyp_nov",
            candidate_code="def novel_model(): return 'unique_arch'",
            prior_art_corpus=["def prior_model(): return 'standard_arch'"],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
            state_node=node,
        )
        assert len(node.governor_event_logs) == 1
        assert node.governor_event_logs[0]["event_type"] == "novelty_audit"

        mem = StructuralFalsificationMemory()
        code = "def fatal(): return 1 / 0"
        mem.record_fatal_failure(
            code=code,
            stage=SearchStage.POT_VERIFICATION,
            reason="ZeroDivisionError",
            state_node=node,
        )

        assert len(node.executioner_event_logs) == 1
        assert node.executioner_event_logs[0]["event_type"] == "ast_falsification_recorded"

        # Check filter also logs blocked
        is_blocked, _ = mem.check_hard_filter(code, state_node=node)
        assert is_blocked is True
        assert len(node.executioner_event_logs) == 2
        assert node.executioner_event_logs[1]["event_type"] == "ast_falsification_blocked"


class TestTAPAScoreNumericalDefinition:
    """Test suite for TAPAScore numerical calculations in Modality A and Modality B."""

    def test_tapa_score_hard_gates(self):
        # Proof failure zeroes out score immediately
        s_proof_fail = compute_tapa_score(
            r_proof=0.0,
            r_legality=1.0,
            r_stat=5.0,
            delta_robust=0.10,
            var_ratio=0.90,
        )
        assert s_proof_fail == 0.0

        # Legality failure zeroes out score immediately
        s_legal_fail = compute_tapa_score(
            r_proof=1.0,
            r_legality=0.0,
            r_stat=5.0,
            delta_robust=0.10,
            var_ratio=0.90,
        )
        assert s_legal_fail == 0.0

    def test_tapa_score_variance_and_robustness_penalties(self):
        # Variance collapse (var_ratio < 0.50) zeroes out variance pillar
        s_collapsed = compute_tapa_score(
            r_proof=1.0,
            r_legality=1.0,
            r_stat=2.0,
            delta_robust=0.01,
            var_ratio=0.20,  # Collapsed towards constant mean
        )
        # Optimal variance
        s_optimal = compute_tapa_score(
            r_proof=1.0,
            r_legality=1.0,
            r_stat=2.0,
            delta_robust=0.01,
            var_ratio=1.0,
        )
        assert s_collapsed < s_optimal

        # Sub-zero robustness margin (loses to a null model)
        s_beaten_by_null = compute_tapa_score(
            r_proof=1.0,
            r_legality=1.0,
            r_stat=2.0,
            delta_robust=-0.01,  # Worse than a null model
            var_ratio=1.0,
        )
        assert s_beaten_by_null < s_optimal

    def test_tapa_score_scalar_bounds(self):
        s_100 = compute_tapa_score(
            r_proof=1.0,
            r_legality=1.0,
            r_stat=10.0,
            delta_robust=0.02,
            var_ratio=1.0,
            scale_to_100=True,
        )
        assert 90.0 <= s_100 <= 100.0

        s_norm = compute_tapa_score(
            r_proof=1.0,
            r_legality=1.0,
            r_stat=10.0,
            delta_robust=0.02,
            var_ratio=1.0,
            scale_to_100=False,
        )
        assert 0.90 <= s_norm <= 1.0

    def test_tapa_score_modality_b_sparse_autoencoders(self):
        # Dead features >= 1% hard-fails immediately
        s_dead = compute_tapa_score_modality_b(
            r_proof=1.0,
            r_legality=1.0,
            l0_sparsity=32.0,
            l0_target=32.0,
            loss_recovery=0.96,
            dead_feature_ratio=0.015,  # 1.5% > 1% threshold
        )
        assert s_dead == 0.0

        # Loss recovery < 80% hard-fails immediately
        s_rec_fail = compute_tapa_score_modality_b(
            r_proof=1.0,
            r_legality=1.0,
            l0_sparsity=32.0,
            l0_target=32.0,
            loss_recovery=0.75,  # 75% < 80% floor
            dead_feature_ratio=0.002,
        )
        assert s_rec_fail == 0.0

        # Valid SAE candidate
        s_sae_pass = compute_tapa_score_modality_b(
            r_proof=1.0,
            r_legality=1.0,
            l0_sparsity=33.0,
            l0_target=32.0,
            loss_recovery=0.96,  # 96% loss recovery
            dead_feature_ratio=0.002,  # 0.2% dead features
            r_stat=1.5,
        )
        assert 80.0 <= s_sae_pass <= 100.0
