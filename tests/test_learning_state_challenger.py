"""Empirical Challenger stress tests for SwarmLearningStateNode (Milestone M3).

Authored by m3_challenger_1.
Adversarially challenges SwarmLearningStateNode with:
1. Chaotic and adversarial score sequences (alternating spikes, sudden drops, negative scores, zero scores).
2. Strict non-decreasing upper envelope invariance across all step queries.
3. Incremental net gain vs theoretical delta and floating-point error stability.
4. Large volume appends to active branches and dual log sinks (governor_event_logs, executioner_event_logs).
5. Boundary conditions, numerical edge cases (NaN, Inf), and transactional state consistency.
"""

from datetime import datetime
import json
import math
from pathlib import Path
import random
import time
from typing import List, Tuple
import pytest
from pydantic import ValidationError

from frame.brain import learning_state as brain_learning_state
from frame.schemas.learning_state import SwarmLearningStateNode, TrajectoryPoint


class TestChaoticAndAdversarialScoreSequences:
    """Stress tests evaluating SwarmLearningStateNode across chaotic, alternating,

    dropping, negative, and zero score distributions.
    """

    def test_strictly_alternating_high_frequency_spikes(self):
        """Alternating spikes (+M, -M) with increasing amplitudes.

        Verifies the upper envelope strictly locks to cumulative peaks and never drops.
        """
        node = SwarmLearningStateNode(node_id="node_alt_spikes")
        num_cycles = 100
        raw_scores: List[float] = []

        for i in range(1, num_cycles + 1):
            positive_spike = float(i * 10)
            negative_spike = float(-i * 10)

            raw_scores.append(positive_spike)
            node.record_evaluation(f"hyp_pos_{i}", positive_spike, "architecture", depth=1)

            raw_scores.append(negative_spike)
            node.record_evaluation(f"hyp_neg_{i}", negative_spike, "optimization", depth=1)

        assert node.total_budget_n == num_cycles * 2
        assert len(node.trajectories) == num_cycles * 2

        curve = node.get_upper_envelope_curve()
        assert len(curve) == num_cycles * 2

        # Verify envelope matches theoretical running max at every step
        running_max = -float("inf")
        for idx, (step, env_val) in enumerate(curve):
            running_max = max(running_max, raw_scores[idx])
            assert step == idx + 1
            assert math.isclose(env_val, running_max, abs_tol=1e-9)

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        expected_gain = float(num_cycles * 10 - 10)  # max peak (1000) - first score (10)
        assert math.isclose(gain, expected_gain, abs_tol=1e-9)

    def test_sudden_catastrophic_drops(self):
        """Early positive peak followed by deep negative scores and near-zero values.

        Verifies the upper envelope remains unaffected by subsequent regressions.
        """
        node = SwarmLearningStateNode(node_id="node_catastrophic_drops")
        # Step 1: baseline
        node.record_evaluation("h_base", 50.0, "objectives")
        # Step 2: high peak
        node.record_evaluation("h_peak", 1000.0, "objectives")

        # 200 consecutive severe regressions
        catastrophic_scores = [-1e6, -500.0, -1e8, -0.01, -99999.0, 0.0, 10.0, -50.0] * 25
        for idx, s in enumerate(catastrophic_scores, 3):
            node.record_evaluation(f"h_drop_{idx}", s, "objectives")

        assert node.total_budget_n == 202
        curve = node.get_upper_envelope_curve()

        # Step 1 envelope is 50.0, step 2 onwards must be strictly 1000.0
        assert math.isclose(curve[0][1], 50.0, abs_tol=1e-9)
        for step, env in curve[1:]:
            assert math.isclose(env, 1000.0, abs_tol=1e-9)

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isclose(gain, 950.0, abs_tol=1e-9)

    def test_strictly_negative_score_sequences(self):
        """Sequences strictly in the negative domain (e.g., negative log loss).

        Verifies upper envelope progression and net gain calculation in negative space.
        """
        node = SwarmLearningStateNode(node_id="node_negative_domain")
        neg_scores = [-1000.0, -750.5, -800.0, -600.25, -600.25, -450.0, -500.0, -100.125]
        for idx, s in enumerate(neg_scores, 1):
            node.record_evaluation(f"h_neg_{idx}", s, "features")

        curve = node.get_upper_envelope_curve()
        expected_envelope = [-1000.0, -750.5, -750.5, -600.25, -600.25, -450.0, -450.0, -100.125]

        for idx, (step, env) in enumerate(curve):
            assert step == idx + 1
            assert math.isclose(env, expected_envelope[idx], abs_tol=1e-9)

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        # Gain: (-100.125) - (-1000.0) = 899.875
        expected_gain = -100.125 - (-1000.0)
        assert math.isclose(gain, expected_gain, abs_tol=1e-9)

    def test_pure_zero_and_zero_crossing_sequences(self):
        """Pure zeros followed by negative-to-positive zero crossings.

        Verifies numerical stability when scores cross zero.
        """
        node_zeros = SwarmLearningStateNode(node_id="node_pure_zeros")
        for i in range(50):
            node_zeros.record_evaluation(f"h_zero_{i}", 0.0, "architecture")

        is_mono, gain = node_zeros.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0
        assert node_zeros.compute_epistemic_entropy() == 0.0

        # Crossing zero: negative -> zero -> positive
        node_crossing = SwarmLearningStateNode(node_id="node_zero_crossing")
        crossing_scores = [-25.0, -10.0, 0.0, 0.0, 15.0, -5.0, 30.0]
        for idx, s in enumerate(crossing_scores, 1):
            node_crossing.record_evaluation(f"h_cross_{idx}", s, "architecture")

        is_mono_cross, gain_cross = node_crossing.verify_monotonic_progression()
        assert is_mono_cross is True
        # Gain: 30.0 - (-25.0) = 55.0
        assert math.isclose(gain_cross, 55.0, abs_tol=1e-9)

    def test_chaotic_pseudo_random_walk_invariants(self):
        """1000 steps of pseudo-random walk with heavy fluctuations.

        Checks invariant at every single step: E(t) = max(s_1, ..., s_t).
        """
        random.seed(1337)
        node = SwarmLearningStateNode(node_id="node_random_walk")
        current_val = 0.0
        raw_history: List[float] = []

        for t in range(1, 1001):
            # Heavy tailed jump
            step_jump = random.gauss(0, 5.0)
            if random.random() < 0.05:
                step_jump *= 10.0  # rare massive jump
            current_val += step_jump
            raw_history.append(current_val)

            pt = node.record_evaluation(f"h_rw_{t}", current_val, "optimization")

            # Check point invariant immediately
            expected_max = max(raw_history)
            assert math.isclose(pt.upper_envelope_quality, expected_max, abs_tol=1e-9)

        curve = node.get_upper_envelope_curve()
        assert len(curve) == 1000

        # Invariant: curve is strictly non-decreasing
        for i in range(1, len(curve)):
            assert curve[i][1] >= curve[i - 1][1]

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        expected_gain = max(raw_history) - raw_history[0]
        assert math.isclose(gain, expected_gain, abs_tol=1e-9)


class TestUpperEnvelopeMonotonicityInvariants:
    """Rigorous mathematical tests ensuring the upper envelope is strictly

    non-decreasing across all step queries and configurations.
    """

    def test_upper_envelope_non_decreasing_across_all_step_queries(self):
        """Exhaustive invariant check: for any arbitrary score sequence,

        E(t) >= E(t-1) for all t in [2, N].
        """
        test_patterns = [
            # Sawtooth
            [float(i % 5) for i in range(50)],
            # Reverse sorted
            [100.0 - float(i) for i in range(100)],
            # Strictly increasing
            [float(i * 1.5) for i in range(60)],
            # Sinusoidal oscillation
            [math.sin(i * 0.2) * 50.0 for i in range(80)],
            # Clustered duplicate spikes
            [10.0, 10.0, 5.0, 10.0, 20.0, 20.0, 15.0, 20.0, 30.0],
        ]

        for p_idx, pattern in enumerate(test_patterns):
            node = SwarmLearningStateNode(node_id=f"node_pattern_{p_idx}")
            for idx, s in enumerate(pattern, 1):
                node.record_evaluation(f"h_{idx}", s, "features")

            curve = node.get_upper_envelope_curve()
            for step_idx in range(1, len(curve)):
                prev_step, prev_val = curve[step_idx - 1]
                curr_step, curr_val = curve[step_idx]
                assert curr_step == prev_step + 1
                assert curr_val >= prev_val, (
                    f"Monotonicity violation at step {curr_step}: {curr_val} < {prev_val}"
                )

            is_mono, _ = node.verify_monotonic_progression()
            assert is_mono is True

    def test_envelope_curve_matches_trajectories_exactly(self):
        """Verifies 1:1 parity between get_upper_envelope_curve and trajectories list."""
        node = SwarmLearningStateNode(node_id="node_parity")
        for i in range(30):
            node.record_evaluation(f"h_{i}", float(i ** 2 - 10 * i), "objectives")

        curve = node.get_upper_envelope_curve()
        assert len(curve) == len(node.trajectories)
        for idx, pt in enumerate(node.trajectories):
            assert curve[idx][0] == pt.step
            assert curve[idx][1] == pt.upper_envelope_quality

    def test_tampered_envelope_detected_by_verification(self):
        """Adversarially injects an artificial dip in upper envelope to verify

        that verify_monotonic_progression correctly rejects non-monotonic trajectories.
        """
        node = SwarmLearningStateNode(node_id="node_tampered")
        node.record_evaluation("h1", 10.0, "architecture")
        node.record_evaluation("h2", 20.0, "architecture")
        node.record_evaluation("h3", 15.0, "architecture")

        # Invariant should be (True, 10.0)
        assert node.verify_monotonic_progression()[0] is True

        # Maliciously corrupt step 3's envelope to be lower than step 2
        node.trajectories[2] = TrajectoryPoint(
            hypothesis_id="h3",
            step=3,
            quality_score=15.0,
            lane="architecture",
            upper_envelope_quality=18.0,  # Below step 2 (20.0)!
        )

        is_mono, gain = node.verify_monotonic_progression(tolerance=1e-9)
        assert is_mono is False
        # Gain is 18.0 - 10.0 = 8.0
        assert math.isclose(gain, 8.0, abs_tol=1e-9)

    def test_custom_monotonicity_tolerances(self):
        """Verifies monotonicity behavior under tight vs loose numerical tolerances."""
        node = SwarmLearningStateNode(node_id="node_tolerances")
        pt1 = TrajectoryPoint(hypothesis_id="h1", step=1, quality_score=1.0, lane="l", upper_envelope_quality=1.0)
        # Slight decrease within 1e-4 but beyond 1e-6
        pt2 = TrajectoryPoint(hypothesis_id="h2", step=2, quality_score=0.99995, lane="l", upper_envelope_quality=0.99995)
        node.trajectories = [pt1, pt2]
        node.total_budget_n = 2

        # Under strict tolerance 1e-6: non-monotonic
        is_strict, _ = node.verify_monotonic_progression(tolerance=1e-6)
        assert is_strict is False

        # Under loose tolerance 1e-3: accepted as monotonic
        is_loose, _ = node.verify_monotonic_progression(tolerance=1e-3)
        assert is_loose is True


class TestIncrementalNetGainPrecisionAndStability:
    """Tests evaluating floating-point arithmetic quirks, representation errors,

    and precision preservation in net gain calculations.
    """

    def test_floating_point_representation_quirks(self):
        """Verifies classical IEEE 754 precision issues (e.g. 0.3 - 0.1, 0.7 - 0.1)."""
        node1 = SwarmLearningStateNode(node_id="node_fp_03_01")
        node1.record_evaluation("h1", 0.1, "features")
        node1.record_evaluation("h2", 0.3, "features")
        is_mono1, gain1 = node1.verify_monotonic_progression()
        assert is_mono1 is True
        # In raw IEEE 754, 0.3 - 0.1 == 0.19999999999999998
        # The schema uses round(gain, 12), which must produce exact 0.2
        assert gain1 == 0.2

        node2 = SwarmLearningStateNode(node_id="node_fp_08_01")
        node2.record_evaluation("h1", 0.1, "features")
        node2.record_evaluation("h2", 0.8, "features")
        is_mono2, gain2 = node2.verify_monotonic_progression()
        assert is_mono2 is True
        assert gain2 == 0.7

    def test_sub_tolerance_micro_gains(self):
        """Tests gains near or below the default tolerance threshold (1e-9)."""
        node = SwarmLearningStateNode(node_id="node_micro_gain")
        node.record_evaluation("h1", 1.0, "optimization")
        # Step with 1e-10 increment
        node.record_evaluation("h2", 1.0 + 1e-10, "optimization")

        # With default tolerance=1e-9, math.isclose(1e-10, 0.0, abs_tol=1e-9) is True -> returns 0.0
        is_mono, gain_default = node.verify_monotonic_progression(tolerance=1e-9)
        assert is_mono is True
        assert gain_default == 0.0

        # With tighter tolerance=1e-12, gain must reflect exact 1e-10
        is_mono_tight, gain_tight = node.verify_monotonic_progression(tolerance=1e-12)
        assert is_mono_tight is True
        assert math.isclose(gain_tight, 1e-10, abs_tol=1e-13)

    def test_large_magnitude_gain_stability(self):
        """Verifies net gain calculation when scores have large absolute magnitudes."""
        node = SwarmLearningStateNode(node_id="node_large_mag")
        node.record_evaluation("h1", 1e12, "architecture")
        node.record_evaluation("h2", 1e12 + 5000.25, "architecture")
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isclose(gain, 5000.25, abs_tol=1e-2)

    def test_strictly_decreasing_scores_yield_exact_zero_gain(self):
        """When all subsequent evaluations are inferior to the initial evaluation,

        upper envelope remains flat and net incremental gain is strictly 0.0.
        """
        node = SwarmLearningStateNode(node_id="node_strictly_decreasing")
        node.record_evaluation("h0", 100.0, "architecture")
        for i in range(1, 25):
            node.record_evaluation(f"h_{i}", 100.0 - float(i) * 3.5, "architecture")

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0
        assert node.trajectories[-1].upper_envelope_quality == 100.0

    def test_all_identical_scores_zero_gain(self):
        """When all scores are identical, gain is 0.0 and entropy is 0.0."""
        node = SwarmLearningStateNode(node_id="node_identical")
        for i in range(30):
            node.record_evaluation(f"h_{i}", 42.125, "objectives")

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0
        assert node.compute_epistemic_entropy() == 0.0


class TestHighVolumeAppendsAndDualLogSinks:
    """Stress tests evaluating high-volume data ingestion across active branches,

    Information Governor audit logs, and Executioner event logs.
    """

    def test_active_branches_massive_appends(self):
        """Appends 20,000 active branches and validates integrity, lookups, and serialization."""
        node = SwarmLearningStateNode(node_id="node_mass_branches")
        num_branches = 20000

        t0 = time.perf_counter()
        for i in range(num_branches):
            node.active_branches.append(f"branch_mcts_lane_{i % 4}_id_{i}")
        t1 = time.perf_counter()

        assert len(node.active_branches) == num_branches
        assert node.active_branches[0] == "branch_mcts_lane_0_id_0"
        assert node.active_branches[-1] == f"branch_mcts_lane_{(num_branches - 1) % 4}_id_{num_branches - 1}"
        assert (t1 - t0) < 1.0, f"Appends took {t1 - t0:.2f}s, expected < 1.0s"

    def test_dual_log_sinks_massive_appends(self):
        """Appends 25,000 Governor audit logs and 25,000 Executioner event logs

        with realistic nested JSON payloads.
        """
        node = SwarmLearningStateNode(node_id="node_dual_logs")
        num_logs = 25000

        t0 = time.perf_counter()
        for i in range(num_logs):
            node.governor_event_logs.append({
                "step": i,
                "event": "GOVERNOR_AUDIT",
                "verdict": "APPROVED" if i % 3 != 0 else "REJECTED_LIT_PROXIMITY",
                "score": round(0.40 + (i % 60) * 0.01, 4),
                "source_count": 5,
            })

            node.executioner_event_logs.append({
                "step": i,
                "event_type": "AUDIT_PASS" if i % 5 != 0 else "ORTHOGONAL_WARM_RESTART",
                "branch_id": f"branch_{i % 100}",
                "metrics": {"norm_grad": 1e-4, "loss_curvature": 1e-3, "ast_jaccard": 0.45},
            })
        t1 = time.perf_counter()

        assert len(node.governor_event_logs) == num_logs
        assert len(node.executioner_event_logs) == num_logs
        assert (t1 - t0) < 2.0, f"Logging took {t1 - t0:.2f}s, expected < 2.0s"

        summary = node.export_summary()
        assert summary["governor_events_count"] == num_logs
        assert summary["executioner_events_count"] == num_logs

    def test_combined_high_volume_persistence_roundtrip(self, tmp_path: Path):
        """Full stress lifecycle: 5,000 evaluations + 10,000 branches + 15,000 Governor logs

        + 15,000 Executioner logs (45,000 total elements) saved to disk and restored.
        """
        node = SwarmLearningStateNode(node_id="node_combined_stress")

        # 5,000 evaluations
        for i in range(5000):
            node.record_evaluation(
                hypothesis_id=f"hyp_{i}",
                score=0.2 + (i % 500) * 0.0015,
                lane=["architecture", "objectives", "optimization", "features"][i % 4],
                depth=i % 10,
                metadata={"worker_id": f"worker_{i % 8}"},
            )

        # 10,000 branches
        for i in range(10000):
            node.active_branches.append(f"b_{i}")

        # 15,000 governor logs
        for i in range(15000):
            node.governor_event_logs.append({
                "audit_id": i,
                "status": "PASSED",
                "details": {"score": 0.88, "notes": "audit verified"},
            })

        # 15,000 executioner logs
        for i in range(15000):
            node.executioner_event_logs.append({
                "event_id": i,
                "action": "MONITOR",
                "stagnation": False,
            })

        file_path = tmp_path / "large_stress_state.json"

        # Benchmark serialization
        t_save_start = time.perf_counter()
        saved_file = brain_learning_state.save_learning_state(node, file_path)
        t_save_end = time.perf_counter()
        assert Path(saved_file).is_file()
        assert (t_save_end - t_save_start) < 4.0, f"Save took {t_save_end - t_save_start:.2f}s"

        # Benchmark deserialization
        t_load_start = time.perf_counter()
        loaded = brain_learning_state.load_learning_state(file_path)
        t_load_end = time.perf_counter()
        assert (t_load_end - t_load_start) < 2.0, f"Load took {t_load_end - t_load_start:.2f}s"

        # Verify full state recovery
        assert loaded.node_id == node.node_id
        assert loaded.total_budget_n == 5000
        assert len(loaded.trajectories) == 5000
        assert len(loaded.active_branches) == 10000
        assert len(loaded.governor_event_logs) == 15000
        assert len(loaded.executioner_event_logs) == 15000

        # Verify summary export on restored node
        summary = loaded.export_summary()
        assert summary["total_budget_n"] == 5000
        assert summary["trajectory_count"] == 5000
        assert summary["active_branches_count"] == 10000
        assert summary["governor_events_count"] == 15000
        assert summary["executioner_events_count"] == 15000
        assert summary["monotonic_progression"] is True


class TestAdversarialBoundaryVulnerabilities:
    """Empirical tests probing boundary conditions, numerical vulnerabilities,

    and transactional integrity flaws.
    """

    def test_nan_score_envelope_poisoning_and_entropy_crash(self):
        """EMPIRICAL VULNERABILITY DISCOVERY:

        When an initial evaluation score is NaN (float('nan')):
        1. curr_env is permanently poisoned to NaN for all future steps due to max(nan, x) == nan.
        2. compute_epistemic_entropy() crashes with ValueError('cannot convert float NaN to integer').
        """
        node = SwarmLearningStateNode(node_id="node_vuln_nan")
        node.record_evaluation("h_nan", float("nan"), "architecture")
        node.record_evaluation("h_valid", 0.95, "architecture")

        curve = node.get_upper_envelope_curve()
        # Both envelope values evaluate to NaN
        assert math.isnan(curve[0][1])
        assert math.isnan(curve[1][1])

        # verify_monotonic_progression evaluates True because `nan < nan - 1e-9` is False
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isnan(gain)

        # Calling compute_epistemic_entropy crashes with ValueError
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            node.compute_epistemic_entropy()

        # Calling export_summary also crashes due to epistemic_entropy computation
        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            node.export_summary()

    def test_inf_score_entropy_crash(self):
        """EMPIRICAL VULNERABILITY DISCOVERY:

        When a score is infinite (+inf or -inf), compute_epistemic_entropy() computes
        width = (inf - low) / bins = inf, leading to (s - low) / width = nan, which
        crashes with ValueError('cannot convert float NaN to integer').
        """
        node_inf = SwarmLearningStateNode(node_id="node_vuln_inf")
        node_inf.record_evaluation("h_norm", 0.5, "architecture")
        node_inf.record_evaluation("h_inf", float("inf"), "architecture")

        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            node_inf.compute_epistemic_entropy()

        node_neginf = SwarmLearningStateNode(node_id="node_vuln_neginf")
        node_neginf.record_evaluation("h_neginf", float("-inf"), "architecture")
        node_neginf.record_evaluation("h_norm", 0.5, "architecture")

        with pytest.raises(ValueError, match="cannot convert float NaN to integer"):
            node_neginf.compute_epistemic_entropy()

    def test_pre_validation_budget_mutation_inconsistency(self):
        """EMPIRICAL VULNERABILITY DISCOVERY:

        In SwarmLearningStateNode.record_evaluation:
        self.total_budget_n += 1 is executed BEFORE TrajectoryPoint(...) schema validation.
        If TrajectoryPoint validation fails (e.g. depth < 0), total_budget_n is permanently
        incremented while trajectories is NOT updated, resulting in desynchronized state
        and skipped step numbers.
        """
        node = SwarmLearningStateNode(node_id="node_vuln_desync")
        assert node.total_budget_n == 0
        assert len(node.trajectories) == 0

        # Attempt invalid record with depth = -1 (violates ge=0 constraint)
        with pytest.raises(ValidationError):
            node.record_evaluation("h_bad", 0.5, "architecture", depth=-1)

        # Budget was incremented to 1 despite validation failure!
        assert node.total_budget_n == 1
        assert len(node.trajectories) == 0  # Inconsistent: budget=1, trajectories=0

        # Next valid evaluation gets step=2 instead of step=1
        pt_next = node.record_evaluation("h_good", 0.8, "architecture", depth=0)
        assert pt_next.step == 2
        assert node.total_budget_n == 2
        assert len(node.trajectories) == 1

    def test_empty_and_single_element_edge_cases(self):
        """Verifies boundary state on empty and single-element state nodes."""
        empty_node = SwarmLearningStateNode(node_id="node_boundary_empty")
        assert empty_node.total_budget_n == 0
        assert empty_node.get_upper_envelope_curve() == []
        is_mono, gain = empty_node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0
        assert empty_node.compute_epistemic_entropy() == 0.0

        single_node = SwarmLearningStateNode(node_id="node_boundary_single")
        single_node.record_evaluation("h1", 0.75, "features")
        assert single_node.total_budget_n == 1
        assert single_node.get_upper_envelope_curve() == [(1, 0.75)]
        is_mono_s, gain_s = single_node.verify_monotonic_progression()
        assert is_mono_s is True
        assert gain_s == 0.0
        assert single_node.compute_epistemic_entropy() == 0.0

    def test_invalid_json_handling(self, tmp_path: Path):
        """Verifies clear error reporting on malformed JSON load."""
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("{ this is not valid json }", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            SwarmLearningStateNode.load_from_json(bad_file)
