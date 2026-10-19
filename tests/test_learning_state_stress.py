"""Adversarial stress test suite for Milestone M3 (Swarm Learning State Node).

Evaluates numerical and scale properties under pathological, degenerate,
and extreme high-throughput conditions.
"""

import math
import tempfile
import time
from pathlib import Path
import pytest

from frame.schemas.learning_state import SwarmLearningStateNode, TrajectoryPoint
from frame.brain.learning_state import save_learning_state, load_learning_state


class TestAdversarialShannonEpistemicEntropy:
    """Stress tests for Shannon epistemic entropy under degenerate and pathological score sets."""

    def test_high_volume_identical_scores_zero_entropy(self):
        """High volume (10,000) of identical scores must yield exact 0.0 entropy."""
        test_values = [0.0, 0.5, 1.0, -1000.0, 1000.0, 1e-12, 1e8]
        for val in test_values:
            node = SwarmLearningStateNode(node_id=f"ident_{val}")
            for i in range(10000):
                node.record_evaluation(f"h_{i}", val, "architecture")
            
            ent_norm = node.compute_epistemic_entropy(num_bins=5, normalized=True)
            ent_raw = node.compute_epistemic_entropy(num_bins=5, normalized=False)
            assert ent_norm == 0.0, f"Expected exact 0.0 for val={val}, got {ent_norm}"
            assert ent_raw == 0.0, f"Expected exact 0.0 for val={val}, got {ent_raw}"

    def test_identical_within_float_tolerance(self):
        """Scores varying by <= 1e-11 (within 1e-9 tolerance) must yield exact 0.0 entropy."""
        node = SwarmLearningStateNode(node_id="ident_tol")
        for i in range(5000):
            node.record_evaluation(f"h_{i}", 0.5 + i * 1e-15, "optimization")
        
        ent = node.compute_epistemic_entropy(num_bins=5)
        assert ent == 0.0

    def test_single_point_and_empty_evaluations(self):
        """Empty and single-point evaluations must yield exact 0.0 entropy."""
        node_empty = SwarmLearningStateNode(node_id="empty")
        assert node_empty.compute_epistemic_entropy() == 0.0
        assert node_empty.compute_epistemic_entropy(num_bins=10) == 0.0

        for val in [0.0, -999.0, 0.5, 1e-12, 1e8]:
            node_single = SwarmLearningStateNode(node_id=f"single_{val}")
            node_single.record_evaluation("h_0", val, "features")
            assert node_single.compute_epistemic_entropy() == 0.0

    def test_uniform_distribution_across_bins_maximum_entropy(self):
        """Uniform distribution across all bins must yield maximum normalized entropy (1.0)."""
        # Test 5 bins, 10 bins, 20 bins with 10,000 points
        for num_bins in [5, 10, 20]:
            node = SwarmLearningStateNode(node_id=f"uniform_{num_bins}")
            n_points = 10000
            for i in range(n_points):
                bin_idx = i % num_bins
                # Place score in the center of bin_idx in [0, 1]
                score = (bin_idx + 0.5) / num_bins
                node.record_evaluation(f"h_{i}", score, "objectives")
            
            ent_norm = node.compute_epistemic_entropy(num_bins=num_bins, normalized=True)
            assert math.isclose(ent_norm, 1.0, abs_tol=1e-7), f"Expected 1.0 for {num_bins} bins, got {ent_norm}"

            ent_raw = node.compute_epistemic_entropy(num_bins=num_bins, normalized=False)
            assert math.isclose(ent_raw, math.log(num_bins), abs_tol=1e-7)

    def test_arbitrary_scale_negative_to_positive(self):
        """Arbitrary scale [-1000 to +1000] partitions and computes entropy without crash or NaN."""
        node = SwarmLearningStateNode(node_id="scale_neg_pos")
        num_bins = 5
        # Uniform spread across [-1000, 1000]
        # Range width = 2000, bin width = 400
        for i in range(5000):
            bin_idx = i % num_bins
            score = -1000.0 + (bin_idx + 0.5) * 400.0  # -800, -400, 0, 400, 800
            node.record_evaluation(f"h_{i}", score, "architecture")
        
        ent_norm = node.compute_epistemic_entropy(num_bins=num_bins, normalized=True)
        assert math.isclose(ent_norm, 1.0, abs_tol=1e-7)

        # Monotonicity and net gain on this scale
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        # Upper envelope will peak at 800.0, started at -800.0 -> gain 1600.0
        assert math.isclose(gain, 1600.0, abs_tol=1e-5)

    def test_arbitrary_scale_micro_to_macro(self):
        """Arbitrary scale [1e-12 to 1e8] spanning 20 orders of magnitude."""
        node = SwarmLearningStateNode(node_id="scale_micro_macro")
        num_bins = 5
        # Dynamic range [1e-12, 1e8]
        low = 1e-12
        high = 1e8
        width = (high - low) / num_bins
        for i in range(5000):
            bin_idx = i % num_bins
            score = low + (bin_idx + 0.5) * width
            node.record_evaluation(f"h_{i}", score, "features")

        ent_norm = node.compute_epistemic_entropy(num_bins=num_bins, normalized=True)
        assert math.isclose(ent_norm, 1.0, abs_tol=1e-7)

        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain > 0.0

    def test_boundary_values_at_exact_interval_edges(self):
        """Scores exactly at 0.0 and 1.0 (or min and max) partition cleanly into bins."""
        node = SwarmLearningStateNode(node_id="edge_values")
        node.record_evaluation("h_min", 0.0, "arch")
        node.record_evaluation("h_max", 1.0, "arch")
        ent = node.compute_epistemic_entropy(num_bins=5)
        # Bins 0 and 4 each have 1 score
        # p_0 = 0.5, p_4 = 0.5 -> H = -2 * 0.5 * ln(0.5) = ln(2)
        expected_norm = math.log(2) / math.log(5)
        assert math.isclose(ent, expected_norm, abs_tol=1e-7)

    def test_pathological_num_bins_inputs(self):
        """Pathological num_bins inputs (0, 1, negative) handle gracefully returning 0.0."""
        node = SwarmLearningStateNode(node_id="pathological_bins")
        node.record_evaluation("h1", 0.1, "arch")
        node.record_evaluation("h2", 0.9, "arch")
        assert node.compute_epistemic_entropy(num_bins=1) == 0.0
        assert node.compute_epistemic_entropy(num_bins=0) == 0.0
        assert node.compute_epistemic_entropy(num_bins=-5) == 0.0


class TestLargeScaleStressAndPersistence:
    """Stress tests on large-scale (10,000+ points) trajectories and atomic persistence."""

    def test_10k_evaluations_performance_and_upper_envelope(self):
        """10,000 evaluations append and upper envelope calculation completes with high throughput."""
        node = SwarmLearningStateNode(node_id="stress_10k")
        n = 10000

        t0 = time.perf_counter()
        for i in range(n):
            score = 0.2 + (i / n) * 0.7  # Monotonically increasing from 0.2 to 0.9
            node.record_evaluation(f"hyp_{i}", score, "features", depth=i % 8)
        t_append = time.perf_counter() - t0

        # Throughput requirement: > 20,000 evaluations / second (< 0.5s for 10k)
        assert t_append < 1.0, f"10k appends took {t_append:.3f}s, expected < 1.0s"
        assert node.total_budget_n == n
        assert len(node.trajectories) == n

        # Upper envelope curve extraction
        t0 = time.perf_counter()
        curve = node.get_upper_envelope_curve()
        t_curve = time.perf_counter() - t0
        assert len(curve) == n
        assert t_curve < 0.1, f"Curve extraction took {t_curve:.3f}s, expected < 0.1s"

        # Monotonicity verification
        t0 = time.perf_counter()
        is_mono, gain = node.verify_monotonic_progression()
        t_mono = time.perf_counter() - t0
        assert is_mono is True
        assert math.isclose(gain, 0.7, abs_tol=1e-4)
        assert t_mono < 0.1, f"Monotonicity verification took {t_mono:.3f}s"

    def test_10k_evaluations_atomic_persistence_roundtrip(self, tmp_path: Path):
        """10,000 evaluations atomic save and reload round-trip preserves all fields."""
        node = SwarmLearningStateNode(node_id="stress_persist_10k")
        n = 10000
        for i in range(n):
            score = 0.1 + ((i * 13) % 100) * 0.008
            node.record_evaluation(f"hyp_{i}", score, "lane_" + str(i % 4), depth=i % 5)
        
        target_file = tmp_path / "learning_state_10k.json"

        # Save to disk
        t0 = time.perf_counter()
        saved_path = node.save_to_json(target_file)
        t_save = time.perf_counter() - t0
        assert Path(saved_path).is_file()
        assert t_save < 2.0, f"Save took {t_save:.3f}s"

        # Verify no temporary files remain
        tmp_files = list(tmp_path.glob("*.tmp.*"))
        assert len(tmp_files) == 0, f"Temporary files were not cleaned up: {tmp_files}"

        # Load from disk
        t0 = time.perf_counter()
        loaded = SwarmLearningStateNode.load_from_json(target_file)
        t_load = time.perf_counter() - t0
        assert t_load < 2.0, f"Load took {t_load:.3f}s"

        # Assert data integrity
        assert loaded.node_id == node.node_id
        assert loaded.total_budget_n == n
        assert len(loaded.trajectories) == n
        assert loaded.trajectories[0].hypothesis_id == "hyp_0"
        assert loaded.trajectories[-1].hypothesis_id == f"hyp_{n-1}"
        assert loaded.trajectories[-1].step == n
        assert loaded.trajectories[-1].upper_envelope_quality == node.trajectories[-1].upper_envelope_quality

        # Verify monotonic progression matches
        mono_orig, gain_orig = node.verify_monotonic_progression()
        mono_loaded, gain_loaded = loaded.verify_monotonic_progression()
        assert mono_orig == mono_loaded
        assert math.isclose(gain_orig, gain_loaded, abs_tol=1e-7)

        # Verify entropy matches
        ent_orig = node.compute_epistemic_entropy()
        ent_loaded = loaded.compute_epistemic_entropy()
        assert math.isclose(ent_orig, ent_loaded, abs_tol=1e-7)

    def test_brain_module_atomic_helpers_with_large_dataset(self, tmp_path: Path):
        """Brain module save_learning_state / load_learning_state works with large datasets."""
        node = SwarmLearningStateNode(node_id="brain_helper_stress")
        for i in range(5000):
            node.record_evaluation(f"h_{i}", 0.5 + (i / 10000.0), "opt")
        
        target = tmp_path / "brain_stress.json"
        path_out = save_learning_state(node, target)
        assert Path(path_out).is_file()

        loaded = load_learning_state(target)
        assert loaded.node_id == "brain_helper_stress"
        assert loaded.total_budget_n == 5000
        assert len(loaded.trajectories) == 5000
