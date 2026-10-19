"""Unit and Integration Tests for Milestone M3: Persistent Learning State Node & Test-Time Compute."""

from datetime import datetime
import json
import math
from pathlib import Path
import pytest

from frame.schemas.learning_state import SwarmLearningStateNode, TrajectoryPoint
from frame.brain import learning_state as brain_learning_state
from frame.schemas import (
    SwarmLearningStateNode as SchemaSwarmLearningStateNode,
    TrajectoryPoint as SchemaTrajectoryPoint,
)
from frame.brain import (
    SwarmLearningStateNode as BrainSwarmLearningStateNode,
    TrajectoryPoint as BrainTrajectoryPoint,
    save_learning_state,
    load_learning_state,
)


class TestTrajectoryPointSchema:
    """Tests for TrajectoryPoint Pydantic schema and properties."""

    def test_trajectory_point_instantiation_and_fields(self):
        pt = TrajectoryPoint(
            hypothesis_id="hyp_001",
            step=1,
            quality_score=0.82,
            lane="architecture",
            depth=2,
            upper_envelope_quality=0.82,
            metadata={"source": "mcts_worker_1"},
        )
        assert pt.hypothesis_id == "hyp_001"
        assert pt.step == 1
        assert pt.quality_score == 0.82
        assert pt.lane == "architecture"
        assert pt.depth == 2
        assert pt.upper_envelope_quality == 0.82
        assert pt.metadata == {"source": "mcts_worker_1"}
        assert isinstance(pt.timestamp, str)
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(pt.timestamp)

    def test_trajectory_point_score_alias_property(self):
        pt = TrajectoryPoint(
            hypothesis_id="hyp_002",
            step=2,
            quality_score=0.68,
            lane="objectives",
            upper_envelope_quality=0.82,
        )
        assert pt.score == 0.68
        assert pt.score == pt.quality_score

    def test_trajectory_point_score_constructor_normalization(self):
        pt = TrajectoryPoint(
            hypothesis_id="hyp_003",
            step=1,
            score=0.77,
            lane="features",
            upper_envelope_quality=0.77,
        )
        assert pt.quality_score == 0.77
        assert pt.score == 0.77

    def test_trajectory_point_defaults(self):
        pt = TrajectoryPoint(
            hypothesis_id="hyp_004",
            step=1,
            quality_score=0.55,
            lane="optimization",
            upper_envelope_quality=0.55,
        )
        assert pt.depth == 0
        assert pt.metadata == {}
        assert len(pt.timestamp) > 0

    def test_trajectory_point_extreme_and_negative_scores(self):
        pt_neg = TrajectoryPoint(
            hypothesis_id="hyp_neg",
            step=1,
            quality_score=-42.5,
            lane="architecture",
            upper_envelope_quality=-42.5,
        )
        assert pt_neg.quality_score == -42.5

        pt_large = TrajectoryPoint(
            hypothesis_id="hyp_large",
            step=2,
            quality_score=1e7,
            lane="architecture",
            upper_envelope_quality=1e7,
        )
        assert pt_large.quality_score == 1e7


class TestSwarmLearningStateNodeSchema:
    """Tests for SwarmLearningStateNode structure, active branches, and event logging."""

    def test_node_instantiation_and_defaults(self):
        node = SwarmLearningStateNode(node_id="node_init")
        assert node.node_id == "node_init"
        assert node.total_budget_n == 0
        assert len(node.trajectories) == 0
        assert node.active_branches == []
        assert node.governor_event_logs == []
        assert node.executioner_event_logs == []
        assert node.metadata == {}
        assert isinstance(node.created_at, str)
        assert isinstance(node.updated_at, str)

    def test_active_branches_management(self):
        node = SwarmLearningStateNode(node_id="node_branches")
        node.active_branches.append("branch_arch_01")
        node.active_branches.append("branch_opt_02")
        assert len(node.active_branches) == 2
        assert "branch_arch_01" in node.active_branches
        assert "branch_opt_02" in node.active_branches

    def test_event_logs_recording(self):
        node = SwarmLearningStateNode(node_id="node_logs")
        node.governor_event_logs.append({
            "event": "NOVELTY_AUDIT_APPROVED",
            "hypothesis_id": "hyp_10",
            "verdict": "APPROVED",
        })
        node.executioner_event_logs.append({
            "event_type": "ORTHOGONAL_WARM_RESTART",
            "branch_id": "b_stagnant",
            "trigger": "LOSS_PLATEAU",
            "target_lane": "features",
        })
        assert len(node.governor_event_logs) == 1
        assert len(node.executioner_event_logs) == 1
        assert node.governor_event_logs[0]["hypothesis_id"] == "hyp_10"
        assert node.executioner_event_logs[0]["branch_id"] == "b_stagnant"


class TestEvaluationRecordingAndUpperEnvelope:
    """Tests for record_evaluation and cumulative upper envelope computation."""

    def test_record_evaluation_first_step(self):
        node = SwarmLearningStateNode(node_id="node_rec_1")
        pt = node.record_evaluation(
            hypothesis_id="hyp_first",
            score=0.72,
            lane="architecture",
            depth=1,
            metadata={"seed": 42},
        )
        assert pt.step == 1
        assert pt.quality_score == 0.72
        assert pt.upper_envelope_quality == 0.72
        assert pt.depth == 1
        assert pt.metadata == {"seed": 42}
        assert node.total_budget_n == 1
        assert len(node.trajectories) == 1

    def test_record_evaluation_monotonic_scaling(self):
        node = SwarmLearningStateNode(node_id="node_scaling")
        scores = [0.50, 0.60, 0.55, 0.75, 0.70, 0.85]
        for idx, score in enumerate(scores, 1):
            node.record_evaluation(f"hyp_{idx}", score, "features", depth=idx)

        assert node.total_budget_n == 6
        assert len(node.trajectories) == 6

        expected_envelopes = [0.50, 0.60, 0.60, 0.75, 0.75, 0.85]
        curve = node.get_upper_envelope_curve()
        assert len(curve) == 6
        for idx, (step, env) in enumerate(curve):
            assert step == idx + 1
            assert math.isclose(env, expected_envelopes[idx], abs_tol=1e-5)

    def test_upper_envelope_curve_empty_and_single(self):
        node = SwarmLearningStateNode(node_id="node_empty")
        assert node.get_upper_envelope_curve() == []

        node.record_evaluation("hyp_single", 0.65, "objectives")
        assert node.get_upper_envelope_curve() == [(1, 0.65)]

    def test_strictly_decreasing_scores_preserves_upper_envelope(self):
        node = SwarmLearningStateNode(node_id="node_dec")
        scores = [0.95, 0.80, 0.70, 0.60, 0.50]
        for idx, score in enumerate(scores):
            node.record_evaluation(f"h_{idx}", score, "optimization")

        curve = node.get_upper_envelope_curve()
        assert len(curve) == 5
        for _, env in curve:
            assert math.isclose(env, 0.95, abs_tol=1e-5)


class TestMonotonicProgressionVerification:
    """Tests for verify_monotonic_progression and net incremental gain."""

    def test_verify_empty_and_single_step(self):
        node = SwarmLearningStateNode(node_id="node_mono_empty")
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0

        node.record_evaluation("h1", 0.80, "architecture")
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0

    def test_verify_identical_scores_zero_gain(self):
        node = SwarmLearningStateNode(node_id="node_mono_ident")
        for i in range(5):
            node.record_evaluation(f"h_{i}", 0.62, "features")
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert gain == 0.0

    def test_verify_strictly_increasing_progression(self):
        node = SwarmLearningStateNode(node_id="node_mono_inc")
        node.record_evaluation("h1", 0.40, "optimization", 1)
        node.record_evaluation("h2", 0.65, "optimization", 2)
        node.record_evaluation("h3", 0.90, "optimization", 3)
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isclose(gain, 0.50, abs_tol=1e-5)

    def test_verify_oscillating_raw_scores(self):
        node = SwarmLearningStateNode(node_id="node_mono_osc")
        raw_scores = [0.50, 0.30, 0.70, 0.40, 0.80]
        for idx, s in enumerate(raw_scores):
            node.record_evaluation(f"h_{idx}", s, "objectives")
        is_mono, gain = node.verify_monotonic_progression()
        assert is_mono is True
        assert math.isclose(gain, 0.80 - 0.50, abs_tol=1e-5)

    def test_floating_point_epsilon_stability(self):
        node = SwarmLearningStateNode(node_id="node_mono_eps")
        node.record_evaluation("h1", 0.50000000001, "features")
        node.record_evaluation("h2", 0.50000000002, "features")
        is_mono, _ = node.verify_monotonic_progression(tolerance=1e-9)
        assert is_mono is True


class TestEpistemicEntropyComputation:
    """Tests for compute_epistemic_entropy Shannon formulation."""

    def test_entropy_empty_and_single_point(self):
        node = SwarmLearningStateNode(node_id="node_ent_empty")
        assert node.compute_epistemic_entropy() == 0.0

        node.record_evaluation("h1", 0.75, "architecture")
        assert node.compute_epistemic_entropy() == 0.0

    def test_entropy_identical_scores_zero(self):
        node = SwarmLearningStateNode(node_id="node_ent_ident")
        for i in range(10):
            node.record_evaluation(f"h_{i}", 0.50, "architecture")
        assert math.isclose(node.compute_epistemic_entropy(), 0.0, abs_tol=1e-5)

    def test_entropy_all_scores_single_bin_zero(self):
        node = SwarmLearningStateNode(node_id="node_ent_one_bin")
        for i in range(5):
            node.record_evaluation(f"h_{i}", 0.81, "optimization")
        assert node.compute_epistemic_entropy() == 0.0

    def test_entropy_two_points_distinct_bins(self):
        node = SwarmLearningStateNode(node_id="node_ent_two")
        node.record_evaluation("h1", 0.10, "features")
        node.record_evaluation("h2", 0.90, "features")
        h = node.compute_epistemic_entropy()
        assert h > 0.0
        assert h <= 1.0

    def test_entropy_diverse_scores_high_entropy(self):
        node = SwarmLearningStateNode(node_id="node_ent_diverse")
        scores = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8, 0.95]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"h_{idx}", s, "architecture")
        h = node.compute_epistemic_entropy()
        assert h > 0.5
        assert h <= 1.0

    def test_entropy_uniform_distribution_across_bins(self):
        node = SwarmLearningStateNode(node_id="node_ent_uniform")
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        for idx, s in enumerate(scores):
            node.record_evaluation(f"h_{idx}", s, "objectives")
        h_norm = node.compute_epistemic_entropy(num_bins=5, normalized=True)
        assert math.isclose(h_norm, 1.0, abs_tol=1e-5)

        h_raw = node.compute_epistemic_entropy(num_bins=5, normalized=False)
        assert math.isclose(h_raw, math.log(5), abs_tol=1e-5)

    def test_entropy_bounded_0_to_1(self):
        node = SwarmLearningStateNode(node_id="node_ent_bounded")
        for i in range(25):
            node.record_evaluation(f"h_{i}", (i % 5) * 0.2, "architecture")
        h = node.compute_epistemic_entropy()
        assert 0.0 <= h <= 1.0

    def test_entropy_dynamic_updates_across_evaluations(self):
        node = SwarmLearningStateNode(node_id="node_ent_dynamic")
        node.record_evaluation("h1", 0.5, "architecture")
        node.record_evaluation("h2", 0.5, "architecture")
        assert node.compute_epistemic_entropy() == 0.0
        node.record_evaluation("h3", 0.9, "architecture")
        assert node.compute_epistemic_entropy() > 0.0

    def test_entropy_scores_outside_unit_interval(self):
        node = SwarmLearningStateNode(node_id="node_ent_extreme")
        node.record_evaluation("h_neg", -10.5, "architecture")
        node.record_evaluation("h_pos", 100000.0, "architecture")
        h = node.compute_epistemic_entropy()
        assert h > 0.0
        assert h <= 1.0


class TestPersistenceAndSerialization:
    """Tests for file serialization, atomic saving, and model dump/validation."""

    def test_model_dump_and_json_roundtrip(self):
        node = SwarmLearningStateNode(node_id="node_dump")
        node.record_evaluation("hyp_1", 0.85, "features", depth=1)
        node.active_branches.append("branch_01")
        node.governor_event_logs.append({"audit": "pass"})
        node.executioner_event_logs.append({"action": "monitor"})

        data = node.model_dump()
        assert data["node_id"] == "node_dump"
        assert len(data["trajectories"]) == 1
        assert data["active_branches"] == ["branch_01"]

        json_str = json.dumps(data)
        assert "node_dump" in json_str

        restored = SwarmLearningStateNode.model_validate(json.loads(json_str))
        assert restored.node_id == node.node_id
        assert restored.total_budget_n == 1
        assert len(restored.trajectories) == 1
        assert restored.trajectories[0].quality_score == 0.85
        assert restored.trajectories[0].upper_envelope_quality == 0.85

    def test_save_and_load_from_json_file(self, tmp_path: Path):
        file_path = tmp_path / "state_node.json"
        node = SwarmLearningStateNode(node_id="node_file_io")
        node.record_evaluation("h_a", 0.60, "architecture", depth=1)
        node.record_evaluation("h_b", 0.80, "architecture", depth=2)

        saved_path = node.save_to_json(file_path)
        assert Path(saved_path).is_file()

        loaded = SwarmLearningStateNode.load_from_json(file_path)
        assert loaded.node_id == "node_file_io"
        assert loaded.total_budget_n == 2
        assert len(loaded.trajectories) == 2
        assert loaded.get_upper_envelope_curve() == [(1, 0.60), (2, 0.80)]

    def test_load_from_json_raw_string(self):
        node = SwarmLearningStateNode(node_id="node_str")
        node.record_evaluation("h1", 0.70, "lane_a")
        raw_json = json.dumps(node.model_dump())

        loaded = SwarmLearningStateNode.load_from_json(raw_json)
        assert loaded.node_id == "node_str"
        assert loaded.total_budget_n == 1

    def test_load_from_nonexistent_file_raises(self, tmp_path: Path):
        non_existent = tmp_path / "does_not_exist.json"
        with pytest.raises(FileNotFoundError):
            SwarmLearningStateNode.load_from_json(non_existent)


class TestModuleBridgeAndUtilities:
    """Tests for package exports, brain helper functions, and summary exports."""

    def test_schemas_package_reexports(self):
        assert SchemaSwarmLearningStateNode is SwarmLearningStateNode
        assert SchemaTrajectoryPoint is TrajectoryPoint

    def test_brain_package_reexports(self):
        assert BrainSwarmLearningStateNode is SwarmLearningStateNode
        assert BrainTrajectoryPoint is TrajectoryPoint
        assert callable(save_learning_state)
        assert callable(load_learning_state)

    def test_brain_learning_state_module_helpers(self, tmp_path: Path):
        node = SwarmLearningStateNode(node_id="node_helpers")
        node.record_evaluation("h1", 0.55, "features")
        target_path = tmp_path / "brain_helper_state.json"

        path_out = brain_learning_state.save_to_file(node, target_path)
        assert Path(path_out).is_file()

        loaded = brain_learning_state.load_from_file(target_path)
        assert loaded.node_id == "node_helpers"
        assert loaded.total_budget_n == 1

    def test_export_summary(self):
        node = SwarmLearningStateNode(node_id="node_summary")
        node.record_evaluation("h1", 0.50, "architecture")
        node.record_evaluation("h2", 0.75, "architecture")
        summary = node.export_summary()

        assert summary["node_id"] == "node_summary"
        assert summary["total_budget_n"] == 2
        assert summary["trajectory_count"] == 2
        assert summary["upper_envelope_peak"] == 0.75
        assert summary["monotonic_progression"] is True
        assert math.isclose(summary["monotonic_gain"], 0.25, abs_tol=1e-5)
        assert isinstance(summary["epistemic_entropy"], float)

    def test_export_scaling_data(self):
        node = SwarmLearningStateNode(node_id="node_scaling_data")
        node.record_evaluation("h1", 0.40, "optimization")
        node.record_evaluation("h2", 0.70, "optimization")
        data = brain_learning_state.export_scaling_data(node)

        assert data["node_id"] == "node_scaling_data"
        assert data["total_budget_n"] == 2
        assert data["upper_envelope_curve"] == [(1, 0.40), (2, 0.70)]
        assert data["is_monotonic"] is True
        assert math.isclose(data["net_gain"], 0.30, abs_tol=1e-5)
        assert len(data["trajectories"]) == 2
