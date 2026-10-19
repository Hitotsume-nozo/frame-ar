"""Unit and Integration Tests for Milestone M4: The Executioner Watchdog & Orthogonal Warm-Restart.

Verifies epistemic stagnation detection across M stacks, radical early-stage architecture
safeguards, period-4 orthogonal lane cycling, event persistence in SwarmLearningStateNode,
and boundary threshold invariants.
"""

import json
import time
from typing import Any, Dict, List
import pytest

from frame.schemas.executioner import StagnationTrigger, ExecutionerEvent
from frame.brain.executioner import (
    ExecutionerWatchdog,
    LANE_CYCLE,
    DEFAULT_LANE_PRIORS,
)
from frame.schemas import (
    StagnationTrigger as SchemaStagnationTrigger,
    ExecutionerEvent as SchemaExecutionerEvent,
)
from frame.brain import (
    ExecutionerWatchdog as BrainExecutionerWatchdog,
    StagnationTrigger as BrainStagnationTrigger,
    ExecutionerEvent as BrainExecutionerEvent,
)
from frame.schemas.learning_state import SwarmLearningStateNode


class TestExecutionerSchemas:
    """Tests for StagnationTrigger enum and ExecutionerEvent Pydantic schema."""

    def test_stagnation_trigger_enum_members(self):
        assert StagnationTrigger.NONE.value == "NONE"
        assert StagnationTrigger.VANISHING_GRADIENTS.value == "VANISHING_GRADIENTS"
        assert StagnationTrigger.LOSS_PLATEAU.value == "LOSS_PLATEAU"
        assert StagnationTrigger.CIRCULAR_AST.value == "CIRCULAR_AST"
        assert len(StagnationTrigger) == 4

    def test_executioner_event_default_instantiation(self):
        ev = ExecutionerEvent(branch_id="b_test_01")
        assert ev.event_type == "ORTHOGONAL_WARM_RESTART"
        assert ev.branch_id == "b_test_01"
        assert ev.trigger == StagnationTrigger.NONE
        assert ev.target_lane is None
        assert ev.source_lane is None
        assert ev.details == {}
        assert ev.injected_priors == []
        assert isinstance(ev.timestamp, float)
        assert ev.timestamp > 0

    def test_executioner_event_full_instantiation(self):
        now = time.time()
        ev = ExecutionerEvent(
            event_type="ORTHOGONAL_WARM_RESTART",
            branch_id="b_restarted_123",
            target_lane="features",
            source_lane="architecture",
            trigger=StagnationTrigger.VANISHING_GRADIENTS,
            details={"grad_norm": 1e-8, "steps_stagnant": 3},
            injected_priors=["fourier_feature_mapping", "orthogonal_polynomials"],
            timestamp=now,
        )
        assert ev.branch_id == "b_restarted_123"
        assert ev.target_lane == "features"
        assert ev.source_lane == "architecture"
        assert ev.trigger == StagnationTrigger.VANISHING_GRADIENTS
        assert ev.details["grad_norm"] == 1e-8
        assert len(ev.injected_priors) == 2
        assert ev.timestamp == now

    def test_executioner_event_string_trigger_supported(self):
        ev = ExecutionerEvent(
            branch_id="b_str",
            trigger="CIRCULAR_AST",
        )
        assert ev.trigger == "CIRCULAR_AST"

    def test_executioner_event_extra_fields_ignored(self):
        raw = {
            "branch_id": "b_extra",
            "arbitrary_unmodeled_field": 42,
            "extra_metric": 0.99,
        }
        ev = ExecutionerEvent(**raw)
        assert ev.branch_id == "b_extra"
        assert not hasattr(ev, "arbitrary_unmodeled_field")

    def test_executioner_event_json_serialization_roundtrip(self):
        ev = ExecutionerEvent(
            branch_id="b_roundtrip",
            target_lane="optimization",
            source_lane="objectives",
            trigger=StagnationTrigger.LOSS_PLATEAU,
            details={"plateau_range": 5e-5},
            injected_priors=["cosine_annealing"],
        )
        data = ev.model_dump(mode="json")
        assert data["branch_id"] == "b_roundtrip"
        assert data["trigger"] == "LOSS_PLATEAU"
        assert data["target_lane"] == "optimization"

        restored = ExecutionerEvent.model_validate(data)
        assert restored.branch_id == ev.branch_id
        assert restored.target_lane == ev.target_lane


class TestExecutionerExports:
    """Verifies that all schemas and brain classes are correctly exported at package boundaries."""

    def test_schema_package_exports(self):
        assert SchemaStagnationTrigger is StagnationTrigger
        assert SchemaExecutionerEvent is ExecutionerEvent

    def test_brain_package_exports(self):
        assert BrainExecutionerWatchdog is ExecutionerWatchdog
        assert BrainStagnationTrigger is StagnationTrigger
        assert BrainExecutionerEvent is ExecutionerEvent

    def test_constants_exported(self):
        assert isinstance(LANE_CYCLE, dict)
        assert isinstance(DEFAULT_LANE_PRIORS, dict)
        for lane in ("architecture", "features", "objectives", "optimization"):
            assert lane in LANE_CYCLE
            assert lane in DEFAULT_LANE_PRIORS
            assert len(DEFAULT_LANE_PRIORS[lane]) > 0


class TestStagnationDetection:
    """Tests epistemic stagnation detection logic across M stacks."""

    def test_detect_vanishing_gradients(self):
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        history = [
            {"grad_norm": 5e-7, "loss": 1.2, "ast_tokens": {"x", "y"}},
            {"grad_norm": 4e-7, "loss": 1.1, "ast_tokens": {"x", "z"}},
            {"grad_norm": 1e-7, "loss": 1.0, "ast_tokens": {"x", "w"}},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_vanish", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.VANISHING_GRADIENTS
        assert "Vanishing gradients" in reason

    def test_detect_loss_plateau(self):
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        history = [
            {"grad_norm": 0.05, "loss": 0.42000, "ast_tokens": {"t1"}},
            {"grad_norm": 0.04, "loss": 0.42005, "ast_tokens": {"t2"}},
            {"grad_norm": 0.06, "loss": 0.42002, "ast_tokens": {"t3"}},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_plateau", depth=2, visits=5, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.LOSS_PLATEAU
        assert "Loss curvature plateau" in reason

    def test_detect_circular_ast_mutations(self):
        dog = ExecutionerWatchdog(m_stacks=3, tau_jaccard=0.85)
        # Jaccard similarities:
        # t1 & t2: 96 / 101 = 0.9505 > 0.85
        # t2 & t3: 96 / 98 = 0.9796 > 0.85
        t1 = {f"k_{i}" for i in range(100)}
        t2 = {f"k_{i}" for i in range(96)} | {"new_1"}
        t3 = {f"k_{i}" for i in range(95)} | {"new_1", "new_2"}
        history = [
            {"grad_norm": 0.08, "loss": 0.6, "ast_tokens": t1},
            {"grad_norm": 0.07, "loss": 0.5, "ast_tokens": t2},
            {"grad_norm": 0.09, "loss": 0.4, "ast_tokens": t3},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_circular", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.CIRCULAR_AST
        assert "Circular AST" in reason

    def test_precedence_vanishing_gradients_over_loss_plateau(self):
        """When both vanishing gradients and flat loss occur, vanishing gradients must trigger first."""
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6, epsilon_plateau=1e-4)
        history = [
            {"grad_norm": 1e-8, "loss": 0.500000, "ast_tokens": {"a"}},
            {"grad_norm": 1e-8, "loss": 0.500000, "ast_tokens": {"a"}},
            {"grad_norm": 1e-8, "loss": 0.500000, "ast_tokens": {"a"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_dual", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.VANISHING_GRADIENTS

    def test_healthy_branch_returns_none(self):
        dog = ExecutionerWatchdog(m_stacks=3)
        history = [
            {"grad_norm": 0.10, "loss": 1.5, "ast_tokens": {"fn_a", "layer_1"}},
            {"grad_norm": 0.08, "loss": 1.1, "ast_tokens": {"fn_b", "layer_2"}},
            {"grad_norm": 0.05, "loss": 0.7, "ast_tokens": {"fn_c", "layer_3"}},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_healthy", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE
        assert "Branch healthy" in reason


class TestStagnationBoundaries:
    """Tests exact threshold mathematical boundaries and edge cases."""

    def test_gradient_norm_strict_inequality_boundary(self):
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        # Exact boundary: grad_norm == 1e-6 -> strict < condition does NOT trigger
        history_exact = [{"grad_norm": 1e-6, "loss": 0.8 - i * 0.1, "ast_tokens": {f"t_{i}"}} for i in range(3)]
        stagnant, trigger, _ = dog.audit_branch("b_exact", depth=3, visits=5, history=history_exact)
        assert trigger != StagnationTrigger.VANISHING_GRADIENTS

        # Slightly below boundary: 9.99e-7 -> triggers
        history_below = [{"grad_norm": 9.99e-7, "loss": 0.8 - i * 0.1, "ast_tokens": {f"t_{i}"}} for i in range(3)]
        stagnant, trigger, _ = dog.audit_branch("b_below", depth=3, visits=5, history=history_below)
        assert stagnant is True
        assert trigger == StagnationTrigger.VANISHING_GRADIENTS

    def test_loss_plateau_inclusive_inequality_boundary(self):
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        # Exact boundary: max - min == 1e-4 -> inclusive <= condition DOES trigger
        history_exact = [
            {"grad_norm": 0.05, "loss": 0.5000, "ast_tokens": {"a"}},
            {"grad_norm": 0.05, "loss": 0.5001, "ast_tokens": {"b"}},
            {"grad_norm": 0.05, "loss": 0.50005, "ast_tokens": {"c"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_plateau_exact", depth=3, visits=5, history=history_exact)
        assert stagnant is True
        assert trigger == StagnationTrigger.LOSS_PLATEAU

        # Just above boundary: max - min == 1.1e-4 -> does NOT trigger
        history_above = [
            {"grad_norm": 0.05, "loss": 0.50000, "ast_tokens": {"a"}},
            {"grad_norm": 0.05, "loss": 0.50011, "ast_tokens": {"b"}},
            {"grad_norm": 0.05, "loss": 0.50005, "ast_tokens": {"c"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_plateau_above", depth=3, visits=5, history=history_above)
        assert trigger != StagnationTrigger.LOSS_PLATEAU

    def test_jaccard_strict_inequality_boundary(self):
        dog = ExecutionerWatchdog(m_stacks=2, tau_jaccard=0.80)
        # 8 out of 10 overlap: Jaccard = 8 / 10 = 0.80
        # Exact boundary: 0.80 -> strict > condition does NOT trigger
        t1 = {f"x_{i}" for i in range(10)}
        t2 = {f"x_{i}" for i in range(8)} | {"extra_1", "extra_2"}
        history_exact = [
            {"grad_norm": 0.05, "loss": 1.0, "ast_tokens": t1},
            {"grad_norm": 0.05, "loss": 0.8, "ast_tokens": t2},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_jaccard_exact", depth=3, visits=5, history=history_exact)
        assert trigger != StagnationTrigger.CIRCULAR_AST

        # 9 out of 10 overlap: Jaccard = 9 / 11 = 0.818 > 0.80 -> triggers
        t3 = {f"x_{i}" for i in range(9)} | {"extra_3"}
        history_above = [
            {"grad_norm": 0.05, "loss": 1.0, "ast_tokens": t1},
            {"grad_norm": 0.05, "loss": 0.8, "ast_tokens": t3},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_jaccard_above", depth=3, visits=5, history=history_above)
        assert stagnant is True
        assert trigger == StagnationTrigger.CIRCULAR_AST

    def test_history_length_shorter_than_m_stacks(self):
        dog = ExecutionerWatchdog(m_stacks=4)
        history_short = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, reason = dog.audit_branch("b_short", depth=3, visits=5, history=history_short)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE
        assert "Insufficient history" in reason

    def test_empty_history_handled(self):
        dog = ExecutionerWatchdog(m_stacks=3)
        stagnant, trigger, reason = dog.audit_branch("b_empty", depth=3, visits=5, history=[])
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE
        assert "Insufficient history" in reason

    def test_sliding_window_ignores_older_stagnant_steps(self):
        dog = ExecutionerWatchdog(m_stacks=3)
        # 6 steps: steps 0-2 stagnant, steps 3-5 healthy
        history = [
            {"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"old"}},
            {"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"old"}},
            {"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"old"}},
            {"grad_norm": 0.10, "loss": 1.2, "ast_tokens": {"new_1"}},
            {"grad_norm": 0.08, "loss": 0.9, "ast_tokens": {"new_2"}},
            {"grad_norm": 0.06, "loss": 0.6, "ast_tokens": {"new_3"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_window", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_missing_or_none_fields_in_history_handled_safely(self):
        dog = ExecutionerWatchdog(m_stacks=3)
        history = [
            {"other_telemetry": 1},
            {"loss": None, "grad_norm": None},
            {"loss": 0.5},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_missing", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE


class TestEarlyStageSafeguard:
    """Tests radical early-stage architecture safeguard suppression."""

    @pytest.mark.parametrize(
        "depth,visits,expected_exempt",
        [
            (0, 0, True),
            (0, 1, True),
            (1, 1, True),
            (1, 2, True),
            (-1, -1, True),
            (-1, 0, True),
            (1, 3, False),
            (2, 2, False),
            (2, 3, False),
            (3, 5, False),
        ],
    )
    def test_safeguard_truth_table(self, depth: int, visits: int, expected_exempt: bool):
        dog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        # Stagnant history that would otherwise trigger vanishing gradients
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, reason = dog.audit_branch("b_sg", depth=depth, visits=visits, history=history)

        if expected_exempt:
            assert stagnant is False
            assert trigger == StagnationTrigger.NONE
            assert "Exempt" in reason or "early" in reason.lower()
        else:
            assert stagnant is True
            assert trigger == StagnationTrigger.VANISHING_GRADIENTS

    def test_custom_safeguard_parameters(self):
        dog = ExecutionerWatchdog(early_depth=3, early_visits=5)
        history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, _, _ = dog.audit_branch("b_custom_sg", depth=3, visits=5, history=history)
        assert stagnant is False

        stagnant, _, _ = dog.audit_branch("b_custom_sg_mature", depth=4, visits=5, history=history)
        assert stagnant is True


class TestOrthogonalWarmRestart:
    """Tests orthogonal lane cycling and warm restart event generation."""

    def test_lane_cycle_mapping(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_cycle_test")

        ev1 = dog.trigger_orthogonal_warm_restart("b1", "architecture", node)
        assert ev1.target_lane == "features"

        ev2 = dog.trigger_orthogonal_warm_restart("b2", "features", node)
        assert ev2.target_lane == "objectives"

        ev3 = dog.trigger_orthogonal_warm_restart("b3", "objectives", node)
        assert ev3.target_lane == "optimization"

        ev4 = dog.trigger_orthogonal_warm_restart("b4", "optimization", node)
        assert ev4.target_lane == "architecture"

    def test_period_four_cycle_invariance(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_p4")
        lane = "architecture"
        for _ in range(4):
            ev = dog.trigger_orthogonal_warm_restart("b", lane, node)
            lane = ev.target_lane
        assert lane == "architecture"

    def test_unknown_lane_fallback(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_fallback")
        ev = dog.trigger_orthogonal_warm_restart("b_unk", "exotic_hyper_lane", node)
        assert ev.target_lane in ("architecture", "features", "objectives", "optimization")
        assert ev.target_lane == "architecture"

    def test_lane_case_and_whitespace_insensitivity(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_case")
        ev = dog.trigger_orthogonal_warm_restart("b_case", "  FEATURES  ", node)
        assert ev.target_lane == "objectives"

    def test_unique_branch_id_generated(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_unique_id")
        ev1 = dog.trigger_orthogonal_warm_restart("b_orig", "architecture", node)
        ev2 = dog.trigger_orthogonal_warm_restart("b_orig", "architecture", node)
        assert ev1.branch_id != "b_orig"
        assert ev2.branch_id != "b_orig"
        assert ev1.branch_id != ev2.branch_id
        assert ev1.branch_id.startswith("b_orig_warm_")

    def test_injected_priors_non_empty_for_all_lanes(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_priors")
        for lane in ("architecture", "features", "objectives", "optimization"):
            ev = dog.trigger_orthogonal_warm_restart("b", lane, node)
            assert ev.injected_priors is not None
            assert len(ev.injected_priors) >= 2
            for prior in ev.injected_priors:
                assert isinstance(prior, str)
                assert len(prior) > 0


class TestExecutionerEventLogging:
    """Tests executioner event logging, queryability, and serialization in SwarmLearningStateNode."""

    def test_event_logged_to_state_node(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_log_test")
        assert len(node.executioner_event_logs) == 0

        ev = dog.trigger_orthogonal_warm_restart(
            branch_id="b_parent_1",
            current_lane="architecture",
            state_node=node,
            trigger=StagnationTrigger.VANISHING_GRADIENTS,
            details={"grad_norm": 1e-8},
        )
        assert len(node.executioner_event_logs) == 1
        log = node.executioner_event_logs[0]
        assert log["branch_id"] == "b_parent_1"
        assert log["new_branch_id"] == ev.branch_id
        assert log["target_lane"] == "features"
        assert log["source_lane"] == "architecture"
        assert log["trigger"] == "VANISHING_GRADIENTS"
        assert log["details"]["grad_norm"] == 1e-8

    def test_logs_queryable_by_parent_branch_id(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_query_test")
        dog.trigger_orthogonal_warm_restart("branch_A", "architecture", node)
        dog.trigger_orthogonal_warm_restart("branch_B", "objectives", node)
        dog.trigger_orthogonal_warm_restart("branch_A", "features", node)

        matches_a = [e for e in node.executioner_event_logs if e["branch_id"] == "branch_A"]
        matches_b = [e for e in node.executioner_event_logs if e["branch_id"] == "branch_B"]
        matches_c = [e for e in node.executioner_event_logs if e["branch_id"] == "branch_C"]

        assert len(matches_a) == 2
        assert len(matches_b) == 1
        assert len(matches_c) == 0

    def test_timestamp_monotonicity(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_time_test")
        ev1 = dog.trigger_orthogonal_warm_restart("b1", "architecture", node)
        ev2 = dog.trigger_orthogonal_warm_restart("b2", "features", node)
        assert ev2.timestamp >= ev1.timestamp

    def test_large_volume_event_log_json_serialization(self):
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_large_log")
        for i in range(100):
            dog.trigger_orthogonal_warm_restart(f"branch_{i}", "architecture", node)
        assert len(node.executioner_event_logs) == 100
        serialized = json.dumps(node.model_dump())
        assert "branch_0" in serialized
        assert "branch_99" in serialized

        restored = SwarmLearningStateNode.model_validate(json.loads(serialized))
        assert len(restored.executioner_event_logs) == 100
        assert restored.export_summary()["executioner_events_count"] == 100

    def test_none_or_mock_state_node_handled(self):
        dog = ExecutionerWatchdog()
        ev = dog.trigger_orthogonal_warm_restart("b_standalone", "features", state_node=None)
        assert ev.target_lane == "objectives"
        assert ev.branch_id.startswith("b_standalone_warm_")


class TestExecutionerIntegration:
    """Full lifecycle integration tests linking audit, watchdog decision, warm restart, and MCTS node."""

    def test_end_to_end_audit_and_warm_restart_recovery(self):
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        node = SwarmLearningStateNode(node_id="n_e2e_rec")

        # 1. Search trapped in vanishing gradients
        history = [
            {"grad_norm": 2e-7, "loss": 0.50, "ast_tokens": {"x", "y"}},
            {"grad_norm": 1e-7, "loss": 0.50, "ast_tokens": {"x", "y"}},
            {"grad_norm": 5e-8, "loss": 0.50, "ast_tokens": {"x", "y"}},
        ]
        stagnant, trigger, reason = dog.audit_branch("b_trapped", depth=3, visits=6, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.VANISHING_GRADIENTS

        # 2. Trigger warm restart into features lane
        event = dog.trigger_orthogonal_warm_restart(
            branch_id="b_trapped",
            current_lane="architecture",
            state_node=node,
            trigger=trigger,
            details={"audit_reason": reason},
        )
        assert event.target_lane == "features"
        assert len(node.executioner_event_logs) == 1

        # 3. New exploration begins in target lane
        pt = node.record_evaluation(
            hypothesis_id="hyp_recovered_01",
            score=0.85,
            lane=event.target_lane,
            depth=1,
            metadata={"resumed_from": event.branch_id},
        )
        assert pt.lane == "features"
        assert pt.score == 0.85

        # 4. State summary reflects both evaluation and executioner event
        summary = node.export_summary()
        assert summary["trajectory_count"] == 1
        assert summary["executioner_events_count"] == 1
        assert summary["upper_envelope_peak"] == 0.85
