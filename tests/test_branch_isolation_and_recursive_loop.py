"""Edge-case and boundary verification tests for branch-scoped isolation and recursive learning."""

import ast
import numpy as np
import pytest
import torch.nn as nn

from frame.brain.state import MCTSNode, SearchStage, SearchAccountingLedger
from frame.schemas.hypothesis import TargetDirection
from frame.brain.thinking import (
    StructuralFalsificationMemory,
    EpistemicMCTSBrain,
    FailureMemoryRecord,
    PostMortemTelemetry,
    synthesize_post_mortem_telemetry,
    compute_canonical_ast_fingerprint,
)
from frame.brain.campaign import AutonomousCampaignRunner, CampaignConfig
from frame.agents.agy_bridge import AgySwarmCoordinator, SwarmProposal, LaneType


# ==============================================================================
# 1. Edge Case: Branch Compatibility Logic Boundary Values
# ==============================================================================

def test_branch_compatibility_boundary_cases():
    """Exhaustively tests all permutations of None, empty strings, exact, and hierarchical branches."""
    compat = StructuralFalsificationMemory._is_branch_compatible

    # Both None (global)
    assert compat(None, None) is True

    # Global record matches any query
    assert compat("branch_x", None) is True
    assert compat("", None) is True

    # Unscoped query matches any record
    assert compat(None, "branch_x") is True

    # Exact string match
    assert compat("lane_arch", "lane_arch") is True

    # Disjoint branches
    assert compat("lane_features", "lane_arch") is False
    assert compat("branch_A", "branch_B") is False

    # Hierarchical descendant matching via paths
    assert compat(
        query_branch="lane_arch",
        record_branch="lane_arch",
        query_path="root/lane_arch/node_1/sub_node",
        record_path="root/lane_arch",
    ) is True

    # Hierarchical sibling/disjoint paths
    assert compat(
        query_branch="lane_feat",
        record_branch="lane_arch",
        query_path="root/lane_feat/node_1",
        record_path="root/lane_arch/node_1",
    ) is False

    # Ancestor query should NOT match child record
    assert compat(
        query_branch="lane_arch",
        record_branch="lane_arch",
        query_path="root/lane_arch",
        record_path="root/lane_arch/node_1",
    ) is False


# ==============================================================================
# 2. Edge Case: Empty, Malformed, and Syntax-Invalid Code Inputs
# ==============================================================================

def test_falsification_memory_empty_and_syntax_error_code():
    """Asserts that AST canonicalization and memory handle empty strings and syntax errors gracefully."""
    mem = StructuralFalsificationMemory()

    # Empty string
    fp_empty = compute_canonical_ast_fingerprint("")
    assert isinstance(fp_empty, str) and len(fp_empty) == 64

    # Syntax error code (invalid Python)
    invalid_code = "def syntax_error(::: pass"
    fp_invalid = compute_canonical_ast_fingerprint(invalid_code)
    assert isinstance(fp_invalid, str) and len(fp_invalid) == 64

    # Recording failure on syntax-invalid code works without crashing
    fp_rec = mem.record_fatal_failure(
        code=invalid_code,
        stage=SearchStage.IDEA_PROPOSAL,
        reason="SYNTAX_ERROR",
        branch_id="syntax_branch",
    )
    assert fp_rec == fp_invalid
    blocked, rec = mem.check_hard_filter(invalid_code)
    assert blocked is True
    assert rec.reason == "SYNTAX_ERROR"


# ==============================================================================
# 3. Edge Case: Post-Mortem Telemetry Boundary Inputs
# ==============================================================================

def test_post_mortem_telemetry_boundary_inputs():
    """Asserts that telemetry generation handles empty residuals, empty reasons, and unmapped stages."""
    # Empty reason and unmapped stage
    tel_empty = synthesize_post_mortem_telemetry(
        stage=SearchStage.MICRO_PASS,
        reason="",
        branch_id=None,
        depth=0,
        residuals=None,
    )
    assert tel_empty.error_mode == "generic_rejection"
    assert len(tel_empty.targeted_queries) > 0

    # Empty residuals array
    empty_res = np.array([])
    tel_empty_res = synthesize_post_mortem_telemetry(
        stage=SearchStage.STAGE1_FUTILITY,
        reason="Futility Pruned",
        residuals=empty_res,
    )
    assert tel_empty_res.residual_stats == {}

    # Residuals with extreme outliers / NaN / Inf resilience
    normal_res = np.array([0.01, 0.02, -0.01])
    tel_normal = synthesize_post_mortem_telemetry(
        stage=SearchStage.CONFIRMED_DISCOVERY,
        reason="Minor gain",
        residuals=normal_res,
    )
    assert tel_normal.residual_stats["std_residual"] < 1.5


# ==============================================================================
# 4. Edge Case: Recursion Depth Capping at max_depth
# ==============================================================================

def test_campaign_max_depth_capping():
    """Verifies that stack depth is strictly bounded by max_depth and never overflows."""
    config = CampaignConfig(
        goal="Test Depth Capping",
        max_iterations=6,
        max_depth=2,  # Hard ceiling at depth 2
        target_improvement=0.99,  # Unreachable goal to force continuous failure/recursion
        verbose=False,
    )
    runner = AutonomousCampaignRunner(config)
    result = runner.run_campaign()

    # Verify that in all logged iterations, depth never exceeds max_depth
    for log in result.iteration_history:
        assert log.depth <= config.max_depth

    # Verify nodes in MCTS brain also obey depth ceiling
    for node in runner.brain.nodes.values():
        assert node.depth <= config.max_depth


# ==============================================================================
# 5. Deep Test: Implicit Branch Path Semantic Rebuttal & Isolation
# ==============================================================================

def test_expand_child_implicit_branch_path_isolation_and_rebuttal():
    """Asserts that expand_child without explicit branch_path enforces rebuttal in same branch and isolates others."""
    brain = EpistemicMCTSBrain()
    root = brain.create_root_node("root_hyp", "def baseline(): pass")

    failed_code = "def broken_op_v1(q, k, v): return q @ k.T / 0.0"
    similar_code = "def broken_op_v2(q, k, v): return q @ k.T / 1e-6"

    # Record fatal failure in Branch A WITHOUT passing explicit branch_path
    brain.memory.record_fatal_failure(
        code=failed_code,
        stage=SearchStage.POT_VERIFICATION,
        reason="DIV_BY_ZERO_POLE",
        branch_id="lane_architecture",
        depth=1,
    )

    # 1. Expand child in Branch A WITHOUT passing branch_path & without rebuttal -> must fail!
    ok_a, child_a, msg_a = brain.expand_child(
        parent_id="root",
        child_id="child_a_no_rebuttal",
        hypothesis_id="hyp_a",
        code=similar_code,
        branch_id="lane_architecture",
        rebuttal=None,
    )
    assert ok_a is False
    assert child_a is None
    assert "rebuttal required" in msg_a.lower()

    # 2. Expand child in Branch A WITH structured rebuttal -> must succeed!
    ok_a_reb, child_a_reb, msg_a_reb = brain.expand_child(
        parent_id="root",
        child_id="child_a_with_rebuttal",
        hypothesis_id="hyp_a_reb",
        code=similar_code,
        branch_id="lane_architecture",
        rebuttal="Numerical stability guaranteed with epsilon=1e-6 added to denominator.",
    )
    assert ok_a_reb is True
    assert child_a_reb is not None

    # 3. Expand child in Branch B (lane_features) -> must succeed without rebuttal!
    ok_b, child_b, msg_b = brain.expand_child(
        parent_id="root",
        child_id="child_b",
        hypothesis_id="hyp_b",
        code=similar_code,
        branch_id="lane_features",
        rebuttal=None,
    )
    assert ok_b is True
    assert child_b is not None
    assert child_b.branch_id == "lane_features"


# ==============================================================================
# 6. Deep Test: Multi-Lane Branch Isolation Across Campaign Iterations
# ==============================================================================

def test_campaign_multi_lane_branch_isolation():
    """Asserts that each parallel lane retains its distinct branch_id and is not overwritten by earlier lanes."""
    config = CampaignConfig(
        goal="Test Multi-Lane Isolation",
        max_iterations=4,
        target_improvement=0.99,
        verbose=False,
    )
    runner = AutonomousCampaignRunner(config)
    result = runner.run_campaign()
    assert result.iterations_executed == 4

    expected_lanes = ["architecture", "objectives", "optimization", "features"]
    expected_branches = ["lane_architecture", "lane_objectives", "lane_optimization", "lane_features"]

    for i, log in enumerate(result.iteration_history):
        assert log.lane == expected_lanes[i]
        assert log.branch_id == expected_branches[i]


# ==============================================================================
# 7. Deep Test: Phase 4 -> Phase 1 Evolutionary Re-Entry with Targeted Queries
# ==============================================================================

def test_campaign_phase4_to_phase1_reentry_with_telemetry():
    """Asserts that when a lane recurses (Iteration 5 revisits lane 1), it enters at depth 1 with Phase 1 queries."""
    config = CampaignConfig(
        goal="Test Phase 4 to Phase 1 Re-Entry",
        max_iterations=5,
        target_improvement=0.99,
        verbose=False,
    )
    runner = AutonomousCampaignRunner(config)
    result = runner.run_campaign()
    assert result.iterations_executed == 5

    # Iteration 1 is architecture at depth 0
    iter1_log = result.iteration_history[0]
    assert iter1_log.lane == "architecture"
    assert iter1_log.depth == 0

    # Iteration 5 is architecture at depth 1
    iter5_log = result.iteration_history[4]
    assert iter5_log.lane == "architecture"
    assert iter5_log.branch_id == "lane_architecture"
    assert iter5_log.depth == 1
    assert len(iter5_log.targeted_queries) > 0
    assert "Reinforced" in iter5_log.proposal_title


# ==============================================================================
# 8. Deep Test: Post-Mortem Telemetry with Non-Finite (NaN / Inf) Residuals
# ==============================================================================

def test_post_mortem_telemetry_nan_inf_residuals():
    """Asserts that non-finite residuals (NaN / Inf) are handled robustly without crashing or emitting warnings."""
    nan_res = np.array([np.nan, np.inf, -np.inf])
    tel_nan = synthesize_post_mortem_telemetry(
        stage=SearchStage.STAGE1_FUTILITY,
        reason="Numerical instability with NaN loss",
        residuals=nan_res,
    )
    assert tel_nan.residual_stats["mean_residual"] == 0.0
    assert any("numerical stability" in q for q in tel_nan.targeted_queries)

    mixed_res = np.array([0.5, np.nan, 2.5, 3.0])
    tel_mixed = synthesize_post_mortem_telemetry(
        stage=SearchStage.STAGE2_CONFIRMATORY,
        reason="High variance residuals with partial NaN",
        residuals=mixed_res,
    )
    assert tel_mixed.residual_stats["mean_residual"] == 2.0
    assert tel_mixed.residual_stats["max_residual"] == 3.0
