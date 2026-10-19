"""Unit Tests for Concrete Epistemic Brain, MCTS, and Accounting Ledger (brain/)."""

import math
import pytest

from frame.brain.state import MCTSNode, SearchStage, SearchAccountingLedger
from frame.brain.experimental_design import ScreeningDecision, SequentialStage
from frame.schemas.hypothesis import TargetDirection
from frame.brain.thinking import (
    compute_adaptive_uct_c,
    compute_uct_score,
    compute_canonical_ast_fingerprint,
    StructuralFalsificationMemory,
    RecombinationLegalityGrammar,
    LegalityViolation,
    EpistemicMCTSBrain,
    PostMortemTelemetry,
    synthesize_post_mortem_telemetry,
)


# ==============================================================================
# 1. MCTS State & Accounting Ledger Tests
# ==============================================================================

def test_mcts_node_and_accounting_ledger():
    """Verifies node creation, resource propagation, and ledger stage metrics."""
    ledger = SearchAccountingLedger()

    # Record entries and throughput across stages
    ledger.record_entry(SearchStage.IDEA_PROPOSAL)
    ledger.record_entry(SearchStage.IDEA_PROPOSAL)
    ledger.record_pass(SearchStage.IDEA_PROPOSAL, gpu_hours=0.01, cpu_hours=0.05)
    ledger.record_rejection(SearchStage.IDEA_PROPOSAL, reason="FATAL_AST_MATCH", gpu_hours=0.0, cpu_hours=0.02)

    assert ledger.total_hypotheses == 2
    summary = ledger.summary()
    assert summary["stage_pass_rates"][SearchStage.IDEA_PROPOSAL.value] == 0.5

    # Record a discovery
    ledger.record_discovery()
    assert ledger.total_discoveries == 1
    assert ledger.get_discovery_yield() == 0.5
    assert ledger.get_total_gpu_hours() == 0.01


# ==============================================================================
# 2. Adaptive UCT Schedule Tests
# ==============================================================================

def test_adaptive_uct_schedule():
    """Verifies that exploration constant c(t) follows the calibrated Pareto schedule."""
    # At step 0: c ≈ 1.4142
    c_0 = compute_adaptive_uct_c(0)
    assert abs(c_0 - 1.41421356) < 1e-5

    # At step 24: still in first interval
    c_24 = compute_adaptive_uct_c(24)
    assert abs(c_24 - c_0) < 1e-5

    # At step 25: first decay step 1.4142 * 0.95 ≈ 1.3435
    c_25 = compute_adaptive_uct_c(25)
    assert abs(c_25 - (c_0 * 0.95)) < 1e-5
    assert c_25 < c_0

    # At step 50: second decay step 1.4142 * 0.95^2 ≈ 1.2763
    c_50 = compute_adaptive_uct_c(50)
    assert abs(c_50 - (c_0 * 0.95**2)) < 1e-5

    # UCT score evaluates unvisited node as infinity
    unvisited = MCTSNode(node_id="child_new", visits_n=0, value_q=0.0)
    assert compute_uct_score(unvisited, parent_visits=10, c_t=c_0) == math.inf

    # Visited node balances Q and exploration
    visited = MCTSNode(node_id="child_1", visits_n=4, value_q=3.0)  # Q_mean = 0.75
    score = compute_uct_score(visited, parent_visits=10, c_t=c_0)
    assert score > 0.75
    assert math.isfinite(score)


# ==============================================================================
# 3. Structural AST Falsification Memory Tests
# ==============================================================================

def test_canonical_ast_fingerprint_invariance():
    """Verifies that variable renaming and docstrings produce identical fingerprints."""
    code_a = """
def activate(tensor_x):
    '''Docstring explaining activation.'''
    temp_val = tensor_x * 2.0
    return temp_val + 1.0
"""

    code_b = """
def activate(different_arg):
    alpha_var = different_arg * 2.0
    return alpha_var + 1.0
"""

    fp_a = compute_canonical_ast_fingerprint(code_a)
    fp_b = compute_canonical_ast_fingerprint(code_b)
    assert fp_a == fp_b


def test_falsification_memory_rejection_and_rebuttal():
    """Verifies that known fatal failures are rejected O(1) and semantic matches require rebuttal."""
    brain = EpistemicMCTSBrain()
    fatal_code = "def bad_op(x): return x / (x - x)"
    brain.memory.record_fatal_failure(
        code=fatal_code,
        stage=SearchStage.POT_VERIFICATION,
        reason="DIV_BY_ZERO",
    )

    root = brain.create_root_node("root_hyp", "def baseline(): pass")

    # 1. Exact match to fatal failure -> rejected O(1)
    passed_fatal, _, msg_fatal = brain.expand_child(
        parent_id="root",
        child_id="child_fatal",
        hypothesis_id="hyp_fatal",
        code=fatal_code,
    )
    assert passed_fatal is False
    assert "FATAL_AST_MATCH" in brain.ledger.stages[SearchStage.IDEA_PROPOSAL.value].rejection_reasons

    # 2. Similar failure missing rebuttal -> rejected
    similar_code = "def bad_op(y): return y / (y - y) + 1.0"
    passed_sim, _, msg_sim = brain.expand_child(
        parent_id="root",
        child_id="child_sim",
        hypothesis_id="hyp_sim",
        code=similar_code,
        rebuttal=None,
    )
    assert passed_sim is False
    assert "rebuttal required" in msg_sim.lower()

    # 3. Similar failure WITH structured rebuttal -> accepted
    passed_ok, child_node, _ = brain.expand_child(
        parent_id="root",
        child_id="child_ok",
        hypothesis_id="hyp_ok",
        code=similar_code,
        rebuttal="Numerical stability guaranteed via epsilon=1e-6 added to denominator.",
    )
    assert passed_ok is True
    assert child_node is not None


# ==============================================================================
# 4. Recombination Operator Legality Grammar Tests
# ==============================================================================

def test_recombination_legality_grammar():
    """Verifies typed compositional rules for (Model A, Loss B, Optimizer D)."""
    # 1. Double-Softmax Crush (Model has explicit softmax + CrossEntropyLoss)
    m_softmax = {"output_activation": "softmax"}
    l_ce = {"name": "cross_entropy_loss"}
    opt_adam = {"name": "adamw"}
    is_legal, code, msg = RecombinationLegalityGrammar.validate_recombination(m_softmax, l_ce, opt_adam)
    assert is_legal is False
    assert code == LegalityViolation.DOUBLE_SOFTMAX_CRUSH

    # 2. Non-Smooth Hessian with 2nd-order optimizer (Sophia + L1Loss)
    m_linear = {"output_activation": "identity"}
    l_l1 = {"name": "l1loss"}
    opt_sophia = {"name": "sophia"}
    is_legal2, code2, msg2 = RecombinationLegalityGrammar.validate_recombination(m_linear, l_l1, opt_sophia)
    assert is_legal2 is False
    assert code2 == LegalityViolation.NON_SMOOTH_HESSIAN

    # 3. Tensor Rank Mismatch (Muon applied to 1D vector parameters)
    opt_muon_1d = {"name": "muon", "includes_1d_params": True}
    l_mse = {"name": "mseloss"}
    is_legal3, code3, msg3 = RecombinationLegalityGrammar.validate_recombination(m_linear, l_mse, opt_muon_1d)
    assert is_legal3 is False
    assert code3 == LegalityViolation.TENSOR_RANK_MISMATCH

    # 4. Valid legal combination (Linear -> CrossEntropyLoss -> AdamW)
    is_legal4, code4, _ = RecombinationLegalityGrammar.validate_recombination(m_linear, l_ce, opt_adam)
    assert is_legal4 is True
    assert code4 is None

    # 5. Simplex Domain Mismatch (Model has linear output + KLDivLoss)
    l_kldiv = {"name": "kldivloss"}
    is_legal5, code5, msg5 = RecombinationLegalityGrammar.validate_recombination(m_linear, l_kldiv, opt_adam)
    assert is_legal5 is False
    assert code5 == LegalityViolation.SIMPLEX_DOMAIN_MISMATCH

    # 6. Valid Simplex combination (LogSoftmax -> KLDivLoss -> AdamW)
    m_log_softmax = {"output_activation": "log_softmax"}
    is_legal6, code6, _ = RecombinationLegalityGrammar.validate_recombination(m_log_softmax, l_kldiv, opt_adam)
    assert is_legal6 is True
    assert code6 is None


def test_canonical_ast_preserves_framework_and_class_docstrings():
    """Verifies that framework namespaces like torch are preserved and class docstrings are stripped."""
    code_with_doc = """
class Model(nn.Module):
    '''Class docstring that should be stripped.'''
    def forward(self, x):
        return torch.sin(x)
"""

    code_without_doc = """
class Model(nn.Module):
    def forward(self, input_tensor):
        return torch.sin(input_tensor)
"""

    fp_doc = compute_canonical_ast_fingerprint(code_with_doc)
    fp_no_doc = compute_canonical_ast_fingerprint(code_without_doc)
    assert fp_doc == fp_no_doc


# ==============================================================================
# 5. Oracle Statistical Reward & Backpropagation Tests
# ==============================================================================

def test_oracle_reward_statistical_backpropagation():
    """Verifies that MCTS reward computation and backpropagation connect to statistical outcomes."""
    brain = EpistemicMCTSBrain()
    root = brain.create_root_node("hyp_root", "class Baseline: pass")
    ok, child, _ = brain.expand_child(parent_id="root", child_id="child_1", hypothesis_id="hyp_cand", code="class Cand: pass")
    assert ok is True

    # 1. Futility pruned decision -> 0.0 reward
    dec_futility = ScreeningDecision(
        stage=SequentialStage.STAGE_1_FUTILITY,
        proceed=False,
        status="FUTILITY_PRUNED",
        sample_size_evaluated=3,
        mean_difference=-0.02,
        details={},
    )
    r_fut = brain.compute_oracle_reward(dec_futility, baseline_score=2.0)
    assert r_fut == 0.0

    # 2. Confirmatory rejection (sub-threshold) -> attenuated reward <= 0.05
    dec_sub = ScreeningDecision(
        stage=SequentialStage.STAGE_2_CONFIRMATORY,
        proceed=False,
        status="REJECTED_NULL",
        sample_size_evaluated=6,
        mean_difference=0.005,
        details={"p_value_rope": 0.15, "allocated_alpha_t": 0.010937, "rope_epsilon": 0.01},
    )
    r_sub = brain.compute_oracle_reward(dec_sub, baseline_score=2.0)
    assert 0.0 <= r_sub <= 0.05

    # 3. Confirmatory discovery -> significant oracle reward (discovery bonus 1.0 + effect * significance)
    dec_disc = ScreeningDecision(
        stage=SequentialStage.STAGE_2_CONFIRMATORY,
        proceed=True,
        status="CONFIRMED_CANDIDATE",
        sample_size_evaluated=6,
        mean_difference=0.15,
        details={"p_value_rope": 0.0001, "allocated_alpha_t": 0.010937, "rope_epsilon": 0.01},
    )
    r_disc = brain.compute_oracle_reward(dec_disc, baseline_score=2.0)
    assert r_disc > 1.0  # Includes discovery base bonus + effect size scaling

    # 4. Backpropagation connects to statistical decision
    reward_backprop = brain.backpropagate(
        node_id="child_1",
        statistical_decision=dec_disc,
        baseline_score=2.0,
        gpu_hours=0.05,
    )
    assert reward_backprop == r_disc
    assert child.visits_n == 1
    assert child.value_q == r_disc
    assert root.visits_n == 1
    assert root.value_q == r_disc
    assert root.gpu_hours_spent == 0.05


def test_falsification_memory_writeback_real_failure_signatures():
    """Verifies writeback of diverse failure signatures to StructuralFalsificationMemory."""
    mem = StructuralFalsificationMemory()

    # Record Stage 1 Futility failure
    code_futility = "def futile_kernel(x): return x * 1.0001"
    fp1 = mem.record_fatal_failure(
        code=code_futility,
        stage=SearchStage.STAGE1_FUTILITY,
        reason="Stage 1 Futility Pruned: mean diff -0.004 <= 0",
        rebuttal_req="Demonstrate positive paired seed gain on pilot sample k=3.",
    )
    assert fp1 in mem.fatal_fingerprints

    # Record Stage 2 Confirmatory failure
    code_confirmatory = "def weak_kernel(x): return x + 0.0002"
    fp2 = mem.record_fatal_failure(
        code=code_confirmatory,
        stage=SearchStage.STAGE2_CONFIRMATORY,
        reason="Stage 2 Confirmatory Rejected: p_rope=0.082 > alpha_t=0.010937",
        rebuttal_req="Demonstrate effect size exceeding ROPE with streaming FDR significance.",
    )
    assert fp2 in mem.fatal_fingerprints

    # O(1) Hard filter detects exact match
    blocked, rec = mem.check_hard_filter(code_futility)
    assert blocked is True
    assert rec.failure_stage == SearchStage.STAGE1_FUTILITY
    assert "Futility Pruned" in rec.reason

    # Query similar failures returns failure record and similarity score
    code_mutated = "def futile_kernel(input_tensor): return input_tensor * 1.0001"
    # Canonical AST should match exactly because variable renaming makes it identical
    blocked_mutated, _ = mem.check_hard_filter(code_mutated)
    assert blocked_mutated is True


def test_oracle_reward_statistical_verification_report():
    """Verifies that compute_oracle_reward and backpropagate correctly accept StatisticalVerificationReport."""
    from frame.schemas.verification import StatisticalVerificationReport, PairedSeedMetrics
    brain = EpistemicMCTSBrain()
    root = brain.create_root_node("hyp_root", "class Baseline: pass")
    ok, child, _ = brain.expand_child(parent_id="root", child_id="child_report", hypothesis_id="hyp_cand", code="class Cand: pass")
    assert ok is True

    # 1. Confirmed discovery report
    report_disc = StatisticalVerificationReport(
        hypothesis_id="hyp_cand",
        metric_name="accuracy",
        sample_size_n=5,
        mean_baseline=0.75,
        mean_candidate=0.85,
        paired_metrics=PairedSeedMetrics(
            metric_name="accuracy",
            paired_differences={"s1": 0.10, "s2": 0.10, "s3": 0.10},
            mean_difference=0.10,
            standard_error=0.005,
            t_statistic=20.0,
        ),
        rope_epsilon=0.01,
        rope_t_statistic=18.0,
        p_value_rope=0.00005,
        lord_step=1,
        allocated_alpha_t=0.010937,
        is_discovery=True,
        verdict="CONFIRMED_DISCOVERY",
    )

    reward = brain.compute_oracle_reward(report_disc)
    assert reward > 1.0  # Milestone bonus + effect size
    assert brain.backpropagate(node_id="child_report", statistical_decision=report_disc) == reward
    assert child.visits_n == 1
    assert child.value_q == reward

    # 2. Confirmed null report with positive drift (sub-threshold credit)
    report_null = StatisticalVerificationReport(
        hypothesis_id="hyp_cand",
        metric_name="accuracy",
        sample_size_n=5,
        mean_baseline=0.75,
        mean_candidate=0.755,
        paired_metrics=PairedSeedMetrics(
            metric_name="accuracy",
            paired_differences={"s1": 0.005, "s2": 0.005, "s3": 0.005},
            mean_difference=0.005,
            standard_error=0.005,
            t_statistic=1.0,
        ),
        rope_epsilon=0.01,
        rope_t_statistic=-1.0,
        p_value_rope=0.18,
        lord_step=2,
        allocated_alpha_t=0.008,
        is_discovery=False,
        verdict="CONFIRMED_NULL",
    )
    r_null = brain.compute_oracle_reward(report_null)
    assert 0.0 < r_null <= 0.05


def test_campaign_stage_level_accounting_and_node_lifecycle():
    """Verifies that SearchAccountingLedger correctly records entries, passes, and rejections."""
    from frame.brain.campaign import AutonomousCampaignRunner, CampaignConfig
    runner = AutonomousCampaignRunner(CampaignConfig(goal="Test Accounting Lifecycle", max_iterations=2, verbose=False))
    result = runner.run_campaign()

    ledger = runner.brain.ledger
    # Verify stage entries are recorded consistently
    for stage_name, st in ledger.stages.items():
        if st.hypotheses_passed > 0 or st.hypotheses_rejected > 0:
            assert st.hypotheses_entered >= (st.hypotheses_passed + st.hypotheses_rejected)
            assert 0.0 <= st.pass_rate() <= 1.0

    # Verify MCTS node states match lifecycle
    for node_id, node in runner.brain.nodes.items():
        if node_id != "root":
            assert node.stage in (
                SearchStage.STAGE1_FUTILITY,
                SearchStage.STAGE2_CONFIRMATORY,
                SearchStage.CONFIRMED_DISCOVERY,
                SearchStage.REJECTED,
            )


# ==============================================================================
# 6. Branch-Scoped Epistemic Isolation, Recursion Depth & Telemetry Tests
# ==============================================================================

def test_branch_scoped_failure_isolation_and_global_ast_blocking():
    """Asserts that failure in branch A does not poison branch B, while exact AST is blocked globally."""
    brain = EpistemicMCTSBrain()
    root = brain.create_root_node("root_hyp", "def baseline(): pass")

    failed_code_a = "def broken_attention_v1(q, k, v): return q @ k.T / 0.0"
    similar_code_b = "def broken_attention_v2(q, k, v): return q @ k.T / 1e-6"

    # Record fatal failure in Branch A (e.g. Lane: Architecture)
    fp_a = brain.memory.record_fatal_failure(
        code=failed_code_a,
        stage=SearchStage.POT_VERIFICATION,
        reason="DIV_BY_ZERO_POLE",
        rebuttal_req="Ensure denominator is bounded away from zero.",
        branch_id="lane_architecture",
        depth=1,
        branch_path="root/lane_architecture",
    )
    assert fp_a in brain.memory.fatal_fingerprints

    # 1. Exact identical AST is blocked globally across ANY branch
    ok_exact, child_exact, msg_exact = brain.expand_child(
        parent_id="root",
        child_id="child_exact_in_features",
        hypothesis_id="hyp_exact",
        code=failed_code_a,
        branch_id="lane_features",  # Different branch!
    )
    assert ok_exact is False
    assert child_exact is None
    assert "FATAL_AST_MATCH" in brain.ledger.stages[SearchStage.IDEA_PROPOSAL.value].rejection_reasons

    # 2. Semantic similar failure in Branch B (Lane: Features) does NOT get poisoned
    # Unrelated branch should expand successfully without requiring rebuttal for Branch A's failure
    ok_b, child_b, msg_b = brain.expand_child(
        parent_id="root",
        child_id="child_b",
        hypothesis_id="hyp_b",
        code=similar_code_b,
        branch_id="lane_features",
    )
    assert ok_b is True
    assert child_b is not None
    assert child_b.branch_id == "lane_features"
    assert child_b.depth == 1

    # 3. Branch A itself (or descendant in same branch) DOES require rebuttal
    ok_a, child_a, msg_a = brain.expand_child(
        parent_id="root",
        child_id="child_a_no_rebuttal",
        hypothesis_id="hyp_a",
        code=similar_code_b,
        branch_id="lane_architecture",
        branch_path="root/lane_architecture/sub_lane",
        rebuttal=None,
    )
    assert ok_a is False
    assert "rebuttal required" in msg_a.lower()

    # 4. Branch A WITH structured rebuttal expands successfully
    ok_a_reb, child_a_reb, _ = brain.expand_child(
        parent_id="root",
        child_id="child_a_with_rebuttal",
        hypothesis_id="hyp_a_reb",
        code=similar_code_b,
        branch_id="lane_architecture",
        branch_path="root/lane_architecture/sub_lane",
        rebuttal="Numerical stability guaranteed with non-zero epsilon=1e-6 added to denominator.",
    )
    assert ok_a_reb is True
    assert child_a_reb is not None


def test_recursion_depth_tracking_and_stack_propagation():
    """Verifies that MCTS tree nodes and search runner track recursion depth (N stacks deep)."""
    brain = EpistemicMCTSBrain()
    root = brain.create_root_node("root_hyp", "def baseline(): pass")
    assert root.depth == 0

    # Expand depth 1
    ok1, node1, _ = brain.expand_child(
        parent_id="root",
        child_id="node_d1",
        hypothesis_id="hyp_d1",
        code="def step_1(): pass",
        branch_id="branch_search_1",
    )
    assert ok1 is True
    assert node1.depth == 1
    assert node1.branch_id == "branch_search_1"

    # Expand depth 2 (grandchild)
    ok2, node2, _ = brain.expand_child(
        parent_id="node_d1",
        child_id="node_d2",
        hypothesis_id="hyp_d2",
        code="def step_2(): pass",
    )
    assert ok2 is True
    assert node2.depth == 2
    assert node2.branch_id == "branch_search_1"  # Inherited from parent

    # Verify AutonomousCampaignRunner tracks depth in history and nodes
    from frame.brain.campaign import AutonomousCampaignRunner, CampaignConfig
    config = CampaignConfig(
        goal="Test Stack Depth Tracking",
        max_iterations=4,
        target_improvement=0.50,  # High target to ensure multi-iteration recursive search
        verbose=False,
    )
    runner = AutonomousCampaignRunner(config)
    result = runner.run_campaign()
    assert result.iterations_executed == 4
    for log in result.iteration_history:
        assert hasattr(log, "depth")
        assert hasattr(log, "branch_id")
        assert hasattr(log, "targeted_queries")


def test_post_mortem_telemetry_targeted_queries():
    """Asserts that post-mortem error telemetry synthesizes targeted external search queries."""
    import numpy as np

    # 1. Math Singularity
    tel_math = synthesize_post_mortem_telemetry(
        stage=SearchStage.POT_VERIFICATION,
        reason="Poles detected at x=0",
        branch_id="lane_architecture",
        depth=2,
    )
    assert tel_math.error_mode == "mathematical_singularity"
    assert any("numerical stability" in q for q in tel_math.targeted_queries)

    # 2. Compositional Typing Violation: Double-Softmax
    tel_grammar = synthesize_post_mortem_telemetry(
        stage=SearchStage.EXPERIMENTAL_DESIGN,
        reason="LegalityViolation:DOUBLE_SOFTMAX_CRUSH - Model terminates in explicit Softmax",
        branch_id="lane_objectives",
        depth=1,
    )
    assert tel_grammar.error_mode == "compositional_typing_violation"
    assert any("unnormalized logits" in q for q in tel_grammar.targeted_queries)

    # 3. Stage 1 Futility Stagnation
    tel_futility = synthesize_post_mortem_telemetry(
        stage=SearchStage.STAGE1_FUTILITY,
        reason="Stage 1 Futility Pruned: mean diff -0.012 <= 0",
        branch_id="lane_optimization",
        depth=3,
    )
    assert tel_futility.error_mode == "futility_stagnation"
    assert any("gradient clipping" in q for q in tel_futility.targeted_queries)
    assert any("quantile scaling" in q for q in tel_futility.targeted_queries)

    # 4. Stage 2 Confirmatory Rejection (Null not rejected)
    tel_null = synthesize_post_mortem_telemetry(
        stage=SearchStage.STAGE2_CONFIRMATORY,
        reason="Stage 2 Confirmatory Rejected: p_rope=0.08 > alpha_t=0.0109",
        branch_id="lane_features",
        depth=2,
    )
    assert tel_null.error_mode == "confirmatory_null_insufficient_effect"
    assert any("interaction features" in q for q in tel_null.targeted_queries)

    # 5. Heavy-tailed residuals
    heavy_res = np.array([0.1, -0.2, 5.8, -4.2, 0.3, 0.05, -0.1])
    tel_res = synthesize_post_mortem_telemetry(
        stage=SearchStage.CONFIRMED_DISCOVERY,
        reason="Sub-goal reached with heavy residual error",
        branch_id="lane_features",
        depth=1,
        residuals=heavy_res,
    )
    assert any("robust normalization" in q or "Huber" in q for q in tel_res.targeted_queries)


def test_agy_swarm_coordinator_branch_isolation_and_telemetry():
    """Asserts that AgySwarmCoordinator supports branch-scoped registration, evaluation, and telemetry."""
    import torch.nn as nn
    from frame.agents.agy_bridge import AgySwarmCoordinator, SwarmProposal, LaneType
    coord = AgySwarmCoordinator(goal="Benchmark Branch Isolation")

    # 1. Register proposal with branch_id and depth
    prop = SwarmProposal(
        hypothesis_id="hyp_branch_arch",
        lane=LaneType.ARCHITECTURE,
        title="Isolated Arch Proposal",
        description="Branch testing",
        code_snippet="class IsolatedArch(nn.Module): pass",
        branch_id="lane_architecture",
        depth=2,
    )
    ok, err = coord.register_proposal(prop)
    assert ok is True

    # 2. Evaluate failing hypothesis produces post-mortem telemetry with targeted queries
    eval_futility = coord.evaluate_candidate_hypothesis(
        hypothesis_id="hyp_branch_arch",
        baseline_seed_scores={"s1": 1.0, "s2": 1.0, "s3": 1.0},
        candidate_seed_scores={"s1": 1.5, "s2": 1.5, "s3": 1.5},  # Worse for MINIMIZE
        branch_id="lane_architecture",
        depth=2,
    )
    assert eval_futility["passed"] is False
    assert eval_futility["branch_id"] == "lane_architecture"
    assert eval_futility["depth"] == 2
    assert "post_mortem_telemetry" in eval_futility
    assert len(eval_futility["post_mortem_telemetry"]["targeted_queries"]) > 0


