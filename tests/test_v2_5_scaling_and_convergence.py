"""Unit and integration tests for FRAME-AR v2.5 Scaling & Convergence Upgrades.

Verifies:
1. Dynamic Baseline-Anchored Timeout Calibration (verifier.py)
2. Gate-0 Runtime Projection Guard (frame/cli.py cmd_smoke_check)
3. Ancestor Retirement & Disk Synchronization (frame/brain/tree.py)
4. Branch Exhaustion Penalty & UCT Selection (frame/brain/tree.py)
5. Stagnation Tracking & Early-Stop in Hypervisor (frame-orchestrate.sh)
"""

import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
from unittest.mock import patch
import numpy as np
import pytest

from frame.bench.registry import approve_benchmark, init_benchmark
from frame.bench.verifier import verify_benchmark
from frame.brain.tree import CampaignTree, OPERATOR_CYCLE
from frame.cli import main as cli_main
from frame.eval.funnel import FunnelEvaluator
from frame.eval.sandbox import SandboxExecutionResult
from frame.schemas import (
    BenchSpec,
    DirectiveTicket,
    EdgeType,
    NodeRecord,
    NodeStatus,
    OperatorRole,
    TrustGrade,
)


def test_v2_5_dynamic_timeout_calibration(tmp_path: Path):
    """Verifies that verify_benchmark auto-calibrates spec.timeout_sec."""
    topic = "calib_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")

    # Manually configure spec.timeout_sec to 30.0s
    spec_path = bench_dir / "bench_spec.json"
    spec = BenchSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    spec.timeout_sec = 30.0
    spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    # Run verification
    report = verify_benchmark(bench_dir)
    assert report.trust_grade != TrustGrade.FAIL

    # Read updated bench_spec.json
    updated_spec = BenchSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
    # min clamp is 45.0s, so timeout_sec must have scaled up from 30.0 to at least 45.0
    assert updated_spec.timeout_sec >= 45.0
    assert updated_spec.timeout_sec <= 600.0


def test_v2_5_gate0_runtime_projection_guard(tmp_path: Path):
    """Verifies that cmd_smoke_check triggers GATE0_RUNTIME_PROJECTION_EXCEEDED (code 5) when slow."""
    topic = "runtime_proj_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)

    # Candidate with non-duplicate structural variation
    test_code = textwrap.dedent(
        """
        import numpy as np

        def select_action(obs, policy_state):
            return np.zeros(1)

        def run_experiment(task, seed, mode, artifact_dir):
            return {"policy_weights": np.array([0.15, -0.25, 0.35, -0.45])}
        """
    )
    cand_path = tmp_path / "cand.py"
    cand_path.write_text(test_code, encoding="utf-8")

    # Mock run_candidate_isolated to simulate a slow execution (e.g. 15.0s)
    # smoke_budget_sec is max(5.0, 45.0 * 0.22) = 9.9s
    mock_res = SandboxExecutionResult(
        passed=True,
        aborted=False,
        candidate_output={"policy_weights": np.array([0.15, -0.25, 0.35, -0.45])},
        error_message="",
        stdout="",
        stderr="",
        wall_clock_sec=12.5,
    )

    from contextlib import redirect_stderr
    err_buf = io.StringIO()
    with patch("frame.eval.sandbox.run_candidate_isolated", return_value=mock_res):
        with redirect_stderr(err_buf):
            rc = cli_main([
                "--workspace", str(tmp_path),
                "smoke-check",
                "--topic", topic,
                "--candidate", str(cand_path),
            ])

    assert rc == 5
    err_out = err_buf.getvalue()
    assert "GATE0_RUNTIME_PROJECTION_EXCEEDED" in err_out
    assert "Smoke test took 12.50s" in err_out


def test_v2_5_ancestor_retirement_and_disk_sync(tmp_path: Path):
    """Verifies that superseded ancestor nodes are retired from UCT and on-disk records are synchronized."""
    topic = "retire_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree
    root = tree.nodes["node_0"]
    root.pilot_primary_mean = -35.48
    root.pilot_primary_iqm = -35.48

    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    # Record node_1 as a promoted child of node_0 with much better score (-30.41)
    node_1 = NodeRecord(
        node_id="node_1",
        parent_id="node_0",
        branch_id="b1",
        status=NodeStatus.PROMOTED_DISCOVERY,
        promoted=True,
        pilot_primary_mean=-30.41,
        pilot_primary_iqm=-30.41,
        pilot_delta_vs_parent=+5.07,
    )
    tree.record_node(node_1, dummy_cand, reward=1.0)

    # 1. Verify disk synchronization: node_0's on-disk node_record.json must have node_1 in children_ids
    root_rec_path = tree.nodes_dir / "node_0" / "node_record.json"
    assert root_rec_path.is_file()
    root_rec_disk = NodeRecord.model_validate_json(root_rec_path.read_text(encoding="utf-8"))
    assert "node_1" in root_rec_disk.children_ids
    assert root_rec_disk.visits >= 2

    # 2. Verify Ancestor Retirement:
    # rope_epsilon = 0.5. node_0 trails incumbent node_1 by 5.07 > 2 * 0.5 = 1.0.
    # node_0 has spawned a promoted child (node_1). Therefore, node_0 is retired!
    best_parent, role, directive = tree.select_next_task(rope_epsilon=0.5)
    assert best_parent == "node_1"


def test_v2_5_branch_exhaustion_penalty(tmp_path: Path):
    """Verifies that 4 consecutive failures apply full cycle exhaustion penalty in UCT."""
    topic = "exhaust_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree

    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    # node_1: Promoted
    node_1 = NodeRecord(
        node_id="node_1",
        parent_id="node_0",
        branch_id="b1",
        status=NodeStatus.PROMOTED_DISCOVERY,
        promoted=True,
        pilot_primary_mean=-30.0,
        pilot_primary_iqm=-30.0,
    )
    tree.record_node(node_1, dummy_cand, reward=1.0)

    # node_2: Also Promoted with similar score
    node_2 = NodeRecord(
        node_id="node_2",
        parent_id="node_0",
        branch_id="b2",
        status=NodeStatus.PROMOTED_DISCOVERY,
        promoted=True,
        pilot_primary_mean=-30.0,
        pilot_primary_iqm=-30.0,
    )
    tree.record_node(node_2, dummy_cand, reward=1.0)

    # Spawn 4 consecutive unpromoted children on node_1
    for i in range(3, 7):
        fail_node = NodeRecord(
            node_id=f"node_{i}",
            parent_id="node_1",
            branch_id="b1",
            status=NodeStatus.FUTILITY_PRUNED,
            promoted=False,
            pilot_primary_mean=-32.0,
            pilot_primary_iqm=-32.0,
        )
        tree.record_node(fail_node, dummy_cand, reward=0.0)

    # Now node_1 has 4 failed children (full operator cycle exhausted).
    # Its UCT penalty is heavy (+2.5 extra penalty).
    # select_next_task should now choose node_2 instead of node_1!
    best_parent, role, _ = tree.select_next_task(rope_epsilon=0.5)
    assert best_parent == "node_2"


def test_v2_5_stagnation_counter_and_meta_env(tmp_path: Path):
    """Verifies get_steps_since_last_promotion and its export to meta.env in prepare-jail."""
    topic = "stag_counter_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree
    assert tree.get_steps_since_last_promotion() == 0

    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    # Add 2 failed nodes
    n1 = NodeRecord(node_id="node_1", parent_id="node_0", status=NodeStatus.FUTILITY_PRUNED, promoted=False)
    tree.record_node(n1, dummy_cand, reward=0.0)
    assert tree.get_steps_since_last_promotion() == 1

    n2 = NodeRecord(node_id="node_2", parent_id="node_0", status=NodeStatus.CONFIRMATORY_NULL, promoted=False)
    tree.record_node(n2, dummy_cand, reward=0.0)
    assert tree.get_steps_since_last_promotion() == 2

    # Check prepare-jail exports STAGNANT_STEPS and UNCONSUMED_DIRECTIVES
    jail_dir = tmp_path / "jail"
    rc = cli_main([
        "--workspace", str(tmp_path),
        "prepare-jail",
        "--topic", topic,
        "--jail-dir", str(jail_dir),
    ])
    assert rc == 0
    meta_env = (jail_dir / "meta.env").read_text(encoding="utf-8")
    assert "STAGNANT_STEPS='2'" in meta_env
    assert "UNCONSUMED_DIRECTIVES='0'" in meta_env

    # Now promote node_3
    n3 = NodeRecord(node_id="node_3", parent_id="node_0", status=NodeStatus.PROMOTED_DISCOVERY, promoted=True)
    tree.record_node(n3, dummy_cand, reward=1.0)
    assert tree.get_steps_since_last_promotion() == 0


def test_v2_5_orchestrate_convergence_early_stop():
    """Verifies that frame-orchestrate.sh halts early when max_stagnant_steps is reached."""
    root_dir = Path(__file__).resolve().parent.parent
    script_path = root_dir / "frame-orchestrate.sh"
    topic = "e2e_early_stop_test"

    shutil.rmtree(root_dir / ".frame" / "benchmarks" / topic, ignore_errors=True)
    shutil.rmtree(root_dir / ".frame" / "campaigns" / topic, ignore_errors=True)

    bench_dir = init_benchmark(root_dir, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(root_dir, topic)
    dummy_cand = root_dir / ".frame" / "campaigns" / topic / "nodes" / "node_0" / "candidate.py"

    try:
        # Seed 2 failed nodes
        n1 = NodeRecord(node_id="node_1", parent_id="node_0", status=NodeStatus.FUTILITY_PRUNED, promoted=False)
        evaluator.tree.record_node(n1, dummy_cand, reward=0.0)
        n2 = NodeRecord(node_id="node_2", parent_id="node_0", status=NodeStatus.FUTILITY_PRUNED, promoted=False)
        evaluator.tree.record_node(n2, dummy_cand, reward=0.0)

        # Run frame-orchestrate.sh with --max-stagnant-steps 2 and --steps 5
        # Since stagnant_steps is already 2, step 1 should trigger Convergence Early-Stop immediately
        res = subprocess.run(
            [
                str(script_path),
                "--topic", topic,
                "--steps", "5",
                "--max-stagnant-steps", "2",
            ],
            cwd=str(root_dir),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "[Convergence Early-Stop]" in res.stdout
        assert "2 consecutive steps completed" in res.stdout
        assert "Halting MCTS search to prevent deep-tree thrashing and proceeding to Vault Audit." in res.stdout
    finally:
        shutil.rmtree(root_dir / ".frame" / "benchmarks" / topic, ignore_errors=True)
        shutil.rmtree(root_dir / ".frame" / "campaigns" / topic, ignore_errors=True)


def test_v2_5_deep_disk_sync_and_ancestor_backprop(tmp_path: Path):
    """Verifies that 3-level deep tree backpropagation updates visits, rewards, and children on all ancestors' node_record.json files."""
    topic = "deep_sync_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree
    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    # node_0 -> node_1 (promoted) -> node_2 (promoted) -> node_3 (futility)
    n1 = NodeRecord(node_id="node_1", parent_id="node_0", branch_id="b1", status=NodeStatus.PROMOTED_DISCOVERY, promoted=True, pilot_primary_mean=-30.0)
    tree.record_node(n1, dummy_cand, reward=1.0)

    n2 = NodeRecord(node_id="node_2", parent_id="node_1", branch_id="b1", status=NodeStatus.PROMOTED_DISCOVERY, promoted=True, pilot_primary_mean=-25.0)
    tree.record_node(n2, dummy_cand, reward=1.5)

    n3 = NodeRecord(node_id="node_3", parent_id="node_2", branch_id="b1", status=NodeStatus.FUTILITY_PRUNED, promoted=False, pilot_primary_mean=-28.0)
    tree.record_node(n3, dummy_cand, reward=0.0)

    # Check node_0 on disk
    rec0 = NodeRecord.model_validate_json((tree.nodes_dir / "node_0" / "node_record.json").read_text(encoding="utf-8"))
    assert "node_1" in rec0.children_ids
    # Initial visits=1 on root + 3 non-buggy children = 4 visits
    assert rec0.visits == 4
    assert rec0.total_reward == pytest.approx(1.0 + 1.0 + 1.5 + 0.0)

    # Check node_1 on disk
    rec1 = NodeRecord.model_validate_json((tree.nodes_dir / "node_1" / "node_record.json").read_text(encoding="utf-8"))
    assert "node_2" in rec1.children_ids
    # Initial visits=1 + 1 (its own record_node) + 2 (descendants node_2, node_3) = 4
    assert rec1.visits == 4
    assert rec1.total_reward == pytest.approx(1.0 + 1.5 + 0.0)

    # Check node_2 on disk
    rec2 = NodeRecord.model_validate_json((tree.nodes_dir / "node_2" / "node_record.json").read_text(encoding="utf-8"))
    assert "node_3" in rec2.children_ids
    assert rec2.visits == 3
    assert rec2.total_reward == pytest.approx(1.5 + 0.0)


def test_v2_5_tiny_rope_and_multi_ancestor_retirement(tmp_path: Path):
    """Verifies ancestor retirement stability under small ROPE thresholds (e.g. Kaggle tabular scale 1e-4)."""
    topic = "tiny_rope_test"
    bench_dir = init_benchmark(tmp_path, topic, "kaggle_tabular")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree
    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    root = tree.nodes["node_0"]
    root.pilot_primary_mean = 0.8500
    root.pilot_primary_iqm = 0.8500

    # node_1: beats root by 0.0003 (promoted) -> incumbent
    n1 = NodeRecord(
        node_id="node_1",
        parent_id="node_0",
        branch_id="b1",
        status=NodeStatus.PROMOTED_DISCOVERY,
        promoted=True,
        pilot_primary_mean=0.8503,
        pilot_primary_iqm=0.8503,
    )
    tree.record_node(n1, dummy_cand, reward=1.0)

    # node_0 has promoted child (node_1).
    # rope_epsilon = 1e-4. 2 * rope_epsilon = 2e-4 = 0.0002.
    # Gap is 0.8503 - 0.8500 = 0.0003 > 0.0002. node_0 MUST be retired!
    best_parent, _, _ = tree.select_next_task(rope_epsilon=1e-4)
    assert best_parent == "node_1"

    # Now if rope_epsilon = 2e-4, 2 * rope = 4e-4 = 0.0004 > 0.0003. node_0 should NOT be retired!
    best_parent_loose, _, _ = tree.select_next_task(rope_epsilon=2e-4)
    assert best_parent_loose in ("node_0", "node_1")


def test_v2_5_legacy_state_total_steps_recovery(tmp_path: Path):
    """Verifies that legacy tree_state.json missing 'total_steps' accurately recovers evaluated step count."""
    topic = "legacy_tree_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    tree = evaluator.tree
    dummy_cand = tmp_path / "cand.py"
    dummy_cand.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {}", encoding="utf-8")

    # Add 1 evaluated, 1 buggy, 1 aborted
    n1 = NodeRecord(node_id="node_1", parent_id="node_0", status=NodeStatus.FUTILITY_PRUNED, promoted=False)
    tree.record_node(n1, dummy_cand, reward=0.0)

    n2 = NodeRecord(node_id="node_2", parent_id="node_0", status=NodeStatus.BUGGY, promoted=False)
    tree.record_node(n2, dummy_cand, reward=0.0)

    n3 = NodeRecord(node_id="node_3", parent_id="node_0", status=NodeStatus.ABORTED, promoted=False)
    tree.record_node(n3, dummy_cand, reward=0.0)

    # Intentionally corrupt tree_state.json by stripping 'total_steps'
    state_file = tree.state_path
    raw_data = json.loads(state_file.read_text(encoding="utf-8"))
    del raw_data["total_steps"]
    state_file.write_text(json.dumps(raw_data), encoding="utf-8")

    # Reload fresh CampaignTree
    reloaded_tree = CampaignTree(tmp_path, topic)
    # Only node_1 is an evaluated non-buggy/non-aborted step (total_steps = 1)
    assert reloaded_tree.total_steps == 1


def test_v2_5_gate0_runtime_boundary_condition(tmp_path: Path):
    """Verifies the exact boundary: wall_clock_sec == smoke_budget_sec passes, while > triggers code 5."""
    topic = "boundary_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    evaluator = FunnelEvaluator(tmp_path, topic)
    cand_path = tmp_path / "cand.py"
    cand_path.write_text("import numpy as np\ndef run_experiment(t, s, m, a): return {'policy_weights': np.zeros(4)}", encoding="utf-8")

    smoke_budget_sec = max(5.0, evaluator.spec.timeout_sec * 0.22)

    # 1. Exactly on budget: wall_clock_sec == smoke_budget_sec -> PASSED
    res_exact = SandboxExecutionResult(
        passed=True,
        aborted=False,
        candidate_output={"policy_weights": np.zeros(4)},
        error_message="",
        stdout="",
        stderr="",
        wall_clock_sec=smoke_budget_sec,
    )
    with patch("frame.eval.sandbox.run_candidate_isolated", return_value=res_exact):
        rc = cli_main(["--workspace", str(tmp_path), "smoke-check", "--topic", topic, "--candidate", str(cand_path)])
    assert rc == 0

    # 2. Exceeds budget by 0.05s -> FAILED with code 5
    res_exceeded = SandboxExecutionResult(
        passed=True,
        aborted=False,
        candidate_output={"policy_weights": np.zeros(4)},
        error_message="",
        stdout="",
        stderr="",
        wall_clock_sec=smoke_budget_sec + 0.05,
    )
    with patch("frame.eval.sandbox.run_candidate_isolated", return_value=res_exceeded):
        rc = cli_main(["--workspace", str(tmp_path), "smoke-check", "--topic", topic, "--candidate", str(cand_path)])
    assert rc == 5

