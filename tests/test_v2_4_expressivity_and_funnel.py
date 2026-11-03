"""Unit and integration tests for FRAME-AR v2.4 Open Policy Expressivity and Incumbent Filter."""

from pathlib import Path
import tempfile
import textwrap
import numpy as np
import pytest

from frame.bench.registry import approve_benchmark, generate_human_review_markdown, init_benchmark
from frame.bench.verifier import _run_expressivity_probe, verify_benchmark
from frame.brain.tree import CampaignTree
from frame.eval.funnel import FunnelEvaluator
from frame.eval.sandbox import bind_candidate_select_action, load_harness_module, run_candidate_isolated
from frame.schemas import BenchSpec, EdgeType, NodeStatus, OperatorRole, StatisticalEstimator, TrustGrade


def test_sandbox_ast_binding_select_action(tmp_path: Path) -> None:
    """Verifies that bind_candidate_select_action safely binds select_action into restricted math namespace."""
    code = textwrap.dedent(
        """
        import numpy as np

        def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
            return policy_state["gain"] * obs[:, 0]

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir):
            return {"gain": 2.5}
        """
    )
    cand_file = tmp_path / "cand.py"
    cand_file.write_text(code, encoding="utf-8")

    raw_output = {"gain": 2.5}
    bound = bind_candidate_select_action(cand_file, raw_output)
    assert "_select_action" in bound
    assert callable(bound["_select_action"])

    obs = np.array([[1.5, 2.0], [-2.0, 0.0]], dtype=np.float64)
    act = bound["_select_action"](obs, bound)
    np.testing.assert_allclose(act, [3.75, -5.0])


def test_sandbox_audit_hook_blocks_candidate_sys_getframe(tmp_path: Path) -> None:
    """Verifies that sys._getframe inside candidate execution is intercepted and raises PermissionError."""
    code = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            frame = sys._getframe(0)
            return {"status": "infiltrated"}
        """
    )
    cand_file = tmp_path / "cand_frame_snoop.py"
    cand_file.write_text(code, encoding="utf-8")

    res = run_candidate_isolated(
        candidate_path=cand_file,
        public_task={"dummy": 1},
        seed=42,
        mode="smoke",
        timeout_sec=5.0,
        forbidden_bench_dir=tmp_path / "forbidden",
    )
    assert not res.passed
    assert "sys._getframe" in res.error_message or "sys._getframe" in res.stderr
    assert "Sandbox violation" in res.error_message or "Sandbox violation" in res.stderr


def test_check7_expressivity_probe_catches_hardcoded_representation(tmp_path: Path) -> None:
    """Verifies that _run_expressivity_probe catches lock-in when harness expects fixed policy_weights."""
    bench_dir = init_benchmark(tmp_path, "probe_test", "rl_control")
    spec = BenchSpec.model_validate_json((bench_dir / "bench_spec.json").read_text(encoding="utf-8"))
    harness = load_harness_module(bench_dir / "harness.py")
    pub_smoke, sec_smoke = harness.prepare_split(seed=spec.pilot_seeds[0], split_tier="smoke")

    # 1. Normal upgraded harness passes Check #7
    passed, errors = _run_expressivity_probe(
        spec=spec,
        harness=harness,
        pub_smoke=pub_smoke,
        sec_smoke=sec_smoke,
        seed0=spec.pilot_seeds[0],
        bench_dir=bench_dir,
    )
    assert passed is True
    assert len(errors) == 0

    # 2. Mock a rigid locked harness that demands candidate_output["rigid_4dim_vector"]
    class RigidHarnessMock:
        def score_candidate_output(self, candidate_output, sec, seed, split_tier):
            if "rigid_4dim_vector" not in candidate_output:
                raise KeyError("rigid_4dim_vector")
            return {"primary_metric": -30.0}

        def degrade_output(self, candidate_output, sec, noise_level, seed):
            return dict(candidate_output)

    rigid_harness = RigidHarnessMock()
    rigid_passed, rigid_errors = _run_expressivity_probe(
        spec=spec,
        harness=rigid_harness,
        pub_smoke=pub_smoke,
        sec_smoke=sec_smoke,
        seed0=spec.pilot_seeds[0],
        bench_dir=bench_dir,
    )
    assert rigid_passed is False
    assert any("REPRESENTATION_LOCK_IN_DETECTED" in err for err in rigid_errors)


def test_expressivity_probe_in_verification_report_and_human_review(tmp_path: Path) -> None:
    """Verifies expressivity_probe_passed is in BenchVerificationReport and BENCHMARK_REVIEW.md."""
    bench_dir = init_benchmark(tmp_path, "osc_report", "rl_control")
    rep = verify_benchmark(bench_dir)
    assert rep.expressivity_probe_passed is True
    assert rep.trust_grade in (TrustGrade.A, TrustGrade.B)

    review_md_path = generate_human_review_markdown(bench_dir)
    review_content = review_md_path.read_text(encoding="utf-8")
    assert "Expressivity Probe" in review_content
    assert "`True`" in review_content


def test_gate1_incumbent_superiority_filter(tmp_path: Path) -> None:
    """
    Verifies Section 3: A candidate must beat BOTH its parent node AND the current Global Incumbent M*.
    Scenario:
      - node_0 is root baseline (rep = -37.5)
      - node_1 promotes to incumbent M* (rep = -33.6)
      - branch from node_0 produces node_2 (rep = -35.0):
        Beats parent node_0 (-35.0 > -37.5, delta > 0)
        BUT worse than incumbent node_1 (-35.0 < -33.6, delta <= 0)
      -> Must be FUTILITY_PRUNED at Gate 1 without advancing to Gate 2!
    """
    bench_dir = init_benchmark(tmp_path, "incumbent_filter_test", "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")

    funnel = FunnelEvaluator(tmp_path, "incumbent_filter_test")
    assert funnel.tree.incumbent_node_id == "node_0"

    # Step 1: Promote strong candidate to become node_1 and incumbent M*
    pos_path = bench_dir / "positive_control.py"
    r1 = funnel.evaluate_step(pos_path, parent_id="node_0")
    assert r1.status == NodeStatus.PROMOTED_DISCOVERY
    assert funnel.tree.incumbent_node_id == r1.node_id
    assert r1.node_id == "node_1"

    # Step 2: Create a candidate branching from node_0 that beats node_0 (-37.5)
    # but does NOT beat incumbent node_1 (-33.6).
    # e.g., controller with slightly better linear gains beating node_0 but worse than node_1.
    med_code = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np

        def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
            x, v = obs[:, 0], obs[:, 1]
            return -1.50 * x - 0.85 * v - 0.15 * (x**3)

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            return {"_marker": "medium_linear_cubic"}
        """
    )
    med_path = tmp_path / "med_cand.py"
    med_path.write_text(med_code, encoding="utf-8")

    # Evaluate with parent_id="node_0"
    r2 = funnel.evaluate_step(med_path, parent_id="node_0")
    # Must be pruned because it doesn't beat incumbent node_1
    assert r2.status == NodeStatus.FUTILITY_PRUNED
    # Lord step must not have been spent on Gate 2
    assert funnel.lord.current_step == 1  # only spent 1 step for node_1


def test_sandbox_ast_binding_annassign_and_helper_class_and_lambda(tmp_path: Path) -> None:
    """Verifies that bind_candidate_select_action supports AnnAssign constants, helper classes, and lambda definitions."""
    # 1. AnnAssign and helper class
    code1 = textwrap.dedent(
        """
        import numpy as np

        GAIN_X: float = 2.0
        BIAS_V: float = -1.0

        class LinearGainHelper:
            def __init__(self, gx, bv):
                self.gx = gx
                self.bv = bv

            def compute(self, obs):
                return self.gx * obs[:, 0] + self.bv * obs[:, 1]

        def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
            helper = LinearGainHelper(GAIN_X, BIAS_V)
            return helper.compute(obs)

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir):
            return {"ready": True}
        """
    )
    cand_file1 = tmp_path / "cand_annassign.py"
    cand_file1.write_text(code1, encoding="utf-8")
    bound1 = bind_candidate_select_action(cand_file1, {"ready": True})
    assert "_select_action" in bound1
    obs = np.array([[1.0, 3.0], [2.0, -1.0]], dtype=np.float64)
    act1 = bound1["_select_action"](obs, bound1)
    np.testing.assert_allclose(act1, [-1.0, 5.0])

    # 2. Lambda select_action
    code2 = textwrap.dedent(
        """
        import numpy as np

        select_action = lambda obs, ps: -1.5 * obs[:, 0] + 0.5 * obs[:, 1]

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir):
            return {}
        """
    )
    cand_file2 = tmp_path / "cand_lambda.py"
    cand_file2.write_text(code2, encoding="utf-8")
    bound2 = bind_candidate_select_action(cand_file2, {})
    assert "_select_action" in bound2
    act2 = bound2["_select_action"](obs, bound2)
    np.testing.assert_allclose(act2, [0.0, -3.5])


def test_sandbox_audit_hook_blocks_getframe_even_in_site_packages_path(tmp_path: Path) -> None:
    """Verifies that sys._getframe is unconditionally blocked even if executed from a path with site-packages."""
    sp_dir = tmp_path / "custom_site-packages_dir"
    sp_dir.mkdir(parents=True)
    cand_file = sp_dir / "candidate_with_site_packages_in_path.py"
    cand_file.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path

            def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
                f = sys._getframe(0)
                return {"leaked": True}
            """
        ),
        encoding="utf-8",
    )

    res = run_candidate_isolated(
        candidate_path=cand_file,
        public_task={"dummy": 1},
        seed=42,
        mode="smoke",
        timeout_sec=5.0,
        forbidden_bench_dir=tmp_path / "forbidden",
    )
    assert not res.passed
    assert "sys._getframe" in res.error_message or "sys._getframe" in res.stderr
    assert "Sandbox violation" in res.error_message or "Sandbox violation" in res.stderr

