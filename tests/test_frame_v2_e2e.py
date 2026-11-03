"""Comprehensive end-to-end verification suite for FRAME-AR v2."""

from pathlib import Path
import textwrap
import pytest

from frame.bench.registry import (
    approve_benchmark,
    export_stage1_blind_packet,
    export_stage2_cross_packet,
    init_benchmark,
    record_stage1_review,
    record_stage2_critique,
)
from frame.bench.verifier import verify_benchmark
from frame.cli import main as cli_main
from frame.eval.funnel import FunnelEvaluator
from frame.schemas import EdgeType, NodeStatus, OperatorRole
from frame.swarm.brief import generate_context_brief


def test_phase0_committee_and_human_gate_kaggle(tmp_path: Path) -> None:
    """Verifies Rule 1 (3-stage blind-to-crossed committee), 6-check verifier, and Human Lock."""
    bench_dir = init_benchmark(tmp_path, "housing_tab", "kaggle_tabular")
    rep = verify_benchmark(bench_dir)

    assert rep.smoke_passed is True
    assert rep.baseline_competent is True
    assert rep.null_separation_passed is True
    assert rep.positive_control_passed is True
    assert rep.sensitivity_monotonic_passed is True
    assert rep.trust_grade.value in ("A", "B")

    # Stage 1 blind packets & reviews
    s1_pkt = export_stage1_blind_packet(bench_dir)
    assert s1_pkt.exists()

    # Stage 2 must block until BOTH blind reviews are recorded
    record_stage1_review(bench_dir, "gemini", "Gemini blind review: check tail errors.")
    with pytest.raises(RuntimeError):
        export_stage2_cross_packet(bench_dir, "gemini")

    record_stage1_review(bench_dir, "claude", "Claude blind review: check variance ratio.")
    s2_gem = export_stage2_cross_packet(bench_dir, "gemini")
    s2_cld = export_stage2_cross_packet(bench_dir, "claude")
    assert "Claude blind review" in s2_gem.read_text(encoding="utf-8")
    assert "Gemini blind review" in s2_cld.read_text(encoding="utf-8")

    record_stage2_critique(bench_dir, "gemini", "Agreed on variance ratio.")
    record_stage2_critique(bench_dir, "claude", "Agreed on tail error cap.")

    # Funnel must refuse to run BEFORE human approval
    with pytest.raises(PermissionError):
        FunnelEvaluator(tmp_path, "housing_tab")

    lock = approve_benchmark(bench_dir, reviewer="sparsh", require_committee=True)
    assert lock.approved_by == "sparsh"


def test_sandbox_audit_hook_blocks_benchmark_snooping(tmp_path: Path) -> None:
    """Verifies that a malicious candidate attempting to read `harness.py` is caught and marked BUGGY."""
    bench_dir = init_benchmark(tmp_path, "sec_test", "kaggle_tabular")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="sparsh")

    funnel = FunnelEvaluator(tmp_path, "sec_test")

    malicious_code = textwrap.dedent(
        f"""
        from pathlib import Path

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            # Attempt to snoop locked harness source code
            secret = Path({str(bench_dir / 'harness.py')!r}).read_text()
            return {{"predictions": [0.0] * len(task["X_eval"])}}
        """
    )
    mal_path = tmp_path / "malicious_cand.py"
    mal_path.write_text(malicious_code, encoding="utf-8")

    rec = funnel.evaluate_step(mal_path)
    assert rec.status == NodeStatus.BUGGY
    assert "Sandbox Audit Violation" in (rec.error_traceback or "")


def test_full_funnel_rl_iqm_directives_and_vault(tmp_path: Path) -> None:
    """Verifies RL IQM bootstrap, AST deduplication, futility pruning, promotion, and `frame inject`."""
    bench_dir = init_benchmark(tmp_path, "oscillator_rl", "rl_control")
    rep = verify_benchmark(bench_dir)
    assert rep.trust_grade.value in ("A", "B")
    approve_benchmark(bench_dir, reviewer="sparsh")

    funnel = FunnelEvaluator(tmp_path, "oscillator_rl")

    # 1. Exact duplicate of baseline (with renamed local variables) must be blocked at Gate 0 by OperatorASTMemory
    dup_code = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np
        from scipy.linalg import solve_discrete_are
        from scipy.optimize import minimize

        def select_action(obs_renamed: np.ndarray, state_renamed: dict) -> np.ndarray:
            x_renamed = obs_renamed[:, 0]
            v_renamed = obs_renamed[:, 1]
            gain_renamed = np.asarray(state_renamed["k_lin"], dtype=np.float64)
            return gain_renamed[0] * x_renamed + gain_renamed[1] * v_renamed

        def run_experiment(task_renamed: dict, seed_renamed: int, mode_renamed: str, artifact_renamed: Path) -> dict:
            step_dt = float(task_renamed["dt"])
            ctrl_max = float(task_renamed["u_max"])
            weights_cost = task_renamed["cost_weights"]
            params_plant = task_renamed["plant_params"]
            init_states = task_renamed["train_init"]
            noise_seq = task_renamed["train_noise"]

            A_mat = np.array([[1.0, step_dt], [params_plant["k_lin"] * step_dt, 1.0 + params_plant["c_lin"] * step_dt]], dtype=np.float64)
            B_mat = np.array([[0.0], [step_dt]], dtype=np.float64)
            Q_mat = np.diag([weights_cost["q_x"], weights_cost["q_v"]])
            R_mat = np.array([[weights_cost["r_u"]]], dtype=np.float64)
            P_mat = solve_discrete_are(A_mat, B_mat, Q_mat, R_mat)
            K_feedback = (-np.linalg.inv(R_mat + B_mat.T @ P_mat @ B_mat) @ (B_mat.T @ P_mat @ A_mat)).reshape(2)

            def sim_linear(w_param: np.ndarray) -> float:
                x_val = init_states[:, 0].astype(np.float64).copy()
                v_val = init_states[:, 1].astype(np.float64).copy()
                accum_cost = np.zeros(len(x_val), dtype=np.float64)
                for t_step in range(noise_seq.shape[1]):
                    u_val = np.clip(w_param[0] * x_val + w_param[1] * v_val, -ctrl_max, ctrl_max)
                    accum_cost += weights_cost["q_x"] * (x_val**2) + weights_cost["q_v"] * (v_val**2) + weights_cost["r_u"] * (u_val**2)
                    nl_force = params_plant["k_cubic"] * (x_val**3) + params_plant["k_quad_wind"] * (x_val**2) + params_plant["c_shear"] * (x_val * v_val)
                    nl_sat = params_plant["wind_cap"] * np.tanh(nl_force / params_plant["wind_cap"])
                    acc_val = params_plant["k_lin"] * x_val + params_plant["c_lin"] * v_val + nl_sat + u_val + noise_seq[:, t_step, 0]
                    x_val = x_val + step_dt * v_val
                    v_val = v_val + step_dt * acc_val
                return float(np.mean(accum_cost))

            solver_res = minimize(sim_linear, x0=K_feedback, method="Nelder-Mead", options={"maxiter": 80, "xatol": 1e-3, "fatol": 1e-3})
            lin_res = solver_res.x if solver_res.fun < sim_linear(K_feedback) else K_feedback
            return {"k_lin": np.asarray(lin_res, dtype=np.float64)}
        """
    )
    dup_path = tmp_path / "dup_cand.py"
    dup_path.write_text(dup_code, encoding="utf-8")
    r_dup = funnel.evaluate_step(dup_path)
    assert r_dup.status == NodeStatus.BUGGY
    assert "duplicate" in (r_dup.error_traceback or "").lower()

    # 2. Weak candidate must be pruned at Gate 1 (`FUTILITY_PRUNED`) without spending LORD step
    weak_code = textwrap.dedent(
        """
        from pathlib import Path
        import numpy as np

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            W = np.array([-0.45, -0.35, 0.0, 0.0], dtype=np.float64)
            return {"policy_weights": W, "policy_bias": np.zeros(1)}
        """
    )
    weak_path = tmp_path / "weak_cand.py"
    weak_path.write_text(weak_code, encoding="utf-8")
    r_weak = funnel.evaluate_step(weak_path)
    assert r_weak.status == NodeStatus.FUTILITY_PRUNED
    assert funnel.lord.current_step == 0

    # 3. Enqueue a human/frontier hypothesis via `frame inject` (`enqueue_directive`)
    ticket = funnel.tree.enqueue_directive(
        parent_node_id="node_0",
        hypothesis_spec="Add cubic stiffness compensation W[2] = -0.55 and cross-damping W[3] = -0.15.",
        operator_role=OperatorRole.SPEC_IMPLEMENTATION,
        priority=0,
    )
    brief_pkt = generate_context_brief(tmp_path, "oscillator_rl")
    assert ticket.directive_id in brief_pkt["brief_markdown"]
    assert "cubic stiffness compensation" in brief_pkt["brief_markdown"]

    # 4. Evaluate the strong nonlinear policy (`positive_control.py`) -> Must promote to Incumbent M*
    strong_path = bench_dir / "positive_control.py"
    r_strong = funnel.evaluate_step(
        strong_path,
        parent_id="node_0",
        operator_role=OperatorRole.SPEC_IMPLEMENTATION,
        directive_id=ticket.directive_id,
    )
    assert r_strong.status == NodeStatus.PROMOTED_DISCOVERY
    assert r_strong.promoted is True
    assert funnel.tree.incumbent_node_id == r_strong.node_id
    assert len(funnel.tree.get_queued_directives()) == 0

    # 5. Evaluate Vault-Holdout on the promoted incumbent
    vault_rep = funnel.evaluate_vault_holdout()
    assert vault_rep["passed_vault_audit"] is True

    # 6. CLI smoke test on `status`
    rc = cli_main(["--workspace", str(tmp_path), "status", "--topic", "oscillator_rl"])
    assert rc == 0
