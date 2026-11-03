"""7-Check Benchmark Meta-Verifier (`frame bench verify` — Auditing the Auditor)."""

from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import numpy as np

from frame.eval.sandbox import load_harness_module, run_candidate_isolated
from frame.eval.statistics import calibrate_required_k, compute_iqm
from frame.schemas import (
    BenchSpec,
    BenchVerificationReport,
    DomainType,
    TargetDirection,
    TrustGrade,
)


def _run_expressivity_probe(
    spec: BenchSpec,
    harness: Any,
    pub_smoke: Dict[str, Any],
    sec_smoke: Dict[str, Any],
    seed0: int,
    bench_dir: Path,
) -> Tuple[bool, List[str]]:
    """Check #7: Verifies that harness.py does not lock candidates into a fixed feature vector or weight shape."""
    dom_val = spec.domain.value if hasattr(spec.domain, "value") else str(spec.domain)
    if dom_val != "rl_control":
        return True, []

    probe_scripts = {
        "probe_linear_2d": """from pathlib import Path
import numpy as np

def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
    return policy_state["k_x"] * obs[:, 0] + policy_state["k_v"] * obs[:, 1]

def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
    return {"k_x": -1.45, "k_v": -0.75}
""",
        "probe_mlp_shear_48p": """from pathlib import Path
import numpy as np

def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
    x, v = obs[:, 0], obs[:, 1]
    feats = np.stack([x, v, x * v], axis=-1)
    h = np.tanh(feats @ policy_state["W1"] + policy_state["b1"])
    return (h @ policy_state["W2"]).ravel()

def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.4, size=(3, 12)),
        "b1": np.zeros(12, dtype=np.float64),
        "W2": rng.normal(0.0, 0.5, size=(12, 1)),
    }
""",
        "probe_piecewise_lookup": """from pathlib import Path
import numpy as np

def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:
    x, v = obs[:, 0], obs[:, 1]
    s = x + 0.5 * v
    t_low, t_high = policy_state["thresholds"]
    l_neg, l_mid, l_pos = policy_state["levels"]
    return np.where(s < t_low, l_neg, np.where(s > t_high, l_pos, l_mid))

def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
    return {"thresholds": [-0.6, 0.6], "levels": [2.2, -0.4, -2.8]}
""",
    }

    probe_scores: List[float] = []
    errors: List[str] = []

    with tempfile.TemporaryDirectory(prefix="frame_expr_probe_") as tmp_d:
        tmp_path = Path(tmp_d)
        for p_name, p_code in probe_scripts.items():
            p_file = tmp_path / f"{p_name}.py"
            p_file.write_text(p_code, encoding="utf-8")
            res = run_candidate_isolated(
                candidate_path=p_file,
                public_task=pub_smoke,
                seed=seed0,
                mode="smoke",
                timeout_sec=15.0,
                forbidden_bench_dir=bench_dir,
            )
            if not res.passed or res.candidate_output is None:
                errors.append(
                    f"REPRESENTATION_LOCK_IN_DETECTED: {p_name} failed sandbox execution: {res.error_message}"
                )
                continue
            try:
                scored = harness.score_candidate_output(
                    res.candidate_output, sec_smoke, seed=seed0, split_tier="smoke"
                )
                val = float(scored["primary_metric"])
                if not np.isfinite(val):
                    errors.append(f"REPRESENTATION_LOCK_IN_DETECTED: {p_name} returned non-finite metric {val}")
                else:
                    probe_scores.append(round(val, 5))
                # Also verify degrade_output works on arbitrary policy dicts
                deg = harness.degrade_output(res.candidate_output, sec_smoke, noise_level=0.25, seed=seed0)
                harness.score_candidate_output(deg, sec_smoke, seed=seed0, split_tier="smoke")
            except Exception as e:
                errors.append(
                    f"REPRESENTATION_LOCK_IN_DETECTED: harness.py rejected polymorphic architecture '{p_name}' "
                    f"with {type(e).__name__}: {e}. Move fixed feature/shape assumptions out of harness.py "
                    f"and into candidate.select_action(obs, policy_state)."
                )

    if not errors and len(set(probe_scores)) < 3:
        errors.append(
            f"REPRESENTATION_LOCK_IN_DETECTED: Polymorphic probes yielded identical scores {probe_scores}; "
            f"harness.py is ignoring candidate.select_action."
        )

    return (len(errors) == 0), errors


def _is_better(cand_val: float, ref_val: float, direction: TargetDirection, margin: float = 0.0) -> bool:
    if direction == TargetDirection.MINIMIZE:
        return (ref_val - cand_val) >= margin
    return (cand_val - ref_val) >= margin


def verify_benchmark(bench_dir: Path) -> BenchVerificationReport:
    """Runs all 7 meta-verification checks on a Phase 0 benchmark directory."""
    bench_dir = bench_dir.resolve()
    spec = BenchSpec.model_validate_json((bench_dir / "bench_spec.json").read_text(encoding="utf-8"))
    harness = load_harness_module(bench_dir / "harness.py")
    baseline_path = bench_dir / "baseline.py"
    pos_ctrl_path = bench_dir / "positive_control.py"

    failures: List[str] = []

    # 1. Smoke Test on `baseline.py`
    pub_smoke, sec_smoke = harness.prepare_split(seed=spec.pilot_seeds[0], split_tier="smoke")
    smoke_res = run_candidate_isolated(
        candidate_path=baseline_path,
        public_task=pub_smoke,
        seed=spec.pilot_seeds[0],
        mode="smoke",
        timeout_sec=min(15.0, spec.timeout_sec),
        forbidden_bench_dir=bench_dir,
    )
    smoke_passed = smoke_res.passed and smoke_res.candidate_output is not None
    if not smoke_passed:
        failures.append(f"Baseline smoke test failed: {smoke_res.error_message}")

    # 2. Baseline Pilot Evaluation & External Competence Check
    base_scores: Dict[str, float] = {}
    base_outputs_by_seed: Dict[int, dict] = {}
    sec_by_seed: Dict[int, dict] = {}
    pub_by_seed: Dict[int, dict] = {}
    max_seed_wall_sec: float = 0.0

    for s in spec.pilot_seeds:
        pub, sec = harness.prepare_split(seed=s, split_tier="dev_diagnostic")
        pub_by_seed[s] = pub
        sec_by_seed[s] = sec
        res = run_candidate_isolated(
            candidate_path=baseline_path,
            public_task=pub,
            seed=s,
            mode="full",
            timeout_sec=spec.timeout_sec,
            forbidden_bench_dir=bench_dir,
        )
        if hasattr(res, "wall_clock_sec") and res.wall_clock_sec is not None:
            max_seed_wall_sec = max(max_seed_wall_sec, float(res.wall_clock_sec))
        if not res.passed or res.candidate_output is None:
            failures.append(f"Baseline failed on pilot seed {s}: {res.error_message}")
            continue
        base_outputs_by_seed[s] = res.candidate_output
        scored = harness.score_candidate_output(res.candidate_output, sec, seed=s, split_tier="dev_diagnostic")
        base_scores[f"seed_{s}"] = round(float(scored["primary_metric"]), 5)
        for g in spec.guardrails:
            g_val = float(scored["guardrails"][g.name])
            if not g.evaluate(g_val):
                failures.append(f"Baseline violated guardrail {g.name}={g_val:.4f} ({g.op} {g.threshold}) on seed {s}")

    base_vals = list(base_scores.values()) or [0.0]
    base_mean = round(float(np.mean(base_vals)), 5)
    base_iqm = round(compute_iqm(base_vals), 5)

    baseline_competent = _is_better(
        base_mean, spec.external_anchor.competence_floor, spec.direction, margin=0.0
    )
    if not baseline_competent:
        failures.append(
            f"Baseline score ({base_mean}) failed competence floor ({spec.external_anchor.competence_floor})."
        )

    # 3. Adversarial Null Separation Check
    s0 = spec.pilot_seeds[0]
    null_outputs = harness.generate_null_outputs(pub_by_seed[s0], sec_by_seed[s0], seed=s0)
    null_scores: Dict[str, float] = {}
    null_sep_passed = True
    for null_name, n_out in null_outputs.items():
        sc = harness.score_candidate_output(n_out, sec_by_seed[s0], seed=s0, split_tier="dev_diagnostic")
        n_metric = round(float(sc["primary_metric"]), 5)
        null_scores[null_name] = n_metric
        guards_ok = all(
            g.evaluate(float(sc["guardrails"][g.name])) for g in spec.guardrails
        )
        beats_null = _is_better(base_scores[f"seed_{s0}"], n_metric, spec.direction, margin=spec.rope_epsilon)
        if not beats_null and guards_ok:
            null_sep_passed = False
            failures.append(f"Null probe '{null_name}' ({n_metric}) was not separated from baseline and passed guardrails.")

    # 4. Positive Control Check (`positive_control.py` must beat `baseline.py` by > ROPE)
    pos_scores: Dict[str, float] = {}
    for s in spec.pilot_seeds:
        res_p = run_candidate_isolated(
            candidate_path=pos_ctrl_path,
            public_task=pub_by_seed[s],
            seed=s,
            mode="full",
            timeout_sec=spec.timeout_sec,
            forbidden_bench_dir=bench_dir,
        )
        if hasattr(res_p, "wall_clock_sec") and res_p.wall_clock_sec is not None:
            max_seed_wall_sec = max(max_seed_wall_sec, float(res_p.wall_clock_sec))
        if not res_p.passed or res_p.candidate_output is None:
            failures.append(f"Positive control failed on seed {s}: {res_p.error_message}")
            continue
        sc_p = harness.score_candidate_output(res_p.candidate_output, sec_by_seed[s], seed=s, split_tier="dev_diagnostic")
        pos_scores[f"seed_{s}"] = round(float(sc_p["primary_metric"]), 5)
        for g in spec.guardrails:
            g_val = float(sc_p["guardrails"][g.name])
            if not g.evaluate(g_val):
                failures.append(f"Positive control violated guardrail {g.name}={g_val:.4f} on seed {s}")

    pos_vals = list(pos_scores.values()) or [0.0]
    pos_mean = round(float(np.mean(pos_vals)), 5)
    pos_iqm = round(compute_iqm(pos_vals), 5)
    pos_passed = _is_better(pos_mean, base_mean, spec.direction, margin=spec.rope_epsilon)
    if not pos_passed:
        failures.append(
            f"Positive control ({pos_mean}) did not beat baseline ({base_mean}) by >= ROPE ({spec.rope_epsilon})."
        )

    # 5. Monotonic Sensitivity Degradation Ladder ([0.10, 0.25, 0.50])
    noise_levels = [0.10, 0.25, 0.50]
    ladder_scores: Dict[str, float] = {}
    prev_val = base_mean
    monotonic_ok = True
    for nl in noise_levels:
        nl_vals: List[float] = []
        for s in spec.pilot_seeds:
            deg_out = harness.degrade_output(base_outputs_by_seed[s], sec_by_seed[s], noise_level=nl, seed=s)
            sc_d = harness.score_candidate_output(deg_out, sec_by_seed[s], seed=s, split_tier="dev_diagnostic")
            nl_vals.append(float(sc_d["primary_metric"]))
        mean_nl = round(float(np.mean(nl_vals)), 5)
        ladder_scores[f"noise_{int(nl * 100)}pct"] = mean_nl
        if not _is_better(prev_val, mean_nl, spec.direction, margin=1e-5):
            monotonic_ok = False
        prev_val = mean_nl

    if not monotonic_ok:
        failures.append(f"Sensitivity degradation ladder was not strictly monotonic: {ladder_scores}")

    # 6. Empirical Paired Seed-to-Seed Noise & Required k* Calibration
    paired_diffs = [
        (base_scores[k] - pos_scores[k]) if spec.direction == TargetDirection.MINIMIZE else (pos_scores[k] - base_scores[k])
        for k in base_scores.keys()
        if k in pos_scores
    ]
    sigma_d = round(float(np.std(paired_diffs, ddof=1)) if len(paired_diffs) >= 2 else 0.001, 5)
    k_star = calibrate_required_k(sigma_d, spec.rope_epsilon)

    # 7. Polymorphic Expressivity Probe
    expr_ok, expr_errors = _run_expressivity_probe(
        spec=spec,
        harness=harness,
        pub_smoke=pub_smoke,
        sec_smoke=sec_smoke,
        seed0=spec.pilot_seeds[0],
        bench_dir=bench_dir,
    )
    if not expr_ok:
        failures.extend(expr_errors)

    if failures:
        grade = TrustGrade.FAIL
    elif k_star <= 6:
        grade = TrustGrade.A
    elif k_star <= 10:
        grade = TrustGrade.B
    else:
        grade = TrustGrade.C

    recommended_timeout = round(min(600.0, max(45.0, float(spec.timeout_sec), 8.0 * max_seed_wall_sec)), 1)
    if recommended_timeout > spec.timeout_sec and grade != TrustGrade.FAIL:
        spec.timeout_sec = recommended_timeout
        (bench_dir / "bench_spec.json").write_text(spec.model_dump_json(indent=2), encoding="utf-8")

    report = BenchVerificationReport(
        benchmark_id=spec.benchmark_id,
        smoke_passed=smoke_passed,
        baseline_competent=baseline_competent,
        null_separation_passed=null_sep_passed,
        positive_control_passed=pos_passed,
        sensitivity_monotonic_passed=monotonic_ok,
        expressivity_probe_passed=expr_ok,
        baseline_pilot_scores=base_scores,
        baseline_mean=base_mean,
        baseline_iqm=base_iqm,
        positive_control_scores=pos_scores,
        positive_control_mean=pos_mean,
        positive_control_iqm=pos_iqm,
        null_probe_scores=null_scores,
        degradation_ladder_scores=ladder_scores,
        empirical_sigma_d=sigma_d,
        recommended_k_star=k_star,
        trust_grade=grade,
        failure_reasons=failures,
    )
    (bench_dir / "verification_report.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    return report
