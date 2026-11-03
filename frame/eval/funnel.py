"""4-Gate Out-of-Process Verification Funnel (`frame step`) for FRAME-AR v2."""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import numpy as np

from frame.bench.registry import verify_bench_lock
from frame.brain.tree import CampaignTree
from frame.eval.sandbox import load_harness_module, run_candidate_isolated
from frame.eval.statistics import (
    LORDController,
    compute_iqm,
    evaluate_iqm_bootstrap_rope,
    evaluate_paired_t_rope,
    generate_step_seeds,
)
from frame.schemas import (
    BenchSpec,
    BenchVerificationReport,
    EdgeType,
    NodeRecord,
    NodeStatus,
    OperatorRole,
    StatisticalEstimator,
    TargetDirection,
)
from frame.utils.canonical_json import compute_file_sha256


class FunnelEvaluator:
    """Executes the tamper-proof 4-Gate evaluation cascade on a candidate script."""

    def __init__(self, workspace_root: Path, topic: str, secret_key: str = "frame_v2_default_hmac_key") -> None:
        self.workspace_root = workspace_root.resolve()
        self.topic = topic.strip().lower().replace(" ", "_")
        self.secret_key = secret_key
        self.tree = CampaignTree(self.workspace_root, self.topic)
        self.bench_dir = self.tree.bench_dir

        # 1. Verify Human Approval Gate & Cryptographic Lock
        self.lock = verify_bench_lock(self.bench_dir)
        self.spec = BenchSpec.model_validate_json(
            (self.bench_dir / "bench_spec.json").read_text(encoding="utf-8")
        )
        self.harness = load_harness_module(self.bench_dir / "harness.py")
        self.lord_path = self.tree.campaign_dir / "lord_state.json"
        self.lord = LORDController.load(self.lord_path, default_alpha=0.05)

        # Ensure root node_0 is initialized
        if "node_0" not in self.tree.nodes:
            rep = BenchVerificationReport.model_validate_json(
                (self.bench_dir / "verification_report.json").read_text(encoding="utf-8")
            )
            pub0, sec0 = self.harness.prepare_split(seed=self.spec.pilot_seeds[0], split_tier="dev_diagnostic")
            res0 = run_candidate_isolated(
                self.bench_dir / "baseline.py",
                pub0,
                self.spec.pilot_seeds[0],
                "full",
                timeout_sec=self.spec.timeout_sec,
                forbidden_bench_dir=self.bench_dir,
            )
            diag0 = {}
            if res0.passed and res0.candidate_output is not None:
                sc0 = self.harness.score_candidate_output(
                    res0.candidate_output, sec0, self.spec.pilot_seeds[0], "dev_diagnostic"
                )
                diag0 = sc0.get("diagnostics", {})
            self.tree.initialize_from_benchmark(self.spec, rep, diag0)

    def _directional_gain(self, base_val: float, cand_val: float) -> float:
        if self.spec.direction == TargetDirection.MINIMIZE:
            return float(base_val - cand_val)
        return float(cand_val - base_val)

    def evaluate_step(
        self,
        candidate_path: Path,
        parent_id: Optional[str] = None,
        operator_role: OperatorRole = OperatorRole.SPEC_IMPLEMENTATION,
        edge_type: EdgeType = EdgeType.ATOMIC_TRANSITION,
        title: str = "",
        description: str = "",
        directive_id: Optional[str] = None,
        allocated_node_id: Optional[str] = None,
    ) -> NodeRecord:
        """Runs Gate 0 -> Gate 1 -> Gate 2 on `candidate_path`."""
        t_start = time.perf_counter()
        node_id = allocated_node_id or self.tree.allocate_next_node_id()
        eff_parent_id = parent_id or self.tree.incumbent_node_id
        if eff_parent_id not in self.tree.nodes:
            raise KeyError(f"Parent node '{eff_parent_id}' not found in campaign tree.")
        parent_node = self.tree.nodes[eff_parent_id]

        # Enforce Blend Edge Protocol: blends are always screened against Incumbent Best Single Model (M*)
        reference_node_id = (
            self.tree.incumbent_node_id
            if edge_type == EdgeType.BLEND_ENSEMBLE
            else eff_parent_id
        )
        reference_node = self.tree.nodes[reference_node_id]

        code_text = candidate_path.read_text(encoding="utf-8")
        code_sha = compute_file_sha256(candidate_path)
        is_dup, dup_node, ast_fp, op_tokens = self.tree.check_duplicate_ast(code_text)

        node_rec = NodeRecord(
            node_id=node_id,
            parent_id=eff_parent_id,
            branch_id=f"{parent_node.branch_id}/{operator_role.value}" if parent_node.depth == 0 else parent_node.branch_id,
            depth=parent_node.depth + 1,
            edge_type=edge_type,
            operator_role=operator_role,
            directive_id=directive_id,
            title=title or f"{operator_role.value} mutation from {eff_parent_id}",
            description=description,
            code_sha256=code_sha,
            ast_fingerprint=ast_fp,
            operator_tokens=op_tokens,
        )

        # ======================================================================
        # GATE 0: AST Deduplication & Out-of-Process Smoke Test
        # ======================================================================
        if is_dup and dup_node is not None:
            node_rec.status = NodeStatus.BUGGY
            node_rec.error_traceback = (
                f"Gate 0 Blocked: Exact canonical AST duplicate of {dup_node.node_id} (status={dup_node.status.value})."
            )
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        pid_file = self.tree.active_dir / f"{node_id}.pid"
        node_artifact_dir = self.tree.nodes_dir / node_id / "artifacts"

        pub_smoke, sec_smoke = self.harness.prepare_split(seed=self.spec.pilot_seeds[0], split_tier="smoke")
        smoke_res = run_candidate_isolated(
            candidate_path=candidate_path,
            public_task=pub_smoke,
            seed=self.spec.pilot_seeds[0],
            mode="smoke",
            timeout_sec=min(15.0, self.spec.timeout_sec),
            forbidden_bench_dir=self.bench_dir,
            pid_file=pid_file,
        )
        if smoke_res.aborted:
            node_rec.status = NodeStatus.ABORTED
            node_rec.error_traceback = smoke_res.error_message
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        if not smoke_res.passed or smoke_res.candidate_output is None:
            node_rec.status = NodeStatus.BUGGY
            node_rec.error_traceback = f"Gate 0 Smoke Execution Failed:\n{smoke_res.error_message}"
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        try:
            smoke_scored = self.harness.score_candidate_output(
                smoke_res.candidate_output, sec_smoke, seed=self.spec.pilot_seeds[0], split_tier="smoke"
            )
            if not np.isfinite(float(smoke_scored["primary_metric"])):
                raise ValueError("Primary metric is NaN or Inf on smoke check.")
        except Exception as e:
            node_rec.status = NodeStatus.BUGGY
            node_rec.error_traceback = f"Gate 0 Smoke Output Scoring Failed: {e}"
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        # ======================================================================
        # GATE 1: `Dev-Diagnostic` Guardrails + k=3 CRN Pilot Futility Screen
        # ======================================================================
        pilot_scores: Dict[str, float] = {}
        guardrail_accum: Dict[str, List[float]] = {}
        failed_guards: List[str] = []
        latest_diag: Dict[str, Any] = {}

        for idx, s in enumerate(self.spec.pilot_seeds):
            pub_dev, sec_dev = self.harness.prepare_split(seed=s, split_tier="dev_diagnostic")
            res_dev = run_candidate_isolated(
                candidate_path=candidate_path,
                public_task=pub_dev,
                seed=s,
                mode="full",
                timeout_sec=self.spec.timeout_sec,
                forbidden_bench_dir=self.bench_dir,
                pid_file=pid_file,
                persist_artifacts_to=(node_artifact_dir if idx == 0 else None),
            )
            if res_dev.aborted:
                node_rec.status = NodeStatus.ABORTED
                node_rec.error_traceback = res_dev.error_message
                node_rec.wall_clock_sec = time.perf_counter() - t_start
                self.tree.record_node(node_rec, candidate_path, reward=0.0)
                return node_rec

            if not res_dev.passed or res_dev.candidate_output is None:
                node_rec.status = NodeStatus.BUGGY
                node_rec.error_traceback = f"Gate 1 Failed on pilot seed {s}:\n{res_dev.error_message}"
                node_rec.wall_clock_sec = time.perf_counter() - t_start
                self.tree.record_node(node_rec, candidate_path, reward=0.0)
                return node_rec

            scored_dev = self.harness.score_candidate_output(
                res_dev.candidate_output, sec_dev, seed=s, split_tier="dev_diagnostic"
            )
            pilot_scores[f"seed_{s}"] = round(float(scored_dev["primary_metric"]), 5)
            if idx == 0:
                latest_diag = scored_dev.get("diagnostics", {})

            for g_spec in self.spec.guardrails:
                g_val = float(scored_dev.get("guardrails", {}).get(g_spec.name, float("nan")))
                guardrail_accum.setdefault(g_spec.name, []).append(g_val)
                if not np.isfinite(g_val) or not g_spec.evaluate(g_val):
                    msg = f"{g_spec.name}={g_val:.4f} (violated {g_spec.op} {g_spec.threshold} on seed {s})"
                    if msg not in failed_guards:
                        failed_guards.append(msg)

        p_vals = list(pilot_scores.values())
        node_rec.pilot_scores = pilot_scores
        node_rec.pilot_primary_mean = round(float(np.mean(p_vals)), 5)
        node_rec.pilot_primary_iqm = round(compute_iqm(p_vals), 5)
        node_rec.guardrail_values = {k: round(float(np.mean(v)), 4) for k, v in guardrail_accum.items()}
        node_rec.guardrails_passed = len(failed_guards) == 0
        node_rec.failed_guardrails = failed_guards
        node_rec.dev_diagnostics = latest_diag

        use_iqm = self.spec.estimator == StatisticalEstimator.IQM_BOOTSTRAP_ROPE
        cand_pilot_rep = node_rec.pilot_primary_iqm if use_iqm else node_rec.pilot_primary_mean

        parent_pilot_rep = (
            parent_node.pilot_primary_iqm if use_iqm else parent_node.pilot_primary_mean
        )
        if parent_pilot_rep is None:
            parent_pilot_rep = cand_pilot_rep

        incumbent_node = self.tree.nodes[self.tree.incumbent_node_id]
        incumbent_pilot_rep = (
            incumbent_node.pilot_primary_iqm if use_iqm else incumbent_node.pilot_primary_mean
        )
        if incumbent_pilot_rep is None:
            incumbent_pilot_rep = parent_pilot_rep

        pilot_delta_vs_parent = self._directional_gain(float(parent_pilot_rep), float(cand_pilot_rep))
        pilot_delta_vs_incumbent = self._directional_gain(float(incumbent_pilot_rep), float(cand_pilot_rep))
        node_rec.pilot_delta_vs_parent = round(pilot_delta_vs_parent, 5)

        if not node_rec.guardrails_passed:
            node_rec.status = NodeStatus.GUARDRAIL_REJECTED
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        # Candidate must beat BOTH its parent AND the reigning Incumbent M* on pilot seeds
        if pilot_delta_vs_parent <= 0.0 or pilot_delta_vs_incumbent <= 0.0:
            node_rec.status = NodeStatus.FUTILITY_PRUNED
            node_rec.wall_clock_sec = time.perf_counter() - t_start
            self.tree.record_node(node_rec, candidate_path, reward=0.0)
            return node_rec

        # ======================================================================
        # GATE 2: `Confirmatory` Split (Fresh Step-Indexed Seeds + ROPE + LORD-3)
        # Re-evaluates BOTH Challenger and Incumbent M* on fresh seeds.
        # ======================================================================
        next_lord_step = self.lord.current_step + 1
        conf_seeds = generate_step_seeds(
            secret_key=self.secret_key,
            benchmark_id=self.spec.benchmark_id,
            confirmatory_step=next_lord_step,
            count=self.spec.confirmatory_k,
        )

        incumbent_code_path = self.tree.get_node_code_path(self.tree.incumbent_node_id)
        chal_conf_scores: Dict[str, float] = {}
        inc_conf_scores: Dict[str, float] = {}

        for s in conf_seeds:
            pub_conf, sec_conf = self.harness.prepare_split(seed=s, split_tier="confirmatory")

            res_chal = run_candidate_isolated(
                candidate_path=candidate_path,
                public_task=pub_conf,
                seed=s,
                mode="full",
                timeout_sec=self.spec.timeout_sec,
                forbidden_bench_dir=self.bench_dir,
                pid_file=pid_file,
            )
            if res_chal.aborted:
                node_rec.status = NodeStatus.ABORTED
                node_rec.wall_clock_sec = time.perf_counter() - t_start
                self.tree.record_node(node_rec, candidate_path, reward=0.0)
                return node_rec
            if not res_chal.passed or res_chal.candidate_output is None:
                node_rec.status = NodeStatus.BUGGY
                node_rec.error_traceback = f"Gate 2 Challenger failed on confirmatory seed {s}:\n{res_chal.error_message}"
                node_rec.wall_clock_sec = time.perf_counter() - t_start
                self.tree.record_node(node_rec, candidate_path, reward=0.0)
                return node_rec

            sc_chal = self.harness.score_candidate_output(
                res_chal.candidate_output, sec_conf, seed=s, split_tier="confirmatory"
            )
            chal_conf_scores[f"seed_{s}"] = round(float(sc_chal["primary_metric"]), 5)

            for g_spec in self.spec.guardrails:
                g_val = float(sc_chal.get("guardrails", {}).get(g_spec.name, float("nan")))
                if not np.isfinite(g_val) or not g_spec.evaluate(g_val):
                    node_rec.status = NodeStatus.GUARDRAIL_REJECTED
                    node_rec.failed_guardrails.append(f"Confirmatory guardrail {g_spec.name}={g_val:.4f}")
                    node_rec.wall_clock_sec = time.perf_counter() - t_start
                    self.tree.record_node(node_rec, candidate_path, reward=0.0)
                    return node_rec

            res_inc = run_candidate_isolated(
                candidate_path=incumbent_code_path,
                public_task=pub_conf,
                seed=s,
                mode="full",
                timeout_sec=self.spec.timeout_sec,
                forbidden_bench_dir=self.bench_dir,
            )
            sc_inc = self.harness.score_candidate_output(
                res_inc.candidate_output, sec_conf, seed=s, split_tier="confirmatory"
            )
            inc_conf_scores[f"seed_{s}"] = round(float(sc_inc["primary_metric"]), 5)

        active_rope = (
            self.spec.blend_rope_epsilon
            if edge_type == EdgeType.BLEND_ENSEMBLE
            else self.spec.rope_epsilon
        )

        if self.spec.estimator == StatisticalEstimator.IQM_BOOTSTRAP_ROPE:
            stat_res = evaluate_iqm_bootstrap_rope(
                incumbent_scores=inc_conf_scores,
                challenger_scores=chal_conf_scores,
                direction=self.spec.direction,
                rope_epsilon=active_rope,
                lord_controller=self.lord,
            )
        else:
            stat_res = evaluate_paired_t_rope(
                incumbent_scores=inc_conf_scores,
                challenger_scores=chal_conf_scores,
                direction=self.spec.direction,
                rope_epsilon=active_rope,
                lord_controller=self.lord,
            )

        self.lord.save(self.lord_path)

        node_rec.confirmatory_step = stat_res.lord_step
        node_rec.confirmatory_seeds = conf_seeds
        node_rec.confirmatory_challenger_scores = chal_conf_scores
        node_rec.confirmatory_incumbent_scores = inc_conf_scores
        node_rec.confirmatory_effect = round(stat_res.directional_effect, 5)
        node_rec.confirmatory_ci_lower = round(stat_res.ci_lower_bound, 5)
        node_rec.p_value_rope = round(stat_res.p_value_rope, 6)
        node_rec.allocated_alpha_t = round(stat_res.allocated_alpha_t, 6)
        node_rec.promoted = stat_res.is_discovery
        node_rec.status = (
            NodeStatus.PROMOTED_DISCOVERY if stat_res.is_discovery else NodeStatus.CONFIRMATORY_NULL
        )
        node_rec.wall_clock_sec = time.perf_counter() - t_start

        reward = 1.0 + max(0.0, stat_res.directional_effect / active_rope) if stat_res.is_discovery else 0.15
        self.tree.record_node(node_rec, candidate_path, reward=reward)
        return node_rec

    def evaluate_vault_holdout(self, node_id: Optional[str] = None) -> Dict[str, Any]:
        """Gate 3 (Campaign Closure): Evaluates promoted incumbent on untouched Vault-Holdout split."""
        target_id = node_id or self.tree.incumbent_node_id
        node = self.tree.nodes[target_id]
        code_path = self.tree.get_node_code_path(target_id)

        pub_v, sec_v = self.harness.prepare_split(seed=99999, split_tier="vault_holdout")
        res_v = run_candidate_isolated(
            candidate_path=code_path,
            public_task=pub_v,
            seed=99999,
            mode="full",
            timeout_sec=self.spec.timeout_sec,
            forbidden_bench_dir=self.bench_dir,
        )
        if not res_v.passed or res_v.candidate_output is None:
            raise RuntimeError(f"Vault evaluation failed for {target_id}: {res_v.error_message}")

        scored_v = self.harness.score_candidate_output(
            res_v.candidate_output, sec_v, seed=99999, split_tier="vault_holdout"
        )
        vault_score = float(scored_v["primary_metric"])
        ref_score = float(
            node.pilot_primary_iqm
            if self.spec.estimator == StatisticalEstimator.IQM_BOOTSTRAP_ROPE
            else node.pilot_primary_mean
        )
        drift = abs(vault_score - ref_score)
        passed_overfit = drift <= self.spec.vault_overfit_tolerance

        report = {
            "node_id": target_id,
            "vault_score": round(vault_score, 5),
            "dev_pilot_score": round(ref_score, 5),
            "generalization_drift": round(drift, 5),
            "vault_overfit_tolerance": self.spec.vault_overfit_tolerance,
            "passed_vault_audit": passed_overfit,
            "guardrails": scored_v.get("guardrails", {}),
        }
        (self.tree.campaign_dir / "vault_audit.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
