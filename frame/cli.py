"""Unified Zero-Mock CLI for FRAME-AR v2."""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

from frame.bench.registry import (
    approve_benchmark,
    export_stage1_blind_packet,
    export_stage2_cross_packet,
    generate_human_review_markdown,
    get_bench_dir,
    get_benchmark_dir,
    init_benchmark,
    record_stage1_review,
    record_stage2_critique,
    verify_bench_lock,
)
from frame.bench.verifier import verify_benchmark
from frame.brain.tree import CampaignTree
from frame.eval.funnel import FunnelEvaluator
from frame.schemas import EdgeType, OperatorRole
from frame.swarm.brief import export_swarm_specs, generate_context_brief


def _read_text_arg(text: Optional[str], file_path: Optional[str]) -> str:
    if file_path:
        return Path(file_path).read_text(encoding="utf-8")
    if text:
        return text
    raise ValueError("Provide either `--text` or `--file`.")


def cmd_bench_check_lock(args: argparse.Namespace) -> int:
    """Verifies that bench.lock exists and matches SHA-256 digests of all benchmark files."""
    ws = Path(args.workspace).resolve()
    bdir = get_benchmark_dir(ws, args.topic)
    try:
        lock = verify_bench_lock(bdir)
        print(json.dumps({
            "locked": True,
            "benchmark_id": lock.benchmark_id,
            "trust_grade": lock.trust_grade.value,
            "approved_by": lock.approved_by_human,
            "lock_signature": lock.lock_signature,
        }, indent=2))
        return 0
    except Exception as e:
        print(json.dumps({"locked": False, "error": str(e)}, indent=2), file=sys.stderr)
        return 1


def cmd_prepare_jail(args: argparse.Namespace) -> int:
    """Populates an isolated /tmp jail directory with BRIEF.md, parent candidate.py, and meta.env."""
    import shutil
    ws = Path(args.workspace).resolve()
    jail_dir = Path(args.jail_dir).resolve()
    jail_dir.mkdir(parents=True, exist_ok=True)

    # Ensure root node_0 is initialized
    evaluator = FunnelEvaluator(ws, args.topic)
    role_enum = OperatorRole(args.role) if getattr(args, "role", None) else None
    res = generate_context_brief(
        workspace_root=ws,
        topic=args.topic,
        parent_node_id=getattr(args, "parent", None),
        operator_role=role_enum,
    )

    parent_id = res["parent_node_id"]
    role_str = res["operator_role"]
    dir_id = res["directive_id"] or ""

    # 1. Write BRIEF.md
    (jail_dir / "BRIEF.md").write_text(res["brief_markdown"], encoding="utf-8")

    # 2. Copy parent candidate.py into jail
    parent_code_path = evaluator.tree.get_node_code_path(parent_id)
    shutil.copyfile(parent_code_path, jail_dir / "candidate.py")

    # 3. Write shell-sourceable meta.env
    stagnant_steps = evaluator.tree.get_steps_since_last_promotion()
    unconsumed_dirs = len(evaluator.tree.list_directives(unconsumed_only=True))
    meta_lines = [
        f"PARENT_ID='{parent_id}'",
        f"ROLE='{role_str}'",
        f"DIRECTIVE_ID='{dir_id}'",
        f"STAGNANT_STEPS='{stagnant_steps}'",
        f"UNCONSUMED_DIRECTIVES='{unconsumed_dirs}'",
    ]
    (jail_dir / "meta.env").write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
    return 0


def cmd_smoke_check(args: argparse.Namespace) -> int:
    """Runs Gate 0 (AST duplicate check + out-of-process 5s smoke test) without mutating MCTS state."""
    ws = Path(args.workspace).resolve()
    cand_path = Path(args.candidate).resolve()
    if not cand_path.is_file():
        print(f"Candidate file not found: {cand_path}", file=sys.stderr)
        return 1

    evaluator = FunnelEvaluator(ws, args.topic)
    code_text = cand_path.read_text(encoding="utf-8")

    # 1. Check AST duplicate against existing nodes
    is_dup, dup_node, _, _ = evaluator.tree.check_duplicate_ast(code_text)
    if is_dup and dup_node is not None:
        print(
            f"GATE0_AST_DUPLICATE: Candidate code is an exact canonical AST duplicate of existing node "
            f"'{dup_node.node_id}' (status={dup_node.status.value}). You MUST make a structural or "
            f"parametric change to candidate.py.",
            file=sys.stderr,
        )
        return 2

    # 2. Run out-of-process 5s Smoke Test
    from frame.eval.sandbox import run_candidate_isolated
    seed0 = evaluator.spec.pilot_seeds[0]
    pub_smoke, sec_smoke = evaluator.harness.prepare_split(seed=seed0, split_tier="smoke")
    smoke_res = run_candidate_isolated(
        candidate_path=cand_path,
        public_task=pub_smoke,
        seed=seed0,
        mode="smoke",
        timeout_sec=min(15.0, evaluator.spec.timeout_sec),
        forbidden_bench_dir=evaluator.bench_dir,
    )
    if not smoke_res.passed or smoke_res.candidate_output is None:
        print(f"GATE0_SMOKE_CRASH:\n{smoke_res.error_message}", file=sys.stderr)
        return 3

    # Estimate full-split runtime multiplier (full split typically has ~4x episodes/iterations of smoke)
    smoke_budget_sec = max(5.0, evaluator.spec.timeout_sec * 0.22)
    if smoke_res.wall_clock_sec > smoke_budget_sec:
        print(
            f"GATE0_RUNTIME_PROJECTION_EXCEEDED: Smoke test took {smoke_res.wall_clock_sec:.2f}s "
            f"(limit: {smoke_budget_sec:.2f}s for full-seed timeout_sec={evaluator.spec.timeout_sec:.1f}s). "
            f"Your candidate will time out on the full training split in Gate 1! "
            f"Vectorize your simulation/rollout loop or reduce population size / maxiter.",
            file=sys.stderr,
        )
        return 5

    try:
        scored = evaluator.harness.score_candidate_output(
            smoke_res.candidate_output, sec_smoke, seed=seed0, split_tier="smoke"
        )
        val = float(scored["primary_metric"])
        if not __import__("numpy").isfinite(val):
            raise ValueError(f"Primary metric is non-finite ({val})")
    except Exception as e:
        print(f"GATE0_SMOKE_SCORING_ERROR:\n{e}", file=sys.stderr)
        return 4

    print(f"GATE0_SMOKE_OK: metric={val:.5f} ({smoke_res.wall_clock_sec:.2f}s)")
    return 0


def cmd_extract_code(args: argparse.Namespace) -> int:
    """Checks if candidate.py was modified in-place on disk, or extracts a fenced Python block from stdout."""
    import re
    cand_path = Path(args.candidate).resolve()
    backup_path = Path(args.backup).resolve()
    stdout_path = Path(args.stdout_log).resolve()

    # 1. Primary path: agy modified candidate.py in-place using its file edit tool
    if cand_path.is_file() and backup_path.is_file():
        if cand_path.read_text(encoding="utf-8").strip() != backup_path.read_text(encoding="utf-8").strip():
            print("IN_PLACE_MODIFIED")
            return 0

    # 2. Fallback path: extract ```python ... ``` block from agy_stdout.log
    if not stdout_path.is_file():
        print("NO_STDOUT_LOG", file=sys.stderr)
        return 1

    raw_out = stdout_path.read_text(encoding="utf-8", errors="replace")
    if not raw_out.strip():
        print("EMPTY_STDOUT_LOG", file=sys.stderr)
        return 1

    blocks = re.findall(r"```[ \t]*(?:python[0-9.]*|py)?[ \t]*\r?\n(.*?)```", raw_out, flags=re.DOTALL | re.IGNORECASE)
    for block in reversed(blocks):
        if "def run_experiment" in block:
            cand_path.write_text(block.strip() + "\n", encoding="utf-8")
            print("EXTRACTED_FENCED_BLOCK")
            return 0

    if "def run_experiment" in raw_out and "```" not in raw_out:
        lines = raw_out.splitlines()
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ", "def ", '"""', "'''", "#")):
                start_idx = i
                break
        extracted = "\n".join(lines[start_idx:]).strip() + "\n"
        cand_path.write_text(extracted, encoding="utf-8")
        print("EXTRACTED_RAW_PYTHON")
        return 0

    print("NO_MODIFICATION_DETECTED: NO_PYTHON_CODE_FOUND_IN_STDOUT", file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frame", description="FRAME-AR v2 Autonomous Research Engine")
    parser.add_argument("--workspace", default=".", help="Workspace root directory")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # 1. `frame bench` subcommands
    p_bench = sub.add_parser("bench", help="Phase 0 Benchmark Synthesis, Audit & Lock")
    bench_sub = p_bench.add_subparsers(dest="bench_cmd", required=True)

    b_init = bench_sub.add_parser("init", help="Initialize benchmark from domain template")
    b_init.add_argument("--topic", required=True)
    b_init.add_argument("--domain", choices=["kaggle_tabular", "rl_control"], required=True)

    b_ver = bench_sub.add_parser("verify", help="Run 6-check benchmark meta-verification")
    b_ver.add_argument("--topic", required=True)

    b_s1_exp = bench_sub.add_parser("export-stage1", help="Export Stage 1 blind review packet")
    b_s1_exp.add_argument("--topic", required=True)

    b_s1_rec = bench_sub.add_parser("record-stage1", help="Record Stage 1 blind review from Gemini or Claude")
    b_s1_rec.add_argument("--topic", required=True)
    b_s1_rec.add_argument("--model", choices=["gemini", "claude"], required=True)
    b_s1_rec.add_argument("--text", default=None)
    b_s1_rec.add_argument("--file", default=None)

    b_s2_exp = bench_sub.add_parser("export-stage2", help="Export Stage 2 cross-critique packet")
    b_s2_exp.add_argument("--topic", required=True)
    b_s2_exp.add_argument("--model", choices=["gemini", "claude"], required=True)

    b_s2_rec = bench_sub.add_parser("record-stage2", help="Record Stage 2 cross-critique from Gemini or Claude")
    b_s2_rec.add_argument("--topic", required=True)
    b_s2_rec.add_argument("--model", choices=["gemini", "claude"], required=True)
    b_s2_rec.add_argument("--text", default=None)
    b_s2_rec.add_argument("--file", default=None)

    b_app = bench_sub.add_parser("approve", help="Human Gate: Approve benchmark and create bench.lock")
    b_app.add_argument("--topic", required=True)
    b_app.add_argument("--reviewer", required=True)
    b_app.add_argument("--notes", default="")
    b_app.add_argument("--require-committee", action="store_true")

    # Inside bench_sub in build_parser():
    b_chk = bench_sub.add_parser("check-lock", help="Verify cryptographic bench.lock integrity")
    b_chk.add_argument("--topic", type=str, required=True)
    b_chk.set_defaults(func=cmd_bench_check_lock)

    # 2. `frame brief`
    p_brief = sub.add_parser("brief", help="Generate leak-free Context Firewall Brief for a subagent")
    p_brief.add_argument("--topic", required=True)
    p_brief.add_argument("--parent", default=None)
    p_brief.add_argument("--role", choices=[r.value for r in OperatorRole], default=None)

    # 3. `frame step`
    p_step = sub.add_parser("step", help="Execute 4-Gate Verification Funnel on a candidate script")
    p_step.add_argument("--topic", required=True)
    p_step.add_argument("--candidate", required=True)
    p_step.add_argument("--parent", default=None)
    p_step.add_argument("--role", choices=[r.value for r in OperatorRole], default=OperatorRole.SPEC_IMPLEMENTATION.value)
    p_step.add_argument("--edge-type", choices=[e.value for e in EdgeType], default=EdgeType.ATOMIC_TRANSITION.value)
    p_step.add_argument("--title", default="")
    p_step.add_argument("--directive", default=None)

    # 4. `frame inject`
    p_inj = sub.add_parser("inject", help="Enqueue a priority Human/Frontier hypothesis onto any node")
    p_inj.add_argument("--topic", required=True)
    p_inj.add_argument("--parent", required=True)
    p_inj.add_argument("--spec", required=True)
    p_inj.add_argument("--role", choices=[r.value for r in OperatorRole], default=OperatorRole.SPEC_IMPLEMENTATION.value)
    p_inj.add_argument("--priority", type=int, default=0)
    p_inj.add_argument("--author", default="human_or_frontier")

    # 5. `frame abort`
    p_abt = sub.add_parser("abort", help="Preempt an active worker PID and optionally inject a replacement spec")
    p_abt.add_argument("--topic", required=True)
    p_abt.add_argument("--node", required=True)
    p_abt.add_argument("--inject-spec", default=None)

    # 6. `frame swarm`
    p_swm = sub.add_parser("swarm", help="Export 4 mechanical operator briefs for parallel Flash subagents")
    p_swm.add_argument("--topic", required=True)

    # 7. `frame status`
    p_sts = sub.add_parser("status", help="Inspect MCTS campaign tree, Incumbent M*, and directive queue")
    p_sts.add_argument("--topic", required=True)

    # 8. `frame vault-eval`
    p_vlt = sub.add_parser("vault-eval", help="Evaluate Incumbent M* on the sealed Vault-Holdout split")
    p_vlt.add_argument("--topic", required=True)
    p_vlt.add_argument("--node", default=None)

    # Top-level subparsers in build_parser():
    p_jail = sub.add_parser("prepare-jail", help="Populate isolated /tmp jail with BRIEF.md and candidate.py")
    p_jail.add_argument("--topic", type=str, required=True)
    p_jail.add_argument("--jail-dir", type=str, required=True)
    p_jail.add_argument("--parent", type=str, default=None)
    p_jail.add_argument("--role", type=str, default=None)
    p_jail.set_defaults(func=cmd_prepare_jail)

    p_smoke = sub.add_parser("smoke-check", help="Run stateless Gate 0 AST + 5s smoke check on a candidate")
    p_smoke.add_argument("--topic", type=str, required=True)
    p_smoke.add_argument("--candidate", type=str, required=True)
    p_smoke.set_defaults(func=cmd_smoke_check)

    p_ext = sub.add_parser("extract-code", help="Verify in-place edit or extract Python block from worker stdout")
    p_ext.add_argument("--candidate", type=str, required=True)
    p_ext.add_argument("--backup", type=str, required=True)
    p_ext.add_argument("--stdout-log", type=str, required=True)
    p_ext.set_defaults(func=cmd_extract_code)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        return args.func(args)
    ws = Path(args.workspace).resolve()

    if args.cmd == "bench":
        bdir = get_bench_dir(ws, args.topic)
        if args.bench_cmd == "init":
            path = init_benchmark(ws, args.topic, args.domain)
            print(json.dumps({"status": "initialized", "bench_dir": str(path)}, indent=2))
            return 0
        if args.bench_cmd == "verify":
            rep = verify_benchmark(bdir)
            rev_md = generate_human_review_markdown(bdir)
            print(rep.model_dump_json(indent=2))
            print(f"\n[Review Card Generated]: {rev_md}")
            return 0 if rep.trust_grade.value != "FAIL" else 1
        if args.bench_cmd == "export-stage1":
            p = export_stage1_blind_packet(bdir)
            print(json.dumps({"stage1_blind_packet": str(p)}, indent=2))
            return 0
        if args.bench_cmd == "record-stage1":
            txt = _read_text_arg(args.text, args.file)
            rec = record_stage1_review(bdir, args.model, txt)
            print(rec.model_dump_json(indent=2))
            return 0
        if args.bench_cmd == "export-stage2":
            p = export_stage2_cross_packet(bdir, args.model)
            print(json.dumps({"stage2_cross_packet": str(p)}, indent=2))
            return 0
        if args.bench_cmd == "record-stage2":
            txt = _read_text_arg(args.text, args.file)
            rec = record_stage2_critique(bdir, args.model, txt)
            print(rec.model_dump_json(indent=2))
            return 0
        if args.bench_cmd == "approve":
            lock = approve_benchmark(bdir, args.reviewer, args.notes, args.require_committee)
            print(lock.model_dump_json(indent=2))
            return 0

    if args.cmd == "brief":
        role = OperatorRole(args.role) if args.role else None
        pkt = generate_context_brief(ws, args.topic, parent_node_id=args.parent, operator_role=role)
        print(pkt["brief_markdown"])
        return 0

    if args.cmd == "step":
        funnel = FunnelEvaluator(ws, args.topic)
        rec = funnel.evaluate_step(
            candidate_path=Path(args.candidate),
            parent_id=args.parent,
            operator_role=OperatorRole(args.role),
            edge_type=EdgeType(args.edge_type),
            title=args.title,
            directive_id=args.directive,
        )
        # Enforce 1-bit Confirmatory telemetry privacy: mask raw per-seed confirmatory scores in stdout receipt
        receipt = {
            "node_id": rec.node_id,
            "parent_id": rec.parent_id,
            "status": rec.status.value,
            "promoted": rec.promoted,
            "incumbent_node_id": funnel.tree.incumbent_node_id,
            "pilot_primary_mean": rec.pilot_primary_mean,
            "pilot_primary_iqm": rec.pilot_primary_iqm,
            "pilot_delta_vs_parent": rec.pilot_delta_vs_parent,
            "guardrails_passed": rec.guardrails_passed,
            "failed_guardrails": rec.failed_guardrails,
            "dev_diagnostics": rec.dev_diagnostics,
            "confirmatory_verdict": (
                {
                    "lord_step": rec.confirmatory_step,
                    "directional_effect": rec.confirmatory_effect,
                    "p_value_rope": rec.p_value_rope,
                    "allocated_alpha_t": rec.allocated_alpha_t,
                    "is_discovery": rec.promoted,
                }
                if rec.confirmatory_step is not None
                else None
            ),
            "error_traceback": rec.error_traceback,
            "wall_clock_sec": round(rec.wall_clock_sec, 3),
        }
        print(json.dumps(receipt, indent=2))
        return 0

    if args.cmd == "inject":
        FunnelEvaluator(ws, args.topic)
        tree = CampaignTree(ws, args.topic)
        ticket = tree.enqueue_directive(
            parent_node_id=args.parent,
            hypothesis_spec=args.spec,
            operator_role=OperatorRole(args.role),
            priority=args.priority,
            author=args.author,
        )
        print(ticket.model_dump_json(indent=2))
        return 0

    if args.cmd == "abort":
        tree = CampaignTree(ws, args.topic)
        res = tree.abort_running_node(args.node, inject_replacement_spec=args.inject_spec)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "swarm":
        pkts = export_swarm_specs(ws, args.topic)
        print(json.dumps({"exported_operator_briefs": len(pkts)}, indent=2))
        return 0

    if args.cmd == "status":
        tree = CampaignTree(ws, args.topic)
        summary = {
            "topic": tree.topic,
            "incumbent_single_model_M_star": tree.incumbent_node_id,
            "best_blend_node_id": tree.best_blend_node_id,
            "total_nodes": len(tree.nodes),
            "queued_directives": [d.model_dump(mode="json") for d in tree.get_queued_directives()],
            "nodes": [
                {
                    "node_id": n.node_id,
                    "parent_id": n.parent_id,
                    "role": n.operator_role.value,
                    "status": n.status.value,
                    "pilot_mean": n.pilot_primary_mean,
                    "pilot_iqm": n.pilot_primary_iqm,
                    "promoted": n.promoted,
                }
                for n in tree.nodes.values()
            ],
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "vault-eval":
        funnel = FunnelEvaluator(ws, args.topic)
        rep = funnel.evaluate_vault_holdout(node_id=args.node)
        print(json.dumps(rep, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
