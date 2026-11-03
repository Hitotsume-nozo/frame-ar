#!/usr/bin/env python3
"""Out-of-Band Campaign Monitor & Lock-Free Hypothesis Injector (`frame-mon.py`).

Run in a second Kitty terminal pane while `frame-orchestrate.sh` runs in Pane 1:
  python3 frame-mon.py --topic <slug> --watch
  python3 frame-mon.py --topic <slug> --inject-parent node_0 --spec "..."
  python3 frame-mon.py --topic <slug> --pause
  python3 frame-mon.py --topic <slug> --abort node_3 --spec "..."
"""

import argparse
import os
from pathlib import Path
import time

from frame.brain.tree import CampaignTree
from frame.eval.statistics import LORDController
from frame.schemas import BenchSpec, OperatorRole


def render_dashboard(ws: Path, topic: str) -> str:
    tree = CampaignTree(ws, topic)
    spec_path = tree.bench_dir / "bench_spec.json"
    metric_info = ""
    if spec_path.is_file():
        spec = BenchSpec.model_validate_json(spec_path.read_text(encoding="utf-8"))
        metric_info = f"{spec.primary_metric_name} ({spec.direction.value}, ROPE={spec.rope_epsilon})"

    lord = LORDController.load(tree.campaign_dir / "lord_state.json")
    directives = tree.list_directives(unconsumed_only=True)
    active_pids = [p.stem for p in tree.active_dir.glob("*.pid")]
    paused = (tree.campaign_dir / "signals" / "pause").is_file()

    lines = [
        "=" * 88,
        f" FRAME-AR v2.1 LIVE MONITOR | Topic: {tree.topic} | {'[PAUSED]' if paused else '[ACTIVE]'}",
        f" Metric: {metric_info} | Incumbent M*: {tree.incumbent_node_id} | Best Blend: {tree.best_blend_node_id or 'None'}",
        f" LORD-3 Step: {lord.current_step} | Next Alpha: {lord.peek_next_alpha():.6f} | Discoveries: {len(lord.discovery_steps)}",
        f" In-Flight Sandbox Worker(s): {active_pids if active_pids else 'None'}",
        "=" * 88,
        f" {'Node':8s} | {'Parent':8s} | {'Operator':19s} | {'Status':19s} | {'Pilot IQM/Mean':14s} | {'Gate1 Δ':9s} | {'Gate2 p':9s}",
        "-" * 88,
    ]

    for nid, n in tree.nodes.items():
        p_score = n.pilot_primary_iqm if n.pilot_primary_iqm is not None else n.pilot_primary_mean
        p_str = f"{p_score:.4f}" if p_score is not None else "N/A"
        d_str = f"{n.pilot_delta_vs_parent:+.4f}" if n.pilot_delta_vs_parent is not None else "N/A"
        pv_str = f"{n.p_value_rope:.5f}" if n.p_value_rope is not None else "-"
        marker = "*" if nid == tree.incumbent_node_id else " "
        lines.append(
            f"{marker}{nid:8s} | {str(n.parent_id or '-'):8s} | {n.operator_role.value:19s} | "
            f"{n.status.value:19s} | {p_str:14s} | {d_str:9s} | {pv_str:9s}"
        )

    lines.append("-" * 88)
    lines.append(f" Queued Priority Directives ({len(directives)} unconsumed):")
    if directives:
        for d in directives:
            lines.append(f"  • [{d.directive_id}] P{d.priority} on {d.parent_node_id} ({d.operator_role.value}): {d.hypothesis_spec[:65]}")
    else:
        lines.append("  • None")
    lines.append("=" * 88)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FRAME-AR Out-of-Band Monitor & Hypothesis Injector")
    parser.add_argument("--workspace", type=str, default=".")
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--watch", action="store_true", help="Continuously refresh dashboard every 2 seconds")
    parser.add_argument("--inject-parent", type=str, default=None, help="Parent node_id to attach a directive to")
    parser.add_argument("--spec", type=str, default=None, help="Hypothesis specification text to inject")
    parser.add_argument("--role", type=str, default="spec_implementation", help="Operator role for injected directive")
    parser.add_argument("--priority", type=int, default=0, help="Priority 0 (highest) to 2")
    parser.add_argument("--pause", action="store_true", help="Drop a pause signal for frame-orchestrate.sh")
    parser.add_argument("--abort", type=str, default=None, help="Abort a running node_id via SIGTERM")
    args = parser.parse_args()

    ws = Path(args.workspace).resolve()
    tree = CampaignTree(ws, args.topic)

    if args.pause:
        sig_dir = tree.campaign_dir / "signals"
        sig_dir.mkdir(parents=True, exist_ok=True)
        (sig_dir / "pause").write_text("paused\n", encoding="utf-8")
        print(f"[+] Pause signal written to {sig_dir / 'pause'}. Hypervisor will pause before next step.")
        return

    if args.abort:
        res = tree.abort_running_node(node_id=args.abort, replacement_spec=args.spec)
        print(res)
        return

    if args.inject_parent and args.spec:
        ticket = tree.inject_directive(
            parent_node_id=args.inject_parent,
            hypothesis_spec=args.spec,
            operator_role=OperatorRole(args.role),
            priority=args.priority,
            author="kitty_monitor_user",
        )
        print(f"[+] Injected Priority Directive {ticket.directive_id} attached to {ticket.parent_node_id}.")
        return

    if args.watch:
        try:
            while True:
                os.system("clear")
                print(render_dashboard(ws, args.topic))
                time.sleep(2.0)
        except KeyboardInterrupt:
            return
    else:
        print(render_dashboard(ws, args.topic))


if __name__ == "__main__":
    main()
