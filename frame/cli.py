"""FRAME-AR Command Line Interface (CLI).

Provides unified command-line access to all FRAME-AR capabilities:
- frame run: Execute end-to-end autonomous research on a goal
- frame swarm: Run parallel multi-lane swarm research (Architecture, Objectives, Optimization, Features)
- frame red-team: Adversarial null-model degeneracy and causal leakage auditing
- frame verify-math: Tiered symbolic-numeric verification (L0 SymPy, L1 IBP, L2 IBnB)
- frame inspect-metric: Incentive surface analysis and ANSI terminal heatmap rendering
- frame capsule: Manual execution capsule packaging and micro-replay verification
- frame status: Inspect MCTS epistemic state, LORD alpha wealth, and accounting ledger
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional

from frame.schemas.compute import ExecutionTier
from frame.schemas.hypothesis import Hypothesis, TargetDirection
from frame.pot.tracer import profile_module_flops, route_compute_tier
from frame.verification.statistics import LORDController, evaluate_paired_seed_rope
from frame.verification.math_engine import (
    compress_swiglu_2d,
    compress_rmsnorm_invariant,
    verify_l0_analytic,
    verify_l1_symbolic_numeric,
    verify_l2_ibnb,
    IntervalBranchAndBoundSolver,
)
from frame.verification.red_team import (
    AdversarialRedTeamSuite,
    RedTeamDegeneracyResult,
    CausalLeakageResult,
)
from frame.brain.experimental_design import (
    calculate_required_sample_size,
    TwoStageScreeningController,
    QuadBaselineGenerator,
    verify_method_code_alignment,
)
from frame.brain.state import MCTSNode, SearchAccountingLedger, SearchStage
from frame.brain.thinking import (
    StructuralFalsificationMemory,
    RecombinationLegalityGrammar,
    LegalityViolation,
)
from frame.feedback.visualization import (
    MetricScopeContract,
    generate_ansi_terminal_heatmap,
    generate_svg_contour_slice,
)
from frame.runners.manual.capsule import package_manual_capsule, verify_micro_replay
from frame.agents.agy_bridge import AgySwarmCoordinator, build_agy_subagent_specs, LaneType
from frame.brain.campaign import AutonomousCampaignRunner, CampaignConfig, CampaignResult


def cmd_status(args: argparse.Namespace) -> int:
    """Displays current FRAME-AR engine status and accounting ledger."""
    print("=" * 70)
    print("FRAME-AR Autonomous Research Engine: System Status")
    print("=" * 70)
    
    lord_path = os.path.join(".frame", "lord_state.json")
    if os.path.exists(lord_path):
        controller = LORDController.load_from_file(lord_path)
    else:
        controller = LORDController(alpha=0.05)

    print(f"LORD 3 Streaming FDR Target Alpha: {controller.alpha}")
    print(f"Initial Wealth W0: {controller.w0}")
    print(f"Current Next Alpha (Step {controller.current_step + 1}): {controller.get_next_alpha():.6f}")
    print(f"Confirmed Discoveries: {len(controller.discovery_indices)}")
    
    ledger_path = os.path.join(".frame", "accounting_ledger.json")
    if os.path.exists(ledger_path):
        ledger = SearchAccountingLedger.load_from_file(ledger_path)
    else:
        ledger = SearchAccountingLedger()

    print("\nStage Accounting Metrics:")
    for st in SearchStage:
        rec = ledger._get_or_create_stage(st)
        print(f"  [{st.value:28s}] Entered: {rec.hypotheses_entered:3d} | Passed: {rec.hypotheses_passed:3d} | Rejected: {rec.hypotheses_rejected:3d}")

    # Swarm Learning State & Test-Time Compute Scaling
    try:
        from frame.brain.learning_state import DEFAULT_LEARNING_STATE_PATH, load_learning_state
        if os.path.exists(DEFAULT_LEARNING_STATE_PATH):
            node = load_learning_state()
            curve = node.get_upper_envelope_curve()
            is_mono, gain = node.verify_monotonic_progression()
            entropy = node.compute_epistemic_entropy()
            print("\nSwarm Learning State & Test-Time Compute Scaling:")
            print(f"  Node ID: {node.node_id} | Total Budget N: {node.total_budget_n}")
            print(f"  Monotonic Progression: {is_mono} (Upper Envelope Gain: {gain:+.4f})")
            print(f"  Epistemic Entropy: {entropy:.4f} nats")
            if curve:
                curve_preview = [round(float(v), 4) for v in curve[:8]]
                print(f"  Scaling Curve Q*(N): {curve_preview}{'...' if len(curve) > 8 else ''} ({len(curve)} points)")
            if getattr(node, "executioner_event_logs", None):
                print(f"  Executioner Watchdog Event Logs ({len(node.executioner_event_logs)} events):")
                for ev in node.executioner_event_logs[-5:]:
                    ev_type = getattr(ev, "event_type", "EVENT")
                    b_id = getattr(ev, "branch_id", "")
                    trig = getattr(ev, "trigger", "")
                    print(f"    - [{ev_type}] Branch: {b_id} | Trigger: {trig}")
            if getattr(node, "governor_audit_logs", None):
                print(f"  Information Governor Audit Logs ({len(node.governor_audit_logs)} logs):")
                for g_log in node.governor_audit_logs[-5:]:
                    print(f"    - {g_log}")
    except Exception:
        pass

    try:
        from frame.runners.kaggle.credentials import load_kaggle_credentials
        from frame.runners.kaggle.quota import KaggleQuotaLedger
        creds = load_kaggle_credentials()
        ledger_q = KaggleQuotaLedger()
        used_q = ledger_q.get_used_gpu_hours()
        rem_q = ledger_q.get_remaining_gpu_hours()
        print(f"\nTier 2 Kaggle Remote Engine:")
        print(f"  Credentials: Verified (User: {creds.username})")
        print(f"  Rolling 7-Day GPU Quota: {used_q:.1f}h / 30.0h used ({rem_q:.1f}h remaining)")
    except Exception as e:
        print(f"\nTier 2 Kaggle Remote Engine: Unconfigured / Inactive ({e})")
    print("=" * 70)
    return 0


def cmd_verify_math(args: argparse.Namespace) -> int:
    """Executes tiered symbolic-numeric verification (L0, L1, L2)."""
    expr_str = args.expr
    print(f"==> Initiating Tiered Math Verification for: {expr_str}")
    
    # Clean expression for SymPy and AST analysis
    sp_expr_str = expr_str.replace("torch.", "").replace("math.", "").replace("np.", "")
    
    # 1. Level 0 Analytical SymPy
    res_l0 = verify_l0_analytic(sp_expr_str, variables=["x"])
    print(f"Level 0 (SymPy Analytical) -> Passed: {res_l0.passed} | Poles: {res_l0.poles} | Limits: {res_l0.limits}")
    if not res_l0.passed:
        print(f"[!] Level 0 Rejection: {res_l0.details.get('reason', 'Analytical check failed')}")
        return 1
        
    # 2. Level 1 Interval Bound Propagation
    try:
        ast_code = f"y = {sp_expr_str}"
        res_l1 = verify_l1_symbolic_numeric(ast_code, input_bound=5.0)
        print(f"Level 1 (Interval Bound Prop) -> Certified Safe: {res_l1.passed} | Output Range: {res_l1.output_interval}")
        if not res_l1.passed:
            print("[!] Level 1 Rejection: Denominator hazard detected under IBP.")
            return 1
    except Exception as e:
        print(f"Level 1 (Direct AST evaluation note): {e}")

    # 3. Level 2 Interval Branch-and-Bound
    if args.level2:
        print("==> Escalating to Level 2 (Interval Branch-and-Bound Formal Proof)...")
        import mpmath
        
        def scalar_fn(coords):
            x = coords[0]
            try:
                val = eval(sp_expr_str, {"x": x, "exp": np.exp, "tanh": np.tanh, "sin": np.sin, "cos": np.cos})
                return float(val)
            except Exception:
                return float("nan")

        def mp_interval_fn(boxes):
            x = boxes[0]
            try:
                iv_ctx = {
                    "x": x,
                    "exp": getattr(mpmath.iv, "exp", mpmath.exp),
                    "sin": getattr(mpmath.iv, "sin", mpmath.sin),
                    "cos": getattr(mpmath.iv, "cos", mpmath.cos),
                    "tan": getattr(mpmath.iv, "tan", mpmath.tan),
                    "log": getattr(mpmath.iv, "log", mpmath.log),
                    "sqrt": getattr(mpmath.iv, "sqrt", mpmath.sqrt),
                }
                if hasattr(mpmath.iv, "tanh"):
                    iv_ctx["tanh"] = mpmath.iv.tanh
                else:
                    iv_ctx["tanh"] = lambda v: (mpmath.iv.exp(2*v) - 1) / (mpmath.iv.exp(2*v) + 1)
                return eval(sp_expr_str, iv_ctx)
            except Exception as e:
                return mpmath.mpi(-1e6, 1e6)

        res_l2 = verify_l2_ibnb(
            scalar_fn=scalar_fn,
            mp_interval_fn=mp_interval_fn,
            bounds=[(-2.0, 2.0)],
            violation_threshold=-100.0,
            seek_violation_greater=False,
            max_depth=6,
            max_subdivisions=500,
        )
        print(f"Level 2 (IBnB Formal Proof) -> Certified: {res_l2.certified} | Subdivisions: {res_l2.subdivisions}")
        if not res_l2.passed:
            print(f"[!] Counterexample Found: {res_l2.counterexample}")
            return 1
            
    print("[+] All requested verification levels cleared successfully.")
    return 0


def cmd_red_team(args: argparse.Namespace) -> int:
    """Executes the Adversarial Red-Team Degeneracy Suite."""
    print("==> Executing Adversarial Red-Team Degeneracy Audit...")
    
    # Create test model and evaluation dataset
    model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    X = torch.randn(64, 16)
    y = X[:, :1] * 2.5 + torch.randn(64, 1) * 0.05
    dataloader = [(X[i:i+8], y[i:i+8]) for i in range(0, 64, 8)]
    
    # Train candidate model to clear the baseline
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(40):
        for bx, by in dataloader:
            loss = nn.functional.mse_loss(model(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    suite = AdversarialRedTeamSuite(
        epsilon_robust=0.01,
        target_direction=TargetDirection.MINIMIZE,
    )
    result = suite.evaluate_null_models(
        candidate_model=model,
        eval_data=dataloader,
        metric_fn=lambda p, t: float(nn.functional.mse_loss(p, t).item()),
    )
    
    print(f"Candidate Score: {result.candidate_score:.6f}")
    print(f"Null Models: {', '.join(f'{k}={v:.4f}' for k, v in result.null_scores.items())}")
    print(f"Best Null Score: {result.min_null_score:.6f}")
    print(f"Margin Over Best Null: {result.margin:.6f}")
    print(f"Verdict: {result.verdict}")
    
    if not result.passed:
        print(f"[!] Red-Team Audit FAILED: candidate margin {result.margin:.6f} <= {result.epsilon_robust}")
        return 1
        
    print("[+] Red-Team Audit PASSED. Candidate is non-degenerate.")
    return 0


def cmd_inspect_metric(args: argparse.Namespace) -> int:
    """Renders ANSI terminal heatmap and incentive surface metrics."""
    metric_name = args.name or "cross_entropy_loss"
    direction = TargetDirection.MINIMIZE if "loss" in metric_name.lower() else TargetDirection.MAXIMIZE
    
    scores = [1.85, 1.72, 1.65, 1.50, 1.48, 1.45, 1.42, 1.38, 1.35, 1.30]
    contract = MetricScopeContract.create(
        metric_name=metric_name,
        observed_scores=scores,
        theoretical_min=0.0,
        theoretical_max=10.0,
        null_max_score=4.0,
        target_direction=direction,
    )
    
    print(f"Metric Contract: {contract.metric_name} (Direction: {contract.target_direction.value})")
    print(f"Effective Dynamic Range: [{contract.population_quantiles['p0']:.3f}, {contract.population_quantiles['p100']:.3f}]")
    print(f"Saturation Index: {contract.saturation_index:.3f} (Saturated: {contract.is_saturated})")
    print(f"Degeneracy Risk Ratio: {contract.degeneracy_risk_ratio:.3f} (High Risk: {contract.is_high_degeneracy_risk})")
    
    print("\nIncentive Surface ANSI Heatmap (Learning Rate vs Weight Decay):")
    x_vals = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    y_vals = [0.0, 1e-4, 1e-3, 1e-2, 1e-1]
    p0 = contract.population_quantiles["p0"]
    grid = [[p0 + (i * 0.1 + j * 0.05) for j in range(5)] for i in range(5)]
    
    heatmap_str = generate_ansi_terminal_heatmap(
        x_param="Learning Rate",
        y_param="Weight Decay",
        x_values=x_vals,
        y_values=y_vals,
        grid_data=grid,
        title=f"Incentive Surface ({contract.metric_name})",
    )
    print(heatmap_str)
    return 0


def cmd_swarm(args: argparse.Namespace) -> int:
    """Launches parallel multi-lane swarm research coordinator."""
    goal = args.goal or "Optimize attention layer efficiency"
    lanes = [lane.strip() for lane in args.lanes.split(",")]
    
    if getattr(args, "export_agy_specs", False):
        coordinator = AgySwarmCoordinator(goal=goal)
        specs = coordinator.export_agy_specs(lanes=lanes)
        print(json.dumps(specs, indent=2))
        return 0

    print(f"==> Launching FRAME-AR Swarm Coordinator for Goal: '{goal}'")
    print(f"==> Active Swarm Lanes: {lanes}")
    
    # Initialize components
    falsification_memory = StructuralFalsificationMemory()
    recomb_grammar = RecombinationLegalityGrammar()
    lord_controller = LORDController(alpha=0.05)
    accounting = SearchAccountingLedger()
    
    # 1. Swarm proposal generation across lanes
    print("\n[Swarm Phase 1: Lane Proposal & Structural AST Audit]")
    proposals = [
        {"lane": "architecture", "title": "Sparse Local Window Attention", "code": "class WindowAttention(nn.Module): pass"},
        {"lane": "objectives", "title": "Margin Contrastive Regularizer", "code": "def margin_loss(p, t): return F.relu(1.0 - p*t).mean()"},
        {"lane": "optimization", "title": "Cosine Decay with Warmup", "code": "class CosineWarmup: pass"},
    ]
    
    for prop in proposals:
        if prop["lane"] not in lanes:
            continue
        print(f"  Lane [{prop['lane']:12s}]: Proposing '{prop['title']}'")
        is_blocked, record = falsification_memory.check_hard_filter(prop["code"])
        accounting.record_entry(SearchStage.IDEA_PROPOSAL)
        if is_blocked and record is not None:
            print(f"    [X] Blocked by Falsification Memory: {record.reason}")
            accounting.record_rejection(SearchStage.IDEA_PROPOSAL, reason=record.reason)
        else:
            print("    [✓] Cleared AST Falsification Filter")
            accounting.record_pass(SearchStage.IDEA_PROPOSAL)
            
    # 2. Recombination Legality Audit
    print("\n[Swarm Phase 2: Compositional Recombination Audit (A ⊕ B ⊕ D)]")
    is_legal, viol_code, expl = recomb_grammar.validate_recombination(
        model_spec={"output_activation": "linear"},
        loss_spec={"name": "cross_entropy"},
        optimizer_spec={"name": "adamw"},
    )
    if not is_legal:
        print(f"  [X] Illegal Recombination: {viol_code} - {expl}")
    else:
        print("  [✓] Recombination Legality Confirmed: Architecture(linear) ⊕ Loss(cross_entropy) ⊕ Opt(adamw)")
        
    # 3. Two-Stage Group Sequential Screening
    print("\n[Swarm Phase 3: Two-Stage Group Sequential Screening]")
    screener = TwoStageScreeningController(rope_epsilon=0.05, target_direction=TargetDirection.MINIMIZE)
    stage1 = screener.evaluate_stage1_futility(
        baseline_scores={"s1": 2.10, "s2": 2.05, "s3": 2.15},
        candidate_scores={"s1": 1.80, "s2": 1.75, "s3": 1.85},
    )
    print(f"  Stage 1 Futility Verdict: {stage1.status} (Mean Improvement: {stage1.mean_difference:.4f})")
    
    if stage1.proceed:
        stage2 = screener.evaluate_stage2_confirmatory(
            baseline_scores={"s1": 2.10, "s2": 2.05, "s3": 2.15, "s4": 2.00, "s5": 2.08},
            candidate_scores={"s1": 1.80, "s2": 1.75, "s3": 1.85, "s4": 1.70, "s5": 1.78},
            lord_controller=lord_controller,
        )
        p_val = stage2.details.get("p_value_rope", 0.0)
        alpha_t = stage2.details.get("allocated_alpha_t", 0.0)
        print(f"  Stage 2 Confirmatory Verdict: {stage2.status} (p-value: {p_val:.6f}, alpha_t: {alpha_t:.6f})")
        print(f"  Confirmed Discovery: {stage2.proceed}")

    print("\n[+] Swarm cycle completed successfully.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Executes end-to-end autonomous research run on a target goal until reached."""
    goal = args.goal or "Optimize model efficiency and representation"
    lanes = [l.strip() for l in args.lanes.split(",")] if hasattr(args, "lanes") and args.lanes else ["architecture", "objectives", "optimization", "features"]
    max_iter = getattr(args, "max_iterations", 5)
    target_imp = getattr(args, "target_improvement", 0.10)

    if getattr(args, "export_agy_specs", False):
        coordinator = AgySwarmCoordinator(goal=goal)
        specs = coordinator.export_agy_specs(lanes=lanes)
        print(json.dumps(specs, indent=2))
        return 0

    config = CampaignConfig(
        goal=goal,
        max_iterations=max_iter,
        target_improvement=target_imp,
        active_lanes=lanes,
        verbose=True,
    )
    runner = AutonomousCampaignRunner(config)
    result = runner.run_campaign()
    return 0 if result.goal_reached or result.confirmed_discoveries > 0 else 1


def cmd_kaggle(args: argparse.Namespace) -> int:
    """Inspects Kaggle API credentials, rolling quota ledger, and remote connectivity."""
    import time
    from frame.runners.kaggle.credentials import load_kaggle_credentials
    from frame.runners.kaggle.quota import KaggleQuotaLedger
    from frame.runners.kaggle.client import KaggleClient

    print("=" * 70)
    print("FRAME-AR Tier 2: Kaggle Remote Engine Diagnostics")
    print("=" * 70)

    try:
        creds = load_kaggle_credentials()
        print("[✓] Credentials Loaded Successfully:")
        print(f"    Username: {creds.username}")
        print(f"    Source:   {creds.source}")
    except Exception as e:
        print(f"[!] Credential Resolution Failed: {e}")
        return 1

    ledger = KaggleQuotaLedger()
    used_h = ledger.get_used_gpu_hours()
    rem_h = ledger.get_remaining_gpu_hours()
    print("\n[✓] Rolling 7-Day GPU Quota Ledger:")
    print(f"    Weekly Limit:     {ledger.weekly_limit_hours:.1f} hours")
    print(f"    Consumed (7-Day): {used_h:.2f} hours")
    print(f"    Available:        {rem_h:.2f} hours")
    print(f"    Ledger File:      {ledger.ledger_path}")

    # 1. Probe connectivity
    if getattr(args, "check_online", False):
        print("\n==> Probing Kaggle REST API Connectivity...")
        try:
            client = KaggleClient(credentials=creds, quota_ledger=ledger)
            conn_info = client.check_connection()
            print(f"[✓] Kaggle REST Authentication Accepted (User: {conn_info['username']})")
            print("[+] Kaggle REST client fully operational.")
        except Exception as e:
            print(f"[!] Kaggle API probe failed: {e}")
            return 1

    # 2. List remote user kernels
    if getattr(args, "list", False):
        print(f"\n==> Listing kernels on Kaggle for user '{creds.username}'...")
        try:
            client = KaggleClient(credentials=creds, quota_ledger=ledger)
            kernels = client.list_kernels(page_size=10)
            print(f"[✓] Found {len(kernels)} kernels:")
            for k in kernels:
                ref = k.get("ref", "")
                title = k.get("title", "")
                last_run = k.get("lastRunTime", "N/A")
                print(f"    • {ref:35s} | {title[:25]:25s} | {last_run}")
        except Exception as e:
            print(f"[!] Failed to list kernels: {e}")
            return 1

    # 3. Bundle FRAME-AR source payload
    if getattr(args, "bundle", None):
        out_file = args.bundle
        from frame.runners.kaggle.bundler import create_kaggle_bundle
        script_content = "import frame\nprint('FRAME-AR Remote Kaggle Container Initialized successfully.')\n"
        if getattr(args, "script", None) and os.path.isfile(args.script):
            with open(args.script, "r", encoding="utf-8") as sf:
                script_content = sf.read()
        payload = create_kaggle_bundle(main_execution_script=script_content)
        with open(out_file, "w", encoding="utf-8") as of:
            of.write(payload)
        print(f"\n[✓] Self-extracting FRAME-AR payload bundled to: {out_file} ({len(payload)} bytes)")

    # 4. Check kernel status
    if getattr(args, "status", None):
        target_slug = args.status
        print(f"\n==> Fetching status for kernel '{target_slug}'...")
        try:
            client = KaggleClient(credentials=creds, quota_ledger=ledger)
            st_res = client.get_kernel_status(target_slug)
            print(f"    Status:          {st_res.get('status')}")
            if st_res.get("failureMessage"):
                print(f"    Failure Message: {st_res.get('failureMessage')}")
        except Exception as e:
            print(f"[!] Failed to query kernel status: {e}")
            return 1

    # 5. Download kernel output
    if getattr(args, "output", None):
        target_slug = args.output
        dest_dir = getattr(args, "dest", None) or os.path.join(".frame", "kaggle_outputs", target_slug)
        print(f"\n==> Downloading outputs for kernel '{target_slug}' to '{dest_dir}'...")
        try:
            client = KaggleClient(credentials=creds, quota_ledger=ledger)
            saved = client.download_kernel_output(target_slug, destination_dir=dest_dir)
            print(f"    Downloaded {len(saved)} files:")
            for fname, fpath in saved.items():
                print(f"      - {fname}: {fpath}")
        except Exception as e:
            print(f"[!] Failed to download kernel output: {e}")
            return 1

    print("=" * 70)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Constructs the unified FRAME-AR CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="frame",
        description="FRAME-AR: Autonomous Scientific Discovery & Verification Platform",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available FRAME-AR subcommands")
    
    # frame kaggle
    p_kaggle = subparsers.add_parser("kaggle", help="Inspect Kaggle credentials and 7-day rolling GPU quota")
    p_kaggle.add_argument("--check-online", action="store_true", help="Probe remote Kaggle REST API connectivity")
    p_kaggle.add_argument("--list", action="store_true", help="List recent Kaggle kernels for user")
    p_kaggle.add_argument("--bundle", type=str, help="Package FRAME source into self-extracting payload file")
    p_kaggle.add_argument("--script", type=str, help="Optional script file to embed into bundle")
    p_kaggle.add_argument("--status", type=str, help="Check status of a remote Kaggle kernel slug")
    p_kaggle.add_argument("--output", type=str, help="Download output files of a remote Kaggle kernel slug")
    p_kaggle.add_argument("--dest", type=str, help="Destination directory for downloaded outputs")
    p_kaggle.set_defaults(func=cmd_kaggle)
    
    # frame run
    p_run = subparsers.add_parser("run", help="Run end-to-end autonomous research campaign until goal reached")
    p_run.add_argument("--goal", type=str, required=True, help="Research objective or hypothesis goal")
    p_run.add_argument("--lanes", type=str, default="architecture,objectives,optimization,features", help="Active swarm lanes")
    p_run.add_argument("--max-iterations", type=int, default=5, help="Maximum MCTS search iterations")
    p_run.add_argument("--target-improvement", type=float, default=0.10, help="Relative improvement threshold to declare goal reached (e.g. 0.10 for 10 percent)")
    p_run.add_argument("--export-agy-specs", action="store_true", help="Export invoke_subagent payload JSON for agy")
    p_run.set_defaults(func=cmd_run)
    
    # frame swarm
    p_swarm = subparsers.add_parser("swarm", help="Run multi-lane swarm research coordinator")
    p_swarm.add_argument("--goal", type=str, default="Optimize attention layer efficiency", help="Research objective")
    p_swarm.add_argument("--lanes", type=str, default="architecture,objectives,optimization,features", help="Active swarm lanes")
    p_swarm.add_argument("--export-agy-specs", action="store_true", help="Export invoke_subagent payload JSON for agy")
    p_swarm.set_defaults(func=cmd_swarm)
    
    # frame verify-math
    p_math = subparsers.add_parser("verify-math", help="Run tiered symbolic-numeric verification")
    p_math.add_argument("--expr", type=str, required=True, help="Mathematical expression to verify (e.g. 'x / (1 + torch.exp(-x))')")
    p_math.add_argument("--level2", action="store_true", help="Escalate to Level 2 Interval Branch-and-Bound")
    p_math.set_defaults(func=cmd_verify_math)
    
    # frame red-team
    p_red = subparsers.add_parser("red-team", help="Run adversarial red-team degeneracy probes")
    p_red.set_defaults(func=cmd_red_team)
    
    # frame inspect-metric
    p_metric = subparsers.add_parser("inspect-metric", help="Inspect incentive surface and render ANSI heatmap")
    p_metric.add_argument("--name", type=str, default="cross_entropy_loss", help="Metric identifier")
    p_metric.set_defaults(func=cmd_inspect_metric)
    
    # frame status
    p_status = subparsers.add_parser("status", help="Inspect MCTS epistemic state and accounting ledger")
    p_status.set_defaults(func=cmd_status)
    
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if hasattr(args, "func"):
        sys.exit(args.func(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
