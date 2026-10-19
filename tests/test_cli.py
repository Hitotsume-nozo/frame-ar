"""Unit and Integration Tests for FRAME-AR CLI & Antigravity Swarm Bridge."""

import argparse
import json
import pytest
import torch
import torch.nn as nn

from frame.cli import (
    cmd_status,
    cmd_verify_math,
    cmd_red_team,
    cmd_inspect_metric,
    cmd_swarm,
    cmd_run,
    cmd_kaggle,
    build_parser,
)
from frame.brain.state import SearchStage
from frame.agents.agy_bridge import (
    LaneType,
    SubagentSpec,
    SwarmProposal,
    AgySwarmCoordinator,
    build_agy_subagent_specs,
)


def test_cli_parser_build():
    """Validates parser structure and argument definitions."""
    parser = build_parser()
    assert parser.prog == "frame"
    
    # Test verify-math args
    args_math = parser.parse_args(["verify-math", "--expr", "x / (1 + torch.exp(-x))", "--level2"])
    assert args_math.expr == "x / (1 + torch.exp(-x))"
    assert args_math.level2 is True

    # Test swarm args
    args_swarm = parser.parse_args(["swarm", "--goal", "Optimize attention", "--export-agy-specs"])
    assert args_swarm.goal == "Optimize attention"
    assert args_swarm.export_agy_specs is True


def test_cmd_status_execution(capsys):
    """Asserts cmd_status displays LORD wealth and all stage metrics without error."""
    parser = build_parser()
    args = parser.parse_args(["status"])
    exit_code = cmd_status(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "FRAME-AR Autonomous Research Engine: System Status" in captured
    assert "LORD 3 Streaming FDR Target Alpha: 0.05" in captured
    assert "idea_proposal" in captured


def test_cmd_verify_math_l0_l1(capsys):
    """Asserts cmd_verify_math succeeds on valid SwiGLU activation for Level 0 and Level 1."""
    parser = build_parser()
    args = parser.parse_args(["verify-math", "--expr", "x / (1 + torch.exp(-x))"])
    exit_code = cmd_verify_math(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Level 0 (SymPy Analytical) -> Passed: True" in captured
    assert "Level 1 (Interval Bound Prop) -> Certified Safe: True" in captured
    assert "All requested verification levels cleared successfully." in captured


def test_cmd_verify_math_l2_ibnb(capsys):
    """Asserts cmd_verify_math Level 2 Interval Branch-and-Bound certifies bounded activation."""
    parser = build_parser()
    args = parser.parse_args(["verify-math", "--expr", "x / (1 + torch.exp(-x))", "--level2"])
    exit_code = cmd_verify_math(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Level 2 (IBnB Formal Proof) -> Certified: True" in captured


def test_cmd_verify_math_rejects_pole(capsys):
    """Asserts cmd_verify_math properly rejects singularity expression 1 / x."""
    parser = build_parser()
    args = parser.parse_args(["verify-math", "--expr", "1.0 / x"])
    exit_code = cmd_verify_math(args)
    assert exit_code == 1

    captured = capsys.readouterr().out
    assert "Level 0 (SymPy Analytical) -> Passed: False" in captured
    assert "[!] Level 0 Rejection:" in captured


def test_cmd_inspect_metric(capsys):
    """Asserts cmd_inspect_metric renders contract and ANSI terminal heatmap."""
    parser = build_parser()
    args = parser.parse_args(["inspect-metric", "--name", "cross_entropy_loss"])
    exit_code = cmd_inspect_metric(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Metric Contract: cross_entropy_loss" in captured
    assert "Incentive Surface ANSI Heatmap" in captured


def test_cmd_red_team_passes_trained_candidate(capsys):
    """Asserts cmd_red_team evaluates candidate against 5 null models and passes."""
    parser = build_parser()
    args = parser.parse_args(["red-team"])
    exit_code = cmd_red_team(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "Verdict: PASSED_RED_TEAM" in captured
    assert "[+] Red-Team Audit PASSED." in captured


def test_cmd_swarm_standard_and_export(capsys):
    """Asserts cmd_swarm executes multi-lane cycle and supports export-agy-specs."""
    parser = build_parser()
    
    # 1. Standard swarm execution
    args = parser.parse_args(["swarm", "--goal", "Optimize attention layer efficiency"])
    exit_code = cmd_swarm(args)
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Swarm Phase 1: Lane Proposal & Structural AST Audit" in captured
    assert "Swarm Phase 2: Compositional Recombination Audit" in captured
    assert "Swarm Phase 3: Two-Stage Group Sequential Screening" in captured
    assert "Confirmed Discovery: True" in captured

    # 2. Export agy specs
    args_export = parser.parse_args(["swarm", "--export-agy-specs"])
    exit_code_export = cmd_swarm(args_export)
    assert exit_code_export == 0
    captured_export = capsys.readouterr().out
    specs = json.loads(captured_export)
    assert isinstance(specs, list)
    assert len(specs) == 8
    types = [s["TypeName"] for s in specs]
    assert "frame_information_governor" in types
    assert "frame_literature_proximity_analyzer" in types
    assert "frame_meta_reviewer" in types
    assert "frame_executioner" in types
    assert "frame_lane_architecture" in types
    assert "frame_lane_objectives" in types
    assert "frame_lane_optimization" in types
    assert "frame_lane_features" in types


def test_cmd_run(capsys):
    """Asserts cmd_run end-to-end campaign execution."""
    parser = build_parser()
    args = parser.parse_args(["run", "--goal", "Optimize transformer feedforward representation"])
    exit_code = cmd_run(args)
    assert exit_code == 0

    captured = capsys.readouterr().out
    assert "FRAME-AR Autonomous Research Campaign" in captured
    assert "CAMPAIGN COMPLETED SUMMARY" in captured


def test_build_agy_subagent_specs():
    """Asserts subagent specifications match invoke_subagent schema."""
    # 1. Default (backward-compatible) returns 4 lane specialists
    specs = build_agy_subagent_specs("Optimize ViT patch embeddings")
    assert len(specs) == 4
    for s in specs:
        assert isinstance(s, SubagentSpec)
        assert s.Model == "inherit"
        assert s.Workspace == "inherit"
        assert len(s.Prompt) > 50

    # 2. Cognitive swarm roles (all 8 roles when requested)
    specs_8 = build_agy_subagent_specs("Optimize ViT patch embeddings", include_cognitive_roles=True)
    assert len(specs_8) == 8
    types_8 = [s.TypeName for s in specs_8]
    assert "frame_information_governor" in types_8
    assert "frame_literature_proximity_analyzer" in types_8
    assert "frame_meta_reviewer" in types_8
    assert "frame_executioner" in types_8
    assert "frame_lane_architecture" in types_8
    assert "frame_lane_objectives" in types_8
    assert "frame_lane_optimization" in types_8
    assert "frame_lane_features" in types_8


def test_agy_swarm_coordinator_full_cycle():
    """Asserts AgySwarmCoordinator proposal registration, falsification memory, and two-stage screening."""
    coordinator = AgySwarmCoordinator(goal="Benchmark Transformer Layers")
    
    # 1. Valid proposal registration
    p1 = SwarmProposal(
        hypothesis_id="hyp_001",
        lane=LaneType.ARCHITECTURE,
        title="SwiGLU Layer",
        description="Gated feedforward",
        code_snippet="class SwiGLU(nn.Module): pass",
        math_expression="x / (1 + exp(-x))",
    )
    ok, err = coordinator.register_proposal(p1)
    assert ok is True
    assert err is None

    # 2. Duplicate AST blocked by Falsification Memory if recorded
    coordinator.falsification_memory.record_fatal_failure(
        code="class BadAttention(nn.Module): pass",
        stage=SearchStage.POT_VERIFICATION,
        reason="Non-invertible gradient",
    )
    p2 = SwarmProposal(
        hypothesis_id="hyp_002",
        lane=LaneType.ARCHITECTURE,
        title="Bad Attention",
        description="Fails",
        code_snippet="class BadAttention(nn.Module): pass",
    )
    ok2, err2 = coordinator.register_proposal(p2)
    assert ok2 is False
    assert "Duplicate AST blocked by Falsification Memory" in err2

    # 3. Swarm synthesis cycle
    cycle_res = coordinator.run_swarm_synthesis_cycle()
    assert cycle_res["goal"] == "Benchmark Transformer Layers"
    assert cycle_res["recombination"]["is_legal"] is True
    assert cycle_res["evaluation"]["passed"] is True
    assert cycle_res["evaluation"]["status"] == "CONFIRMED_CANDIDATE"
    assert len(coordinator.confirmed_discoveries) == 1


def test_cmd_kaggle_diagnostics(capsys):
    """Asserts that frame kaggle displays credentials and quota ledger status."""
    parser = build_parser()
    args = parser.parse_args(["kaggle"])
    exit_code = cmd_kaggle(args)
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "FRAME-AR Tier 2: Kaggle Remote Engine Diagnostics" in out
    assert "Rolling 7-Day GPU Quota Ledger" in out


def test_cmd_kaggle_bundle(capsys, tmp_path):
    """Asserts that frame kaggle --bundle writes a self-extracting payload file."""
    parser = build_parser()
    bundle_dest = str(tmp_path / "payload.py")
    args = parser.parse_args(["kaggle", "--bundle", bundle_dest])
    exit_code = cmd_kaggle(args)
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "Self-extracting FRAME-AR payload bundled to:" in out
    assert (tmp_path / "payload.py").is_file()
    assert (tmp_path / "payload.py").stat().st_size > 10000


def test_cmd_kaggle_online_and_list_mocked(capsys):
    """Asserts that frame kaggle --check-online and --list interact with client."""
    from unittest import mock
    parser = build_parser()
    args = parser.parse_args(["kaggle", "--check-online", "--list"])

    with mock.patch("frame.runners.kaggle.client.KaggleClient.check_connection") as mock_conn, \
         mock.patch("frame.runners.kaggle.client.KaggleClient.list_kernels") as mock_list:
        mock_conn.return_value = {"authenticated": True, "username": "mock_user"}
        mock_list.return_value = [{"ref": "mock_user/exp-1", "title": "Mock Experiment", "lastRunTime": "2026-09-30"}]

        exit_code = cmd_kaggle(args)
        assert exit_code == 0

        out = capsys.readouterr().out
        assert "Kaggle REST Authentication Accepted" in out
        assert "mock_user/exp-1" in out

