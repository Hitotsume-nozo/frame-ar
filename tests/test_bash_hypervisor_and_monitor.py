"""Verification tests for FRAME-AR v2.1: Static Bash Hypervisor & Out-of-Band Monitor Upgrade."""

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import pytest

from frame.bench.registry import approve_benchmark, init_benchmark
from frame.bench.verifier import verify_benchmark
from frame.cli import main as cli_main
from frame.eval.funnel import FunnelEvaluator
from frame.eval.statistics import LORDController, ZETA_1_6, gamma_lord


def test_section1_riemann_zeta_lord3_decay():
    """Verifies Riemann Zeta (s=1.6) LORD-3 decay schedule and step 1 alpha."""
    assert abs(ZETA_1_6 - 2.28576595) < 1e-7
    assert gamma_lord(0) == 0.0
    assert gamma_lord(-5) == 0.0

    gamma_1 = gamma_lord(1)
    assert abs(gamma_1 - (1.0 / 2.28576595)) < 1e-7

    lord = LORDController(alpha=0.05, w0_ratio=0.5)
    assert lord.current_step == 0
    alpha_1 = lord.peek_next_alpha()
    # Step 1 alpha = w0 * gamma(1) = 0.025 / 2.28576595 = 0.010937256...
    assert abs(alpha_1 - 0.01093725) < 1e-5

    # Check peek does not mutate state
    assert lord.current_step == 0
    alpha_step, is_disc, step_num = lord.step(0.001)
    assert abs(alpha_step - alpha_1) < 1e-7
    assert is_disc is True
    assert step_num == 1
    assert lord.current_step == 1


def test_section2_cli_subcommands(tmp_path: Path):
    """Verifies `bench check-lock`, `prepare-jail`, and `smoke-check` CLI subcommands."""
    topic = "test_subcommands"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")

    # 1. bench check-lock before lock should fail (code 1)
    rc = cli_main(["--workspace", str(tmp_path), "bench", "check-lock", "--topic", topic])
    assert rc == 1

    # Verify and approve to create bench.lock
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="test_user")

    # Now bench check-lock should succeed (code 0)
    rc = cli_main(["--workspace", str(tmp_path), "bench", "check-lock", "--topic", topic])
    assert rc == 0

    # 2. prepare-jail
    jail_dir = tmp_path / "jail_test"
    rc = cli_main([
        "--workspace", str(tmp_path),
        "prepare-jail",
        "--topic", topic,
        "--jail-dir", str(jail_dir),
    ])
    assert rc == 0
    assert (jail_dir / "BRIEF.md").is_file()
    assert (jail_dir / "candidate.py").is_file()
    assert (jail_dir / "meta.env").is_file()
    meta_env = (jail_dir / "meta.env").read_text(encoding="utf-8")
    assert "PARENT_ID='node_0'" in meta_env
    assert "ROLE=" in meta_env

    # 3. smoke-check: AST duplicate check
    # candidate.py in jail is a direct copy of node_0, so it must trigger GATE0_AST_DUPLICATE (code 2)
    rc = cli_main([
        "--workspace", str(tmp_path),
        "smoke-check",
        "--topic", topic,
        "--candidate", str(jail_dir / "candidate.py"),
    ])
    assert rc == 2

    # candidate file not found (code 1)
    rc = cli_main([
        "--workspace", str(tmp_path),
        "smoke-check",
        "--topic", topic,
        "--candidate", str(jail_dir / "nonexistent.py"),
    ])
    assert rc == 1

    # candidate that crashes (code 3)
    crash_code = textwrap.dedent(
        """
        def run_experiment(task, seed, mode, artifact_dir):
            raise RuntimeError("Intentional crash")
        """
    )
    crash_cand = tmp_path / "crash_cand.py"
    crash_cand.write_text(crash_code, encoding="utf-8")
    rc = cli_main([
        "--workspace", str(tmp_path),
        "smoke-check",
        "--topic", topic,
        "--candidate", str(crash_cand),
    ])
    assert rc == 3

    # candidate with parametric modification that runs cleanly (code 0)
    good_code = textwrap.dedent(
        """
        import numpy as np

        def run_experiment(task, seed, mode, artifact_dir):
            W = np.array([-0.90, -0.80, 0.05, -0.02], dtype=np.float64)
            return {"policy_weights": W, "policy_bias": np.zeros(1)}
        """
    )
    good_cand = tmp_path / "good_cand.py"
    good_cand.write_text(good_code, encoding="utf-8")
    rc = cli_main([
        "--workspace", str(tmp_path),
        "smoke-check",
        "--topic", topic,
        "--candidate", str(good_cand),
    ])
    assert rc == 0


def test_section3_orchestrate_script_executable():
    """Verifies that frame-orchestrate.sh is executable and runs help/validation."""
    script_path = Path(__file__).resolve().parent.parent / "frame-orchestrate.sh"
    assert script_path.is_file()
    assert (script_path.stat().st_mode & 0o111) != 0

    # Test --help
    res = subprocess.run([str(script_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Usage: ./frame-orchestrate.sh" in res.stdout

    # Test missing --topic
    res_err = subprocess.run([str(script_path)], capture_output=True, text=True)
    assert res_err.returncode == 1
    assert "Error: --topic <slug> is required" in res_err.stderr


def test_section4_frame_mon_script(tmp_path: Path):
    """Verifies frame-mon.py CLI and lock-free injection."""
    script_path = Path(__file__).resolve().parent.parent / "frame-mon.py"
    assert script_path.is_file()
    assert (script_path.stat().st_mode & 0o111) != 0

    # Test --help
    res = subprocess.run([sys.executable, str(script_path), "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "--inject-parent" in res.stdout
    assert "--pause" in res.stdout

    topic = "mon_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="tester")
    FunnelEvaluator(tmp_path, topic)

    # Test pause signal
    res_pause = subprocess.run(
        [sys.executable, str(script_path), "--workspace", str(tmp_path), "--topic", topic, "--pause"],
        capture_output=True,
        text=True,
    )
    assert res_pause.returncode == 0
    assert "Pause signal written" in res_pause.stdout
    pause_file = tmp_path / ".frame" / "campaigns" / topic / "signals" / "pause"
    assert pause_file.is_file()

    # Test dashboard render
    res_dash = subprocess.run(
        [sys.executable, str(script_path), "--workspace", str(tmp_path), "--topic", topic],
        capture_output=True,
        text=True,
    )
    assert res_dash.returncode == 0
    assert "[PAUSED]" in res_dash.stdout
    assert "LORD-3 Step: 0" in res_dash.stdout
    assert "Next Alpha: 0.010937" in res_dash.stdout

    # Test hypothesis injection
    res_inj = subprocess.run(
        [
            sys.executable, str(script_path),
            "--workspace", str(tmp_path),
            "--topic", topic,
            "--inject-parent", "node_0",
            "--spec", "Test non-linear spring hypothesis",
        ],
        capture_output=True,
        text=True,
    )
    assert res_inj.returncode == 0
    assert "Injected Priority Directive" in res_inj.stdout

    # Test abort command
    res_abort = subprocess.run(
        [
            sys.executable, str(script_path),
            "--workspace", str(tmp_path),
            "--topic", topic,
            "--abort", "node_0",
            "--spec", "Replacement hypothesis after abort",
        ],
        capture_output=True,
        text=True,
    )
    assert res_abort.returncode == 0
    assert "aborted_node_id" in res_abort.stdout
    assert "node_0" in res_abort.stdout


def test_section3_orchestrate_end_to_end_with_gate0_retry(tmp_path: Path):
    """Verifies frame-orchestrate.sh full execution lifecycle including Gate 0 retry loop."""
    import os
    script_path = Path(__file__).resolve().parent.parent / "frame-orchestrate.sh"
    topic = "orch_e2e_retry_test"

    # Create mock agy worker that fails on attempt 1 (crash) and succeeds on retry 2
    mock_agy = tmp_path / "mock_agy.sh"
    state_file = tmp_path / "agy_invocations.txt"
    mock_script = (
        "#!/usr/bin/env bash\n"
        f"touch {state_file}\n"
        f"INVOCATIONS=$(wc -l < {state_file})\n"
        f"echo call >> {state_file}\n"
        "if [ \"$INVOCATIONS\" -eq 0 ]; then\n"
        "  cat << 'EOF' > candidate.py\n"
        "def run_experiment(task, seed, mode, artifact_dir):\n"
        "    raise RuntimeError('Gate 0 retry simulation')\n"
        "EOF\n"
        "else\n"
        "  cat << 'EOF' > candidate.py\n"
        "import numpy as np\n"
        "def run_experiment(task, seed, mode, artifact_dir):\n"
        "    W = np.array([-0.90, -0.80, 0.05, -0.02], dtype=np.float64)\n"
        "    return {'policy_weights': W, 'policy_bias': np.zeros(1)}\n"
        "EOF\n"
        "fi\n"
    )
    mock_agy.write_text(mock_script, encoding="utf-8")
    mock_agy.chmod(0o755)

    env = os.environ.copy()
    env["AGY_BIN"] = str(mock_agy)
    env["AGY_TIMEOUT"] = "30"

    root_dir = Path(__file__).resolve().parent.parent
    try:
        # Run frame-orchestrate.sh with 1 step, feeding '\ny\n' to complete Phase 0 manual audit loop
        proc = subprocess.Popen(
            [str(script_path), "--topic", topic, "--domain", "rl_control", "--steps", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(root_dir),
            env=env,
        )
        stdout, stderr = proc.communicate(input="\ny\n", timeout=60)
        assert proc.returncode == 0, f"Script failed with code {proc.returncode}:\n{stderr}\n{stdout}"
        assert "[Phase 0] Benchmark locked successfully." in stdout
        assert "Gate 0 Failed (Attempt 1 / 3)" in stdout
        assert "Gate 0 Smoke Check Passed (Attempt 2)" in stdout
        assert "[Phase 5] Running Final Untouched Vault-Holdout Audit..." in stdout
    finally:
        import shutil
        shutil.rmtree(root_dir / ".frame" / "benchmarks" / topic, ignore_errors=True)
        shutil.rmtree(root_dir / ".frame" / "campaigns" / topic, ignore_errors=True)


def test_section5_extract_code_unit_and_e2e(tmp_path: Path, capsys):
    """Verifies extract-code CLI subcommand across all extraction paths and error modes."""
    from frame.cli import main as cli_main

    cand = tmp_path / "candidate.py"
    backup = tmp_path / "parent_backup.py"
    stdout_log = tmp_path / "agy_stdout.log"

    # 1. In-place modified on disk
    backup.write_text("def run_experiment(): pass # v1", encoding="utf-8")
    cand.write_text("def run_experiment(): pass # v2 modified", encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "IN_PLACE_MODIFIED" in out

    # Reset candidate to match backup
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")

    # 2. Missing stdout log
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "NO_STDOUT_LOG" in err

    # 3. Empty stdout log
    stdout_log.write_text("   \n\n  ", encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "EMPTY_STDOUT_LOG" in err

    # 4. Fenced code block with def run_experiment
    fenced_content = textwrap.dedent("""
        Certainly! Here is the mutated controller:
        ```python
        import numpy as np

        def run_experiment(task, seed, mode, artifact_dir):
            return {"policy_weights": np.array([1.0, 2.0])}
        ```
        Let me know if you need any adjustments.
    """)
    stdout_log.write_text(fenced_content, encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXTRACTED_FENCED_BLOCK" in out
    assert "return {\"policy_weights\": np.array([1.0, 2.0])}" in cand.read_text(encoding="utf-8")

    # 5. Multiple fenced blocks (picks the last containing def run_experiment)
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    multi_fenced = textwrap.dedent("""
        First idea:
        ```python
        def helper(): pass
        ```
        Final solution:
        ```python
        def run_experiment(task, seed, mode, artifact_dir):
            return {"policy_weights": "final_choice"}
        ```
    """)
    stdout_log.write_text(multi_fenced, encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXTRACTED_FENCED_BLOCK" in out
    assert '"final_choice"' in cand.read_text(encoding="utf-8")

    # 5b. Fenced block with ```python3 tag
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    stdout_log.write_text("```python3\ndef run_experiment(task, seed, mode, artifact_dir):\n    return {'python3_block': True}\n```", encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXTRACTED_FENCED_BLOCK" in out
    assert "'python3_block': True" in cand.read_text(encoding="utf-8")

    # 5c. Fenced block with ``` python tag (space after backticks)
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    stdout_log.write_text("``` python\ndef run_experiment(task, seed, mode, artifact_dir):\n    return {'space_tag': True}\n```", encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXTRACTED_FENCED_BLOCK" in out
    assert "'space_tag': True" in cand.read_text(encoding="utf-8")

    # 6. Fallback raw python code without backticks
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    raw_python = textwrap.dedent("""
        # Leading comments
        import numpy as np
        from pathlib import Path

        def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:
            return {"raw_success": True}
    """)
    stdout_log.write_text(raw_python, encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "EXTRACTED_RAW_PYTHON" in out
    assert '"raw_success": True' in cand.read_text(encoding="utf-8")

    # 7. No valid python code found
    cand.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    stdout_log.write_text("Sorry, I am an AI and cannot process this request.", encoding="utf-8")
    rc = cli_main(["extract-code", "--candidate", str(cand), "--backup", str(backup), "--stdout-log", str(stdout_log)])
    assert rc == 2
    err = capsys.readouterr().err
    assert "NO_PYTHON_CODE_FOUND_IN_STDOUT" in err


def test_section6_orchestrate_with_stdout_pipe_worker(tmp_path: Path):
    """Verifies frame-orchestrate.sh end-to-end with a worker that prints ```python fenced code to stdout."""
    import os
    script_path = Path(__file__).resolve().parent.parent / "frame-orchestrate.sh"
    topic = "orch_stdout_pipe_test"

    # Mock worker simulates `agy -p`: leaves candidate.py untouched and outputs fenced code block to stdout
    mock_agy = tmp_path / "mock_pipe_agy.sh"
    mock_script = textwrap.dedent("""
        #!/usr/bin/env bash
        cat << 'EOF'
        Here is the optimized controller candidate:

        ```python
        import numpy as np

        def run_experiment(task, seed, mode, artifact_dir):
            W = np.array([-0.91, -0.79, 0.045, -0.015], dtype=np.float64)
            return {'policy_weights': W, 'policy_bias': np.zeros(1)}
        ```
        EOF
    """)
    mock_agy.write_text(mock_script.strip(), encoding="utf-8")
    mock_agy.chmod(0o755)

    env = os.environ.copy()
    env["AGY_BIN"] = str(mock_agy)
    env["AGY_TIMEOUT"] = "30"

    root_dir = Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.Popen(
            [str(script_path), "--topic", topic, "--domain", "rl_control", "--steps", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(root_dir),
            env=env,
        )
        stdout, stderr = proc.communicate(input="\ny\n", timeout=60)
        assert proc.returncode == 0, f"Script failed with code {proc.returncode}:\n{stderr}\n{stdout}"
        assert "[Phase 0] Benchmark locked successfully." in stdout
        assert "Candidate code captured (EXTRACTED_FENCED_BLOCK)" in stdout
        assert "Gate 0 Smoke Check Passed (Attempt 1)" in stdout
        assert "[Phase 5] Running Final Untouched Vault-Holdout Audit..." in stdout

        # Verify logs were persisted
        log_dir = root_dir / ".frame" / "campaigns" / topic / "logs"
        assert log_dir.is_dir()
        step_logs = list(log_dir.glob("step_1_*"))
        assert len(step_logs) == 1
        assert (step_logs[0] / "BRIEF.md").is_file()
        assert (step_logs[0] / "attempt1_stdout.log").is_file()
        assert (step_logs[0] / "candidate_final.py").is_file()
    finally:
        import shutil
        shutil.rmtree(root_dir / ".frame" / "benchmarks" / topic, ignore_errors=True)
        shutil.rmtree(root_dir / ".frame" / "campaigns" / topic, ignore_errors=True)


def test_section7_orchestrate_with_stdout_pipe_worker_gate0_retry(tmp_path: Path):
    """Verifies that frame-orchestrate.sh successfully extracts code on Gate 0 retry with a stdout pipe worker."""
    import os
    script_path = Path(__file__).resolve().parent.parent / "frame-orchestrate.sh"
    topic = "orch_stdout_pipe_retry_test"

    mock_agy = tmp_path / "mock_pipe_retry_agy.sh"
    state_file = tmp_path / "invocations.txt"
    mock_script = textwrap.dedent(f"""
        #!/usr/bin/env bash
        touch {state_file}
        INVOCATIONS=$(wc -l < {state_file})
        echo call >> {state_file}

        if [ "$INVOCATIONS" -eq 0 ]; then
          cat << 'EOF'
        Attempt 1 (fails smoke check):
        ```python
        def run_experiment(task, seed, mode, artifact_dir):
            raise RuntimeError('Attempt 1 failed smoke')
        ```
        EOF
        else
          cat << 'EOF'
        Attempt 2 (valid):
        ```python
        import numpy as np
        def run_experiment(task, seed, mode, artifact_dir):
            W = np.array([-0.91, -0.79, 0.045, -0.015], dtype=np.float64)
            return {{'policy_weights': W, 'policy_bias': np.zeros(1)}}
        ```
        EOF
        fi
    """)
    mock_agy.write_text(mock_script.strip(), encoding="utf-8")
    mock_agy.chmod(0o755)

    env = os.environ.copy()
    env["AGY_BIN"] = str(mock_agy)
    env["AGY_TIMEOUT"] = "30"

    root_dir = Path(__file__).resolve().parent.parent
    try:
        proc = subprocess.Popen(
            [str(script_path), "--topic", topic, "--domain", "rl_control", "--steps", "1"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(root_dir),
            env=env,
        )
        stdout, stderr = proc.communicate(input="\ny\n", timeout=60)
        assert proc.returncode == 0, f"Script failed with code {proc.returncode}:\n{stderr}\n{stdout}"
        assert "[Phase 0] Benchmark locked successfully." in stdout
        assert "Candidate code captured (EXTRACTED_FENCED_BLOCK)" in stdout
        assert "Gate 0 Failed (Attempt 1 / 3)" in stdout
        assert "Gate 0 Smoke Check Passed (Attempt 2)" in stdout
        assert "[Phase 5] Running Final Untouched Vault-Holdout Audit..." in stdout

        # Verify logs were persisted for both attempts
        log_dir = root_dir / ".frame" / "campaigns" / topic / "logs"
        assert log_dir.is_dir()
        step_logs = list(log_dir.glob("step_1_*"))
        assert len(step_logs) == 1
        assert (step_logs[0] / "BRIEF.md").is_file()
        assert (step_logs[0] / "attempt1_stdout.log").is_file()
        assert (step_logs[0] / "attempt1_smoke_error.log").is_file()
        assert (step_logs[0] / "attempt2_stdout.log").is_file()
        assert (step_logs[0] / "candidate_final.py").is_file()
    finally:
        import shutil
        shutil.rmtree(root_dir / ".frame" / "benchmarks" / topic, ignore_errors=True)
        shutil.rmtree(root_dir / ".frame" / "campaigns" / topic, ignore_errors=True)


def test_section1_brief_mandatory_jail_instructions(tmp_path: Path):
    """Verifies Section 6 of generate_context_brief markdown matches spec verbatim."""
    from frame.swarm.brief import generate_context_brief
    from frame.schemas import OperatorRole

    topic = "brief_spec_test"
    bench_dir = init_benchmark(tmp_path, topic, "rl_control")
    verify_benchmark(bench_dir)
    approve_benchmark(bench_dir, reviewer="test_user")

    for role in OperatorRole:
        pkt = generate_context_brief(tmp_path, topic, operator_role=role)
        md = pkt["brief_markdown"]

        expected_section_6 = (
            "## 6. Mandatory Jail Worker Instructions\n"
            f"1. Modify `candidate.py` IN-PLACE in the current working directory to implement the `{role.value}` mutation.\n"
            "2. Preserve the exact function signature:\n"
            "   `def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:`\n"
            f"3. Do NOT create new files (do NOT write `cand_{role.value}.py`).\n"
            "4. Do NOT run `python3`, `pytest`, or `frame step` via shell/command tools inside this directory — the host hypervisor will automatically execute Gate 0 (`smoke-check`), Gate 1, and Gate 2 on `candidate.py` the instant you exit.\n"
            "5. Do NOT access any paths outside the current working directory.\n"
        )
        assert expected_section_6 in md, f"Section 6 mismatch for {role.value}:\n{md}"




