import ast
from dataclasses import dataclass
import importlib.util
import math
import os
import pickle
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, Optional
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.special


def bind_candidate_select_action(candidate_path: Path, candidate_output: Dict[str, Any]) -> Dict[str, Any]:
    """Safely extracts `select_action(obs, policy_state)` from candidate_path into a restricted math namespace."""
    if not candidate_path.is_file() or not isinstance(candidate_output, dict):
        return candidate_output

    source = candidate_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return candidate_output

    has_select = any(
        (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "select_action")
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "select_action" for t in node.targets)
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "select_action"
        )
        for node in tree.body
    )
    if not has_select:
        return candidate_output

    # Keep pure functions, helper classes, and constant/annotated assignments (strip imports and run_experiment)
    safe_body = [
        node for node in tree.body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != "run_experiment")
        or isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef))
    ]
    mod_ast = ast.Module(body=safe_body, type_ignores=[])
    ast.fix_missing_locations(mod_ast)

    safe_builtins = {
        "__build_class__": __build_class__,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "filter": filter, "float": float, "hasattr": hasattr,
        "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
        "max": max, "min": min, "pow": pow, "range": range, "round": round,
        "set": set, "sorted": sorted, "str": str, "sum": sum, "super": super,
        "property": property, "classmethod": classmethod, "staticmethod": staticmethod,
        "tuple": tuple, "zip": zip, "Exception": Exception, "ValueError": ValueError,
        "KeyError": KeyError, "TypeError": TypeError,
    }
    restricted_env: Dict[str, Any] = {
        "__builtins__": safe_builtins,
        "__name__": "__frame_candidate__",
        "np": np,
        "numpy": np,
        "math": math,
        "scipy": scipy,
    }

    try:
        code_obj = compile(mod_ast, filename=str(candidate_path.name), mode="exec")
        exec(code_obj, restricted_env)
        fn = restricted_env.get("select_action")
        if callable(fn):
            candidate_output["_select_action"] = fn
    except Exception:
        pass

    return candidate_output



@dataclass
class SandboxExecutionResult:
    passed: bool
    aborted: bool
    candidate_output: Optional[Dict[str, Any]]
    error_message: str
    stdout: str
    stderr: str
    wall_clock_sec: float = 0.0


def load_harness_module(harness_path: Path) -> Any:
    """Loads the trusted benchmark `harness.py` in the parent verifier process ONLY."""
    spec = importlib.util.spec_from_file_location("frame_trusted_harness", str(harness_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load harness at {harness_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_WORKER_RUNNER_TEMPLATE = textwrap.dedent(
    """
    import importlib.util
    import os
    from pathlib import Path
    import pickle
    import sys

    FORBIDDEN_DIR = Path(sys.argv[1]).resolve()
    CANDIDATE_PATH = Path(sys.argv[2]).resolve()
    TASK_PKL_PATH = Path(sys.argv[3]).resolve()
    OUTPUT_PKL_PATH = Path(sys.argv[4]).resolve()
    ARTIFACT_DIR = Path(sys.argv[5]).resolve()
    SEED = int(sys.argv[6])
    MODE = str(sys.argv[7])

    with TASK_PKL_PATH.open("rb") as f:
        public_task = pickle.load(f)

    # Pre-load candidate source code before activating the filesystem audit hook
    cand_spec = importlib.util.spec_from_file_location("candidate_module", str(CANDIDATE_PATH))
    cand_mod = importlib.util.module_from_spec(cand_spec)

    def _audit_hook(event, args):
        if event in ("os.chmod", "os.chown", "socket.connect", "socket.bind"):
            raise PermissionError(f"Sandbox violation: operation '{event}' is prohibited inside candidate code.")
        if event == "sys._getframe":
            if args:
                fname = getattr(args[0].f_code, "co_filename", "")
                if (
                    fname == str(CANDIDATE_PATH)
                    or fname.endswith("candidate_under_test.py")
                    or fname.endswith("_worker_entry.py")
                    or ("site-packages" not in fname and "/usr/lib" not in fname)
                ):
                    raise PermissionError(f"Sandbox violation: operation '{event}' is prohibited inside candidate code.")
                return
            raise PermissionError(f"Sandbox violation: operation '{event}' is prohibited inside candidate code.")
        if event in ("os.remove", "os.unlink", "os.rename"):
            if args:
                try:
                    p = Path(str(args[0])).resolve()
                    if FORBIDDEN_DIR in p.parents or p == FORBIDDEN_DIR:
                        raise PermissionError(f"Sandbox Audit Violation: {event} blocked on {p}")
                except PermissionError:
                    raise
                except Exception:
                    pass
        elif event == "open" and args:
            path_arg = args[0]
            if isinstance(path_arg, (str, bytes, os.PathLike)):
                try:
                    p = Path(os.fsdecode(path_arg)).resolve()
                    if FORBIDDEN_DIR in p.parents or p == FORBIDDEN_DIR:
                        raise PermissionError(
                            f"Sandbox Audit Violation: Candidate attempted to access locked benchmark path {p}"
                        )
                except PermissionError:
                    raise
                except Exception:
                    pass
        elif event.startswith("socket."):
            raise PermissionError(f"Sandbox Audit Violation: Network access ({event}) is prohibited.")

    sys.addaudithook(_audit_hook)

    cand_spec.loader.exec_module(cand_mod)
    if not hasattr(cand_mod, "run_experiment"):
        raise AttributeError("Candidate file must define `run_experiment(task, seed, mode, artifact_dir) -> dict`.")

    result = cand_mod.run_experiment(
        task=public_task,
        seed=SEED,
        mode=MODE,
        artifact_dir=ARTIFACT_DIR,
    )
    if not isinstance(result, dict):
        raise TypeError("`run_experiment` must return a Python dict.")

    with OUTPUT_PKL_PATH.open("wb") as f:
        pickle.dump(result, f)
    """
)


def run_candidate_isolated(
    candidate_path: Path,
    public_task: Dict[str, Any],
    seed: int,
    mode: str,
    timeout_sec: float,
    forbidden_bench_dir: Path,
    pid_file: Optional[Path] = None,
    persist_artifacts_to: Optional[Path] = None,
) -> SandboxExecutionResult:
    """Executes `candidate_path` in a separate OS process with zero access to secret targets or bench_dir."""
    candidate_path = candidate_path.resolve()
    forbidden_bench_dir = forbidden_bench_dir.resolve()
    t_start = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="frame_sandbox_") as tmp_str:
        tmp_dir = Path(tmp_str).resolve()
        cand_copy = tmp_dir / "candidate_under_test.py"
        shutil.copy2(candidate_path, cand_copy)

        task_pkl = tmp_dir / "public_task.pkl"
        output_pkl = tmp_dir / "candidate_output.pkl"
        worker_py = tmp_dir / "_worker_entry.py"
        artifact_dir = tmp_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with task_pkl.open("wb") as f:
            pickle.dump(public_task, f)
        worker_py.write_text(_WORKER_RUNNER_TEMPLATE, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        cmd = [
            sys.executable,
            str(worker_py),
            str(forbidden_bench_dir),
            str(cand_copy),
            str(task_pkl),
            str(output_pkl),
            str(artifact_dir),
            str(seed),
            str(mode),
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(tmp_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        if pid_file is not None:
            pid_file.parent.mkdir(parents=True, exist_ok=True)
            pid_file.write_text(str(proc.pid), encoding="utf-8")

        try:
            stdout, stderr = proc.communicate(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxExecutionResult(
                passed=False,
                aborted=False,
                candidate_output=None,
                error_message=f"Candidate timed out after {timeout_sec:.1f}s.",
                stdout=stdout,
                stderr=stderr,
                wall_clock_sec=time.perf_counter() - t_start,
            )
        finally:
            if pid_file is not None and pid_file.exists():
                try:
                    pid_file.unlink()
                except OSError:
                    pass

        abort_marker = pid_file.with_suffix(".aborted") if pid_file is not None else None
        was_aborted = proc.returncode in (-signal.SIGTERM, -signal.SIGINT, 143, 137) or (
            abort_marker is not None and abort_marker.exists()
        )
        if abort_marker is not None and abort_marker.exists():
            try:
                abort_marker.unlink()
            except OSError:
                pass

        if was_aborted:
            return SandboxExecutionResult(
                passed=False,
                aborted=True,
                candidate_output=None,
                error_message="Execution aborted by user/governor via `frame abort`.",
                stdout=stdout,
                stderr=stderr,
                wall_clock_sec=time.perf_counter() - t_start,
            )

        if proc.returncode != 0 or not output_pkl.exists():
            return SandboxExecutionResult(
                passed=False,
                aborted=False,
                candidate_output=None,
                error_message=stderr.strip() or f"Process exited with code {proc.returncode}",
                stdout=stdout,
                stderr=stderr,
                wall_clock_sec=time.perf_counter() - t_start,
            )

        with output_pkl.open("rb") as f:
            output_data = pickle.load(f)

        output_data = bind_candidate_select_action(candidate_path, output_data)

        if persist_artifacts_to is not None and any(artifact_dir.iterdir()):
            persist_artifacts_to.mkdir(parents=True, exist_ok=True)
            for item in artifact_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, persist_artifacts_to / item.name)

        return SandboxExecutionResult(
            passed=True,
            aborted=False,
            candidate_output=output_data,
            error_message="",
            stdout=stdout,
            stderr=stderr,
            wall_clock_sec=time.perf_counter() - t_start,
        )
