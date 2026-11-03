---
name: frame-ar
description: >-
  Out-of-process verified autonomous research engine for Reinforcement Learning (RL) and Kaggle tabular/empirical tasks using FRAME-AR v2. Use when setting up a verified benchmark (Phase 0), running context-isolated mechanical operator swarms (Phases 1-4), injecting human/frontier hypotheses, or auditing generalization on vault holdouts.
---

# FRAME-AR v2: Out-of-Process Autonomous Research Engine

FRAME-AR v2 enforces a strict **Separation of Powers** between ** the Environment (`frame` CLI)** and **the Research Agent (`candidate.py`)**. All data splitting, out-of-process sandboxing, guardrail enforcement, Common Random Numbers (CRN) pilot screening, `rliable` Interquartile Mean (IQM) / ROPE bootstrap testing, and LORD-3 online False Discovery Rate (FDR) accounting are executed deterministically by the `frame` CLI.

---

## ⚠️ 5 IRONCLAD RULES FOR AGENTS

1. **NEVER IMPORT `frame.*` IN CANDIDATE CODE**: Candidate scripts (`candidate.py`) are pure experiment implementations executed inside an isolated subprocess. Never write evaluation metrics, K-fold splitters, or LORD statistical tests inside `candidate.py`.
2. **NEVER BYPASS THE HUMAN GATE IN PHASE 0**: An AI agent must **NEVER** run `frame bench approve` autonomously unless the human user explicitly instructs it to approve after reviewing `BENCHMARK_REVIEW.md`.
3. **RESPECT THE CPYTHON AUDIT SANDBOX**: Candidate scripts run under a mandatory `sys.addaudithook`. Attempting to read `.frame/benchmarks/`, `harness.py`, `bench.lock`, call `os.chmod`/`os.chown`, or open network sockets will immediately terminate the candidate with a `PermissionError`.
4. **DEBUG `BUGGY` NODES LOCALLY (MAX 3 RETRIES)**: If `frame step` returns `status: "buggy"` (Gate 0 smoke failure, syntax error, or shape mismatch), fix the bug in your candidate file and re-run `frame step` (up to 3 attempts). Once `frame step` reaches Gate 1 or Gate 2 (`guardrail_rejected`, `futility_pruned`, `confirmatory_null`, or `promoted_discovery`), your turn for that mutation is complete.
5. **ENFORCE THE BLEND EDGE PROTOCOL**: Whenever a candidate combines, stacks, or ensembles predictions/policies from multiple models, you **MUST** pass `--edge-type blend_ensemble` to `frame step` so it is screened against the Incumbent Best Single Model ($M^*$) under `blend_rope_epsilon`.

---

## 1. Architecture & Workflow Overview

```
=====================================================================================
STAGE A: PHASE 0 — BENCHMARK QUALIFICATION & COMMITTEE LOCK (`frame bench`)
=====================================================================================
1. Initialize / Adapt Benchmark (`harness.py`, `baseline.py`, `positive_control.py`)
2. Run Empirical 6-Point Meta-Verifier:
     `frame bench verify --topic <slug>`
   • Checks Smoke Test, External Competence Floor, 4 Null Probes, Positive Control
     Superiority (> ROPE), [10%, 25%, 50%] Monotonic Degradation Ladder, & k* Power.
3. Rule 1 Blind-to-Crossed Frontier Audit (For custom benchmarks):
   • `frame bench export-stage1 --topic <slug>` -> Blind reviews by Gemini & Claude
   • `frame bench export-stage2 --topic <slug> --model <id>` -> Cross-critiques
4. MANDATORY HUMAN GATE:
   • Human reviews `.frame/benchmarks/<slug>/BENCHMARK_REVIEW.md` (<= 4 guardrails)
   • Human runs: `frame bench approve --topic <slug> --reviewer <name>` -> Writes `bench.lock`

=====================================================================================
STAGE B: PHASES 1–4 — ASYMMETRIC SWARM LOOP (`frame brief` -> `frame step`)
=====================================================================================
1. Context Firewall Packet (`frame brief --topic <slug>`):
   • Selects highest-UCT node (or top P0 `DirectiveTicket` from `frame inject`)
   • Assigns 1 of 4 Mechanical Operators (`error_slice`, `spec_implementation`,
     `hyperparam_schedule`, `ablation_simplify`)
   • Emits parent `candidate.py` + `Dev-Diagnostic` error telemetry + rejected siblings
2. Candidate Implementation (`cand_<role>.py`):
   • Implements `run_experiment(task, seed, mode, artifact_dir) -> dict`
3. Out-of-Process 4-Gate Funnel (`frame step --topic <slug> --candidate cand_<role>.py`):
   • Gate 0: `bench.lock` verification + Operator AST deduplication + 5s Smoke Test
   • Gate 1 (`dev_diagnostic`): Hard Guardrail Invariants + k=3 CRN Pilot Futility Screen
   • Gate 2 (`confirmatory`): Fresh HMAC step-indexed seeds k*, re-evaluating BOTH
     Challenger and Incumbent M* (eliminating Winner's Curse) via IQM Bootstrap / ROPE t-test
     under LORD-3 alpha_t. (Returns 1-bit verdict only; zero confirmatory telemetry leaked).

=====================================================================================
STAGE C: EPOCH INTERVENTION & CAMPAIGN CLOSURE (`frame inject` / `frame vault-eval`)
=====================================================================================
• Inject Frontier / Human Hypothesis onto any past or running branch:
    `frame inject --topic <slug> --parent <node_id> --spec "<mutation_spec>" --priority 0`
• Preempt & terminate a flawed running candidate (0 LORD alpha wealth spent):
    `frame abort --topic <slug> --node <node_id> --inject-spec "<replacement_spec>"`
• Final Untouched Vault-Holdout Generalization Audit:
    `frame vault-eval --topic <slug>`
```

---

## 2. The Candidate Code Contract (`candidate.py`)

Every candidate file submitted to `frame step` must be a self-contained Python module defining:

```python
from pathlib import Path
from typing import Any, Dict
import numpy as np

def run_experiment(
    task: Dict[str, Any],
    seed: int,
    mode: str,          # "smoke" (fast <5s sanity check) or "full" (standard evaluation)
    artifact_dir: Path  # Directory to save checkpoints, .npy predictions, or policy weights
) -> Dict[str, Any]:
    """Executes training/inference for a single deterministic seed."""
    ...
```

### Domain Return Signatures
* **Kaggle / Tabular (`domain == "kaggle_tabular"`)**:
  * `task` provides `"X_train"`, `"y_train"`, `"X_eval"`, `"feature_names"`. (Note: `"y_eval"` is held secretly in the parent verifier process and is never in `task`).
  * Must return `{"predictions": np.ndarray}` matching `len(task["X_eval"])` with 100% finite values.
  * Must fit all preprocessors, scalers, and target encoders strictly inside `run_experiment` on `X_train`/`y_train` only.
* **RL Control (`domain == "rl_control"`)**:
  * `task` provides `"train_episode_seeds"`, `"obs_dim"`, `"basis_dim"`, `"action_bounds"`, `"horizon"`.
  * Scale iterations/rollouts down when `mode == "smoke"` so Gate 0 finishes in $<5$ seconds.
  * Must return policy parameters (e.g. `{"W": np.ndarray, "b": np.ndarray}`) which the parent verifier will evaluate out-of-process on sealed evaluation episode seeds.

---

## 3. Mechanical Swarm Operators (Replacing Persona Hats)

To prevent single-model echo chambers when running Flash subagents, workers do not roleplay generic personas. Each worker executes a specific mechanical operator assigned by `frame brief`:

| Operator Role (`--role`) | Mechanical Mandate |
| :--- | :--- |
| **`error_slice`** | Inspect the `Dev-Diagnostic Error Telemetry` in the brief (e.g., subgroup RMSE by regime, top nonlinear error correlations, or wind-disturbance cost correlation). Add a targeted feature interaction, state basis term, or regime-conditional branch addressing that exact failure slice. |
| **`spec_implementation`** | Implement the Priority Directive specification (`DirectiveTicket`) if present; otherwise replace the estimator/controller family (e.g., linear Ridge $\to$ HistGradientBoosting, or static PD $\to$ Cross-Entropy Method / actor-critic). |
| **`hyperparam_schedule`** | Hold the structural model/features fixed. Mutate regularization (`l2_regularization`, shrinkage, weight decay), learning rate / step schedule, tree depth / leaf constraints, or control gain damping. |
| **`ablation_simplify`** | Remove at least 10–15% of complexity (prune weak features, simplify basis terms, or reduce model capacity) while maintaining or improving the primary metric and tightening guardrail margins. |

---

## 4. Complete CLI Command Reference

### Phase 0: Benchmark Setup & Approval
```bash
# 1. Initialize a benchmark workspace (kaggle_tabular or rl_control)
python3 -m frame.cli bench init --topic <slug> --domain <kaggle_tabular|rl_control>

# 2. Run the 6-point empirical meta-verifier (emits BENCHMARK_REVIEW.md)
python3 -m frame.cli bench verify --topic <slug>

# 3. (Rule 1 Committee) Export & record Stage 1 blind reviews + Stage 2 cross-critiques
python3 -m frame.cli bench export-stage1 --topic <slug>
python3 -m frame.cli bench record-stage1 --topic <slug> --model gemini_frontier --file gemini_s1.md
python3 -m frame.cli bench record-stage1 --topic <slug> --model claude_sonnet --file claude_s1.md
python3 -m frame.cli bench export-stage2 --topic <slug> --model gemini_frontier
python3 -m frame.cli bench export-stage2 --topic <slug> --model claude_sonnet
python3 -m frame.cli bench record-stage2 --topic <slug> --model gemini_frontier --file gemini_s2.md
python3 -m frame.cli bench record-stage2 --topic <slug> --model claude_sonnet --file claude_s2.md

# 4. HUMAN GATE ONLY: Approve and cryptographically lock the benchmark
python3 -m frame.cli bench approve --topic <slug> --reviewer <human_name> --notes "Verified anchor and guardrails"
```

### Phases 1–4: Swarm Execution & Hypothesis Injection
```bash
# 1. Generate the Context Firewall Brief for the next UCT node (or top priority directive)
python3 -m frame.cli brief --topic <slug>

# 2. Evaluate a candidate script through Gate 0 (Smoke), Gate 1 (Dev), and Gate 2 (Confirmatory)
python3 -m frame.cli step --topic <slug> --candidate <cand.py> --parent <node_id> --role <role>

# 3. Evaluate an ensemble/blend against the Incumbent Best Single Model (M*)
python3 -m frame.cli step --topic <slug> --candidate <blend.py> --parent <node_id> --edge-type blend_ensemble

# 4. Inject a high-priority Human or Frontier (/boost) hypothesis onto any node
python3 -m frame.cli inject --topic <slug> --parent <node_id> --priority 0 --spec "<concrete_mutation_instructions>"

# 5. Abort a running node mid-execution (0 alpha spent) and queue a replacement spec
python3 -m frame.cli abort --topic <slug> --node <node_id> --inject-spec "<corrected_instructions>"

# 6. Inspect MCTS tree status, incumbent M*, LORD alpha wealth, and active directives
python3 -m frame.cli status --topic <slug>

# 7. Run final untouched Vault-Holdout audit at campaign completion
python3 -m frame.cli vault-eval --topic <slug>
```
