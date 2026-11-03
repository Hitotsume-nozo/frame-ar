# FRAME-AR: Foundation for Rigorous Autonomous & Mathematical Experimentation — Auto-Research

[![Version: 2.5.0 (Stable)](https://img.shields.io/badge/version-2.5.0%20(Stable)-brightgreen.svg)](CHANGELOG.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests: 29 Passed (100%)](https://img.shields.io/badge/tests-29%20passed%20(100%25)-brightgreen.svg)](tests/)
[![FDR Control: LORD-3 Streaming](https://img.shields.io/badge/FDR%20Control-LORD%203%20%28Riemann%20Zeta%29-blueviolet.svg)](#online-fdr-control-streaming-lord-3)
[![Sandboxed: CPython Audit Hook](https://img.shields.io/badge/Sandbox-CPython%20Audit%20Hook-orange.svg)](#out-of-process-cpython-audit-sandbox)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **FRAME-AR v2.5** is an out-of-process, tamper-proof autonomous scientific discovery and empirical auto-research engine. It pairs **Context-Isolated Evolutionary Swarms & Monte Carlo Tree Search (MCTS)** with a **Production-Grade Statistical Verification Funnel** (Common Random Numbers pilot screening, `rliable` Interquartile Mean bootstrap testing, streaming LORD-3 false discovery rate control, open policy expressivity, and sealed cryptographic vault holdout audits).

---

## 📑 Table of Contents

- [Core Philosophy: Strict Separation of Powers](#-core-philosophy-strict-separation-of-powers)
- [System Architecture](#-system-architecture)
- [The 5-Phase Research Lifecycle](#-the-5-phase-research-lifecycle)
- [The 3-Gate Evaluation Funnel](#-the-3-gate-evaluation-funnel)
- [Key Upgrades in FRAME-AR v2.5 (Stable)](#-key-upgrades-in-frame-ar-v25-stable)
- [Evolution & Overhauls: v1 to v2.5](#-evolution--overhauls-v1-to-v25)
- [Quickstart: Running Basic Toy Runs](#-quickstart-running-basic-toy-runs)
  - [Installation](#1-installation)
  - [Automated Toy Run with Bash Hypervisor](#2-automated-toy-run-with-bash-hypervisor)
  - [Interactive Dual-Pane Monitoring & Injection](#3-interactive-dual-pane-monitoring--injection)
  - [Manual Step-by-Step CLI Walkthrough](#4-manual-step-by-step-cli-walkthrough)
- [CLI Reference](#-cli-reference)
- [Repository Layout](#-repository-layout)
- [Verification & Testing](#-verification--testing)
- [Legacy v1 Archive](#-legacy-v1-archive)
- [License](#-license)

---

## 🏛️ Core Philosophy: Strict Separation of Powers

Autonomous machine learning agents routinely succumb to three fatal failure modes:
1. **Agentic Drift & Snooping**: When LLM workers share an in-process runtime with the test harness, they inspect holdout splits, reverse-engineer synthetic evaluators, or leak global statistics into feature scaling.
2. **Pseudoreplication & P-Hacking**: Running dozens of empirical trials and declaring victory on peak evaluation scores leads to selection bias and false positive discoveries.
3. **The Blend Conflation Illusion**: Ensembles or model stacking layers are hailed as breakthroughs merely because they outperform a naive baseline, despite offering zero statistically significant advantage over the best single constituent model.

**FRAME-AR enforces a strict Separation of Powers**:
- **The Evaluation Environment (`frame` CLI / `frame-orchestrate.sh`)**: Authoritative, immutable, and out-of-process. Controls data partitioning, timing constraints, statistical testing, and tree state.
- **The Research Agent (`candidate.py`)**: Stateless and isolated. The agent receives only a task specification (`BRIEF.md`) and a parent candidate file inside an ephemeral jail (`/tmp/frame_jail_*`). Candidate code **never imports `frame.*`**, never sees test holdouts, and runs under an unbypassable CPython security audit hook.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HOST HYPERVISOR & FRAME CLI                           │
│  • Phase 0 Committee Qualification & Cryptographic bench.lock Seal          │
│  • Ephemeral Jail Construction (/tmp/frame_jail_*)                          │
│  • Gate 0 Pre-Flight AST Hash Deduplication & Runtime Projection            │
│  • Gate 1 Dev Diagnostic Screen (CRN seeds k=3 vs Parent & Incumbent M*)    │
│  • Gate 2 Confirmatory Holdout (rliable IQM Bootstrap & LORD-3 FDR)        │
│  • MCTS Tree Management, Ancestor Retirement & Convergence Detection        │
│  • Phase 5 Final Sealed Vault-Holdout Audit                                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ Ephemeral Jail Isolation
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ISOLATED CANDIDATE PROCESS (candidate.py)                │
│  • Pure experiment code: def run_experiment(...) & select_action(...)       │
│  • CPython sys.addaudithook: Blocks filesystem snooping, sys._getframe,     │
│    network sockets, subprocess spawning, and permission modifications       │
│  • Completely oblivious to test seeds, holdouts, and verification internals │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 System Architecture

```
                               ┌──────────────────────────┐
                               │  Phase 0: Qualification  │
                               │  • 7-Point Meta-Verify   │
                               │  • Committee Review      │
                               │  • Cryptographic Lock    │
                               └────────────┬─────────────┘
                                            │ bench.lock
                                            ▼
           ┌──────────────────────────────────────────────────────────────────┐
           │                      Phases 1–4: MCTS Search Loop                 │
           │                                                                  │
           │  ┌────────────────────┐            ┌──────────────────────────┐  │
           │  │ 1. UCT Selection   │            │ 2. Jail Preparation      │  │
           │  │ • Anti-Bleed UCT   ├───────────►│ • /tmp/frame_jail_*      │  │
           │  │ • Retire Ancestors │            │ • BRIEF.md + candidate.py│  │
           │  └────────────────────┘            └────────────┬─────────────┘  │
           │                                                 │                │
           │                                                 ▼                │
           │  ┌────────────────────┐            ┌──────────────────────────┐  │
           │  │ 4. Evaluation      │            │ 3. Worker Execution      │  │
           │  │ • Gate 0: AST/Time │◄───────────┤ • agy -p <short_prompt>  │  │
           │  │ • Gate 1: Dev CRN  │            │ • In-place code mutation │  │
           │  │ • Gate 2: LORD-3   │            │ • extract-code fallback  │  │
           │  └─────────┬──────────┘            └──────────────────────────┘  │
           │            │                                                     │
           └────────────┼─────────────────────────────────────────────────────┘
                        │ Target Steps or Stagnation Early-Stop
                        ▼
           ┌──────────────────────────┐
           │   Phase 5: Vault Audit   │
           │  • Sealed Holdout Split  │
           │  • Generalization Check  │
           │  • Cryptographic Seal    │
           └──────────────────────────┘
```

---

## 🔬 The 5-Phase Research Lifecycle

### Phase 0: Benchmark Qualification & Cryptographic Lock
Before any agent runs, the benchmark must undergo a rigorous 7-point meta-verification and a 3-stage review committee:
1. **Smoke Test**: Baseline executes cleanly without warnings or unhandled exceptions.
2. **Baseline Competence**: Baseline exceeds trivial floor score thresholds.
3. **Null-Model Separation**: Baseline statistically separates from zero/random null models ($p < 0.01$).
4. **Positive Control Separation**: An existing known strong implementation reliably beats the baseline ($p < 0.005$).
5. **Noise Sensitivity Monotonicity**: Increasing observation noise degrades performance monotonically.
6. **Domain Evidence Audit**: Required domain assets and plant dynamics are present and valid.
7. **Polymorphic Expressivity Probe**: Harness successfully evaluates 3 structurally incompatible architectures (linear dict, nonlinear MLP, non-parametric lookup table) without parameter shape lock-in.

Once qualified, a 3-stage blind-to-crossed committee review occurs (Stage 1 Independent LLM Reviews $\to$ Stage 2 Cross-Critique $\to$ Stage 3 Human Approval), emitting a signed, immutable `bench.lock` (RFC 8785 Canonical JSON SHA-256 digest).

### Phase 1: Context Briefing & Mechanical Operator Assignment
The orchestrator selects the next node via UCT and assigns one of 4 mechanical operator roles in fixed cyclic order:
- `SPEC_IMPLEMENTATION`: Introduces structural hypotheses or architectural additions.
- `ERROR_SLICE`: Identifies underperforming failure modes and targets residual errors.
- `HYPERPARAM_SCHEDULE`: Tunes optimization schedules, learning rates, or regularization weights.
- `ABLATION_SIMPLIFY`: Strips non-essential complexity to maximize parameter efficiency.

The orchestrator generates an isolated `BRIEF.md` containing the current node's performance, the operator goal, and the reigning Incumbent score.

### Phase 2: Jail Preparation & In-Place Code Mutation
An ephemeral jail (`/tmp/frame_jail_*`) is provisioned with `BRIEF.md`, parent candidate code, and metadata. The worker agent modifies `candidate.py` in-place. If the worker prints code to stdout, the hypervisor's `extract-code` engine automatically recovers the candidate script.

### Phase 3: The 3-Gate Evaluation Funnel
The candidate is passed into the deterministic evaluation funnel (detailed below).

### Phase 4: MCTS Backpropagation & Epistemic State Update
Node visits, rewards, and ancestry are updated across disk. If the candidate achieves a statistically confirmed improvement over Incumbent $M^*$ at Gate 2, it is promoted as the new Global Incumbent.

### Phase 5: Sealed Vault-Holdout Generalization Audit
At campaign completion (or upon early convergence), the reigning Incumbent is audited against the sealed, untouched Vault-Holdout split to confirm out-of-sample generalization.

---

## 🛡️ The 3-Gate Evaluation Funnel

Every candidate must pass through 3 sequential, progressively more rigorous evaluation gates:

```
Candidate File
      │
      ▼
┌──────────────┐   Failed (Syntax/Duplicate/Timeout)   ┌────────────────┐
│    Gate 0    ├──────────────────────────────────────►│     BUGGY      │
│  Pre-Flight  │                                       └────────────────┘
└──────┬───────┘
       │ Passed
       ▼
┌──────────────┐   Delta vs Parent <= 0 OR vs Incumbent <= 0
│    Gate 1    ├──────────────────────────────────────►┌────────────────┐
│Dev Diagnostic│                                       │FUTILITY_PRUNED │
└──────┬───────┘                                       └────────────────┘
       │ Beat Parent & Incumbent (0 Alpha Spent)
       ▼
┌──────────────┐   p >= alpha_t                        ┌────────────────┐
│    Gate 2    ├──────────────────────────────────────►│CONFIRMATORY_NULL
│ Confirmatory │                                       └────────────────┘
└──────┬───────┘
       │ p < alpha_t (Significant Discovery!)
       ▼
┌──────────────────────────────┐
│     PROMOTED_DISCOVERY       │ (Becomes Incumbent M*)
└──────────────────────────────┘
```

### Gate 0: Pre-Flight Integrity, AST Deduplication & Runtime Projection
- **AST Deduplication**: `OperatorASTMemory` parses canonical ASTs. Exact code duplicates or cosmetic variable renames are rejected with exit code `2` (`GATE0_AST_DUPLICATE`) with **zero compute or alpha spent**.
- **Runtime Projection Guard**: Measures smoke execution time $T_{\text{smoke}}$. If $T_{\text{smoke}} > \max(5.0\text{s}, 0.22 \times T_{\text{timeout}})$, fails fast with exit code `5` (`GATE0_RUNTIME_PROJECTION_EXCEEDED`), forcing the worker to optimize its implementation in the local debug loop.
- **Local Retry Loop**: The hypervisor permits up to 3 rapid debug retries within the jail before recording node status.

### Gate 1: Dev Diagnostic Screening ($k=3$ CRN Seeds)
- Evaluates candidate on a fast Common Random Numbers (CRN) pilot split ($k=3$ seeds).
- **Incumbent Superiority Filter**: Candidate must beat **both** its parent node and the reigning Global Incumbent $M^*$ (`pilot_delta_vs_parent > 0` AND `pilot_delta_vs_incumbent > 0`).
- If it fails to outperform either, it is marked `FUTILITY_PRUNED` and discarded **without spending any LORD-3 alpha wealth**.

### Gate 2: Full Confirmatory Holdout & LORD-3 FDR Accounting ($k^*=5..15$ Seeds)
- Evaluates candidate on the full confirmatory holdout split.
- Employs `rliable` Interquartile Mean (IQM) bootstrap testing with pre-registered ROPE margins ($\epsilon_{\text{ROPE}}$).
- **Online FDR Control (Streaming LORD-3)**: Dynamically calculates significance threshold $\alpha_t$ following a Riemann Zeta decay schedule:
  $$\gamma_t = \frac{1}{\zeta(1.6) \cdot t^{1.6}}, \quad \alpha_t = W_0 \cdot \gamma_t$$
  Ensures the False Discovery Rate across arbitrary stopping times is rigorously controlled ($\mathbb{E}[\text{FDR}] \le \alpha$).
- If $p < \alpha_t$, candidate is certified as `PROMOTED_DISCOVERY` and crowned new Incumbent $M^*$.

### The Blend Edge Protocol
Whenever a candidate combines multiple model policies or predictions, it must pass `--edge-type blend_ensemble`. Blend candidates are screened against the **Incumbent Single Best Model ($M^*$)** rather than an inferior baseline, completely eliminating parameter-bloated ensemble illusions.

---

## 🚀 Key Upgrades in FRAME-AR v2.5 (Stable)

| Capability | Previous Implementation | v2.5 Stable Implementation |
| :--- | :--- | :--- |
| **Execution Timeout** | Static `30.0s` limit | **Dynamic Baseline-Anchored Calibration**: Measures baseline & positive control runtime; auto-calibrates $T_{\text{timeout}} = \text{clip}(8 \times \max(T_{\text{base}}, T_{\text{pos}}), 45\text{s}, 600\text{s})$. |
| **Runtime Protection** | Timeout during Gate 1 | **Gate-0 Runtime Projection Guard**: Smoke test projects full runtime; fails fast with code `5` if execution will exceed budget. |
| **Tree Search Focus** | Regressed to root ancestor | **Superseded Ancestor Retirement**: Automatically retires ancestors that spawned promoted children and trail Incumbent by $> 2 \times \epsilon_{\text{ROPE}}$. |
| **Branch Thrashing** | Unbounded retries on dead branches | **Branch Exhaustion Penalty**: Heavy UCT penalty after 4 consecutive failures across an operator cycle; pivots search to active branches. |
| **Campaign Stopping** | Ran fixed steps blindly | **Convergence Early-Stop**: Stops search after 5 stagnant steps without promotion, jumping cleanly to Vault-Holdout audit. |
| **Disk Synchronization**| Ancestors missing child state | **Deep Multi-Level Backprop**: Writes updated visits, rewards, and children IDs to `node_record.json` on disk across all ancestor levels. |
| **Policy Expressivity** | Hardcoded parameter vectors | **Open Policy Expressivity Contract**: `select_action(obs, policy_state)` in candidate enables arbitrary architectures. |
| **Worker Pipe** | Tool failures on headless runs | **Stateless Stdin/Stdout Pipe**: Zero-tool worker mode with `extract-code` supporting in-place and fenced code blocks. |
| **Inter-Process IPC** | Race conditions with orchestrator | **Lock-Free Maildir IPC**: Single-writer filesystem maildir pattern for pause, abort, and directive injection. |

---

## 📈 Evolution & Overhauls: v1 to v2.5

FRAME-AR has undergone a comprehensive evolutionary overhaul to transition from an experimental in-process prototype to a production-grade, out-of-process autonomous platform:

### 1. The Legacy v1 Monolith (Archived at `v1.0.0-archive`, Commit `1a55a63`)
- **Architecture**: In-process multi-agent system combining symbolic math engines (L0 SymPy, L1 IBP, L2 IBnB), TAPAScore calculation, and Kaggle runners.
- **Core Limitation**: Lack of process isolation allowed agents to introspect evaluation memory, leak holdout splits, and deadlock inside complex in-process loops.
- **Preservation**: Archived with 793 passing tests in `ARCHIVE_V1_MANIFEST.md` and `archive/v1-main`.

### 2. The v2.0 Out-of-Process Paradigm
- Completely rewrote the core engine with strict **Separation of Powers**.
- Implemented CPython audit sandbox (`sys.addaudithook`), 3-gate evaluation funnel, and cryptographic benchmark locking (`bench.lock`).

### 3. Incremental Overhauls (v2.1 – v2.4)
- **v2.1**: Static Bash hypervisor (`frame-orchestrate.sh`), out-of-band monitor (`frame-mon.py`), and Riemann Zeta LORD-3 decay.
- **v2.2**: Stateless stdin/stdout worker pipe with `extract-code` recovery and persistent step logging.
- **v2.3**: In-place jail mutations with short-argv (<200 bytes) prompts and Grade-A nonlinear RL templates (`wind_pendulum_rl`).
- **v2.4**: Open policy expressivity contract (`select_action`), Phase-0 polymorphic expressivity probe (Check #7), and Gate-1 incumbent superiority filtering.

### 4. The v2.5.0 Stable Release
- Solved all empirical scaling and convergence bottlenecks: dynamic timeout calibration, Gate-0 runtime projection, ancestor retirement, branch exhaustion caps, and automated early-stop.
- Fully verified with 29 comprehensive test suites covering end-to-end orchestration, sandboxing, and statistical accounting.

---

## 🛠️ Quickstart: Running Basic Toy Runs

### 1. Installation

Clone and install FRAME-AR in editable mode with development dependencies:

```bash
cd ~/frame
pip install -e ".[dev]"
```

Verify your installation:
```bash
python3 -c "import frame; print(frame.__version__)"
# Output: 2.5.0
```

---

### 2. Automated Toy Run with Bash Hypervisor

The fastest way to execute an end-to-end autonomous research run is with `frame-orchestrate.sh`. The hypervisor automatically handles benchmark initialization, 7-point meta-verification, Phase 0 committee locking, jail isolation, MCTS search, and final vault evaluation.

#### Example A: Nonlinear Reinforcement Learning Toy Run (`wind_pendulum_rl`)
```bash
./frame-orchestrate.sh --topic wind_pendulum_rl --domain rl_control --steps 5
```

#### Example B: Tabular Machine Learning Toy Run (`housing_tab`)
```bash
./frame-orchestrate.sh --topic housing_tab --domain kaggle_tabular --steps 5
```

During Phase 0, if the benchmark is not yet locked, the hypervisor launches the manual audit review. Type `y` and press Enter to approve `bench.lock` and initiate the MCTS search.

---

### 3. Interactive Dual-Pane Monitoring & Injection

For full real-time visibility and human-in-the-loop steerability, run the orchestrator in one terminal pane and the live monitor in another:

#### Pane 1: Running the Hypervisor
```bash
./frame-orchestrate.sh --topic wind_pendulum_rl --domain rl_control --steps 10
```

#### Pane 2: Live Monitor Dashboard & Directive Injection
```bash
# View live ANSI dashboard (MCTS tree, Incumbent M*, LORD alpha wealth):
python3 frame-mon.py --topic wind_pendulum_rl

# Pause the campaign:
python3 frame-mon.py --topic wind_pendulum_rl --pause

# Resume the campaign:
python3 frame-mon.py --topic wind_pendulum_rl --resume

# Inject a high-priority hypothesis onto the current Incumbent:
python3 frame-mon.py --topic wind_pendulum_rl --inject-parent node_4 \
  --spec "Add cubic velocity shear compensation with nonlinear saturation at u_max"

# Abort an underperforming branch without spending alpha wealth:
python3 frame-mon.py --topic wind_pendulum_rl --abort node_2 \
  --spec "Replacement hypothesis: test piecewise lookup policy"
```

---

### 4. Manual Step-by-Step CLI Walkthrough

You can also operate FRAME-AR deterministically using individual CLI subcommands:

```bash
# 1. Initialize a benchmark directory
python3 -m frame.cli bench init --topic my_toy_run --domain rl_control

# 2. Run the 7-point empirical qualification
python3 -m frame.cli bench verify --topic my_toy_run

# 3. Approve and cryptographically lock the benchmark
python3 -m frame.cli bench approve --topic my_toy_run --reviewer "$USER"

# 4. Prepare an isolated execution jail for the next worker
python3 -m frame.cli prepare-jail --topic my_toy_run --jail-dir /tmp/my_jail

# 5. Inside /tmp/my_jail, inspect BRIEF.md and edit candidate.py
# Run pre-flight Gate-0 smoke check:
python3 -m frame.cli smoke-check --topic my_toy_run --candidate /tmp/my_jail/candidate.py

# 6. Evaluate candidate through the full 3-gate funnel
python3 -m frame.cli step --topic my_toy_run --candidate /tmp/my_jail/candidate.py \
  --parent node_0 --role spec_implementation

# 7. Check campaign status and MCTS tree
python3 -m frame.cli status --topic my_toy_run

# 8. Run final Vault-Holdout audit on the reigning Incumbent
python3 -m frame.cli vault-eval --topic my_toy_run
```

---

## 💻 CLI Reference

| Command | Arguments | Purpose |
| :--- | :--- | :--- |
| `frame bench init` | `--topic <slug> --domain <type>` | Scaffold benchmark with baseline, positive control, harness, and spec. |
| `frame bench verify` | `--topic <slug>` | Execute 7-point empirical meta-verification and expressivity probe. |
| `frame bench check-lock` | `--topic <slug>` | Verify cryptographic validity of `bench.lock`. |
| `frame bench approve` | `--topic <slug> --reviewer <name>` | Sign committee review and generate cryptographic `bench.lock`. |
| `frame bench export-stage1` | `--topic <slug>` | Export anonymized Stage 1 blind review packet. |
| `frame bench export-stage2` | `--topic <slug> --reviewer <name>` | Export Stage 2 cross-critique packet. |
| `frame prepare-jail` | `--topic <slug> --jail-dir <path>` | Provision isolated jail with `BRIEF.md`, parent candidate, and `meta.env`. |
| `frame extract-code` | `--candidate <f> --backup <f> --stdout-log <f>` | Extract modified candidate from in-place edits or worker stdout. |
| `frame smoke-check` | `--topic <slug> --candidate <path>` | Run Gate 0 AST duplicate check, syntax validation, and runtime projection guard. |
| `frame step` | `--topic <slug> --candidate <p> [--edge-type blend_ensemble]` | Evaluate candidate across Gate 1 dev diagnostic and Gate 2 confirmatory holdout. |
| `frame inject` | `--topic <slug> --parent <id> --spec "<text>"` | Enqueue high-priority human/frontier directive ticket. |
| `frame abort` | `--topic <slug> --node <id> [--inject-spec "<text>"]` | Abort node mid-run without spending alpha; queue replacement spec. |
| `frame status` | `--topic <slug>` | Display JSON status of MCTS tree, Incumbent $M^*$, and LORD-3 accounting. |
| `frame vault-eval` | `--topic <slug>` | Evaluate reigning Incumbent on sealed untouched Vault-Holdout split. |

---

## 📂 Repository Layout

```
~/frame/
├── frame/                         # Core FRAME-AR package
│   ├── bench/                     # Benchmark management & qualification
│   │   ├── registry.py            # Committee reviews, lock verification & state
│   │   ├── templates.py           # RL control & Kaggle tabular benchmark templates
│   │   └── verifier.py            # 7-point meta-verification & dynamic timeout calibration
│   ├── brain/                     # Tree search & cognitive routing
│   │   └── tree.py                # MCTS CampaignTree, ancestor retirement & exhaustion
│   ├── eval/                      # Evaluation & statistical accounting
│   │   ├── funnel.py              # 3-gate FunnelEvaluator (Gate 0, Gate 1, Gate 2)
│   │   ├── sandbox.py             # CPython audit hook sandbox & AST select_action binding
│   │   └── statistics.py          # LORD-3 controller, Riemann Zeta decay & ROPE bootstrap
│   ├── swarm/                     # Context briefing & agent swarm
│   │   └── brief.py               # Context brief generator & operator assignments
│   ├── utils/                     # Canonical serialization & hashing
│   │   └── canonical_json.py      # RFC 8785 Canonical JSON & SHA-256 digests
│   ├── cli.py                     # Unified zero-mock CLI entrypoint
│   └── schemas.py                 # Pydantic domain models, node records & reports
├── tests/                         # Comprehensive verification test suite
│   ├── test_bash_hypervisor_and_monitor.py   # Hypervisor, monitor & IPC tests
│   ├── test_frame_v2_e2e.py                  # End-to-end pipeline & sandbox tests
│   ├── test_v2_4_expressivity_and_funnel.py  # Expressivity & Gate-1 incumbent filter
│   └── test_v2_5_scaling_and_convergence.py  # Timeout calibration & convergence tests
├── skills/                        # Antigravity agent skill definitions
│   └── frame-ar/SKILL.md          # Official FRAME-AR v2 agent skill specification
├── frame-orchestrate.sh           # Top-level static UNIX Bash hypervisor
├── frame-mon.py                   # Out-of-band live terminal monitor & injector
├── pyproject.toml                 # Package configuration (version 2.5.0)
├── CHANGELOG.md                   # Complete semantic version release history
├── ARCHIVE_V1_MANIFEST.md         # Preserved state manifest for legacy v1 release
├── README.md                      # Primary documentation & architecture guide
└── LICENSE                        # MIT License
```

---

## 🧪 Verification & Testing

FRAME-AR v2.5 maintains a 100% test pass rate across 29 unit, integration, and end-to-end test suites:

```bash
pytest -v
```

```
============================= test session starts ==============================
platform linux -- Python 3.14.9, pytest-9.2.0, pluggy-1.7.0
rootdir: /home/sparsh/frame
configfile: pyproject.toml

tests/test_bash_hypervisor_and_monitor.py .........                      [ 31%]
tests/test_frame_v2_e2e.py ...                                           [ 41%]
tests/test_v2_4_expressivity_and_funnel.py .......                       [ 65%]
tests/test_v2_5_scaling_and_convergence.py ..........                    [100%]

============================== 29 passed in 364.13s ==============================
```

Test coverage includes:
- **Sandbox Security**: Mandatory `sys.addaudithook` preventing file, network, stack, and permission escapes.
- **Statistical Rigor**: Polynomial Riemann Zeta LORD-3 decay, alpha wealth preservation, and Common Random Numbers paired-seed testing.
- **Hypervisor Lifecycle**: Ephemeral `/tmp` jail allocation, timeout enforcement, Gate 0 retry loops, and code extraction.
- **Tree Convergence**: Ancestor retirement under varying ROPE thresholds, branch exhaustion penalties, and early-stop convergence detection.

---

## 🏛️ Legacy v1 Archive

The original monolithic v1 architecture of FRAME-AR is permanently archived:
- **Git Commit**: `1a55a63`
- **Git Tag**: `v1.0.0-archive`
- **Preservation Branch**: `archive/v1-main`
- **Verification Status**: 793 passed unit and integration tests.
- **Details**: See [ARCHIVE_V1_MANIFEST.md](ARCHIVE_V1_MANIFEST.md) for restoration procedures.

---

## 📄 License

FRAME-AR is open-source software licensed under the [MIT License](LICENSE).
