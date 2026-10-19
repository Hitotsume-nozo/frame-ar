# FRAME-AR: Foundation for Rigorous Autonomous & Mathematical Experimentation — Auto-Research

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests Passing](https://img.shields.io/badge/tests-793%20passed-brightgreen.svg)](tests/)
[![FDR Control](https://img.shields.io/badge/FDR%20Control-LORD%203%20Streaming-blueviolet.svg)](#4-rigorous-statistical-superiority--streaming-lord-3-fdr)
[![Cryptographic CAS](https://img.shields.io/badge/Evidence%20Chain-Merkle%20CAS%20%28RFC%208785%29-orange.svg)](#10-cryptographic-merkle-cas-evidence-binding)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **FRAME-AR** is an autonomous scientific discovery and empirical auto-research platform. It pairs **Agentic Swarm Thinking & Recursive Evolutionary MCTS Search** (inspired by AlphaEvolve and AI-Scientist v2) with a **Production-Grade Mathematical & Statistical Verification Engine** (tiered symbolic proofs, adversarial red-teaming, streaming false discovery rate control, edge-indexed residual routing, and cryptographic claim-evidence binding).

---

## 📑 Table of Contents

- [Core Philosophy & Architecture](#-core-philosophy--architecture)
- [System Architecture Diagram](#-system-architecture-diagram)
- [The 10 Distinctive Subsystems](#-the-10-distinctive-subsystems)
  - [1. Cognitive Swarm & Information Governor](#1-cognitive-swarm--information-governor)
  - [2. Dual-Key Novelty Contract](#2-dual-key-novelty-contract)
  - [3. Tiered Symbolic-Numeric Math Engine (L0–L2)](#3-tiered-symbolic-numeric-math-engine-l0l2)
  - [4. Rigorous Statistical Superiority & Streaming LORD 3 FDR](#4-rigorous-statistical-superiority--streaming-lord-3-fdr)
  - [5. Edge-Indexed FDR & The Blend Edge Protocol](#5-edge-indexed-fdr--the-blend-edge-protocol)
  - [6. Adversarial Red-Team Suite & Anti-Reward Hacking](#6-adversarial-red-team-suite--anti-reward-hacking)
  - [7. Numerical TAPAScore Formulation](#7-numerical-tapascore-formulation)
  - [8. Three-Tier Compute Dispatch Router](#8-three-tier-compute-dispatch-router)
  - [9. Persistent Dual-Sink Epistemic Ledgers](#9-persistent-dual-sink-epistemic-ledgers)
  - [10. Cryptographic Merkle CAS Evidence Binding](#10-cryptographic-merkle-cas-evidence-binding)
- [Repository Layout](#-repository-layout)
- [Installation & Setup](#-installation--setup)
- [CLI Reference & Usage](#-cli-reference--usage)
- [Empirical Verification Matrix (793 Tests)](#-empirical-verification-matrix-793-tests)
- [Master Documentation & Publications](#-master-documentation--publications)
- [License](#-license)

---

## 🏛️ Core Philosophy & Architecture

Modern automated machine learning (AutoML) tools routinely suffer from three structural pathologies:
1. **The Epistemic Closed-World Trap**: Mutating code purely via internal LLM priors without grounding against external scientific literature causes agents to hallucinate repetitive, degenerate failure modes.
2. **Pseudoreplication & P-Hacking**: Running dozens of empirical trials and picking the peak score on static validation splits guarantees false discoveries through selection bias ($p$-hacking) and rank barriers.
3. **The Blend Conflation Illusion**: Declaring ensemble/stacking layers as "discoveries" merely because they outperform a naive baseline, despite adding massive parameter bloat with zero statistically verifiable advantage over the best single constituent model.

**FRAME-AR solves these pathologies from first principles**:
- **Closed-Loop Recursive Thinking**: Failures extract post-mortem error telemetry that translates directly into targeted scientific queries, re-entering Phase 1 literature ingestion.
- **Online False Discovery Rate (FDR) Control**: Streaming **LORD 3** dynamically allocates significance thresholds $\alpha_t$, guaranteeing $\mathbb{E}[\text{FDR}_{\mathcal{T}}] \le \alpha$ for arbitrary stopping times $\mathcal{T}$.
- **Edge-Indexed Residual Routing**: Hypotheses are modeled as a directed acyclic graph (DAG) where every edge carries an independent two-stage statistical test with pre-registered margins ($\epsilon_{\text{ROPE}}$).
- **The Blend Edge Protocol**: An ensemble cannot be certified without passing a two-stage screen against the *incumbent best single model*.
- **Cryptographic Provenance**: Every empirical claim links to an unbroken SHA-256 Merkle content-addressable storage (CAS) root CID under RFC 8785 Canonical JSON.

---

## 🔄 System Architecture Diagram

```
   ┌─────────────────────────────────────────────────────────────────────────────────────────┐
   │                                                                                         │
   │  ┌──────────────────────────────────────────────────────────┐                           │
   └─►│  PHASE 1: TARGETED SWARM THINKING & EXTERNAL INGESTION   │                           │
      │  • Ingest domain profile & theoretical noise floor       │                           │
      │  • Active web mining for external SOTA & failure cures   │                           │
      │  • Information Governor evidence audit G(s) >= 0.40      │                           │
      │  • Exploration entropy check H_phase1 >= 0.75 ln(4)      │                           │
      │  • Spawn orthogonal swarm specialist lanes (A ⊕ B ⊕ D ⊕ F)│                          │
      └────────────────────────────┬─────────────────────────────┘                           │
                                   │ Swarm proposals with branch provenance                  │
                                   ▼                                                         │
      ┌──────────────────────────────────────────────────────────┐                           │
      │    PHASE 2: COMPOSITIONAL SYNTHESIS & PRE-FLIGHT PROOFS  │                           │
      │  • Structural AST Falsification Memory (O(1) hard block) │                           │
      │  • Branch-isolated semantic memory & rebuttal filter     │                           │
      │  • Recombination Legality Grammar (A ⊕ B ⊕ D typing)     │                           │
      │  • Dual-Key Novelty Contract (d_lit > 0.15, d_func > 0.10)│                          │
      │  • Tiered Math Verification (L0 SymPy, L1 IBP, L2 IBnB)  │                           │
      └────────────────────────────┬─────────────────────────────┘                           │
                                   │ Cleared for compute                                     │
                                   ▼                                                         │
      ┌──────────────────────────────────────────────────────────┐                           │
      │    PHASE 3: MULTI-TIER EMPIRICAL EXECUTION & ORACLES     │                           │
      │  • 5-Fold Cross-Validation with strict per-fold hygiene  │                           │
      │  • Common Random Numbers (CRN) paired-seed testing (k>=5)│                           │
      │  • Two-Stage Screening (k=3 futility -> k* confirmatory) │                           │
      │  • Edge-Indexed FDR under streaming LORD 3 (alpha_t > 0) │                           │
      │  • The Blend Edge Protocol (screen blends vs incumbent)  │                           │
      │  • Adversarial Red-Team Gate (5 null models, variance)   │                           │
      │  • Compute Hierarchical Oracle Reward & backpropagate    │                           │
      └────────────────────────────┬─────────────────────────────┘                           │
                                   │                                                         │
        ┌──────────────────────────┴──────────────────────────┐                              │
        │                                                     │                              │
[If Target Not Reached]                                 [If Target Reached]                  │
        │                                                     │                              │
        ▼                                                     ▼                              │
┌─────────────────────────────┐                       ┌─────────────────────────────┐        │
│ PHASE 4: EVOLUTIONARY LOOP  │                       │ PHASE 5: EVIDENCE BINDING   │        │
│ • Extract residual errors   │                       │ • Compute Numerical TAPAScore│       │
│ • Record failure signature  │                       │ • Assemble Merkle CAS CID   │        │
│ • Synthesize targeted       │                       │ • S_LB Leaderboard check    │        │
│   external search queries   │                       │ • Assert |L_CV - S_LB| <= τ │        │
│ • Increment recursion depth │                       │ • Emit GoalMark Certificate │        │
└──────────────┬──────────────┘                       └─────────────────────────────┘        │
               │                                                                             │
               └──────────────► Loop back to Phase 1 (Targeted External Ingestion) ──────────┘
```

---

## ⚙️ The 10 Distinctive Subsystems

### 1. Cognitive Swarm & Information Governor
- **Role Specialization**: Divides hypothesis generation into four orthogonal cognitive lanes: `Architecture` ($A$), `Objectives` ($B$), `Optimization` ($D$), and `Features` ($F$).
- **Evidence Hierarchy Standard**: The `InformationGovernor` (`frame.agents.governor`) audits and grades all ingested literature before proposals enter synthesis:
  $$G(s) = w_{\text{hier}}(T(s)) \cdot \left[0.40 \cdot r(s) + 0.40 \cdot m(s) + 0.20 \cdot v(s)\right] \ge \tau_{\text{gov}} = 0.40$$
  where $w_{\text{hier}}$ weights peer-reviewed Cochrane/Journal publications ($1.0$), Preprints ($0.7$), Forums ($0.3$), and Unverified sources ($0.1$).
- **Shannon Exploration Entropy**: Prevents greedy premature convergence into local optima by requiring:
  $$H_{\text{phase1}} = -\sum_{k=1}^K p_k \ln(p_k) \ge 0.75 \ln(4) \approx 1.0397 \text{ nats}$$

### 2. Dual-Key Novelty Contract
Pre-confirmatory gating engine (`frame.verification.novelty`) that **consumes zero LORD alpha wealth**:
- **Key 1 (Literature Distance $d_{\text{lit}} > 0.15$)**: Evaluates unigram/bigram cosine distance and Szymkiewicz-Simpson containment against prior art to detect surface-form variable renames.
- **Key 2 (Functional Prediction Manifold Divergence $d_{\text{func}} > 0.10$)**: Computes uncentered cosine angle $\rho$ between holdout residual vectors:
  $$d_{\text{func}} = 1 - |\rho|, \quad \rho = \frac{\mathbf{r}_{\text{base}} \cdot \mathbf{r}_{\text{cand}}}{\|\mathbf{r}_{\text{base}}\|_2 \|\mathbf{r}_{\text{cand}}\|_2}$$

### 3. Tiered Symbolic-Numeric Math Engine (L0–L2)
Unified formal math verifier (`frame.verification.math_engine`) evaluating candidates through dimensional compression:
- **Level 0 (SymPy Analytic, $<200$ ms)**: Evaluates closed-form singularities, denominators $\text{denom}(x) \neq 0$, continuous differentiability, and asymptotic limits $\lim_{x \to \pm \infty} f(x)$.
- **Level 1 (Interval Bound Propagation, $<1.5$ s)**: Executes symbolic tensor interval propagation over AST DAGs on $[-B, B]$ and computes Lanczos spectral condition numbers $\kappa(W) \le 10^7$.
- **Level 2 (Interval Branch-and-Bound, $<8$ s)**: Recursive box bisection using `mpmath` multi-precision interval arithmetic and L-BFGS-B boundary exploration, producing formal mathematical proof or explicit counterexample coordinates $x_{\text{fail}}$.

### 4. Rigorous Statistical Superiority & Streaming LORD 3 FDR
- **Common Random Numbers (CRN)**: Evaluates candidates across paired seeds $k \ge 5$, isolating algorithmic improvements from data-split variance:
  $$D_i = Y_{\text{cand}}^{(i)} - Y_{\text{base}}^{(i)} \quad (\text{or } Y_{\text{base}}^{(i)} - Y_{\text{cand}}^{(i)} \text{ for minimization})$$
- **Region of Practical Equivalence (ROPE)**: Tests directional superiority beyond an operational margin $\epsilon_{\text{ROPE}}$:
  $$H_0: \mu_D \le \epsilon_{\text{ROPE}} \quad \text{vs.} \quad H_1: \mu_D > \epsilon_{\text{ROPE}}, \quad t_{\text{ROPE}} = \frac{\bar{D} - \epsilon_{\text{ROPE}}}{S_D / \sqrt{k}}$$
- **Online Streaming LORD 3**: Evaluates dynamic thresholds with Riemann zeta normalizing constant $\zeta(1.6) \approx 2.285766$:
  $$\alpha_t = W_0 \gamma_t + (\alpha - W_0) \gamma_{t - \tau_1}\mathbf{1}_{\{\tau_1 < t\}} + \sum_{j \ge 2 : \tau_j < t} \alpha \gamma_{t - \tau_j}$$

### 5. Edge-Indexed FDR & The Blend Edge Protocol
- **Directed Graph Routing**: Every statistical test indexes a specific edge $e = (u \to v)$ in `StatisticalVerificationReport`:
  - `edge_id`: Unique transition string (`edge_gen0_to_gen1_lgbm`, `edge_incumbent_to_blend`).
  - `edge_type`: `"atomic_transition"`, `"blend_ensemble"`, or `"cumulative_campaign"`.
  - Pre-registered per-edge margins: $\epsilon_{\text{ROPE}} = 0.005$ for architectural leaps, $\epsilon_{\text{ROPE}} = 0.001$ for stacking refinements.
  - Distinct $\alpha_t$ consumption per edge (telemetry conflation strictly prohibited).
- **The Blend Edge Protocol**: Ensembles cannot claim discovery against a naive root baseline (Gen 0); they **must pass two-stage screening against the incumbent best single constituent model**. If sub-threshold, the blend is rejected as parameter bloat.

### 6. Adversarial Red-Team Suite & Anti-Reward Hacking
- **5-Null Model Degeneracy Battery** (`frame.verification.red_team`): Candidates must outperform Zero ($f \equiv 0$), Constant ($\bar{y}$), Random ($\mathcal{N}(0, I)$), Identity ($f(x) = x$), and Label-Permuted models by $\Delta_{\text{null}} > \epsilon_{\text{robust}}$.
- **Variance Preservation Ratio**: Asserts output standard deviation satisfies $s_{\hat{y}} \ge 0.50 s_y$ to prevent trivial mean collapse.
- **Causal Lookahead Gradient Probe**: Enforces future-token gradient $\frac{\partial y_t}{\partial x_{t+k}} \equiv 0$ in autoregressive sequence architectures.

### 7. Numerical TAPAScore Formulation
The **Targeted Analytical Performance Alignment Score** (`frame.verification.tapa`) computes a deterministic scalar $\in [0, 100]$:
$$\text{TAPAScore} = 100 \times R_{\text{proof}} \times R_{\text{legality}} \times \left[ 0.50 \cdot \tanh(\max(0, R_{\text{stat}})) + 0.25 \cdot \eta_{\text{robust}} + 0.25 \cdot \eta_{\text{var}} \right]$$
- $R_{\text{proof}}, R_{\text{legality}} \in \{0, 1\}$: Hard gates for formal proof and recombination typing.
- $R_{\text{stat}}$: Statistical discovery power $\max(0, \bar{D} - \epsilon_{\text{ROPE}}) \cdot \ln(1 + \alpha_t / \max(p_{\text{ROPE}}, 10^{-9}))$.
- $\eta_{\text{robust}}$: Clipped margin over best adversarial null model.
- $\eta_{\text{var}}$: Variance preservation penalty function preventing variance collapse or explosion.
- **Modality B (SAEs)**: Incorporates hard gates for dead feature ratio $<1\%$, loss recovery $R \ge 80\%$, and $L_0$ target sparsity alignment.

### 8. Three-Tier Compute Dispatch Router
Deterministic Program-of-Thought (PoT) dry-run profiler (`frame.pot.tracer`) measures forward/backward ATen FLOPs via PyTorch `FlopCounterMode` ($B=1$) and routes compute:
- **Tier 1 (Host Local)**: $F_{\text{total}} \le 3 \times 10^{14}$ FLOPs, Peak VRAM $\le 3$ GB, runtime $\le 300$ s. Memory fraction capped at 75% with expandable CUDA segments.
- **Tier 2 (Kaggle Remote REST)**: $3 \times 10^{14} < F_{\text{total}} \le 10^{17}$ FLOPs, Peak VRAM $\le 14.5$ GB (Tesla P100 / Dual T4). Dispatched via self-extracting Base64 payload ([`bundler.py`](frame/runners/kaggle/bundler.py)) over a zero-dependency REST client with a 7-day 30h quota ledger.
- **Tier 3 (Manual Air-Gapped Capsule)**: $F_{\text{total}} > 10^{17}$ FLOPs, multi-GPU FSDP runs. Generates an air-gapped Docker bundle ([`capsule.py`](frame/runners/manual/capsule.py)) with deterministic host micro-replay verification ($|\Delta \mathcal{L}_{\text{replay}}| \le 10^{-4}$).

### 9. Persistent Dual-Sink Epistemic Ledgers
Persistent state node (`frame.schemas.learning_state`) maintains queryable audit event streams:
- `governor_event_logs`: Ingests `source_vetting`, `exploration_entropy_audit`, and `novelty_audit` events.
- `executioner_event_logs`: Ingests `upper_envelope_update`, `watchdog_monotonicity`, `stagnation_check` (vanishing gradients, loss plateaus, circular ASTs), and `ast_falsification_blocked` events.

### 10. Cryptographic Merkle CAS Evidence Binding
Every verified claim is hashed using **RFC 8785 JSON Canonicalization Scheme (JCS)**:
$$\text{Merkle Root CID} = \text{SHA256}(\text{Hypothesis Hash} \parallel \text{Execution Receipt Hash} \parallel \text{Statistical Report Hash})$$
Yielding an unbroken audit trail:
$$\text{Scientific Claim} \longrightarrow \text{Report}_{\text{CID}} \longrightarrow \text{Receipt}_{\text{CID}} \longrightarrow \text{Code}_{\text{CID}} \longrightarrow \text{Dataset}_{\text{CID}}$$

---

## 📁 Repository Layout

```
frame/
├── frame/                              # Core Framework Package
│   ├── agents/                         # Cognitive Swarm Roles & Bridges
│   │   ├── agy_bridge.py               # Antigravity CLI swarm coordinator
│   │   └── governor.py                 # Information Governor evidence grading & entropy
│   ├── brain/                          # Search, Planning & Experimental Design
│   │   ├── campaign.py                 # Autonomous closed-loop campaign runner
│   │   ├── executioner.py              # Watchdog monitoring & orthogonal warm restarts
│   │   ├── experimental_design.py      # Power analysis, two-stage screening, quad-baselines
│   │   ├── state.py                    # MCTS nodes & SearchAccountingLedger
│   │   └── thinking.py                 # Adaptive UCT, AST memory, recombination grammar
│   ├── feedback/                       # Visualization & Incentive Surfaces
│   │   └── visualization.py            # SVG contour plots & 24-bit ANSI terminal heatmaps
│   ├── pot/                            # Program of Thought Tracing
│   │   └── tracer.py                   # FlopCounterMode profiler & compute router
│   ├── runners/                        # Hardware Execution Tiers
│   │   ├── kaggle/                     # Native REST client, bundler, quota ledger
│   │   └── manual/                     # Air-gapped capsule generator & replay verifier
│   ├── schemas/                        # Pydantic v2 Production Data Contracts
│   │   ├── capsule.py                  # Capsule manifests & replay verification schemas
│   │   ├── compute.py                  # ExecutionTier & ComputeProfile
│   │   ├── hypothesis.py               # Sealed seed commitments & Hypothesis schema
│   │   ├── learning_state.py           # SwarmLearningStateNode & dual-sink event ledgers
│   │   ├── novelty.py                  # DualKeyNoveltyContract results & verdicts
│   │   ├── receipt.py                  # HardwareTelemetry & ExecutionReceipt
│   │   └── verification.py             # StatisticalVerificationReport & PairedSeedMetrics
│   ├── utils/                          # Utilities
│   │   └── canonical_json.py           # RFC 8785 Canonical JSON serializer
│   ├── verification/                   # Formal Mathematics & Statistics
│   │   ├── domain_evidence.py          # Data provenance & ethical standards
│   │   ├── harness.py                  # Decoupled isolated verifier harness
│   │   ├── invariants.py               # Physics, biology & boundary condition plugins
│   │   ├── math_engine.py              # Tiered symbolic-numeric verifier (L0, L1, L2)
│   │   ├── novelty.py                  # Dual-key novelty contract engine
│   │   ├── red_team.py                 # Adversarial red-team suite (5 null models)
│   │   ├── statistics.py               # CRN Paired ROPE t-test & LORD 3 streaming FDR
│   │   └── tapa.py                     # Numerical TAPAScore formulations (Modality A & B)
│   └── cli.py                          # Unified CLI entrypoint
├── docs/                               # Master Documentation & Publications
│   ├── FRAME_AR_SYSTEM_REPORT.pdf      # 12-page compiled LaTeX publication report
│   ├── FRAME_AR_SYSTEM_REPORT.tex      # LaTeX source code with TikZ diagrams
│   └── FRAME_AR_SYSTEM_REPORT.toc      # Table of contents
├── skills/                             # Antigravity & Agentic Skills
│   └── frame-ar/
│       └── SKILL.md                    # Official FRAME-AR skill definition
├── tests/                              # Comprehensive Test Suite (793 tests)
│   ├── test_anti_reward_hacking.py
│   ├── test_brain.py
│   ├── test_branch_isolation_and_recursive_loop.py
│   ├── test_cli.py
│   ├── test_domain_evidence.py
│   ├── test_domain_evidence_challenger.py
│   ├── test_e2e_cognitive_tiers.py     # 218 E2E cognitive tier tests
│   ├── test_empirical_m5_challenger.py
│   ├── test_executioner.py
│   ├── test_executioner_mcts_simulation_challenger.py
│   ├── test_experimental_design.py
│   ├── test_framework.py
│   ├── test_governor.py
│   ├── test_kaggle_client.py
│   ├── test_learning_state.py
│   ├── test_learning_state_challenger.py
│   ├── test_learning_state_stress.py
│   ├── test_math_engine.py
│   ├── test_novelty_adversarial.py
│   ├── test_novelty_challenger.py
│   ├── test_novelty_contract.py
│   ├── test_preflight_verifications.py # Edge-indexed FDR, blend edge, TAPAScore
│   └── test_visualization.py
├── FRAME_AR_SPECIFICATION.md           # Master System Specification
├── PROJECT.md                          # Cognitive Overhaul Architecture Specification
├── TEST_INFRA.md                       # E2E Test Suite Specification
├── pyproject.toml                      # PEP 621 Build & Package Configuration
└── README.md                           # Master Documentation & Quickstart
```

---

## 🚀 Installation & Setup

### Requirements
- Python $\ge 3.10$
- PyTorch $\ge 2.0.0$
- NumPy, SciPy, Pydantic $\ge 2.0$, SymPy, mpmath

### Install from Source
```bash
git clone https://github.com/Hitotsume-nozo/frame-ar.git
cd frame-ar
pip install -e .
```

### Install with Developer Dependencies
```bash
pip install -e ".[dev]"
```

---

## 💻 CLI Reference & Usage

The unified CLI entrypoint `frame` exposes all cognitive and verification capabilities:

```bash
# 1. Run an end-to-end autonomous research campaign
frame run --goal "Optimize Tabular California Housing" --max-iterations 5 --target-improvement 0.10

# 2. Export subagent role specifications for Antigravity (agy)
frame swarm --goal "Optimize Transformer Attention" --export-agy-specs

# 3. Formally verify mathematical stability of an activation or loss formula
frame verify-math --expr "x / (1 + torch.exp(-x))" --level2

# 4. Audit a PyTorch candidate model against 5 adversarial null models
frame red-team

# 5. Inspect metric incentive surface and display a 24-bit ANSI terminal heatmap
frame inspect-metric --name cross_entropy_loss

# 6. Inspect current MCTS tree depth, stage yields, and active LORD alpha wealth
frame status

# 7. Check Kaggle GPU quota and remote worker status
frame kaggle --status
```

---

## 🧪 Empirical Verification Matrix (793 Tests)

The framework includes a comprehensive 22-suite test matrix with **793 passing tests**:

```bash
pytest tests/
```

```
============================= test session starts ==============================
platform linux -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/sparsh/frame
configfile: pyproject.toml
collected 793 items

tests/test_anti_reward_hacking.py .......                                [  0%]
tests/test_brain.py ..............                                       [  2%]
tests/test_branch_isolation_and_recursive_loop.py ........               [  3%]
tests/test_cli.py ..............                                         [  5%]
tests/test_domain_evidence.py .......................................... [ 10%]
tests/test_domain_evidence_challenger.py ...............                 [ 14%]
tests/test_e2e_cognitive_tiers.py ...................................... [ 41%]
tests/test_empirical_m5_challenger.py .................................. [ 48%]
tests/test_executioner.py ............................................   [ 53%]
tests/test_executioner_mcts_simulation_challenger.py ................... [ 57%]
tests/test_experimental_design.py .......                                [ 58%]
tests/test_framework.py ..........                                       [ 59%]
tests/test_governor.py ........................                          [ 62%]
tests/test_kaggle_client.py ................................             [ 66%]
tests/test_learning_state.py ...................................         [ 71%]
tests/test_learning_state_challenger.py ......................           [ 73%]
tests/test_learning_state_stress.py ...........                          [ 75%]
tests/test_math_engine.py ..........                                     [ 76%]
tests/test_novelty_adversarial.py ...................................... [ 89%]
tests/test_novelty_challenger.py ......................                  [ 92%]
tests/test_novelty_contract.py .......................................   [ 97%]
tests/test_preflight_verifications.py .............                      [ 98%]
tests/test_visualization.py ........                                     [100%]

============================= 793 passed in 22.73s =============================
```

---

## 📚 Master Documentation & Publications

| Document | Format | Location | Description |
| :--- | :---: | :--- | :--- |
| **Comprehensive System Report** | **PDF** | [`docs/FRAME_AR_SYSTEM_REPORT.pdf`](docs/FRAME_AR_SYSTEM_REPORT.pdf) | 12-page compiled publication report with TikZ architecture diagrams, proofs, and hardware calibration. |
| **System Report Source** | **LaTeX** | [`docs/FRAME_AR_SYSTEM_REPORT.tex`](docs/FRAME_AR_SYSTEM_REPORT.tex) | Complete compilable LaTeX source code. |
| **Architecture Specification** | **Markdown** | [`FRAME_AR_SPECIFICATION.md`](FRAME_AR_SPECIFICATION.md) | Full mathematical specification, CRN paired seeds, ROPE, LORD 3, edge-indexed FDR, and Merkle CAS. |
| **FRAME-AR Agent Skill** | **Markdown** | [`skills/frame-ar/SKILL.md`](skills/frame-ar/SKILL.md) | Official agent skill instructions, non-deviation rules, and execution templates. |
| **Cognitive Architecture Spec** | **Markdown** | [`PROJECT.md`](PROJECT.md) | Technical specification for Information Governor, Novelty Contracts, and Swarm roles. |
| **E2E Test Infrastructure** | **Markdown** | [`TEST_INFRA.md`](TEST_INFRA.md) | Test architecture detailing all 20 cognitive features across 4 verification tiers. |

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.
