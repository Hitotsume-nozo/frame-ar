# Test Infrastructure Specification: FRAME-AR Cognitive Research Overhaul

## 1. Overview & Architecture
This document specifies the comprehensive opaque-box End-to-End (E2E) testing infrastructure for the FRAME-AR Cognitive Research Overhaul.
The test suite validates all 20 cognitive features spanning Requirements R1–R5 across four distinct verification tiers:
- **Tier 1: Feature Isolation Coverage** (>=5 test cases per feature covering happy-path behavioral contracts = >=100 tests)
- **Tier 2: Boundary & Corner Cases** (>=5 test cases per feature covering degenerate inputs, strict inequalities, zero-variance, and limits = >=100 tests)
- **Tier 3: Cross-Feature Interactions** (pairwise lifecycle integrations: Governor → Novelty, Novelty → Learning State, Executioner → Restart, Invariants → L0–L2 math proofs = >=12 tests)
- **Tier 4: Real-World Research Scenarios** (complete multi-agent discovery campaigns with synthetic data, adversary simulation, and compute scaling = >=6 tests)

Total E2E test inventory: **218 test cases** in `tests/test_e2e_cognitive_tiers.py`.

```
========================================================================================
                          FRAME-AR E2E COGNITIVE TEST SUITE
========================================================================================
  [Tier 1: Feature Isolation]   --> 100 tests (20 features x 5 happy-path contracts)
  [Tier 2: Boundary & Corners]  --> 100 tests (20 features x 5 edge/degenerate cases)
  [Tier 3: Cross-Feature]       --> 12 tests (Pipeline interactions & pairwise contracts)
  [Tier 4: Real-World Scenarios]--> 6 tests (Autonomous discovery campaigns & stress runs)
========================================================================================
  Baseline Non-Regression: 110 tests across existing 9 test suites preserved intact
  Total Test Suite Capacity: 328 tests
========================================================================================
```

---

## 2. Feature Coverage Matrix (20 Features)

| Feature # | Category | Feature Name | Behavioral Contract & Invariant | Module Location |
|---|---|---|---|---|
| **F01** | R1 Swarm | Information Governor Evidence Vetting | Audits and grades ingested literature sources; accepts iff $G(s) \ge 0.40$ | `frame.brain.governor` / `frame.agents.governor` |
| **F02** | R1 Swarm | High-Entropy Exploratory Mode | Enforces Phase 1 lane exploration entropy $H_{\text{phase1}} \ge 0.75 \ln 4 \approx 1.0397$ | `frame.brain.governor` |
| **F03** | R1 Swarm | Cognitive Role Spec Generator | Generates Antigravity subagent specs for Governor, Analyzer, Reviewer, Executioner, Specialists | `frame.agents.agy_bridge` |
| **F04** | R2 Novelty | Literature Graph Proximity | Computes cosine distance $d_{\text{lit}} = 1 - \max \text{sim}$; accepts iff $d_{\text{lit}} > 0.15$ | `frame.verification.novelty` |
| **F05** | R2 Novelty | Functional Manifold Divergence | Computes holdout residual non-collinearity $d_{\text{func}} = 1 - \|\rho\|$; accepts iff $d_{\text{func}} > 0.10$ | `frame.verification.novelty` |
| **F06** | R2 Novelty | Automated Dual-Key Novelty Gate | Combines $d_{\text{lit}} > 0.15 \land d_{\text{func}} > 0.10$; strictly rejects surface rephrasing; 0 alpha spent | `frame.verification.novelty` |
| **F07** | R3 State | Persistent Swarm Learning State Node | Queryable artifact recording hypotheses, quality scores, epistemic entropy, and trajectories | `frame.brain.learning_state` |
| **F08** | R3 State | Epistemic Entropy Tracker | Computes Shannon entropy over quality quantile bins and lane distributions | `frame.brain.learning_state` |
| **F09** | R3 State | Test-Time Compute Scaling Upper Envelope | Tracks upper envelope $\mathcal{E}(N) = \max_{n \le N} Q(n)$; verifies monotonic non-decreasing property | `frame.brain.learning_state` |
| **F10** | R4 Executioner| Epistemic Stagnation Detector | Monitors $M=3$ stacks for vanishing gradients ($\le 10^{-6}$), plateau ($\le 10^{-4}$), circular AST ($J > 0.85$) | `frame.brain.executioner` |
| **F11** | R4 Executioner| Radical Architecture Safeguard | Suppresses pruning for radical early-stage architectures ($\text{depth} \le 1 \land \text{visits} \le 2$) | `frame.brain.executioner` |
| **F12** | R4 Executioner| Orthogonal Warm-Restart Controller | Injects fresh orthogonal priors (`arch` $\to$ `features`, `obj` $\to$ `opt`, `opt` $\to$ `arch`, `feat` $\to$ `obj`) | `frame.brain.executioner` |
| **F13** | R4 Executioner| Executioner Event Logging | Persists audit verdicts, pruned branches, and warm-restart events to Learning State Node | `frame.brain.executioner` |
| **F14** | R5 Evidence | Evidence Hierarchy Weighting System | Calculates composite score weighting Cochrane ($1.0$) > Preprint ($0.7$) > Forum ($0.3$) > Unverified ($0.1$) | `frame.verification.domain_evidence` |
| **F15** | R5 Evidence | Data Provenance & Ethics Tracker | Enforces permissive licensing, privacy budget $\epsilon \le 1.0$, and bias disparity $\le 0.10$ | `frame.verification.domain_evidence` |
| **F16** | R5 Evidence | Domain Invariant Plug-in Architecture | Modular registration of symbolic constraints (conservation laws, biological monotonicity, boundaries) | `frame.verification.domain_evidence` |
| **F17** | R5 Evidence | L0–L2 Math Verifier Invariant Binding | Binds symbolic constraints into tiered L0 SymPy, L1 IBP, and L2 IBnB proof engine | `frame.verification.domain_evidence` |
| **F18** | CLI System | Enhanced `frame status` Observability | Displays Learning State, compute scaling curve, Governor logs, and Executioner logs | `frame.cli` |
| **F19** | CLI System | Expanded `frame swarm --export-agy-specs`| Exports CLI JSON specifications for all 8 cognitive subagent roles | `frame.cli` |
| **F20** | Non-Regress | Statistical Screener Non-Regression | Preserves LORD-3 streaming FDR, two-stage screening, ROPE superiority, and FlopCounterMode | `frame.verification.statistics` |

---

## 3. Mathematical Specifications & Decision Oracles

### 3.1 Information Governor Grading Formula
$$G(s) = w_{\text{hier}}(T(s)) \cdot \left(0.40 \cdot r(s) + 0.40 \cdot m(s) + 0.20 \cdot v(s)\right)$$
Where:
- $w_{\text{hier}}(\text{cochrane}) = 1.0$
- $w_{\text{hier}}(\text{preprint}) = 0.7$
- $w_{\text{hier}}(\text{public\_forum}) = 0.3$
- $w_{\text{hier}}(\text{unverified}) = 0.1$
- $r(s) \in [0, 1]$ (relevance score)
- $m(s) \in [0, 1]$ (methodological rigor score)
- $v(s) \in \{0.0, 1.0\}$ (peer review flag)
- Decision Rule: `ACCEPTED` iff $G(s) \ge 0.40$.

### 3.2 Phase 1 Exploratory Entropy Formula
$$H_{\text{phase1}} = -\sum_{k=1}^K p_k \ln(p_k)$$
- $K = 4$ exploration lanes (`architecture`, `objectives`, `optimization`, `features`)
- $H_{\max} = \ln 4 \approx 1.38629$
- Decision Rule: `ENTROPY_SUFFICIENT` iff $H_{\text{phase1}} \ge 0.75 \ln 4 \approx 1.03972$.

### 3.3 Dual-Key Novelty Contract
- **Key 1 (Literature Distance)**:
  $$d_{\text{lit}} = 1.0 - \max_{p \in \mathcal{P}} (\mathbf{e}_h^\top \mathbf{e}_p) > 0.15$$
- **Key 2 (Functional Manifold Divergence)**:
  $$d_{\text{func}} = 1.0 - |\rho(\mathbf{r}_{\text{cand}}, \mathbf{r}_{\text{base}})| > 0.10$$
  Where $\mathbf{r} = \mathbf{y}_{\text{true}} - \hat{\mathbf{y}}$, $\rho$ is Pearson correlation.
- **Contract Gate Invariant**: Candidate is `APPROVED` iff $d_{\text{lit}} > 0.15 \land d_{\text{func}} > 0.10$.
- **Alpha Wealth Preservation**: Does not invoke `LORDController.record_result()`. Consumes $0.000$ alpha wealth.

### 3.4 Test-Time Compute Scaling Upper Envelope
$$\mathcal{E}(N) = \max_{1 \le n \le N} Q(n)$$
- Monotonic progression invariant: $\forall n_2 \ge n_1: \mathcal{E}(n_2) \ge \mathcal{E}(n_1)$.
- Compounding Gain: $\Delta \mathcal{E}(N) = \mathcal{E}(N) - \mathcal{E}(1) \ge 0$.

### 3.5 Epistemic Stagnation Detection
Window: recent $M=3$ steps in branch history.
1. **Vanishing Gradient**: $\max_{m \in [1, M]} \|\nabla \theta_m\|_2 < 1.0 \times 10^{-6}$.
2. **Loss Plateau**: $\frac{\max \mathcal{L} - \min \mathcal{L}}{\text{mean}(\mathcal{L})} < 1.0 \times 10^{-4}$.
3. **Circular AST**: Token Jaccard $J(T_{t-1}, T_t) = \frac{|T_{t-1} \cap T_t|}{|T_{t-1} \cup T_t|} > 0.85$.
- **Early-Stage Exemption**: If $\text{depth} \le 1 \land \text{visits} \le 2$, pruning is suppressed.

---

## 4. Test Runner & Verification Commands

### Standard Test Execution
```bash
# Run the entire test suite (baseline 110 tests + cognitive E2E suite)
python3 -m pytest tests/

# Run the cognitive E2E tier suite specifically with verbose output
python3 -m pytest tests/test_e2e_cognitive_tiers.py -v

# Run individual test tiers
python3 -m pytest tests/test_e2e_cognitive_tiers.py -k "tier1" -v
python3 -m pytest tests/test_e2e_cognitive_tiers.py -k "tier2" -v
python3 -m pytest tests/test_e2e_cognitive_tiers.py -k "tier3" -v
python3 -m pytest tests/test_e2e_cognitive_tiers.py -k "tier4" -v
```

### Progressive Testability Architecture
The E2E test suite adheres to progressive testability:
- Tests access modules through safe dynamic loaders (`get_governor_module()`, `get_novelty_module()`, etc.).
- During incremental milestone development (M1 $\to$ M6), tests for pending milestones are skipped cleanly without breaking test collection.
- Tests for already-implemented features (such as Feature 20, LORD-3 streaming FDR, two-stage screening, ROPE superiority, and FlopCounterMode) execute and pass immediately.
- Once a milestone implements its corresponding module, all its Tier 1–4 tests execute automatically against the live implementation code.
- At Milestone M7, 100% of tests execute with zero skips and zero failures.
