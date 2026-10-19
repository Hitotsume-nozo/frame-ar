# Project: FRAME-AR Cognitive Research Overhaul

## Architecture
The cognitive research overhaul transforms FRAME-AR into a novelty-guaranteed, compounding discovery platform inspired by Google Co-Scientist and AI-Scientist v2.
The enhanced architecture establishes a dual-track cognitive cycle:

```
[Phase 1: Cognitive Ingestion & Swarm Proposal]
  ├── Information Governor (audits/vets/grades literature & data provenance)
  ├── High-Entropy Exploratory Sampling (prevents greedy premature exploitation)
  ├── 4 Lane Specialists (Architecture, Objectives, Optimization, Features)
  └── Meta-Reviewer (cross-audits proposals & enforces methodological rigor)
             │
             ▼
[Phase 1.5: Dual-Key Novelty Contract Gate (Pre-Confirmatory)]
  ├── Key 1: Literature Graph Proximity (d_lit > tau_lit = 0.15 vs prior art)
  └── Key 2: Functional Prediction Manifold Divergence (d_func > epsilon_func = 0.10 vs baselines)
  └── Strict rejection of surface-form rephrased baselines (0 LORD alpha wealth used)
             │
             ▼
[Phase 2: Program-of-Thought Tracing & Domain Invariants]
  ├── Static AST Safety Check & PyTorch FX Tracing
  ├── FlopCounterMode Dry-Run Profiler (B=1) & Hardware Tier Routing
  └── Domain Invariant Contracts (Symbolic physical/biological laws via L0–L2 Math Verifiers)
             │
             ▼
[Phase 3: Multi-Tier Empirical Execution & Two-Stage Screening]
  ├── Tier 1 Local / Tier 2 Kaggle Execution
  ├── Stage 1 Futility Filter (k=3 seeds, abort if D_bar <= 0)
  └── Stage 2 Confirmatory Testing (k* seeds, ROPE superiority under LORD alpha_t)
             │
             ▼
[Phase 4: MCTS Backprop & Learning State Node Compounding]
  ├── Swarm Learning State Node (records hypothesis quality, epistemic entropy, trajectories)
  └── Test-Time Compute Scaling Curves (verifies monotonic upper envelope over budget N)
             │
             ▼
[Cross-Stack Watchdog: The Executioner & Orthogonal Warm-Restart]
  ├── Queries Swarm Learning State across M=3 recursion stacks
  ├── Detects Stagnation: ||grad theta||_2 <= 1e-6, loss curvature <= 1e-4, circular AST J > 0.85
  ├── Early-Stage Safeguard: Exemption for radical architectures (depth <= 1, visits <= 2)
  └── Triggers Orthogonal Warm Restart (injects fresh orthogonal lane priors)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Information Governor Evidence Vetting | Audits, vets, and grades ingested literature, benchmarks, and data sources before synthesis ($G(s) \ge 0.40$) | M1 | survey |
| 2 | High-Entropy Exploratory Mode | Enforces exploratory entropy in Phase 1 ($H \ge 0.75 \ln 4$) to prevent greedy premature exploitation | M1 | survey |
| 3 | Cognitive Role Specification Generator | Generates Antigravity subagent specifications for Governor, Proximity Analyzer, Meta-Reviewer, Executioner, and Lane Specialists | M1 | survey |
| 4 | Literature Graph Proximity Evaluator | Calculates semantic distance from candidate hypothesis to vetted literature graph ($d_{\text{lit}} > 0.15$) | M2 | survey |
| 5 | Functional Prediction Manifold Divergence | Measures non-collinearity between candidate residuals and baseline predictions on holdouts ($d_{\text{func}} > 0.10$) | M2 | survey |
| 6 | Automated Dual-Key Novelty Gate | Combines $d_{\text{lit}}$ and $d_{\text{func}}$ into unified gate strictly rejecting surface-form paraphrased baselines | M2 | survey |
| 7 | Persistent Swarm Learning State Node | Queryable persistent artifact tracking hypothesis quality, epistemic entropy, and trajectories across MCTS generations | M3 | survey |
| 8 | Epistemic Entropy Tracker | Computes Shannon entropy over hypothesis quality distribution and exploration lane allocations | M3 | survey |
| 9 | Test-Time Compute Scaling Upper Envelope | Tracks and verifies monotonic progression of the upper envelope of hypothesis quality over search budget $N$ ($\mathcal{E}(N) \ge \mathcal{E}(N-1)$) | M3 | survey |
| 10 | Epistemic Stagnation Detector across $M$ Stacks | Evaluates active branch history across $M=3$ stacks for vanishing gradients, loss plateau, and circular AST mutations | M4 | survey |
| 11 | Radical Early-Stage Architecture Safeguard | Prevents Executioner from prematurely killing radical early-stage architectures ($d \le 1, n \le 2$) | M4 | survey |
| 12 | Orthogonal Warm-Restart Controller | Triggers warm restart injecting fresh orthogonal lane priors into stagnant branches rather than unguided abort | M4 | survey |
| 13 | Executioner Event Logging | Records all audit decisions, pruned branches, and warm restart actions in Learning State Node | M4 | survey |
| 14 | Evidence Hierarchy Weighting System | Calculates composite evidence score weighting peer-reviewed Cochrane/journal ($1.0$) > pre-print ($0.7$) > forum ($0.3$) | M5 | survey |
| 15 | Data Provenance & Ethical Constraint Tracker | Verifies dataset provenance (CID/hash), license safety, privacy constraints ($\epsilon$), and bias disparity bounds | M5 | survey |
| 16 | Domain Invariant Plug-in Architecture | Modular interface allowing domain plugins to register symbolic constraints (conservation, monotonicity, boundary) | M5 | survey |
| 17 | L0–L2 Math Verifier Invariant Binding | Binds domain invariant symbolic constraints into tiered L0 SymPy, L1 IBP, and L2 IBnB proof engine | M5 | survey |
| 18 | Enhanced `frame status` Observability | Displays Swarm Learning State, test-time compute scaling curve, Information Governor audit logs, and Executioner event logs | M6 | survey |
| 19 | Expanded `frame swarm --export-agy-specs` | Exports Antigravity specifications for Governor, Proximity Analyzer, Meta-Reviewer, Executioner, and Lane Specialists | M1, M6 | survey |
| 20 | Statistical Screener Non-Regression | Guarantees existing LORD-3 streaming FDR, two-stage screening, ROPE superiority, and FlopCounterMode are preserved (110 baseline tests pass) | M7 | survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Cognitive Swarm Roles & Information Governor (R1) | `frame/agents/governor.py`, `frame/agents/agy_bridge.py`, role specs | none | PLANNED |
| M2 | Dual-Key Novelty Contract (R2) | `frame/verification/novelty.py`, `frame/schemas/novelty.py`, novelty gate | M1 | PLANNED |
| M3 | Persistent Learning State Node & Test-Time Compute (R3) | `frame/brain/learning_state.py`, `frame/schemas/learning_state.py`, scaling envelope | M1 | PLANNED |
| M4 | Executioner Watchdog & Orthogonal Warm-Restart (R4) | `frame/brain/executioner.py`, `frame/brain/campaign.py`, stagnation audit | M3 | PLANNED |
| M5 | Domain Evidence Standards & Invariant Plugins (R5) | `frame/verification/domain_evidence.py`, `frame/verification/invariants.py`, L0–L2 binding | none | PLANNED |
| M6 | System Integration & CLI Observability | `frame/cli.py`, `frame/brain/campaign.py` status and swarm integration | M1, M2, M3, M4, M5 | PLANNED |
| M7 | Final Milestone: E2E Test Pass & Adversarial Hardening | Pass 100% of E2E test suite (Tiers 1–4) + Tier 5 Adversarial Coverage Hardening | M6, E2E Track | PLANNED |

## Interface Contracts

### `InformationGovernor` ↔ `AgySwarmCoordinator`
```python
class InformationGovernor(BaseModel):
    grading_threshold: float = 0.40
    def grade_source(self, source: EvidenceSource) -> VettingResult
    def audit_and_filter(self, sources: List[EvidenceSource]) -> List[EvidenceSource]
    def check_exploration_entropy(self, lane_counts: Dict[str, int]) -> Dict[str, Any]
```

### `DualKeyNoveltyContract` ↔ `AgySwarmCoordinator` / `AutonomousCampaignRunner`
```python
class DualKeyNoveltyContract:
    def evaluate_novelty(
        self,
        hypothesis_id: str,
        candidate_code: str,
        prior_art_corpus: List[str],
        y_true: np.ndarray,
        y_baseline_pred: np.ndarray,
        y_candidate_pred: np.ndarray,
    ) -> NoveltyContractResult
```
*Contract Invariant*: Pre-confirmatory check. Does NOT call `lord_controller.record_result()`; consumes zero alpha wealth.

### `SwarmLearningStateNode` ↔ `ExecutionerWatchdog` & `frame status`
```python
class SwarmLearningStateNode(BaseModel):
    node_id: str
    total_budget_n: int
    trajectories: List[TrajectoryPoint]
    governor_event_logs: List[Dict[str, Any]]
    executioner_event_logs: List[Dict[str, Any]]
    def record_evaluation(self, hypothesis_id: str, score: float, lane: str, depth: int) -> TrajectoryPoint
    def get_upper_envelope_curve(self) -> List[Tuple[int, float]]
    def verify_monotonic_progression(self) -> Tuple[bool, float]
    def compute_epistemic_entropy(self) -> float
```

### `ExecutionerWatchdog` ↔ `AutonomousCampaignRunner`
```python
class ExecutionerWatchdog:
    def audit_branch(
        self,
        branch_id: str,
        depth: int,
        visits: int,
        history: List[Dict[str, Any]]
    ) -> Tuple[bool, StagnationTrigger, str]
    def trigger_orthogonal_warm_restart(
        self,
        branch_id: str,
        current_lane: str,
        state_node: SwarmLearningStateNode
    ) -> ExecutionerEvent
```

### `DomainEvidenceVerifier` ↔ `TieredMathVerifier`
```python
class DomainEvidenceVerifier:
    def verify_evidence_hierarchy(self, sources: List[EvidenceSource]) -> Tuple[bool, float]
    def verify_provenance_and_ethics(self, provenance: DataProvenance) -> Tuple[bool, List[str]]
    def verify_symbolic_invariants(self, candidate_expr: str, domain: Optional[str] = None) -> Dict[str, Any]
```

## Code Layout
- Existing conventions in `/home/sparsh/frame`:
  - `frame/schemas/`: Data models and schemas (`compute.py`, `hypothesis.py`, `receipt.py`, `verification.py`, `capsule.py`, `novelty.py`, `learning_state.py`)
  - `frame/brain/`: MCTS, state, campaign, and reasoning (`state.py`, `thinking.py`, `campaign.py`, `experimental_design.py`, `governor.py`, `executioner.py`, `learning_state.py`)
  - `frame/verification/`: Verification engines (`statistics.py`, `math_engine.py`, `red_team.py`, `harness.py`, `novelty.py`, `domain_evidence.py`, `invariants.py`)
  - `frame/agents/`: Swarm coordination (`agy_bridge.py`, `governor.py`)
  - `frame/cli.py`: CLI entry points
  - `tests/`: Existing 9 test modules + new test modules (`test_governor.py`, `test_novelty_contract.py`, `test_learning_state.py`, `test_executioner.py`, `test_domain_evidence.py`, `test_cognitive_e2e.py`)
