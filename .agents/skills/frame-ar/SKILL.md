---
name: frame-ar
description: >-
  Autonomous scientific discovery, tiered math verification (L0-L2), anti-reward hacking, and swarm experimentation using the FRAME-AR framework. Use when the user asks to run an auto-research task, scientific exploration, Kaggle competition optimization, or model interpretability search.
---

# FRAME-AR: Autonomous Scientific Discovery & Verification Platform

The **FRAME-AR** (Framework for Autonomous Mathematics & Empirical Auto-Research) system is an end-to-end autonomous research and discovery engine. It pairs **Agentic Swarm Thinking & Evolutionary Search** (inspired by AlphaEvolve and AI-Scientist v2) with a **Rigorous Mathematical & Statistical Verification Engine** (tiered symbolic proofs, adversarial red-teaming, and online streaming false discovery rate control).

---

## ⚠️ MANDATORY PROTOCOL: NON-DEVIATION ENFORCEMENT & ANTI-SHORTCUT MANDATE

When tasked with solving a machine learning problem, optimizing a model, or conducting research using the FRAME-AR framework:

1. **NO MONOLITHIC SHORTCUTS**: You must **NEVER** write a single linear script that skips the thinking loops, hardcodes dummy evidence scores, bypasses math proofs, or hallucinates completed audit gates.
2. **NO TELEMETRY CONFLATION**: You must **NEVER** combine, average, or conflate multi-edge transitions into a single statistical row. Every transition between hypotheses is an explicit directed edge with its own independent two-stage test.
3. **NO UNBACKED CLAIMS**: Every claim in an audit checklist, scorecard, or report **MUST** correspond to a concrete, queryable event log in the epistemic ledger (`governor_event_logs` or `executioner_event_logs`). An unbacked checklist box is a catastrophic integrity failure.
4. **NO UNCERTIFIED BLENDS**: You must **NEVER** certify an ensemble, blend, or stacking layer as a discovery solely because it beats a naive root baseline (Gen 0). Any blend **MUST** undergo dedicated two-stage screening against the incumbent single best constituent model.
5. **MANDATORY TAPASCORE EVALUATION**: You must **NEVER** emit or freeze a GoalMark completion certificate without computing the numerical scalar `TAPAScore` $\in [0, 100]$ using the official formulation.

---

## 1. The 5-Phase Closed-Loop Research Architecture

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

## 2. The 4 Pre-Flight Production Protocols

### Protocol 1: Edge-Indexed FDR Reporting (Residual Graph Architecture)

To prevent conflating cumulative campaign gains with atomic architectural contributions, statistical verification is strictly indexed over directed edges $e = (u \to v)$ in the hypothesis search tree:

1. **Explicit Edge Fields on `StatisticalVerificationReport`**:
   - `edge_id: str`: Canonical edge key (e.g. `edge_gen0_to_gen1_lgbm`, `edge_gen1_to_gen4_stack`, `campaign_gen0_to_final`).
   - `source_hypothesis_id: str`: The reference incumbent baseline for this edge.
   - `target_hypothesis_id: str`: The candidate hypothesis under evaluation.
   - `source_node_id: Optional[str]`, `target_node_id: Optional[str]`: Graph node identifiers.
   - `edge_type: str`: Strictly one of:
     - `"atomic_transition"`: Single-lane mutation from parent hypothesis to descendant.
     - `"blend_ensemble"`: Meta-stacking or ensembling layer, tested strictly against incumbent best single model.
     - `"cumulative_campaign"`: End-to-end campaign performance tracking candidate vs root baseline (Gen 0).
   - `tapa_score: Optional[float]`: Numerical alignment score bound to the edge.

2. **Pre-Registered Per-Edge Superiority Margin ($\epsilon_{\text{ROPE}}$)**:
   - **Root Architectural Transitions** (Gen 0 $\to$ Gen 1): $\epsilon_{\text{ROPE}} = 0.005$ (meaningful structural leap).
   - **Stacking & Ensembling Refinements** (Constituent $\to$ Meta-Stack): $\epsilon_{\text{ROPE}} = 0.001$ (fine residual gain).

3. **Distinct Alpha Wealth Allocation**:
   - Every edge consumes its own sequential step $t$ and allocated significance threshold $\alpha_t > 0$ from the `LORDController`.
   - **Never reuse a previously consumed $\alpha_t$ across multiple edges.** Each edge must record its own `p_value_rope`, `allocated_alpha_t`, and `lord_step`.

---

### Protocol 2: The Blend Edge Protocol (Ensemble Screening vs Incumbent)

Ensembling introduces parameter bloat, memory footprint, and inference latency. A common pathology in automated ML is certifying an ensemble as a "discovery" solely because it beats a naive baseline (Gen 0), even when it offers no statistically verifiable advantage over the best single model.

**The Blend Invariant**:
> **Any ensemble, blend, or stacking layer CANNOT be certified or declared a discovery without passing its own dedicated two-stage screening test against the INCUMBENT BEST SINGLE CONSTITUENT MODEL.**

```
   [Gen 0 Baseline] ──(Edge 0->1: atomic_transition)──► [Gen 1 LightGBM] (CV: 0.56250)
                                                               │
                                                       (Incumbent Best Single)
                                                               │
                                                   (Edge 1->4: blend_ensemble)
                                                   H0: mu_D <= 0.001
                                                   alpha_t distinct
                                                               ▼
                                                        [Gen 4 Meta-Stack] (CV: 0.55668)
```

1. **Execution Rule**:
   - Identify the single best constituent model across all evaluated single architectures:
     $$M^* = \arg\min_{m \in \{M_1, \dots, M_K\}} \text{CV}(M_m)$$
   - Evaluate the candidate blend against $M^*$ on the identical paired seeds $k \ge 3$:
     $$D_i^{\text{blend}} = Y_{M^*}^{(i)} - Y_{\text{blend}}^{(i)}$$
   - Run Stage 1 Futility Filter: $\bar{D}^{\text{blend}} > 0$.
   - Run Stage 2 Confirmatory Test: Reject $H_0: \mu_D^{\text{blend}} \le \epsilon_{\text{ROPE}}^{\text{blend}} = 0.001$ at allocated threshold $\alpha_t^{\text{blend}}$.
2. **Rejection Rule**: If the blend edge fails confirmatory testing ($p_{\text{ROPE}} > \alpha_t$), the blend **MUST BE REJECTED** as unverified parameter bloat. The pipeline must fall back to the incumbent single model $M^*$.

---

### Protocol 3: Dual-Sink Event Ledger Appends & Degeneracy Probes

The `SwarmLearningStateNode` maintains two independent, queryable persistent event sinks. Every action taken by the cognitive agents must be actively recorded:

#### 1. Information Governor Sink (`learning_state.governor_event_logs`)
- **`source_vetting`**: Recorded on every audited literature or benchmark source.
  ```python
  learning_state.append_governor_event("source_vetting", {
      "source_id": "arxiv_2305_xxxxx",
      "title": "TabPFN / GBDT Meta-Features for California Housing",
      "tier": "preprint",  # cochrane_or_journal, preprint, public_forum, unverified
      "grade": 0.72,
      "accepted": True,
      "rejection_reason": None,
  })
  ```
- **`exploration_entropy_audit`**: Recorded during Phase 1 entropy checks ($H \ge 0.75 \ln 4$).
  ```python
  learning_state.append_governor_event("exploration_entropy_audit", {
      "total_proposals": 4,
      "entropy": 1.3863,
      "threshold": 1.0397,
      "is_high_entropy": True,
      "greedy_exploitation_flag": False,
      "lane_probabilities": {"features": 0.25, "architecture": 0.25, "objectives": 0.25, "optimization": 0.25},
  })
  ```
- **`novelty_audit`**: Recorded by `DualKeyNoveltyContract`.
  ```python
  learning_state.append_governor_event("novelty_audit", {
      "hypothesis_id": "hyp_gen1_spatial",
      "passed": True,
      "verdict": "APPROVED",
      "d_lit": 0.421,
      "tau_lit": 0.15,
      "d_func": 0.312,
      "epsilon_func": 0.10,
      "nearest_prior_art": "Baseline OLS",
  })
  ```

#### 2. Watchdog & Executioner Sink (`learning_state.executioner_event_logs`)
- **`upper_envelope_update`**: Recorded on every candidate evaluation recording the running maximum quality score ($1 / \text{RMSE}$ or accuracy).
  ```python
  learning_state.record_evaluation(
      hypothesis_id="hyp_gen1_spatial",
      score=1.0 / cv_score,
      lane="features",
      depth=1,
      log_event=True,  # Automatically appends upper_envelope_update
  )
  ```
- **`watchdog_monotonicity`**: Recorded when asserting upper envelope non-decreasing progression ($E(i) \ge E(i-1)$).
  ```python
  is_mono, gain = learning_state.verify_monotonic_progression(log_event=True)
  ```
- **`stagnation_check`**: Recorded on watchdog branch audits.
  ```python
  is_stagnant, trigger, reason = watchdog.audit_branch(..., state_node=learning_state)
  ```
- **`ast_falsification_recorded` / `ast_falsification_blocked`**: Recorded on AST filter checks.
  ```python
  is_blocked, rec = falsification_memory.check_hard_filter(code, state_node=learning_state)
  ```
- **`modality_b_degeneracy_probe`** (For Sparse Autoencoders & Modality B):
  ```python
  learning_state.append_executioner_event("modality_b_degeneracy_probe", {
      "hypothesis_id": "hyp_sae_l0",
      "l0_sparsity": 48.2,
      "loss_recovery": 0.962,
      "dead_feature_ratio": 0.003,
      "dead_features_gate_pass": True,  # dead_ratio < 0.01
      "loss_recovery_gate_pass": True,   # R >= 0.80
  })
  ```

---

### Protocol 4: Numerical TAPAScore Formulation

Before GoalMark completion certificates freeze, the pipeline must evaluate and emit a concrete numerical scalar `TAPAScore` $\in [0, 100]$:

#### Modality A: Empirical & Competitive Modeling (Kaggle / Tabular / Vision / NLP)

$$\text{TAPAScore} = 100 \times R_{\text{proof}} \times R_{\text{legality}} \times \left[ 0.50 \cdot \tanh(\max(0, R_{\text{stat}})) + 0.25 \cdot \eta_{\text{robust}} + 0.25 \cdot \eta_{\text{var}} \right]$$

where:
1. **$R_{\text{proof}} \in \{0, 1\}$**: Indicator from Tiered Math Proofs (L0 pole-free, L1 IBP bounded, L2 IBnB verified).
2. **$R_{\text{legality}} \in \{0, 1\}$**: Indicator from Recombination Legality Grammar ($A \oplus B \oplus D$).
3. **$R_{\text{stat}}$**: Normalized statistical discovery strength:
   $$R_{\text{stat}} = \max(0, \bar{D} - \epsilon_{\text{ROPE}}) \cdot \ln\left(1 + \frac{\alpha_t}{\max(p_{\text{ROPE}}, 10^{-9})}\right)$$
4. **$\eta_{\text{robust}} \in [0, 1]$**: Minimum margin over adversarial null models:
   $$\eta_{\text{robust}} = \text{clip}\left(\frac{\Delta_{\text{robust}}}{\epsilon_{\text{robust}}}, 0.0, 1.0\right), \quad \Delta_{\text{robust}} = \min_{m \in \text{NullModels}} (L_{\text{null}, m} - L_{\text{cand}})$$
5. **$\eta_{\text{var}} \in [0, 1]$**: Prediction variance preservation ratio ($\text{VarRatio} = s_{\hat{y}} / s_y$):
   $$\eta_{\text{var}} = \begin{cases}
   0.0 & \text{if } \text{VarRatio} < 0.50 \text{ (mean-collapse degeneracy)} \\
   \frac{\text{VarRatio} - 0.50}{0.50} & \text{if } 0.50 \le \text{VarRatio} \le 1.0 \\
   \max(0.0, 1.0 - 0.5(\text{VarRatio} - 1.0)) & \text{if } 1.0 < \text{VarRatio} \le 2.0 \\
   \max(0.0, 0.5 - 0.5(\text{VarRatio} - 2.0)) & \text{if } \text{VarRatio} > 2.0 \text{ (variance explosion)}
   \end{cases}$$

#### Modality B: Open-Ended / Interpretability / Sparse Autoencoders (SAEs)

$$\text{TAPAScore}_{\text{SAE}} = 100 \times R_{\text{proof}} \times R_{\text{legality}} \times \mathbf{1}_{\{\text{dead\_ratio} < 0.01\}} \times \left[ 0.45 \cdot \eta_{\text{recovery}} + 0.35 \cdot \eta_{\text{sparsity}} + 0.20 \cdot \tanh(R_{\text{stat}}) \right]$$

where:
- $\mathbf{1}_{\{\text{dead\_ratio} < 0.01\}}$: **Hard gate** failing any architecture with $\ge 1\%$ dead features.
- $\eta_{\text{recovery}} = \text{clip}\left(\frac{R - 0.80}{1.0 - 0.80}, 0.0, 1.0\right)$: Downstream loss recovery floor at $80\%$ with target $\ge 95\%$.
- $\eta_{\text{sparsity}} = \exp\left(-\frac{|L_0 - L_{0, \text{target}}|}{L_{0, \text{target}}}\right)$: Proximity to pre-registered target $L_0$ sparsity budget.

---

## 3. Mandatory Component & Ledger Initialization Schema

Every FRAME-AR execution pipeline (`pipeline.py`) MUST instantiate and wire the following core components:

```python
import os, sys, math, time, json, hashlib
import numpy as np
import torch

# 1. Framework Imports
from frame.schemas.hypothesis import TargetDirection
from frame.schemas.compute import ExecutionTier
from frame.schemas.receipt import ExecutionReceipt, HardwareTelemetry
from frame.schemas.verification import StatisticalVerificationReport, PairedSeedMetrics
from frame.schemas.learning_state import SwarmLearningStateNode, TrajectoryPoint, save_learning_state
from frame.brain.state import MCTSNode, SearchAccountingLedger, SearchStage
from frame.brain.thinking import EpistemicMCTSBrain, StructuralFalsificationMemory, RecombinationLegalityGrammar
from frame.brain.experimental_design import TwoStageScreeningController, ScreeningDecision
from frame.verification.statistics import LORDController, evaluate_paired_seed_rope
from frame.verification.novelty import DualKeyNoveltyContract
from frame.verification.red_team import AdversarialRedTeamSuite
from frame.verification.math_engine import verify_l0_analytic, verify_l1_symbolic_numeric, verify_l2_ibnb
from frame.verification.tapa import compute_tapa_score, compute_tapa_score_modality_b
from frame.agents.governor import InformationGovernor, EvidenceSource, SourceTier
from frame.brain.executioner import ExecutionerWatchdog

# 2. Mandatory Component Instantiation
learning_state = SwarmLearningStateNode(node_id="learning_state_campaign", total_budget_n=0)
governor = InformationGovernor(grading_threshold=0.40, entropy_threshold_ratio=0.75, num_expected_lanes=4)
brain = EpistemicMCTSBrain()
falsification_memory = StructuralFalsificationMemory()
recomb_grammar = RecombinationLegalityGrammar()
novelty_contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
screening_controller = TwoStageScreeningController(rope_epsilon=0.005, target_direction=TargetDirection.MINIMIZE)
lord_controller = LORDController(alpha=0.05)
red_team_suite = AdversarialRedTeamSuite(epsilon_robust=0.005, target_direction=TargetDirection.MINIMIZE)
watchdog = ExecutionerWatchdog()
```

---

## 4. Phase-by-Phase Mandatory Execution Template

### Phase 1: Problem Ingestion & Targeted Swarm Thinking
1. Ingest dataset, calculate target distribution ($\mu_y, \sigma_y$), determine noise floor ($f(x) \equiv \bar{y}$).
2. Perform external literature mining via `search_web` / `read_url_content`.
3. Grade evidence sources using `governor.grade_source(source, state_node=learning_state)`.
4. Audit Phase 1 exploration entropy using `governor.check_exploration_entropy(lane_counts, state_node=learning_state)`.
5. Establish formal Gen 0 baseline on paired seeds $k \ge 5$ (Common Random Numbers).
6. Create MCTS root node via `brain.create_root_node(...)`.
7. Record baseline evaluation with `learning_state.record_evaluation(score=..., lane="root", depth=0, log_event=True)`.

### Phase 2: Compositional Synthesis & Pre-Flight Verification
1. Run Tiered Math Proofs:
   - Level 0: `verify_l0_analytic(expr, variables=...)`
   - Level 1: `verify_l1_symbolic_numeric(expr, input_bound=...)`
   - Level 2: `verify_l2_ibnb(...)`
2. Check Structural AST Falsification Memory:
   `is_blocked, _ = falsification_memory.check_hard_filter(code, state_node=learning_state)`
3. Audit Recombination Legality:
   `is_legal, code_viol, expl = recomb_grammar.validate_recombination(model_spec, loss_spec, opt_spec)`
4. Evaluate Dual-Key Novelty Contract:
   `novelty = novelty_contract.evaluate_novelty(..., state_node=learning_state)`

### Phase 3: Multi-Tier Empirical Execution & Hierarchical Screening
1. Train candidate model across 5-fold CV with strict per-fold preprocessing hygiene.
2. Evaluate across paired seeds $k \ge 5$.
3. Two-Stage Sequential Screening:
   - **Stage 1 Futility Filter ($k=3$)**: Immediate abort if $\bar{D} \le 0$.
   - **Stage 2 Confirmatory Test**: Screen under streaming LORD $\alpha_t$ with pre-registered per-edge $\epsilon_{\text{ROPE}}$:
     ```python
     s2 = screening_controller.evaluate_stage2_confirmatory(
         baseline_scores=incumbent_seed_scores,
         candidate_scores=candidate_seed_scores,
         lord_controller=lord_controller,
         rope_epsilon_override=edge_rope_margin,
         edge_id=edge_id,
         source_hypothesis_id=incumbent_hyp_id,
         target_hypothesis_id=candidate_hyp_id,
         edge_type=edge_type,
     )
     ```
4. **Apply The Blend Edge Protocol**: If evaluating a blend/stack, screen strictly against the incumbent best single model with $\epsilon_{\text{ROPE}} = 0.001$.
5. Evaluate Adversarial Red-Team Suite (5 null models + variance preservation ratio).
6. Compute numerical `TAPAScore` via `compute_tapa_score(...)`.
7. Record evaluation and assert upper envelope monotonicity:
   ```python
   learning_state.record_evaluation(..., log_event=True)
   is_mono, gain = learning_state.verify_monotonic_progression(log_event=True)
   ```

### Phase 4: Recursive Evolutionary Feedback Loop (On Sub-Goal / Rejection)
1. If candidate is rejected or goal is not reached, **NEVER terminate prematurely**.
2. Extract error residuals $(y - \hat{y})$ and failure signature.
3. Record fatal AST to `falsification_memory.record_fatal_failure(..., state_node=learning_state)`.
4. Synthesize post-mortem telemetry and generate targeted external search queries via `synthesize_post_mortem_telemetry(...)`.
5. Increment MCTS recursion depth ($d \le \text{max\_depth}$) and backpropagate oracle reward.
6. **REROUTE DIRECTLY BACK TO PHASE 1**: Ingest external research cures, mutate Generation $N+1$, and re-enter Phase 2.

### Phase 5: Cryptographic CAS Binding & Modality A Closure
1. Generate `hypothesis_hash`, `receipt_hash`, and `report_hash`.
2. Populate edge-indexed `StatisticalVerificationReport` linking `edge_id`, `edge_type`, `tapa_score`, `allocated_alpha_t`, and `p_value_rope`.
3. Assemble Merkle CAS root CID:
   $$\text{Merkle Root} = \text{SHA256}(\text{Hypothesis Hash} \parallel \text{Receipt Hash} \parallel \text{Report Hash})$$
4. Check generalization gap against frozen leaderboard: $|L_{\text{CV}} - S_{\text{LB}}| \le \tau_{\text{leak}}$.
5. Emit `goalmark_certificate.json` binding `merkle_root_cid`, `tapa_score`, CV score, and leaderboard ranking.

---

## 5. Forbidden Anti-Patterns & Deviation Prohibitions

| Anti-Pattern | Violation Description | Enforced Penalty / Guardrail |
| :--- | :--- | :--- |
| **Monolithic Hardcoding** | Jumping directly to training a single model without Phase 1 web search or literature vetting. | Immediate pipeline invalidation; Governor audit required. |
| **Merged Telemetry Collision** | Pairing Stage 1 $\bar{D}$ from Edge A with Stage 2 $p$-value from Edge B in scorecards. | Schema validation failure; `edge_id` must uniquely index every test. |
| **Unbacked Checklist Lines** | Checking boxes in reports while `governor_event_logs` or `executioner_event_logs` are empty. | Integrity assertion failure; report rejected as unbacked claim. |
| **Unverified Blend Certification** | Declaring a blend a discovery solely because it beats Gen 0, without screening against the best single model. | Immediate certificate revocation; Blend Edge protocol mandatory. |
| **Undefined TAPAScore** | Referencing TAPAScore as a qualitative buzzword without numerical computation. | GoalMark certificate generation blocked; formula must be executed. |
| **Global Data Leakage** | Fitting scalers or target encodings on the full dataset before cross-validation splitting. | V6 protocol violation; strict per-fold fit/transform mandatory. |
| **Alpha Re-Use** | Using the same LORD $\alpha_t$ allocation across multiple distinct transitions. | Streaming FDR failure; each edge must consume its own $\alpha_t$. |

---

## 6. Comprehensive Discovery Verification Checklist

Before emitting a GoalMark certificate or claiming confirmed discovery:

- [ ] **Phase 1 Executed**: Web research conducted, external literature mined, evidence sources vetted ($G(s) \ge 0.40$).
- [ ] **Governor Ledger Populated**: `learning_state.governor_event_logs` contains `source_vetting` and `exploration_entropy_audit` events.
- [ ] **Level 0 SymPy Proof**: Verified pole-free with continuous limits at $\pm \infty$.
- [ ] **Level 1 IBP Proof**: Certified safe across operational interval $[-B, B]$.
- [ ] **Structural AST Memory**: Canonical AST hash is unique and unblocked globally.
- [ ] **Recombination Grammar**: $A \oplus B \oplus D$ certified legal without typing violations.
- [ ] **Dual-Key Novelty Contract**: $d_{\text{lit}} > 0.15$ and $d_{\text{func}} > 0.10$ evaluated with zero alpha wealth spent.
- [ ] **5-Fold CV Hygiene**: Scalers fit strictly per-fold without global leakage (V6).
- [ ] **Edge-Indexed FDR**: Report explicitly indexes `edge_id`, `edge_type`, and pre-registered $\epsilon_{\text{ROPE}}$.
- [ ] **The Blend Edge Verified**: Blended/stacked model passed two-stage screening against the incumbent best single model.
- [ ] **Streaming LORD 3**: Wealth $W_t$ updated, distinct allocated $\alpha_t > 0$ strictly consumed per edge.
- [ ] **Adversarial Red-Team**: Candidate beats all 5 null models ($\Delta_{\text{null}} > \epsilon_{\text{robust}}$).
- [ ] **Variance Preservation**: Predicted standard deviation $s_{\hat{y}} \ge 0.50 s_y$.
- [ ] **Executioner Ledger Populated**: `learning_state.executioner_event_logs` contains `upper_envelope_update` and `watchdog_monotonicity` events.
- [ ] **Numerical TAPAScore Evaluated**: Evaluated via `compute_tapa_score` and bound to report and certificate.
- [ ] **Cryptographic Binding**: Merkle CAS root CID binds hypothesis hash, receipt hash, and report hash.
- [ ] **Modality A Closure**: Leaderboard score $S_{\text{LB}}$ recorded and $|L_{\text{CV}} - S_{\text{LB}}| \le \tau_{\text{leak}}$.
