# FRAME-AR: Foundation for Rigorous Autonomous & Mathematical Experimentation — Auto-Research
## Complete System Architecture, Mathematical Foundations & Production Contracts

---

## 1. System Overview & Core Philosophy

**FRAME-AR** is an autonomous scientific discovery and experimentation framework designed specifically for resource-constrained local environments paired with remote accelerator APIs and high-performance compute clusters.

```
                                  [ Research Goal / Idea ]
                                             │
                                             ▼
                             [ Planner Core: MCTS & Mutation ]
                                             │
                                             ▼
                        [ Program of Thought (PoT) Verification ]
                         ├── Static AST Safety Check
                         ├── PyTorch FX Symbolic Tracing
                         ├── FlopCounterMode Dry-Run Profiler (B=1)
                         └── Dynamic Shape & Gradient Proofs
                                             │
                                             ▼
                              [ Compute Dispatch Router ]
                                             │
          ┌──────────────────────────────────┼──────────────────────────────────┐
          ▼                                  ▼                                  ▼
[ Tier 1: Host Local ]            [ Tier 2: Kaggle Remote ]          [ Tier 3: Manual Capsule ]
• GTX 1650 4GB / CPU              • Tesla P100 16GB / Dual T4        • Multi-GPU / External Clusters
• F_total <= 3e14 FLOPs           • 3e14 < F_total <= 1e17 FLOPs     • F_total > 1e17 FLOPs
• Peak VRAM <= 3.0 GB             • Peak VRAM <= 14.5 GB             • Peak VRAM > 14.5 GB
• Runtime <= 300s                 • Runtime <= 8.5h (Quota aware)    • Runtime > 8.5h or Air-Gapped
• Immediate Execution             • Self-Extracting REST Payload     • Air-Gapped Tarball Capsule
          │                                  │                                  │
          └──────────────────────────────────┼──────────────────────────────────┘
                                             │
                                             ▼
                               [ Host Replay Verification ]
                                (Assert |ΔL_replay| <= 1e-4)
                                             │
                                             ▼
                         [ Mathematical Verification Engine ]
                         ├── Common Random Numbers (k >= 5 Paired Seeds)
                         ├── Paired Difference Analysis (D_i = Y_exp - Y_ctrl)
                         ├── Region of Practical Equivalence (ROPE) Test
                         └── LORD 3 (LORD++) Online Streaming FDR Controller
                                             │
                                             ▼
                          [ Epistemic Merkle Ledger & CAS ]
                           (RFC 8785 Canonical JSON Hashing)
```

---

## 2. Three-Tier Hardware-Calibrated Execution Model

The host machine features an **AMD Ryzen 5 3550H (8 threads)**, **13 GiB RAM**, and an **NVIDIA GeForce GTX 1650 (4096 MiB VRAM)**. To maximize research throughput without causing out-of-memory kernel kills or draining remote quotas on trivial tasks, execution is routed across three distinct tiers:

### Tier 1: Host Local Sandbox (Low Compute)
* **Hardware Target**: Local CPU or GTX 1650 (enforcing `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and `torch.cuda.set_per_process_memory_fraction(0.75)`).
* **Boundaries**:
  * Total Compute: $F_{\text{total}} \le 3 \times 10^{14}$ FLOPs ($\sim 5$ minutes at effective 1 TFLOP/s MFU).
  * Peak VRAM: $\le 3.0$ GiB.
  * Maximum Runtime: $\le 300$ seconds.
* **Workloads**: PoT dry-runs, symbolic graph tracing, synthetic micro-baselines, and toy analytical prototypes.

### Tier 2: Remote Kaggle Engine (Medium-High Compute)
* **Hardware Target**: Kaggle Tesla P100 (16GB VRAM) or Dual T4 (2x15GB VRAM), 4 vCPUs, 30GB RAM.
* **Boundaries**:
  * Total Compute: $3 \times 10^{14} < F_{\text{total}} \le 1.0 \times 10^{17}$ FLOPs.
  * Peak VRAM: $\le 14.5$ GiB.
  * Session Runtime: $\le 8.5$ hours (providing a 30-minute safety buffer below Kaggle's 9.0-hour GPU timeout).
  * Weekly Quota: Verified against remaining weekly GPU quota (rolling 30 hours).
* **Single-File Payload Architecture**:
  * Because Kaggle's API (`POST /api/v1/kernels/push`) only transmits the raw script body in `code_file` and ignores auxiliary directory files, FRAME-AR implements an **in-memory self-extracting payload generator**.
  * The source tree, configuration, and evaluation harness are compressed via `zlib` into a Base64 archive string embedded in the header of `entrypoint.py`.
  * On remote startup, `entrypoint.py` automatically decompresses the archive into `/kaggle/working/frame_src/` and registers it on `sys.path`.
* **Zero-Dependency Native REST Client**:
  * Uses Python's standard `urllib.request` with HTTP Basic Authentication (`Authorization: Basic base64("{username}:{key}")`), eliminating any dependency on external CLI binaries.

### Tier 3: Manual Execution System Requirement (Massive Compute)
* **Trigger Conditions**:
  * Workloads requiring $F_{\text{total}} > 1.0 \times 10^{17}$ FLOPs.
  * Peak VRAM $> 14.5$ GiB (e.g. 70B parameter models, large context windows).
  * Multi-GPU distributed training (FSDP / Megatron-LM).
  * Exceeding Kaggle's 8.5-hour runtime safety ceiling or 30-hour weekly quota.
  * Proprietary / air-gapped data requiring offline training.
* **The Manual Execution Capsule Protocol**:
  1. **Packaging**: The framework generates an air-gapped bundle in `.frame/capsules/<capsule_id>/`:
     * `manifest.json`: Cryptographic specification containing code hash, dataset hash, seed lists, and expected artifact SHA-256 schemas.
     * `entrypoint.py`: Standalone execution script enforcing strict IEEE 754 precision (`allow_tf32 = False`).
     * `run.sh`: Self-bootstrapping script using `uv` to create an ephemeral virtual environment and install exact locked dependencies.
     * `Dockerfile` / `compose.yaml`: Containerized runner with GPU pass-through (`--gpus all`) for cloud providers (Lambda Labs, RunPod, AWS, university HPC clusters).
     * `singularity.def`: HPC cluster definition for rootless academic supercomputers.
  2. **Non-Blocking Multi-Lane Execution**: The orchestrator marks only the dependent hypothesis as `AWAITING_MANUAL_EXECUTION`. Other local symbolic derivations and independent Kaggle jobs continue executing concurrently on separate lanes.
  3. **Deterministic Host Ingestion & Replay Verification**:
     * The researcher returns `results_<capsule_id>.tar.gz` and executes `frame ingest --capsule <id> --bundle <tar.gz>`.
     * **Step 1 (Integrity)**: Verifies SHA-256 digests of all checkpoints, metric logs, and execution traces against `manifest.json`.
     * **Step 2 (Deterministic Replay Pass)**: The local host loads the reported checkpoint weights and executes a 3-step deterministic micro-forward-backward pass using the recorded validation seeds. It computes:
       $$\left| \mathcal{L}_{\text{val}}^{\text{replay}} - \mathcal{L}_{\text{val}}^{\text{reported}} \right| \le 10^{-4}$$
       If $|\Delta \mathcal{L}| > 10^{-4}$, the receipt is **rejected as fraudulent or non-reproducible**.
     * **Step 3 (TF32 Protection)**: By disabling TensorFloat-32 (`torch.backends.cuda.matmul.allow_tf32 = False`) during both remote and local execution, cross-architecture mantissa truncation drift is eliminated.

---

## 3. Mathematical Execution Verification Engine

### 3.1 The Small-Sample Non-Parametric Failure
Traditional auto-research frameworks often attempt to use non-parametric rank tests (e.g. Mann-Whitney $U$) on small sample sizes ($k = 5$ seeds) combined with FDR control. In FRAME-AR, this was proven mathematically impossible:

The minimum attainable two-sided $p$-value for Mann-Whitney $U$ with $n_1 = n_2 = 5$ is:
$$p_{\min}^{\text{MWU}}(5, 5) = \frac{2}{\binom{10}{5}} = \frac{2}{252} \approx \mathbf{0.007936}$$

Under standard Benjamini-Hochberg FDR control with $m \ge 8$ hypotheses, the rank-1 significance threshold is:
$$\alpha_{\text{BH}}^{(1)} = \frac{1}{m} \alpha = \frac{0.05}{8} = \mathbf{0.00625}$$

Because $0.007936 > 0.00625$, Mann-Whitney $U$ at $k=5$ can **never** reject the null hypothesis at rank 1, regardless of how massive the experimental improvement is. The framework is permanently blinded.

### 3.2 Common Random Numbers (CRN) Paired Seed Design
FRAME-AR resolves this by evaluating baseline and candidate models on identical deterministic random seeds ($k \ge 5$):
$$D_i = Y_{\text{candidate}}^{(i)} - Y_{\text{baseline}}^{(i)}, \quad i \in \{1, \dots, k\}$$

Using paired differences eliminates inter-seed dataset variance, isolating the true architectural effect:
$$\bar{D} = \frac{1}{k} \sum_{i=1}^k D_i, \qquad S_D^2 = \frac{1}{k - 1} \sum_{i=1}^k (D_i - \bar{D})^2, \qquad SE = \frac{S_D}{\sqrt{k}}$$

### 3.3 Region of Practical Equivalence (ROPE) Superiority
A finding is not considered a scientific discovery merely because $p < \alpha$; tiny, negligible improvements ($+0.001\%$) with low variance can achieve statistical significance. FRAME-AR tests directional superiority beyond a practical margin $\epsilon_{\text{ROPE}}$:
$$H_{0}: \mu_D \le \epsilon_{\text{ROPE}} \quad \text{vs.} \quad H_{1}: \mu_D > \epsilon_{\text{ROPE}}$$
$$t_{\text{ROPE}} = \frac{\bar{D} - \epsilon_{\text{ROPE}}}{SE}, \qquad p_{\text{ROPE}} = 1 - F_{t, k-1}(t_{\text{ROPE}})$$

### 3.4 LORD 3 (LORD++) Online Streaming FDR Control
Because experiments complete asynchronously across hours or days, static batch FDR procedures (such as Benjamini-Hochberg) cannot be applied. FRAME-AR integrates **LORD 3 (Learning with Ordered Representations of Decisions)**:

* **Initial Alpha Wealth**: $W_0 = \frac{\alpha}{2} = 0.025$ (for target $\alpha = 0.05$).
* **Normalizing Decay Factor**: Evaluated using the exact Riemann zeta function with exponent $s = 1.6$:
  $$\zeta(1.6) = \sum_{n=1}^\infty \frac{1}{n^{1.6}} \approx \mathbf{2.285766}$$
  $$\gamma_t = \frac{1}{\zeta(1.6) \cdot t^{1.6}}, \qquad \sum_{t=1}^\infty \gamma_t = 1.0$$
* **Dynamic Alpha Allocation**:
  $$\alpha_t = W_0 \gamma_t + (\alpha - W_0) \gamma_{t - \tau_1} \mathbf{1}_{\{\tau_1 < t\}} + \sum_{j \ge 2 : \tau_j < t} \alpha \gamma_{t - \tau_j}$$
  where $\tau_j$ is the index of the $j$-th confirmed discovery ($p_{\text{ROPE}, \tau_j} \le \alpha_{\tau_j}$).
* **Epistemic Guarantee**: For any arbitrary, data-dependent stopping time $\mathcal{T}$, the False Discovery Rate satisfies:
  $$\mathbb{E}[\text{FDR}_{\mathcal{T}}] \le \alpha$$

### 3.5 Edge-Indexed FDR Reporting & Residual Graph Architecture
To prevent conflating cumulative campaign progression with atomic architectural contributions, all statistical verification in FRAME-AR is indexed over directed edges in the hypothesis search graph:
* **Edge Identifier (`edge_id`)**: Explicit string key indexing the transition (e.g. `edge_gen0_to_gen1`, `edge_incumbent_to_blend`, `campaign_gen0_to_final`).
* **Source & Target Hypotheses**: `source_hypothesis_id` (the reference incumbent) and `target_hypothesis_id` (the candidate under test), paired with graph node IDs (`source_node_id`, `target_node_id`).
* **Edge Taxonomy (`edge_type`)**:
  1. `atomic_transition`: Discrete step from parent hypothesis to mutated descendant within an exploration lane.
  2. `blend_ensemble`: Meta-learning, stacking, or ensembling layer combining existing models, tested strictly against the incumbent single best constituent.
  3. `cumulative_campaign`: End-to-end campaign performance tracking candidate vs initial root baseline (Gen 0).
* **Pre-Registered Per-Edge Margin ($\epsilon_{\text{ROPE}}$)**:
  - Root architectural leaps: $\epsilon_{\text{ROPE}} = 0.005$
  - Stacking and ensembling refinements: $\epsilon_{\text{ROPE}} = 0.001$
* **Distinct Alpha Consumption**: Each edge consumes its own distinct $\alpha_t$ allocated by the streaming LORD 3 controller. Reusing previously consumed $\alpha_t$ across edges is strictly forbidden.

### 3.6 The Blend Edge Protocol
Ensembling introduces parameter bloat, inference latency, and memory footprint. A common failure in empirical research is declaring an ensemble a "discovery" solely because it outperforms a naive baseline (Gen 0), even when it offers no statistically verifiable advantage over the best constituent model.

**The Blend Invariant**:
> **Any ensembled or blended artifact CANNOT be certified or declared a discovery without undergoing its own explicit two-stage paired-seed screening against the incumbent single best model (the blend edge). If not evaluated, it cannot be certified.**

1. Identify the incumbent single best model: $M^* = \arg\max_{m \in \text{constituents}} \text{Metric}(m)$.
2. Register the blend edge: `edge_id = f"edge_{M*.id}_to_{blend.id}"`, `edge_type = "blend_ensemble"`.
3. Pre-register the refinement margin: $\epsilon_{\text{ROPE}} = 0.001$.
4. Request a fresh alpha allocation $\alpha_t$ from `LORDController`.
5. Execute Stage 1 Futility Filter ($k=3$) and Stage 2 Confirmatory Test ($k^* \ge 5$).
6. If $p_{\text{ROPE}} > \alpha_t$ or $\bar{D} \le \epsilon_{\text{ROPE}}$, the blend is **CONFIRMED_NULL** and must NOT receive a GoalMark certificate.

### 3.7 Targeted Analytical Performance Alignment Score (TAPAScore)
Prior to issuing a GoalMark completion certificate, the candidate artifact is evaluated via the closed-form numerical TAPAScore scalar $\in [0, 100]$:

$$\text{TAPAScore} = \Phi(R_{\text{proof}}, R_{\text{legality}}, R_{\text{stat}}, \Delta_{\text{robust}}, \text{VarRatio})$$

#### Mathematical Formulation (Modality A - Supervised / Tabular):
$$\text{TAPAScore} = 100 \times R_{\text{proof}} \times R_{\text{legality}} \times \left[ w_{\text{stat}} \cdot \tanh(\max(0, R_{\text{stat}})) + w_{\text{robust}} \cdot \eta_{\text{robust}} + w_{\text{var}} \cdot \eta_{\text{var}} \right]$$

where:
1. **Hard Epistemic Gates**:
   - $R_{\text{proof}} \in \{0, 1\}$: Level 0–Level 2 mathematical proof verification. Poles or unbounded operational intervals yield $R_{\text{proof}} = 0$.
   - $R_{\text{legality}} \in \{0, 1\}$: Recombination legality grammar ($A \oplus B \oplus D$). Non-smooth or ill-posed combinations yield $R_{\text{legality}} = 0$.
   - If either gate is zero, $\text{TAPAScore} = 0.0$.
2. **Statistical Discovery Strength**:
   $$R_{\text{stat}} = \max\left(0, \frac{\bar{D} - \epsilon_{\text{ROPE}}}{\max(\epsilon_{\text{ROPE}}, 10^{-6})}\right) \cdot \ln\left(1 + \frac{\alpha_t}{\max(p_{\text{ROPE}}, 10^{-9})}\right)$$
3. **Adversarial Robustness Margin**:
   $$\eta_{\text{robust}} = \text{clip}\left(\frac{\Delta_{\text{robust}}}{\epsilon_{\text{robust}}}, 0.0, 1.0\right), \quad \Delta_{\text{robust}} = \min_{m \in \text{NullModels}} (L_{\text{null}, m} - L_{\text{cand}})$$
4. **Variance Preservation Ratio**:
   $$\text{VarRatio} = \frac{s_{\hat{y}}}{s_y}$$
   $$\eta_{\text{var}} = \begin{cases}
   0.0 & \text{if } \text{VarRatio} < 0.50 \text{ (mean-collapse degeneracy)} \\
   \frac{\text{VarRatio} - 0.50}{0.50} & \text{if } 0.50 \le \text{VarRatio} \le 1.0 \\
   \max(0.0, 1.0 - 0.5(\text{VarRatio} - 1.0)) & \text{if } 1.0 < \text{VarRatio} \le 2.0 \\
   \max(0.0, 0.5 - 0.5(\text{VarRatio} - 2.0)) & \text{if } \text{VarRatio} > 2.0 \text{ (variance explosion)}
   \end{cases}$$
5. **Canonical Weights**: $w_{\text{stat}} = 0.50, w_{\text{robust}} = 0.25, w_{\text{var}} = 0.25$.

#### Modality B Formulation (Open-Ended / Sparse Autoencoders):
$$\text{TAPAScore}_{\text{SAE}} = 100 \times R_{\text{proof}} \times R_{\text{legality}} \times \mathbf{1}_{\{\text{dead\_ratio} < 0.01\}} \times \left[ 0.45 \cdot \eta_{\text{recovery}} + 0.35 \cdot \eta_{\text{sparsity}} + 0.20 \cdot \tanh(R_{\text{stat}}) \right]$$
where:
- $\mathbf{1}_{\{\text{dead\_ratio} < 0.01\}}$ hard-fails any model with $\ge 1\%$ dead features.
- $\eta_{\text{recovery}} = \text{clip}\left(\frac{R - 0.80}{1.0 - 0.80}, 0.0, 1.0\right)$ enforces downstream loss recovery floor at $80\%$ with target $\ge 95\%$.
- $\eta_{\text{sparsity}} = \exp\left(-\frac{|L_0 - L_{0, \text{target}}|}{L_{0, \text{target}}}\right)$ measures proximity to the pre-registered $L_0$ budget.


---

## 4. Program of Thought (PoT) Tracing & Verification

Before code is scheduled for training, it passes through four static and micro-execution barriers:

```
[Candidate PyTorch Code]
         │
         ▼
[1. Static AST Sandbox] ──► Disallows os.system, socket, eval, monkey-patching
         │
         ▼
[2. PyTorch FX Tracing] ──► Symbolic graph construction (detects dynamic shape mismatches)
         │
         ▼
[3. FlopCounterMode]    ──► Micro-pass (B=1) measures exact forward & backward ATen FLOPs
         │
         ▼
[4. Gradient Check]     ──► Finite-difference verification of custom autograd functions
```

### Exact Compute Formulation
The total training FLOPs are determined deterministically:
$$\text{FLOPs}_{\text{total}} = S_{\text{seeds}} \times E_{\text{epochs}} \times B_{\text{batches}} \times \left(3 \times \text{FLOPs}_{\text{forward}}(B=1)\right)$$
where the factor of $3$ accounts for 1 forward pass and 2 backward passes (gradient computation and weight updates).

---

## 5. Epistemic Evidence Chain & Merkle CAS

To ensure total reproducibility and tamper-evident science, every artifact is addressed by its cryptographic hash using the **RFC 8785 JSON Canonicalization Scheme (JCS)**:
* Keys are sorted lexicographically by UTF-8 code point.
* All superfluous whitespace is eliminated.
* Floats are encoded in strict canonical representation.

```
       [ Goal Merkle Node ]
                 │
                 ▼
    [ Hypothesis Merkle Node ] ◄── Linked to Parent Idea
                 │
                 ▼
    [ PoT Verification Node ]  ◄── Contains FX Graph Hash & FLOPs
                 │
                 ▼
    [ Execution Receipt Node ] ◄── Contains Checkpoint & Trace Hashes
                 │
                 ▼
  [ Replay Verification Node ] ◄── Proves |ΔL| <= 1e-4 on Host
                 │
                 ▼
   [ Statistical Report Node ] ◄── Evaluates ROPE t-test + LORD FDR
```

Every claim in the final synthesized paper links directly to an unbroken audit trail:
$$\text{Claim} \longrightarrow \text{Report}_{\text{CID}} \longrightarrow \text{Replay}_{\text{CID}} \longrightarrow \text{Receipt}_{\text{CID}} \longrightarrow \text{Code}_{\text{CID}} \longrightarrow \text{Dataset}_{\text{CID}}$$

---

## 6. Swarm Learning State Node & Dual Sinks Provenance

The `SwarmLearningStateNode` (`frame.schemas.learning_state`) tracks compounding compute scaling, epistemic entropy, and persistent audit telemetry across recursive MCTS generations. To ensure strict non-deviation, two independent audit sinks are continuously populated throughout pipeline execution:

### 6.1 Information Governor Sink (`governor_event_logs`)
Maintains evidence quality and epistemic exploration logs:
1. `source_vetting`: Emitted when external literature or benchmark sources are graded.
   - Payload: `source_id`, `title`, `tier` (`cochrane_or_journal`, `preprint`, `public_forum`, `unverified`), `grade`, `accepted`, `rejection_reason`.
2. `exploration_entropy_audit`: Emitted during Phase 1 entropy checks.
   - Payload: `total_proposals`, `entropy`, `threshold`, `is_high_entropy`, `greedy_exploitation_flag`, `lane_probabilities`.
3. `novelty_audit`: Emitted by `DualKeyNoveltyContract`.
   - Payload: `hypothesis_id`, `passed`, `verdict`, `d_lit`, `tau_lit`, `d_func`, `epsilon_func`, `nearest_prior_art`, `rejection_reason`.

### 6.2 Watchdog & Executioner Sink (`executioner_event_logs`)
Maintains falsification memory, watchdog monotonicity, and degeneracy probe logs:
1. `upper_envelope_update`: Emitted on every candidate evaluation recording the running maximum.
   - Payload: `hypothesis_id`, `step`, `score`, `upper_envelope`, `lane`, `depth`.
2. `watchdog_monotonicity`: Emitted when Upper Envelope curve monotonicity ($E(i) \ge E(i-1)$) is asserted.
   - Payload: `is_monotonic`, `net_gain`, `tolerance`, `step_count`.
3. `stagnation_check`: Emitted during branch history watchdog audits.
   - Payload: `branch_id`, `depth`, `visits`, `is_stagnant`, `trigger` (`VANISHING_GRADIENTS`, `LOSS_PLATEAU`, `CIRCULAR_AST`), `reason`.
4. `orthogonal_warm_restart`: Emitted when the executioner cycles a stagnant branch to an orthogonal lane.
   - Payload: `branch_id`, `new_branch_id`, `source_lane`, `target_lane`, `trigger`, `injected_priors`.
5. `ast_falsification_recorded` / `ast_falsification_blocked`: Emitted when code is fingerprinted and rejected by `StructuralFalsificationMemory`.
   - Payload: `fingerprint`, `stage`, `reason`, `branch_id`, `depth`.
6. `modality_b_degeneracy_probe`: Emitted for Modality B (SAE) threshold evaluations.
   - Payload: `hypothesis_id`, `l0_sparsity`, `loss_recovery`, `dead_feature_ratio`, `tapa_score`, `sparsity_gate_pass`, `dead_features_gate_pass`.

