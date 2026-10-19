# FRAME-AR Architecture & Implementation Roadmap (`todo.md`)

## Phase 1: Kaggle REST Infrastructure & Credential Management
- [x] Add Kaggle API credentials configuration:
  - Resolution order: Environment variables `KAGGLE_USERNAME` & `KAGGLE_KEY` -> `~/.kaggle/kaggle.json` -> `.frame/kaggle.json`.
  - Enforce file permissions check (POSIX `0600`) on credential files.
- [x] Implement zero-dependency native REST client in `frame/runners/kaggle/client.py`:
  - Basic Auth header: `Authorization: Basic base64("{username}:{key}")`.
  - Kernel Push: `POST https://www.kaggle.com/api/v1/kernels/push` with payload (`slug`, `newTitle`, `text`, `enableGpu=True`, `enableInternet=True`, `isPrivate=True`).
  - Kernel Status: `GET https://www.kaggle.com/api/v1/kernels/status?userName={username}&kernelSlug={slug}` (`queued` | `running` | `complete` | `error`).
  - Kernel Output Download: `GET https://www.kaggle.com/api/v1/kernels/output?userName={username}&kernelSlug={slug}`.
  - Active execution watchdog (8.5h safety ceiling) to prevent timeout without checkpoint saving.
- [x] Implement local rolling 7-day quota ledger in `frame/runners/kaggle/quota.py` (`.frame/kaggle_quota.json`) tracking GPU hours (30h weekly limit).
- [x] Write unit tests with mock HTTP responses in `tests/test_kaggle_client.py`.

## Phase 2: Tiered Symbolic-Numeric Verification Engine (L0, L1, L2)
- [x] Fix Metric Directionality in `frame/verification/statistics.py` (Completed & Tested: supports `MAXIMIZE` and `MINIMIZE`).
- [x] Implement Dimensional Compression Mappings:
  - Symmetric / Pairwise projection (Softmax over $\mathbb{R}^d \to$ Logistic 1D).
  - Invariant variable reduction (RMSNorm $s = \text{mean}(x^2) \in [0, \infty)$).
  - Radial projection & Spectral envelopes ($W \to [\sigma_{\min}, \sigma_{\max}]$).
- [x] Level 0 (Fast Analytical SymPy Path, $<200$ ms) in `frame/verification/math_engine.py`:
  - Analytic differentiability and positive domain assumptions (`positive=True`).
  - Closed-form pole / singularity certification ($\text{denom}(x) \neq 0, \text{arg}_{\log}(x) > 0$).
  - Asymptotic growth and gradient limits ($\lim_{x \to \pm\infty} f(x), \lim_{x \to \pm\infty} f'(x)$).
- [x] Level 1 (Symbolic-Numeric Hybrid System, $<1.5$ s) in `frame/verification/math_engine.py`:
  - PyTorch Dynamo / AOTAutograd graph tracing with AST fallback.
  - Forward Interval Bound Propagation (IBP): propagate $[l_i, u_i]$ under $\|x\|_\infty \le B$ to certify non-zero denominator intervals.
  - Lanczos spectral power iteration for operator norms $\|W\|_2$ and SVD condition number $\kappa(W)$.
- [x] Level 2 (Formal Escalation & Interval Branch-and-Bound, $<8$ s) in `frame/verification/math_engine.py`:
  - Pure-Python recursive Interval Branch-and-Bound (IBnB) solver using `mpmath` interval arithmetic and `scipy.optimize`.
  - Evaluate violation predicate $\exists x \in \mathcal{B}_\infty(\mathbf{0}, R) \text{ s.t. } \text{Violation}(f(x)) = \text{True}$.
  - Certify property or extract concrete counterexample tensor $x_{\text{fail}}$ for Falsification Memory.

## Phase 3: Anti-Reward Hacking & Adversarial Defense Architecture
- [x] Public vs. Sealed Seed Protocol (HMAC Commitment):
  - Refactor `Hypothesis` schema: remove public hardcoded seeds. Provide $S_{\text{pub}} = [101, 102]$ for local drafting only.
  - Verifier generates sealed seeds $S_{\text{sealed}}$ at dispatch via $s_i = \text{HMAC-SHA256}(K_{\text{verifier}}, \text{hypothesis\_id} \parallel i) \pmod{2^{31} - 1}$.
- [x] Decoupled Isolated Verifier Harness in `frame/verification/harness.py`:
  - Candidate code supplies only model weights/forward logic.
  - Verifier loads weights into isolated sub-process, feeds private holdout split $\mathcal{D}_{\text{holdout}}$ under `torch.inference_mode()`, and computes metrics with trusted verifier functions.
- [x] Adversarial Red-Team Degeneracy Suite in `frame/verification/red_team.py`:
  - Mandatory evaluation against 5 null models: (1) Zero $f \equiv 0$, (2) Constant $f \equiv \bar{y}$, (3) Random $f \sim \mathcal{N}(0, I)$, (4) Identity $f(x) = x$, (5) Label-permuted model.
  - Hard gate: candidate must beat $\max(\text{Score}_{\text{null}})$ by $\Delta_{\text{null}} > \epsilon_{\text{robust}}$.
- [x] Causal Mask Lookahead Leakage Probe:
  - Assert future-token gradient $\frac{\partial y_t}{\partial x_{t+k}} \equiv 0$ in autoregressive architectures.
- [x] Verifier-Signed Claim-Evidence Binding in Merkle CAS:
  - Verifier commits intermediate activation hashes at steps $t \in \{1, 10, 50\}$ to append-only Merkle tree.

## Phase 4: Explicit Experimental Design Stage
- [x] Dynamic Power Analysis & Sample Sizing in `frame/brain/experimental_design.py`:
  - Calculate required seeds $k^*$ using non-central Student's $t$-distribution for target power $\beta = 0.80$ under current LORD threshold $\alpha_t$.
  - Enforce variance reduction via strict Common Random Numbers (CRN) synchronization (batch order, augmentations, projections), compressing $\sigma_D$ by $4\times$ to $10\times$.
- [x] Two-Stage Group Sequential Screening:
  - Stage 1 (Futility Screening at $k=3$): abort immediately if $\bar{D} < 0$, pruning $\sim 70\%$ suboptimal branches at negligible cost.
  - Stage 2 (Confirmatory Expansion): expand to $k^*$ seeds on Tier 2 Kaggle for LORD 3 discovery evaluation.
- [x] Quad-Baseline Control Protocol:
  - Synthesize $C_{\text{FLOP}}$ (matched-FLOPs), $C_{\text{param}}$ (matched-parameters), $C_{\text{ablate}}$ (core mutation ablation), and $C_{\text{neg}}$ (inverted/scrambled control).
- [x] Method-to-Implementation Alignment Verification:
  - Structural AST diff verification between baseline and candidate.
  - Active gradient flow audit: assert new parameters have `requires_grad=True`, $\|\nabla \theta_{\text{new}}\|_2 > 10^{-8}$, and $\frac{\partial \mathcal{L}}{\partial \theta_{\text{new}}} \neq 0$.
  - Abort if alignment score $< 0.85$ (`METHOD_CODE_MISALIGNMENT`).

## Phase 5: Concrete Research Brain & Search-Level Accounting
- [x] Epistemic MCTS Engine in `frame/brain/state.py` & `frame/brain/thinking.py`:
  - Implement `MCTSNode` schema with `SearchStage` lifecycle and resource accounting.
  - Set UCT exploration constant $c = \sqrt{2} \approx 1.414$ with adaptive schedule $c(t) = 1.414 \cdot (0.95)^{\lfloor t/25 \rfloor}$.
- [x] Structural Falsification Memory:
  - Canonical AST fingerprinting (stripping variable names, docstrings, line numbers).
  - $O(1)$ hard AST filter matching known fatal failures.
  - Dense k-NN semantic retrieval with mandatory structured evidentiary rebuttal field.
- [x] Recombination Operator Legality Grammar ($C = A \oplus B \oplus D$):
  - Formal compositional typing rules preventing double-softmax gradient crushing, simplex domain mismatches, non-smooth Hessians with 2nd-order optimizers, and tensor rank mismatches.
- [x] Search-Level Accounting & Stage Observability:
  - Implement `SearchAccountingLedger` tracking hypotheses entered, passed, and rejected per stage.
  - Measure discovery yield and mean GPU hours between discoveries for stage-by-stage debugging.

## Phase 6: Incentive Surface & Reward Scope Visualization
- [x] Metric Scope Contract in `frame/feedback/visualization.py`:
  - Publish `MetricScopeContract` for every active benchmark metric: theoretical bounds, population range, saturation index ($> 0.50 \implies$ saturated).
  - Perturbation sensitivity profile under input, parameter, and label noise.
  - Degeneracy risk ratio: pause automated exploration if ratio $> 0.50$.
- [x] 2D Contour Slice Generator & Terminal Exporter:
  - Generate standalone SVG contour plots and ANSI terminal heatmaps across 2D hyperparameter/architecture slices.
  - Stream `METRIC_INCENTIVE_SURFACE` JSON telemetry events.

