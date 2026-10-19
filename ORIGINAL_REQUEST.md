# Original User Request

## Initial Request — 2026-10-04T21:18:42Z

Use a very large team of agents.

Architect and implement an end-to-end cognitive research overhaul for FRAME-AR in `/home/sparsh/frame`. Transform the framework into a novelty-guaranteed, compounding discovery platform inspired by Google Co-Scientist and AI-Scientist v2, featuring cognition-based swarm roles, an active Information Governor, an Executioner watchdog tracking persistent learning artifacts, and domain evidence standards.

Working directory: /home/sparsh/frame
Integrity mode: development

## Requirements

### R1. Cognition-Based Swarm Roles & Information Governor
Implement specialized cognition-based agent roles in Phase 1 (Information Governor, Literature Proximity Analyzer, Meta-Reviewer, and Lane Specialists). The Information Governor must audit, vet, and grade all ingested literature, benchmarks, and data sources before proposals are synthesized. Phase 1 must operate in high-entropy exploratory mode, preventing greedy premature exploitation.

### R2. Dual-Key Novelty Contract
Formulate and implement an automated Novelty Contract evaluating candidate hypotheses against prior art across two orthogonal axes:
1. Literature Graph Proximity ($d_{\text{lit}} > \tau_{\text{lit}}$) verifying non-trivial distance from existing published literature and competition writeups.
2. Functional Prediction Manifold Divergence ($d_{\text{func}} > \epsilon_{\text{func}}$) verifying non-collinear residual predictions relative to established baselines on holdout distributions.
Proposals that merely re-certify known baselines with surface-form vocabulary shifts must be strictly rejected.

### R3. Persistent Learning Artifact & Test-Time Compute Compounding
Establish a queryable Swarm Learning State Node artifact tracking hypothesis quality, epistemic entropy, and learning trajectories across recursive MCTS generations. The framework must record and verify test-time compute scaling (monotonic progression in the upper envelope of hypothesis quality over search budget $N$).

### R4. The Executioner Watchdog & Orthogonal Warm-Restart Controller
Implement an active Executioner agent that queries the Swarm Learning State Node across $M$ recursion stacks. The Executioner must prune epistemically stagnant branches (vanishing gradients $\|\nabla \theta\|_2 \to 0$, loss curvature plateau, or circular AST mutations with Jaccard similarity $> 0.85$). When stagnation is detected, the Executioner must trigger an Orthogonal Warm Restart (injecting fresh orthogonal lane priors) rather than an unguided abort or prematurely killing radical early-stage architectures.

### R5. Domain Evidence Standards & Invariant Plug-in Architecture
Expand Modality A/B formalization beyond statistical metric matching ($|L_{\text{CV}} - S_{\text{LB}}| \le \tau_{\text{leak}}$) to support domain evidence standards:
1. Evidence Hierarchy Weighting (e.g., peer-reviewed Cochrane/journal $> $ pre-print $> $ public forum).
2. Data Provenance and ethical constraint tracking.
3. Domain Invariant Contracts (e.g., physical conservation laws, biological monotonicity, boundary conditions) formulated as symbolic constraints evaluated by the tiered L0–L2 mathematical verifier.

## Acceptance Criteria

### Unit & Integration Verification
- [ ] Full test suite executes with zero failures: `python3 -m pytest tests/` passes all existing 110 tests plus new test modules covering R1–R5.
- [ ] Novelty Contract test verifies rejection of surface-form rephrased baselines while accepting functionally distinct hypotheses.
- [ ] Executioner watchdog test verifies pruning of gradient-stagnant/circular branches and successful triggering of orthogonal warm restarts.
- [ ] Information Governor test validates grading and filtering of ingested evidence sources.
- [ ] Domain Evidence plugin test confirms verification of symbolic domain invariant constraints alongside L0–L2 math proofs.

### System & CLI Integration
- [ ] `frame status` displays Swarm Learning State, test-time compute scaling curves, and Information Governor / Executioner event logs.
- [ ] `frame swarm --export-agy-specs` exports specifications for all cognitive roles (Governor, Proximity Analyzer, Meta-Reviewer, Executioner, and Lane Specialists).
- [ ] No regressions in existing statistical screeners (LORD-3 streaming FDR, two-stage screening, ROPE superiority, and FlopCounterMode).
