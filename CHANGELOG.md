# Changelog — FRAME-AR

All notable changes to the FRAME-AR platform are documented in this file.
The project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.5.0] - Stable Release - 2026-11-03

### Summary
FRAME-AR v2.5.0 marks the first **production-stable release** of the v2 out-of-process autonomous research engine. This release solves three fundamental scaling and tree-search bottlenecks discovered during initial empirical campaigns on nonlinear RL benchmarks (`wind_pendulum_rl`), establishing robust baseline calibration, ancestor pruning, and convergence detection.

### Added
- **Dynamic Baseline-Anchored Timeout Calibration (`frame.bench.verifier`)**:
  - Replaced fixed static `30.0s` timeouts with empirical runtime profiling during Phase 0 (`frame bench verify`).
  - Auto-calibrates execution timeout based on observed per-seed wall-clock times of `baseline.py` ($T_{\text{base}}$) and `positive_control.py` ($T_{\text{pos}}$):
    $$T_{\text{timeout}} = \text{clip}\left(8.0 \times \max(T_{\text{base}}, T_{\text{pos}}), \; 45.0\text{s}, \; 600.0\text{s}\right)$$
  - Updates `bench_spec.json` automatically during verification.
- **Gate-0 Runtime Projection Guard (`frame.cli:cmd_smoke_check`)**:
  - Measures Gate-0 smoke execution time $T_{\text{smoke}}$ and projects full-split runtime.
  - If $T_{\text{smoke}} > \max(5.0\text{s}, 0.22 \times T_{\text{timeout}})$, fails fast with exit code `5` (`GATE0_RUNTIME_PROJECTION_EXCEEDED`).
  - Forces workers to vectorize or optimize loops in the rapid local Gate-0 retry loop rather than hanging on full multi-seed confirmatory splits in Gate 1.
- **Ancestor Retirement & Anti-Bleed UCT Selection (`frame.brain.tree`)**:
  - Prevents UCT search from regressing to obsolete ancestor nodes when active child branches have already achieved superior performance.
  - Any node that has spawned at least one promoted discovery and trails the reigning Incumbent $M^*$ by $> 2.0 \times \epsilon_{\text{ROPE}}$ is retired from future UCT task selection.
- **Deep Disk Synchronization (`frame.brain.tree:CampaignTree.record_node`)**:
  - Fixed backpropagation loop so updated `visits`, `total_reward`, and `children_ids` are synchronized to disk across all ancestor levels (`nodes/<id>/node_record.json`).
- **Branch Exhaustion Penalty (`frame.brain.tree`)**:
  - Adds a dynamic penalty to UCT score after 4 consecutive failed child mutations (covering a full cycle of all 4 mechanical operators), redirecting search to alternative promoted branches.
- **Campaign Stagnation Early-Stop (`frame-orchestrate.sh` & `frame.brain.tree`)**:
  - Added `get_steps_since_last_promotion()` tracking consecutive non-promoted steps across the campaign.
  - Added `--max-stagnant-steps` (default: 5) to `frame-orchestrate.sh`. Halts MCTS search early if search stagnates without human intervention, transitioning cleanly to Phase 5 Vault-Holdout audit.

### Verified
- 100% test pass rate across 29 unit, integration, and end-to-end test suites (`pytest -q`).

---

## [2.4.0] - 2026-11-01

### Added
- **Open Policy Expressivity Contract (`select_action`)**:
  - Decoupled policy evaluation logic from `harness.py` into candidate code via `def select_action(obs: np.ndarray, policy_state: dict) -> np.ndarray:`.
  - Enables arbitrary policy architectures (linear, polynomial, multi-layer MLPs, non-parametric lookup tables) without harness shape lock-in.
- **Restricted AST Compilation in Audit Sandbox (`frame.eval.sandbox`)**:
  - Added `bind_candidate_select_action` to compile `select_action` in a sandboxed math namespace (`numpy`, `scipy`, `math`), completely preventing `PicklingError` and disk snooping.
  - Enforced `sys.addaudithook` interception of `sys._getframe` to prevent stack introspection.
- **Phase-0 Check #7: Automated Polymorphic Expressivity Probe (`expressivity_probe_passed`)**:
  - Synthesizes and tests 3 structurally incompatible policy representations during `frame bench verify`.
  - Blocks `bench.lock` if harness exhibits parameter shape or representation lock-in (`REPRESENTATION_LOCK_IN_DETECTED`).
- **Gate-1 Incumbent Superiority Filter (`frame.eval.funnel`)**:
  - Candidates in Gate 1 must beat **both** parent node and reigning Incumbent $M^*$ (`pilot_delta_vs_parent > 0` AND `pilot_delta_vs_incumbent > 0`) before advancing to Gate 2, eliminating wasted LORD-3 alpha wealth on stale branches.

---

## [2.3.0] - 2026-10-30

### Added & Fixed
- **In-Place Jail Mutation Workflow**:
  - Updated `generate_context_brief` in `frame.swarm.brief` with strict Section 6 instructions: workers edit `candidate.py` in-place and immediately exit.
  - Workers prohibited from running slow sub-shells, nested pytest, or manual `frame step` inside jails.
- **Short-Argv Invocation**:
  - Reduced `agy` CLI prompt to <200 bytes in `frame-orchestrate.sh`, preventing command-line buffer overflows.
- **Grade-A Nonlinear RL Benchmark Template**:
  - Added permanent DARE LQR + Powell nonlinear controller template for `wind_pendulum_rl` with BLAKE2b salt and 4 empirical guardrails.

---

## [2.2.0] - 2026-10-30

### Fixed
- **Stateless Stdin/Stdout Worker Pipe**:
  - Resolved `GATE0_AST_DUPLICATE` failures caused by headless `agy -p` stdout output when `/dev/null` was bound to stdin.
  - Implemented `extract-code` CLI subcommand supporting in-place edits, fenced ```python code blocks, and raw code extraction.
- **Persistent Worker Telemetry**:
  - All attempt stdout, stderr, and smoke logs saved to `.frame/campaigns/<topic>/logs/step_<N>_<timestamp>/`.
- **MCTS Tree Hygiene**:
  - Skipped step recording when worker fails to modify code across all Gate-0 retries, preventing dummy `buggy` node pollution.

---

## [2.1.0] - 2026-10-30

### Added
- **Static UNIX Bash Hypervisor (`frame-orchestrate.sh`)**:
  - Top-level OS hypervisor managing process lifecycle, `mktemp -d` isolation, timeout handling, and Phase 0 manual webapp review loops.
- **Out-of-Band Monitor & Hypothesis Injector (`frame-mon.py`)**:
  - Terminal status monitor and human-in-the-loop directive injector.
  - Lock-free single-writer IPC maildir pattern (`signals/` and `directives/`).
- **Riemann Zeta LORD-3 Decay Schedule (`frame.eval.statistics`)**:
  - Restored exact polynomial Riemann Zeta ($\zeta(1.6) \approx 2.285766$) decay schedule ($w_0 = 0.025, \alpha_1 = 0.010937$).

---

## [2.0.0] - 2026-10-29

### Complete Architectural Redesign (Overhaul from v1)
- **Strict Separation of Powers**: Complete architectural decoupling between host evaluation environment (`frame` CLI) and candidate code (`candidate.py`).
- **Out-of-Process CPython Audit Sandbox**: All candidate execution isolated in child processes with strict `sys.addaudithook` preventing file, network, and permission tampering.
- **3-Gate Evaluation Funnel**:
  - Gate 0: AST deduplication via `OperatorASTMemory` (zero compute).
  - Gate 1: Dev Diagnostic screening on fast Common Random Numbers (CRN) pilot split ($k=3$).
  - Gate 2: Full confirmatory evaluation ($k^*=5..15$) with `rliable` IQM bootstrap and LORD-3 online FDR accounting.
- **Phase 0 Cryptographic Benchmark Lock (`bench.lock`)**:
  - Multi-check empirical qualification with 3-stage blind-to-crossed committee review.
  - RFC 8785 Canonical JSON SHA-256 seal.
- **Blend Edge Protocol**:
  - Dedicated screening of ensemble models against the incumbent best single model ($M^*$).
- **Phase 5 Vault-Holdout Generalization Audit**:
  - Sealed holdout split evaluated only after campaign completion.

---

## [1.0.0-archive] - 2026-10-29

### Preserved State
- **Archive Commit**: `1a55a63`
- **Git Branch**: `archive/v1-main`
- **Git Tag**: `v1.0.0-archive`
- **Verification Status**: 793 passed unit and integration tests.
- **Archive Manifest**: Documented in `ARCHIVE_V1_MANIFEST.md` and packaged in `frame_v1_preserved_state.tar.xz`.
- Legacy in-process monolithic architecture preserved for reference.
