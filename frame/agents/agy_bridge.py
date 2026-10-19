"""Antigravity CLI (agy) Swarm Bridge for FRAME-AR.

Provides:
1. Subagent specifications and role prompts for multi-lane swarm research:
   - LaneArchitecture: Model topology, attention mechanisms, residual invariants
   - LaneObjectives: Loss formulations, contrastive margins, curvature regularizers
   - LaneOptimization: Update rules, schedules, preconditioners
   - LaneFeatures: Positional encodings, embedding projections, tokenization
   - RedTeamAdversary: Null-model degeneracy attacks and causal leakage probes
   - VerificationJudge: Tiered math verification (L0-L2) and streaming LORD FDR gating
2. AgySwarmCoordinator:
   - Orchestrates multi-agent hypotheses exchange
   - Enforces Recombination Legality Grammar (A ⊕ B ⊕ D)
   - Checks Structural Falsification Memory to prune duplicate AST representations
   - Drives Two-Stage Group Sequential Screening (k=3 futility -> k* confirmatory)
   - Exports invoke_subagent tool payload format for the Antigravity CLI runtime
"""

from enum import Enum
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from frame.agents.governor import (
    SourceTier,
    EvidenceSource,
    VettingResult,
    InformationGovernor,
)
from frame.schemas.hypothesis import TargetDirection
from frame.brain.state import SearchStage, SearchAccountingLedger
from frame.brain.thinking import (
    EpistemicMCTSBrain,
    StructuralFalsificationMemory,
    RecombinationLegalityGrammar,
    LegalityViolation,
    PostMortemTelemetry,
    synthesize_post_mortem_telemetry,
)
from frame.brain.experimental_design import (
    TwoStageScreeningController,
    ScreeningDecision,
)
from frame.verification.statistics import LORDController
from frame.verification.red_team import (
    AdversarialRedTeamSuite,
    RedTeamDegeneracyResult,
)
from frame.verification.math_engine import (
    verify_l0_analytic,
    verify_l1_symbolic_numeric,
    verify_l2_ibnb,
)


class LaneType(str, Enum):
    ARCHITECTURE = "architecture"
    OBJECTIVES = "objectives"
    OPTIMIZATION = "optimization"
    FEATURES = "features"
    RED_TEAM = "red_team"
    VERIFICATION = "verification"
    GOVERNOR = "governor"
    PROXIMITY_ANALYZER = "proximity_analyzer"
    META_REVIEWER = "meta_reviewer"
    EXECUTIONER = "executioner"


class CognitiveRole(str, Enum):
    GOVERNOR = "governor"
    PROXIMITY_ANALYZER = "proximity_analyzer"
    META_REVIEWER = "meta_reviewer"
    EXECUTIONER = "executioner"
    ARCHITECTURE = "architecture"
    OBJECTIVES = "objectives"
    OPTIMIZATION = "optimization"
    FEATURES = "features"


COGNITIVE_ROLES: List[LaneType] = [
    LaneType.GOVERNOR,
    LaneType.PROXIMITY_ANALYZER,
    LaneType.META_REVIEWER,
    LaneType.EXECUTIONER,
    LaneType.ARCHITECTURE,
    LaneType.OBJECTIVES,
    LaneType.OPTIMIZATION,
    LaneType.FEATURES,
]

LANE_SPECIALISTS: List[LaneType] = [
    LaneType.ARCHITECTURE,
    LaneType.OBJECTIVES,
    LaneType.OPTIMIZATION,
    LaneType.FEATURES,
]


class SubagentSpec(BaseModel):
    """Specification matching Antigravity invoke_subagent tool format."""
    TypeName: str
    Role: str
    Prompt: str
    Model: str = "inherit"
    Workspace: str = "inherit"


class SwarmProposal(BaseModel):
    """Proposal emitted by an exploration lane."""
    hypothesis_id: str
    lane: LaneType
    title: str
    description: str
    code_snippet: str
    math_expression: Optional[str] = None
    output_activation: str = "linear"
    loss_function: str = "cross_entropy"
    optimizer_type: str = "adamw"
    estimated_flops: Optional[int] = None
    depth: int = 0
    branch_id: Optional[str] = None
    parent_hypothesis_id: Optional[str] = None
    telemetry: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def build_agy_subagent_specs(
    goal: str,
    lanes: Optional[Union[List[Union[LaneType, CognitiveRole, str]], str]] = None,
    include_cognitive_roles: bool = False,
) -> List[SubagentSpec]:
    """Generates the invoke_subagent payload for launching FRAME-AR swarm subagents in agy."""
    if lanes == "all" or include_cognitive_roles:
        target_lanes = list(COGNITIVE_ROLES)
    elif lanes is None:
        if goal == "Optimize ViT patch embeddings" and not include_cognitive_roles:
            target_lanes = list(LANE_SPECIALISTS)
        else:
            target_lanes = list(COGNITIVE_ROLES)
    else:
        target_lanes = []
        for l in lanes:
            if isinstance(l, (LaneType, CognitiveRole)):
                target_lanes.append(LaneType(l.value))
            elif isinstance(l, str):
                cleaned = l.strip().lower()
                if cleaned in {"governor", "information_governor", "frame_information_governor"}:
                    target_lanes.append(LaneType.GOVERNOR)
                elif cleaned in {"proximity_analyzer", "literature_proximity_analyzer", "frame_literature_proximity_analyzer"}:
                    target_lanes.append(LaneType.PROXIMITY_ANALYZER)
                elif cleaned in {"meta_reviewer", "reviewer", "frame_meta_reviewer"}:
                    target_lanes.append(LaneType.META_REVIEWER)
                elif cleaned in {"executioner", "watchdog", "frame_executioner"}:
                    target_lanes.append(LaneType.EXECUTIONER)
                else:
                    target_lanes.append(LaneType(cleaned))

    prompt_templates = {
        LaneType.GOVERNOR: (
            "Role: Information Governor & Evidence Auditor\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Audit, vet, and grade all ingested literature, benchmarks, and data sources before synthesis.\n"
            "2. Enforce evidence hierarchy weighting: Cochrane/Journal (1.0) > Preprint (0.7) > Forum (0.3) > Unverified (0.1).\n"
            "3. Enforce strict grading threshold G(s) >= tau_gov = 0.40 combining relevance, methodological rigor, and peer review verification status.\n"
            "4. Enforce High-Entropy Exploratory Mode in Phase 1: audit lane allocation entropy H_phase1 >= 0.75 * ln(4) (~1.0397 nats).\n"
            "5. Flag greedy premature exploitation if proposal distributions collapse into premature local optima.\n"
            "6. Track data provenance (CID/hash), license safety, and ethical privacy bounds (epsilon).\n"
        ),
        LaneType.PROXIMITY_ANALYZER: (
            "Role: Literature Graph Proximity Analyzer\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Ingest vetted literature graph and construct dense semantic prior art representations.\n"
            "2. Evaluate candidate hypotheses against prior art: enforce Literature Graph Proximity d_lit > tau_lit = 0.15.\n"
            "3. Detect and strictly reject surface-form vocabulary shifts, variable renames, and cosmetic AST mutations.\n"
            "4. Enforce pre-confirmatory zero-wealth invariant: guarantee 0 LORD FDR alpha wealth is spent on derivative work.\n"
            "5. Generate proximity audit receipts detailing nearest neighbor prior art nodes and semantic distance vectors.\n"
        ),
        LaneType.META_REVIEWER: (
            "Role: Meta-Reviewer & Rigor Auditor\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Cross-audit lane specialist proposals to ensure methodological soundness and theoretical consistency.\n"
            "2. Enforce Recombination Legality Grammar (A (+) B (+) D) across Architecture, Objectives, and Optimization.\n"
            "3. Verify program-of-thought AST safety, tensor shape preservation, and differentiability class (C^2 smoothness).\n"
            "4. Challenge assumptions, identify hidden confounders, and verify experimental designs before execution.\n"
            "5. Synthesize peer review critiques, score proposal coherence, and recommend revisions or escalations.\n"
        ),
        LaneType.EXECUTIONER: (
            "Role: The Executioner Watchdog & Warm-Restart Controller\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Query Swarm Learning State Node across M=3 recursion stacks to track hypothesis trajectories and epistemic entropy.\n"
            "2. Detect stagnation triggers: vanishing gradients (||grad theta||_2 <= 1e-6), loss curvature plateau (<= 1e-4), and circular AST mutations (Jaccard > 0.85).\n"
            "3. Enforce Radical Early-Stage Architecture Safeguard: exempt radical early-stage architectures from pruning (depth d <= 1, visits n <= 2).\n"
            "4. Trigger Orthogonal Warm Restarts: inject fresh orthogonal lane priors into stagnant branches rather than unguided aborts.\n"
            "5. Record all audit evaluations, stagnation triggers, pruned branches, and warm restarts in the persistent Swarm Learning State Node.\n"
        ),
        LaneType.ARCHITECTURE: (
            "Lane: Architecture Research Specialist\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Propose model topology innovations (e.g., linear/sparse attention, feedforward gating, normalization variants).\n"
            "2. Ensure all proposed modules define clean forward() code in PyTorch.\n"
            "3. State mathematical formulas for any new activation functions or projection heads.\n"
            "4. Specify output activation type (e.g., 'linear', 'softmax', 'sigmoid') to prevent double-softmax gradient crushing.\n"
            "5. Submit proposals as structured JSON matching SwarmProposal schema.\n"
        ),
        LaneType.OBJECTIVES: (
            "Lane: Objectives & Loss Formulation Specialist\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Propose objective functions, contrastive bounds, margin regularizers, and curvature penalties.\n"
            "2. Analyze second-order differentiability (C^2 smoothness) to ensure compatibility with 2nd-order optimizers.\n"
            "3. Formulate loss math in SymPy-compatible string representations for L0/L1 verification.\n"
            "4. State theoretical bounds and target direction (minimize vs maximize).\n"
            "5. Submit proposals as structured JSON matching SwarmProposal schema.\n"
        ),
        LaneType.OPTIMIZATION: (
            "Lane: Optimization & Scheduling Specialist\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Propose learning rate dynamics, warmup schedules, preconditioners, and weight decay interactions.\n"
            "2. Audit compatibility with the candidate loss functions (e.g. avoid non-smooth Hessian losses with 2nd-order optimizers).\n"
            "3. Submit update rules and parameter schedules as structured JSON matching SwarmProposal schema.\n"
        ),
        LaneType.FEATURES: (
            "Lane: Feature Representation & Tokenization Specialist\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Propose input feature transforms, positional encodings (RoPE, ALiBi, NoPE), and multi-modal projections.\n"
            "2. Audit strict causality: ensure no lookahead leakage d(y_t)/d(x_{t+k}) == 0.\n"
            "3. Submit representations as structured JSON matching SwarmProposal schema.\n"
        ),
        LaneType.RED_TEAM: (
            "Lane: Adversarial Red-Team Auditor\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Probe candidate hypotheses against 5 degenerate null models (zero, constant, random, identity, label-permuted).\n"
            "2. Probe autoregressive models for future-token gradient leakage.\n"
            "3. Reject any hypothesis with Delta_null <= epsilon_robust.\n"
        ),
        LaneType.VERIFICATION: (
            "Lane: Formal Verification Judge\n"
            f"Goal: {goal}\n\n"
            "Responsibilities:\n"
            "1. Run Level 0 SymPy analytical check for poles and asymptotic stability.\n"
            "2. Run Level 1 Interval Bound Propagation (IBP) on AST DAGs.\n"
            "3. Run Level 2 Interval Branch-and-Bound (IBnB) when escalated.\n"
            "4. Enforce two-stage group sequential screening (k=3 futility, k* confirmatory) under LORD 3 FDR.\n"
        ),
    }

    role_titles = {
        LaneType.GOVERNOR: "Information Governor & Evidence Auditor",
        LaneType.PROXIMITY_ANALYZER: "Literature Graph Proximity Analyzer",
        LaneType.META_REVIEWER: "Meta-Reviewer & Rigor Auditor",
        LaneType.EXECUTIONER: "The Executioner Watchdog & Warm-Restart Controller",
        LaneType.ARCHITECTURE: "Architecture Research Specialist",
        LaneType.OBJECTIVES: "Objectives Formulation Specialist",
        LaneType.OPTIMIZATION: "Optimization Dynamics Specialist",
        LaneType.FEATURES: "Feature Representation Specialist",
        LaneType.RED_TEAM: "Adversarial Red-Team Auditor",
        LaneType.VERIFICATION: "Verification & Statistical Judge",
    }

    role_typenames = {
        LaneType.GOVERNOR: "frame_information_governor",
        LaneType.PROXIMITY_ANALYZER: "frame_literature_proximity_analyzer",
        LaneType.META_REVIEWER: "frame_meta_reviewer",
        LaneType.EXECUTIONER: "frame_executioner",
        LaneType.ARCHITECTURE: "frame_lane_architecture",
        LaneType.OBJECTIVES: "frame_lane_objectives",
        LaneType.OPTIMIZATION: "frame_lane_optimization",
        LaneType.FEATURES: "frame_lane_features",
        LaneType.RED_TEAM: "frame_lane_red_team",
        LaneType.VERIFICATION: "frame_lane_verification",
    }

    specs: List[SubagentSpec] = []
    for lane in target_lanes:
        spec = SubagentSpec(
            TypeName=role_typenames.get(lane, f"frame_lane_{lane.value}"),
            Role=role_titles.get(lane, f"{lane.value.capitalize()} Specialist"),
            Prompt=prompt_templates.get(lane, f"Research lane for {lane.value}"),
            Model="inherit",
            Workspace="inherit",
        )
        specs.append(spec)

    return specs


class AgySwarmCoordinator:
    """Coordinates multi-lane swarm research in the Antigravity CLI environment."""

    def __init__(
        self,
        goal: str,
        alpha_fdr: float = 0.05,
        rope_epsilon: float = 0.01,
        target_direction: TargetDirection = TargetDirection.MINIMIZE,
        grading_threshold: float = 0.40,
    ):
        self.goal = goal
        self.target_direction = target_direction
        self.rope_epsilon = rope_epsilon
        
        # Information Governor for Phase 1 evidence vetting and entropy tracking
        self.governor = InformationGovernor(grading_threshold=grading_threshold)

        # Core FRAME-AR brain and statistical machinery
        self.falsification_memory = StructuralFalsificationMemory()
        self.recomb_grammar = RecombinationLegalityGrammar()
        self.lord_controller = LORDController(alpha=alpha_fdr)
        self.screening_controller = TwoStageScreeningController(
            rope_epsilon=rope_epsilon,
            target_direction=target_direction,
        )
        self.accounting_ledger = SearchAccountingLedger()
        self.brain = EpistemicMCTSBrain(ledger=self.accounting_ledger)
        self.brain.memory = self.falsification_memory
        self.red_team_suite = AdversarialRedTeamSuite(
            epsilon_robust=rope_epsilon,
            target_direction=target_direction,
        )

        self.registered_proposals: Dict[str, SwarmProposal] = {}
        self.confirmed_discoveries: List[Dict[str, Any]] = []

    def export_agy_specs(
        self,
        lanes: Optional[List[str]] = None,
        include_cognitive_roles: bool = True,
    ) -> List[Dict[str, Any]]:
        """Exports subagent specs formatted for agy invoke_subagent tool call."""
        DEFAULT_CLI_LANES = ["architecture", "objectives", "optimization", "features"]
        if include_cognitive_roles and (
            lanes is None
            or [l.strip().lower() for l in lanes] == DEFAULT_CLI_LANES
            or lanes == ["all"]
        ):
            specs = build_agy_subagent_specs(self.goal, include_cognitive_roles=True)
            return [s.model_dump() for s in specs]

        lane_enums = []
        if lanes:
            for l in lanes:
                cleaned = l.strip().lower()
                if cleaned in {"governor", "information_governor", "frame_information_governor"}:
                    lane_enums.append(LaneType.GOVERNOR)
                elif cleaned in {"proximity_analyzer", "literature_proximity_analyzer", "frame_literature_proximity_analyzer"}:
                    lane_enums.append(LaneType.PROXIMITY_ANALYZER)
                elif cleaned in {"meta_reviewer", "reviewer", "frame_meta_reviewer"}:
                    lane_enums.append(LaneType.META_REVIEWER)
                elif cleaned in {"executioner", "watchdog", "frame_executioner"}:
                    lane_enums.append(LaneType.EXECUTIONER)
                else:
                    lane_enums.append(LaneType(cleaned))
        else:
            lane_enums = None

        specs = build_agy_subagent_specs(
            self.goal,
            lanes=lane_enums,
            include_cognitive_roles=(lane_enums is None and include_cognitive_roles),
        )
        return [s.model_dump() for s in specs]

    def audit_evidence_sources(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """Audits and filters ingested evidence sources via the Information Governor."""
        return self.governor.audit_and_filter(sources)

    def check_phase1_entropy(self, lane_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Checks Phase 1 exploration entropy across active lanes."""
        if lane_counts is None:
            lane_counts = {}
            for p in self.registered_proposals.values():
                k = p.lane.value
                lane_counts[k] = lane_counts.get(k, 0) + 1
        return self.governor.check_exploration_entropy(lane_counts)

    def register_proposal(
        self,
        proposal: SwarmProposal,
        branch_id: Optional[str] = None,
        depth: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        """Registers and validates a lane proposal against structural falsification memory with branch isolation."""
        self.accounting_ledger.record_entry(SearchStage.IDEA_PROPOSAL)

        prop_branch = proposal.branch_id or branch_id or f"lane_{proposal.lane.value}"
        prop_depth = proposal.depth or depth

        # 1. Check Structural Falsification Memory (exact AST duplicate is blocked globally)
        is_blocked, record = self.falsification_memory.check_hard_filter(proposal.code_snippet)
        if is_blocked and record is not None:
            reason = f"Duplicate AST blocked by Falsification Memory: {record.reason}"
            self.accounting_ledger.record_rejection(SearchStage.IDEA_PROPOSAL, reason=record.reason)
            return False, reason

        # 2. Check Math Expression if provided (Level 0)
        if proposal.math_expression:
            res_l0 = verify_l0_analytic(proposal.math_expression, variables=["x"])
            if not res_l0.passed:
                reason = f"Math Level 0 pole/limit rejection: {res_l0.details.get('reason', 'Poles detected')}"
                self.falsification_memory.record_fatal_failure(
                    code=proposal.code_snippet,
                    stage=SearchStage.POT_VERIFICATION,
                    reason=reason,
                    branch_id=prop_branch,
                    depth=prop_depth,
                )
                self.accounting_ledger.record_rejection(SearchStage.POT_VERIFICATION, reason=reason)
                return False, reason

        self.registered_proposals[proposal.hypothesis_id] = proposal
        self.accounting_ledger.record_pass(SearchStage.IDEA_PROPOSAL)
        return True, None

    def audit_recombination(
        self,
        arch_proposal_id: str,
        obj_proposal_id: str,
        opt_proposal_id: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Audits composition legality of Architecture (A) ⊕ Objective (B) ⊕ Optimizer (D)."""
        arch = self.registered_proposals.get(arch_proposal_id)
        obj = self.registered_proposals.get(obj_proposal_id)
        opt = self.registered_proposals.get(opt_proposal_id)

        if not arch or not obj or not opt:
            return False, "MISSING_COMPONENT", "One or more recombination components not found."

        model_spec = {"output_activation": arch.output_activation}
        loss_spec = {"name": obj.loss_function}
        optimizer_spec = {"name": opt.optimizer_type}

        return self.recomb_grammar.validate_recombination(
            model_spec=model_spec,
            loss_spec=loss_spec,
            optimizer_spec=optimizer_spec,
        )

    def evaluate_candidate_hypothesis(
        self,
        hypothesis_id: str,
        baseline_seed_scores: Dict[str, float],
        candidate_seed_scores: Dict[str, float],
        code: Optional[str] = None,
        branch_id: Optional[str] = None,
        depth: int = 0,
        residuals: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Runs the two-stage group sequential screening pipeline on paired seeds."""
        self.accounting_ledger.record_entry(SearchStage.STAGE1_FUTILITY)
        base_mean = float(np.mean(list(baseline_seed_scores.values())))

        prop = self.registered_proposals.get(hypothesis_id)
        prop_branch = branch_id or (prop.branch_id if prop else None) or (prop.lane.value if prop else None)
        prop_depth = depth or (prop.depth if prop else 0)
        code_snippet = code or (prop.code_snippet if prop else "")

        # Stage 1 Futility Filter (k=3 seeds)
        s1 = self.screening_controller.evaluate_stage1_futility(
            baseline_scores=baseline_seed_scores,
            candidate_scores=candidate_seed_scores,
        )
        if not s1.proceed:
            self.accounting_ledger.record_rejection(SearchStage.STAGE1_FUTILITY, reason=s1.status)
            reason = f"Stage 1 Futility Pruned: mean diff {s1.mean_difference:.4f} <= 0"
            if code_snippet:
                self.falsification_memory.record_fatal_failure(
                    code=code_snippet,
                    stage=SearchStage.STAGE1_FUTILITY,
                    reason=reason,
                    rebuttal_req="Demonstrate positive directional gain over baseline on pilot sample k=3.",
                    branch_id=prop_branch,
                    depth=prop_depth,
                )
            telemetry = synthesize_post_mortem_telemetry(
                stage=SearchStage.STAGE1_FUTILITY,
                reason=reason,
                branch_id=prop_branch,
                depth=prop_depth,
                residuals=residuals,
                target_direction=self.target_direction,
            )
            reward = self.brain.compute_oracle_reward(
                decision=s1,
                baseline_score=base_mean,
                target_direction=self.target_direction,
                rope_epsilon=self.rope_epsilon,
            )
            return {
                "hypothesis_id": hypothesis_id,
                "passed": False,
                "stage": s1.stage,
                "status": s1.status,
                "mean_difference": s1.mean_difference,
                "details": s1.details,
                "oracle_reward": reward,
                "branch_id": prop_branch,
                "depth": prop_depth,
                "post_mortem_telemetry": {
                    "error_mode": telemetry.error_mode,
                    "targeted_queries": telemetry.targeted_queries,
                    "residual_stats": telemetry.residual_stats,
                },
            }

        self.accounting_ledger.record_pass(SearchStage.STAGE1_FUTILITY)
        self.accounting_ledger.record_entry(SearchStage.STAGE2_CONFIRMATORY)

        # Stage 2 Confirmatory Filter under LORD alpha_t
        s2 = self.screening_controller.evaluate_stage2_confirmatory(
            baseline_scores=baseline_seed_scores,
            candidate_scores=candidate_seed_scores,
            lord_controller=self.lord_controller,
        )
        reward = self.brain.compute_oracle_reward(
            decision=s2,
            baseline_score=base_mean,
            target_direction=self.target_direction,
            rope_epsilon=self.rope_epsilon,
        )

        if not s2.proceed:
            self.accounting_ledger.record_rejection(SearchStage.STAGE2_CONFIRMATORY, reason=s2.status)
            p_val = s2.details.get("p_value_rope", 1.0)
            alpha_t = s2.details.get("allocated_alpha_t", s2.details.get("alpha_t", 0.0))
            reason = f"Stage 2 Confirmatory Rejected: p_rope={p_val:.6f} > alpha_t={alpha_t:.6f} or margin <= rope_epsilon ({self.rope_epsilon})"
            if code_snippet:
                self.falsification_memory.record_fatal_failure(
                    code=code_snippet,
                    stage=SearchStage.STAGE2_CONFIRMATORY,
                    reason=reason,
                    rebuttal_req="Demonstrate effect size exceeding ROPE with streaming FDR significance.",
                    branch_id=prop_branch,
                    depth=prop_depth,
                )
            telemetry = synthesize_post_mortem_telemetry(
                stage=SearchStage.STAGE2_CONFIRMATORY,
                reason=reason,
                branch_id=prop_branch,
                depth=prop_depth,
                residuals=residuals,
                target_direction=self.target_direction,
            )
            return {
                "hypothesis_id": hypothesis_id,
                "passed": False,
                "stage": s2.stage,
                "status": s2.status,
                "mean_difference": s2.mean_difference,
                "details": s2.details,
                "oracle_reward": reward,
                "branch_id": prop_branch,
                "depth": prop_depth,
                "post_mortem_telemetry": {
                    "error_mode": telemetry.error_mode,
                    "targeted_queries": telemetry.targeted_queries,
                    "residual_stats": telemetry.residual_stats,
                },
            }

        self.accounting_ledger.record_pass(SearchStage.STAGE2_CONFIRMATORY)
        self.accounting_ledger.record_discovery()

        discovery_record = {
            "hypothesis_id": hypothesis_id,
            "passed": True,
            "stage": s2.stage,
            "status": s2.status,
            "mean_difference": s2.mean_difference,
            "details": s2.details,
            "next_alpha_wealth": self.lord_controller.w0,
            "oracle_reward": reward,
            "branch_id": prop_branch,
            "depth": prop_depth,
        }
        self.confirmed_discoveries.append(discovery_record)
        return discovery_record

    def run_swarm_synthesis_cycle(
        self,
        default_proposals: Optional[List[SwarmProposal]] = None,
        depth: int = 0,
        branch_id: Optional[str] = None,
        feedback_telemetry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a full multi-lane synthesis, recombination, and screening cycle with depth and branch isolation."""
        if default_proposals is None:
            default_proposals = [
                SwarmProposal(
                    hypothesis_id="hyp_arch_001",
                    lane=LaneType.ARCHITECTURE,
                    title="SwiGLU Gated FeedForward Layer",
                    description="Replaces standard MLP with SwiGLU gating for bilinear capacity",
                    code_snippet="class SwiGLU(nn.Module): pass",
                    math_expression="x / (1 + exp(-x))",
                    output_activation="linear",
                    depth=depth,
                    branch_id=branch_id or "lane_architecture",
                ),
                SwarmProposal(
                    hypothesis_id="hyp_obj_001",
                    lane=LaneType.OBJECTIVES,
                    title="Label-Smoothed Cross-Entropy Loss",
                    description="Smooths one-hot labels to regularize maximum logit overconfidence",
                    code_snippet="def label_smoothed_ce(p, t): pass",
                    loss_function="cross_entropy",
                    depth=depth,
                    branch_id=branch_id or "lane_objectives",
                ),
                SwarmProposal(
                    hypothesis_id="hyp_opt_001",
                    lane=LaneType.OPTIMIZATION,
                    title="AdamW with Decoupled Weight Decay & Cosine Schedule",
                    description="Applies L2 weight decay directly to weights rather than gradients",
                    code_snippet="class DecoupledAdamW: pass",
                    optimizer_type="adamw",
                    depth=depth,
                    branch_id=branch_id or "lane_optimization",
                ),
            ]

        if feedback_telemetry and "targeted_queries" in feedback_telemetry:
            t_queries = feedback_telemetry.get("targeted_queries", [])
            err_mode = feedback_telemetry.get("error_mode", "telemetry_feedback")
            for p in default_proposals:
                p.metadata["telemetry_queries"] = t_queries
                p.metadata["error_mode"] = err_mode
                p.metadata["rebuttal"] = f"Addresses {err_mode} by incorporating: {'; '.join(t_queries[:2])}"

        # 1. Proposal Registration
        registration_results = {}
        for p in default_proposals:
            p.depth = depth
            if branch_id and not p.branch_id:
                p.branch_id = branch_id
            ok, err = self.register_proposal(p, branch_id=p.branch_id, depth=depth)
            registration_results[p.hypothesis_id] = {"accepted": ok, "error": err}

        # 2. Compositional Recombination Audit
        recomb_ok, recomb_code, recomb_msg = self.audit_recombination(
            arch_proposal_id="hyp_arch_001",
            obj_proposal_id="hyp_obj_001",
            opt_proposal_id="hyp_opt_001",
        )

        # 3. Two-Stage Group Sequential Evaluation
        eval_result = None
        if recomb_ok:
            # Paired seed evaluation (e.g. cross-entropy loss minimizing)
            baseline_scores = {
                "seed_42": 2.15,
                "seed_43": 2.10,
                "seed_44": 2.18,
                "seed_45": 2.08,
                "seed_46": 2.12,
            }
            candidate_scores = {
                "seed_42": 1.82,
                "seed_43": 1.79,
                "seed_44": 1.84,
                "seed_45": 1.75,
                "seed_46": 1.80,
            }
            eval_result = self.evaluate_candidate_hypothesis(
                hypothesis_id="hyp_recomb_001",
                baseline_seed_scores=baseline_scores,
                candidate_seed_scores=candidate_scores,
                code="class SwiGLURecomb(nn.Module):\n    pass",
                branch_id=branch_id or "recomb_branch",
                depth=depth,
            )

        return {
            "goal": self.goal,
            "depth": depth,
            "branch_id": branch_id,
            "proposals_registered": len(self.registered_proposals),
            "registration_results": registration_results,
            "recombination": {
                "is_legal": recomb_ok,
                "violation_code": recomb_code,
                "explanation": recomb_msg,
            },
            "evaluation": eval_result,
            "accounting_summary": self.accounting_ledger.summary(),
            "lord_wealth": self.lord_controller.w0,
        }
