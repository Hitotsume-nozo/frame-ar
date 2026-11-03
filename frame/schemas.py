"""Consolidated Pydantic v2 data contracts for FRAME-AR v2."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DomainType(str, Enum):
    KAGGLE_TABULAR = "kaggle_tabular"
    RL_CONTROL = "rl_control"


class TargetDirection(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class StatisticalEstimator(str, Enum):
    PAIRED_T_ROPE = "paired_t_rope"
    IQM_BOOTSTRAP_ROPE = "iqm_bootstrap_rope"


class OperatorRole(str, Enum):
    ERROR_SLICE = "error_slice"
    ABLATION_SIMPLIFY = "ablation_simplify"
    HYPERPARAM_SCHEDULE = "hyperparam_schedule"
    SPEC_IMPLEMENTATION = "spec_implementation"


class EdgeType(str, Enum):
    ATOMIC_TRANSITION = "atomic_transition"
    BLEND_ENSEMBLE = "blend_ensemble"


class TrustGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    FAIL = "FAIL"


class NodeStatus(str, Enum):
    ROOT = "root"
    ROOT_BASELINE = "root_baseline"
    BUGGY = "buggy"
    ABORTED = "aborted"
    GUARDRAIL_REJECTED = "guardrail_rejected"
    FUTILITY_PRUNED = "futility_pruned"
    CONFIRMATORY_NULL = "confirmatory_null"
    PROMOTED_DISCOVERY = "promoted_discovery"


class GuardrailSpec(BaseModel):
    name: str
    op: Literal["<=", ">=", "<", ">"]
    threshold: float
    description: str = ""

    def evaluate(self, value: float) -> bool:
        if self.op == "<=":
            return value <= self.threshold
        if self.op == ">=":
            return value >= self.threshold
        if self.op == "<":
            return value < self.threshold
        if self.op == ">":
            return value > self.threshold
        return False


class ExternalAnchor(BaseModel):
    source_name: str = Field(..., description="External reference source (e.g. Kaggle Leaderboard / CleanRL)")
    reference_url_or_citation: str
    competence_floor: float = Field(..., description="Worst acceptable score for baseline.py to be considered competent")
    published_sota_target: Optional[float] = None


class BenchSpec(BaseModel):
    benchmark_id: str
    title: str
    domain: DomainType
    primary_metric_name: str
    direction: TargetDirection
    estimator: StatisticalEstimator
    rope_epsilon: float = Field(..., gt=0.0)
    blend_rope_epsilon: float = Field(default=0.001, gt=0.0)
    vault_overfit_tolerance: float = Field(default=0.05, gt=0.0)
    pilot_seeds: List[int] = Field(default_factory=lambda: [101, 102, 103])
    confirmatory_k: int = Field(default=5, ge=3, le=15)
    timeout_sec: float = Field(default=60.0, gt=1.0)
    external_anchor: ExternalAnchor
    guardrails: List[GuardrailSpec] = Field(default_factory=list)

    @field_validator("guardrails")
    @classmethod
    def validate_max_four_guardrails(cls, v: List[GuardrailSpec]) -> List[GuardrailSpec]:
        if len(v) > 4:
            raise ValueError("Committee bloat cap: At most 4 guardrails are permitted in BenchSpec.")
        return v


class CommitteeAuditRecord(BaseModel):
    stage1_gemini_blind_review: Optional[str] = None
    stage1_claude_blind_review: Optional[str] = None
    stage2_gemini_on_claude_critique: Optional[str] = None
    stage2_claude_on_gemini_critique: Optional[str] = None
    stage3_human_reviewer: Optional[str] = None
    stage3_human_notes: Optional[str] = None
    approved_at: Optional[str] = None


class BenchVerificationReport(BaseModel):
    benchmark_id: str
    timestamp: str = Field(default_factory=utc_now_iso)
    smoke_passed: bool
    baseline_competent: bool
    null_separation_passed: bool
    positive_control_passed: bool
    sensitivity_monotonic_passed: bool
    expressivity_probe_passed: bool = Field(
        default=True,
        description="Check #7: True if harness evaluates 3 structurally distinct policy architectures without shape/key lock-in",
    )
    baseline_pilot_scores: Dict[str, float]
    baseline_mean: float
    baseline_iqm: float
    positive_control_scores: Dict[str, float]
    positive_control_mean: float
    positive_control_iqm: float
    null_probe_scores: Dict[str, float]
    degradation_ladder_scores: Dict[str, float]
    empirical_sigma_d: float
    recommended_k_star: int
    trust_grade: TrustGrade
    failure_reasons: List[str] = Field(default_factory=list)


class BenchLock(BaseModel):
    benchmark_id: str
    bench_spec_sha256: str
    harness_sha256: str
    baseline_sha256: str
    positive_control_sha256: str
    verification_report_sha256: str
    trust_grade: TrustGrade
    approved_by: str
    approved_at: str = Field(default_factory=utc_now_iso)

    @property
    def approved_by_human(self) -> str:
        return self.approved_by

    @property
    def lock_signature(self) -> str:
        return self.verification_report_sha256[:16]


class DirectiveTicket(BaseModel):
    directive_id: str
    parent_node_id: str
    operator_role: OperatorRole = OperatorRole.SPEC_IMPLEMENTATION
    priority: int = Field(default=0, description="0 is highest priority")
    hypothesis_spec: str
    author: str = "human_or_frontier"
    status: Literal["queued", "consumed"] = "queued"
    created_at: str = Field(default_factory=utc_now_iso)


class NodeRecord(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    branch_id: str = "root"
    depth: int = 0
    edge_type: EdgeType = EdgeType.ATOMIC_TRANSITION
    operator_role: OperatorRole = OperatorRole.SPEC_IMPLEMENTATION
    directive_id: Optional[str] = None
    title: str = ""
    description: str = ""
    status: NodeStatus = NodeStatus.BUGGY
    code_sha256: str = ""
    ast_fingerprint: str = ""
    operator_tokens: List[str] = Field(default_factory=list)
    pilot_scores: Dict[str, float] = Field(default_factory=dict)
    pilot_primary_mean: Optional[float] = None
    pilot_primary_iqm: Optional[float] = None
    pilot_delta_vs_parent: Optional[float] = None
    guardrail_values: Dict[str, float] = Field(default_factory=dict)
    guardrails_passed: bool = False
    failed_guardrails: List[str] = Field(default_factory=list)
    dev_diagnostics: Dict[str, Any] = Field(default_factory=dict)
    confirmatory_step: Optional[int] = None
    confirmatory_seeds: List[int] = Field(default_factory=list)
    confirmatory_challenger_scores: Dict[str, float] = Field(default_factory=dict)
    confirmatory_incumbent_scores: Dict[str, float] = Field(default_factory=dict)
    confirmatory_effect: Optional[float] = None
    confirmatory_ci_lower: Optional[float] = None
    p_value_rope: Optional[float] = None
    allocated_alpha_t: Optional[float] = None
    promoted: bool = False
    error_traceback: Optional[str] = None
    visits: int = 1
    total_reward: float = 0.0
    children_ids: List[str] = Field(default_factory=list)
    wall_clock_sec: float = 0.0
    created_at: str = Field(default_factory=utc_now_iso)
