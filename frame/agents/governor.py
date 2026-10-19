"""Information Governor for FRAME-AR.

Audits, vets, and grades literature, benchmarks, and data sources before
hypotheses are synthesized in Phase 1 of the cognitive research swarm.
Also enforces Phase 1 high-entropy exploratory mode to prevent premature
greedy exploitation.
"""

from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class SourceTier(str, Enum):
    """Hierarchy tiers for ingested evidence sources."""
    COCHRANE_OR_JOURNAL = "cochrane_or_journal"
    PREPRINT = "preprint"
    PUBLIC_FORUM = "public_forum"
    UNVERIFIED = "unverified"

    @classmethod
    def from_str(cls, val: str) -> "SourceTier":
        """Normalizes and maps string representations to SourceTier enum."""
        v = val.strip().lower()
        if v in {
            "cochrane_or_journal",
            "cochrane_or_peer_reviewed_journal",
            "cochrane",
            "journal",
            "peer_reviewed",
            "peer_reviewed_journal",
        }:
            return cls.COCHRANE_OR_JOURNAL
        elif v in {"preprint", "arxiv", "biorxiv", "medrxiv"}:
            return cls.PREPRINT
        elif v in {
            "public_forum",
            "forum",
            "kaggle",
            "competition_writeup",
            "discussion",
            "reddit",
        }:
            return cls.PUBLIC_FORUM
        elif v in {"unverified", "blog", "scratchpad", "unknown", "anonymous"}:
            return cls.UNVERIFIED
        else:
            try:
                return cls(v)
            except ValueError:
                return cls.UNVERIFIED


# Canonical evidence hierarchy weights w_hier(T(s))
DEFAULT_HIERARCHY_WEIGHTS: Dict[SourceTier, float] = {
    SourceTier.COCHRANE_OR_JOURNAL: 1.0,
    SourceTier.PREPRINT: 0.7,
    SourceTier.PUBLIC_FORUM: 0.3,
    SourceTier.UNVERIFIED: 0.1,
}


class EvidenceSource(BaseModel):
    """Schema representing an ingested scientific or empirical evidence source."""
    source_id: str = Field(default_factory=lambda: f"src_{uuid.uuid4().hex[:8]}")
    title: str
    source_type: SourceTier
    peer_reviewed: bool = False
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    methodological_rigor: float = Field(default=0.5, ge=0.0, le=1.0)
    abstract_text: str = ""
    doi_or_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_type", mode="before")
    @classmethod
    def coerce_source_type(cls, v: Any) -> SourceTier:
        if isinstance(v, str):
            return SourceTier.from_str(v)
        return v


class VettingResult(BaseModel):
    """Schema representing the vetting and grading verdict for an evidence source."""
    source_id: str
    title: str = ""
    source_type: Optional[SourceTier] = None
    grade: float
    accepted: bool
    rejection_reason: Optional[str] = None
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class InformationGovernor(BaseModel):
    """Active Information Governor for literature vetting and exploratory entropy enforcement.

    Responsibilities:
    1. Audits, vets, and grades literature, benchmarks, and data sources:
       G(s) = w_hier(T(s)) * (0.40 * r(s) + 0.40 * m(s) + 0.20 * v(s))
       where:
         - w_hier(T(s)): Hierarchy weight (1.0 Cochrane/journal, 0.7 preprint, 0.3 forum, 0.1 unverified)
         - r(s): Relevance score in [0.0, 1.0]
         - m(s): Methodological rigor in [0.0, 1.0]
         - v(s): Peer review indicator in {0.0, 1.0}
       Accepts sources with G(s) >= tau_gov = 0.40.
    2. Enforces Phase 1 high-entropy exploratory mode across research lanes:
       H_phase1 = -sum(p_k * ln(p_k)) >= 0.75 * ln(4)
       Flags greedy premature exploitation if H_phase1 < gamma * ln(K).
    """

    grading_threshold: float = 0.40
    entropy_threshold_ratio: float = 0.75
    num_expected_lanes: int = 4
    tier_weights: Dict[SourceTier, float] = Field(default_factory=lambda: dict(DEFAULT_HIERARCHY_WEIGHTS))
    weight_relevance: float = 0.40
    weight_rigor: float = 0.40
    weight_peer_review: float = 0.20

    audit_history: List[VettingResult] = Field(default_factory=list)
    event_logs: List[Dict[str, Any]] = Field(default_factory=list)

    def grade_source(
        self,
        source: EvidenceSource,
        record: bool = True,
        state_node: Optional[Any] = None,
    ) -> VettingResult:
        """Grades an ingested evidence source against the evidence hierarchy standard.

        Formula:
            G(s) = w_hier(T(s)) * (w_r * r(s) + w_m * m(s) + w_v * v(s))
        """
        w_hier = self.tier_weights.get(source.source_type, 0.1)
        v = 1.0 if source.peer_reviewed else 0.0
        r = float(source.relevance_score)
        m = float(source.methodological_rigor)

        inner_score = (
            self.weight_relevance * r
            + self.weight_rigor * m
            + self.weight_peer_review * v
        )
        grade = round(float(w_hier * inner_score), 4)

        accepted = grade >= self.grading_threshold
        rejection_reason = None
        if not accepted:
            rejection_reason = "LOW_EVIDENCE_QUALITY"

        result = VettingResult(
            source_id=source.source_id,
            title=source.title,
            source_type=source.source_type,
            grade=grade,
            accepted=accepted,
            rejection_reason=rejection_reason,
            raw_scores={
                "w_hier": w_hier,
                "relevance": r,
                "rigor": m,
                "peer_reviewed_val": v,
                "inner_score": round(inner_score, 4),
            },
        )

        log_entry = {
            "event_type": "source_vetting",
            "source_id": source.source_id,
            "title": source.title,
            "tier": source.source_type.value,
            "grade": grade,
            "accepted": accepted,
            "rejection_reason": rejection_reason,
            "timestamp": result.timestamp,
        }

        if record:
            self.audit_history.append(result)
            self.event_logs.append(log_entry)

        if state_node is not None and hasattr(state_node, "governor_event_logs") and isinstance(state_node.governor_event_logs, list):
            state_node.governor_event_logs.append(dict(log_entry))

        return result


    def audit_and_filter(self, sources: List[EvidenceSource]) -> List[EvidenceSource]:
        """Audits a batch of evidence sources, updates audit history, and returns only accepted sources."""
        filtered: List[EvidenceSource] = []
        for s in sources:
            res = self.grade_source(s, record=True)
            if res.accepted:
                filtered.append(s)
        return filtered

    def check_exploration_entropy(
        self,
        lane_counts: Dict[str, int],
        strict: bool = False,
        state_node: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Computes Phase 1 exploratory epistemic entropy and detects greedy premature exploitation.

        H_phase1 = -sum_{k=1}^K p_k * ln(p_k)
        Requires:
            H_phase1 >= gamma * ln(K) (default: 0.75 * ln(4) = 1.0397)
        """
        total = sum(lane_counts.values())
        k_lanes = self.num_expected_lanes
        max_entropy = round(float(math.log(k_lanes)), 4)
        threshold = round(float(self.entropy_threshold_ratio * math.log(k_lanes)), 4)

        if total == 0:
            h_phase1 = 0.0
            probs: Dict[str, float] = {}
            normalized_entropy = 0.0
        else:
            h_val = 0.0
            probs = {}
            for lane, count in lane_counts.items():
                if count < 0:
                    raise ValueError(f"Lane proposal count cannot be negative: {count} for {lane}")
                p_k = count / total
                probs[lane] = round(float(p_k), 4)
                if p_k > 0.0:
                    h_val -= p_k * math.log(p_k)
            h_phase1 = round(float(h_val), 4)
            normalized_entropy = round(float(h_phase1 / max_entropy), 4) if max_entropy > 0 else 0.0

        is_high_entropy = (h_phase1 >= threshold) and (total > 0)
        greedy_flag = not is_high_entropy

        status = "ENTROPY_SUFFICIENT" if is_high_entropy else "GREEDY_EXPLOITATION_BLOCKED"
        warning = None
        if greedy_flag:
            warning = (
                f"Greedy premature exploitation detected: Phase 1 exploration entropy {h_phase1:.4f} "
                f"is below required threshold {threshold:.4f} "
                f"(gamma={self.entropy_threshold_ratio}, expected_lanes={k_lanes})"
            )

        log_entry = {
            "event_type": "exploration_entropy_audit",
            "total_proposals": total,
            "entropy": h_phase1,
            "threshold": threshold,
            "is_high_entropy": is_high_entropy,
            "greedy_exploitation_flag": greedy_flag,
            "lane_probabilities": probs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.event_logs.append(log_entry)

        if state_node is not None and hasattr(state_node, "governor_event_logs") and isinstance(state_node.governor_event_logs, list):
            state_node.governor_event_logs.append(dict(log_entry))

        if strict and greedy_flag:
            raise AssertionError(warning)

        return {
            "passed": is_high_entropy,
            "sufficient": is_high_entropy,
            "is_high_entropy": is_high_entropy,
            "greedy_exploitation_flag": greedy_flag,
            "status": status,
            "entropy": h_phase1,
            "max_entropy": max_entropy,
            "normalized_entropy": normalized_entropy,
            "threshold": threshold,
            "lane_probabilities": probs,
            "total_proposals": total,
            "warning": warning,
        }

    def sync_to_learning_state(self, state_node: Any) -> int:
        """Flushes/appends all recorded governor event logs to state_node.governor_event_logs."""
        if hasattr(state_node, "governor_event_logs") and isinstance(state_node.governor_event_logs, list):
            count = len(self.event_logs)
            for entry in self.event_logs:
                state_node.governor_event_logs.append(dict(entry))
            return count
        return 0


    def get_audit_summary(self) -> Dict[str, Any]:
        """Returns statistical summary of all sources audited by this governor."""
        total = len(self.audit_history)
        accepted = sum(1 for r in self.audit_history if r.accepted)
        rejected = total - accepted
        rate = round(accepted / total, 4) if total > 0 else 0.0
        return {
            "total_audited": total,
            "accepted": accepted,
            "rejected": rejected,
            "acceptance_rate": rate,
            "grading_threshold": self.grading_threshold,
        }

    def clear_audit_history(self) -> None:
        """Clears audit history and event logs."""
        self.audit_history.clear()
        self.event_logs.clear()
