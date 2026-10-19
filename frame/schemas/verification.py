import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, model_validator
from frame.utils.canonical_json import canonical_json_dumps


class PairedSeedMetrics(BaseModel):
    metric_name: str
    paired_differences: Dict[str, float]
    mean_difference: float
    standard_error: float
    t_statistic: float


class StatisticalVerificationReport(BaseModel):
    hypothesis_id: str
    metric_name: str
    sample_size_n: int
    mean_baseline: float
    mean_candidate: float
    paired_metrics: PairedSeedMetrics
    rope_epsilon: float
    rope_t_statistic: float
    p_value_rope: float
    lord_step: int
    allocated_alpha_t: float
    is_discovery: bool
    verdict: str  # CONFIRMED_DISCOVERY | CONFIRMED_NULL
    hypothesis_hash: Optional[str] = None
    receipt_hash: Optional[str] = None
    report_hash: Optional[str] = None

    # Edge-indexed FDR fields (F2 Residuals & Graph Routing)
    edge_id: Optional[str] = None
    source_hypothesis_id: Optional[str] = None
    target_hypothesis_id: Optional[str] = None
    source_node_id: Optional[str] = None
    target_node_id: Optional[str] = None
    edge_type: str = "atomic_transition"  # "atomic_transition" | "blend_ensemble" | "cumulative_campaign"
    tapa_score: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _populate_edge_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            hyp_id = data.get("hypothesis_id")
            if hyp_id:
                if not data.get("target_hypothesis_id"):
                    data["target_hypothesis_id"] = hyp_id
                if not data.get("edge_id"):
                    src = data.get("source_hypothesis_id") or "baseline"
                    tgt = data.get("target_hypothesis_id") or hyp_id
                    data["edge_id"] = f"edge_{src}_to_{tgt}"
        return data

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"report_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()

