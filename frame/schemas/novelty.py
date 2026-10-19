"""Data schemas and contracts for the Dual-Key Novelty Contract (Milestone M2).

Evaluates candidate hypotheses across two orthogonal axes:
1. Literature Graph Proximity (d_lit > tau_lit)
2. Functional Prediction Manifold Divergence (d_func > epsilon_func)
"""

from datetime import datetime, timezone
from enum import Enum
import hashlib
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, model_validator

from frame.utils.canonical_json import canonical_json_dumps


class NoveltyVerdict(str, Enum):
    """Verdicts emitted by the Dual-Key Novelty Contract gate."""

    APPROVED = "APPROVED"
    REJECTED_LIT_PROXIMITY = "REJECTED_LIT_PROXIMITY"
    REJECTED_FUNC_COLLINEAR = "REJECTED_FUNC_COLLINEAR"
    REJECTED_DUAL_FAILURE = "REJECTED_DUAL_FAILURE"

    @classmethod
    def _missing_(cls, value: object):
        if isinstance(value, str):
            for member in cls:
                if member.value == value.upper() or member.name == value.upper():
                    return member
        return super()._missing_(value)


class NoveltyContractResult(BaseModel):
    """Immutable audit record and outcome of Dual-Key Novelty evaluation."""

    hypothesis_id: str = Field(..., description="Hypothesis identifier under audit")
    passed: bool = Field(..., description="Whether the hypothesis passed both novelty keys strictly")
    verdict: NoveltyVerdict = Field(..., description="Verdict outcome of the dual-key evaluation")
    d_lit: float = Field(..., ge=0.0, le=1.0, description="Semantic literature distance to nearest prior art")
    tau_lit: float = Field(default=0.15, ge=0.0, le=1.0, description="Threshold for literature proximity")
    nearest_prior_art: str = Field(default="", description="Nearest prior art item from the corpus")
    d_func: float = Field(..., ge=0.0, le=1.0, description="Functional divergence metric: 1 - |rho|")
    epsilon_func: float = Field(default=0.10, ge=0.0, le=1.0, description="Threshold for functional divergence")
    residual_correlation: float = Field(default=0.0, ge=-1.0, le=1.0, description="Residual manifold correlation: rho")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic breakdown and metadata")
    rejection_reason: Optional[str] = Field(default=None, description="Human-readable rejection reason if rejected")
    contract_hash: Optional[str] = Field(default=None, description="SHA-256 hash of canonical JSON representation")
    created_at: Optional[str] = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of audit"
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support alias rho -> residual_correlation
            if "residual_correlation" not in data and "rho" in data:
                data["residual_correlation"] = data["rho"]
            # Support alias nearest_prior_reference -> nearest_prior_art
            if "nearest_prior_art" not in data and "nearest_prior_reference" in data:
                data["nearest_prior_art"] = data["nearest_prior_reference"]
            # Support alias timestamp -> created_at
            if "created_at" not in data and "timestamp" in data:
                data["created_at"] = data["timestamp"]
            # Synchronize rejection_reason with details
            if not data.get("rejection_reason") and isinstance(data.get("details"), dict):
                data["rejection_reason"] = data["details"].get("rejection_reason")
            elif data.get("rejection_reason") and isinstance(data.get("details"), dict):
                if "rejection_reason" not in data["details"]:
                    data["details"]["rejection_reason"] = data["rejection_reason"]
        return data

    def compute_hash(self) -> str:
        """Computes deterministic SHA-256 hash of canonical JSON excluding contract_hash."""
        data = self.model_dump(mode="json", exclude={"contract_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()

    @property
    def rho(self) -> float:
        """Alias for residual_correlation."""
        return self.residual_correlation

    @property
    def nearest_prior_reference(self) -> str:
        """Alias for nearest_prior_art."""
        return self.nearest_prior_art

    @property
    def timestamp(self) -> Optional[str]:
        """Alias for created_at."""
        return self.created_at

    @property
    def is_lit_novel(self) -> bool:
        """True if candidate passed literature proximity key strictly."""
        return self.d_lit > self.tau_lit

    @property
    def is_func_novel(self) -> bool:
        """True if candidate passed functional divergence key strictly."""
        return self.d_func > self.epsilon_func
