"""Mathematical verification package for FRAME-AR."""

from frame.verification.statistics import (
    LORDController,
    evaluate_paired_seed_rope,
    gamma_lord,
    ZETA_1_6,
)
from frame.verification.novelty import (
    DualKeyNoveltyContract,
    NoveltyVerdict,
    NoveltyContractResult,
)
from frame.verification.invariants import (
    SymbolicDomainConstraint,
    DomainInvariantPlugin,
    PhysicsConservationPlugin,
    BiologyMonotonicityPlugin,
    BoundaryConditionsPlugin,
    DefaultPhysicsPlugin,
    DefaultBiologyPlugin,
    DefaultBoundaryPlugin,
)
from frame.verification.domain_evidence import (
    DataProvenance,
    DomainEvidenceVerifier,
    APPROVED_LICENSES,
    DEFAULT_HIERARCHY_WEIGHTS,
)
from frame.verification.tapa import (
    compute_tapa_score,
    compute_tapa_score_modality_b,
)

__all__ = [
    "LORDController",
    "evaluate_paired_seed_rope",
    "gamma_lord",
    "ZETA_1_6",
    "DualKeyNoveltyContract",
    "NoveltyVerdict",
    "NoveltyContractResult",
    "SymbolicDomainConstraint",
    "DomainInvariantPlugin",
    "PhysicsConservationPlugin",
    "BiologyMonotonicityPlugin",
    "BoundaryConditionsPlugin",
    "DefaultPhysicsPlugin",
    "DefaultBiologyPlugin",
    "DefaultBoundaryPlugin",
    "DataProvenance",
    "DomainEvidenceVerifier",
    "APPROVED_LICENSES",
    "DEFAULT_HIERARCHY_WEIGHTS",
    "compute_tapa_score",
    "compute_tapa_score_modality_b",
]

