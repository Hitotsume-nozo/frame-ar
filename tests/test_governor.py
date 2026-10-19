"""Unit and Integration Tests for Information Governor & Cognitive Swarm Roles (Milestone M1).

Covers:
1. Evidence Hierarchy Grading: Cochrane / Journal vs Preprint vs Public Forum vs Unverified.
2. Boundary Cases: Exact 0.40 boundary grade, below 0.40 rejection, empty source list.
3. Exploration Epistemic Entropy: Uniform pass, skewed greedy exploitation warning, single-lane 0.0 entropy.
4. Cognitive Subagent Specifications: All 8 roles present in build_agy_subagent_specs and frame swarm --export-agy-specs.
5. Coordinator Integration & Event Logging.
6. Schema validation, normalization, and strict entropy mode.
"""

import json
import math
import numpy as np
import pytest
from typing import Dict, List

from frame.agents.governor import (
    SourceTier,
    EvidenceSource,
    VettingResult,
    InformationGovernor,
    DEFAULT_HIERARCHY_WEIGHTS,
)
from frame.agents.agy_bridge import (
    LaneType,
    CognitiveRole,
    SubagentSpec,
    AgySwarmCoordinator,
    build_agy_subagent_specs,
)
from frame.cli import build_parser, cmd_swarm


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def default_governor() -> InformationGovernor:
    """Returns an InformationGovernor with default grading threshold 0.40."""
    return InformationGovernor(grading_threshold=0.40)


@pytest.fixture
def cochrane_high_rigor_source() -> EvidenceSource:
    """Peer-reviewed Cochrane systematic review with high relevance and rigor."""
    return EvidenceSource(
        source_id="src_cochrane_001",
        title="Systematic Review of Gated Feedforward Activations",
        source_type=SourceTier.COCHRANE_OR_JOURNAL,
        peer_reviewed=True,
        relevance_score=0.85,
        methodological_rigor=0.90,
        abstract_text="Meta-analysis demonstrating bilateral capacity of SwiGLU across 42 trials.",
        doi_or_url="doi:10.1002/14651858.MR000001",
    )


@pytest.fixture
def preprint_strong_source() -> EvidenceSource:
    """Strong preprint with solid empirical methods but not peer-reviewed."""
    return EvidenceSource(
        source_id="src_arxiv_001",
        title="Scaling Laws for Precision-Aware Transformer Preconditioning",
        source_type=SourceTier.PREPRINT,
        peer_reviewed=False,
        relevance_score=0.80,
        methodological_rigor=0.80,
        abstract_text="Empirical investigation of decoupled weight decay under FP8.",
        doi_or_url="arxiv:2401.09999",
    )


@pytest.fixture
def preprint_weak_source() -> EvidenceSource:
    """Preprint with moderate relevance but poor methodological rigor."""
    return EvidenceSource(
        source_id="src_arxiv_002",
        title="Preliminary Thoughts on Attention Without Softmax",
        source_type=SourceTier.PREPRINT,
        peer_reviewed=False,
        relevance_score=0.50,
        methodological_rigor=0.40,
        abstract_text="Single-seed toy demonstration on MNIST.",
        doi_or_url="arxiv:2402.11111",
    )


@pytest.fixture
def public_forum_source() -> EvidenceSource:
    """Public forum post with high relevance but low rigor and no peer review."""
    return EvidenceSource(
        source_id="src_forum_001",
        title="Kaggle Discussion: Use this 1 Weird Trick with AdamW",
        source_type=SourceTier.PUBLIC_FORUM,
        peer_reviewed=False,
        relevance_score=1.00,
        methodological_rigor=0.20,
        abstract_text="I bumped my LB score by 0.003 by setting lr=0.007.",
        doi_or_url="https://kaggle.com/discussion/12345",
    )


@pytest.fixture
def unverified_source() -> EvidenceSource:
    """Unverified claim with no citation or provenance."""
    return EvidenceSource(
        source_id="src_anon_001",
        title="Anonymous Pastebin: SOTA Activation Function Formula",
        source_type=SourceTier.UNVERIFIED,
        peer_reviewed=False,
        relevance_score=0.50,
        methodological_rigor=0.50,
        abstract_text="Trust me this formula is faster.",
    )


# =====================================================================
# 1. Evidence Hierarchy Grading Tests
# =====================================================================

class TestEvidenceGradingHierarchy:
    """Tests evidence scoring across hierarchy tiers (Cochrane > Preprint > Forum > Unverified)."""

    def test_cochrane_high_rigor_source_accepted(
        self, default_governor: InformationGovernor, cochrane_high_rigor_source: EvidenceSource
    ):
        """Cochrane journal source (weight 1.0) with high rigor clears threshold easily."""
        res = default_governor.grade_source(cochrane_high_rigor_source)
        # Expected: 1.0 * (0.40 * 0.85 + 0.40 * 0.90 + 0.20 * 1.0) = 0.34 + 0.36 + 0.20 = 0.90
        assert res.source_id == "src_cochrane_001"
        assert res.accepted is True
        assert res.grade == pytest.approx(0.90, abs=1e-4)
        assert res.rejection_reason is None

    def test_preprint_strong_source_accepted(
        self, default_governor: InformationGovernor, preprint_strong_source: EvidenceSource
    ):
        """Preprint (weight 0.7) with high rigor (0.80) clears the 0.40 threshold."""
        res = default_governor.grade_source(preprint_strong_source)
        # Expected: 0.7 * (0.40 * 0.80 + 0.40 * 0.80 + 0.20 * 0.0) = 0.7 * 0.64 = 0.448
        assert res.source_id == "src_arxiv_001"
        assert res.accepted is True
        assert res.grade == pytest.approx(0.448, abs=1e-4)
        assert res.rejection_reason is None

    def test_preprint_weak_source_rejected(
        self, default_governor: InformationGovernor, preprint_weak_source: EvidenceSource
    ):
        """Preprint (weight 0.7) with lower rigor (0.40) falls below 0.40 threshold."""
        res = default_governor.grade_source(preprint_weak_source)
        # Expected: 0.7 * (0.40 * 0.50 + 0.40 * 0.40 + 0.20 * 0.0) = 0.7 * 0.36 = 0.252
        assert res.source_id == "src_arxiv_002"
        assert res.accepted is False
        assert res.grade == pytest.approx(0.252, abs=1e-4)
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"

    def test_public_forum_source_rejected(
        self, default_governor: InformationGovernor, public_forum_source: EvidenceSource
    ):
        """Public forum (weight 0.3) is discounted and falls below 0.40 threshold."""
        res = default_governor.grade_source(public_forum_source)
        # Expected: 0.3 * (0.40 * 1.00 + 0.40 * 0.20 + 0.20 * 0.0) = 0.3 * 0.48 = 0.144
        assert res.source_id == "src_forum_001"
        assert res.accepted is False
        assert res.grade == pytest.approx(0.144, abs=1e-4)
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"

    def test_public_forum_max_scores_still_rejected(self, default_governor: InformationGovernor):
        """Even with maximum relevance (1.0) and rigor (1.0), forum weight 0.3 caps grade at 0.24."""
        perfect_forum = EvidenceSource(
            source_id="src_forum_max",
            title="Max Score Forum Post",
            source_type=SourceTier.PUBLIC_FORUM,
            peer_reviewed=False,
            relevance_score=1.0,
            methodological_rigor=1.0,
        )
        res = default_governor.grade_source(perfect_forum)
        # Expected: 0.3 * (0.40 * 1.0 + 0.40 * 1.0 + 0.20 * 0.0) = 0.3 * 0.80 = 0.24 < 0.40
        assert res.accepted is False
        assert res.grade == pytest.approx(0.24, abs=1e-4)
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"

    def test_unverified_source_rejected(
        self, default_governor: InformationGovernor, unverified_source: EvidenceSource
    ):
        """Unverified sources (weight 0.1) are heavily penalized and rejected."""
        res = default_governor.grade_source(unverified_source)
        # Expected: 0.1 * (0.40 * 0.50 + 0.40 * 0.50 + 0.20 * 0.0) = 0.1 * 0.40 = 0.04
        assert res.source_id == "src_anon_001"
        assert res.accepted is False
        assert res.grade == pytest.approx(0.04, abs=1e-4)
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"


# =====================================================================
# 2. Boundary Cases Tests
# =====================================================================

class TestGovernorBoundaryCases:
    """Tests exact edge cases: 0.40 boundary, 0.399 rejection, empty lists, custom thresholds."""

    def test_exact_boundary_grade_0_40_accepted(self, default_governor: InformationGovernor):
        """Source scoring exactly 0.40 clears the G(s) >= 0.40 invariant."""
        # Journal (1.0) with r=0.5, m=0.5, v=0.0 -> 1.0 * (0.2 + 0.2) = 0.4000
        boundary_source = EvidenceSource(
            source_id="src_boundary_pass",
            title="Boundary Journal Paper",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=False,
            relevance_score=0.50,
            methodological_rigor=0.50,
        )
        res = default_governor.grade_source(boundary_source)
        assert res.grade == pytest.approx(0.40, abs=1e-5)
        assert res.accepted is True
        assert res.rejection_reason is None

    def test_grade_just_below_0_40_rejected(self, default_governor: InformationGovernor):
        """Source scoring 0.3992 (< 0.40) is strictly rejected."""
        # Journal (1.0) with r=0.498, m=0.500, v=0.0 -> 1.0 * (0.1992 + 0.2000) = 0.3992
        near_boundary_source = EvidenceSource(
            source_id="src_boundary_fail",
            title="Near Boundary Journal Paper",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=False,
            relevance_score=0.498,
            methodological_rigor=0.500,
        )
        res = default_governor.grade_source(near_boundary_source)
        assert res.grade == pytest.approx(0.3992, abs=1e-4)
        assert res.accepted is False
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"

    def test_empty_source_list_audit_and_filter(self, default_governor: InformationGovernor):
        """Filtering an empty list returns an empty list without raising errors."""
        filtered = default_governor.audit_and_filter([])
        assert filtered == []
        assert isinstance(filtered, list)
        assert len(default_governor.audit_history) == 0

    def test_audit_and_filter_mixed_corpus(
        self,
        default_governor: InformationGovernor,
        cochrane_high_rigor_source: EvidenceSource,
        preprint_strong_source: EvidenceSource,
        preprint_weak_source: EvidenceSource,
        public_forum_source: EvidenceSource,
    ):
        """audit_and_filter retains only accepted sources and records all in audit_history."""
        corpus = [
            cochrane_high_rigor_source,
            preprint_strong_source,
            preprint_weak_source,
            public_forum_source,
        ]
        accepted = default_governor.audit_and_filter(corpus)
        assert len(accepted) == 2
        accepted_ids = [s.source_id for s in accepted]
        assert "src_cochrane_001" in accepted_ids
        assert "src_arxiv_001" in accepted_ids
        assert "src_arxiv_002" not in accepted_ids
        assert "src_forum_001" not in accepted_ids
        assert len(default_governor.audit_history) == 4

    def test_custom_grading_threshold_configuration(self):
        """Custom grading threshold (e.g. 0.60) raises the bar for acceptance."""
        strict_gov = InformationGovernor(grading_threshold=0.60)
        source = EvidenceSource(
            source_id="src_moderate",
            title="Moderate Paper",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            peer_reviewed=False,
            relevance_score=0.60,
            methodological_rigor=0.65,
        )
        # Grade: 1.0 * (0.4 * 0.60 + 0.4 * 0.65) = 0.24 + 0.26 = 0.50
        res = strict_gov.grade_source(source)
        assert res.grade == pytest.approx(0.50, abs=1e-4)
        assert res.accepted is False  # 0.50 < 0.60 threshold
        assert res.rejection_reason == "LOW_EVIDENCE_QUALITY"


# =====================================================================
# 3. High-Entropy Exploration Verification Tests
# =====================================================================

class TestExplorationEntropyVerification:
    """Tests Shannon entropy over swarm lane distributions (prevents greedy premature exploitation)."""

    def test_uniform_distribution_passes_entropy_check(self, default_governor: InformationGovernor):
        """Balanced proposals across 4 lanes achieves maximum entropy H = ln(4) and passes."""
        lane_counts = {
            "architecture": 5,
            "objectives": 5,
            "optimization": 5,
            "features": 5,
        }
        res = default_governor.check_exploration_entropy(lane_counts)
        expected_entropy = np.log(4)  # ~1.38629
        expected_threshold = 0.75 * np.log(4)  # ~1.03972

        assert res["passed"] is True
        assert res["sufficient"] is True
        assert res["is_high_entropy"] is True
        assert res["greedy_exploitation_flag"] is False
        assert res["entropy"] == pytest.approx(expected_entropy, rel=1e-3)
        assert res["max_entropy"] == pytest.approx(expected_entropy, rel=1e-3)
        assert res["normalized_entropy"] == pytest.approx(1.0, rel=1e-3)
        assert res["threshold"] == pytest.approx(expected_threshold, rel=1e-3)
        assert res["status"] == "ENTROPY_SUFFICIENT"
        assert res["warning"] is None

    def test_skewed_distribution_fails_greedy_exploitation(self, default_governor: InformationGovernor):
        """Skewed distribution concentrating 90% on architecture fails with greedy exploitation warning."""
        lane_counts = {
            "architecture": 18,
            "objectives": 1,
            "optimization": 1,
            "features": 0,
        }
        res = default_governor.check_exploration_entropy(lane_counts)
        # Total = 20. p = [0.90, 0.05, 0.05, 0.00]
        # H = -(0.9 * ln(0.9) + 2 * 0.05 * ln(0.05)) ~ 0.3944
        expected_threshold = 0.75 * np.log(4)  # ~1.03972

        assert res["passed"] is False
        assert res["sufficient"] is False
        assert res["is_high_entropy"] is False
        assert res["greedy_exploitation_flag"] is True
        assert res["entropy"] < expected_threshold
        assert res["status"] == "GREEDY_EXPLOITATION_BLOCKED"
        assert res["warning"] is not None
        assert "greedy" in res["warning"].lower() or "exploitation" in res["warning"].lower()

    def test_single_lane_concentration_zero_entropy(self, default_governor: InformationGovernor):
        """Single-lane 100% allocation has entropy 0.0 and strictly fails."""
        lane_counts = {
            "architecture": 10,
            "objectives": 0,
            "optimization": 0,
            "features": 0,
        }
        res = default_governor.check_exploration_entropy(lane_counts)
        assert res["entropy"] == pytest.approx(0.0, abs=1e-5)
        assert res["passed"] is False
        assert res["sufficient"] is False
        assert res["status"] == "GREEDY_EXPLOITATION_BLOCKED"

    def test_empty_lane_counts_handled_gracefully(self, default_governor: InformationGovernor):
        """Empty or zero proposal counts does not cause ZeroDivisionError or crash."""
        res_empty = default_governor.check_exploration_entropy({})
        assert res_empty["passed"] is False
        assert res_empty["sufficient"] is False
        assert res_empty["entropy"] == pytest.approx(0.0, abs=1e-5)
        assert res_empty["status"] == "GREEDY_EXPLOITATION_BLOCKED"

        res_zeros = default_governor.check_exploration_entropy({
            "architecture": 0,
            "objectives": 0,
            "optimization": 0,
            "features": 0,
        })
        assert res_zeros["passed"] is False
        assert res_zeros["sufficient"] is False
        assert res_zeros["entropy"] == pytest.approx(0.0, abs=1e-5)


# =====================================================================
# 4. Cognitive Subagent Specifications Tests (All 8 Roles)
# =====================================================================

class TestCognitiveSubagentSpecifications:
    """Validates specification generation and CLI export for all 8 cognitive roles."""

    EXPECTED_8_ROLES = [
        "frame_information_governor",
        "frame_literature_proximity_analyzer",
        "frame_meta_reviewer",
        "frame_executioner",
        "frame_lane_architecture",
        "frame_lane_objectives",
        "frame_lane_optimization",
        "frame_lane_features",
    ]

    def test_build_agy_subagent_specs_all_8_roles_when_requested(self):
        """build_agy_subagent_specs generates valid specs for all 8 cognitive roles."""
        specs = build_agy_subagent_specs(
            goal="Design novel attention mechanisms",
            include_cognitive_roles=True,
        )
        assert len(specs) == 8
        type_names = [s.TypeName for s in specs]
        for expected in self.EXPECTED_8_ROLES:
            assert expected in type_names

        # Validate spec properties
        for spec in specs:
            assert isinstance(spec, SubagentSpec)
            assert spec.Model == "inherit"
            assert spec.Workspace == "inherit"
            assert len(spec.Role) > 0
            assert len(spec.Prompt) > 50

    def test_cli_swarm_export_agy_specs_contains_all_8_roles(self, capsys):
        """frame swarm --export-agy-specs outputs valid JSON containing all 8 roles."""
        parser = build_parser()
        args = parser.parse_args(["swarm", "--export-agy-specs"])
        exit_code = cmd_swarm(args)
        assert exit_code == 0

        captured = capsys.readouterr().out
        specs_json = json.loads(captured)
        assert isinstance(specs_json, list)
        assert len(specs_json) == 8

        exported_type_names = [s["TypeName"] for s in specs_json]
        for expected in self.EXPECTED_8_ROLES:
            assert expected in exported_type_names

    def test_cognitive_role_prompts_contain_responsibilities(self):
        """Verifies each cognitive role prompt contains key responsibility keywords."""
        specs = build_agy_subagent_specs(
            goal="Optimize attention layer efficiency",
            include_cognitive_roles=True,
        )
        specs_by_type = {s.TypeName: s for s in specs}

        # 1. Governor
        gov_prompt = specs_by_type["frame_information_governor"].Prompt
        assert "evidence" in gov_prompt.lower() or "grade" in gov_prompt.lower()
        assert "entropy" in gov_prompt.lower() or "audit" in gov_prompt.lower()

        # 2. Proximity Analyzer
        prox_prompt = specs_by_type["frame_literature_proximity_analyzer"].Prompt
        assert "proximity" in prox_prompt.lower() or "prior art" in prox_prompt.lower() or "distance" in prox_prompt.lower()

        # 3. Meta-Reviewer
        meta_prompt = specs_by_type["frame_meta_reviewer"].Prompt
        assert "review" in meta_prompt.lower() or "synerg" in meta_prompt.lower() or "rigor" in meta_prompt.lower()

        # 4. Executioner
        exec_prompt = specs_by_type["frame_executioner"].Prompt
        assert "stagnat" in exec_prompt.lower() or "prun" in exec_prompt.lower() or "restart" in exec_prompt.lower()


# =====================================================================
# 5. Coordinator Integration & Event Logging Tests
# =====================================================================

class TestGovernorCoordinatorIntegration:
    """Tests integration between InformationGovernor and AgySwarmCoordinator."""

    def test_coordinator_has_governor_instance(self):
        """AgySwarmCoordinator initializes with an InformationGovernor."""
        coordinator = AgySwarmCoordinator(goal="Benchmark Transformer Layers")
        assert hasattr(coordinator, "governor")
        assert isinstance(coordinator.governor, InformationGovernor)

    def test_coordinator_audit_ingested_sources_records_logs(
        self,
        cochrane_high_rigor_source: EvidenceSource,
        public_forum_source: EvidenceSource,
    ):
        """Coordinator audits ingested sources and records event logs."""
        coordinator = AgySwarmCoordinator(goal="Benchmark Transformer Layers")
        vetted = coordinator.governor.audit_and_filter([
            cochrane_high_rigor_source,
            public_forum_source,
        ])
        assert len(vetted) == 1
        assert vetted[0].source_id == "src_cochrane_001"

        logs = coordinator.governor.audit_history
        assert len(logs) == 2
        assert logs[0].accepted is True
        assert logs[1].accepted is False


# =====================================================================
# 6. Schema, Normalization & Extended Coverage Tests
# =====================================================================

class TestSourceTierAndSchemaValidation:
    """Tests schema validation, enum coercion, and audit summary features."""

    def test_source_tier_enum_values_and_normalization(self):
        """Verifies SourceTier enum members and robust string normalization."""
        assert SourceTier.COCHRANE_OR_JOURNAL.value == "cochrane_or_journal"
        assert SourceTier.PREPRINT.value == "preprint"
        assert SourceTier.PUBLIC_FORUM.value == "public_forum"
        assert SourceTier.UNVERIFIED.value == "unverified"

        assert SourceTier.from_str("cochrane") == SourceTier.COCHRANE_OR_JOURNAL
        assert SourceTier.from_str("cochrane_or_peer_reviewed_journal") == SourceTier.COCHRANE_OR_JOURNAL
        assert SourceTier.from_str("arxiv") == SourceTier.PREPRINT
        assert SourceTier.from_str("kaggle") == SourceTier.PUBLIC_FORUM
        assert SourceTier.from_str("blog") == SourceTier.UNVERIFIED
        assert SourceTier.from_str("anonymous") == SourceTier.UNVERIFIED
        assert SourceTier.from_str("unknown_xyz") == SourceTier.UNVERIFIED

    def test_evidence_source_creation_and_defaults(self):
        """Verifies EvidenceSource field defaults and string coercion."""
        src = EvidenceSource(
            title="Attention Is All You Need",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
        )
        assert src.source_id.startswith("src_")
        assert src.peer_reviewed is False
        assert src.relevance_score == 0.5
        assert src.methodological_rigor == 0.5

        # Coercion from string
        src2 = EvidenceSource(
            title="BERT Pre-training",
            source_type="preprint",
            peer_reviewed=True,
            relevance_score=0.9,
            methodological_rigor=0.85,
        )
        assert src2.source_type == SourceTier.PREPRINT

    def test_governor_entropy_strict_mode(self, default_governor: InformationGovernor):
        """Verifies strict mode raises AssertionError when greedy exploitation is detected."""
        uniform_counts = {"a": 5, "b": 5, "c": 5, "d": 5}
        # Should not raise
        res = default_governor.check_exploration_entropy(uniform_counts, strict=True)
        assert res["passed"] is True

        skewed_counts = {"a": 20, "b": 0, "c": 0, "d": 0}
        with pytest.raises(AssertionError, match="Greedy premature exploitation detected"):
            default_governor.check_exploration_entropy(skewed_counts, strict=True)

    def test_governor_audit_summary_and_clear(self, default_governor: InformationGovernor, cochrane_high_rigor_source: EvidenceSource):
        """Verifies get_audit_summary and clear_audit_history."""
        default_governor.grade_source(cochrane_high_rigor_source)
        summary = default_governor.get_audit_summary()
        assert summary["total_audited"] == 1
        assert summary["accepted"] == 1
        assert summary["rejected"] == 0
        assert summary["acceptance_rate"] == 1.0

        default_governor.clear_audit_history()
        assert len(default_governor.audit_history) == 0
        assert len(default_governor.event_logs) == 0
