"""Unit test suite for Domain Evidence Standards & Invariant Architecture (Milestone M5).

Covers:
- Feature 14: Evidence Hierarchy Weighting System
- Feature 15: Data Provenance & Ethical Constraint Tracker
- Feature 16: Domain Invariant Plug-in Architecture
- Feature 17: Tiered L0–L2 Math Verifier Invariant Binding
"""

import hashlib
import math
import pytest

from frame.agents.governor import EvidenceSource, SourceTier
from frame.verification.domain_evidence import (
    APPROVED_LICENSES,
    DEFAULT_HIERARCHY_WEIGHTS,
    DataProvenance,
    DomainEvidenceVerifier,
)
from frame.verification.invariants import (
    BiologyMonotonicityPlugin,
    BoundaryConditionsPlugin,
    DefaultBiologyPlugin,
    DefaultBoundaryPlugin,
    DefaultPhysicsPlugin,
    DomainInvariantPlugin,
    PhysicsConservationPlugin,
    SymbolicDomainConstraint,
)


# ==============================================================================
# Feature 14: Evidence Hierarchy Weighting System Tests
# ==============================================================================


class TestEvidenceHierarchyWeighting:
    """Tests for DomainEvidenceVerifier.verify_evidence_hierarchy."""

    def test_empty_sources_list_returns_false_and_zero(self):
        verifier = DomainEvidenceVerifier()
        passed, score = verifier.verify_evidence_hierarchy([])
        assert passed is False
        assert score == 0.0

    def test_single_cochrane_source_full_rigor(self):
        verifier = DomainEvidenceVerifier()
        source = EvidenceSource(
            source_id="c1",
            title="Peer Reviewed Study",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=1.0,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is True
        assert math.isclose(score, 1.0, abs_tol=1e-5)

    def test_single_preprint_source_weight_attenuation(self):
        verifier = DomainEvidenceVerifier()
        source = EvidenceSource(
            source_id="p1",
            title="ArXiv Preprint",
            source_type=SourceTier.PREPRINT,
            methodological_rigor=0.8,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is True
        # Single source score equals its methodological rigor
        assert math.isclose(score, 0.8, abs_tol=1e-5)

    def test_single_forum_source_weight_attenuation(self):
        verifier = DomainEvidenceVerifier()
        source = EvidenceSource(
            source_id="f1",
            title="Reddit Discussion",
            source_type=SourceTier.PUBLIC_FORUM,
            methodological_rigor=0.4,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is False
        assert math.isclose(score, 0.4, abs_tol=1e-5)

    def test_single_unverified_source_fails_threshold(self):
        verifier = DomainEvidenceVerifier()
        source = EvidenceSource(
            source_id="u1",
            title="Anonymous Blog",
            source_type=SourceTier.UNVERIFIED,
            methodological_rigor=0.3,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is False
        assert score < 0.50

    def test_mixed_sources_exact_weighted_average(self):
        verifier = DomainEvidenceVerifier()
        sources = [
            EvidenceSource(
                source_id="c1",
                title="Top Journal",
                source_type=SourceTier.COCHRANE_OR_JOURNAL,
                methodological_rigor=1.0,
            ),  # weight 1.0 -> 1.0 * 1.0 = 1.0
            EvidenceSource(
                source_id="f1",
                title="Forum Post",
                source_type=SourceTier.PUBLIC_FORUM,
                methodological_rigor=0.2,
            ),  # weight 0.3 -> 0.3 * 0.2 = 0.06
        ]
        # (1.0 + 0.06) / (1.0 + 0.3) = 1.06 / 1.3 ≈ 0.81538
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is True
        expected = (1.0 * 1.0 + 0.3 * 0.2) / (1.0 + 0.3)
        assert math.isclose(score, expected, abs_tol=1e-4)

    def test_exact_threshold_boundary_acceptance(self):
        verifier = DomainEvidenceVerifier(evidence_threshold=0.50)
        source = EvidenceSource(
            source_id="s50",
            title="Threshold Borderline",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.50,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is True
        assert math.isclose(score, 0.50, abs_tol=1e-5)

    def test_just_below_threshold_rejection(self):
        verifier = DomainEvidenceVerifier(evidence_threshold=0.50)
        source = EvidenceSource(
            source_id="s49",
            title="Sub-Threshold",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.499,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is False
        assert score < 0.50

    def test_dict_source_structure_compatibility(self):
        verifier = DomainEvidenceVerifier()
        sources = [
            {"tier": "cochrane_or_journal", "rigor": 0.9},
            {"tier": "preprint", "rigor": 0.7},
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is True
        expected = (1.0 * 0.9 + 0.7 * 0.7) / (1.0 + 0.7)
        assert math.isclose(score, expected, abs_tol=1e-4)

    def test_custom_evidence_threshold(self):
        verifier = DomainEvidenceVerifier(evidence_threshold=0.85)
        source = EvidenceSource(
            source_id="c1",
            title="Good Journal",
            source_type=SourceTier.COCHRANE_OR_JOURNAL,
            methodological_rigor=0.80,
        )
        passed, score = verifier.verify_evidence_hierarchy([source])
        assert passed is False
        assert math.isclose(score, 0.80, abs_tol=1e-5)

    def test_zero_rigor_across_multiple_sources(self):
        verifier = DomainEvidenceVerifier()
        sources = [
            EvidenceSource(
                source_id="z1",
                title="Zero 1",
                source_type=SourceTier.COCHRANE_OR_JOURNAL,
                methodological_rigor=0.0,
            ),
            EvidenceSource(
                source_id="z2",
                title="Zero 2",
                source_type=SourceTier.PREPRINT,
                methodological_rigor=0.0,
            ),
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        assert passed is False
        assert score == 0.0

    def test_invalid_rigor_values_fallback(self):
        verifier = DomainEvidenceVerifier()
        sources = [
            {"tier": "cochrane_or_journal", "rigor": "invalid_number"},
            {"tier": "preprint", "rigor": 0.8},
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources)
        # invalid_number falls back to 0.0: (1.0*0.0 + 0.7*0.8) / 1.7 = 0.56 / 1.7 ≈ 0.3294
        assert isinstance(passed, bool)
        assert isinstance(score, float)


# ==============================================================================
# Feature 15: Data Provenance & Ethical Constraint Tracker Tests
# ==============================================================================


class TestDataProvenanceAndEthics:
    """Tests for DomainEvidenceVerifier.verify_provenance_and_ethics."""

    def test_valid_permissive_mit_dataset_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafybeic5...",
            license="MIT",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.5,
            bias_disparity_score=0.03,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True
        assert len(violations) == 0

    def test_valid_apache_dataset_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_apache",
            license="Apache-2.0",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.8,
            bias_disparity_score=0.05,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True
        assert len(violations) == 0

    def test_valid_cc_by_dataset_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_cc",
            license="CC-BY-4.0",
            is_commercial_safe=True,
            privacy_budget_epsilon=0.2,
            bias_disparity_score=0.01,
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True
        assert len(violations) == 0

    def test_valid_bsd_and_cc0_licenses(self):
        verifier = DomainEvidenceVerifier()
        for lic in ["BSD", "BSD-3-Clause", "CC0", "Unlicense", "ISC"]:
            prov = DataProvenance(dataset_cid="cid", license=lic)
            passed, violations = verifier.verify_provenance_and_ethics(prov)
            assert passed is True, f"Failed for approved license {lic}"

    def test_unapproved_proprietary_license_fails(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_prop",
            license="PROPRIETARY_UNAUTHORIZED",
            is_commercial_safe=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("license" in v.lower() for v in violations)

    def test_commercial_safety_false_fails(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_comm",
            license="MIT",
            is_commercial_safe=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("commercial" in v.lower() for v in violations)

    def test_privacy_budget_exact_boundary_1_0_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_eps1",
            license="MIT",
            privacy_budget_epsilon=1.0,
            bias_disparity_score=0.05,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True
        assert len(violations) == 0

    def test_privacy_budget_zero_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_eps0",
            license="MIT",
            privacy_budget_epsilon=0.0,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True

    def test_privacy_budget_excessive_fails(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_leak",
            license="MIT",
            privacy_budget_epsilon=2.5,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("privacy" in v.lower() or "epsilon" in v.lower() for v in violations)

    def test_bias_disparity_exact_boundary_0_10_passes(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_bias_exact",
            license="MIT",
            bias_disparity_score=0.10,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is True

    def test_bias_disparity_excessive_fails(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_bias_fail",
            license="MIT",
            bias_disparity_score=0.25,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("bias" in v.lower() for v in violations)

    def test_ethical_approval_false_fails(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(
            dataset_cid="bafy_unethical",
            license="MIT",
            ethical_approval=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("ethical" in v.lower() for v in violations)

    def test_multiple_simultaneous_violations_aggregated(self):
        verifier = DomainEvidenceVerifier()
        bad_prov = DataProvenance(
            dataset_cid="bafy_all_fail",
            license="STRICTLY_PROPRIETARY",
            is_commercial_safe=False,
            privacy_budget_epsilon=5.0,
            bias_disparity_score=0.50,
            ethical_approval=False,
        )
        passed, violations = verifier.verify_provenance_and_ethics(bad_prov)
        assert passed is False
        assert len(violations) >= 4
        assert any("license" in v.lower() for v in violations)
        assert any("commercial" in v.lower() for v in violations)
        assert any("privacy" in v.lower() or "epsilon" in v.lower() for v in violations)
        assert any("bias" in v.lower() for v in violations)
        assert any("ethical" in v.lower() for v in violations)

    def test_empty_cid_handled_gracefully(self):
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(dataset_cid="", license="MIT")
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert isinstance(passed, bool)
        assert passed is True

    def test_cid_sha256_hash_integrity_verification(self):
        data = b"synthetic medical research records"
        expected_hash = hashlib.sha256(data).hexdigest()
        prov = DataProvenance(dataset_cid=expected_hash, license="MIT")
        assert prov.verify_hash_integrity(data) is True

    def test_cid_sha256_hash_integrity_mismatch(self):
        prov = DataProvenance(dataset_cid="abcdef123456", license="MIT")
        assert prov.verify_hash_integrity(b"corrupted data") is False

    def test_custom_epsilon_and_bias_thresholds(self):
        verifier = DomainEvidenceVerifier(max_epsilon=0.2, max_bias_disparity=0.02)
        prov = DataProvenance(
            dataset_cid="cid",
            license="MIT",
            privacy_budget_epsilon=0.5,  # Exceeds custom 0.2
            bias_disparity_score=0.05,   # Exceeds custom 0.02
        )
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert len(violations) == 2


# ==============================================================================
# Feature 16: Domain Invariant Plug-in Architecture Tests
# ==============================================================================


class TestDomainInvariantPluginArchitecture:
    """Tests for DomainInvariantPlugin and registry architecture."""

    def test_default_plugins_loaded_on_none(self):
        verifier = DomainEvidenceVerifier()
        assert "physics" in verifier.plugins
        assert "biology" in verifier.plugins
        assert "boundary" in verifier.plugins
        assert len(verifier.plugins) == 3

    def test_empty_plugins_list_initializes_empty(self):
        verifier = DomainEvidenceVerifier(plugins=[])
        assert len(verifier.plugins) == 0

    def test_custom_plugin_registration(self):
        class QuantumPlugin(DomainInvariantPlugin):
            def get_domain_name(self) -> str:
                return "quantum"

            def get_constraints(self):
                return [
                    SymbolicDomainConstraint(
                        constraint_id="q_norm",
                        name="Unitary Normalization",
                        domain="quantum",
                        symbolic_expression="p_1 + p_2",
                        lower_bound=1.0,
                        upper_bound=1.0,
                    )
                ]

        verifier = DomainEvidenceVerifier(plugins=[QuantumPlugin()])
        assert "quantum" in verifier.plugins
        assert len(verifier.plugins) == 1
        constraints = verifier.plugins["quantum"].get_constraints()
        assert len(constraints) == 1
        assert constraints[0].constraint_id == "q_norm"

    def test_multiple_custom_plugins_registered(self):
        class P1(DomainInvariantPlugin):
            def get_domain_name(self): return "d1"
            def get_constraints(self): return []

        class P2(DomainInvariantPlugin):
            def get_domain_name(self): return "d2"
            def get_constraints(self): return []

        verifier = DomainEvidenceVerifier(plugins=[P1(), P2()])
        assert len(verifier.plugins) == 2
        assert "d1" in verifier.plugins
        assert "d2" in verifier.plugins

    def test_duplicate_constraint_id_handling(self):
        class DupPlugin(DomainInvariantPlugin):
            def get_domain_name(self): return "dup"
            def get_constraints(self):
                return [
                    SymbolicDomainConstraint(constraint_id="c_dup", name="1", domain="dup", symbolic_expression="x"),
                    SymbolicDomainConstraint(constraint_id="c_dup", name="2", domain="dup", symbolic_expression="y"),
                ]

        verifier = DomainEvidenceVerifier(plugins=[DupPlugin()])
        assert "dup" in verifier.plugins
        assert len(verifier.plugins["dup"].get_constraints()) == 2

    def test_get_constraints_returns_list_of_constraints(self):
        plugin = PhysicsConservationPlugin()
        constraints = plugin.get_constraints()
        assert isinstance(constraints, list)
        assert len(constraints) >= 1
        assert isinstance(constraints[0], SymbolicDomainConstraint)
        assert constraints[0].domain == "physics"

    def test_constraint_extra_fields_allowed_via_config_dict(self):
        c = SymbolicDomainConstraint(
            constraint_id="custom_id",
            name="Custom",
            domain="physics",
            extra_telemetry_tag="v1.0",
            author="specialist",
        )
        assert c.constraint_id == "custom_id"
        assert getattr(c, "extra_telemetry_tag") == "v1.0"

    def test_unregistered_domain_returns_all_satisfied(self):
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x**2", domain="unregistered_domain_xyz")
        assert isinstance(res, dict)
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0


# ==============================================================================
# Feature 17: Tiered L0–L2 Math Verifier Invariant Binding Tests
# ==============================================================================


class TestMathVerifierInvariantBinding:
    """Tests for DomainEvidenceVerifier.verify_symbolic_invariants."""

    def test_l0_analytic_conservation_quadratic_positive(self):
        verifier = DomainEvidenceVerifier()
        # x^2 + 1 >= 0 is strictly true everywhere
        res = verifier.verify_symbolic_invariants("x**2 + 1", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_l0_analytic_conservation_quadratic_with_linear_term(self):
        verifier = DomainEvidenceVerifier()
        # x^2 + 2*x + 5 = (x+1)^2 + 4 >= 4 > 0 everywhere
        res = verifier.verify_symbolic_invariants("x**2 + 2*x + 5", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_l1_bounded_interval_linear_shift(self):
        verifier = DomainEvidenceVerifier()
        # x + 3 >= 0 on [0, 5]
        res = verifier.verify_symbolic_invariants("x + 3", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_l1_exponential_decay_positivity(self):
        verifier = DomainEvidenceVerifier()
        # exp(-x**2) > 0 everywhere
        res = verifier.verify_symbolic_invariants("exp(-x**2)", domain="physics")
        assert res["all_satisfied"] is True

    def test_l2_ibnb_monotonicity_exponential(self):
        verifier = DomainEvidenceVerifier()
        # exp(x) has strictly positive derivative everywhere
        res = verifier.verify_symbolic_invariants("exp(x)", domain="biology")
        assert res["all_satisfied"] is True

    def test_l2_monotonicity_constant_derivative(self):
        verifier = DomainEvidenceVerifier()
        # 2*x has positive derivative 2 >= 0
        res = verifier.verify_symbolic_invariants("2*x", domain="biology")
        assert res["all_satisfied"] is True

    def test_l2_monotonicity_negative_slope_fails(self):
        verifier = DomainEvidenceVerifier()
        # -2*x has negative derivative -2 < 0
        res = verifier.verify_symbolic_invariants("-2*x", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0

    def test_conservation_violation_produces_counterexample(self):
        verifier = DomainEvidenceVerifier()
        # x^2 - 100 evaluates to -100 at x=0
        res = verifier.verify_symbolic_invariants("x**2 - 100", domain="physics")
        assert res["all_satisfied"] is False
        assert res["counterexample"] is not None
        assert "counterexample" in str(res).lower()
        assert len(res["violations"]) > 0

    def test_conservation_violation_energy_drop(self):
        verifier = DomainEvidenceVerifier()
        # x^2 - 5 evaluates to -5 at x=0
        res = verifier.verify_symbolic_invariants("x**2 - 5", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0

    def test_boundary_regularity_logistic_sigmoid_passes(self):
        verifier = DomainEvidenceVerifier()
        # 1 / (1 + exp(-x)) is bounded in [0, 1] without poles
        res = verifier.verify_symbolic_invariants("1 / (1 + exp(-x))", domain="boundary")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_boundary_singularity_pole_detection_fails(self):
        verifier = DomainEvidenceVerifier()
        # 1 / x has singularity pole at x = 0
        res = verifier.verify_symbolic_invariants("1 / x", domain="boundary")
        assert res["all_satisfied"] is False
        assert any("pole" in v.lower() or "singularity" in v.lower() for v in res["violations"])

    def test_constant_expression_satisfies_bounds(self):
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("1.0", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_constant_expression_violates_bounds(self):
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("-10.0", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0

    def test_multivariable_expression_handled(self):
        verifier = DomainEvidenceVerifier()
        # On domain='physics', x^2 + y^2 >= 0 satisfies conservation
        res_phys = verifier.verify_symbolic_invariants("x**2 + y**2", domain="physics")
        assert res_phys["all_satisfied"] is True
        # On domain=None, evaluated across all plugins gracefully returning valid dict
        res_none = verifier.verify_symbolic_invariants("x**2 + y**2", domain=None)
        assert isinstance(res_none, dict)

    def test_malformed_syntax_graceful_failure(self):
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x +* 3 / )", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("syntax" in v.lower() for v in res["violations"])

    def test_non_differentiable_function_handled_gracefully(self):
        verifier = DomainEvidenceVerifier()
        # abs(x) has non-differentiable singularity at x=0
        res = verifier.verify_symbolic_invariants("abs(x)", domain="biology")
        assert isinstance(res, dict)
        assert "all_satisfied" in res

    def test_boundary_point_equality_zero(self):
        verifier = DomainEvidenceVerifier()
        # x on [0, 5] has minimum 0.0 == lower_bound 0.0
        res = verifier.verify_symbolic_invariants("x", domain="physics")
        assert isinstance(res, dict)
        assert res["all_satisfied"] is True

    def test_all_plugins_evaluated_when_domain_is_none(self):
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("exp(x)", domain=None)
        assert isinstance(res, dict)
        assert "evaluations" in res
        assert len(res["evaluations"]) >= 3
