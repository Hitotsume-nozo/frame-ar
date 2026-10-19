"""Empirical Adversarial Verification Test Suite for Milestone M5.

Agent: m5_challenger_2 (Empirical Challenger)
Milestone: M5 — Domain Evidence Standards & Invariant Plug-in Architecture
Scope: Features 16 & 17 (Domain Invariants & L0–L2 Math Verifier Binding)

Test Categories:
1. Custom plugin registration and domain filtering (Feature 16)
2. Physical conservation invariants: positive quadratics pass, negative quadratics fail with counterexamples (Feature 17)
3. Biological monotonicity invariants: exponential, sigmoid, Hill functions pass; non-monotonic / decreasing functions fail (Feature 17)
4. Boundary condition singularity / pole detection: poles fail, sigmoids pass, range overflows fail (Feature 17)
5. Ill-formed mathematical expressions: syntax traps, malformed ASTs caught gracefully (Feature 17)
6. Challenger Stress Tests & Bug Reproducers:
   - Bug 1: Domain-ignorant pole detection flags poles strictly outside the evaluation domain (e.g., x / (1 + x), log(x + 1))
   - Bug 2: Origin-only multi-variable evaluation falsely certifies negative-energy expressions (e.g., 1 - x - y)
"""

import math
from typing import List, Optional
import numpy as np
import pytest
import sympy as sp

from frame.verification.domain_evidence import DomainEvidenceVerifier
from frame.verification.invariants import (
    BiologyMonotonicityPlugin,
    BoundaryConditionsPlugin,
    DomainInvariantPlugin,
    PhysicsConservationPlugin,
    SymbolicDomainConstraint,
)


# ==============================================================================
# Category 1: Custom Plugin Registration and Domain Filtering
# ==============================================================================


class TestPluginRegistrationAndDomainFilteringChallenger:
    """Empirical verification of custom plugin lifecycle and domain isolation."""

    def test_custom_plugin_registration_and_domain_query(self):
        """Custom domain plugin registers successfully and can be queried specifically."""
        class QuantumInvariantsPlugin(DomainInvariantPlugin):
            def get_domain_name(self) -> str:
                return "quantum"

            def get_constraints(self) -> List[SymbolicDomainConstraint]:
                return [
                    SymbolicDomainConstraint(
                        constraint_id="quantum_born_rule_positivity",
                        name="Born Rule Probability Positivity",
                        domain="quantum",
                        symbolic_expression="psi_sq",
                        lower_bound=0.0,
                        upper_bound=1.0,
                        interval=(0.0, 1.0),
                        constraint_type="positivity",
                        description="Probabilities in quantum mechanics must be non-negative and <= 1.",
                    ),
                    SymbolicDomainConstraint(
                        constraint_id="quantum_uncertainty_lower_bound",
                        name="Heisenberg Product Lower Bound",
                        domain="quantum",
                        symbolic_expression="delta_x * delta_p",
                        lower_bound=0.5,
                        interval=(0.5, 10.0),
                        constraint_type="positivity",
                        description="Product of uncertainties must satisfy dx*dp >= hbar/2.",
                    ),
                ]

        verifier = DomainEvidenceVerifier(plugins=[QuantumInvariantsPlugin()])
        assert "quantum" in verifier.plugins
        assert len(verifier.plugins) == 1

        # Query quantum domain with valid expression
        res_pass = verifier.verify_symbolic_invariants("0.75", domain="quantum")
        assert res_pass["all_satisfied"] is True
        assert len(res_pass["violations"]) == 0
        assert len(res_pass["evaluations"]) == 2

        # Query quantum domain with violating constant
        res_fail = verifier.verify_symbolic_invariants("-0.1", domain="quantum")
        assert res_fail["all_satisfied"] is False
        assert len(res_fail["violations"]) > 0

    def test_domain_filtering_isolates_unqueried_domains(self):
        """Filtering by domain='physics' evaluates only physics constraints and ignores biology."""
        verifier = DomainEvidenceVerifier()
        res_phys = verifier.verify_symbolic_invariants("x**2 + 10", domain="physics")
        assert res_phys["all_satisfied"] is True
        assert len(res_phys["evaluations"]) == 1
        assert res_phys["evaluations"][0]["domain"] == "physics"

        res_bio = verifier.verify_symbolic_invariants("exp(x)", domain="biology")
        assert res_bio["all_satisfied"] is True
        assert len(res_bio["evaluations"]) == 1
        assert res_bio["evaluations"][0]["domain"] == "biology"

    def test_unregistered_domain_handled_gracefully(self):
        """Querying an unknown domain returns all_satisfied=True without crashing."""
        verifier = DomainEvidenceVerifier()
        res = verifier.verify_symbolic_invariants("x**3", domain="nonexistent_domain_xyz")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0
        assert res["evaluations"] == []
        assert "not registered" in res["details"].get("msg", "").lower()

    def test_empty_plugins_verifier_initialization(self):
        """Verifier initialized with empty list plugins=[] handles any query gracefully."""
        verifier = DomainEvidenceVerifier(plugins=[])
        assert len(verifier.plugins) == 0

        res_none = verifier.verify_symbolic_invariants("x**2 + 1", domain=None)
        assert res_none["all_satisfied"] is True
        assert len(res_none["violations"]) == 0
        assert len(res_none["evaluations"]) == 0

        res_domain = verifier.verify_symbolic_invariants("x**2 + 1", domain="physics")
        assert res_domain["all_satisfied"] is True
        assert len(res_domain["violations"]) == 0

    def test_multiple_plugins_with_overlapping_constraint_ids(self):
        """Multiple plugins registering duplicate constraint IDs coexistence without crash."""
        class PluginA(DomainInvariantPlugin):
            def get_domain_name(self) -> str: return "dom_a"
            def get_constraints(self) -> List[SymbolicDomainConstraint]:
                return [SymbolicDomainConstraint(constraint_id="shared_id", name="A", domain="dom_a")]

        class PluginB(DomainInvariantPlugin):
            def get_domain_name(self) -> str: return "dom_b"
            def get_constraints(self) -> List[SymbolicDomainConstraint]:
                return [SymbolicDomainConstraint(constraint_id="shared_id", name="B", domain="dom_b")]

        verifier = DomainEvidenceVerifier(plugins=[PluginA(), PluginB()])
        assert len(verifier.plugins) == 2
        res = verifier.verify_symbolic_invariants("x + 1", domain=None)
        assert isinstance(res, dict)
        assert "all_satisfied" in res


# ==============================================================================
# Category 2: Physical Conservation Invariants (Positivity & Energy Bounds)
# ==============================================================================


class TestPhysicalConservationInvariantsChallenger:
    """Empirical verification of physical conservation and energy positivity laws."""

    @pytest.fixture
    def verifier(self):
        return DomainEvidenceVerifier()

    def test_strictly_positive_quadratic_global(self, verifier):
        """Global positive quadratic (x^2 + 10) satisfies conservation at L0_ANALYTIC."""
        res = verifier.verify_symbolic_invariants("x**2 + 10", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0
        assert res["evaluations"][0]["satisfied"] is True
        assert res["evaluations"][0]["tier_used"] == "L0_ANALYTIC"

    def test_positive_quadratic_nonzero_linear_shift(self, verifier):
        """Positive quadratic with linear shift (x^2 - 2*x + 5 = (x-1)^2 + 4 >= 4) passes."""
        res = verifier.verify_symbolic_invariants("x**2 - 2*x + 5", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0
        assert res["evaluations"][0]["tier_used"] == "L0_ANALYTIC"

    def test_positive_semidefinite_quadratic_at_zero(self, verifier):
        """x^2 on [0, 5] with minimum 0.0 == lower_bound 0.0 satisfies conservation."""
        res = verifier.verify_symbolic_invariants("x**2", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_positive_semidefinite_shifted_vertex_at_zero(self, verifier):
        """(x - 2)^2 = x^2 - 4*x + 4 has minimum 0.0 at x=2 in [0, 5], passes."""
        res = verifier.verify_symbolic_invariants("x**2 - 4*x + 4", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_negative_energy_quadratic_fails_with_concrete_counterexample(self, verifier):
        """x^2 - 100 violates energy conservation; must fail and return counterexample."""
        res = verifier.verify_symbolic_invariants("x**2 - 100", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        assert isinstance(res["counterexample"], list)
        assert len(res["counterexample"]) >= 1
        ce_x = res["counterexample"][0]
        val_at_ce = ce_x**2 - 100
        assert val_at_ce < 0.0
        assert any("violated" in v.lower() for v in res["violations"])

    def test_negative_energy_quadratic_subtle_dip(self, verifier):
        """x^2 - 3*x + 1 on [0, 5] dips below 0 at x=1.5 (value = -1.25), fails with counterexample."""
        res = verifier.verify_symbolic_invariants("x**2 - 3*x + 1", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        ce_x = res["counterexample"][0]
        val = ce_x**2 - 3 * ce_x + 1
        assert val < 0.0

    def test_strictly_negative_everywhere_quadratic(self, verifier):
        """-(x^2 + 5) is strictly negative everywhere on [0, 5], fails with counterexample."""
        res = verifier.verify_symbolic_invariants("-(x**2 + 5)", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None

    def test_multivariable_energy_conservation_positive(self, verifier):
        """Multi-variable kinetic + potential energy x^2 + y^2 satisfies conservation."""
        res = verifier.verify_symbolic_invariants("x**2 + y**2", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_multivariable_energy_conservation_negative_at_origin(self, verifier):
        """Multi-variable x^2 + y^2 - 10 evaluates to -10 at origin, must fail with counterexample."""
        res = verifier.verify_symbolic_invariants("x**2 + y**2 - 10", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        assert res["counterexample"] == [0.0, 0.0]


# ==============================================================================
# Category 3: Biological Monotonicity Invariants (Dose-Response)
# ==============================================================================


class TestBiologicalMonotonicityInvariantsChallenger:
    """Empirical verification of biological dose-response monotonicity invariants."""

    @pytest.fixture
    def verifier(self):
        return DomainEvidenceVerifier()

    def test_monotonic_exponential_growth_passes(self, verifier):
        """exp(x) and exp(2*x) have strictly positive derivatives, satisfying monotonicity."""
        res1 = verifier.verify_symbolic_invariants("exp(x)", domain="biology")
        assert res1["all_satisfied"] is True
        assert len(res1["violations"]) == 0

        res2 = verifier.verify_symbolic_invariants("exp(2*x)", domain="biology")
        assert res2["all_satisfied"] is True
        assert len(res2["violations"]) == 0

    def test_monotonic_logistic_sigmoid_passes(self, verifier):
        """1 / (1 + exp(-x)) has strictly positive derivative everywhere, passes biology monotonicity."""
        res = verifier.verify_symbolic_invariants("1 / (1 + exp(-x))", domain="biology")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_monotonic_linear_positive_slope_passes(self, verifier):
        """3*x + 5 has constant positive slope 3.0 >= 0, passes monotonicity."""
        res = verifier.verify_symbolic_invariants("3*x + 5", domain="biology")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_monotonic_odd_polynomial_passes(self, verifier):
        """x**3 + 2*x on [0, 5] has positive derivative 3*x^2 + 2 > 0, passes."""
        res = verifier.verify_symbolic_invariants("x**3 + 2*x", domain="biology")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_decreasing_linear_negative_slope_fails(self, verifier):
        """-5*x has negative derivative -5 < 0, must fail monotonicity with counterexample."""
        res = verifier.verify_symbolic_invariants("-5*x", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        assert any("monotonicity" in v.lower() for v in res["violations"])

    def test_decreasing_exponential_decay_fails(self, verifier):
        """exp(-x) has negative derivative -exp(-x) < 0 on [0, 5], must fail monotonicity."""
        res = verifier.verify_symbolic_invariants("exp(-x)", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None

    def test_inverted_hill_response_decreasing_fails(self, verifier):
        """1 / (1 + x) decreases with dose on [0, 5], must fail monotonicity."""
        res = verifier.verify_symbolic_invariants("1 / (1 + x)", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None

    def test_oscillating_sinusoidal_function_fails(self, verifier):
        """sin(x) is not monotonic on [0, 5] (derivative cos(x) < 0 for x in (pi/2, 3pi/2)), must fail."""
        res = verifier.verify_symbolic_invariants("sin(x)", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        ce_x = res["counterexample"][0]
        assert math.cos(ce_x) < 0.0

    def test_non_monotonic_parabola_fails(self, verifier):
        """x**2 - 4*x has negative slope on [0, 2), must fail monotonicity with counterexample."""
        res = verifier.verify_symbolic_invariants("x**2 - 4*x", domain="biology")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] is not None
        ce_x = res["counterexample"][0]
        assert 2 * ce_x - 4 < 0.0


# ==============================================================================
# Category 4: Boundary Condition Singularity / Pole Detection & Bounds
# ==============================================================================


class TestBoundaryConditionSingularityPoleDetectionChallenger:
    """Empirical verification of boundary singularities, asymptotic poles, and bounds."""

    @pytest.fixture
    def verifier(self):
        return DomainEvidenceVerifier()

    def test_pole_at_origin_reciprocal_fails(self, verifier):
        """1 / x has a singularity pole at x=0; must fail boundary invariant."""
        res = verifier.verify_symbolic_invariants("1 / x", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("pole" in v.lower() or "singularity" in v.lower() for v in res["violations"])
        assert res["counterexample"] is not None
        assert res["counterexample"] == [0.0]

    def test_pole_at_positive_point_fails(self, verifier):
        """1 / (x - 2) has a singularity pole at x=2; must fail boundary invariant."""
        res = verifier.verify_symbolic_invariants("1 / (x - 2)", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("pole" in v.lower() or "singularity" in v.lower() for v in res["violations"])
        assert res["counterexample"] is not None
        assert res["counterexample"] == [2.0]

    def test_higher_order_pole_fails(self, verifier):
        """1 / x**2 has a pole at x=0; must fail boundary invariant."""
        res = verifier.verify_symbolic_invariants("1 / x**2", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("pole" in v.lower() or "singularity" in v.lower() for v in res["violations"])

    def test_rational_with_numerator_pole_fails(self, verifier):
        """(x**2 + 5) / x has denominator zero at x=0; must fail boundary invariant."""
        res = verifier.verify_symbolic_invariants("(x**2 + 5) / x", domain="boundary")
        assert res["all_satisfied"] is False
        assert any("pole" in v.lower() or "singularity" in v.lower() for v in res["violations"])

    def test_logistic_sigmoid_regularity_no_poles_passes(self, verifier):
        """1 / (1 + exp(-x)) is pole-free and bounded in [0, 1]; must pass boundary invariant."""
        res = verifier.verify_symbolic_invariants("1 / (1 + exp(-x))", domain="boundary")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_scaled_sigmoid_regularity_no_poles_passes(self, verifier):
        """1 / (1 + exp(-2*x)) is pole-free and bounded in [0, 1]; must pass boundary invariant."""
        res = verifier.verify_symbolic_invariants("1 / (1 + exp(-2*x))", domain="boundary")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_boundary_upper_bound_overflow_fails(self, verifier):
        """2 / (1 + exp(-x)) has range (0, 2), exceeding upper_bound 1.0; must fail."""
        res = verifier.verify_symbolic_invariants("2 / (1 + exp(-x))", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("boundary" in v.lower() or "bound" in v.lower() for v in res["violations"])

    def test_boundary_lower_bound_underflow_fails(self, verifier):
        """-1 / (1 + exp(-x)) has range (-1, 0), violating lower_bound 0.0; must fail."""
        res = verifier.verify_symbolic_invariants("-1 / (1 + exp(-x))", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0

    def test_unbounded_linear_function_fails_boundary(self, verifier):
        """x on [-10, 10] exceeds [0, 1] range; must fail boundary invariant."""
        res = verifier.verify_symbolic_invariants("x", domain="boundary")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0


# ==============================================================================
# Category 5: Ill-formed Expressions & Edge Cases Handling
# ==============================================================================


class TestIllFormedExpressionsAndEdgeCasesChallenger:
    """Empirical verification of syntax error trapping, edge cases, and robustness."""

    @pytest.fixture
    def verifier(self):
        return DomainEvidenceVerifier()

    @pytest.mark.parametrize(
        "bad_expr",
        [
            "x +* 3 / )",
            "((((x + 1)",
            "x ^^ 2",
            "x + / 4",
            "1.0.0 + x",
            "",
            "   ",
            "def foo(): return 1",
            "!",
        ],
    )
    def test_malformed_syntax_trapped_gracefully(self, verifier, bad_expr):
        """Malformed syntax strings are trapped without raising unhandled exceptions."""
        res = verifier.verify_symbolic_invariants(bad_expr, domain="physics")
        assert isinstance(res, dict)
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert any("syntax" in v.lower() or "error" in v.lower() for v in res["violations"])
        assert res["counterexample"] is None
        assert "execution_time_ms" in res
        assert res["execution_time_ms"] >= 0.0

    def test_constant_expression_satisfies_bounds(self, verifier):
        """Positive constant '42.0' satisfies physics lower bound (0.0)."""
        res = verifier.verify_symbolic_invariants("42.0", domain="physics")
        assert res["all_satisfied"] is True
        assert len(res["violations"]) == 0

    def test_constant_expression_violates_bounds(self, verifier):
        """Negative constant '-42.0' violates physics lower bound (0.0)."""
        res = verifier.verify_symbolic_invariants("-42.0", domain="physics")
        assert res["all_satisfied"] is False
        assert len(res["violations"]) > 0
        assert res["counterexample"] == [-42.0]

    def test_domain_none_evaluates_all_three_plugins(self, verifier):
        """When domain=None, verifier evaluates constraints across physics, biology, and boundary."""
        res = verifier.verify_symbolic_invariants("exp(x)", domain=None)
        assert isinstance(res, dict)
        assert "evaluations" in res
        domains_evaluated = {e["domain"] for e in res["evaluations"]}
        assert "physics" in domains_evaluated
        assert "biology" in domains_evaluated
        assert "boundary" in domains_evaluated

    def test_non_differentiable_function_handled(self, verifier):
        """Non-differentiable abs(x) at origin handled cleanly without unhandled exception."""
        res = verifier.verify_symbolic_invariants("abs(x)", domain="biology")
        assert isinstance(res, dict)
        assert "all_satisfied" in res

    def test_verification_latency_budget(self, verifier):
        """Evaluation executes well within the 200 ms latency budget."""
        res = verifier.verify_symbolic_invariants("x**2 + 10", domain="physics")
        assert res["execution_time_ms"] < 200.0, f"Execution took {res['execution_time_ms']} ms"


# ==============================================================================
# Category 6: Adversarial Bug Reproducers & Ground-Truth Challenges
# ==============================================================================


class TestChallengerBugReproducers:
    """Reproducible empirical proofs of identified bugs in domain_evidence.py."""

    @pytest.fixture
    def verifier(self):
        return DomainEvidenceVerifier()

    def test_bug1_canonical_hill_equation_false_pole_rejection(self, verifier):
        """BUG 1 PROOF: Canonical Hill dose-response x / (1 + x) on [0, 5] is falsely rejected.
        
        Mathematical Ground Truth:
        - Function f(x) = x / (1 + x) is monotonically increasing: f'(x) = 1 / (1 + x)^2 > 0 for all x >= 0.
        - Denominator root x = -1 is strictly outside the domain [0.0, 5.0].
        - This is the EXACT reference symbolic_expression registered in BiologyMonotonicityPlugin.
        
        Failure Mode Observed:
        - verify_symbolic_invariants checks verify_l0_analytic without checking domain bounds.
        - Returns all_satisfied=False with pole 'x = -1' and counterexample [-1.0].
        """
        res = verifier.verify_symbolic_invariants("x / (1 + x)", domain="biology")
        # Ground truth expectation: all_satisfied should be True
        assert res["all_satisfied"] is True, (
            f"BUG 1 REPRODUCED: Canonical dose-response 'x / (1 + x)' falsely failed verification! "
            f"Violations: {res['violations']}, Counterexample: {res['counterexample']}"
        )

    def test_bug1_logarithmic_dose_response_false_pole_rejection(self, verifier):
        """BUG 1 PROOF: log(x + 1) on [0, 5] is monotonically increasing, but falsely rejected.
        
        Mathematical Ground Truth:
        - d/dx log(x + 1) = 1 / (x + 1) > 0 for all x in [0, 5].
        - Log singularity argument x + 1 = 0 => x = -1, strictly outside [0, 5].
        
        Failure Mode Observed:
        - Falsely flagged as having pole 'x = -1 (log pole)' on biology domain [0, 5].
        """
        res = verifier.verify_symbolic_invariants("log(x + 1)", domain="biology")
        assert res["all_satisfied"] is True, (
            f"BUG 1 REPRODUCED: 'log(x + 1)' falsely failed verification! Violations: {res['violations']}"
        )

    def test_bug1_rational_physics_positive_energy_false_pole_rejection(self, verifier):
        """BUG 1 PROOF: (x^2 + 1) / (x + 10) on [0, 5] has positive energy everywhere, but falsely rejected.
        
        Mathematical Ground Truth:
        - For x in [0, 5], x^2 + 1 >= 1 and x + 10 >= 10 => (x^2 + 1) / (x + 10) >= 0.10 > 0.
        - Denominator pole is at x = -10, strictly outside physical domain [0, 5].
        
        Failure Mode Observed:
        - Falsely flagged as having pole 'x = -10' on physics domain [0, 5].
        """
        res = verifier.verify_symbolic_invariants("(x**2 + 1) / (x + 10)", domain="physics")
        assert res["all_satisfied"] is True, (
            f"BUG 1 REPRODUCED: '(x**2 + 1) / (x + 10)' falsely rejected! Violations: {res['violations']}"
        )

    def test_bug2_multivariable_negative_energy_away_from_origin_false_pass(self, verifier):
        """BUG 2 PROOF: 1 - x - y violates energy conservation on [0, 5]x[0, 5], but falsely passes.
        
        Mathematical Ground Truth:
        - At origin (0, 0), f(0, 0) = 1.0 >= 0.
        - At (5, 5), f(5, 5) = 1 - 5 - 5 = -9.0 < 0 (violates lower_bound 0.0).
        - Over 90% of the domain [0, 5]x[0, 5], f(x, y) < 0.
        
        Failure Mode Observed:
        - verify_symbolic_invariants only evaluates at origin {s: 0.0 for s in free_symbols}.
        - Since f(0, 0) >= 0, it falsely reports all_satisfied=True, tier_used='L0_ANALYTIC'!
        """
        res = verifier.verify_symbolic_invariants("1 - x - y", domain="physics")
        # Ground truth expectation: all_satisfied must be False
        assert res["all_satisfied"] is False, (
            "BUG 2 REPRODUCED: '1 - x - y' falsely passed physical energy conservation! "
            "Evaluated only at origin (0, 0), ignoring domain [0, 5]x[0, 5] where f(5, 5) = -9.0."
        )

    def test_bug2_multivariable_quadratic_negative_energy_false_pass(self, verifier):
        """BUG 2 PROOF: 1 - x^2 - y^2 violates conservation on [0, 5]x[0, 5], but falsely passes.
        
        Mathematical Ground Truth:
        - At origin (0, 0), f(0, 0) = 1.0 >= 0.
        - At (5, 5), f(5, 5) = 1 - 25 - 25 = -49.0 < 0.
        
        Failure Mode Observed:
        - Falsely returns all_satisfied=True because it only tests (0, 0).
        """
        res = verifier.verify_symbolic_invariants("1 - x**2 - y**2", domain="physics")
        assert res["all_satisfied"] is False, (
            "BUG 2 REPRODUCED: '1 - x**2 - y**2' falsely passed physical conservation! "
            "Evaluated only at origin (0, 0), ignoring domain [0, 5]x[0, 5] where f(5, 5) = -49.0."
        )
