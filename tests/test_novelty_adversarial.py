"""Adversarial challenge test suite for Milestone M2: Dual-Key Novelty Contract.

Rigorous stress-testing of numerical and statistical manifold divergence logic:
1. Numerical stability under pathological input vectors:
   - Zero vectors, constant vectors, highly skewed distributions
   - Extreme scales (1e-15 up to 1e12)
   - Massive dimension vectors (N = 100,000) and tiny dimension vectors (N = 1, 2)
   - Negative correlations, collinear scaling with positive and negative multipliers
2. Verification of strict boundedness:
   - Output ranges strictly bounded in [0.0, 1.0] for d_func, [-1.0, 1.0] for rho
   - No ZeroDivisionError, no NaN, no Inf, no unexpected crashes
   - Valid Pydantic NoveltyContractResult generation across all edge cases
"""

import math
import numpy as np
import pytest

from frame.schemas.novelty import NoveltyContractResult, NoveltyVerdict
from frame.verification.novelty import DualKeyNoveltyContract


@pytest.fixture
def contract():
    return DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)


def assert_valid_divergence_metrics(d_func: float, rho: float):
    """Universal oracle asserting non-NaN, non-Inf, and strict bounding."""
    assert isinstance(d_func, float), f"d_func is not float: {type(d_func)}"
    assert isinstance(rho, float), f"rho is not float: {type(rho)}"
    assert not math.isnan(d_func), f"d_func is NaN: {d_func}"
    assert not math.isnan(rho), f"rho is NaN: {rho}"
    assert not math.isinf(d_func), f"d_func is Inf: {d_func}"
    assert not math.isinf(rho), f"rho is Inf: {rho}"
    assert 0.0 <= d_func <= 1.0, f"d_func out of [0.0, 1.0]: {d_func}"
    assert -1.0 <= rho <= 1.0, f"rho out of [-1.0, 1.0]: {rho}"


class TestZeroVectorsAndPathologicalBaselines:
    """Adversarial suite for zero vectors and baseline identity edge cases."""

    @pytest.mark.parametrize("n", [1, 2, 5, 100, 10000, 100000])
    def test_all_zero_inputs(self, contract, n):
        y_true = np.zeros(n)
        y_base = np.zeros(n)
        y_cand = np.zeros(n)
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 0.0
        assert rho == 1.0

    def test_empty_vector_holdout(self, contract):
        d_func, rho = contract.compute_functional_divergence(
            np.array([]), np.array([]), np.array([])
        )
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 0.0
        assert rho == 1.0

    def test_candidate_exact_ground_truth_zero_candidate_residual(self, contract):
        y_true = np.linspace(1.0, 10.0, 50)
        y_base = y_true + 2.0
        y_cand = y_true.copy()  # r_c is exactly zero vector
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 1.0
        assert rho == 0.0

    def test_baseline_exact_ground_truth_zero_baseline_residual(self, contract):
        y_true = np.linspace(1.0, 10.0, 50)
        y_base = y_true.copy()  # r_b is exactly zero vector
        y_cand = y_true + 2.0
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 1.0
        assert rho == 0.0

    def test_both_exact_ground_truth_zero_residuals(self, contract):
        y_true = np.array([42.0, 13.37, 99.0])
        y_base = y_true.copy()
        y_cand = y_true.copy()
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 0.0
        assert rho == 1.0


class TestConstantVectorsAndShifts:
    """Adversarial suite for constant vectors and constant shift offsets."""

    @pytest.mark.parametrize("n", [1, 2, 10, 500])
    @pytest.mark.parametrize("c", [0.0, 1.0, -5.0, 1e-12, 1e8])
    def test_identical_constant_predictions(self, contract, n, c):
        y_true = np.zeros(n)
        y_base = np.full(n, c)
        y_cand = np.full(n, c)
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 0.0
        assert rho == 1.0

    def test_positive_constant_shifts_collinear(self, contract):
        y_true = np.zeros(20)
        y_base = np.full(20, 2.0)
        y_cand = np.full(20, 10.0)  # r_c = 5 * r_b
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-10)
        assert math.isclose(rho, 1.0, abs_tol=1e-10)

    def test_opposite_constant_shifts_anti_collinear(self, contract):
        y_true = np.zeros(20)
        y_base = np.full(20, 2.0)
        y_cand = np.full(20, -10.0)  # r_c = -5 * r_b
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-10)
        assert math.isclose(rho, -1.0, abs_tol=1e-10)


class TestExtremeScales:
    """Adversarial suite for extreme magnitudes spanning 1e-15 to 1e12."""

    @pytest.mark.parametrize("scale", [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12])
    def test_collinear_at_extreme_scales(self, contract, scale):
        y_true = np.zeros(30)
        base = np.linspace(1.0, 2.0, 30) * scale
        cand = base * 2.5
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)

    @pytest.mark.parametrize("scale", [1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e6, 1e9, 1e12])
    def test_anti_collinear_at_extreme_scales(self, contract, scale):
        y_true = np.zeros(30)
        base = np.linspace(1.0, 2.0, 30) * scale
        cand = -base * 3.0
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        # Note: if scale < 1e-12, atol=1e-12 in np.allclose may clamp to 1.0 or -1.0; both yield d_func=0.0
        assert rho in (1.0, -1.0) or math.isclose(rho, -1.0, abs_tol=1e-5)

    @pytest.mark.parametrize("scale", [1e-10, 1e-6, 1.0, 1e6, 1e12])
    def test_orthogonal_at_resolvable_scales(self, contract, scale):
        y_true = np.zeros(4)
        base = np.array([scale, scale, 0.0, 0.0])
        cand = np.array([0.0, 0.0, scale, scale])
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(rho, 0.0, abs_tol=1e-8)
        assert math.isclose(d_func, 1.0, abs_tol=1e-8)

    def test_sub_picolevel_atol_resolution_boundary(self, contract):
        """Documents and stresses behavior when residuals are below atol=1e-12."""
        y_true = np.zeros(2)
        # Sub-pico scale: both vectors have norm ~ 1e-15
        base = np.array([1e-15, 0.0])
        cand = np.array([0.0, 1e-15])
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        # Verify strict boundedness and no crash
        assert_valid_divergence_metrics(d_func, rho)
        # Because |base - cand| = sqrt(2)*1e-15 <= atol=1e-12, np.allclose treats as identical
        assert d_func == 0.0
        assert rho == 1.0

    def test_disparate_magnitudes_base_huge_candidate_tiny(self, contract):
        y_true = np.zeros(10)
        base = np.ones(10) * 1e12
        cand = np.ones(10) * 1e-15
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        # Both are positive constant vectors -> collinear
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)

    def test_disparate_magnitudes_base_huge_candidate_tiny_orthogonal(self, contract):
        y_true = np.zeros(2)
        base = np.array([1e12, 0.0])
        cand = np.array([0.0, 1e-10])  # resolvable scale > 1e-12
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(rho, 0.0, abs_tol=1e-5)
        assert math.isclose(d_func, 1.0, abs_tol=1e-5)

    def test_extreme_dynamic_range_within_single_vector(self, contract):
        y_true = np.zeros(4)
        base = np.array([1e-15, 1.0, 1e6, 1e12])
        cand = base * 2.0
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)


class TestDimensionalityExtremes:
    """Adversarial suite for tiny (N=1, 2) and massive (N=100,000) dimensions."""

    def test_tiny_dimension_n1_collinear_positive(self, contract):
        y_true = np.array([5.0])
        y_base = np.array([7.0])   # r_b = +2.0
        y_cand = np.array([11.0])  # r_c = +6.0
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-10)
        assert math.isclose(rho, 1.0, abs_tol=1e-10)

    def test_tiny_dimension_n1_collinear_negative(self, contract):
        y_true = np.array([5.0])
        y_base = np.array([7.0])   # r_b = +2.0
        y_cand = np.array([1.0])   # r_c = -4.0
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-10)
        assert math.isclose(rho, -1.0, abs_tol=1e-10)

    def test_tiny_dimension_n2_orthogonal(self, contract):
        y_true = np.array([0.0, 0.0])
        y_base = np.array([1.0, 0.0])
        y_cand = np.array([0.0, 1.0])
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 1.0, abs_tol=1e-10)
        assert math.isclose(rho, 0.0, abs_tol=1e-10)

    def test_tiny_dimension_n2_angle_45_deg(self, contract):
        y_true = np.zeros(2)
        y_base = np.array([1.0, 0.0])
        y_cand = np.array([1.0, 1.0])  # cos(45 deg) = 1/sqrt(2) approx 0.70710678
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        expected_rho = 1.0 / math.sqrt(2.0)
        assert math.isclose(rho, expected_rho, abs_tol=1e-7)
        assert math.isclose(d_func, 1.0 - expected_rho, abs_tol=1e-7)

    def test_massive_dimension_n100k_collinear(self, contract):
        n = 100000
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(n)
        r_b = rng.standard_normal(n)
        y_base = y_true + r_b
        y_cand = y_true + (r_b * 3.75)  # collinear
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)

    def test_massive_dimension_n100k_orthogonal_random(self, contract):
        n = 100000
        rng = np.random.default_rng(123)
        y_true = np.zeros(n)
        y_base = rng.standard_normal(n)
        y_cand = rng.standard_normal(n)
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        # Independent N(0, 1) vectors in R^100000 have correlation ~ O(1/sqrt(100000)) ~ 0.003
        assert abs(rho) < 0.02
        assert d_func > 0.98


class TestHighlySkewedDistributionsAndOutliers:
    """Adversarial suite for skewed distributions, sparse spikes, and heavy tails."""

    def test_single_sparse_spike_identical(self, contract):
        n = 1000
        y_true = np.zeros(n)
        y_base = np.zeros(n)
        y_base[0] = 1e6
        y_cand = np.zeros(n)
        y_cand[0] = 1e6
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert d_func == 0.0
        assert rho == 1.0

    def test_single_sparse_spike_disjoint_support(self, contract):
        n = 1000
        y_true = np.zeros(n)
        y_base = np.zeros(n)
        y_base[0] = 1e6
        y_cand = np.zeros(n)
        y_cand[1] = 1e6  # Completely orthogonal support
        d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 1.0, abs_tol=1e-10)
        assert math.isclose(rho, 0.0, abs_tol=1e-10)

    def test_heavy_tailed_pareto_residuals(self, contract):
        rng = np.random.default_rng(42)
        n = 5000
        y_true = np.zeros(n)
        # Pareto distribution with alpha = 1.1 (heavy-tailed, infinite variance in limit)
        r_b = (rng.pareto(a=1.1, size=n) + 1.0) * 10.0
        r_c = r_b * (-4.2)  # Negatively collinear heavy-tailed
        d_func, rho = contract.compute_functional_divergence(y_true, y_true + r_b, y_true + r_c)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, -1.0, abs_tol=1e-5)

    def test_lognormal_extreme_skewness(self, contract):
        rng = np.random.default_rng(999)
        n = 2000
        y_true = np.zeros(n)
        # Log-normal with sigma = 4.0 creates massive multi-order-of-magnitude skewness
        r_b = rng.lognormal(mean=0.0, sigma=4.0, size=n)
        r_c = rng.lognormal(mean=0.0, sigma=4.0, size=n)
        d_func, rho = contract.compute_functional_divergence(y_true, r_b, r_c)
        assert_valid_divergence_metrics(d_func, rho)


class TestNegativeCorrelationsAndCollinearMultipliers:
    """Adversarial suite for negative correlations and arbitrary multipliers."""

    @pytest.mark.parametrize("multiplier", [
        -1e12, -1e6, -100.0, -2.0, -1.0, -0.5, -0.01, -1e-6, -1e-12, -1e-15,
        1e-15, 1e-12, 1e-6, 0.01, 0.5, 1.0, 2.0, 100.0, 1e6, 1e12,
    ])
    def test_collinear_multipliers_sweep(self, contract, multiplier):
        y_true = np.zeros(50)
        base = np.linspace(1.0, 10.0, 50)
        cand = base * multiplier
        d_func, rho = contract.compute_functional_divergence(y_true, base, cand)
        assert_valid_divergence_metrics(d_func, rho)
        # All collinear scalings must have d_func = 0.0
        assert math.isclose(d_func, 0.0, abs_tol=1e-4)
        if multiplier > 0:
            assert math.isclose(rho, 1.0, abs_tol=1e-4)
        else:
            # If magnitude < 1e-12, atol in np.allclose could match +1.0; otherwise -1.0
            assert rho in (1.0, -1.0) or math.isclose(rho, -1.0, abs_tol=1e-4)

    @pytest.mark.parametrize("target_rho", [-1.0, -0.9, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_synthetic_target_correlation_matching(self, contract, target_rho):
        n = 10000
        rng = np.random.default_rng(2026)
        x = rng.standard_normal(n)
        z = rng.standard_normal(n)
        # Gram-Schmidt to make z orthogonal to x
        z = z - (np.dot(x, z) / np.dot(x, x)) * x
        z = z / np.linalg.norm(z) * np.linalg.norm(x)
        # Construct y with exact correlation target_rho
        y = target_rho * x + math.sqrt(max(0.0, 1.0 - target_rho**2)) * z

        y_true = np.zeros(n)
        d_func, rho = contract.compute_functional_divergence(y_true, x, y)
        assert_valid_divergence_metrics(d_func, rho)
        assert math.isclose(rho, target_rho, abs_tol=1e-3)
        expected_d_func = 1.0 - abs(target_rho)
        assert math.isclose(d_func, expected_d_func, abs_tol=1e-3)


class TestDualKeyContractEndToEndRobustness:
    """Adversarial suite for evaluate_novelty end-to-end integration across all edge cases."""

    def test_evaluate_novelty_pathological_matrix(self, contract):
        scenarios = [
            ("all_zeros", np.zeros(10), np.zeros(10), np.zeros(10)),
            ("constant_scale_1e12", np.zeros(10), np.full(10, 1e12), np.full(10, 2e12)),
            ("constant_scale_1e_minus_15", np.zeros(10), np.full(10, 1e-15), np.full(10, 2e-15)),
            ("n_1_vector", np.array([1.0]), np.array([2.0]), np.array([3.0])),
            ("n_100k_vector", np.zeros(100000), np.ones(100000), np.ones(100000) * 3.0),
            ("anti_collinear", np.zeros(20), np.ones(20), -np.ones(20)),
        ]

        for name, yt, yb, yc in scenarios:
            res = contract.evaluate_novelty(
                hypothesis_id=f"hyp_{name}",
                candidate_code="def novelty_test(): return 42",
                prior_art_corpus=["def prior_art(): return 0"],
                y_true=yt,
                y_baseline_pred=yb,
                y_candidate_pred=yc,
            )
            # Verify result schema integrity
            assert isinstance(res, NoveltyContractResult)
            assert res.hypothesis_id == f"hyp_{name}"
            assert isinstance(res.passed, bool)
            assert isinstance(res.verdict, NoveltyVerdict)
            assert_valid_divergence_metrics(res.d_func, res.residual_correlation)
            assert 0.0 <= res.d_lit <= 1.0
            assert res.contract_hash is not None
            assert len(res.contract_hash) == 64


class TestPropertyBasedRandomizedFuzzing:
    """Randomized fuzz harness generating diverse pathological inputs."""

    def test_randomized_adversarial_vector_fuzzing_1000_iterations(self, contract):
        rng = np.random.default_rng(20261005)
        total_tests = 1000

        for i in range(total_tests):
            n = int(rng.choice([1, 2, 3, 5, 10, 50, 100, 1000]))
            scale_y = float(10.0 ** rng.uniform(-15, 12))
            scale_b = float(10.0 ** rng.uniform(-15, 12))
            scale_c = float(10.0 ** rng.uniform(-15, 12))

            dist_choice = rng.choice(["zeros", "constant", "normal", "cauchy", "pareto", "exponential", "spikes"])
            if dist_choice == "zeros":
                yt = np.zeros(n)
                yb = np.zeros(n)
                yc = np.zeros(n)
            elif dist_choice == "constant":
                yt = np.full(n, rng.uniform(-1, 1) * scale_y)
                yb = np.full(n, rng.uniform(-1, 1) * scale_b)
                yc = np.full(n, rng.uniform(-1, 1) * scale_c)
            elif dist_choice == "spikes":
                yt = np.zeros(n)
                yb = np.zeros(n)
                yc = np.zeros(n)
                idx_b = int(rng.integers(0, n))
                idx_c = int(rng.integers(0, n))
                yb[idx_b] = scale_b
                yc[idx_c] = scale_c
            elif dist_choice == "cauchy":
                yt = rng.standard_cauchy(n) * scale_y
                yb = rng.standard_cauchy(n) * scale_b
                yc = rng.standard_cauchy(n) * scale_c
            elif dist_choice == "pareto":
                yt = (rng.pareto(1.1, n) + 1.0) * scale_y
                yb = (rng.pareto(1.1, n) + 1.0) * scale_b
                yc = (rng.pareto(1.1, n) + 1.0) * scale_c
            elif dist_choice == "exponential":
                yt = rng.exponential(1.0, n) * scale_y
                yb = rng.exponential(1.0, n) * scale_b
                yc = rng.exponential(1.0, n) * scale_c
            else:
                yt = rng.standard_normal(n) * scale_y
                yb = rng.standard_normal(n) * scale_b
                yc = rng.standard_normal(n) * scale_c

            # Inject collinear scaling with positive or negative multipliers
            if rng.random() < 0.25:
                mult = float(rng.choice([-1.0, 1.0, -2.0, 2.0, 0.5, -0.5, 1e-15, -1e-15, 1e12, -1e12]))
                yc = yt + (yb - yt) * mult

            d_func, rho = contract.compute_functional_divergence(yt, yb, yc)
            assert_valid_divergence_metrics(d_func, rho)

            # Periodically test end-to-end evaluate_novelty schema and contract hashing
            if i % 100 == 0:
                res = contract.evaluate_novelty(
                    hypothesis_id=f"fuzz_hyp_{i}",
                    candidate_code=f"def model_{i}(): return {i}",
                    prior_art_corpus=["def prior_model(): return 0"],
                    y_true=yt,
                    y_baseline_pred=yb,
                    y_candidate_pred=yc,
                )
                assert isinstance(res, NoveltyContractResult)
                assert_valid_divergence_metrics(res.d_func, res.residual_correlation)
                assert 0.0 <= res.d_lit <= 1.0
                assert res.contract_hash is not None

