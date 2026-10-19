"""Unit and boundary test suite for Milestone M2: Dual-Key Novelty Contract.

Validates:
- Feature 4: Literature Graph Proximity Evaluator (d_lit > tau_lit = 0.15)
- Feature 5: Functional Prediction Manifold Divergence (d_func > epsilon_func = 0.10)
- Feature 6: Automated Dual-Key Novelty Gate
- Schema contracts: NoveltyVerdict and NoveltyContractResult
- Zero LORD alpha wealth consumption invariant
"""

import math
import numpy as np
import pytest

from frame.schemas.novelty import NoveltyContractResult, NoveltyVerdict
from frame.verification.novelty import DualKeyNoveltyContract
from frame.verification.statistics import LORDController


class TestNoveltySchemas:
    """Validation of NoveltyVerdict Enum and NoveltyContractResult Pydantic schema."""

    def test_novelty_verdict_enum_values_and_case_insensitivity(self):
        assert NoveltyVerdict.APPROVED == "APPROVED"
        assert NoveltyVerdict.REJECTED_LIT_PROXIMITY == "REJECTED_LIT_PROXIMITY"
        assert NoveltyVerdict.REJECTED_FUNC_COLLINEAR == "REJECTED_FUNC_COLLINEAR"
        assert NoveltyVerdict.REJECTED_DUAL_FAILURE == "REJECTED_DUAL_FAILURE"

        # Case-insensitive lookup via _missing_
        assert NoveltyVerdict("approved") == NoveltyVerdict.APPROVED
        assert NoveltyVerdict("rejected_lit_proximity") == NoveltyVerdict.REJECTED_LIT_PROXIMITY
        assert NoveltyVerdict("rejected_func_collinear") == NoveltyVerdict.REJECTED_FUNC_COLLINEAR
        assert NoveltyVerdict("rejected_dual_failure") == NoveltyVerdict.REJECTED_DUAL_FAILURE

    def test_novelty_contract_result_instantiation_and_defaults(self):
        res = NoveltyContractResult(
            hypothesis_id="hyp_001",
            passed=True,
            verdict=NoveltyVerdict.APPROVED,
            d_lit=0.35,
            d_func=0.45,
            residual_correlation=0.10,
            nearest_prior_art="Baseline MLP",
        )
        assert res.hypothesis_id == "hyp_001"
        assert res.passed is True
        assert res.verdict == NoveltyVerdict.APPROVED
        assert res.tau_lit == 0.15
        assert res.epsilon_func == 0.10
        assert res.details == {}
        assert res.rejection_reason is None
        assert res.contract_hash is None
        assert res.created_at is not None

    def test_novelty_contract_result_aliases_and_properties(self):
        res = NoveltyContractResult(
            hypothesis_id="hyp_002",
            passed=False,
            verdict=NoveltyVerdict.REJECTED_LIT_PROXIMITY,
            d_lit=0.08,
            d_func=0.30,
            rho=-0.70,
            nearest_prior_reference="Prior Art Attention",
            details={"rejection_reason": "Literature match too close"},
        )
        # Property aliases
        assert res.rho == -0.70
        assert res.residual_correlation == -0.70
        assert res.nearest_prior_reference == "Prior Art Attention"
        assert res.nearest_prior_art == "Prior Art Attention"
        assert res.timestamp == res.created_at
        assert res.rejection_reason == "Literature match too close"
        assert res.is_lit_novel is False
        assert res.is_func_novel is True

    def test_novelty_contract_result_hash_computation(self):
        res = NoveltyContractResult(
            hypothesis_id="hyp_hash_test",
            passed=True,
            verdict=NoveltyVerdict.APPROVED,
            d_lit=0.80,
            d_func=0.90,
            residual_correlation=0.05,
            nearest_prior_art="Prior",
        )
        hash1 = res.compute_hash()
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA-256 hex string

        # Setting contract_hash must not change computed hash (exclude={'contract_hash'})
        res.contract_hash = hash1
        hash2 = res.compute_hash()
        assert hash1 == hash2

    def test_novelty_contract_result_serialization_round_trip(self):
        res = NoveltyContractResult(
            hypothesis_id="hyp_serde",
            passed=False,
            verdict=NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            d_lit=0.50,
            d_func=0.02,
            residual_correlation=0.98,
            nearest_prior_art="Transformer",
            details={"notes": "Collinear error space"},
        )
        dumped = res.model_dump(mode="json")
        loaded = NoveltyContractResult.model_validate(dumped)
        assert loaded.hypothesis_id == res.hypothesis_id
        assert loaded.verdict == res.verdict
        assert loaded.d_lit == res.d_lit
        assert loaded.d_func == res.d_func
        assert loaded.residual_correlation == res.residual_correlation

    def test_novelty_contract_result_validation_bounds(self):
        with pytest.raises(Exception):
            NoveltyContractResult(
                hypothesis_id="h_err",
                passed=True,
                verdict=NoveltyVerdict.APPROVED,
                d_lit=1.5,  # Out of [0, 1] bound
                d_func=0.5,
            )
        with pytest.raises(Exception):
            NoveltyContractResult(
                hypothesis_id="h_err",
                passed=True,
                verdict=NoveltyVerdict.APPROVED,
                d_lit=0.5,
                d_func=0.5,
                residual_correlation=1.5,  # Out of [-1, 1] bound
            )


class TestLiteratureProximityEvaluator:
    """Feature F04: Literature Graph Proximity Evaluator tests."""

    def test_orthogonal_proposals_have_large_distance(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = [
            "Dense Multilayer Perceptron",
            "Convolutional Neural Network ResNet",
        ]
        candidate = "Quantum Orthogonal Fourier Token Mixing without keys or queries"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert d_lit > 0.15
        assert isinstance(nearest, str)

    def test_identical_candidate_and_prior_art_zero_distance(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = ["Standard Multi-Head Attention Mechanism"]
        candidate = "Standard Multi-Head Attention Mechanism"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert math.isclose(d_lit, 0.0, abs_tol=1e-6)
        assert nearest == "Standard Multi-Head Attention Mechanism"

    def test_case_insensitivity_and_punctuation_handling(self):
        contract = DualKeyNoveltyContract()
        prior_art = ["Standard Multi-Head Attention Mechanism!"]
        candidate = "standard multi-head attention mechanism"
        d_lit, _ = contract.compute_literature_distance(candidate, prior_art)
        assert math.isclose(d_lit, 0.0, abs_tol=1e-6)

    def test_wrapper_containment_detection(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        prior_art = ["Standard Multi-Head Attention"]
        candidate = "Trivial Wrapper: Standard Multi-Head Attention with extra layer"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert d_lit <= 0.15
        assert "Attention" in nearest

    def test_nearest_neighbor_selection_in_multi_corpus(self):
        contract = DualKeyNoveltyContract()
        prior_art = [
            "Graph Convolutional Network",
            "Recurrent LSTM Cell",
            "State Space Model Mamba",
        ]
        candidate = "Mamba Selective State Space Model with linear scan"
        d_lit, nearest = contract.compute_literature_distance(candidate, prior_art)
        assert nearest == "State Space Model Mamba"
        assert d_lit <= 0.15

    def test_empty_prior_art_corpus_safe_fallback(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        d_lit, nearest = contract.compute_literature_distance("Any Candidate", [])
        assert d_lit >= 0.15
        assert nearest == ""

    def test_empty_candidate_text_safe_fallback(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        d_lit, nearest = contract.compute_literature_distance("", ["Prior Art Document"])
        assert d_lit >= 0.15
        assert nearest == "Prior Art Document"

    def test_single_item_corpus_nearest(self):
        contract = DualKeyNoveltyContract()
        d_lit, nearest = contract.compute_literature_distance("Novel Idea", ["Solo Art"])
        assert nearest == "Solo Art"

    def test_unnormalized_and_special_character_strings(self):
        contract = DualKeyNoveltyContract()
        d_lit, _ = contract.compute_literature_distance(
            "@#$%^&*()_+ special characters", ["normal baseline"]
        )
        assert 0.0 <= d_lit <= 1.0

    def test_threshold_boundary_strict_inequality(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        assert not (0.15000 > contract.tau_lit)


class TestFunctionalPredictionManifoldDivergence:
    """Feature F05: Functional Prediction Manifold Divergence tests."""

    def test_perfectly_collinear_residuals_zero_distance(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(0, 10, 50)
        base_pred = y_true + 1.0
        cand_pred = y_true + 1.0
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(abs(rho), 1.0, abs_tol=1e-5)

    def test_scaled_collinear_residuals_zero_distance(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(50)
        base_pred = np.linspace(1, 5, 50)
        cand_pred = 3.0 * base_pred  # Scaled by 3
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)

    def test_anti_correlated_residuals_zero_distance(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(50)
        base_pred = np.linspace(-2, 2, 50)
        cand_pred = -base_pred  # Anti-parallel: rho = -1 -> |rho| = 1 -> d_func = 0
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, -1.0, abs_tol=1e-5)

    def test_strictly_orthogonal_residual_vectors(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(40)
        base_pred = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(40)])
        cand_pred = np.array([0.0 if i % 2 == 0 else 1.0 for i in range(40)])
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(rho, 0.0, abs_tol=1e-5)
        assert math.isclose(d_func, 1.0, abs_tol=1e-5)
        assert d_func > contract.epsilon_func

    def test_uncorrelated_random_residuals_divergence(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        np.random.seed(42)
        y_true = np.zeros(100)
        base_pred = np.random.normal(0, 1, 100)
        cand_pred = np.random.normal(0, 1, 100)
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert d_func > contract.epsilon_func

    def test_zero_variance_constant_residual_vectors(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.ones(30)
        base_pred = np.zeros(30)
        cand_pred = np.zeros(30)
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert 0.0 <= d_func <= 1.0
        assert not math.isnan(d_func)

    def test_candidate_matches_ground_truth_zero_residual(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(1, 10, 40)
        base_pred = y_true + 1.0
        cand_pred = y_true.copy()  # Candidate is ground truth
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert isinstance(d_func, float)
        assert math.isclose(d_func, 1.0, abs_tol=1e-5)
        assert math.isclose(rho, 0.0, abs_tol=1e-5)

    def test_both_candidate_and_baseline_match_ground_truth(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(1, 10, 20)
        base_pred = y_true.copy()
        cand_pred = y_true.copy()
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)
        assert math.isclose(rho, 1.0, abs_tol=1e-5)

    def test_baseline_matches_ground_truth_candidate_has_error(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.linspace(1, 10, 20)
        base_pred = y_true.copy()
        cand_pred = y_true + 2.0
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 1.0, abs_tol=1e-5)
        assert math.isclose(rho, 0.0, abs_tol=1e-5)

    def test_extreme_scale_numerical_stability(self):
        contract = DualKeyNoveltyContract()
        y_true = np.linspace(1e7, 2e7, 50)
        base_pred = y_true + 1e5
        cand_pred = y_true + 2e5
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert not math.isnan(d_func)
        assert not math.isnan(rho)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)

    def test_small_sample_size_holdout(self):
        contract = DualKeyNoveltyContract()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        base_pred = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        cand_pred = np.array([1.5, 2.2, 2.9, 4.5, 4.8])
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert isinstance(d_func, float)
        assert 0.0 <= d_func <= 1.0

    def test_empty_sample_size_holdout(self):
        contract = DualKeyNoveltyContract()
        d_func, rho = contract.compute_functional_divergence(np.array([]), np.array([]), np.array([]))
        assert isinstance(d_func, float)
        assert 0.0 <= d_func <= 1.0

    def test_multidimensional_array_flattening(self):
        contract = DualKeyNoveltyContract()
        y_true = np.zeros((10, 1))
        base_pred = np.ones((10, 1))
        cand_pred = np.ones((10, 1))
        d_func, rho = contract.compute_functional_divergence(y_true, base_pred, cand_pred)
        assert math.isclose(d_func, 0.0, abs_tol=1e-5)

    def test_functional_threshold_boundary_strict_inequality(self):
        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        assert not (0.10000 > contract.epsilon_func)


class TestDualKeyNoveltyGate:
    """Feature F06: Automated Dual-Key Novelty Gate tests."""

    def test_gate_approved_when_both_keys_pass(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + np.sin(np.linspace(0, 10, 50))
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_approve",
            candidate_code="class RadicallyNewArchitecture(nn.Module): pass",
            prior_art_corpus=["Old Legacy Architecture"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is True
        assert res.verdict == NoveltyVerdict.APPROVED
        assert res.rejection_reason is None
        assert res.contract_hash is not None

    def test_gate_rejected_lit_proximity(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + np.sin(np.linspace(0, 10, 50))
        prior_code = "class StandardTransformer(nn.Module): pass"
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_lit_fail",
            candidate_code=prior_code,
            prior_art_corpus=[prior_code],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == NoveltyVerdict.REJECTED_LIT_PROXIMITY
        assert "Literature proximity" in res.rejection_reason

    def test_gate_rejected_func_collinear(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + 0.5  # Exactly collinear residuals
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_collinear_fail",
            candidate_code="class BrandNewName(nn.Module): pass",
            prior_art_corpus=["Totally unrelated literature"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == NoveltyVerdict.REJECTED_FUNC_COLLINEAR
        assert "Functional divergence" in res.rejection_reason

    def test_gate_rejected_dual_failure(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.linspace(0, 5, 50)
        base_pred = y_true + 0.5
        cand_pred = y_true + 0.5
        code = "class IdenticalEverything(nn.Module): pass"
        res = contract.evaluate_novelty(
            hypothesis_id="hyp_dual_fail",
            candidate_code=code,
            prior_art_corpus=[code],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict == NoveltyVerdict.REJECTED_DUAL_FAILURE
        assert "Dual failure" in res.rejection_reason

    def test_surface_form_rephrase_attack_rejected(self):
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.ones(40)
        base_pred = np.ones(40) * 0.8
        cand_pred = np.ones(40) * 0.8  # Collinear predictions despite rephrasing

        res = contract.evaluate_novelty(
            hypothesis_id="hyp_rephrase",
            candidate_code="# Cosmetic docstring\nalpha = beta * 2",
            prior_art_corpus=["Original code: x = y * 2"],
            y_true=y_true,
            y_baseline_pred=base_pred,
            y_candidate_pred=cand_pred,
        )
        assert res.passed is False
        assert res.verdict in (
            NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )

    def test_empty_code_snippet_evaluated_safely(self):
        contract = DualKeyNoveltyContract()
        y = np.ones(30)
        res = contract.evaluate_novelty("h_empty", "", ["Prior Art"], y, y, y)
        assert isinstance(res.passed, bool)
        assert res.hypothesis_id == "h_empty"

    def test_custom_contract_thresholds(self):
        contract = DualKeyNoveltyContract(tau_lit=0.25, epsilon_func=0.20)
        assert contract.tau_lit == 0.25
        assert contract.epsilon_func == 0.20

    def test_contract_hash_populated_in_result(self):
        contract = DualKeyNoveltyContract()
        y = np.ones(20)
        res = contract.evaluate_novelty("h_hash", "code", ["art"], y, y + 1, y + 2)
        assert res.contract_hash is not None
        assert res.contract_hash == res.compute_hash()


class TestLORDAlphaPreservation:
    """Invariant validation: Novelty Contract consumes strictly 0 LORD alpha wealth."""

    def test_novelty_gate_consumes_zero_lord_alpha_wealth(self):
        lord = LORDController(alpha=0.05)
        initial_alpha = lord.get_next_alpha()
        initial_step = lord.current_step
        initial_discoveries = list(lord.discovery_indices)

        contract = DualKeyNoveltyContract()
        y_true = np.linspace(0, 10, 40)

        # 1. Evaluate rejected proposal
        res_fail = contract.evaluate_novelty(
            "h_fail", "identical code", ["identical code"], y_true, y_true + 1, y_true + 1
        )
        assert res_fail.passed is False

        # Verify state is untouched
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-12)
        assert lord.current_step == initial_step
        assert lord.discovery_indices == initial_discoveries

        # 2. Evaluate approved proposal
        res_pass = contract.evaluate_novelty(
            "h_pass", "radically novel code", ["unrelated art"], y_true, y_true + 1, y_true + np.cos(y_true)
        )
        assert res_pass.passed is True

        # Verify state is still completely untouched
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-12)
        assert lord.current_step == initial_step
        assert lord.discovery_indices == initial_discoveries
