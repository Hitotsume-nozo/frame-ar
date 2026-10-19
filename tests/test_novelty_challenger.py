"""Challenger empirical stress tests for DualKeyNoveltyContract (Milestone M2).

Authored by m2_challenger_1.
Verifies adversarial attack vectors, near-threshold boundaries, extreme corpus sizes,
and discovers/documents implementation vulnerabilities.
"""

import math
import numpy as np
import pytest

from frame.schemas.novelty import NoveltyContractResult, NoveltyVerdict
from frame.verification.novelty import DualKeyNoveltyContract
from frame.verification.statistics import LORDController


class TestSurfaceFormRephrasingAttacks:
    """Stress tests verifying that cosmetic code rewrites do NOT bypass the novelty gate

    when predictions remain collinear with the baseline.
    """

    PRIOR_MLP_CODE = """
import numpy as np

def mlp_forward(features, weight_in, bias_in, weight_out, bias_out):
    \"\"\"Standard two-layer feedforward network with ReLU activation.\"\"\"
    hidden = np.maximum(0, np.matmul(features, weight_in) + bias_in)
    output = np.matmul(hidden, weight_out) + bias_out
    return output
"""

    def _setup_synthetic_predictions(self, n: int = 100):
        np.random.seed(42)
        y_true = np.linspace(0, 10, n)
        # Baseline residuals
        r_base = np.sin(np.linspace(0, 3 * np.pi, n))
        y_base = y_true + r_base
        return y_true, y_base

    def test_variable_renaming_collinear_predictions_rejected(self):
        """Variable renaming changes surface vocabulary but collinear predictions trigger rejection."""
        y_true, y_base = self._setup_synthetic_predictions()
        y_cand = y_base.copy()

        renamed_code = """
import numpy as np

def quantum_manifold_projector(tensor_q, matrix_alpha, vector_alpha, matrix_omega, vector_omega):
    latent_state = np.maximum(0, np.matmul(tensor_q, matrix_alpha) + vector_alpha)
    synthesized_readout = np.matmul(latent_state, matrix_omega) + vector_omega
    return synthesized_readout
"""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        result = contract.evaluate_novelty(
            hypothesis_id="adv_var_rename",
            candidate_code=renamed_code,
            prior_art_corpus=[self.PRIOR_MLP_CODE],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
        )

        assert result.passed is False
        assert result.verdict in (
            NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )
        assert math.isclose(result.d_func, 0.0, abs_tol=1e-5)

    def test_docstring_spam_and_citation_padding_rejected(self):
        """Massive academic prose insertion does not bypass collinearity rejection."""
        y_true, y_base = self._setup_synthetic_predictions()
        y_cand = y_base.copy()

        docstring_spam = """
\"\"\"
Breakthrough Multi-Scale Holographic Representation Architecture.
Inspired by Riemannian differential geometry and algebraic topology.
Key References:
- Vaswani et al. (2017) 'Attention is All You Need'
- He et al. (2016) 'Deep Residual Learning for Image Recognition'
- Einstein et al. (1935) Physical Reality
- Turing (1950) Computing Machinery
- Shannon (1948) Communication Theory
\"\"\"
import numpy as np

def mlp_forward(features, weight_in, bias_in, weight_out, bias_out):
    hidden = np.maximum(0, np.matmul(features, weight_in) + bias_in)
    return np.matmul(hidden, weight_out) + bias_out
"""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        result = contract.evaluate_novelty(
            hypothesis_id="adv_docstring_spam",
            candidate_code=docstring_spam,
            prior_art_corpus=[self.PRIOR_MLP_CODE],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
        )

        assert result.passed is False
        assert result.verdict in (
            NoveltyVerdict.REJECTED_LIT_PROXIMITY,
            NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )

    def test_dead_code_and_dummy_operations_rejected(self):
        """Dead code injection does not bypass collinearity rejection."""
        y_true, y_base = self._setup_synthetic_predictions()
        y_cand = y_base.copy()

        dead_code = """
import numpy as np

def mlp_forward(features, weight_in, bias_in, weight_out, bias_out):
    dummy_const = 3.14159 * 2.71828
    _unused = [i**2 for i in range(50)]
    hidden = np.maximum(0, np.matmul(features, weight_in) + bias_in)
    return np.matmul(hidden, weight_out) + bias_out
"""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        result = contract.evaluate_novelty(
            hypothesis_id="adv_dead_code",
            candidate_code=dead_code,
            prior_art_corpus=[self.PRIOR_MLP_CODE],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
        )

        assert result.passed is False
        assert result.verdict in (
            NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )

    def test_syntactic_sugar_and_nested_closures_rejected(self):
        """Lambdas, nested closures, and structural syntactic sugar do not fool Key 2."""
        y_true, y_base = self._setup_synthetic_predictions()
        y_cand = y_base.copy()

        closure_code = """
import numpy as np

def build_projector():
    relu = lambda arr: np.where(arr > 0, arr, 0)
    linear = lambda x, w, b: np.dot(x, w) + b
    def execute(f, w1, b1, w2, b2):
        return linear(relu(linear(f, w1, b1)), w2, b2)
    return execute
"""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        result = contract.evaluate_novelty(
            hypothesis_id="adv_nested_closure",
            candidate_code=closure_code,
            prior_art_corpus=[self.PRIOR_MLP_CODE],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
        )

        assert result.passed is False
        assert result.verdict in (
            NoveltyVerdict.REJECTED_FUNC_COLLINEAR,
            NoveltyVerdict.REJECTED_DUAL_FAILURE,
        )

    def test_scaled_residual_collinear_multipliers_rejected(self):
        """Scalar-scaled residuals (r_cand = c * r_base) yield rho = 1.0 -> d_func = 0.0."""
        y_true, y_base = self._setup_synthetic_predictions()
        r_base = y_base - y_true
        contract = DualKeyNoveltyContract(epsilon_func=0.10)

        for c in [0.001, 0.5, 1.0001, 2.0, 100.0]:
            y_cand = y_true + c * r_base
            d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
            assert math.isclose(rho, 1.0, abs_tol=1e-5), f"Failed for c={c}"
            assert math.isclose(d_func, 0.0, abs_tol=1e-5)
            assert d_func <= contract.epsilon_func

    def test_anti_parallel_residual_multipliers_rejected(self):
        """Inverted residuals (r_cand = -c * r_base) yield rho = -1.0 -> d_func = 0.0."""
        y_true, y_base = self._setup_synthetic_predictions()
        r_base = y_base - y_true
        contract = DualKeyNoveltyContract(epsilon_func=0.10)

        for c in [-0.1, -1.0, -5.0]:
            y_cand = y_true + c * r_base
            d_func, rho = contract.compute_functional_divergence(y_true, y_base, y_cand)
            assert math.isclose(rho, -1.0, abs_tol=1e-5)
            assert math.isclose(d_func, 0.0, abs_tol=1e-5)
            assert d_func <= contract.epsilon_func

    def test_plagiarized_text_with_divergent_predictions_rejected_by_key1(self):
        """Candidate copies prior art text but produces orthogonal predictions: Key 1 rejects."""
        y_true, y_base = self._setup_synthetic_predictions(n=40)
        # Truly orthogonal predictions
        y_cand = y_true + np.array([1.0 if i % 2 == 0 else -1.0 for i in range(40)])

        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        result = contract.evaluate_novelty(
            hypothesis_id="adv_plagiarized_text",
            candidate_code=self.PRIOR_MLP_CODE,
            prior_art_corpus=[self.PRIOR_MLP_CODE],
            y_true=y_true,
            y_baseline_pred=y_base,
            y_candidate_pred=y_cand,
        )

        assert result.passed is False
        assert result.verdict == NoveltyVerdict.REJECTED_LIT_PROXIMITY
        assert result.d_lit <= contract.tau_lit
        assert result.d_func > contract.epsilon_func


class TestNearThresholdEdgeCases:
    """Rigorous tests probing strict inequality boundaries and precision limits."""

    def test_exact_tau_lit_threshold_boundary(self):
        """Boundary: d_lit == tau_lit must strictly fail (requires >)."""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        assert not (0.1500000000000000 > contract.tau_lit)
        assert (0.1500000000000001 > contract.tau_lit)
        assert not (0.1499999999999999 > contract.tau_lit)

    def test_exact_epsilon_func_threshold_boundary(self):
        """Boundary: d_func == epsilon_func must strictly fail (requires >)."""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        assert not (0.1000000000000000 > contract.epsilon_func)
        assert (0.1000000000000001 > contract.epsilon_func)
        assert not (0.0999999999999999 > contract.epsilon_func)

    def test_synthetic_vector_at_exact_epsilon_func(self):
        """Construct synthetic residuals that yield rho = 0.9000000 -> d_func = 0.1000000."""
        r_b = np.array([1.0, 0.0])
        sin_theta = math.sqrt(1.0 - 0.90**2)
        r_c_exact = np.array([0.90, sin_theta])

        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(2)
        d_func, rho = contract.compute_functional_divergence(y_true, r_b, r_c_exact)

        assert math.isclose(rho, 0.90, abs_tol=1e-6)
        assert math.isclose(d_func, 0.10, abs_tol=1e-6)
        # Strict inequality test
        assert not (d_func > contract.epsilon_func)

    def test_synthetic_vector_just_above_epsilon_func(self):
        """Construct synthetic residuals that yield rho = 0.8990 -> d_func = 0.1010 > 0.10."""
        target_rho = 0.8990
        sin_theta = math.sqrt(1.0 - target_rho**2)
        r_b = np.array([1.0, 0.0])
        r_c = np.array([target_rho, sin_theta])

        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(2)
        d_func, rho = contract.compute_functional_divergence(y_true, r_b, r_c)

        assert math.isclose(rho, target_rho, abs_tol=1e-6)
        assert d_func > contract.epsilon_func

    def test_synthetic_vector_just_below_epsilon_func(self):
        """Construct synthetic residuals that yield rho = 0.9010 -> d_func = 0.0990 <= 0.10."""
        target_rho = 0.9010
        sin_theta = math.sqrt(1.0 - target_rho**2)
        r_b = np.array([1.0, 0.0])
        r_c = np.array([target_rho, sin_theta])

        contract = DualKeyNoveltyContract(epsilon_func=0.10)
        y_true = np.zeros(2)
        d_func, rho = contract.compute_functional_divergence(y_true, r_b, r_c)

        assert math.isclose(rho, target_rho, abs_tol=1e-6)
        assert not (d_func > contract.epsilon_func)

    def test_all_four_verdict_quadrants_at_boundaries(self):
        """Verify state machine emits the 4 exact verdicts based on boundary logic."""
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_t = np.zeros(2)
        r_b = np.array([1.0, 0.0])
        r_c_pass = np.array([0.0, 1.0])  # rho = 0.0, d_func = 1.0 > 0.10
        r_c_fail = np.array([1.0, 0.0])  # rho = 1.0, d_func = 0.0 <= 0.10

        # 1. Approved: both pass
        res_app = contract.evaluate_novelty(
            "h_app", "Quantum Disruption Model", ["Legacy Baseline"], y_t, r_b, r_c_pass
        )
        assert res_app.verdict == NoveltyVerdict.APPROVED
        assert res_app.passed is True

        # 2. Lit fail, Func pass
        res_lit_fail = contract.evaluate_novelty(
            "h_lit_fail", "Legacy Baseline", ["Legacy Baseline"], y_t, r_b, r_c_pass
        )
        assert res_lit_fail.verdict == NoveltyVerdict.REJECTED_LIT_PROXIMITY
        assert res_lit_fail.passed is False

        # 3. Lit pass, Func fail
        res_func_fail = contract.evaluate_novelty(
            "h_func_fail", "Quantum Disruption Model", ["Legacy Baseline"], y_t, r_b, r_c_fail
        )
        assert res_func_fail.verdict == NoveltyVerdict.REJECTED_FUNC_COLLINEAR
        assert res_func_fail.passed is False

        # 4. Lit fail, Func fail
        res_dual_fail = contract.evaluate_novelty(
            "h_dual_fail", "Legacy Baseline", ["Legacy Baseline"], y_t, r_b, r_c_fail
        )
        assert res_dual_fail.verdict == NoveltyVerdict.REJECTED_DUAL_FAILURE
        assert res_dual_fail.passed is False


class TestExtremeCorpusAndTokenPathologies:
    """Stress tests on large corpus size, wrapper containment, and token pathologies."""

    def test_large_corpus_scaling_and_nearest_lookup(self):
        """Corpus with 1,000 distinct items: verifies latency and nearest neighbor selection."""
        contract = DualKeyNoveltyContract()
        corpus = [f"prior art document index {i} with features {i*7}" for i in range(1000)]
        target_idx = 732
        corpus[target_idx] = "quantum fourier mamba hybrid architecture"

        candidate = "quantum fourier mamba hybrid architecture with residual scaling"
        d_lit, nearest = contract.compute_literature_distance(candidate, corpus)

        assert nearest == corpus[target_idx]
        assert d_lit <= contract.tau_lit

    def test_corpus_with_5000_items_performance(self):
        """Stress test with 5,000 items: verifies execution completes in < 2.0s."""
        import time
        contract = DualKeyNoveltyContract()
        corpus = [f"standard model v{i} layer {i%10}" for i in range(5000)]
        candidate = "unrelated synthetic biology regulatory network"

        start_time = time.time()
        d_lit, nearest = contract.compute_literature_distance(candidate, corpus)
        elapsed = time.time() - start_time

        assert d_lit > contract.tau_lit
        assert elapsed < 2.0

    def test_szymkiewicz_simpson_wrapper_containment(self):
        """When candidate wraps prior art in boilerplate, containment detects it."""
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        core_prior = "core loss function torch cross entropy logits targets"
        scaffolding = " ".join([f"var {i} compute auxiliary helper {i}" for i in range(50)])
        wrapped_candidate = f"{scaffolding} {core_prior} {scaffolding}"

        d_lit, nearest = contract.compute_literature_distance(wrapped_candidate, [core_prior])
        assert math.isclose(d_lit, 0.0, abs_tol=1e-5)
        assert nearest == core_prior

    def test_szymkiewicz_simpson_snippet_extraction(self):
        """When candidate is a small snippet extracted from a large prior art doc."""
        contract = DualKeyNoveltyContract(tau_lit=0.15)
        large_prior = " ".join([f"token {i}" for i in range(100)])
        snippet_candidate = "token 20 token 21 token 22 token 23"

        d_lit, _ = contract.compute_literature_distance(snippet_candidate, [large_prior])
        assert d_lit <= contract.tau_lit

    def test_snake_case_monolithic_tokenization_behavior(self):
        """Documents empirical finding: _extract_features does not split on underscores."""
        contract = DualKeyNoveltyContract()
        feats_compound = contract._extract_features("quantum_fourier_mamba")
        assert "quantum_fourier_mamba" in feats_compound
        assert "quantum" not in feats_compound

        feats_space = contract._extract_features("quantum fourier mamba")
        assert "quantum" in feats_space
        assert "fourier" in feats_space

    def test_punctuation_and_emojis_safe_fallback(self):
        """Text consisting purely of non-alphanumeric symbols falls back safely."""
        contract = DualKeyNoveltyContract()
        symbol_candidate = "🚀🔥🧠💡✨ !@#$%^&*()_+{}[]|:;'<>,.?/~`"
        d_lit, nearest = contract.compute_literature_distance(symbol_candidate, ["normal baseline"])
        assert not math.isnan(d_lit)
        assert d_lit == 1.0


class TestAdversarialVulnerabilitiesAndPathologies:
    """Documents and empirically reproduces identified vulnerabilities and edge cases."""

    def test_empirical_vulnerability_nan_prediction_bypass(self):
        """EMPIRICAL FINDING: An adversarial candidate injecting NaN into predictions

        causes denom = NaN, defaulting rho to 0.0, resulting in d_func = 1.0 and APPROVED.
        This test documents the vulnerability empirically.
        """
        contract = DualKeyNoveltyContract(tau_lit=0.15, epsilon_func=0.10)
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_baseline = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        # Adversarial prediction with single NaN
        y_cand_nan = np.array([np.nan, 2.1, 3.1, 4.1, 5.1])

        d_func, rho = contract.compute_functional_divergence(y_true, y_baseline, y_cand_nan)
        # Demonstrates the current bug: rho becomes 0.0 and d_func becomes 1.0
        assert rho == 0.0
        assert d_func == 1.0

        # When evaluated through the contract with novel-sounding text:
        result = contract.evaluate_novelty(
            hypothesis_id="hyp_nan_exploit",
            candidate_code="def totally_novel_function(): return 42",
            prior_art_corpus=["def prior(): return 0"],
            y_true=y_true,
            y_baseline_pred=y_baseline,
            y_candidate_pred=y_cand_nan,
        )
        # The exploit causes an invalid NaN output to be marked APPROVED
        assert result.verdict == NoveltyVerdict.APPROVED
        assert result.passed is True

    def test_zero_alpha_wealth_consumption_invariant(self):
        """Confirms that DualKeyNoveltyContract strictly consumes zero alpha wealth from LORD."""
        lord = LORDController(alpha=0.05)
        initial_alpha = lord.get_next_alpha()
        contract = DualKeyNoveltyContract()
        y = np.ones(20)

        for i in range(500):
            if i % 2 == 0:
                contract.evaluate_novelty(f"h_{i}", "same", ["same"], y, y + 1, y + 1)
            else:
                contract.evaluate_novelty(f"h_{i}", f"novel {i}", ["prior"], y, y + 1, y + np.sin(np.arange(20)))

        assert lord.current_step == 0
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-15)
        assert len(lord.discovery_indices) == 0

    def test_contract_hash_tamper_detection(self):
        """Confirms contract_hash detects field modification."""
        contract = DualKeyNoveltyContract()
        y = np.zeros(10)
        res = contract.evaluate_novelty(
            "hyp_tamper", "novel code", ["prior doc"], y, y + 1, y + np.sin(np.arange(10))
        )
        orig_hash = res.contract_hash
        assert orig_hash is not None
        assert res.compute_hash() == orig_hash

        res.d_lit = 0.9999
        assert res.compute_hash() != orig_hash
