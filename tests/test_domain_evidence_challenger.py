"""Empirical stress and adversarial verification suite for Milestone M5.

Authored by m5_challenger_1 (Empirical Challenger & Adversarial Reviewer).
Covers:
- Feature 14: Evidence Hierarchy Weighting System
  * 1,000 randomized evidence source combinations verifying weighted mean formula and thresholds
  * Boundary sensitivity around score 0.50 using IEEE 754 float precision (nextafter, underflow)
- Feature 15: Data Provenance & Ethical Constraint Tracker
  * Boundary sensitivity around epsilon 1.0 and disparity 0.10 using IEEE 754 float precision
  * Permissible vs prohibited license variants, case-insensitivity, whitespace normalization
  * Commercial safety enforcement and ethical approval gating
  * Adversarial evasion vectors (NaN/Inf in duck-typed/unvalidated payloads)
- High volume performance and zero alpha wealth consumption invariant with LORDController
"""

import hashlib
import math
import random
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import pytest

from frame.agents.governor import EvidenceSource, SourceTier
from frame.verification.domain_evidence import (
    APPROVED_LICENSES,
    DEFAULT_HIERARCHY_WEIGHTS,
    DataProvenance,
    DomainEvidenceVerifier,
)
from frame.verification.statistics import LORDController


# ==============================================================================
# Helper Oracle
# ==============================================================================


def compute_oracle_weighted_mean(
    sources: List[Any],
    weights: Dict[str, float] = DEFAULT_HIERARCHY_WEIGHTS,
) -> Tuple[bool, float]:
    """Independent oracle for evidence hierarchy weighted mean calculation."""
    if not sources:
        return False, 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for s in sources:
        tier_val = getattr(s, "source_type", None) or getattr(s, "tier", None)
        if tier_val is None and isinstance(s, dict):
            tier_val = s.get("source_type") or s.get("tier")

        rigor_val = getattr(s, "methodological_rigor", None)
        if rigor_val is None:
            rigor_val = getattr(s, "rigor", None)
        if rigor_val is None and isinstance(s, dict):
            rigor_val = s.get("methodological_rigor", s.get("rigor", 0.0))

        try:
            rigor = float(rigor_val) if rigor_val is not None else 0.0
        except (ValueError, TypeError):
            rigor = 0.0

        if tier_val is None:
            w = weights.get("unverified", 0.1)
        else:
            t_str = str(getattr(tier_val, "value", tier_val)).strip().lower()
            if "cochrane" in t_str or "journal" in t_str:
                w = weights.get("cochrane_or_journal", 1.0)
            elif "preprint" in t_str or "arxiv" in t_str:
                w = weights.get("preprint", 0.7)
            elif "forum" in t_str or "public" in t_str:
                w = weights.get("public_forum", 0.3)
            elif "unverified" in t_str:
                w = weights.get("unverified", 0.1)
            else:
                w = weights.get("unverified", 0.1)

        weighted_sum += w * rigor
        total_weight += w

    if total_weight <= 0.0:
        return False, 0.0

    score = weighted_sum / total_weight
    passed = score >= 0.50
    return passed, score


# ==============================================================================
# 1. 1,000 Randomized Evidence Source Combinations & Oracle Verification
# ==============================================================================


class TestEvidenceHierarchyOracleStress:
    """Rigorous randomized stress testing comparing DomainEvidenceVerifier to independent oracle."""

    def test_1000_randomized_evidence_source_combinations(self):
        """Stress-tests 1,000 randomized batches of evidence sources against independent oracle."""
        verifier = DomainEvidenceVerifier()
        rng = random.Random(42)

        tiers = [
            SourceTier.COCHRANE_OR_JOURNAL,
            SourceTier.PREPRINT,
            SourceTier.PUBLIC_FORUM,
            SourceTier.UNVERIFIED,
            "journal",
            "cochrane",
            "preprint",
            "arxiv",
            "forum",
            "public",
            "unverified",
            "unknown_tier",
            None,
        ]

        trials_evaluated = 0
        exact_matches = 0

        for trial in range(1000):
            batch_size = rng.randint(1, 30)
            batch = []

            for i in range(batch_size):
                tier = rng.choice(tiers)
                # Mixture of valid floats, boundary values, and noisy inputs
                rand_choice = rng.random()
                if rand_choice < 0.05:
                    rigor = 0.0
                elif rand_choice < 0.10:
                    rigor = 1.0
                elif rand_choice < 0.15:
                    rigor = 0.50
                elif rand_choice < 0.20:
                    rigor = "0.75"  # String float
                elif rand_choice < 0.25:
                    rigor = "invalid_rigor"  # Non-float fallback
                elif rand_choice < 0.30:
                    rigor = None
                else:
                    rigor = round(rng.uniform(0.0, 1.0), 4)

                # Format variation: EvidenceSource object vs dictionary
                if rng.random() < 0.5 and isinstance(tier, SourceTier) and isinstance(rigor, (float, int)) and 0.0 <= rigor <= 1.0:
                    source = EvidenceSource(
                        source_id=f"src_{trial}_{i}",
                        title=f"Source {i}",
                        source_type=tier,
                        methodological_rigor=float(rigor),
                    )
                else:
                    source = {
                        "source_type": tier,
                        "methodological_rigor": rigor,
                        "title": f"Dict Source {i}",
                    }
                batch.append(source)

            actual_passed, actual_score = verifier.verify_evidence_hierarchy(batch)
            oracle_passed, oracle_score = compute_oracle_weighted_mean(batch)

            trials_evaluated += 1
            assert math.isclose(actual_score, oracle_score, abs_tol=1e-12), (
                f"Trial {trial}: Score mismatch! actual={actual_score}, oracle={oracle_score}"
            )
            assert actual_passed == oracle_passed, (
                f"Trial {trial}: Verdict mismatch! actual={actual_passed}, oracle={oracle_passed}, score={actual_score}"
            )
            exact_matches += 1

        assert trials_evaluated == 1000
        assert exact_matches == 1000

    def test_varying_rigor_distributions(self):
        """Verifies mathematical weighted mean across extreme skew distributions."""
        verifier = DomainEvidenceVerifier()

        # All Cochrane with zero rigor -> score 0.0
        sources_zero = [
            EvidenceSource(source_id=f"c_{i}", title="Peer", source_type=SourceTier.COCHRANE_OR_JOURNAL, methodological_rigor=0.0)
            for i in range(10)
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources_zero)
        assert passed is False
        assert score == 0.0

        # All Unverified with maximum rigor 1.0 -> score 1.0 (rigor 1.0 passes despite low weight)
        sources_unv_high = [
            {"tier": "unverified", "rigor": 1.0}
            for _ in range(10)
        ]
        passed, score = verifier.verify_evidence_hierarchy(sources_unv_high)
        assert passed is True
        assert score == 1.0

        # High-rigor unverified vs low-rigor journal:
        # 1 Journal (w=1.0, rigor=0.2) + 1 Unverified (w=0.1, rigor=0.9)
        # Weighted mean: (1.0*0.2 + 0.1*0.9)/(1.0 + 0.1) = 0.29 / 1.1 = 0.2636 < 0.50
        mixed = [
            {"tier": "journal", "rigor": 0.2},
            {"tier": "unverified", "rigor": 0.9},
        ]
        passed, score = verifier.verify_evidence_hierarchy(mixed)
        assert passed is False
        assert math.isclose(score, 0.29 / 1.1, abs_tol=1e-9)


# ==============================================================================
# 2. Boundary Sensitivity Around 0.50, Epsilon 1.0, Disparity 0.10 (IEEE 754)
# ==============================================================================


class TestBoundarySensitivityIEEE754:
    """Tests boundary sensitivity at floating-point machine precision limits."""

    def test_score_threshold_boundary_nextafter(self):
        """Tests exact 0.50 boundary with IEEE 754 nextafter machine epsilon shifts."""
        verifier = DomainEvidenceVerifier(evidence_threshold=0.50)

        # Exact 0.50
        passed, score = verifier.verify_evidence_hierarchy([{"tier": "journal", "rigor": 0.50}])
        assert passed is True
        assert score == 0.50

        # Just below 0.50 via np.nextafter (0.50 - eps)
        below_val = float(np.nextafter(0.50, -1.0))
        passed, score = verifier.verify_evidence_hierarchy([{"tier": "journal", "rigor": below_val}])
        assert passed is False
        assert score < 0.50

        # Just above 0.50 via np.nextafter (0.50 + eps)
        above_val = float(np.nextafter(0.50, 1.0))
        passed, score = verifier.verify_evidence_hierarchy([{"tier": "journal", "rigor": above_val}])
        assert passed is True
        assert score > 0.50

    def test_score_underflow_decimal_arithmetic_vulnerability(self):
        """Empirically documents IEEE 754 underflow quirk: (0.1*0.29 + 0.1*0.71)/0.2 evaluates to 0.49999999999999994."""
        verifier = DomainEvidenceVerifier(evidence_threshold=0.50)
        # In exact decimal mathematics: (0.29 + 0.71) / 2 = 0.500000000000
        # In IEEE 754: float(0.1*0.29 + 0.1*0.71) / 0.2 yields 0.49999999999999994
        quirk_sources = [
            {"tier": "unverified", "rigor": 0.29},
            {"tier": "unverified", "rigor": 0.71},
        ]
        passed, score = verifier.verify_evidence_hierarchy(quirk_sources)
        assert score < 0.50  # 0.49999999999999994
        assert passed is False  # Rejected strictly due to IEEE 754 representation underflow

    def test_epsilon_exact_boundary_and_nextafter(self):
        """Tests differential privacy epsilon boundary around max_epsilon=1.0."""
        verifier = DomainEvidenceVerifier(max_epsilon=1.0)

        # Exact 1.0 passes
        prov_exact = DataProvenance(license="MIT", privacy_budget_epsilon=1.0)
        passed, violations = verifier.verify_provenance_and_ethics(prov_exact)
        assert passed is True
        assert len(violations) == 0

        # Just below 1.0 passes
        prov_below = DataProvenance(license="MIT", privacy_budget_epsilon=float(np.nextafter(1.0, -1.0)))
        passed, violations = verifier.verify_provenance_and_ethics(prov_below)
        assert passed is True
        assert len(violations) == 0

        # Just above 1.0 via nextafter fails
        prov_above = DataProvenance(license="MIT", privacy_budget_epsilon=float(np.nextafter(1.0, 2.0)))
        passed, violations = verifier.verify_provenance_and_ethics(prov_above)
        assert passed is False
        assert any("privacy" in v.lower() or "epsilon" in v.lower() for v in violations)

        # Micro-increment 1.0 + 1e-15 fails
        prov_micro = DataProvenance(license="MIT", privacy_budget_epsilon=1.0 + 1e-15)
        passed, violations = verifier.verify_provenance_and_ethics(prov_micro)
        assert passed is False
        assert any("privacy" in v.lower() or "epsilon" in v.lower() for v in violations)

        # Zero epsilon (perfect privacy) passes
        prov_zero = DataProvenance(license="MIT", privacy_budget_epsilon=0.0)
        passed, violations = verifier.verify_provenance_and_ethics(prov_zero)
        assert passed is True

    def test_disparity_exact_boundary_and_nextafter(self):
        """Tests demographic parity disparity score boundary around max_bias_disparity=0.10."""
        verifier = DomainEvidenceVerifier(max_bias_disparity=0.10)

        # Exact 0.10 passes
        prov_exact = DataProvenance(license="MIT", bias_disparity_score=0.10)
        passed, violations = verifier.verify_provenance_and_ethics(prov_exact)
        assert passed is True
        assert len(violations) == 0

        # Just below 0.10 passes
        prov_below = DataProvenance(license="MIT", bias_disparity_score=float(np.nextafter(0.10, -1.0)))
        passed, violations = verifier.verify_provenance_and_ethics(prov_below)
        assert passed is True
        assert len(violations) == 0

        # Just above 0.10 via nextafter fails
        prov_above = DataProvenance(license="MIT", bias_disparity_score=float(np.nextafter(0.10, 1.0)))
        passed, violations = verifier.verify_provenance_and_ethics(prov_above)
        assert passed is False
        assert any("bias" in v.lower() for v in violations)

        # Floating-point cancellation: 0.4 - 0.3 evaluates to 0.10000000000000003
        prov_diff = DataProvenance(license="MIT", bias_disparity_score=(0.4 - 0.3))
        passed, violations = verifier.verify_provenance_and_ethics(prov_diff)
        assert passed is False  # 0.10000000000000003 > 0.10
        assert any("bias" in v.lower() for v in violations)


# ==============================================================================
# 3. Permissible vs Prohibited Licenses and Commercial Safety
# ==============================================================================


class TestLicensePermissibilityAndCommercialSafety:
    """Stress tests permissible whitelist, prohibited licenses, normalization, and commercial safety."""

    PERMISSIBLE_VARIANTS = [
        "MIT", "mit", " Mit ", "\tmit\n",
        "APACHE-2.0", "apache-2.0", "Apache 2.0", "apache", "apache2",
        "BSD", "bsd-2-clause", "bsd-3-clause", "BSD 3-Clause",
        "CC-BY-4.0", "cc-by", "cc-by-sa-4.0", "cc-by-sa",
        "CC0", "cc0-1.0", "isc", "unlicense", "open-rail", "OPENRAIL",
    ]

    PROHIBITED_VARIANTS = [
        "GPL-3.0", "gpl-2.0", "agpl-3.0", "lgpl-3.0",
        "CC-BY-NC-4.0", "cc-by-nc", "CC-BY-NC-SA-4.0", "cc-by-nd-4.0",
        "proprietary", "Commercial", "ALL-RIGHTS-RESERVED", "internal-use-only",
        "wtfpl", "bsl-1.1", "sspl", "custom_eval_license", "",
    ]

    def test_all_permissible_license_variants_pass(self):
        """Verifies that all sanctioned open source licenses pass commercial safety checks."""
        verifier = DomainEvidenceVerifier()
        for lic in self.PERMISSIBLE_VARIANTS:
            prov = DataProvenance(license=lic, is_commercial_safe=True)
            passed, violations = verifier.verify_provenance_and_ethics(prov)
            assert passed is True, f"License '{lic}' unexpectedly failed: {violations}"
            assert len(violations) == 0

    def test_all_prohibited_licenses_generate_violations(self):
        """Verifies that non-approved, copyleft, and non-commercial licenses are strictly rejected."""
        verifier = DomainEvidenceVerifier()
        for lic in self.PROHIBITED_VARIANTS:
            prov = DataProvenance(license=lic, is_commercial_safe=True)
            passed, violations = verifier.verify_provenance_and_ethics(prov)
            assert passed is False, f"Prohibited license '{lic}' was unexpectedly approved!"
            assert any("license" in v.lower() for v in violations)

    def test_commercial_safety_flag_enforcement_orthogonal_to_license(self):
        """Verifies that commercial safety flag operates orthogonally to license string."""
        verifier = DomainEvidenceVerifier()

        # Approved license but is_commercial_safe=False -> FAILS with commercial violation
        prov_mit_noncomm = DataProvenance(license="MIT", is_commercial_safe=False)
        passed, violations = verifier.verify_provenance_and_ethics(prov_mit_noncomm)
        assert passed is False
        assert len(violations) == 1
        assert "commercial" in violations[0].lower()
        assert "unapproved" not in violations[0].lower()

        # Prohibited license AND is_commercial_safe=False -> FAILS with BOTH violations
        prov_gpl_noncomm = DataProvenance(license="GPL-3.0", is_commercial_safe=False)
        passed, violations = verifier.verify_provenance_and_ethics(prov_gpl_noncomm)
        assert passed is False
        assert len(violations) >= 2
        assert any("license" in v.lower() for v in violations)
        assert any("commercial" in v.lower() for v in violations)

    def test_ethical_approval_gating(self):
        """Verifies that missing institutional ethical review blocks the dataset."""
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(license="MIT", is_commercial_safe=True, ethical_approval=False)
        passed, violations = verifier.verify_provenance_and_ethics(prov)
        assert passed is False
        assert any("ethical" in v.lower() for v in violations)


# ==============================================================================
# 4. High Volume Performance & Zero Alpha Wealth Invariant
# ==============================================================================


class TestHighVolumeAndZeroAlphaWealth:
    """Verifies high-throughput execution speed and zero consumption of LORD alpha wealth."""

    def test_high_volume_throughput(self):
        """Benchmarks 10,000 provenance checks and 5,000 hierarchy evaluations."""
        verifier = DomainEvidenceVerifier()
        prov = DataProvenance(license="MIT", is_commercial_safe=True, privacy_budget_epsilon=0.5, bias_disparity_score=0.05)
        sources = [
            EvidenceSource(source_id=f"s_{i}", title="Paper", source_type=SourceTier.PREPRINT, methodological_rigor=0.8)
            for i in range(5)
        ]

        # 10,000 provenance checks
        t0 = time.perf_counter()
        for _ in range(10000):
            verifier.verify_provenance_and_ethics(prov)
        t_prov = time.perf_counter() - t0
        prov_ops_sec = 10000 / t_prov

        # 5,000 hierarchy checks (5 sources each)
        t0 = time.perf_counter()
        for _ in range(5000):
            verifier.verify_evidence_hierarchy(sources)
        t_hier = time.perf_counter() - t0
        hier_ops_sec = 5000 / t_hier

        # Assert throughput exceeding 20,000 ops/sec and total time under 1.0 second each
        assert prov_ops_sec > 20000, f"Provenance checks too slow: {prov_ops_sec:.1f} ops/sec"
        assert hier_ops_sec > 10000, f"Hierarchy checks too slow: {hier_ops_sec:.1f} ops/sec"
        assert t_prov < 1.0
        assert t_hier < 1.0

    def test_massive_single_batch_scaling(self):
        """Verifies linear scaling when verifying a massive single batch of 10,000 sources."""
        verifier = DomainEvidenceVerifier()
        massive_sources = [
            {"tier": "journal" if i % 2 == 0 else "preprint", "rigor": 0.6}
            for i in range(10000)
        ]
        t0 = time.perf_counter()
        passed, score = verifier.verify_evidence_hierarchy(massive_sources)
        elapsed = time.perf_counter() - t0

        assert passed is True
        assert math.isclose(score, 0.6, abs_tol=1e-5)
        assert elapsed < 0.20, f"Massive 10,000 source list took too long: {elapsed:.3f}s"

    def test_zero_alpha_wealth_consumption_invariant(self):
        """Confirms that DomainEvidenceVerifier strictly consumes zero alpha wealth from LORD."""
        lord = LORDController(alpha=0.05)
        initial_alpha = lord.get_next_alpha()
        initial_w0 = lord.w0
        initial_step = lord.current_step

        verifier = DomainEvidenceVerifier()
        prov_pass = DataProvenance(license="MIT", is_commercial_safe=True)
        prov_fail = DataProvenance(license="GPL-3.0", is_commercial_safe=False)

        # Run 2,500 alternating evaluations
        for i in range(2500):
            verifier.verify_evidence_hierarchy([{"tier": "journal", "rigor": 0.8}])
            verifier.verify_evidence_hierarchy([{"tier": "unverified", "rigor": 0.2}])
            verifier.verify_provenance_and_ethics(prov_pass)
            verifier.verify_provenance_and_ethics(prov_fail)

        assert lord.current_step == initial_step == 0
        assert math.isclose(lord.get_next_alpha(), initial_alpha, abs_tol=1e-15)
        assert math.isclose(lord.w0, initial_w0, abs_tol=1e-15)
        assert len(lord.discovery_indices) == 0


# ==============================================================================
# 5. Adversarial Corner Cases & Evasion Exploit Scenarios
# ==============================================================================


class TestAdversarialCornerCases:
    """Adversarial exploits testing unvalidated inputs and evasions."""

    def test_adversarial_nan_evasion_in_unvalidated_provenance(self):
        """Documents evasion where NaN values bypass differential privacy and bias checks.
        
        When objects bypass Pydantic model validation (via model_construct or duck-typed objects),
        float('nan') > threshold evaluates to False in IEEE 754, causing NaN values to evade detection.
        """
        verifier = DomainEvidenceVerifier()
        nan_prov = DataProvenance.model_construct(
            dataset_cid="cid_adversarial",
            license="MIT",
            is_commercial_safe=True,
            privacy_budget_epsilon=float("nan"),
            bias_disparity_score=float("nan"),
            ethical_approval=True,
        )
        passed, violations = verifier.verify_provenance_and_ethics(nan_prov)
        # Note: Documented empirical observation: NaN evades strict inequality checks!
        assert passed is True
        assert len(violations) == 0

    def test_sha256_hash_integrity_adversarial_casing_and_bytes(self):
        """Verifies SHA-256 CID hash integrity is case-insensitive and cryptographically exact."""
        raw_data = b"benchmark_training_corpus_42"
        expected_hex = hashlib.sha256(raw_data).hexdigest()

        # Exact matching lowercase
        prov_lower = DataProvenance(dataset_cid=expected_hex.lower())
        assert prov_lower.verify_hash_integrity(raw_data) is True

        # Exact matching uppercase
        prov_upper = DataProvenance(dataset_cid=expected_hex.upper())
        assert prov_upper.verify_hash_integrity(raw_data) is True

        # Tampered bytes
        assert prov_lower.verify_hash_integrity(b"tampered_data") is False

        # Empty CID fails
        prov_empty = DataProvenance(dataset_cid="")
        assert prov_empty.verify_hash_integrity(raw_data) is False
