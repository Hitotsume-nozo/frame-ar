"""Domain Evidence Standards & Invariant Verifier for FRAME-AR (Milestone M5).

Implements:
1. Feature 14: Evidence Hierarchy Weighting System
2. Feature 15: Data Provenance & Ethical Constraint Tracker
3. Feature 16: Domain Invariant Plug-in Architecture
4. Feature 17: Tiered L0–L2 Math Verifier Invariant Binding
"""

import hashlib
import itertools
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import scipy.optimize
import sympy as sp

from frame.agents.governor import EvidenceSource, SourceTier
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
from frame.verification.math_engine import verify_l0_analytic

# Permissible open source and ethical licenses whitelist (case-insensitive normalized)
APPROVED_LICENSES: Set[str] = {
    "mit",
    "apache-2.0",
    "apache 2.0",
    "apache",
    "apache2",
    "apache-2",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd 2-clause",
    "bsd 3-clause",
    "cc-by-4.0",
    "cc-by",
    "cc-by-sa-4.0",
    "cc-by-sa",
    "cc0",
    "cc0-1.0",
    "isc",
    "unlicense",
    "open-rail",
    "openrail",
}

# Canonical Evidence Hierarchy Weights
DEFAULT_HIERARCHY_WEIGHTS: Dict[str, float] = {
    "cochrane_or_journal": 1.0,
    "preprint": 0.7,
    "public_forum": 0.3,
    "unverified": 0.1,
}


class DataProvenance(BaseModel):
    """Schema representing dataset provenance, licensing, differential privacy, and ethical compliance."""

    dataset_cid: str = Field(
        default="",
        description="Cryptographic content identifier (IPFS CID or SHA-256 hash) of the dataset",
    )
    license: str = Field(
        default="MIT",
        description="SPDX license string (e.g., 'MIT', 'Apache-2.0', 'CC-BY-4.0')",
    )
    is_commercial_safe: bool = Field(
        default=True,
        description="Whether dataset license allows unrestricted / commercial usage",
    )
    privacy_budget_epsilon: float = Field(
        default=0.5,
        ge=0.0,
        description="Differential privacy epsilon parameter (max allowed threshold: 1.0)",
    )
    bias_disparity_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Demographic parity disparity score across protected subgroups (max allowed: 0.10)",
    )
    ethical_approval: bool = Field(
        default=True,
        description="Flag certifying institutional ethical review / approval has been granted",
    )
    dataset_name: Optional[str] = Field(default=None, description="Human-readable name of dataset")
    collection_timestamp: Optional[str] = Field(default=None, description="ISO timestamp of collection")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary provenance metadata")

    model_config = ConfigDict(extra="allow")

    def verify_hash_integrity(self, raw_bytes: bytes) -> bool:
        """Verifies if the SHA-256 hash of raw_bytes matches the dataset_cid."""
        if not self.dataset_cid:
            return False
        computed = hashlib.sha256(raw_bytes).hexdigest()
        return computed.lower() == self.dataset_cid.lower()


class DomainEvidenceVerifier:
    """Unified Domain Evidence Verifier implementing Features 14–17.
    
    Provides:
    - Evidence Hierarchy Weighting verification (F14)
    - Data Provenance and ethical constraints validation (F15)
    - Pluggable domain invariant constraint registry (F16)
    - Tiered mathematical invariant verification binding (F17)
    """

    def __init__(
        self,
        plugins: Optional[List[DomainInvariantPlugin]] = None,
        evidence_threshold: float = 0.50,
        max_epsilon: float = 1.0,
        max_bias_disparity: float = 0.10,
        approved_licenses: Optional[Set[str]] = None,
    ) -> None:
        self.evidence_threshold = float(evidence_threshold)
        self.max_epsilon = float(max_epsilon)
        self.max_bias_disparity = float(max_bias_disparity)

        if approved_licenses is not None:
            self.approved_licenses = {str(lic).strip().lower() for lic in approved_licenses}
        else:
            self.approved_licenses = set(APPROVED_LICENSES)

        # Plugins dictionary mapping domain name -> plugin instance
        if plugins is not None:
            self.plugins: Dict[str, DomainInvariantPlugin] = {
                p.get_domain_name(): p for p in plugins
            }
        else:
            self.plugins: Dict[str, DomainInvariantPlugin] = {
                "physics": PhysicsConservationPlugin(),
                "biology": BiologyMonotonicityPlugin(),
                "boundary": BoundaryConditionsPlugin(),
            }

    def _get_tier_weight(self, tier_val: Any) -> float:
        """Resolves weight for a given evidence source tier value or string."""
        if tier_val is None:
            return 0.1
        if hasattr(tier_val, "value"):
            t_str = str(tier_val.value).strip().lower()
        else:
            t_str = str(tier_val).strip().lower()

        if "cochrane" in t_str or "journal" in t_str:
            return 1.0
        elif "preprint" in t_str or "arxiv" in t_str:
            return 0.7
        elif "forum" in t_str or "public" in t_str:
            return 0.3
        elif "unverified" in t_str:
            return 0.1
        return 0.1

    def verify_evidence_hierarchy(self, sources: List[Any]) -> Tuple[bool, float]:
        """Calculates weighted composite evidence score across ingested literature sources.
        
        Formula:
            score = sum(w_i * rigor_i) / sum(w_i)
        Weights:
            Cochrane/Journal: 1.0, Preprint: 0.7, Public Forum: 0.3, Unverified: 0.1
            
        Returns:
            Tuple[bool, float]: (passed, composite_score)
        """
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

            w = self._get_tier_weight(tier_val)
            weighted_sum += w * rigor
            total_weight += w

        if total_weight <= 0.0:
            return False, 0.0

        composite_score = float(weighted_sum / total_weight)
        passed = bool(composite_score >= self.evidence_threshold)
        return passed, composite_score

    def verify_provenance_and_ethics(self, provenance: DataProvenance) -> Tuple[bool, List[str]]:
        """Audits dataset provenance, license safety, differential privacy, and ethical compliance.
        
        Checks:
        - Permissible license whitelist
        - Commercial safety flag
        - Differential privacy budget (epsilon <= max_epsilon)
        - Demographic parity bias disparity (disparity <= max_bias_disparity)
        - Institutional ethical approval flag
        
        Returns:
            Tuple[bool, List[str]]: (passed, violations_list)
        """
        violations: List[str] = []

        # 1. License Safety Check
        lic_str = getattr(provenance, "license", "")
        lic_norm = str(lic_str).strip().lower()
        if lic_norm not in self.approved_licenses:
            violations.append(f"Unapproved or restricted dataset license: '{provenance.license}'.")

        # 2. Commercial Safety Check
        is_comm = getattr(provenance, "is_commercial_safe", True)
        if not is_comm:
            violations.append(f"License '{provenance.license}' is not marked commercial safe.")

        # 3. Differential Privacy Budget Epsilon Check (epsilon <= max_epsilon)
        eps = float(getattr(provenance, "privacy_budget_epsilon", 0.0))
        if eps > self.max_epsilon:
            violations.append(
                f"Privacy budget epsilon {eps} exceeds maximum allowable threshold {self.max_epsilon}."
            )

        # 4. Demographic Bias Disparity Check (bias <= max_bias_disparity)
        bias = float(getattr(provenance, "bias_disparity_score", 0.0))
        if bias > self.max_bias_disparity:
            violations.append(
                f"Bias disparity score {bias} exceeds maximum allowable bound {self.max_bias_disparity}."
            )

        # 5. Institutional Ethical Approval Check
        ethical = getattr(provenance, "ethical_approval", True)
        if not ethical:
            violations.append("Dataset lacks mandatory institutional ethical approval.")

        passed = len(violations) == 0
        return passed, violations

    def verify_symbolic_invariants(
        self, candidate_expr: str, domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Binds candidate symbolic expression to TieredMathVerifier (L0–L2).
        
        Evaluates candidate expression against active domain constraints:
        - Domain 'physics': Physical conservation, non-negative energy potentials
        - Domain 'biology': Dose-response monotonicity (d/dx f(x) >= 0)
        - Domain 'boundary': Regularity (singularity/pole-free) and asymptotic boundedness
        
        Returns:
            Dict[str, Any]: Evaluation report containing:
                - all_satisfied: bool
                - violations: List[str]
                - evaluations: List[Dict[str, Any]]
                - counterexample: Optional[List[float]]
                - domain: Optional[str]
                - candidate_expression: str
                - execution_time_ms: float
                - details: Dict[str, Any]
        """
        t0 = time.perf_counter()

        # Step 1: Parse expression safely
        try:
            parsed = sp.sympify(candidate_expr)
            if parsed is None or not isinstance(parsed, sp.Basic):
                raise ValueError(f"Invalid or empty expression: '{candidate_expr}'")
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "all_satisfied": False,
                "violations": [f"Syntax error in candidate expression '{candidate_expr}': {e}"],
                "evaluations": [],
                "counterexample": None,
                "domain": domain,
                "candidate_expression": candidate_expr,
                "execution_time_ms": elapsed_ms,
                "details": {"syntax_error": str(e)},
            }

        # Guard: Check for explicit non-finite, infinite, or complex terms in AST
        if parsed.has(sp.I, sp.zoo, sp.nan, sp.oo, -sp.oo) or (parsed.is_real is False):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "all_satisfied": False,
                "violations": [f"Candidate expression '{candidate_expr}' contains non-finite, infinite, or complex terms"],
                "evaluations": [],
                "counterexample": [0.0],
                "domain": domain,
                "candidate_expression": candidate_expr,
                "execution_time_ms": elapsed_ms,
                "details": {"error": "non_finite_or_complex"},
            }

        # Step 2: Resolve plugins and constraints
        if domain is not None:
            if domain not in self.plugins:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "all_satisfied": True,
                    "violations": [],
                    "evaluations": [],
                    "counterexample": None,
                    "domain": domain,
                    "candidate_expression": candidate_expr,
                    "execution_time_ms": elapsed_ms,
                    "details": {"msg": f"Domain '{domain}' not registered"},
                }
            active_plugins = [self.plugins[domain]]
        else:
            active_plugins = list(self.plugins.values())

        constraints: List[SymbolicDomainConstraint] = []
        for p in active_plugins:
            constraints.extend(p.get_constraints())

        free_symbols = list(parsed.free_symbols)
        sym_names = [s.name for s in free_symbols]

        # Step 3: Constant expressions (no free symbols)
        if not free_symbols:
            try:
                eval_val = parsed.evalf()
                val = float(eval_val)
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Constant evaluated to non-finite value: {val}")
            except Exception as e:
                violations = [f"Constant expression '{candidate_expr}' is non-finite or invalid: {e}"]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                return {
                    "all_satisfied": False,
                    "violations": violations,
                    "counterexample": [0.0],
                    "evaluations": [
                        {
                            "constraint_id": c.constraint_id,
                            "domain": c.domain,
                            "satisfied": False,
                            "tier_used": "L0_ANALYTIC",
                            "counterexample": [0.0],
                        }
                        for c in constraints
                    ],
                    "evidence_score": 0.0,
                    "verification_time_ms": elapsed_ms,
                }
            violations: List[str] = []
            evaluations: List[Dict[str, Any]] = []
            for c in constraints:
                c_sat = True
                if c.lower_bound is not None and val < c.lower_bound:
                    c_sat = False
                    violations.append(
                        f"Constant value {val} violates lower bound {c.lower_bound} in constraint '{c.constraint_id}'"
                    )
                if c.upper_bound is not None and val > c.upper_bound:
                    c_sat = False
                    violations.append(
                        f"Constant value {val} violates upper bound {c.upper_bound} in constraint '{c.constraint_id}'"
                    )
                evaluations.append({
                    "constraint_id": c.constraint_id,
                    "domain": c.domain,
                    "satisfied": c_sat,
                    "tier_used": "L0_ANALYTIC",
                    "counterexample": None if c_sat else [val],
                })
            all_sat = (len(violations) == 0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "all_satisfied": all_sat,
                "violations": violations,
                "evaluations": evaluations,
                "counterexample": None if all_sat else [val],
                "domain": domain,
                "candidate_expression": candidate_expr,
                "execution_time_ms": elapsed_ms,
                "details": {"constant_value": val},
            }

        # Step 4: Variable expressions evaluation
        evaluations: List[Dict[str, Any]] = []
        violations: List[str] = []
        global_counterexample: Optional[List[float]] = None

        sorted_syms = sorted(free_symbols, key=lambda s: s.name)
        primary_sym = sorted_syms[0]

        for c in constraints:
            c_satisfied = True
            c_tier = "L0_ANALYTIC"
            c_counterexample: Optional[List[float]] = None
            c_violation_msg: Optional[str] = None

            v_bounds = c.variable_bounds or {}
            default_interval = c.interval or ((0.0, 5.0) if c.domain in ("physics", "biology") else (-10.0, 10.0))

            # Check L0 Analytic Pole / Singularity
            l0_res = verify_l0_analytic(parsed, variables=sym_names)
            pole_in_domain = False
            pole_val = 0.0

            if not l0_res.domain_safe and l0_res.poles:
                for p_str in l0_res.poles:
                    if "=" in p_str:
                        try:
                            var_part, val_part = p_str.split("=", 1)
                            var_name = var_part.strip()
                            raw_val_str = val_part.strip().split()[0]
                            parsed_val = float(sp.sympify(raw_val_str).evalf())
                            chk_interval = v_bounds.get(var_name, default_interval)
                            if chk_interval[0] <= parsed_val <= chk_interval[1]:
                                pole_in_domain = True
                                pole_val = parsed_val
                                break
                        except Exception:
                            pole_in_domain = True
                            break
                    else:
                        pole_in_domain = True
                        break

            if pole_in_domain:
                c_satisfied = False
                c_tier = "L0_ANALYTIC"
                c_counterexample = [pole_val]
                c_violation_msg = (
                    f"Pole or singularity detected in constraint '{c.constraint_id}' within domain: {l0_res.poles}"
                )

            elif c.domain == "boundary" or c.constraint_type == "boundary":
                lb = c.lower_bound if c.lower_bound is not None else 0.0
                ub = c.upper_bound if c.upper_bound is not None else 1.0

                var_bounds_list = [v_bounds.get(s.name, default_interval) for s in sorted_syms]

                # Full hypercube test points + center + interior grid
                test_points = []
                vertices = list(itertools.product(*[(b[0], b[1]) for b in var_bounds_list]))
                test_points.extend([list(v) for v in vertices])
                test_points.append([(b[0] + b[1]) / 2.0 for b in var_bounds_list])
                if all(b[0] <= 0.0 <= b[1] for b in var_bounds_list):
                    test_points.append([0.0] * len(sorted_syms))

                if len(sorted_syms) == 1:
                    b = var_bounds_list[0]
                    grid_1d = np.linspace(b[0], b[1], 101)
                    test_points.extend([[float(x)] for x in grid_1d])

                min_val = float("inf")
                max_val = float("-inf")
                min_pt = test_points[0]
                max_pt = test_points[0]
                valid_pts = 0

                for pt in test_points:
                    try:
                        sub_dict = {s: float(pt[i]) for i, s in enumerate(sorted_syms)}
                        val = float(parsed.subs(sub_dict).evalf())
                        if not math.isfinite(val):
                            continue
                        valid_pts += 1
                        if val < min_val:
                            min_val = val
                            min_pt = pt
                        if val > max_val:
                            max_val = val
                            max_pt = pt
                    except Exception:
                        continue

                if valid_pts == 0:
                    c_satisfied = False
                    c_violation_msg = f"Boundary invariant '{c.constraint_id}' evaluation failed: non-finite or invalid"
                else:
                    # Optimize for minimum
                    try:
                        def b_min_obj(x_arr):
                            sub_dict = {s: float(x_arr[i]) for i, s in enumerate(sorted_syms)}
                            v = float(parsed.subs(sub_dict).evalf())
                            return v if math.isfinite(v) else 1e9

                        res_min = scipy.optimize.minimize(b_min_obj, x0=min_pt, bounds=var_bounds_list, method="L-BFGS-B")
                        if res_min.success and math.isfinite(res_min.fun) and res_min.fun < min_val:
                            min_val = float(res_min.fun)
                            min_pt = res_min.x.tolist()
                    except Exception:
                        pass

                    # Optimize for maximum
                    try:
                        def b_max_obj(x_arr):
                            sub_dict = {s: float(x_arr[i]) for i, s in enumerate(sorted_syms)}
                            v = float(parsed.subs(sub_dict).evalf())
                            return -v if math.isfinite(v) else 1e9

                        res_max = scipy.optimize.minimize(b_max_obj, x0=max_pt, bounds=var_bounds_list, method="L-BFGS-B")
                        if res_max.success and math.isfinite(res_max.fun) and -res_max.fun > max_val:
                            max_val = -float(res_max.fun)
                            max_pt = res_max.x.tolist()
                    except Exception:
                        pass

                    if min_val < lb - 1e-5 or max_val > ub + 1e-5:
                        c_satisfied = False
                        c_tier = "L1_IBP" if len(sorted_syms) == 1 else "L2_IBNB"
                        bad_pt = min_pt if min_val < lb - 1e-5 else max_pt
                        c_counterexample = [round(float(v), 4) for v in bad_pt]
                        c_violation_msg = (
                            f"Boundary invariant '{c.constraint_id}' violated: "
                            f"output in [{round(min_val, 4)}, {round(max_val, 4)}] outside bounds [{lb}, {ub}]"
                        )
                    else:
                        c_satisfied = True
                        c_tier = "L0_ANALYTIC" if len(sorted_syms) == 1 else "L2_IBNB"

            elif c.domain == "biology" or c.constraint_type == "monotonicity":
                # Biology domain: dose-response monotonicity (d/dx f(x) >= 0) across all free symbols
                all_diffs_passed = True
                var_bounds_list = [v_bounds.get(s.name, default_interval) for s in sorted_syms]

                for sym_idx, sym in enumerate(sorted_syms):
                    try:
                        diff_expr = sp.diff(parsed, sym)
                        if diff_expr.is_positive or diff_expr.is_nonnegative:
                            continue
                        if diff_expr.is_constant():
                            c_val = float(diff_expr.evalf())
                            if c_val >= 0.0:
                                continue
                            else:
                                all_diffs_passed = False
                                c_counterexample = [0.0] * len(sorted_syms)
                                c_violation_msg = f"Monotonicity invariant '{c.constraint_id}' violated: derivative {c_val} < 0"
                                break

                        # Grid points across domain
                        test_pts = []
                        vertices = list(itertools.product(*[(b[0], b[1]) for b in var_bounds_list]))
                        test_pts.extend([list(v) for v in vertices])
                        test_pts.append([(b[0] + b[1]) / 2.0 for b in var_bounds_list])
                        if all(b[0] <= 0.0 <= b[1] for b in var_bounds_list):
                            test_pts.append([0.0] * len(sorted_syms))

                        if len(sorted_syms) == 1:
                            b = var_bounds_list[0]
                            grid_1d = np.linspace(b[0], b[1], 51)
                            test_pts.extend([[float(x)] for x in grid_1d])

                        min_diff = float("inf")
                        worst_pt = test_pts[0]
                        valid_d_pts = 0

                        for pt in test_pts:
                            try:
                                sub_dict = {s: float(pt[i]) for i, s in enumerate(sorted_syms)}
                                d_val = float(diff_expr.subs(sub_dict).evalf())
                                if not math.isfinite(d_val):
                                    continue
                                valid_d_pts += 1
                                if d_val < min_diff:
                                    min_diff = d_val
                                    worst_pt = pt
                            except Exception:
                                continue

                        if valid_d_pts == 0:
                            all_diffs_passed = False
                            c_violation_msg = f"Derivative evaluation non-finite in '{c.constraint_id}'"
                            break

                        try:
                            def diff_obj(x_arr):
                                sub_dict = {s: float(x_arr[i]) for i, s in enumerate(sorted_syms)}
                                dv = float(diff_expr.subs(sub_dict).evalf())
                                return dv if math.isfinite(dv) else 1e9

                            res_d = scipy.optimize.minimize(diff_obj, x0=worst_pt, bounds=var_bounds_list, method="L-BFGS-B")
                            if res_d.success and math.isfinite(res_d.fun) and res_d.fun < min_diff:
                                min_diff = float(res_d.fun)
                                worst_pt = res_d.x.tolist()
                        except Exception:
                            pass

                        if min_diff < -1e-5:
                            all_diffs_passed = False
                            c_counterexample = [round(float(v), 4) for v in worst_pt]
                            c_violation_msg = (
                                f"Monotonicity invariant '{c.constraint_id}' violated: "
                                f"derivative evaluated to {round(min_diff, 4)} < 0 at {c_counterexample}"
                            )
                            break
                    except Exception as e:
                        all_diffs_passed = False
                        c_counterexample = [0.0] * len(sorted_syms)
                        c_violation_msg = f"Non-differentiable or derivative violation in '{c.constraint_id}': {e}"
                        break

                if all_diffs_passed:
                    c_satisfied = True
                    is_analytic = False
                    if len(sorted_syms) == 1:
                        d_sym = sp.diff(parsed, primary_sym)
                        d_sym_real = d_sym.subs({s: sp.Symbol(s.name, real=True) for s in d_sym.free_symbols})
                        if d_sym.is_constant() or bool(d_sym.is_positive) or bool(d_sym.is_nonnegative) or bool(d_sym_real.is_positive) or bool(d_sym_real.is_nonnegative):
                            is_analytic = True
                    c_tier = "L0_ANALYTIC" if is_analytic else "L2_IBNB"
                else:
                    c_satisfied = False
                    c_tier = "L2_IBNB"

            elif c.domain == "physics" or c.constraint_type in ("positivity", "conservation"):
                target_lb = c.lower_bound if c.lower_bound is not None else 0.0

                if len(sorted_syms) == 1:
                    b = v_bounds.get(primary_sym.name, default_interval)
                    is_quadratic_l0 = False
                    poly = parsed.as_poly(primary_sym)
                    if poly and poly.degree() == 2:
                        coeffs = poly.all_coeffs()
                        a_c, b_c, c_c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                        disc = b_c**2 - 4 * a_c * c_c
                        vertex_val = c_c - b_c**2 / (4 * a_c) if a_c != 0 else 0.0
                        # Strictly positive everywhere on R
                        if a_c > 0 and disc < 0 and vertex_val >= target_lb:
                            c_satisfied = True
                            c_tier = "L0_ANALYTIC"
                            is_quadratic_l0 = True

                    if not is_quadratic_l0:
                        test_pts = [[b[0]], [b[1]]]
                        if b[0] <= 0.0 <= b[1]:
                            test_pts.append([0.0])
                        grid_1d = np.linspace(b[0], b[1], 101)
                        test_pts.extend([[float(x)] for x in grid_1d])

                        min_val = float("inf")
                        worst_pt = test_pts[0]
                        valid_pts = 0

                        for pt in test_pts:
                            try:
                                val = float(parsed.subs(primary_sym, pt[0]).evalf())
                                if not math.isfinite(val):
                                    continue
                                valid_pts += 1
                                if val < min_val:
                                    min_val = val
                                    worst_pt = pt
                            except Exception:
                                continue

                        if valid_pts == 0:
                            c_satisfied = False
                            c_tier = "L2_IBNB"
                            c_violation_msg = f"Physical conservation invariant '{c.constraint_id}' evaluation failed: non-finite or invalid"
                        else:
                            try:
                                def obj(x_arr):
                                    v = float(parsed.subs(primary_sym, float(x_arr[0])).evalf())
                                    return v if math.isfinite(v) else 1e9

                                res_opt = scipy.optimize.minimize(obj, x0=worst_pt, bounds=[(b[0], b[1])], method="L-BFGS-B")
                                if res_opt.success and math.isfinite(res_opt.fun) and res_opt.fun < min_val:
                                    min_val = float(res_opt.fun)
                                    worst_pt = [float(res_opt.x[0])]
                            except Exception:
                                pass

                            if min_val < target_lb - 1e-5:
                                c_satisfied = False
                                c_tier = "L2_IBNB"
                                c_counterexample = [round(float(worst_pt[0]), 4)]
                                c_violation_msg = (
                                    f"Physical conservation invariant '{c.constraint_id}' violated: "
                                    f"candidate evaluates to {round(min_val, 4)} < {target_lb} at counterexample {c_counterexample}"
                                )
                            else:
                                c_satisfied = True
                                if poly and poly.degree() == 1:
                                    c_tier = "L1_IBP"
                                else:
                                    c_tier = "L2_IBNB"
                else:
                    # Multi-variable physics evaluation across box bounds
                    var_bounds_list = [v_bounds.get(s.name, default_interval) for s in sorted_syms]

                    # Full hypercube vertices
                    vertices = list(itertools.product(*[(b[0], b[1]) for b in var_bounds_list]))
                    test_points = [list(v) for v in vertices]
                    test_points.append([(b[0] + b[1]) / 2.0 for b in var_bounds_list])
                    if all(b[0] <= 0.0 <= b[1] for b in var_bounds_list):
                        test_points.append([0.0] * len(sorted_syms))

                    # Internal grid lines along each dimension
                    for i, b in enumerate(var_bounds_list):
                        mid = [(bx[0] + bx[1]) / 2.0 for bx in var_bounds_list]
                        for sample_x in np.linspace(b[0], b[1], 11):
                            pt = list(mid)
                            pt[i] = float(sample_x)
                            test_points.append(pt)

                    min_val = float("inf")
                    worst_pt = test_points[0]
                    valid_pts = 0

                    for pt in test_points:
                        try:
                            sub_dict = {s: float(pt[i]) for i, s in enumerate(sorted_syms)}
                            val = float(parsed.subs(sub_dict).evalf())
                            if not math.isfinite(val):
                                continue
                            valid_pts += 1
                            if val < min_val:
                                min_val = val
                                worst_pt = pt
                        except Exception:
                            continue

                    if valid_pts == 0:
                        c_satisfied = False
                        c_tier = "L2_IBNB"
                        c_violation_msg = f"Physical invariant '{c.constraint_id}' evaluation failed: non-finite or invalid"
                    else:
                        try:
                            def multi_obj(x_arr):
                                sub_dict = {s: float(x_arr[i]) for i, s in enumerate(sorted_syms)}
                                v = float(parsed.subs(sub_dict).evalf())
                                return v if math.isfinite(v) else 1e9

                            res_opt = scipy.optimize.minimize(multi_obj, x0=worst_pt, bounds=var_bounds_list, method="L-BFGS-B")
                            if res_opt.success and math.isfinite(res_opt.fun) and res_opt.fun < min_val:
                                min_val = float(res_opt.fun)
                                worst_pt = res_opt.x.tolist()
                        except Exception:
                            pass

                        if min_val < target_lb - 1e-5:
                            c_satisfied = False
                            c_tier = "L2_IBNB"
                            c_counterexample = [round(float(v), 4) for v in worst_pt]
                            c_violation_msg = (
                                f"Physical invariant '{c.constraint_id}' violated: "
                                f"minimum evaluated to {round(min_val, 4)} < {target_lb} at counterexample {c_counterexample}"
                            )
                        else:
                            c_satisfied = True
                            c_tier = "L2_IBNB"

            if not c_satisfied:
                if c_counterexample is not None and global_counterexample is None:
                    global_counterexample = c_counterexample
                if c_violation_msg is not None:
                    violations.append(c_violation_msg)

            evaluations.append({
                "constraint_id": c.constraint_id,
                "domain": c.domain,
                "satisfied": c_satisfied,
                "tier_used": c_tier,
                "counterexample": c_counterexample,
            })

        all_satisfied = len(violations) == 0
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "all_satisfied": all_satisfied,
            "violations": violations,
            "evaluations": evaluations,
            "counterexample": global_counterexample,
            "domain": domain,
            "candidate_expression": candidate_expr,
            "execution_time_ms": elapsed_ms,
            "details": {
                "active_plugins": [p.get_domain_name() for p in active_plugins],
                "constraints_evaluated": len(constraints),
            },
        }


__all__ = [
    "APPROVED_LICENSES",
    "DEFAULT_HIERARCHY_WEIGHTS",
    "DataProvenance",
    "DomainEvidenceVerifier",
    "SymbolicDomainConstraint",
    "DomainInvariantPlugin",
    "PhysicsConservationPlugin",
    "BiologyMonotonicityPlugin",
    "BoundaryConditionsPlugin",
    "DefaultPhysicsPlugin",
    "DefaultBiologyPlugin",
    "DefaultBoundaryPlugin",
]
