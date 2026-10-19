"""Unit Tests for Tiered Symbolic-Numeric Verification Engine (math_engine.py)."""

import math
import numpy as np
import pytest
import sympy as sp
import torch

from frame.verification.math_engine import (
    Interval,
    compress_swiglu_2d,
    compress_rmsnorm_invariant,
    compress_softmax_pairwise,
    lanczos_spectral_norm,
    verify_l0_analytic,
    verify_l1_symbolic_numeric,
    verify_l2_ibnb,
    TieredMathVerifier,
)


# ==============================================================================
# Level 0 Tests
# ==============================================================================

def test_l0_swiglu_and_rmsnorm_compression():
    """Verifies L0 analytical certification on compressed SwiGLU and RMSNorm primitives."""
    # SwiGLU 2D
    swiglu_expr = compress_swiglu_2d()
    res_swiglu = verify_l0_analytic(swiglu_expr, variables=["u", "v", "beta"])
    assert res_swiglu.passed is True
    assert res_swiglu.is_differentiable is True
    assert len(res_swiglu.poles) == 0
    assert res_swiglu.execution_time_ms < 1000.0

    # RMSNorm 1D invariant energy
    s, rms_expr = compress_rmsnorm_invariant(eps_val=1e-5)
    res_rms = verify_l0_analytic(
        rms_expr,
        variables=["s"],
        domain_assumptions={"s": {"real": True, "nonnegative": True}},
    )
    assert res_rms.passed is True
    assert res_rms.is_differentiable is True
    assert len(res_rms.poles) == 0


def test_l0_singularity_detection():
    """Verifies that L0 catches real poles in unsafe rational expressions."""
    # Unsafe function with pole at x = 2
    unsafe_expr = "1 / (x - 2)"
    res_unsafe = verify_l0_analytic(unsafe_expr, variables=["x"])
    assert res_unsafe.passed is False
    assert res_unsafe.domain_safe is False
    assert any("2" in pole for pole in res_unsafe.poles)


def test_l0_asymptotic_limits():
    """Verifies that L0 correctly computes asymptotic limits."""
    delta_z, softmax_expr = compress_softmax_pairwise()
    res_softmax = verify_l0_analytic(softmax_expr, variables=["delta_z"])
    assert res_softmax.passed is True
    assert "lim_delta_z->+inf" in res_softmax.limits
    assert res_softmax.limits["lim_delta_z->+inf"] == "1"
    assert res_softmax.limits["lim_delta_z->-inf"] == "0"


# ==============================================================================
# Level 1 Tests
# ==============================================================================

def test_l1_forward_interval_bound_propagation():
    """Verifies Level 1 IBP correctly distinguishes safe and unsafe denominators."""
    # Safe operation: f(x) = x / (x^2 + 2) under x in [-5, 5]
    def safe_forward(intervals):
        x = intervals["x"]
        denom = (x * x) + 2.0
        out = x / denom
        return out, [denom]

    res_safe = verify_l1_symbolic_numeric(safe_forward, input_bound=5.0)
    assert res_safe.passed is True
    assert res_safe.denominator_safe is True
    assert res_safe.execution_time_ms < 1500.0

    # Dangerous operation: f(x) = 1 / (x - 1) under x in [-3, 3] -> denom spans [-4, 2] containing 0
    def unsafe_forward(intervals):
        x = intervals["x"]
        denom = x - 1.0
        out = Interval(1.0, 1.0) / denom
        return out, [denom]

    res_unsafe = verify_l1_symbolic_numeric(unsafe_forward, input_bound=3.0)
    assert res_unsafe.passed is False
    assert res_unsafe.denominator_safe is False


def test_l1_lanczos_spectral_norm():
    """Verifies Lanczos operator norm and condition number calculations."""
    w = np.diag([4.0, 2.0, 0.5])
    spectral_norm, cond = lanczos_spectral_norm(w)
    assert abs(spectral_norm - 4.0) < 1e-4
    assert abs(cond - 8.0) < 1e-3

    # PyTorch tensor input
    w_torch = torch.tensor([[3.0, 0.0], [0.0, 1.5]])
    s_norm_t, cond_t = lanczos_spectral_norm(w_torch)
    assert abs(s_norm_t - 3.0) < 1e-4
    assert abs(cond_t - 2.0) < 1e-3


# ==============================================================================
# Level 2 Tests
# ==============================================================================

def test_l2_ibnb_certification():
    """Verifies Level 2 IBnB solver certifies properties that hold strictly."""
    # Property to certify: f(x) = x^2 + 0.1 >= 0 on [-2, 2]
    # Seeking violation f(x) < 0 (threshold = 0.0)
    def scalar_fn(x):
        return x[0]**2 + 0.1

    def mp_interval_fn(box):
        return box[0]**2 + 0.1

    res = verify_l2_ibnb(
        scalar_fn=scalar_fn,
        mp_interval_fn=mp_interval_fn,
        bounds=[(-2.0, 2.0)],
        violation_threshold=0.0,
        seek_violation_greater=False,
    )
    assert res.passed is True
    assert res.certified is True
    assert res.counterexample is None
    assert res.execution_time_ms < 8000.0


def test_l2_ibnb_counterexample_extraction():
    """Verifies Level 2 IBnB solver extracts concrete counterexample when violation occurs."""
    # Property claimed: f(x) >= 0, but f(x) = x^2 - 1.0 which is < 0 on (-1, 1)
    def scalar_fn(x):
        return x[0]**2 - 1.0

    def mp_interval_fn(box):
        return box[0]**2 - 1.0

    res = verify_l2_ibnb(
        scalar_fn=scalar_fn,
        mp_interval_fn=mp_interval_fn,
        bounds=[(-2.0, 2.0)],
        violation_threshold=0.0,
        seek_violation_greater=False,
    )
    assert res.passed is False
    assert res.certified is False
    assert res.counterexample is not None
    # Verify counterexample actually violates condition
    val = scalar_fn(res.counterexample)
    assert val < 0.0
    # Verify counterexample tensor property
    t_cex = res.counterexample_tensor
    assert isinstance(t_cex, torch.Tensor)
    assert t_cex.ndim == 1


def test_l0_log_singularity_and_directional_limits():
    """Verifies that L0 catches logarithmic singularities and computes both directional gradient limits."""
    # 1. log(x - 2) has a singularity at x = 2
    res_log = verify_l0_analytic("log(x - 2)", variables=["x"])
    assert res_log.passed is False
    assert res_log.domain_safe is False
    assert any("2" in pole and "log pole" in pole for pole in res_log.poles)

    # 2. log(x) with positive domain assumption passes
    res_log_pos = verify_l0_analytic(
        "log(x)",
        variables=["x"],
        domain_assumptions={"x": {"real": True, "positive": True}},
    )
    assert res_log_pos.passed is True
    assert len(res_log_pos.poles) == 0

    # 3. Check both lim_diff +inf and -inf on sigmoid/logistic
    delta_z, softmax_expr = compress_softmax_pairwise()
    res_limits = verify_l0_analytic(softmax_expr, variables=["delta_z"])
    assert "lim_diff_delta_z->+inf" in res_limits.limits
    assert "lim_diff_delta_z->-inf" in res_limits.limits
    assert res_limits.limits["lim_diff_delta_z->+inf"] == "0"
    assert res_limits.limits["lim_diff_delta_z->-inf"] == "0"


def test_l1_dynamo_fx_and_ast_dag_ibp():
    """Verifies that Level 1 can trace nn.Module via FX and evaluate Python code strings via AST DAG."""
    import torch.nn as nn

    # 1. Safe PyTorch module: y = x / (x^2 + 2)
    class SafeNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
        def forward(self, x):
            h = self.linear(x)
            return h / (h * h + 2.0)

    safe_net = SafeNet()
    res_fx_safe = verify_l1_symbolic_numeric(safe_net, input_bound=3.0)
    assert res_fx_safe.passed is True
    assert res_fx_safe.denominator_safe is True
    assert res_fx_safe.spectral_norm is not None

    # 2. Unsafe PyTorch module: y = 1.0 / (x - 1.0)
    class UnsafeNet(nn.Module):
        def forward(self, x):
            return 1.0 / (x - 1.0)

    unsafe_net = UnsafeNet()
    res_fx_unsafe = verify_l1_symbolic_numeric(unsafe_net, input_bound=2.0)
    assert res_fx_unsafe.passed is False
    assert res_fx_unsafe.denominator_safe is False

    # 3. Safe Python code string via AST DAG
    code_safe = "y = x / (x * x + 4.0)"
    res_ast_safe = verify_l1_symbolic_numeric(code_safe, input_bound=5.0)
    assert res_ast_safe.passed is True
    assert res_ast_safe.denominator_safe is True

    # 4. Unsafe Python code string via AST DAG
    code_unsafe = "denom = x - 2.0\ny = 1.0 / denom"
    res_ast_unsafe = verify_l1_symbolic_numeric(code_unsafe, input_bound=3.0)
    assert res_ast_unsafe.passed is False
    assert res_ast_unsafe.denominator_safe is False


def test_l2_ibnb_avoids_false_certification_on_unproven_leaves():
    """Verifies that Level 2 IBnB will NOT claim a property is certified if max_depth leaves span the threshold."""
    import mpmath

    def scalar_fn(x):
        # Plateau at 1.0 with a sharp dip to -0.2 in [0.79, 0.81]
        xv = x[0]
        if 0.79 <= xv <= 0.81:
            return -0.2
        return 1.0

    # Loose interval bound that spans the threshold
    def mp_interval_fn(box):
        return mpmath.mpi(-0.3, 1.5)

    res = verify_l2_ibnb(
        scalar_fn=scalar_fn,
        mp_interval_fn=mp_interval_fn,
        bounds=[(-1.0, 1.0)],
        violation_threshold=0.0,
        max_depth=2,
    )
    # Because max_depth=2 was hit without proof of val_low >= 0, it must NOT be certified!
    assert res.certified is False
    assert res.passed is False
    assert res.details.get("uncertified_boxes_exist") is True
