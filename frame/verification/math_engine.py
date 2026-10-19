"""Tiered Symbolic-Numeric Verification Engine for FRAME-AR.

Implements three levels of mathematical verification:
1. Level 0: Fast Analytical SymPy Path (<200 ms) for poles, limits, and domain bounds.
2. Level 1: Forward Interval Bound Propagation (IBP) and Lanczos Spectral Norms (<1.5 s).
3. Level 2: Pure-Python Recursive Interval Branch-and-Bound (IBnB) using mpmath & scipy (<8 s).
"""

import ast
from dataclasses import dataclass, field
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mpmath
import numpy as np
import scipy.optimize
import sympy as sp
import torch
import torch.nn as nn


# ==============================================================================
# 0. Dimensional Compression Mappings
# ==============================================================================

def compress_swiglu_2d() -> sp.Expr:
    """Compresses high-D SwiGLU tensor operation to 2D scalar primitive:
    f(u, v) = u * sigmoid(beta * u) * v
    """
    u, v, beta = sp.symbols("u v beta", real=True)
    sigmoid = 1 / (1 + sp.exp(-beta * u))
    return u * sigmoid * v


def compress_rmsnorm_invariant(eps_val: Optional[float] = None) -> Tuple[sp.Symbol, sp.Expr]:
    """Compresses high-D RMSNorm to 1D invariant energy representation:
    s = mean(x^2) in [0, inf)
    g(s) = 1 / sqrt(s + eps)
    Pole at s = -eps is strictly outside the positive domain [0, inf).
    """
    s = sp.Symbol("s", real=True, nonnegative=True)
    if eps_val is not None:
        eps = sp.Float(eps_val)
    else:
        eps = sp.Symbol("eps", positive=True)
    return s, 1 / sp.sqrt(s + eps)


def compress_softmax_pairwise() -> Tuple[sp.Symbol, sp.Expr]:
    """Compresses high-D softmax to pairwise 1D projection:
    p(delta_z) = 1 / (1 + exp(-delta_z))
    """
    delta_z = sp.Symbol("delta_z", real=True)
    return delta_z, 1 / (1 + sp.exp(-delta_z))


# ==============================================================================
# 1. Level 0: Fast Analytical SymPy Path (<200 ms)
# ==============================================================================

@dataclass
class L0VerificationResult:
    passed: bool
    poles: List[str]
    is_differentiable: bool
    limits: Dict[str, str]
    domain_safe: bool
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


def verify_l0_analytic(
    expr: Union[sp.Expr, str],
    variables: Optional[List[str]] = None,
    domain_assumptions: Optional[Dict[str, Dict[str, bool]]] = None,
    check_limits: bool = True,
) -> L0VerificationResult:
    """Level 0: Evaluates analytical singularities, pole sets, and asymptotic growth.
    
    Target execution: <200 ms.
    """
    t0 = time.perf_counter()
    domain_assumptions = domain_assumptions or {}
    
    # Construct symbols with assumptions
    sym_dict = {}
    if variables:
        for v in variables:
            assumptions = domain_assumptions.get(v, {"real": True})
            sym_dict[v] = sp.Symbol(v, **assumptions)
            
    if isinstance(expr, str):
        parsed_expr = sp.sympify(expr, locals=sym_dict)
    else:
        parsed_expr = expr

    free_symbols = list(parsed_expr.free_symbols)
    poles_found: List[str] = []
    domain_safe = True
    
    # 1. Pole and Singularity Certification (denominator zeros)
    numer, denom = sp.fraction(sp.together(parsed_expr))
    if denom != 1:
        for s in free_symbols:
            try:
                # Solve denom == 0 under domain constraints
                roots = sp.solve(denom, s)
                for r in roots:
                    # Check if root violates the assumed domain
                    is_real = r.is_real if hasattr(r, "is_real") else None
                    if is_real is not False:
                        s_assumptions = domain_assumptions.get(s.name, {})
                        if s_assumptions.get("nonnegative", False):
                            if r.is_real and r < 0:
                                continue  # Root is negative, so outside [0, inf)
                        if s_assumptions.get("positive", False):
                            if r.is_real and r <= 0:
                                continue  # Root is <= 0, so outside (0, inf)
                        poles_found.append(f"{s.name} = {r}")
            except Exception:
                pass

    # 1b. Logarithmic Singularity Certification (arg_log <= 0)
    for log_atom in parsed_expr.atoms(sp.log):
        log_arg = log_atom.args[0]
        for s in free_symbols:
            try:
                log_roots = sp.solve(log_arg, s)
                for r in log_roots:
                    is_real = r.is_real if hasattr(r, "is_real") else None
                    if is_real is not False:
                        s_assumptions = domain_assumptions.get(s.name, {})
                        if s_assumptions.get("nonnegative", False):
                            if r.is_real and r < 0:
                                continue
                        if s_assumptions.get("positive", False):
                            if r.is_real and r <= 0:
                                continue
                        poles_found.append(f"{s.name} = {r} (log pole)")
            except Exception:
                pass
                
    if poles_found:
        domain_safe = False

    # 2. Differentiability Check
    is_differentiable = True
    try:
        for s in free_symbols:
            diff_expr = sp.diff(parsed_expr, s)
            if diff_expr.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                is_differentiable = False
    except Exception:
        is_differentiable = False

    # 3. Asymptotic Growth & Gradient Limits
    limits: Dict[str, str] = {}
    primary_sym = None
    if variables:
        for v in variables:
            for s in free_symbols:
                if s.name == v:
                    primary_sym = s
                    break
            if primary_sym is not None:
                break
    if primary_sym is None and free_symbols:
        primary_sym = sorted(free_symbols, key=lambda s: s.name)[0]

    if check_limits and primary_sym is not None:
        try:
            lim_pos = sp.limit(parsed_expr, primary_sym, sp.oo)
            limits[f"lim_{primary_sym}->+inf"] = str(lim_pos)
        except Exception as e:
            limits[f"lim_{primary_sym}->+inf"] = f"unresolved: {e}"

        try:
            lim_neg = sp.limit(parsed_expr, primary_sym, -sp.oo)
            limits[f"lim_{primary_sym}->-inf"] = str(lim_neg)
        except Exception as e:
            limits[f"lim_{primary_sym}->-inf"] = f"unresolved: {e}"

        try:
            diff_primary = sp.diff(parsed_expr, primary_sym)
            lim_diff_pos = sp.limit(diff_primary, primary_sym, sp.oo)
            limits[f"lim_diff_{primary_sym}->+inf"] = str(lim_diff_pos)
        except Exception as e:
            limits[f"lim_diff_{primary_sym}->+inf"] = f"unresolved: {e}"

        try:
            diff_primary = sp.diff(parsed_expr, primary_sym)
            lim_diff_neg = sp.limit(diff_primary, primary_sym, -sp.oo)
            limits[f"lim_diff_{primary_sym}->-inf"] = str(lim_diff_neg)
        except Exception as e:
            limits[f"lim_diff_{primary_sym}->-inf"] = f"unresolved: {e}"

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    passed = domain_safe and is_differentiable

    return L0VerificationResult(
        passed=passed,
        poles=poles_found,
        is_differentiable=is_differentiable,
        limits=limits,
        domain_safe=domain_safe,
        execution_time_ms=elapsed_ms,
        details={"expression": str(parsed_expr), "denominator": str(denom)},
    )


# ==============================================================================
# 2. Level 1: Symbolic-Numeric Hybrid System (IBP + Lanczos, <1.5 s)
# ==============================================================================

@dataclass
class Interval:
    lower: float
    upper: float

    def __post_init__(self):
        if self.lower > self.upper:
            self.lower, self.upper = self.upper, self.lower

    def __add__(self, other: Union["Interval", float]) -> "Interval":
        if isinstance(other, Interval):
            return Interval(self.lower + other.lower, self.upper + other.upper)
        return Interval(self.lower + other, self.upper + other)

    def __sub__(self, other: Union["Interval", float]) -> "Interval":
        if isinstance(other, Interval):
            return Interval(self.lower - other.upper, self.upper - other.lower)
        return Interval(self.lower - other, self.upper - other)

    def sqr(self) -> "Interval":
        if self.contains_zero():
            return Interval(0.0, max(self.lower**2, self.upper**2))
        return Interval(min(self.lower**2, self.upper**2), max(self.lower**2, self.upper**2))

    def __pow__(self, power: Union[int, float]) -> "Interval":
        if power == 2:
            return self.sqr()
        if isinstance(power, int) and power > 0 and power % 2 == 0:
            if self.contains_zero():
                return Interval(0.0, max(self.lower**power, self.upper**power))
            return Interval(min(self.lower**power, self.upper**power), max(self.lower**power, self.upper**power))
        if isinstance(power, (int, float)) and power < 0:
            inv = Interval(1.0, 1.0) / self
            return inv ** (-power)
        return Interval(self.lower**power, self.upper**power)

    def __mul__(self, other: Union["Interval", float]) -> "Interval":
        if other is self:
            return self.sqr()
        if isinstance(other, Interval):
            p = [
                self.lower * other.lower,
                self.lower * other.upper,
                self.upper * other.lower,
                self.upper * other.upper,
            ]
            return Interval(min(p), max(p))
        if other >= 0:
            return Interval(self.lower * other, self.upper * other)
        return Interval(self.upper * other, self.lower * other)

    def __truediv__(self, other: Union["Interval", float]) -> "Interval":
        if isinstance(other, Interval):
            if other.contains_zero():
                raise ZeroDivisionError(f"Denominator interval [{other.lower}, {other.upper}] contains 0")
            inv = Interval(1.0 / other.upper, 1.0 / other.lower)
            return self * inv
        if abs(other) < 1e-15:
            raise ZeroDivisionError("Division by near-zero float")
        return self * (1.0 / other)

    def contains_zero(self) -> bool:
        return self.lower <= 0.0 <= self.upper

    def width(self) -> float:
        return self.upper - self.lower

    def midpoint(self) -> float:
        return 0.5 * (self.lower + self.upper)

    def exp(self) -> "Interval":
        return Interval(math.exp(self.lower), math.exp(self.upper))

    def sqrt(self) -> "Interval":
        if self.upper < 0.0:
            raise ValueError(f"Cannot compute sqrt of strictly negative interval [{self.lower}, {self.upper}]")
        l = max(0.0, self.lower)
        u = max(0.0, self.upper)
        return Interval(math.sqrt(l), math.sqrt(u))

    def log(self) -> "Interval":
        if self.upper <= 0.0:
            raise ValueError(f"Cannot compute log of non-positive interval [{self.lower}, {self.upper}]")
        l = max(1e-15, self.lower)
        return Interval(math.log(l), math.log(self.upper))

    def sigmoid(self) -> "Interval":
        s_low = 1.0 / (1.0 + math.exp(-self.lower))
        s_upp = 1.0 / (1.0 + math.exp(-self.upper))
        return Interval(s_low, s_upp)

    def relu(self) -> "Interval":
        return Interval(max(0.0, self.lower), max(0.0, self.upper))

    def tanh(self) -> "Interval":
        return Interval(math.tanh(self.lower), math.tanh(self.upper))


def lanczos_spectral_norm(
    w: Union[torch.Tensor, np.ndarray],
    max_iters: int = 30,
    tol: float = 1e-6,
) -> Tuple[float, float]:
    """Computes operator norm ||W||_2 and condition number kappa(W) via tridiagonal Lanczos iteration.
    
    Returns:
        Tuple[spectral_norm, condition_number]
    """
    if isinstance(w, torch.Tensor):
        w_mat = w.detach().cpu().to(torch.float64).numpy()
    else:
        w_mat = np.asarray(w, dtype=np.float64)

    if w_mat.ndim != 2:
        raise ValueError(f"Lanczos spectral norm requires a 2D matrix, got shape {w_mat.shape}")

    m, n = w_mat.shape
    if m < n:
        A_mult = lambda v: w_mat @ (w_mat.T @ v)
        dim = m
    else:
        A_mult = lambda v: w_mat.T @ (w_mat @ v)
        dim = n

    k_max = min(dim, max_iters)
    alpha: List[float] = []
    beta: List[float] = []

    rng = np.random.RandomState(42)
    v = rng.randn(dim)
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-15:
        return 0.0, 1.0
    v /= norm_v
    v_prev = np.zeros(dim)
    b_prev = 0.0

    for j in range(k_max):
        p = A_mult(v) - b_prev * v_prev
        a_j = float(v @ p)
        alpha.append(a_j)
        p = p - a_j * v
        b_j = float(np.linalg.norm(p))
        if b_j < tol or j == k_max - 1:
            break
        beta.append(b_j)
        v_prev = v
        b_prev = b_j
        v = p / b_j

    if len(beta) == 0:
        eigvals = np.array(alpha)
    else:
        import scipy.linalg
        eigvals = scipy.linalg.eigvalsh_tridiagonal(alpha, beta)

    max_eig = max(0.0, float(np.max(eigvals)))
    sigma_max = math.sqrt(max_eig)

    pos_eigs = [e for e in eigvals if e > 1e-14]
    min_eig = min(pos_eigs) if pos_eigs else 1e-15
    sigma_min = math.sqrt(min_eig)
    cond = (sigma_max / sigma_min) if sigma_min > 1e-15 else 1.0

    return sigma_max, cond


# ------------------------------------------------------------------------------
# Dynamo / FX and AST DAG Forward IBP Propagator
# ------------------------------------------------------------------------------

class ASTIBPInterpreter(ast.NodeVisitor):
    """Interprets Python AST using Interval arithmetic and tracks denominators."""

    def __init__(self, env: Dict[str, Interval]):
        self.env = dict(env)
        self.denominators: List[Interval] = []

    def evaluate_code(self, code: str) -> Tuple[Interval, List[Interval]]:
        parsed = ast.parse(code)
        last_val = Interval(0.0, 0.0)
        for stmt in parsed.body:
            val = self.visit(stmt)
            if val is not None:
                last_val = val
        return last_val, self.denominators

    def visit_Expr(self, node: ast.Expr) -> Any:
        return self.visit(node.value)

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value is not None:
            return self.visit(node.value)
        return None

    def visit_Assign(self, node: ast.Assign) -> Any:
        val = self.visit(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.env[target.id] = val
        return val

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        last_val = Interval(0.0, 0.0)
        for stmt in node.body:
            v = self.visit(stmt)
            if v is not None:
                last_val = v
        return last_val

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in self.env:
            return self.env[node.id]
        if node.id == "pi":
            return Interval(math.pi, math.pi)
        if node.id == "e":
            return Interval(math.e, math.e)
        return Interval(0.0, 0.0)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            v = float(node.value)
            return Interval(v, v)
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(operand, Interval):
            if isinstance(node.op, ast.USub):
                return Interval(-operand.upper, -operand.lower)
            if isinstance(node.op, ast.UAdd):
                return operand
        return operand

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(left, (int, float)):
            left = Interval(float(left), float(left))
        if isinstance(right, (int, float)):
            right = Interval(float(right), float(right))

        if not isinstance(left, Interval) or not isinstance(right, Interval):
            return Interval(0.0, 0.0)

        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, (ast.Div, ast.FloorDiv)):
            self.denominators.append(right)
            return left / right
        if isinstance(node.op, ast.Pow):
            if right.lower == right.upper:
                return left ** right.lower
            return left ** int(round(right.midpoint()))
        return left

    def visit_Call(self, node: ast.Call) -> Any:
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        args = [self.visit(a) for a in node.args]
        arg0 = args[0] if args and isinstance(args[0], Interval) else Interval(0.0, 0.0)

        if func_name in {"sin"}:
            return Interval(-1.0, 1.0)
        if func_name in {"cos"}:
            return Interval(-1.0, 1.0)
        if func_name in {"exp"}:
            return arg0.exp()
        if func_name in {"sqrt"}:
            return arg0.sqrt()
        if func_name in {"log"}:
            return arg0.log()
        if func_name in {"sigmoid"}:
            return arg0.sigmoid()
        if func_name in {"relu"}:
            return arg0.relu()
        if func_name in {"tanh"}:
            return arg0.tanh()
        return arg0


def trace_and_propagate_fx_ibp(
    model: nn.Module,
    input_bound: float = 5.0,
) -> Tuple[Interval, List[Interval], List[Union[torch.Tensor, np.ndarray]]]:
    """Uses torch.fx to trace nn.Module into a DAG and performs Forward Interval Bound Propagation."""
    import torch.fx
    try:
        traced = torch.fx.symbolic_trace(model)
    except Exception as e:
        raise RuntimeError(f"FX symbolic tracing failed: {e}")

    env: Dict[torch.fx.Node, Interval] = {}
    denominators: List[Interval] = []
    weight_matrices: List[Union[torch.Tensor, np.ndarray]] = []

    for name, p in model.named_parameters():
        if p.ndim == 2:
            weight_matrices.append(p)

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            env[node] = Interval(-input_bound, input_bound)
        elif node.op == "call_module":
            submod = dict(model.named_modules()).get(node.target)
            arg0 = env.get(node.args[0], Interval(-input_bound, input_bound))
            if isinstance(submod, nn.ReLU):
                env[node] = arg0.relu()
            elif isinstance(submod, nn.Sigmoid):
                env[node] = arg0.sigmoid()
            elif isinstance(submod, nn.Tanh):
                env[node] = arg0.tanh()
            elif isinstance(submod, nn.Linear):
                w = submod.weight.detach().cpu().numpy()
                w_norm, _ = lanczos_spectral_norm(w)
                b = float(submod.bias.norm().item()) if submod.bias is not None else 0.0
                rad = max(abs(arg0.lower), abs(arg0.upper)) * w_norm + b
                env[node] = Interval(-rad, rad)
            else:
                env[node] = arg0
        elif node.op in {"call_function", "call_method"}:
            args = [env.get(a, Interval(float(a), float(a)) if isinstance(a, (int, float)) else Interval(0.0, 0.0)) for a in node.args]
            target_str = str(node.target)
            if "add" in target_str or node.target == "+":
                env[node] = args[0] + args[1]
            elif "sub" in target_str or node.target == "-":
                env[node] = args[0] - args[1]
            elif "mul" in target_str or node.target == "*":
                env[node] = args[0] * args[1]
            elif "div" in target_str or "truediv" in target_str or node.target == "/":
                denominators.append(args[1])
                env[node] = args[0] / args[1]
            elif "relu" in target_str:
                env[node] = args[0].relu()
            elif "sigmoid" in target_str:
                env[node] = args[0].sigmoid()
            elif "tanh" in target_str:
                env[node] = args[0].tanh()
            elif "exp" in target_str:
                env[node] = args[0].exp()
            elif "sqrt" in target_str:
                env[node] = args[0].sqrt()
            else:
                env[node] = args[0] if args else Interval(0.0, 0.0)
        elif node.op == "output":
            out_node = node.args[0]
            out_interval = env.get(out_node, Interval(0.0, 0.0))
            return out_interval, denominators, weight_matrices

    return Interval(0.0, 0.0), denominators, weight_matrices


@dataclass
class L1VerificationResult:
    passed: bool
    denominator_safe: bool
    output_interval: Tuple[float, float]
    spectral_norm: Optional[float]
    condition_number: Optional[float]
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)


def verify_l1_symbolic_numeric(
    target: Union[Callable[[Dict[str, Interval]], Tuple[Interval, List[Interval]]], nn.Module, str],
    input_bound: float = 5.0,
    input_vars: Optional[List[str]] = None,
    weight_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
) -> L1VerificationResult:
    """Level 1: Forward Interval Bound Propagation (IBP) and Lanczos spectral analysis.
    
    Supports:
    - Custom forward interval lambda/callable
    - PyTorch nn.Module (traced via Dynamo / FX into DAG)
    - Python code snippet string (parsed into AST DAG)
    """
    t0 = time.perf_counter()
    input_vars = input_vars or ["x"]
    input_intervals = {var: Interval(-input_bound, input_bound) for var in input_vars}
    
    denominator_safe = True
    output_interval = Interval(0.0, 0.0)
    error_msg = None
    extracted_weights: List[Union[torch.Tensor, np.ndarray]] = list(weight_matrices or [])

    try:
        if isinstance(target, nn.Module):
            out_int, denoms, fx_weights = trace_and_propagate_fx_ibp(target, input_bound=input_bound)
            output_interval = out_int
            extracted_weights.extend(fx_weights)
            for d in denoms:
                if d.contains_zero():
                    denominator_safe = False
                    error_msg = f"Potential division by zero: denominator interval [{d.lower:.4f}, {d.upper:.4f}] spans 0"
                    break
        elif isinstance(target, str):
            ast_interp = ASTIBPInterpreter(input_intervals)
            out_int, denoms = ast_interp.evaluate_code(target)
            output_interval = out_int
            for d in denoms:
                if d.contains_zero():
                    denominator_safe = False
                    error_msg = f"Potential division by zero: denominator interval [{d.lower:.4f}, {d.upper:.4f}] spans 0"
                    break
        elif callable(target):
            out_int, denoms = target(input_intervals)
            output_interval = out_int
            for d in denoms:
                if d.contains_zero():
                    denominator_safe = False
                    error_msg = f"Potential division by zero: denominator interval [{d.lower:.4f}, {d.upper:.4f}] spans 0"
                    break
        else:
            raise TypeError(f"Unsupported target type for Level 1 verification: {type(target)}")
    except ZeroDivisionError as e:
        denominator_safe = False
        error_msg = str(e)
    except Exception as e:
        denominator_safe = False
        error_msg = f"IBP propagation failed: {e}"

    max_spectral_norm = None
    max_condition_number = None
    if extracted_weights:
        for w in extracted_weights:
            s_norm, s_cond = lanczos_spectral_norm(w)
            if max_spectral_norm is None or s_norm > max_spectral_norm:
                max_spectral_norm = s_norm
            if max_condition_number is None or s_cond > max_condition_number:
                max_condition_number = s_cond

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    passed = denominator_safe and (max_condition_number is None or max_condition_number < 1e7)

    return L1VerificationResult(
        passed=passed,
        denominator_safe=denominator_safe,
        output_interval=(output_interval.lower, output_interval.upper),
        spectral_norm=max_spectral_norm,
        condition_number=max_condition_number,
        execution_time_ms=elapsed_ms,
        details={"error": error_msg, "input_bound": input_bound},
    )


# ==============================================================================
# 3. Level 2: Pure-Python Recursive Interval Branch-and-Bound (IBnB, <8 s)
# ==============================================================================

@dataclass
class L2VerificationResult:
    passed: bool
    certified: bool
    counterexample: Optional[List[float]]
    subdivisions: int
    execution_time_ms: float
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def counterexample_tensor(self) -> Optional[torch.Tensor]:
        if self.counterexample is not None:
            return torch.tensor(self.counterexample, dtype=torch.float32)
        return None


class IntervalBranchAndBoundSolver:
    """Recursive Interval Branch-and-Bound (IBnB) solver using mpmath interval arithmetic.
    
    Certifies property over a compact hyper-rectangle or finds a counterexample x_fail.
    """

    def __init__(
        self,
        max_depth: int = 14,
        max_subdivisions: int = 2000,
        tol: float = 1e-5,
    ):
        self.max_depth = max_depth
        self.max_subdivisions = max_subdivisions
        self.tol = tol
        self.subdivisions = 0

    def solve(
        self,
        scalar_fn: Callable[[List[float]], float],
        mp_interval_fn: Callable[[List[Any]], Any],
        bounds: List[Tuple[float, float]],
        violation_threshold: float = 0.0,
        seek_violation_greater: bool = False,
    ) -> L2VerificationResult:
        """Solves whether there exists x in bounds such that violation occurs.
        
        If seek_violation_greater is False:
            Violation occurs when scalar_fn(x) < violation_threshold (e.g. f(x) < 0).
        If seek_violation_greater is True:
            Violation occurs when scalar_fn(x) > violation_threshold.
        """
        t0 = time.perf_counter()
        self.subdivisions = 0
        dim = len(bounds)

        initial_box = [mpmath.mpi(b[0], b[1]) for b in bounds]
        stack = [(initial_box, 0)]
        counterexample: Optional[List[float]] = None
        uncertified_boxes_exist = False

        while stack and self.subdivisions < self.max_subdivisions:
            current_box, depth = stack.pop()
            self.subdivisions += 1

            try:
                val_int = mp_interval_fn(current_box)
                val_low = float(val_int.a)
                val_upp = float(val_int.b)
            except Exception:
                midpoint = [float((b.a + b.b) / 2.0) for b in current_box]
                try:
                    f_mid = scalar_fn(midpoint)
                    if (not seek_violation_greater and f_mid < violation_threshold) or \
                       (seek_violation_greater and f_mid > violation_threshold):
                        counterexample = midpoint
                        break
                except Exception:
                    counterexample = midpoint
                    break
                val_low, val_upp = -math.inf, math.inf

            if not seek_violation_greater:
                if val_low >= violation_threshold:
                    continue  # Box provably satisfies property
                if val_upp < violation_threshold:
                    midpoint = [float((b.a + b.b) / 2.0) for b in current_box]
                    counterexample = midpoint
                    break
            else:
                if val_upp <= violation_threshold:
                    continue  # Box provably satisfies property
                if val_low > violation_threshold:
                    midpoint = [float((b.a + b.b) / 2.0) for b in current_box]
                    counterexample = midpoint
                    break

            widths = [float(b.b - b.a) for b in current_box]
            max_w = max(widths)
            if max_w < self.tol or depth >= self.max_depth:
                midpoint = [float((b.a + b.b) / 2.0) for b in current_box]
                box_bounds = [(float(b.a), float(b.b)) for b in current_box]
                
                if not seek_violation_greater:
                    res = scipy.optimize.minimize(
                        scalar_fn,
                        x0=midpoint,
                        bounds=box_bounds,
                        method="L-BFGS-B",
                    )
                    if res.fun < violation_threshold:
                        counterexample = [float(x) for x in res.x]
                        break
                    # If optimizer did not find a violation, but interval bound val_low < threshold,
                    # this leaf cannot be certified safe!
                    if val_low < violation_threshold:
                        uncertified_boxes_exist = True
                else:
                    res = scipy.optimize.minimize(
                        lambda x: -scalar_fn(x),
                        x0=midpoint,
                        bounds=box_bounds,
                        method="L-BFGS-B",
                    )
                    if -res.fun > violation_threshold:
                        counterexample = [float(x) for x in res.x]
                        break
                    if val_upp > violation_threshold:
                        uncertified_boxes_exist = True
                continue

            split_dim = int(np.argmax(widths))
            b_target = current_box[split_dim]
            mid_val = (b_target.a + b_target.b) / 2.0

            box_left = list(current_box)
            box_left[split_dim] = mpmath.mpi(b_target.a, mid_val)

            box_right = list(current_box)
            box_right[split_dim] = mpmath.mpi(mid_val, b_target.b)

            stack.append((box_left, depth + 1))
            stack.append((box_right, depth + 1))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        has_counterexample = counterexample is not None
        budget_exhausted = (self.subdivisions >= self.max_subdivisions)
        certified = (not has_counterexample) and (not budget_exhausted) and (not uncertified_boxes_exist)
        passed = (not has_counterexample) and (not budget_exhausted) and (not uncertified_boxes_exist)

        details: Dict[str, Any] = {
            "violation_threshold": violation_threshold,
            "seek_violation_greater": seek_violation_greater,
            "budget_exhausted": budget_exhausted,
            "uncertified_boxes_exist": uncertified_boxes_exist,
        }
        if counterexample is not None:
            details["counterexample_tensor"] = torch.tensor(counterexample, dtype=torch.float32)

        return L2VerificationResult(
            passed=passed,
            certified=certified,
            counterexample=counterexample,
            subdivisions=self.subdivisions,
            execution_time_ms=elapsed_ms,
            details=details,
        )


def verify_l2_ibnb(
    scalar_fn: Callable[[List[float]], float],
    mp_interval_fn: Callable[[List[Any]], Any],
    bounds: List[Tuple[float, float]],
    violation_threshold: float = 0.0,
    seek_violation_greater: bool = False,
    max_depth: int = 14,
    max_subdivisions: int = 2000,
) -> L2VerificationResult:
    """Convenience wrapper for Level 2 Interval Branch-and-Bound verification."""
    solver = IntervalBranchAndBoundSolver(
        max_depth=max_depth,
        max_subdivisions=max_subdivisions,
    )
    return solver.solve(
        scalar_fn=scalar_fn,
        mp_interval_fn=mp_interval_fn,
        bounds=bounds,
        violation_threshold=violation_threshold,
        seek_violation_greater=seek_violation_greater,
    )


# ==============================================================================
# 4. Unified Tiered Verifier
# ==============================================================================

class TieredMathVerifier:
    """Unified Tiered Symbolic-Numeric Verification Engine."""

    @staticmethod
    def verify_primitive_l0(expr: Union[sp.Expr, str], variables: Optional[List[str]] = None) -> L0VerificationResult:
        return verify_l0_analytic(expr, variables=variables)

    @staticmethod
    def verify_bounds_l1(
        target: Union[Callable[[Dict[str, Interval]], Tuple[Interval, List[Interval]]], nn.Module, str],
        input_bound: float = 5.0,
        input_vars: Optional[List[str]] = None,
        weight_matrices: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
    ) -> L1VerificationResult:
        return verify_l1_symbolic_numeric(
            target=target,
            input_bound=input_bound,
            input_vars=input_vars,
            weight_matrices=weight_matrices,
        )

    @staticmethod
    def verify_ibnb_l2(
        scalar_fn: Callable[[List[float]], float],
        mp_interval_fn: Callable[[List[Any]], Any],
        bounds: List[Tuple[float, float]],
        violation_threshold: float = 0.0,
        seek_violation_greater: bool = False,
    ) -> L2VerificationResult:
        return verify_l2_ibnb(
            scalar_fn=scalar_fn,
            mp_interval_fn=mp_interval_fn,
            bounds=bounds,
            violation_threshold=violation_threshold,
            seek_violation_greater=seek_violation_greater,
        )
