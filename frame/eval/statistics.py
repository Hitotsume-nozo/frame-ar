"""Statistical estimators (Paired ROPE t-test, rliable IQM Bootstrap, LORD-3 FDR, and Sample Size Power)."""

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np
from scipy import stats

from frame.schemas import TargetDirection


def generate_step_seeds(
    secret_key: str,
    benchmark_id: str,
    confirmatory_step: int,
    count: int,
) -> List[int]:
    """Generates fresh, deterministic step-indexed seeds via HMAC-SHA256 for Gate 2 confirmatory tests."""
    seeds: List[int] = []
    idx = 0
    while len(seeds) < count:
        msg = f"{benchmark_id}:step_{confirmatory_step}:idx_{idx}".encode("utf-8")
        digest = hmac.new(secret_key.encode("utf-8"), msg, hashlib.sha256).digest()
        val = (int.from_bytes(digest[:4], "big") % 900_000) + 10_000
        if val not in seeds:
            seeds.append(val)
        idx += 1
    return seeds


def compute_iqm(values: Sequence[float]) -> float:
    """Computes the 25% trimmed Interquartile Mean (Agarwal et al., rliable)."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute IQM on empty sequence.")
    if arr.size < 4:
        return float(np.mean(arr))
    return float(stats.trim_mean(arr, proportiontocut=0.25))


ZETA_1_6: float = 2.28576595


def gamma_lord(step: int) -> float:
    """Computes decaying weight gamma_t = 1 / (zeta(1.6) * t^1.6)."""
    if step <= 0:
        return 0.0
    return 1.0 / (ZETA_1_6 * (step ** 1.6))


class LORDController:
    """Online False Discovery Rate control via LORD 3 (Javanmard & Montanari, 2018)."""

    def __init__(self, alpha: float = 0.05, w0_ratio: float = 0.5) -> None:
        self.alpha = float(alpha)
        self.w0 = float(alpha * w0_ratio)
        self.b0 = float(alpha - self.w0)
        self.current_step: int = 0
        self.discovery_steps: List[int] = []
        self.alpha_history: List[float] = []

    def _gamma(self, m: int) -> float:
        return gamma_lord(m)

    def peek_next_alpha(self) -> float:
        return self.next_alpha()

    def next_alpha(self) -> float:
        t = self.current_step + 1
        tau_1 = self.discovery_steps[0] if self.discovery_steps else None
        term = self.w0 * self._gamma(t)
        if tau_1 is not None:
            term += (self.alpha - self.w0) * self._gamma(t - tau_1)
        for tau_j in self.discovery_steps[1:]:
            term += self.alpha * self._gamma(t - tau_j)
        return float(max(1e-6, min(self.alpha, term)))

    def step(self, p_value: float) -> tuple[float, bool, int]:
        alpha_t = self.next_alpha()
        self.current_step += 1
        is_discovery = bool(p_value <= alpha_t)
        if is_discovery:
            self.discovery_steps.append(self.current_step)
        self.alpha_history.append(alpha_t)
        return alpha_t, is_discovery, self.current_step

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "w0": self.w0,
            "b0": self.b0,
            "current_step": self.current_step,
            "discovery_steps": self.discovery_steps,
            "alpha_history": self.alpha_history,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, default_alpha: float = 0.05) -> "LORDController":
        ctrl = cls(alpha=default_alpha)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            ctrl.alpha = float(data.get("alpha", default_alpha))
            ctrl.w0 = float(data.get("w0", ctrl.w0))
            ctrl.b0 = float(data.get("b0", ctrl.b0))
            ctrl.current_step = int(data.get("current_step", 0))
            ctrl.discovery_steps = [int(x) for x in data.get("discovery_steps", [])]
            ctrl.alpha_history = [float(x) for x in data.get("alpha_history", [])]
        return ctrl


@dataclass
class ConfirmatoryTestResult:
    directional_effect: float
    ci_lower_bound: float
    p_value_rope: float
    allocated_alpha_t: float
    lord_step: int
    is_discovery: bool


def _compute_paired_gains(
    incumbent_scores: Dict[str, float],
    challenger_scores: Dict[str, float],
    direction: TargetDirection,
) -> np.ndarray:
    common_keys = sorted(set(incumbent_scores.keys()) & set(challenger_scores.keys()))
    if len(common_keys) < 3:
        raise ValueError(f"Need at least 3 paired seeds, got {len(common_keys)}.")
    gains: List[float] = []
    for k in common_keys:
        inc = float(incumbent_scores[k])
        chal = float(challenger_scores[k])
        gain = (inc - chal) if direction == TargetDirection.MINIMIZE else (chal - inc)
        gains.append(gain)
    return np.asarray(gains, dtype=np.float64)


def evaluate_paired_t_rope(
    incumbent_scores: Dict[str, float],
    challenger_scores: Dict[str, float],
    direction: TargetDirection,
    rope_epsilon: float,
    lord_controller: LORDController,
) -> ConfirmatoryTestResult:
    """One-sided paired Student's t-test against H0: mu_gain <= rope_epsilon."""
    gains = _compute_paired_gains(incumbent_scores, challenger_scores, direction)
    k = gains.size
    mean_gain = float(np.mean(gains))
    sd = float(np.std(gains, ddof=1))

    if sd < 1e-12:
        p_val = 0.0 if mean_gain > rope_epsilon else 1.0
        alpha_t, is_disc, step_idx = lord_controller.step(p_val)
        return ConfirmatoryTestResult(
            directional_effect=mean_gain,
            ci_lower_bound=mean_gain,
            p_value_rope=p_val,
            allocated_alpha_t=alpha_t,
            lord_step=step_idx,
            is_discovery=is_disc,
        )

    se = sd / math.sqrt(k)
    t_stat = (mean_gain - rope_epsilon) / se
    p_val = float(stats.t.sf(t_stat, df=k - 1))
    alpha_t, is_disc, step_idx = lord_controller.step(p_val)
    t_crit = float(stats.t.ppf(1.0 - alpha_t, df=k - 1))
    ci_lower = mean_gain - t_crit * se

    return ConfirmatoryTestResult(
        directional_effect=mean_gain,
        ci_lower_bound=float(ci_lower),
        p_value_rope=p_val,
        allocated_alpha_t=alpha_t,
        lord_step=step_idx,
        is_discovery=is_disc,
    )


def evaluate_iqm_bootstrap_rope(
    incumbent_scores: Dict[str, float],
    challenger_scores: Dict[str, float],
    direction: TargetDirection,
    rope_epsilon: float,
    lord_controller: LORDController,
    num_bootstraps: int = 2000,
    rng_seed: int = 12345,
) -> ConfirmatoryTestResult:
    """Paired stratified bootstrap IQM test against H0: IQM(gain) <= rope_epsilon."""
    gains = _compute_paired_gains(incumbent_scores, challenger_scores, direction)
    k = gains.size
    iqm_gain = compute_iqm(gains)

    rng = np.random.default_rng(rng_seed + lord_controller.current_step)
    indices = rng.integers(0, k, size=(num_bootstraps, k))
    samples = gains[indices]
    if k >= 4:
        boot_iqms = stats.trim_mean(samples, proportiontocut=0.25, axis=1)
    else:
        boot_iqms = np.mean(samples, axis=1)

    p_val = float(np.mean(boot_iqms <= rope_epsilon))
    alpha_t, _, step_idx = lord_controller.step(p_val)
    ci_lower = float(np.quantile(boot_iqms, alpha_t))
    is_disc = bool(ci_lower > rope_epsilon and p_val <= alpha_t)

    if is_disc and step_idx not in lord_controller.discovery_steps:
        lord_controller.discovery_steps.append(step_idx)
    elif not is_disc and step_idx in lord_controller.discovery_steps:
        lord_controller.discovery_steps.remove(step_idx)

    return ConfirmatoryTestResult(
        directional_effect=iqm_gain,
        ci_lower_bound=ci_lower,
        p_value_rope=p_val,
        allocated_alpha_t=alpha_t,
        lord_step=step_idx,
        is_discovery=is_disc,
    )


def calibrate_required_k(
    empirical_sigma_d: float,
    rope_epsilon: float,
    alpha: float = 0.025,
    target_power: float = 0.80,
    min_k: int = 5,
    max_k: int = 15,
) -> int:
    """Computes required confirmatory seed count k* for a 2x ROPE true improvement."""
    if empirical_sigma_d <= 1e-9:
        return min_k
    effect_delta = max(rope_epsilon, 1e-6)
    snr = effect_delta / empirical_sigma_d
    for k in range(min_k, max_k + 1):
        df = k - 1
        t_crit = float(stats.t.ppf(1.0 - alpha, df=df))
        ncp = snr * math.sqrt(k)
        power = float(1.0 - stats.nct.cdf(t_crit, df=df, nc=ncp))
        if power >= target_power:
            return k
    return max_k
