"""Dual-Key Novelty Contract Engine for FRAME-AR (Milestone M2).

Evaluates candidate hypotheses against prior art across two orthogonal axes:
1. Literature Graph Proximity (d_lit > tau_lit = 0.15)
2. Functional Prediction Manifold Divergence (d_func > epsilon_func = 0.10)

Contract Invariant:
Pre-confirmatory check. Does NOT consume alpha wealth from LORDController (0 alpha spent).
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from frame.schemas.novelty import NoveltyContractResult, NoveltyVerdict

__all__ = [
    "NoveltyVerdict",
    "NoveltyContractResult",
    "DualKeyNoveltyContract",
]


class DualKeyNoveltyContract:
    """Pre-confirmatory novelty gate enforcing literature and functional divergence."""

    def __init__(self, tau_lit: float = 0.15, epsilon_func: float = 0.10) -> None:
        self.tau_lit = float(tau_lit)
        self.epsilon_func = float(epsilon_func)

    def _extract_features(self, text: Optional[str]) -> List[str]:
        """Extract lowercased alphanumeric tokens and adjacent bigrams."""
        if not text:
            return []
        tokens = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", str(text)) if t]
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def compute_literature_distance(
        self, candidate_code_or_text: str, prior_art_corpus: List[str]
    ) -> Tuple[float, str]:
        """Calculates distance d_lit to prior art corpus and identifies nearest reference.

        Uses unigram and bigram token overlap and Szymkiewicz-Simpson containment to
        accurately detect identical matches, wrapper code, and cosmetic rephrasing
        without external heavy dependencies.
        """
        if not prior_art_corpus:
            return 1.0, ""

        cand_feats = self._extract_features(candidate_code_or_text)
        cand_set = set(cand_feats)
        if not cand_set:
            return 1.0, str(prior_art_corpus[0])

        cand_cnt = Counter(cand_feats)
        norm_cand = math.sqrt(sum(v * v for v in cand_cnt.values()))

        best_dist = 1.0
        nearest = str(prior_art_corpus[0])

        for prior_doc in prior_art_corpus:
            prior_feats = self._extract_features(prior_doc)
            prior_set = set(prior_feats)
            if not prior_set:
                dist = 1.0
            else:
                inter = cand_set & prior_set
                if not inter:
                    dist = 1.0
                else:
                    prior_cnt = Counter(prior_feats)
                    dot = sum(cand_cnt[k] * prior_cnt[k] for k in inter)
                    norm_prior = math.sqrt(sum(v * v for v in prior_cnt.values()))
                    cos = (dot / (norm_cand * norm_prior)) if (norm_cand > 0 and norm_prior > 0) else 0.0

                    overlap = len(inter) / min(len(cand_set), len(prior_set))
                    sim = max(cos, overlap)
                    dist = max(0.0, min(1.0, 1.0 - sim))

            if dist < best_dist:
                best_dist = dist
                nearest = str(prior_doc)

        return float(best_dist), str(nearest)

    def compute_functional_divergence(
        self, y_true: np.ndarray, y_baseline_pred: np.ndarray, y_candidate_pred: np.ndarray
    ) -> Tuple[float, float]:
        """Calculates functional manifold divergence d_func = 1 - |rho| on holdouts.

        Evaluates the uncentered cosine angle between residual vectors:
        r_base = y_baseline_pred - y_true
        r_cand = y_candidate_pred - y_true

        Collinearity implies r_cand = c * r_base, giving |rho| = 1 and d_func = 0.
        Orthogonality gives rho = 0 and d_func = 1.
        """
        y_t = np.asarray(y_true, dtype=np.float64).ravel()
        y_b = np.asarray(y_baseline_pred, dtype=np.float64).ravel()
        y_c = np.asarray(y_candidate_pred, dtype=np.float64).ravel()

        if len(y_t) == 0:
            return 0.0, 1.0

        r_b = y_b - y_t
        r_c = y_c - y_t

        # Exact collinearity checks
        if np.allclose(r_b, r_c, atol=1e-12):
            return 0.0, 1.0
        if np.allclose(r_b, -r_c, atol=1e-12):
            return 0.0, -1.0

        norm_b = float(np.linalg.norm(r_b))
        norm_c = float(np.linalg.norm(r_c))

        if norm_b == 0.0 and norm_c == 0.0:
            return 0.0, 1.0
        if norm_b == 0.0 or norm_c == 0.0:
            return 1.0, 0.0

        dot = float(np.dot(r_b, r_c))
        denom = norm_b * norm_c
        rho = (dot / denom) if denom > 0 else 0.0
        rho = max(-1.0, min(1.0, rho))
        d_func = max(0.0, min(1.0, 1.0 - abs(rho)))
        return float(d_func), float(rho)

    def evaluate_novelty(
        self,
        hypothesis_id: str,
        candidate_code: str,
        prior_art_corpus: List[str],
        y_true: np.ndarray,
        y_baseline_pred: np.ndarray,
        y_candidate_pred: np.ndarray,
        state_node: Optional[Any] = None,
    ) -> NoveltyContractResult:
        """Evaluates candidate hypothesis against dual-key novelty criteria.

        Contract Invariant:
        Pre-confirmatory check. Does NOT call lord_controller.record_result();
        consumes exactly zero alpha wealth.
        """
        d_lit, nearest = self.compute_literature_distance(candidate_code, prior_art_corpus)
        d_func, rho = self.compute_functional_divergence(y_true, y_baseline_pred, y_candidate_pred)

        # Strict inequality invariant: d_lit > tau_lit and d_func > epsilon_func
        lit_pass = bool(d_lit > self.tau_lit)
        func_pass = bool(d_func > self.epsilon_func)

        if lit_pass and func_pass:
            verdict = NoveltyVerdict.APPROVED
            passed = True
            reason = None
        elif (not lit_pass) and func_pass:
            verdict = NoveltyVerdict.REJECTED_LIT_PROXIMITY
            passed = False
            reason = f"Literature proximity {d_lit:.4f} <= threshold {self.tau_lit:.4f} (nearest: {nearest})"
        elif lit_pass and (not func_pass):
            verdict = NoveltyVerdict.REJECTED_FUNC_COLLINEAR
            passed = False
            reason = f"Functional divergence {d_func:.4f} <= threshold {self.epsilon_func:.4f} (|rho|={abs(rho):.4f})"
        else:
            verdict = NoveltyVerdict.REJECTED_DUAL_FAILURE
            passed = False
            reason = f"Dual failure: literature ({d_lit:.4f} <= {self.tau_lit:.4f}) and functional ({d_func:.4f} <= {self.epsilon_func:.4f})"

        details = {
            "lit_pass": lit_pass,
            "func_pass": func_pass,
            "tau_lit": self.tau_lit,
            "epsilon_func": self.epsilon_func,
            "rejection_reason": reason,
        }

        result = NoveltyContractResult(
            hypothesis_id=hypothesis_id,
            passed=passed,
            verdict=verdict,
            d_lit=d_lit,
            tau_lit=self.tau_lit,
            nearest_prior_art=nearest,
            d_func=d_func,
            epsilon_func=self.epsilon_func,
            residual_correlation=rho,
            details=details,
            rejection_reason=reason,
        )
        result.contract_hash = result.compute_hash()

        if state_node is not None and hasattr(state_node, "governor_event_logs") and isinstance(state_node.governor_event_logs, list):
            state_node.governor_event_logs.append({
                "event_type": "novelty_audit",
                "hypothesis_id": hypothesis_id,
                "passed": passed,
                "verdict": verdict.value if hasattr(verdict, "value") else str(verdict),
                "d_lit": d_lit,
                "tau_lit": self.tau_lit,
                "d_func": d_func,
                "epsilon_func": self.epsilon_func,
                "nearest_prior_art": nearest,
                "rejection_reason": reason,
            })

        return result

