"""Decoupled Isolated Verifier Harness for FRAME-AR.

Evaluates candidate model weights on private holdout data using trusted verifier
scoring functions under torch.inference_mode(), with activation hash commitments.
"""

from dataclasses import dataclass, field
import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from frame.utils.canonical_json import canonical_json_dumps


@dataclass
class HarnessEvaluationResult:
    hypothesis_id: str
    passed: bool
    scores: Dict[str, float]
    mean_score: float
    activation_hashes: Dict[int, str]
    execution_time_s: float
    details: Dict[str, Any] = field(default_factory=dict)

    def compute_commitment_hash(self) -> str:
        data = {
            "hypothesis_id": self.hypothesis_id,
            "passed": self.passed,
            "scores": self.scores,
            "mean_score": self.mean_score,
            "activation_hashes": self.activation_hashes,
        }
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()


class DecoupledIsolatedVerifierHarness:
    """Isolated verifier harness that evaluates candidate weights on private holdout data.
    
    Protects against metric fabrication and data leakage by strictly isolating the holdout
    evaluation from candidate training and execution logic.
    """

    def __init__(
        self,
        holdout_data: List[Tuple[torch.Tensor, torch.Tensor]],
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        metric_name: str = "accuracy",
        commitment_steps: Tuple[int, ...] = (1, 10, 50),
    ):
        """
        Args:
            holdout_data: Private holdout split [(x_batch, y_batch), ...].
            metric_fn: Trusted evaluation metric function (preds, targets) -> float.
            metric_name: Name of the evaluation metric.
            commitment_steps: Evaluation steps at which intermediate activation hashes are committed.
        """
        self.holdout_data = holdout_data
        self.metric_fn = metric_fn
        self.metric_name = metric_name
        self.commitment_steps = commitment_steps

    def evaluate_candidate(
        self,
        hypothesis_id: str,
        model: nn.Module,
        seed_list: Optional[List[int]] = None,
    ) -> HarnessEvaluationResult:
        """Evaluates candidate model under torch.inference_mode() across seeds.
        
        Computes intermediate activation hashes at designated commitment steps.
        """
        t0 = time.perf_counter()
        model.eval()
        
        seed_list = seed_list or [42]
        scores_by_seed: Dict[str, float] = {}
        activation_hashes: Dict[int, str] = {}

        # Set evaluation device
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        with torch.inference_mode():
            for seed in seed_list:
                torch.manual_seed(seed)
                np.random.seed(seed)

                batch_scores: List[float] = []
                step = 0

                for x_b, y_b in self.holdout_data:
                    step += 1
                    x_b = x_b.to(device)
                    y_b = y_b.to(device)

                    # Forward pass
                    outputs = model(x_b)

                    # Compute activation hash at commitment steps
                    if step in self.commitment_steps and step not in activation_hashes:
                        act_data = outputs.detach().cpu().to(torch.float32).numpy().tobytes()
                        act_hash = hashlib.sha256(act_data).hexdigest()
                        activation_hashes[step] = act_hash

                    # Trusted metric computation
                    s = self.metric_fn(outputs, y_b)
                    batch_scores.append(float(s))

                mean_seed_score = float(np.mean(batch_scores)) if batch_scores else 0.0
                scores_by_seed[f"seed_{seed}"] = mean_seed_score

        overall_mean = float(np.mean(list(scores_by_seed.values()))) if scores_by_seed else 0.0
        elapsed = time.perf_counter() - t0

        return HarnessEvaluationResult(
            hypothesis_id=hypothesis_id,
            passed=True,
            scores=scores_by_seed,
            mean_score=overall_mean,
            activation_hashes=activation_hashes,
            execution_time_s=elapsed,
            details={"metric_name": self.metric_name, "total_batches": len(self.holdout_data)},
        )
