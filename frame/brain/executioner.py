"""The Executioner Watchdog & Orthogonal Warm-Restart Controller (Milestone M4).

Queries the Swarm Learning State Node across M recursion stacks to detect
epistemic stagnation (vanishing gradients, loss curvature plateau, circular AST mutations),
enforces radical early-stage architecture safeguards, and triggers orthogonal warm restarts.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from frame.schemas.executioner import ExecutionerEvent, StagnationTrigger


# Canonical period-4 orthogonal cognitive lane transition cycle:
# architecture -> features -> objectives -> optimization -> architecture
LANE_CYCLE: Dict[str, str] = {
    "architecture": "features",
    "features": "objectives",
    "objectives": "optimization",
    "optimization": "architecture",
}

DEFAULT_LANE_PRIORS: Dict[str, List[str]] = {
    "features": [
        "temporal_lags",
        "frequency_domain_wavelet",
        "cross_feature_interactions",
        "target_encoding_shrinkage",
    ],
    "objectives": [
        "focal_contrastive_loss",
        "huber_quantile_penalty",
        "sym_boundary_regularizer",
        "multi_task_pareto_frontier",
    ],
    "optimization": [
        "cosine_annealing_warm_restarts",
        "stochastic_weight_averaging",
        "gradient_centralization",
        "muon_newton_schulz_orthogonalization",
    ],
    "architecture": [
        "sparse_linear_attention",
        "rotary_position_embedding",
        "residual_gated_mlp",
        "mixture_of_recurrent_depths",
    ],
}


class ExecutionerWatchdog:
    """Watchdog agent auditing MCTS exploration branches and executing orthogonal warm restarts."""

    def __init__(
        self,
        m_stacks: int = 3,
        epsilon_grad: float = 1e-6,
        epsilon_plateau: float = 1e-4,
        tau_jaccard: float = 0.85,
        early_depth: int = 1,
        early_visits: int = 2,
    ) -> None:
        self.m_stacks = int(m_stacks)
        self.epsilon_grad = float(epsilon_grad)
        self.epsilon_plateau = float(epsilon_plateau)
        self.tau_jaccard = float(tau_jaccard)
        self.early_depth = int(early_depth)
        self.early_visits = int(early_visits)

    def audit_branch(
        self,
        branch_id: str,
        depth: int,
        visits: int,
        history: List[Dict[str, Any]],
        state_node: Optional[Any] = None,
    ) -> Tuple[bool, StagnationTrigger, str]:
        """Audits branch history across m_stacks window for epistemic stagnation.

        Step 1: Early-stage safeguard check.
        Step 2: History length check.
        Step 3: Extract trailing m_stacks window.
        Step 4: Check Vanishing Gradients FIRST (strict inequality < epsilon_grad).
        Step 5: Check Loss Plateau SECOND (inclusive inequality <= epsilon_plateau).
        Step 6: Check Circular AST Mutations THIRD (strict inequality > tau_jaccard).
        Step 7: Healthy branch.

        Returns:
            Tuple of (stagnant: bool, trigger: StagnationTrigger, reason: str).
        """
        is_stagnant = False
        trigger = StagnationTrigger.NONE
        reason = "Branch healthy: no epistemic stagnation detected"

        # Step 1: Radical Early-Stage Architecture Safeguard Check
        if depth <= self.early_depth and visits <= self.early_visits:
            reason = f"Exempt: early-stage architecture safeguard active (depth={depth} <= {self.early_depth}, visits={visits} <= {self.early_visits})"
        # Step 2: History length check
        elif not history or len(history) < self.m_stacks:
            hist_len = len(history) if history else 0
            reason = f"Insufficient history: {hist_len} steps < m_stacks={self.m_stacks}"
        else:
            # Step 3: Extract trailing window of m_stacks steps
            window = history[-self.m_stacks:]

            # Step 4: Check Vanishing Gradients FIRST (strict inequality < epsilon_grad)
            all_vanishing = True
            for step in window:
                norm = step.get("grad_norm")
                if norm is None or float(norm) >= self.epsilon_grad:
                    all_vanishing = False
                    break
            if all_vanishing:
                is_stagnant = True
                trigger = StagnationTrigger.VANISHING_GRADIENTS
                reason = f"Vanishing gradients detected: grad_norm < {self.epsilon_grad} across last {self.m_stacks} steps"
            else:
                # Step 5: Check Loss Plateau SECOND (inclusive inequality <= epsilon_plateau)
                losses = [float(step["loss"]) for step in window if "loss" in step and step["loss"] is not None]
                if len(losses) == self.m_stacks:
                    loss_span = max(losses) - min(losses)
                    if loss_span <= self.epsilon_plateau:
                        is_stagnant = True
                        trigger = StagnationTrigger.LOSS_PLATEAU
                        reason = f"Loss curvature plateau detected: range={loss_span:.2e} <= epsilon_plateau={self.epsilon_plateau:.2e}"

                if not is_stagnant:
                    # Step 6: Check Circular AST Mutations THIRD (strict inequality > tau_jaccard)
                    consecutive_jaccards: List[float] = []
                    has_ast_tokens = False
                    for i in range(len(window) - 1):
                        raw_curr = window[i].get("ast_tokens")
                        raw_next = window[i + 1].get("ast_tokens")
                        toks_curr = set(raw_curr) if raw_curr else set()
                        toks_next = set(raw_next) if raw_next else set()
                        if toks_curr or toks_next:
                            has_ast_tokens = True
                            union = toks_curr | toks_next
                            inter = toks_curr & toks_next
                            jaccard = (len(inter) / len(union)) if union else 0.0
                            consecutive_jaccards.append(jaccard)
                        else:
                            consecutive_jaccards.append(0.0)

                    if has_ast_tokens and consecutive_jaccards and all(j > self.tau_jaccard for j in consecutive_jaccards):
                        is_stagnant = True
                        trigger = StagnationTrigger.CIRCULAR_AST
                        reason = f"Circular AST mutations detected: consecutive Jaccard similarity > {self.tau_jaccard}"

        # Persist event log to state_node if available
        if state_node is not None and hasattr(state_node, "executioner_event_logs") and isinstance(state_node.executioner_event_logs, list):
            state_node.executioner_event_logs.append({
                "event_type": "stagnation_check",
                "branch_id": branch_id,
                "depth": depth,
                "visits": visits,
                "is_stagnant": is_stagnant,
                "trigger": trigger.value if hasattr(trigger, "value") else str(trigger),
                "reason": reason,
                "timestamp": time.time(),
            })

        return (is_stagnant, trigger, reason)


    def trigger_orthogonal_warm_restart(
        self,
        branch_id: str,
        current_lane: str,
        state_node: Any,
        trigger: Optional[Union[StagnationTrigger, str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> ExecutionerEvent:
        """Triggers an Orthogonal Warm Restart injecting fresh lane priors and persists event log.

        Cycles the stagnant branch into an orthogonal cognitive lane (period-4 cycle),
        generates a new unique branch identifier, injects lane-specific domain priors,
        and records the audit event into the state node's executioner_event_logs.
        """
        cleaned_lane = str(current_lane).strip().lower()
        target_lane = LANE_CYCLE.get(cleaned_lane, "architecture")
        priors = list(DEFAULT_LANE_PRIORS.get(target_lane, ["orthogonal_exploratory_prior"]))

        unique_suffix = uuid.uuid4().hex[:8]
        new_branch_id = f"{branch_id}_warm_{unique_suffix}"

        event_details = dict(details) if details else {}
        event_details.update({
            "parent_branch_id": branch_id,
            "source_lane": cleaned_lane,
            "target_lane": target_lane,
            "restarted_branch_id": new_branch_id,
        })

        event = ExecutionerEvent(
            event_type="ORTHOGONAL_WARM_RESTART",
            branch_id=new_branch_id,
            trigger=trigger or StagnationTrigger.NONE,
            target_lane=target_lane,
            source_lane=cleaned_lane,
            injected_priors=priors,
            details=event_details,
            timestamp=time.time(),
        )

        # Persist event log to state_node if available
        if hasattr(state_node, "executioner_event_logs") and isinstance(state_node.executioner_event_logs, list):
            log_entry = event.model_dump(mode="json")
            # Store parent branch_id as audited branch so queries by original branch_id match
            log_entry["branch_id"] = branch_id
            log_entry["new_branch_id"] = new_branch_id
            state_node.executioner_event_logs.append(log_entry)

        return event


__all__ = [
    "ExecutionerWatchdog",
    "StagnationTrigger",
    "ExecutionerEvent",
    "LANE_CYCLE",
    "DEFAULT_LANE_PRIORS",
]
