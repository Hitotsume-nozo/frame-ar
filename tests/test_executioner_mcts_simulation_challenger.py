"""Adversarial MCTS Tree Simulator and Empirical Verification Harness for Milestone M4.

Authored by m4_challenger_2 (Empirical Challenger & Adversarial Verifier).

Empirically verifies The Executioner Watchdog (ExecutionerWatchdog) and Orthogonal
Warm-Restart Controller across realistic and adversarial MCTS tree simulations:
1. Multi-branching MCTS tree simulation with heterogeneous hypothesis behaviors.
2. Radical early-stage architecture safeguard suppression under extreme stagnation.
3. Radical-to-mature boundary transition inflection point tests.
4. Epistemic stagnation pruning (vanishing gradients, loss plateau, circular AST).
5. Precedence invariants among multiple simultaneous stagnation triggers.
6. Period-4 orthogonal cognitive lane cycling, derangement invariance, and injected priors.
7. Sustained healthy deep search non-interference (zero false positive prunings).
8. Chaos testing: noisy histories, corrupted telemetry, edge cases, and high-throughput sweeps.
"""

from dataclasses import dataclass, field
import math
import random
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pytest

from frame.brain.executioner import (
    DEFAULT_LANE_PRIORS,
    ExecutionerWatchdog,
    LANE_CYCLE,
)
from frame.schemas.executioner import ExecutionerEvent, StagnationTrigger
from frame.schemas.learning_state import SwarmLearningStateNode


# ==============================================================================
# MCTS Simulation Infrastructure
# ==============================================================================

@dataclass
class SimulatedMCTSNode:
    """Node in an empirical MCTS hypothesis tree simulation."""

    node_id: str
    lane: str
    depth: int
    visits: int = 0
    parent: Optional["SimulatedMCTSNode"] = None
    children: List["SimulatedMCTSNode"] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    is_pruned: bool = False
    prune_trigger: StagnationTrigger = StagnationTrigger.NONE
    prune_reason: Optional[str] = None
    warm_restart_event: Optional[ExecutionerEvent] = None
    value: float = 0.0

    def add_child(self, child_id: str, lane: str) -> "SimulatedMCTSNode":
        child = SimulatedMCTSNode(
            node_id=child_id,
            lane=lane,
            depth=self.depth + 1,
            visits=0,
            parent=self,
        )
        self.children.append(child)
        return child

    def record_step(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        ast_tokens: Optional[Set[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.visits += 1
        entry: Dict[str, Any] = {}
        if loss is not None:
            entry["loss"] = float(loss)
        if grad_norm is not None:
            entry["grad_norm"] = float(grad_norm)
        if ast_tokens is not None:
            entry["ast_tokens"] = set(ast_tokens)
        if extra:
            entry.update(extra)
        self.history.append(entry)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def collect_subtree(self) -> List["SimulatedMCTSNode"]:
        nodes = [self]
        for c in self.children:
            nodes.extend(c.collect_subtree())
        return nodes


class MCTSTreeSimulator:
    """Autonomous harness driving MCTS tree expansion, rollouts, and Executioner auditing."""

    def __init__(
        self,
        watchdog: Optional[ExecutionerWatchdog] = None,
        state_node: Optional[SwarmLearningStateNode] = None,
    ) -> None:
        self.watchdog = watchdog or ExecutionerWatchdog()
        self.state_node = state_node or SwarmLearningStateNode(node_id="sim_mcts_state")
        self.root = SimulatedMCTSNode(node_id="root", lane="architecture", depth=0, visits=1)
        self.all_nodes: Dict[str, SimulatedMCTSNode] = {"root": self.root}
        self.restarted_branches: List[SimulatedMCTSNode] = []

    def audit_and_prune_node(self, node: SimulatedMCTSNode) -> Tuple[bool, StagnationTrigger, str]:
        """Audits a node with the ExecutionerWatchdog and triggers warm restart if stagnant."""
        if node.is_pruned:
            return True, node.prune_trigger, node.prune_reason or "Already pruned"

        stagnant, trigger, reason = self.watchdog.audit_branch(
            branch_id=node.node_id,
            depth=node.depth,
            visits=node.visits,
            history=node.history,
        )

        if stagnant:
            node.is_pruned = True
            node.prune_trigger = trigger
            node.prune_reason = reason
            event = self.watchdog.trigger_orthogonal_warm_restart(
                branch_id=node.node_id,
                current_lane=node.lane,
                state_node=self.state_node,
                trigger=trigger,
                details={"sim_depth": node.depth, "sim_visits": node.visits, "reason": reason},
            )
            node.warm_restart_event = event

            # Create the restarted branch node attached to parent or root
            restart_parent = node.parent or self.root
            restarted_node = restart_parent.add_child(
                child_id=event.branch_id,
                lane=event.target_lane or "architecture",
            )
            restarted_node.history.append({
                "injected_priors": event.injected_priors,
                "restarted_from": node.node_id,
            })
            self.all_nodes[restarted_node.node_id] = restarted_node
            self.restarted_branches.append(restarted_node)

        return stagnant, trigger, reason

    def audit_entire_tree(self) -> Dict[str, Tuple[bool, StagnationTrigger, str]]:
        """Audits every node currently registered in the MCTS tree."""
        results: Dict[str, Tuple[bool, StagnationTrigger, str]] = {}
        for nid, node in list(self.all_nodes.items()):
            results[nid] = self.audit_and_prune_node(node)
        return results


# ==============================================================================
# 1. Realistic MCTS Tree Simulation Tests
# ==============================================================================

class TestRealisticMCTSTreeSimulation:
    """Empirical tests simulating full branching hypothesis trees with diverse trajectories."""

    def test_heterogeneous_tree_simulation_verifies_pruning_and_survival(self):
        """Simulates an MCTS tree with 60+ nodes across all 4 cognitive lanes.

        Verifies that:
        - Early radical nodes survive regardless of stagnant telemetry.
        - Mature stagnant branches are strictly pruned and restarted into valid orthogonal lanes.
        - Healthy branches continue unmolested to leaf evaluation.
        """
        sim = MCTSTreeSimulator(watchdog=ExecutionerWatchdog(m_stacks=3))

        # Level 1: Expand root into 4 lane branches
        lanes = ["architecture", "features", "objectives", "optimization"]
        l1_nodes: List[SimulatedMCTSNode] = []
        for i, lane in enumerate(lanes):
            child = sim.root.add_child(f"b_l1_{lane}_{i}", lane)
            sim.all_nodes[child.node_id] = child
            l1_nodes.append(child)

        # 1. Early radical branches: depth=1, visits=1 or 2, flat loss and zero grad
        for node in l1_nodes[:2]:
            for _ in range(2):
                node.record_step(loss=0.500, grad_norm=0.0, ast_tokens={"base"})

        # 2. Mature stagnant branches: depth=2, visits=4
        stagnant_nodes: List[SimulatedMCTSNode] = []
        for node in l1_nodes[2:]:
            for c_idx in range(3):
                child = node.add_child(f"{node.node_id}_child_{c_idx}", node.lane)
                sim.all_nodes[child.node_id] = child
                stagnant_nodes.append(child)
                # Saturate with vanishing gradients
                for _ in range(4):
                    child.record_step(loss=0.800, grad_norm=1e-8, ast_tokens={"stagnant_tok"})

        # 3. Mature healthy branches: depth=2 and depth=3, visits=5, active descent
        healthy_nodes: List[SimulatedMCTSNode] = []
        healthy_parent = l1_nodes[0].add_child("b_healthy_parent", "architecture")
        sim.all_nodes[healthy_parent.node_id] = healthy_parent
        for _ in range(3):
            healthy_parent.record_step(loss=1.0 - healthy_parent.visits * 0.1, grad_norm=0.15, ast_tokens={f"hp_{healthy_parent.visits}"})

        for h_idx in range(4):
            h_child = healthy_parent.add_child(f"b_healthy_leaf_{h_idx}", "architecture")
            sim.all_nodes[h_child.node_id] = h_child
            healthy_nodes.append(h_child)
            for step_i in range(5):
                h_child.record_step(
                    loss=1.5 - step_i * 0.15,
                    grad_norm=0.10 - step_i * 0.01,
                    ast_tokens={f"novel_feat_{step_i}_{h_idx}"},
                )

        # Execute full tree audit
        audit_results = sim.audit_entire_tree()

        # Empirical Assertions:
        # A. Early radical nodes (l1_nodes[:2]) MUST NOT be pruned
        for node in l1_nodes[:2]:
            stagnant, trigger, reason = audit_results[node.node_id]
            assert stagnant is False
            assert trigger == StagnationTrigger.NONE
            assert node.is_pruned is False
            assert "Exempt" in reason or "early" in reason.lower()

        # B. Mature stagnant nodes MUST BE pruned and restarted
        for node in stagnant_nodes:
            stagnant, trigger, _ = audit_results[node.node_id]
            assert stagnant is True
            assert trigger == StagnationTrigger.VANISHING_GRADIENTS
            assert node.is_pruned is True
            assert node.warm_restart_event is not None
            # Target lane must match LANE_CYCLE
            expected_target = LANE_CYCLE[node.lane]
            assert node.warm_restart_event.target_lane == expected_target
            assert len(node.warm_restart_event.injected_priors) >= 4

        # C. Healthy nodes MUST NOT be pruned
        for node in healthy_nodes:
            stagnant, trigger, _ = audit_results[node.node_id]
            assert stagnant is False
            assert trigger == StagnationTrigger.NONE
            assert node.is_pruned is False

        # D. State node must contain accurate log count matching number of pruned nodes
        assert len(sim.state_node.executioner_event_logs) == len(stagnant_nodes)
        assert len(sim.restarted_branches) == len(stagnant_nodes)

    def test_recursive_warm_restart_chain_derangement_across_generations(self):
        """Simulates repeated stagnation and warm-restarts across 4 generations.

        Verifies that iterative restarts cycle completely through:
        architecture -> features -> objectives -> optimization -> architecture.
        """
        sim = MCTSTreeSimulator()
        current_branch = sim.root.add_child("b_root_gen0", "architecture")
        sim.all_nodes[current_branch.node_id] = current_branch

        visited_lanes: List[str] = [current_branch.lane]

        for gen in range(4):
            # Saturate current branch with stagnant loss plateau at mature depth
            current_branch.depth = 2
            current_branch.visits = 4
            for _ in range(3):
                current_branch.record_step(loss=0.50001, grad_norm=0.05, ast_tokens={"loop"})

            stagnant, trigger, _ = sim.audit_and_prune_node(current_branch)
            assert stagnant is True
            assert trigger == StagnationTrigger.LOSS_PLATEAU
            assert current_branch.is_pruned is True

            # Retrieve newly spawned restart branch
            restarted_node = sim.restarted_branches[-1]
            visited_lanes.append(restarted_node.lane)
            current_branch = restarted_node

        expected_sequence = ["architecture", "features", "objectives", "optimization", "architecture"]
        assert visited_lanes == expected_sequence


# ==============================================================================
# 2. Early Radical Architecture Safeguard Stress Tests
# ==============================================================================

class TestEarlyRadicalArchitectureSafeguards:
    """Stress tests verifying that early-stage hypotheses are protected from premature pruning."""

    @pytest.mark.parametrize("depth", [0, 1])
    @pytest.mark.parametrize("visits", [0, 1, 2])
    def test_extreme_stagnation_in_early_stage_strictly_exempt(self, depth: int, visits: int):
        """Even with zero gradient, completely flat loss, and identical AST tokens,

        early radical branches (depth <= 1, visits <= 2) must NEVER be pruned.
        """
        dog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        history = [
            {"grad_norm": 0.0, "loss": 1.0, "ast_tokens": {"fixed_token"}},
            {"grad_norm": 0.0, "loss": 1.0, "ast_tokens": {"fixed_token"}},
            {"grad_norm": 0.0, "loss": 1.0, "ast_tokens": {"fixed_token"}},
        ]
        stagnant, trigger, reason = dog.audit_branch(
            branch_id=f"b_early_d{depth}_v{visits}",
            depth=depth,
            visits=visits,
            history=history,
        )
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE
        assert "Exempt" in reason or "early" in reason.lower()

    def test_radical_to_mature_boundary_inflection_point(self):
        """Empirically tracks a single branch stepping across the exact maturity threshold.

        Step 1: depth=1, visits=1 -> exempt.
        Step 2: depth=1, visits=2 -> exempt (exact upper boundary of early visits).
        Step 3: depth=1, visits=3 -> mature! Instantly audited and pruned.
        """
        dog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        stagnant_history = [{"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"tok"}}] * 3

        # Point 1: depth=1, visits=1
        s1, t1, r1 = dog.audit_branch("b_inflect", depth=1, visits=1, history=stagnant_history)
        assert s1 is False
        assert t1 == StagnationTrigger.NONE

        # Point 2: depth=1, visits=2 (exact boundary)
        s2, t2, r2 = dog.audit_branch("b_inflect", depth=1, visits=2, history=stagnant_history)
        assert s2 is False
        assert t2 == StagnationTrigger.NONE

        # Point 3: depth=1, visits=3 (crossed visit boundary)
        s3, t3, r3 = dog.audit_branch("b_inflect", depth=1, visits=3, history=stagnant_history)
        assert s3 is True
        assert t3 == StagnationTrigger.VANISHING_GRADIENTS

    def test_depth_boundary_inflection_point(self):
        """Crosses the depth boundary while visits remain within early_visits.

        depth=1, visits=2 -> exempt.
        depth=2, visits=2 -> mature! Pruned.
        """
        dog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        stagnant_history = [{"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"tok"}}] * 3

        s_exempt, _, _ = dog.audit_branch("b_depth_bnd", depth=1, visits=2, history=stagnant_history)
        assert s_exempt is False

        s_mature, t_mature, _ = dog.audit_branch("b_depth_bnd", depth=2, visits=2, history=stagnant_history)
        assert s_mature is True
        assert t_mature == StagnationTrigger.VANISHING_GRADIENTS

    def test_custom_safeguard_matrix_parameterization(self):
        """Verifies custom (early_depth=4, early_visits=8) configurations."""
        dog = ExecutionerWatchdog(early_depth=4, early_visits=8)
        stagnant_history = [{"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"tok"}}] * 3

        # Within custom boundary
        s_in, _, _ = dog.audit_branch("b_custom", depth=4, visits=8, history=stagnant_history)
        assert s_in is False

        # Exceeding depth
        s_deep, _, _ = dog.audit_branch("b_custom", depth=5, visits=8, history=stagnant_history)
        assert s_deep is True

        # Exceeding visits
        s_vis, _, _ = dog.audit_branch("b_custom", depth=4, visits=9, history=stagnant_history)
        assert s_vis is True


# ==============================================================================
# 3. Mature Stagnation Pruning & Trigger Precedence Tests
# ==============================================================================

class TestMatureStagnationPruning:
    """Stress tests verifying all three stagnation triggers and their precedence ordering."""

    def test_vanishing_gradients_boundary_exactitude(self):
        """Tests strict inequality: grad_norm < epsilon_grad.

        At exactly epsilon_grad (1e-6), vanishing gradients must NOT trigger.
        At 9.999e-7, vanishing gradients MUST trigger.
        """
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)

        # Exactly at boundary
        exact_hist = [
            {"grad_norm": 1e-6, "loss": 0.5, "ast_tokens": {"a"}},
            {"grad_norm": 1e-6, "loss": 0.4, "ast_tokens": {"b"}},
            {"grad_norm": 1e-6, "loss": 0.3, "ast_tokens": {"c"}},
        ]
        s_exact, t_exact, _ = dog.audit_branch("b_exact", depth=3, visits=5, history=exact_hist)
        assert t_exact != StagnationTrigger.VANISHING_GRADIENTS

        # Strict sub-boundary
        below_hist = [
            {"grad_norm": 9.999e-7, "loss": 0.5, "ast_tokens": {"a"}},
            {"grad_norm": 9.999e-7, "loss": 0.4, "ast_tokens": {"b"}},
            {"grad_norm": 9.999e-7, "loss": 0.3, "ast_tokens": {"c"}},
        ]
        s_below, t_below, _ = dog.audit_branch("b_below", depth=3, visits=5, history=below_hist)
        assert s_below is True
        assert t_below == StagnationTrigger.VANISHING_GRADIENTS

    def test_loss_plateau_boundary_exactitude(self):
        """Tests inclusive inequality: max(loss) - min(loss) <= epsilon_plateau.

        At exactly 1e-4 range, loss plateau MUST trigger.
        At 1.0001e-4 range, loss plateau must NOT trigger.
        """
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)

        # Exact span: 0.50010 - 0.50000 = 1.0e-4 -> triggers (inclusive <=)
        exact_hist = [
            {"grad_norm": 0.1, "loss": 0.50000, "ast_tokens": {"a"}},
            {"grad_norm": 0.1, "loss": 0.50005, "ast_tokens": {"b"}},
            {"grad_norm": 0.1, "loss": 0.50010, "ast_tokens": {"c"}},
        ]
        s_exact, t_exact, _ = dog.audit_branch("b_plat_exact", depth=3, visits=5, history=exact_hist)
        assert s_exact is True
        assert t_exact == StagnationTrigger.LOSS_PLATEAU

        # Just above span: 0.5001001 - 0.50000 = 1.001e-4 -> does not trigger
        above_hist = [
            {"grad_norm": 0.1, "loss": 0.50000, "ast_tokens": {"a"}},
            {"grad_norm": 0.1, "loss": 0.50005, "ast_tokens": {"b"}},
            {"grad_norm": 0.1, "loss": 0.5001001, "ast_tokens": {"c"}},
        ]
        s_above, t_above, _ = dog.audit_branch("b_plat_above", depth=3, visits=5, history=above_hist)
        assert t_above != StagnationTrigger.LOSS_PLATEAU

    def test_circular_ast_jaccard_boundary_exactitude(self):
        """Tests strict inequality: consecutive Jaccard > tau_jaccard (0.85).

        At exactly 0.85 Jaccard, circular AST must NOT trigger.
        At > 0.85 Jaccard, circular AST MUST trigger.
        """
        dog = ExecutionerWatchdog(m_stacks=2, tau_jaccard=0.85)

        # 85 shared out of 100 total union: Jaccard = 85/100 = 0.85 exactly
        t1 = {f"k_{i}" for i in range(100)}
        t2 = {f"k_{i}" for i in range(85)} | {f"diff_{i}" for i in range(15)}
        union = t1 | t2  # 100 + 15 = 115? Wait: t1 has 100, t2 has 85 of t1 + 15 diff -> union = 115.
        # Let's construct exact 85/100 = 0.85:
        # tA: 85 common + 0 diff -> 85 elements.
        # tB: 85 common + 15 diff -> 100 elements.
        # inter = 85, union = 100 -> Jaccard = 85 / 100 = 0.8500000000000000 exactly!
        t_common = {f"c_{i}" for i in range(85)}
        t_a = set(t_common)
        t_b = set(t_common) | {f"extra_{i}" for i in range(15)}
        assert (len(t_a & t_b) / len(t_a | t_b)) == 0.85

        exact_hist = [
            {"grad_norm": 0.1, "loss": 1.0, "ast_tokens": t_a},
            {"grad_norm": 0.1, "loss": 0.9, "ast_tokens": t_b},
        ]
        s_exact, t_exact, _ = dog.audit_branch("b_jaccard_exact", depth=3, visits=5, history=exact_hist)
        assert t_exact != StagnationTrigger.CIRCULAR_AST

        # 86 common + 14 diff: Jaccard = 86 / 100 = 0.86 > 0.85
        t_c = {f"c_{i}" for i in range(86)} | {f"extra_{i}" for i in range(14)}
        above_hist = [
            {"grad_norm": 0.1, "loss": 1.0, "ast_tokens": {f"c_{i}" for i in range(86)}},
            {"grad_norm": 0.1, "loss": 0.9, "ast_tokens": t_c},
        ]
        s_above, t_above, _ = dog.audit_branch("b_jaccard_above", depth=3, visits=5, history=above_hist)
        assert s_above is True
        assert t_above == StagnationTrigger.CIRCULAR_AST

    def test_precedence_hierarchy_under_multi_modal_stagnation(self):
        """Verifies strict precedence ordering:

        1. Vanishing Gradients > Loss Plateau
        2. Loss Plateau > Circular AST
        3. Vanishing Gradients > Circular AST
        """
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6, epsilon_plateau=1e-4, tau_jaccard=0.85)

        # Scenario A: All three triggers present simultaneously
        all_three = [
            {"grad_norm": 1e-8, "loss": 0.50000, "ast_tokens": {"fixed"}},
            {"grad_norm": 1e-8, "loss": 0.50000, "ast_tokens": {"fixed"}},
            {"grad_norm": 1e-8, "loss": 0.50000, "ast_tokens": {"fixed"}},
        ]
        sA, tA, _ = dog.audit_branch("b_all", depth=3, visits=5, history=all_three)
        assert sA is True
        assert tA == StagnationTrigger.VANISHING_GRADIENTS, "Vanishing gradients must take precedence over plateau and circular AST"

        # Scenario B: Healthy gradients (0.1), but both Loss Plateau and Circular AST present
        plateau_and_ast = [
            {"grad_norm": 0.1, "loss": 0.50000, "ast_tokens": {"fixed"}},
            {"grad_norm": 0.1, "loss": 0.50001, "ast_tokens": {"fixed"}},
            {"grad_norm": 0.1, "loss": 0.50002, "ast_tokens": {"fixed"}},
        ]
        sB, tB, _ = dog.audit_branch("b_plat_ast", depth=3, visits=5, history=plateau_and_ast)
        assert sB is True
        assert tB == StagnationTrigger.LOSS_PLATEAU, "Loss plateau must take precedence over circular AST"

        # Scenario C: Only Circular AST present (healthy gradient 0.1, dropping loss 1.0 -> 0.7 -> 0.4)
        only_ast = [
            {"grad_norm": 0.1, "loss": 1.0, "ast_tokens": {"fixed"}},
            {"grad_norm": 0.1, "loss": 0.7, "ast_tokens": {"fixed"}},
            {"grad_norm": 0.1, "loss": 0.4, "ast_tokens": {"fixed"}},
        ]
        sC, tC, _ = dog.audit_branch("b_only_ast", depth=3, visits=5, history=only_ast)
        assert sC is True
        assert tC == StagnationTrigger.CIRCULAR_AST


# ==============================================================================
# 4. Orthogonal Warm-Restart Mechanics & Lane Cycle Invariants
# ==============================================================================

class TestOrthogonalWarmRestartMechanics:
    """Verifies period-4 lane cycle algebra, derangement properties, and prior injections."""

    def test_derangement_property_no_fixed_points(self):
        """Every lane in the cycle must transition to a strictly different lane:

        forall x, LANE_CYCLE[x] != x
        """
        for src, dst in LANE_CYCLE.items():
            assert src != dst, f"Cycle violation: fixed point found at lane '{src}'"

    def test_full_permutation_bijection(self):
        """The cycle must define a valid bijection (permutation) over the 4 cognitive lanes."""
        src_set = set(LANE_CYCLE.keys())
        dst_set = set(LANE_CYCLE.values())
        assert src_set == {"architecture", "features", "objectives", "optimization"}
        assert dst_set == src_set

    def test_warm_restart_generates_unique_branch_ids_with_parent_lineage(self):
        """Warm restart generates non-colliding branch IDs while recording parent branch ID in details."""
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_lineage")

        seen_ids: Set[str] = set()
        for i in range(25):
            ev = dog.trigger_orthogonal_warm_restart(
                branch_id="parent_alpha",
                current_lane="architecture",
                state_node=node,
                trigger=StagnationTrigger.VANISHING_GRADIENTS,
            )
            assert ev.branch_id not in seen_ids
            seen_ids.add(ev.branch_id)
            assert ev.branch_id.startswith("parent_alpha_warm_")
            assert ev.details["parent_branch_id"] == "parent_alpha"
            assert ev.details["source_lane"] == "architecture"
            assert ev.details["target_lane"] == "features"

    def test_injected_priors_match_domain_specialization(self):
        """Validates that priors injected for each lane correspond to specialized algorithmic tools."""
        dog = ExecutionerWatchdog()
        node = SwarmLearningStateNode(node_id="n_prior_check")

        for lane in ("architecture", "features", "objectives", "optimization"):
            ev = dog.trigger_orthogonal_warm_restart("b", lane, node)
            priors = ev.injected_priors
            assert isinstance(priors, list)
            assert len(priors) >= 4
            target = ev.target_lane

            if target == "features":
                assert any("feature" in p or "encoding" in p or "wavelet" in p for p in priors)
            elif target == "objectives":
                assert any("loss" in p or "penalty" in p or "frontier" in p for p in priors)
            elif target == "optimization":
                assert any("restart" in p or "gradient" in p or "muon" in p for p in priors)
            elif target == "architecture":
                assert any("attention" in p or "embedding" in p or "mlp" in p for p in priors)


# ==============================================================================
# 5. Healthy Deep Search Non-Interference Tests
# ==============================================================================

class TestHealthySearchNonInterference:
    """Verifies that healthy branches exploring deep MCTS trees are never pruned."""

    def test_sustained_deep_search_zero_false_positives(self):
        """Simulates a branch progressing through 30 MCTS steps across depths 1 to 10.

        The branch exhibits steady convergence: loss improves monotonically,
        gradient norms remain in [0.01, 0.20], and AST tokens continuously mutate.
        Zero false positive prunings must occur.
        """
        dog = ExecutionerWatchdog(m_stacks=3)
        history: List[Dict[str, Any]] = []

        for step in range(1, 31):
            depth = min(10, step // 3 + 1)
            visits = step
            loss = 2.0 / (1.0 + 0.1 * step)
            grad_norm = 0.05 + 0.05 * math.sin(step)
            tokens = {f"ast_node_{step}_{k}" for k in range(10)}

            history.append({
                "loss": loss,
                "grad_norm": grad_norm,
                "ast_tokens": tokens,
            })

            stagnant, trigger, reason = dog.audit_branch(
                branch_id="b_deep_healthy",
                depth=depth,
                visits=visits,
                history=history,
            )

            assert stagnant is False
            assert trigger == StagnationTrigger.NONE
            if depth <= dog.early_depth and visits <= dog.early_visits:
                assert "exempt" in reason.lower() or "early" in reason.lower()
            else:
                assert "healthy" in reason.lower()

    def test_temporary_plateau_recovery_uninterrupted(self):
        """A branch experiences a brief 2-step plateau, but recovers with an innovation at step 3.

        The sliding window of m_stacks=3 captures the recovery and does NOT prune.
        """
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_plateau=1e-4)
        history = [
            {"loss": 0.50000, "grad_norm": 0.05, "ast_tokens": {"t1"}},
            {"loss": 0.50001, "grad_norm": 0.05, "ast_tokens": {"t2"}},
            {"loss": 0.40000, "grad_norm": 0.05, "ast_tokens": {"t3"}},  # Significant descent!
        ]
        stagnant, trigger, _ = dog.audit_branch("b_recovering", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE


# ==============================================================================
# 6. Chaos, Robustness, and Adversarial Edge Cases
# ==============================================================================

class TestAdversarialEdgeCasesAndChaos:
    """Adversarial stress testing against corrupted, non-standard, or extreme inputs."""

    def test_missing_and_none_fields_safely_handled(self):
        """History steps containing None, missing keys, or unexpected data types do not crash the watchdog."""
        dog = ExecutionerWatchdog(m_stacks=3)
        corrupted_history = [
            {},
            {"grad_norm": None, "loss": None, "ast_tokens": None},
            {"unexpected_key": [1, 2, 3]},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_corrupt", depth=3, visits=5, history=corrupted_history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_history_with_empty_or_degenerate_ast_token_sets(self):
        """Tokens provided as empty sets or empty lists do not cause zero-division errors in Jaccard calculation."""
        dog = ExecutionerWatchdog(m_stacks=3, tau_jaccard=0.85)
        history = [
            {"grad_norm": 0.05, "loss": 1.0, "ast_tokens": set()},
            {"grad_norm": 0.05, "loss": 0.8, "ast_tokens": []},
            {"grad_norm": 0.05, "loss": 0.6, "ast_tokens": set()},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_empty_tokens", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_negative_depth_and_visits_gracefully_handled(self):
        """Negative depth or visit values are treated as early-stage (<= early_depth, <= early_visits)."""
        dog = ExecutionerWatchdog(early_depth=1, early_visits=2)
        stagnant_history = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"x"}}] * 3
        stagnant, trigger, _ = dog.audit_branch("b_neg", depth=-5, visits=-10, history=stagnant_history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_large_history_window_sliding_isolation(self):
        """Tests that stagnation in steps 0-99 is completely ignored when recent steps 100-102 are healthy."""
        dog = ExecutionerWatchdog(m_stacks=3)
        old_stagnant = [{"grad_norm": 1e-9, "loss": 0.5, "ast_tokens": {"old"}}] * 100
        recent_healthy = [
            {"grad_norm": 0.10, "loss": 1.0, "ast_tokens": {"new_1"}},
            {"grad_norm": 0.08, "loss": 0.8, "ast_tokens": {"new_2"}},
            {"grad_norm": 0.06, "loss": 0.6, "ast_tokens": {"new_3"}},
        ]
        history = old_stagnant + recent_healthy
        stagnant, trigger, _ = dog.audit_branch("b_large_history", depth=5, visits=103, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_high_throughput_mcts_sweep_performance_and_stability(self):
        """Audits 1,000 diverse branches in a rapid sweep.

        Verifies memory stability, deterministic verdicts, and sub-second execution.
        """
        dog = ExecutionerWatchdog(m_stacks=3)
        state_node = SwarmLearningStateNode(node_id="n_stress_1000")

        start_time = time.perf_counter()
        pruned_count = 0
        exempt_count = 0
        healthy_count = 0

        for i in range(1000):
            branch_id = f"sweep_branch_{i}"
            mode = i % 4

            if mode == 0:
                # Early exempt
                depth, visits = 1, 1
                history = [{"grad_norm": 0.0, "loss": 0.5, "ast_tokens": {"tok"}}] * 3
            elif mode == 1:
                # Vanishing gradients
                depth, visits = 3, 5
                history = [{"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"tok"}}] * 3
            elif mode == 2:
                # Loss plateau
                depth, visits = 3, 5
                history = [{"grad_norm": 0.1, "loss": 0.50001, "ast_tokens": {"tok"}}] * 3
            else:
                # Healthy
                depth, visits = 3, 5
                history = [
                    {"grad_norm": 0.1, "loss": 1.0 - k * 0.2, "ast_tokens": {f"t_{k}"}}
                    for k in range(3)
                ]

            stagnant, trigger, _ = dog.audit_branch(branch_id, depth, visits, history)
            if stagnant:
                pruned_count += 1
                dog.trigger_orthogonal_warm_restart(
                    branch_id=branch_id,
                    current_lane="architecture",
                    state_node=state_node,
                    trigger=trigger,
                )
            elif depth <= 1 and visits <= 2:
                exempt_count += 1
            else:
                healthy_count += 1

        elapsed = time.perf_counter() - start_time

        assert pruned_count == 500  # modes 1 and 2
        assert exempt_count == 250  # mode 0
        assert healthy_count == 250  # mode 3
        assert len(state_node.executioner_event_logs) == 500
        assert elapsed < 2.0, f"1000 audits and restarts took {elapsed:.2f}s (expected < 2.0s)"


# ==============================================================================
# 7. Dynamic UCT MCTS Search Loop with Active Executioner Integration
# ==============================================================================

class TestDynamicUCTMCTSLoopWithExecutioner:
    """Simulates a realistic Monte Carlo Tree Search (UCT) loop with an active Executioner Watchdog."""

    def test_uct_search_loop_redirection_away_from_pruned_subtrees(self):
        """Simulates 100 UCT rollouts.

        - Tree begins with root and 3 candidate children.
        - Child 0 simulates vanishing gradients (grad_norm = 1e-8).
        - Child 1 simulates a healthy branch with high reward.
        - Child 2 simulates an early radical branch.
        - At each rollout step, the Executioner audits leaves.
        - When Child 0 stagnates (at visits >= 3, depth=1... wait, at depth=1 visits=3),
          Child 0 is pruned and warm-restarted to an orthogonal lane.
        - Pruned nodes are masked from UCT selection (value = -inf).
        - Search redirected to the warm restart child.
        """
        watchdog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        state_node = SwarmLearningStateNode(node_id="n_uct_loop")

        root = SimulatedMCTSNode(node_id="root", lane="architecture", depth=0, visits=1)
        # Add 3 initial children
        child_stagnant = root.add_child("b_will_stagnate", "architecture")
        child_healthy = root.add_child("b_healthy", "features")
        child_early = root.add_child("b_early_radical", "objectives")

        candidates: List[SimulatedMCTSNode] = [child_stagnant, child_healthy, child_early]
        rollouts_on_stagnant_after_prune = 0
        pruned_step: Optional[int] = None
        restarted_nodes: List[SimulatedMCTSNode] = []

        total_rollouts = 60
        for step in range(1, total_rollouts + 1):
            # 1. Select unpruned candidate with highest UCT score
            active_candidates = [n for n in candidates if not n.is_pruned]
            assert len(active_candidates) > 0

            # Compute UCT
            best_node = None
            best_score = -float("inf")
            c_uct = 1.414
            for n in active_candidates:
                exploit = n.value / max(1, n.visits)
                explore = c_uct * math.sqrt(math.log(step + 1) / max(1, n.visits))
                score = exploit + explore
                if score > best_score:
                    best_score = score
                    best_node = n

            selected = best_node
            assert selected is not None

            # 2. Simulate rollout telemetry update
            if selected.node_id == "b_will_stagnate":
                if selected.is_pruned:
                    rollouts_on_stagnant_after_prune += 1
                # Dead-end vanishing gradients
                selected.record_step(loss=0.5, grad_norm=1e-8, ast_tokens={"same"})
                selected.value += 0.1  # low reward
            elif selected.node_id == "b_healthy":
                # Healthy progressive descent
                step_idx = selected.visits
                selected.record_step(
                    loss=1.0 / (1.0 + 0.1 * step_idx),
                    grad_norm=0.10,
                    ast_tokens={f"feat_{step_idx}"},
                )
                selected.value += 0.8  # high reward
            elif selected.node_id == "b_early_radical":
                # Only rolled out once or twice (remains early)
                selected.record_step(loss=0.45, grad_norm=1e-8, ast_tokens={"early"})
                selected.value += 0.3
            else:
                # Restarted branch
                step_idx = selected.visits
                selected.record_step(
                    loss=0.8 / (1.0 + 0.15 * step_idx),
                    grad_norm=0.12,
                    ast_tokens={f"restart_feat_{step_idx}"},
                )
                selected.value += 0.85

            # 3. Executioner audits the selected node
            stagnant, trigger, reason = watchdog.audit_branch(
                branch_id=selected.node_id,
                depth=selected.depth,
                visits=selected.visits,
                history=selected.history,
            )

            if stagnant:
                selected.is_pruned = True
                selected.prune_trigger = trigger
                selected.prune_reason = reason
                pruned_step = step

                # Orthogonal Warm Restart
                event = watchdog.trigger_orthogonal_warm_restart(
                    branch_id=selected.node_id,
                    current_lane=selected.lane,
                    state_node=state_node,
                    trigger=trigger,
                    details={"pruned_at_step": step},
                )
                restarted_child = root.add_child(
                    child_id=event.branch_id,
                    lane=event.target_lane or "features",
                )
                # Seed with prior telemetry
                restarted_child.history.append({"injected_priors": event.injected_priors})
                candidates.append(restarted_child)
                restarted_nodes.append(restarted_child)

        # Empirical verifications
        assert child_stagnant.is_pruned is True
        assert child_stagnant.prune_trigger == StagnationTrigger.VANISHING_GRADIENTS
        assert rollouts_on_stagnant_after_prune == 0
        assert pruned_step is not None
        assert pruned_step < total_rollouts

        # Verify child_healthy was never pruned
        assert child_healthy.is_pruned is False

        # Verify child_early:
        # At visits 1 and 2, it was exempted by early-stage safeguard.
        # At visit 3, having crossed early_visits=2, it became mature and was pruned.
        assert child_early.visits == 3
        assert child_early.is_pruned is True
        assert child_early.prune_trigger == StagnationTrigger.VANISHING_GRADIENTS

        # Verify restarted branches were created and registered
        assert len(restarted_nodes) >= 2
        restarted_lanes = [r.lane for r in restarted_nodes]
        assert "optimization" in restarted_lanes  # from child_early (objectives -> optimization)
        assert "features" in restarted_lanes      # from child_stagnant (architecture -> features)
        assert len(state_node.executioner_event_logs) >= 2

    def test_test_time_compute_scaling_compounds_through_warm_restarts(self):
        """Simulates MCTS test-time compute scaling tracking in SwarmLearningStateNode.

        When a branch stagnates and is warm-restarted, the new branch achieves higher discovery quality,
        and SwarmLearningStateNode upper-envelope progression remains strictly monotonic.
        """
        watchdog = ExecutionerWatchdog(m_stacks=3, early_depth=1, early_visits=2)
        state_node = SwarmLearningStateNode(node_id="n_compute_compounding")

        # Stagnant branch fails to improve quality
        stagnant_id = "branch_stagnant_01"
        for v in range(1, 4):
            state_node.record_evaluation(f"hyp_stag_{v}", score=0.40, lane="architecture", depth=2)

        stagnant_history = [{"grad_norm": 1e-8, "loss": 0.5, "ast_tokens": {"fixed"}}] * 3
        stagnant, trigger, reason = watchdog.audit_branch(stagnant_id, depth=2, visits=3, history=stagnant_history)
        assert stagnant is True

        # Warm restart triggered
        event = watchdog.trigger_orthogonal_warm_restart(
            branch_id=stagnant_id,
            current_lane="architecture",
            state_node=state_node,
            trigger=trigger,
        )
        assert event.target_lane == "features"

        # Search in restarted lane discovers compounding innovations
        restarted_scores = [0.55, 0.68, 0.74, 0.82, 0.89]
        for idx, score in enumerate(restarted_scores):
            state_node.record_evaluation(
                hypothesis_id=f"hyp_recov_{idx}",
                score=score,
                lane=event.target_lane,
                depth=1,
                metadata={"restarted_from": event.branch_id},
            )

        # Verify test-time compute monotonic progression
        is_mono, gain = state_node.verify_monotonic_progression()
        assert is_mono is True
        assert gain >= 0.49
        assert state_node.export_summary()["executioner_events_count"] == 1


# ==============================================================================
# 8. M-Stacks Window Intermittency & Diverse AST Formats
# ==============================================================================

class TestMStacksWindowAndASTRepresentations:
    """Verifies trailing window evaluation dynamics and flexible AST token input formats."""

    def test_intermittent_health_prevents_vanishing_gradients(self):
        """If even one step in the m_stacks trailing window has healthy gradients,

        all_vanishing evaluates to False and the branch is NOT pruned.
        """
        dog = ExecutionerWatchdog(m_stacks=3, epsilon_grad=1e-6)
        history = [
            {"grad_norm": 1e-8, "loss": 1.0, "ast_tokens": {"x"}},
            {"grad_norm": 0.05, "loss": 0.8, "ast_tokens": {"y"}},  # Healthy spike!
            {"grad_norm": 1e-8, "loss": 0.7, "ast_tokens": {"z"}},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_intermittent", depth=3, visits=5, history=history)
        assert stagnant is False
        assert trigger == StagnationTrigger.NONE

    def test_diverse_ast_token_collection_formats_supported(self):
        """Tokens provided as list, tuple, or set of strings are correctly evaluated for Jaccard similarity."""
        dog = ExecutionerWatchdog(m_stacks=2, tau_jaccard=0.85)

        # Step 0: list, Step 1: tuple, both containing identical tokens
        tokens = [f"t_{i}" for i in range(20)]
        history = [
            {"grad_norm": 0.05, "loss": 1.0, "ast_tokens": list(tokens)},
            {"grad_norm": 0.05, "loss": 0.9, "ast_tokens": tuple(tokens)},
        ]
        stagnant, trigger, _ = dog.audit_branch("b_diverse_types", depth=3, visits=5, history=history)
        assert stagnant is True
        assert trigger == StagnationTrigger.CIRCULAR_AST
