"""Concrete Epistemic Brain & MCTS Engine for FRAME-AR.

Implements:
1. Adaptive UCT exploration schedule (c = 1.414 * 0.95^(t//25)).
2. Structural AST Falsification Memory (canonical normalization, O(1) filter, k-NN retrieval).
3. Recombination Operator Legality Grammar (C = A (x) B (x) D).
4. EpistemicMCTSBrain managing search tree selection, expansion, and backpropagation.
"""

import ast
from dataclasses import dataclass, field
import hashlib
import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from frame.schemas.hypothesis import TargetDirection
from frame.brain.state import MCTSNode, SearchStage, SearchAccountingLedger


# ==============================================================================
# 1. Adaptive UCT Schedule
# ==============================================================================

def compute_adaptive_uct_c(step_t: int, base_c: float = 1.41421356, decay_rate: float = 0.95, decay_interval: int = 25) -> float:
    """Computes adaptive UCT exploration constant c(t) = c_0 * (decay_rate)^(floor(t / decay_interval))."""
    exponent = step_t // decay_interval
    return float(base_c * (decay_rate ** exponent))


def compute_uct_score(
    node: MCTSNode,
    parent_visits: int,
    c_t: float,
) -> float:
    """Calculates Upper Confidence Bound for Trees (UCT) score:
    UCT(v) = Q_mean(v) + c(t) * sqrt(ln(1 + N_parent) / (1 + N_v))
    """
    if node.visits_n == 0:
        return math.inf  # Prioritize unvisited nodes

    q_bar = node.mean_q()
    exploration = c_t * math.sqrt(math.log(1.0 + parent_visits) / (1.0 + node.visits_n))
    return q_bar + exploration


# ==============================================================================
# 2. Structural AST Falsification Memory
# ==============================================================================

PRESERVED_NAMES = {
    "self", "cls", "range", "len", "print", "float", "int", "str", "bool",
    "list", "dict", "set", "tuple", "True", "False", "None",
    "torch", "nn", "F", "np", "numpy", "math", "sp", "sympy",
    "super", "isinstance", "hasattr", "getattr", "setattr",
    "zip", "enumerate", "reversed", "min", "max", "sum", "abs",
}


class CanonicalASTTransformer(ast.NodeTransformer):
    """Normalizes variable names to var_0, var_1, ... and strips docstrings."""

    def __init__(self):
        self.var_map: Dict[str, str] = {}
        self.counter = 0

    def _get_canonical_name(self, name: str) -> str:
        if name not in self.var_map:
            self.var_map[name] = f"var_{self.counter}"
            self.counter += 1
        return self.var_map[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        # Avoid renaming Python built-ins and framework namespaces
        if node.id in PRESERVED_NAMES:
            return node
        node.id = self._get_canonical_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg not in {"self", "cls"}:
            node.arg = self._get_canonical_name(node.arg)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        # Strip function docstrings
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
            if isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Strip class docstrings
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)):
            if isinstance(node.body[0].value.value, str):
                node.body = node.body[1:]
        self.generic_visit(node)
        return node


def compute_canonical_ast_fingerprint(code: str) -> str:
    """Parses code, canonicalizes AST, and generates cryptographic SHA-256 fingerprint."""
    try:
        parsed = ast.parse(code)
        transformer = CanonicalASTTransformer()
        transformed = transformer.visit(parsed)
        ast.fix_missing_locations(transformed)
        dumped = ast.dump(transformed, annotate_fields=False)
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    except Exception:
        # Fallback to normalized regex token string
        clean = re.sub(r"#.*", "", code)
        clean = re.sub(r"\s+", " ", clean).strip()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()


@dataclass
class FailureMemoryRecord:
    fingerprint: str
    failure_stage: SearchStage
    reason: str
    rebuttal_requirement: str
    code_tokens: Set[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    branch_id: Optional[str] = None
    depth: int = 0
    branch_path: Optional[str] = None


@dataclass
class PostMortemTelemetry:
    """Error telemetry extracted from a failed hypothesis or rejected candidate."""
    failure_stage: SearchStage
    failure_reason: str
    branch_id: Optional[str] = None
    depth: int = 0
    residual_stats: Dict[str, float] = field(default_factory=dict)
    error_mode: str = "generic_rejection"
    targeted_queries: List[str] = field(default_factory=list)
    rebuttal_req: str = ""


def synthesize_post_mortem_telemetry(
    stage: SearchStage,
    reason: str,
    branch_id: Optional[str] = None,
    depth: int = 0,
    residuals: Optional[np.ndarray] = None,
    target_direction: TargetDirection = TargetDirection.MINIMIZE,
) -> PostMortemTelemetry:
    """Analyzes failure telemetry and synthesizes targeted external search queries.
    
    Routes post-mortem findings directly back into Phase 1 (Swarm Thinking & Literature
    Discovery) so that subsequent mutations ingest external cures (e.g. gradient clipping,
    quantile scaling, interaction features, C2-smooth loss functions) rather than
    recycling flawed internal knowledge.
    """
    queries: List[str] = []
    error_mode = "generic_rejection"
    r_stats: Dict[str, float] = {}

    if residuals is not None and len(residuals) > 0:
        clean_res = residuals[np.isfinite(residuals)]
        if len(clean_res) > 0:
            mean_res = float(np.mean(clean_res))
            std_res = float(np.std(clean_res))
            r_stats = {
                "mean_residual": mean_res,
                "std_residual": std_res,
                "max_residual": float(np.max(np.abs(clean_res))),
            }
            if std_res > 1.5:
                queries.append("quantile transformer robust loss Huber for heavy tailed residuals")
                queries.append("quantile scaling and robust normalization tabular regression")
                queries.append("Huber smooth loss heavy tailed error distribution")
                queries.append("gradient boosted trees target encoding high residual variance")
        else:
            r_stats = {"mean_residual": 0.0, "std_residual": 0.0, "max_residual": 0.0}
            queries.append("numerical stability nan inf gradient handling loss function")

    reason_lower = reason.lower()

    if stage == SearchStage.POT_VERIFICATION or "pole" in reason_lower or "singularity" in reason_lower or "div_by_zero" in reason_lower:
        error_mode = "mathematical_singularity"
        queries.extend([
            "numerical stability epsilon non-zero denominator analytic activation functions",
            "smooth bounded activation function replacement sympy torch",
        ])
    elif stage == SearchStage.EXPERIMENTAL_DESIGN or "legalityviolation" in reason_lower:
        error_mode = "compositional_typing_violation"
        if "double_softmax" in reason_lower:
            queries.append("unnormalized logits CrossEntropyLoss softmax output layer classification")
        elif "non_smooth_hessian" in reason_lower:
            queries.append("smooth C2 loss functions second order optimization Sophia AdaHessian")
        elif "tensor_rank" in reason_lower:
            queries.append("Muon optimizer 2D parameter tensor rank split bias layernorm")
        elif "simplex" in reason_lower:
            queries.append("log probability simplex KL divergence activation log_softmax")
        else:
            queries.append("compatible loss function architecture optimizer typing PyTorch")
    elif stage == SearchStage.STAGE1_FUTILITY or "futility" in reason_lower:
        error_mode = "futility_stagnation"
        queries.extend([
            "gradient clipping and warmup schedule for tabular deep learning",
            "quantile scaling and robust normalization tabular regression",
            "residual connection feature interaction neural tabular models",
        ])
    elif stage == SearchStage.STAGE2_CONFIRMATORY or "confirmatory" in reason_lower or "null" in reason_lower:
        error_mode = "confirmatory_null_insufficient_effect"
        queries.extend([
            "interaction features and target encodings for tabular cross validation gain",
            "learning rate annealing and weight decay regularization to beat ROPE margin",
            "ensembled tree GBDT feature importance injection neural network",
        ])
    elif "dispersion" in reason_lower or "fold" in reason_lower:
        error_mode = "fold_divergence"
        queries.extend([
            "fold dispersion cross validation stability regularization weight decay",
            "adversarial validation covariate shift fold variance reduction",
        ])
    elif stage == SearchStage.IDEA_PROPOSAL or "ast" in reason_lower or "rebuttal" in reason_lower:
        error_mode = "falsification_memory_structural_block"
        queries.extend([
            "novel neural network architectural variations for tabular data",
            "alternative loss formulations avoiding known optimization pitfalls",
        ])
    else:
        queries.extend([
            f"techniques to resolve {reason[:40]} deep learning optimization",
            "state of the art tabular model architecture Kaggle winning solutions",
        ])

    # Deduplicate queries while preserving order
    seen = set()
    deduped_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            deduped_queries.append(q)

    return PostMortemTelemetry(
        failure_stage=stage,
        failure_reason=reason,
        branch_id=branch_id,
        depth=depth,
        residual_stats=r_stats,
        error_mode=error_mode,
        targeted_queries=deduped_queries,
    )


class StructuralFalsificationMemory:
    """Maintains falsification memory with O(1) hard AST filtering, branch isolation, and semantic k-NN retrieval."""

    def __init__(self):
        self.fatal_fingerprints: Dict[str, FailureMemoryRecord] = {}
        self.all_failures: List[FailureMemoryRecord] = []

    @staticmethod
    def _extract_tokens(code: str) -> Set[str]:
        try:
            parsed = ast.parse(code)
            t = CanonicalASTTransformer().visit(parsed)
            ast.fix_missing_locations(t)
            dump = ast.dump(t, annotate_fields=False)
            return set(re.findall(r"\w+", dump))
        except Exception:
            return set(re.findall(r"\w+", code.lower()))

    def record_fatal_failure(
        self,
        code: str,
        stage: SearchStage,
        reason: str,
        rebuttal_req: str = "Explain how architecture avoids zero-gradient / instability.",
        branch_id: Optional[str] = None,
        depth: int = 0,
        branch_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        state_node: Optional[Any] = None,
    ) -> str:
        fp = compute_canonical_ast_fingerprint(code)
        tokens = self._extract_tokens(code)
        meta = metadata or {}
        eff_path = branch_path or (f"root/{branch_id}" if branch_id else None)
        record = FailureMemoryRecord(
            fingerprint=fp,
            failure_stage=stage,
            reason=reason,
            rebuttal_requirement=rebuttal_req,
            code_tokens=tokens,
            metadata=meta,
            branch_id=branch_id,
            depth=depth,
            branch_path=eff_path,
        )
        self.fatal_fingerprints[fp] = record
        self.all_failures.append(record)

        if state_node is not None and hasattr(state_node, "executioner_event_logs") and isinstance(state_node.executioner_event_logs, list):
            state_node.executioner_event_logs.append({
                "event_type": "ast_falsification_recorded",
                "fingerprint": fp,
                "stage": stage.value if hasattr(stage, "value") else str(stage),
                "reason": reason,
                "branch_id": branch_id,
                "depth": depth,
            })

        return fp

    def check_hard_filter(
        self,
        code: str,
        branch_id: Optional[str] = None,
        state_node: Optional[Any] = None,
    ) -> Tuple[bool, Optional[FailureMemoryRecord]]:
        """O(1) lookup: returns (True, record) if exact canonical AST matches a fatal failure.
        
        Exact duplicate ASTs are blocked globally across all branches because
        provably invalid or fatal AST syntax/topologies cannot succeed in any branch.
        """
        fp = compute_canonical_ast_fingerprint(code)
        if fp in self.fatal_fingerprints:
            rec = self.fatal_fingerprints[fp]
            if state_node is not None and hasattr(state_node, "executioner_event_logs") and isinstance(state_node.executioner_event_logs, list):
                state_node.executioner_event_logs.append({
                    "event_type": "ast_falsification_blocked",
                    "fingerprint": fp,
                    "reason": rec.reason,
                    "branch_id": branch_id,
                })
            return True, rec
        return False, None


    @staticmethod
    def _is_branch_compatible(
        query_branch: Optional[str],
        record_branch: Optional[str],
        query_path: Optional[str] = None,
        record_path: Optional[str] = None,
    ) -> bool:
        """Determines if a failure record belongs to the same branch lineage or is global.
        
        - Global failures (record_branch is None) apply across all branches.
        - Unscoped queries (query_branch is None) match everything (backwards-compatible).
        - Disjoint branches return False, guaranteeing branch-scoped epistemic isolation.
        - Descendants within the same branch inherit ancestor failures and rebuttal constraints.
        - Sibling branches and ancestor nodes do not get poisoned by deeper child failures.
        """
        if record_branch is None:
            return True
        if query_branch is None:
            return True

        # Check branch lineage compatibility
        same_branch = (query_branch == record_branch)
        is_subbranch = query_branch.startswith(f"{record_branch}/")

        if not (same_branch or is_subbranch):
            # Completely different branch / parallel exploration lane
            return False

        # If both paths are provided, check hierarchical path lineage
        if query_path and record_path:
            q_norm = query_path.removeprefix("root/").strip("/")
            r_norm = record_path.removeprefix("root/").strip("/")

            # 1. Exact path match
            if q_norm == r_norm:
                return True

            # 2. Descendant query inherits ancestor failure
            if q_norm.startswith(f"{r_norm}/"):
                return True

            # 3. Ancestor query must NOT match deeper child failure
            if r_norm.startswith(f"{q_norm}/"):
                return False

            # 4. Check for depth specifiers (e.g. branch/d0 vs branch/d1)
            d_q_match = re.search(r"/d(\d+)$", q_norm)
            d_r_match = re.search(r"/d(\d+)$", r_norm)
            if d_q_match and d_r_match:
                d_q = int(d_q_match.group(1))
                d_r = int(d_r_match.group(1))
                q_base = q_norm[:d_q_match.start()]
                r_base = r_norm[:d_r_match.start()]
                if q_base == r_base or q_base.startswith(f"{r_base}/"):
                    return d_q >= d_r
                return False

            # 5. Check if they diverge into different sibling nodes under same parent
            q_parts = q_norm.split("/")
            r_parts = r_norm.split("/")
            common_len = min(len(q_parts), len(r_parts))
            for i in range(common_len):
                if q_parts[i] != r_parts[i]:
                    # Sibling divergence
                    return False

            # If q_norm has child nodes under r_norm
            return len(q_parts) >= len(r_parts)

        # Same branch or subbranch without conflicting hierarchical path
        return True

    def query_similar_failures(
        self,
        code: str,
        top_k: int = 3,
        similarity_threshold: float = 0.60,
        branch_id: Optional[str] = None,
        branch_path: Optional[str] = None,
    ) -> List[Tuple[FailureMemoryRecord, float]]:
        """Semantic Jaccard k-NN retrieval over known failures with branch isolation.
        
        Failures from unrelated parallel exploration lanes/branches are isolated and
        will not trigger spurious rebuttal requirements for sibling branches.
        """
        tokens = self._extract_tokens(code)
        if not tokens or not self.all_failures:
            return []

        results: List[Tuple[FailureMemoryRecord, float]] = []
        for rec in self.all_failures:
            if not self._is_branch_compatible(branch_id, rec.branch_id, branch_path, rec.branch_path):
                continue
            intersect = len(tokens.intersection(rec.code_tokens))
            union = len(tokens.union(rec.code_tokens))
            sim = (intersect / union) if union > 0 else 0.0
            if sim >= similarity_threshold:
                results.append((rec, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ==============================================================================
# 3. Recombination Operator Legality Grammar (C = A (x) B (x) D)
# ==============================================================================

class LegalityViolation(str):
    DOUBLE_SOFTMAX_CRUSH = "DOUBLE_SOFTMAX_CRUSH"
    NON_SMOOTH_HESSIAN = "NON_SMOOTH_HESSIAN"
    TENSOR_RANK_MISMATCH = "TENSOR_RANK_MISMATCH"
    SIMPLEX_DOMAIN_MISMATCH = "SIMPLEX_DOMAIN_MISMATCH"


class RecombinationLegalityGrammar:
    """Formal compositional typing rules for recombining Model (A), Loss (B), and Optimizer (D)."""

    SECOND_ORDER_OPTIMIZERS = {"sophia", "adahessian", "muon"}
    NON_SMOOTH_LOSSES = {"l1loss", "hingeloss", "smoothl1loss", "l1"}

    @classmethod
    def validate_recombination(
        cls,
        model_spec: Dict[str, Any],
        loss_spec: Dict[str, Any],
        optimizer_spec: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Validates typing constraints between Model A, Loss B, and Optimizer D.
        
        Returns:
            Tuple[is_legal, violation_code, explanation]
        """
        model_last_layer = str(model_spec.get("output_activation", "")).lower()
        loss_name = str(loss_spec.get("name", "")).lower()
        opt_name = str(optimizer_spec.get("name", "")).lower()

        # Rule 1: A -> B (Double-Softmax Gradient Crushing)
        # If loss is CrossEntropyLoss, model must output unnormalized logits
        if "cross_entropy" in loss_name or "nll" in loss_name:
            if "softmax" in model_last_layer:
                return (
                    False,
                    LegalityViolation.DOUBLE_SOFTMAX_CRUSH,
                    "Model terminates in explicit Softmax but Loss is CrossEntropyLoss; leads to double-softmax gradient vanishing.",
                )

        # Rule 2: B -> D (Non-Smooth Hessian with 2nd-Order Optimizers)
        # If optimizer requires 2nd derivatives, loss must be C^2 smooth
        if any(so in opt_name for so in cls.SECOND_ORDER_OPTIMIZERS):
            if any(nsl in loss_name for nsl in cls.NON_SMOOTH_LOSSES):
                return (
                    False,
                    LegalityViolation.NON_SMOOTH_HESSIAN,
                    f"Optimizer {opt_name} requires C^2 Hessian estimates, but loss {loss_name} has discontinuous/non-smooth derivatives.",
                )

        # Rule 3: A -> D (Muon Newton-Schulz Tensor Rank Restriction)
        # Muon applies strictly to >= 2D matrix weights. 1D parameters must be sub-partitioned into AdamW.
        if "muon" in opt_name:
            applies_to_1d = optimizer_spec.get("includes_1d_params", False)
            if applies_to_1d:
                return (
                    False,
                    LegalityViolation.TENSOR_RANK_MISMATCH,
                    "Muon Newton-Schulz orthogonalization is mathematically undefined on 1D vectors (biases/layernorms).",
                )

        # Rule 4: A -> B (Simplex Domain Mismatch)
        # If loss operates on the simplex or log-simplex, outputs must match
        if "kldiv" in loss_name or "kl_div" in loss_name:
            if model_last_layer not in {"log_softmax", "logsoftmax"}:
                return (
                    False,
                    LegalityViolation.SIMPLEX_DOMAIN_MISMATCH,
                    f"Loss {loss_name} expects inputs on the log-probability simplex (log_softmax), but model output activation is {model_last_layer or 'unnormalized'}.",
                )
        elif "wasserstein" in loss_name or "earth_mover" in loss_name:
            if "softmax" not in model_last_layer:
                return (
                    False,
                    LegalityViolation.SIMPLEX_DOMAIN_MISMATCH,
                    f"Loss {loss_name} operates on probability distributions (simplex), but model output activation is {model_last_layer or 'unnormalized'}.",
                )

        return True, None, None


# ==============================================================================
# 4. Epistemic MCTS Brain
# ==============================================================================

class EpistemicMCTSBrain:
    """Epistemic MCTS Engine orchestrating tree search with adaptive UCT and falsification memory."""

    def __init__(self, ledger: Optional[SearchAccountingLedger] = None):
        self.nodes: Dict[str, MCTSNode] = {}
        self.root_id: Optional[str] = None
        self.memory = StructuralFalsificationMemory()
        self.ledger = ledger or SearchAccountingLedger()
        self.current_step = 0

    def create_root_node(
        self,
        hypothesis_id: str,
        code: str,
        branch_id: Optional[str] = "root",
    ) -> MCTSNode:
        root = MCTSNode(
            node_id="root",
            hypothesis_id=hypothesis_id,
            stage=SearchStage.IDEA_PROPOSAL,
            ast_fingerprint=compute_canonical_ast_fingerprint(code),
            depth=0,
            branch_id=branch_id,
            metadata={"branch_path": "root"},
        )
        self.nodes["root"] = root
        self.root_id = "root"
        self.ledger.record_entry(SearchStage.IDEA_PROPOSAL)
        return root

    def select_best_uct_child(self, parent_node: MCTSNode) -> Optional[MCTSNode]:
        """Selects child node maximizing adaptive UCT."""
        if not parent_node.children_ids:
            return None

        c_t = compute_adaptive_uct_c(self.current_step)
        best_child: Optional[MCTSNode] = None
        best_score = -math.inf

        for child_id in parent_node.children_ids:
            child = self.nodes.get(child_id)
            if child is None:
                continue
            score = compute_uct_score(child, parent_node.visits_n, c_t)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def expand_child(
        self,
        parent_id: str,
        child_id: str,
        hypothesis_id: str,
        code: str,
        rebuttal: Optional[str] = None,
        model_spec: Optional[Dict[str, Any]] = None,
        loss_spec: Optional[Dict[str, Any]] = None,
        optimizer_spec: Optional[Dict[str, Any]] = None,
        branch_id: Optional[str] = None,
        depth: Optional[int] = None,
        branch_path: Optional[str] = None,
    ) -> Tuple[bool, Optional[MCTSNode], Optional[str]]:
        """Expands node after subjecting it to:
        1. O(1) Hard AST Filter against known fatal failures (global duplicate blocking).
        2. Semantic k-NN memory retrieval requiring structured rebuttal (branch-isolated).
        3. Recombination Legality Grammar.
        """
        self.current_step += 1
        self.ledger.record_entry(SearchStage.IDEA_PROPOSAL)

        parent_node = self.nodes.get(parent_id)
        parent_depth = getattr(parent_node, "depth", 0) if parent_node else 0
        node_depth = depth if depth is not None else (parent_depth + 1 if parent_node else 1)

        parent_branch = getattr(parent_node, "branch_id", None) if parent_node else None
        node_branch = branch_id if branch_id is not None else parent_branch

        parent_path = parent_node.metadata.get("branch_path") if (parent_node and hasattr(parent_node, "metadata")) else None
        if branch_path:
            node_path = branch_path
        elif parent_path:
            if parent_path == "root" and node_branch and node_branch != "root":
                node_path = f"root/{node_branch}/{child_id}"
            else:
                node_path = f"{parent_path}/{child_id}"
        elif node_branch:
            node_path = f"root/{node_branch}/{child_id}"
        else:
            node_path = None

        # 1. Hard AST Filter (global duplicate blocking across all branches)
        is_fatal, fail_rec = self.memory.check_hard_filter(code, branch_id=node_branch)
        if is_fatal and fail_rec is not None:
            self.ledger.record_rejection(SearchStage.IDEA_PROPOSAL, "FATAL_AST_MATCH")
            return False, None, f"Rejected: Matches fatal failure {fail_rec.fingerprint[:8]} ({fail_rec.reason})"

        # 2. Semantic k-NN Retrieval & Rebuttal Check (branch-isolated)
        similar_fails = self.memory.query_similar_failures(
            code,
            similarity_threshold=0.65,
            branch_id=node_branch,
            branch_path=node_path,
        )
        if similar_fails and (rebuttal is None or len(rebuttal.strip()) < 10):
            top_rec, sim = similar_fails[0]
            self.ledger.record_rejection(SearchStage.IDEA_PROPOSAL, "MISSING_EVIDENTIARY_REBUTTAL")
            return (
                False,
                None,
                f"Rejected: Similar to failure {top_rec.fingerprint[:8]} ({sim:.1%} similarity). Structured rebuttal required: {top_rec.rebuttal_requirement}",
            )

        # 3. Legality Grammar
        if model_spec and loss_spec and optimizer_spec:
            is_legal, code_viol, explanation = RecombinationLegalityGrammar.validate_recombination(
                model_spec, loss_spec, optimizer_spec
            )
            if not is_legal:
                self.ledger.record_rejection(SearchStage.IDEA_PROPOSAL, f"LEGALITY_VIOLATION:{code_viol}")
                return False, None, f"Rejected by Legality Grammar ({code_viol}): {explanation}"

        # Passed all gates: create child node with depth and branch provenance
        child_node = MCTSNode(
            node_id=child_id,
            parent_id=parent_id,
            hypothesis_id=hypothesis_id,
            stage=SearchStage.POT_VERIFICATION,
            ast_fingerprint=compute_canonical_ast_fingerprint(code),
            rebuttal=rebuttal,
            depth=node_depth,
            branch_id=node_branch,
            metadata={"branch_path": node_path} if node_path else {},
        )
        self.nodes[child_id] = child_node
        if parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(child_id)

        self.ledger.record_pass(SearchStage.IDEA_PROPOSAL)
        return True, child_node, None

    def compute_oracle_reward(
        self,
        decision: Any,
        baseline_score: Optional[float] = None,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
        rope_epsilon: Optional[float] = None,
    ) -> float:
        """Computes verifiable oracle reward feedback from statistical verification screening.
        
        Connects MCTS epistemic backpropagation directly to verifiable empirical evidence:
        - Stage 1 Futility pruning: 0.0 reward (penalized, prune branch).
        - Stage 2 Confirmatory Discovery: Positive scalar combining normalized improvement
          beyond ROPE margin, statistical significance scaling (log(alpha_t / p_rope)),
          and a discrete discovery milestone bonus.
        - Stage 2 Sub-threshold Null: Heavily attenuated partial credit (capped at 0.05).
        """
        if hasattr(decision, "is_discovery") and hasattr(decision, "paired_metrics"):
            # StatisticalVerificationReport schema
            proceed = bool(decision.is_discovery)
            mean_diff = float(decision.paired_metrics.mean_difference)
            p_rope = float(decision.p_value_rope)
            alpha_t = float(decision.allocated_alpha_t)
            r_eps = float(rope_epsilon if rope_epsilon is not None else decision.rope_epsilon)
            stage = "stage_2_confirmatory"
            if baseline_score is None:
                baseline_score = float(decision.mean_baseline)
        elif hasattr(decision, "proceed"):
            # ScreeningDecision schema
            proceed = bool(decision.proceed)
            mean_diff = float(getattr(decision, "mean_difference", 0.0))
            details = getattr(decision, "details", {})
            stage = str(getattr(decision, "stage", ""))
            p_rope = float(details.get("p_value_rope", 1.0))
            alpha_t = float(details.get("allocated_alpha_t", details.get("alpha_t", 0.010937)))
            r_eps = float(rope_epsilon if rope_epsilon is not None else details.get("rope_epsilon", 0.01))
        elif isinstance(decision, dict):
            # Dict schema (e.g. AgySwarmCoordinator evaluate_candidate_hypothesis or JSON)
            proceed = bool(decision.get("proceed", decision.get("is_discovery", decision.get("passed", False))))
            mean_diff = float(decision.get("mean_difference", 0.0))
            details = decision.get("details", {})
            stage = str(decision.get("stage", ""))
            p_rope = float(details.get("p_value_rope", decision.get("p_value_rope", 1.0)))
            alpha_t = float(details.get("allocated_alpha_t", details.get("alpha_t", decision.get("allocated_alpha_t", 0.010937))))
            r_eps = float(rope_epsilon if rope_epsilon is not None else details.get("rope_epsilon", decision.get("rope_epsilon", 0.01)))
        else:
            proceed = False
            mean_diff = 0.0
            stage = ""
            p_rope = 1.0
            alpha_t = 0.010937
            r_eps = 0.01

        # Futility pruned at Stage 1
        if not proceed and ("futility" in stage or "stage_1" in stage):
            return 0.0

        base_val = 1.0 if baseline_score is None else float(baseline_score)
        denom = max(1e-6, abs(base_val))

        if proceed:
            # Verified Discovery under streaming FDR (LORD-3)
            # Net margin beyond ROPE
            net_gain = max(0.0, mean_diff - r_eps)
            rel_gain = net_gain / denom

            # Significance boost based on alpha_t / p_rope ratio
            ratio = max(1.0, alpha_t / max(p_rope, 1e-15))
            sig_mult = 1.0 + min(4.0, math.log10(ratio))

            # Discovery bonus = 1.0 for crossing the streaming FDR discovery threshold
            # plus effect size scaled by statistical certainty
            reward = 1.0 + (rel_gain * 10.0 * sig_mult)
            return float(reward)
        else:
            # Failed confirmatory screening (Null hypothesis not rejected)
            if mean_diff > 0:
                rel_gain = mean_diff / denom
                return float(min(0.05, rel_gain * 0.1))
            return 0.0

    def backpropagate(
        self,
        node_id: str,
        reward: Optional[float] = None,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
        statistical_decision: Optional[Any] = None,
        baseline_score: Optional[float] = None,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
        rope_epsilon: Optional[float] = None,
    ) -> float:
        """Propagates reward and compute resources up the search tree.
        
        If statistical_decision is provided, computes oracle reward feedback directly
        from statistical screening outcome (Two-stage ROPE + LORD FDR).
        """
        if reward is None:
            if statistical_decision is not None:
                reward = self.compute_oracle_reward(
                    decision=statistical_decision,
                    baseline_score=baseline_score,
                    target_direction=target_direction,
                    rope_epsilon=rope_epsilon,
                )
            else:
                reward = 0.0

        curr: Optional[MCTSNode] = self.nodes.get(node_id)
        while curr is not None:
            curr.visits_n += 1
            curr.value_q += reward
            curr.gpu_hours_spent += gpu_hours
            curr.cpu_hours_spent += cpu_hours
            curr = self.nodes.get(curr.parent_id) if curr.parent_id else None

        return reward

    def backpropagate_statistical_outcome(
        self,
        node_id: str,
        screening_decision: Any,
        baseline_score: float = 1.0,
        target_direction: TargetDirection = TargetDirection.MAXIMIZE,
        rope_epsilon: Optional[float] = None,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
    ) -> float:
        """Convenience method to compute oracle reward and backpropagate in one step."""
        return self.backpropagate(
            node_id=node_id,
            reward=None,
            gpu_hours=gpu_hours,
            cpu_hours=cpu_hours,
            statistical_decision=screening_decision,
            baseline_score=baseline_score,
            target_direction=target_direction,
            rope_epsilon=rope_epsilon,
        )
