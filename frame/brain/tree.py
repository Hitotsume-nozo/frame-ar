"""MCTS Campaign Tree, OperatorASTMemory, and Asynchronous Priority Directive Queue."""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
from typing import Any, Dict, List, Optional, Tuple

from frame.schemas import (
    BenchSpec,
    BenchVerificationReport,
    DirectiveTicket,
    EdgeType,
    NodeRecord,
    NodeStatus,
    OperatorRole,
)
from frame.utils.canonical_json import compute_file_sha256


class OperatorASTMemory:
    """Extracts operator & API call signatures, stripping variable names and AST node boilerplate."""

    @staticmethod
    def fingerprint_code(code_text: str) -> Tuple[str, List[str]]:
        tree = ast.parse(code_text)
        op_tokens: List[str] = []
        structural_items: List[str] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.operator, ast.unaryop, ast.cmpop, ast.boolop)):
                tok = f"OP:{type(node).__name__}"
                op_tokens.append(tok)
                structural_items.append(tok)
            elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try, ast.ListComp)):
                tok = f"CTRL:{type(node).__name__}"
                op_tokens.append(tok)
                structural_items.append(tok)
            elif isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name:
                    tok = f"CALL:{func_name}"
                    op_tokens.append(tok)
                    structural_items.append(tok)
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                structural_items.append(f"NUM:{round(float(node.value), 6)}")

        digest = hashlib.sha256("|".join(structural_items).encode("utf-8")).hexdigest()
        unique_sorted_ops = sorted(set(op_tokens))
        return digest, unique_sorted_ops

    @staticmethod
    def jaccard_similarity(tokens_a: List[str], tokens_b: List[str]) -> float:
        set_a, set_b = set(tokens_a), set(tokens_b)
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return float(len(set_a & set_b) / len(set_a | set_b))


OPERATOR_CYCLE: List[OperatorRole] = [
    OperatorRole.ERROR_SLICE,
    OperatorRole.SPEC_IMPLEMENTATION,
    OperatorRole.HYPERPARAM_SCHEDULE,
    OperatorRole.ABLATION_SIMPLIFY,
]


class CampaignTree:
    """Manages immutable node snapshots, single-model incumbent M*, UCT selection, and directives."""

    def __init__(self, workspace_root: Path, topic: str) -> None:
        self.workspace_root = workspace_root.resolve()
        self.topic = topic.strip().lower().replace(" ", "_")
        self.bench_dir = self.workspace_root / ".frame" / "benchmarks" / self.topic
        self.campaign_dir = self.workspace_root / ".frame" / "campaigns" / self.topic
        self.nodes_dir = self.campaign_dir / "nodes"
        self.directives_dir = self.campaign_dir / "directives"
        self.active_dir = self.campaign_dir / "active"
        self.state_path = self.campaign_dir / "tree_state.json"

        for d in (self.nodes_dir, self.directives_dir, self.active_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.incumbent_node_id: str = "node_0"
        self.best_blend_node_id: Optional[str] = None
        self.next_node_index: int = 1
        self.total_steps: int = 0
        self.nodes: Dict[str, NodeRecord] = {}
        self.branch_consecutive_failures: Dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.incumbent_node_id = str(data.get("incumbent_node_id", "node_0"))
        self.best_blend_node_id = data.get("best_blend_node_id")
        self.next_node_index = int(data.get("next_node_index", 1))
        self.branch_consecutive_failures = {
            str(k): int(v) for k, v in data.get("branch_consecutive_failures", {}).items()
        }
        self.nodes = {
            nid: NodeRecord.model_validate(n_dict)
            for nid, n_dict in data.get("nodes", {}).items()
        }
        if "total_steps" in data:
            self.total_steps = int(data["total_steps"])
        else:
            self.total_steps = sum(
                1 for n in self.nodes.values()
                if n.node_id != "node_0" and n.status not in (NodeStatus.BUGGY, NodeStatus.ABORTED)
            )

    def save(self) -> None:
        payload = {
            "topic": self.topic,
            "incumbent_node_id": self.incumbent_node_id,
            "best_blend_node_id": self.best_blend_node_id,
            "next_node_index": self.next_node_index,
            "total_steps": self.total_steps,
            "branch_consecutive_failures": self.branch_consecutive_failures,
            "nodes": {nid: n.model_dump(mode="json") for nid, n in self.nodes.items()},
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def initialize_from_benchmark(
        self,
        spec: BenchSpec,
        report: BenchVerificationReport,
        dev_diagnostics: Dict[str, Any],
    ) -> NodeRecord:
        """Creates `node_0` from the verified `baseline.py` snapshot."""
        if "node_0" in self.nodes:
            return self.nodes["node_0"]

        base_path = self.bench_dir / "baseline.py"
        code_txt = base_path.read_text(encoding="utf-8")
        fp, ops = OperatorASTMemory.fingerprint_code(code_txt)

        root = NodeRecord(
            node_id="node_0",
            parent_id=None,
            branch_id="root",
            depth=0,
            title="Verified Phase 0 Baseline",
            status=NodeStatus.ROOT_BASELINE,
            code_sha256=compute_file_sha256(base_path),
            ast_fingerprint=fp,
            operator_tokens=ops,
            pilot_scores=report.baseline_pilot_scores,
            pilot_primary_mean=report.baseline_mean,
            pilot_primary_iqm=report.baseline_iqm,
            pilot_delta_vs_parent=0.0,
            guardrails_passed=True,
            dev_diagnostics=dev_diagnostics,
            promoted=True,
            visits=1,
            total_reward=1.0,
        )
        node_folder = self.nodes_dir / "node_0"
        node_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base_path, node_folder / "candidate.py")
        (node_folder / "node_record.json").write_text(
            root.model_dump_json(indent=2), encoding="utf-8"
        )
        self.nodes["node_0"] = root
        self.incumbent_node_id = "node_0"
        self.save()
        return root

    def allocate_next_node_id(self) -> str:
        nid = f"node_{self.next_node_index}"
        self.next_node_index += 1
        self.save()
        return nid

    def get_node_code_path(self, node_id: str) -> Path:
        return self.nodes_dir / node_id / "candidate.py"

    def check_duplicate_ast(self, code_text: str) -> Tuple[bool, Optional[NodeRecord], str, List[str]]:
        try:
            fp, ops = OperatorASTMemory.fingerprint_code(code_text)
        except SyntaxError:
            return False, None, "syntax_error", []

        for existing in self.nodes.values():
            if existing.ast_fingerprint == fp and existing.status != NodeStatus.ABORTED:
                return True, existing, fp, ops
        return False, None, fp, ops

    def record_node(self, node: NodeRecord, candidate_source_path: Path, reward: float) -> None:
        node_folder = self.nodes_dir / node.node_id
        node_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_source_path, node_folder / "candidate.py")
        (node_folder / "node_record.json").write_text(
            node.model_dump_json(indent=2), encoding="utf-8"
        )
        self.nodes[node.node_id] = node

        if node.parent_id and node.parent_id in self.nodes:
            parent = self.nodes[node.parent_id]
            if node.node_id not in parent.children_ids:
                parent.children_ids.append(node.node_id)
            parent_rec_path = self.nodes_dir / parent.node_id / "node_record.json"
            if parent_rec_path.parent.is_dir():
                parent_rec_path.write_text(parent.model_dump_json(indent=2), encoding="utf-8")

        if node.status not in (NodeStatus.BUGGY, NodeStatus.ABORTED):
            self.total_steps += 1
            curr_id: Optional[str] = node.node_id
            while curr_id and curr_id in self.nodes:
                curr = self.nodes[curr_id]
                if hasattr(curr, "visits_n"):
                    curr.visits_n += 1
                    curr.value_q += float(reward)
                elif hasattr(curr, "visits"):
                    curr.visits += 1
                    curr.total_reward += float(reward)
                rec_p = self.nodes_dir / curr_id / "node_record.json"
                if rec_p.parent.is_dir():
                    rec_p.write_text(curr.model_dump_json(indent=2), encoding="utf-8")
                curr_id = curr.parent_id

        # Track branch stagnation & update Incumbent Best Single Model (M*)
        if node.promoted:
            self.branch_consecutive_failures[node.branch_id] = 0
            if node.edge_type == EdgeType.BLEND_ENSEMBLE:
                self.best_blend_node_id = node.node_id
            else:
                self.incumbent_node_id = node.node_id
        elif node.status != NodeStatus.ABORTED:
            prev_f = self.branch_consecutive_failures.get(node.branch_id, 0)
            self.branch_consecutive_failures[node.branch_id] = prev_f + 1

        if node.directive_id:
            self.mark_directive_consumed(node.directive_id)

        self.save()

    # ==========================================================================
    # Priority Directive Queue (`frame inject`) & Process Preemption (`frame abort`)
    # ==========================================================================
    def enqueue_directive(
        self,
        parent_node_id: str,
        hypothesis_spec: str,
        operator_role: OperatorRole = OperatorRole.SPEC_IMPLEMENTATION,
        priority: int = 0,
        author: str = "human_or_frontier",
    ) -> DirectiveTicket:
        if parent_node_id not in self.nodes:
            raise KeyError(f"Cannot inject onto unknown node '{parent_node_id}'.")
        existing = list(self.directives_dir.glob("dir_*.json"))
        dir_id = f"dir_{len(existing) + 1:03d}"
        ticket = DirectiveTicket(
            directive_id=dir_id,
            parent_node_id=parent_node_id,
            operator_role=operator_role,
            priority=priority,
            hypothesis_spec=hypothesis_spec,
            author=author,
        )
        (self.directives_dir / f"{dir_id}.json").write_text(
            ticket.model_dump_json(indent=2), encoding="utf-8"
        )
        return ticket

    def get_queued_directives(self) -> List[DirectiveTicket]:
        return self.list_directives(unconsumed_only=True)

    def list_directives(self, unconsumed_only: bool = True) -> List[DirectiveTicket]:
        tickets: List[DirectiveTicket] = []
        for p in sorted(self.directives_dir.glob("dir_*.json")):
            t = DirectiveTicket.model_validate_json(p.read_text(encoding="utf-8"))
            if not unconsumed_only or t.status == "queued":
                tickets.append(t)
        tickets.sort(key=lambda x: (x.priority, x.created_at))
        return tickets

    def mark_directive_consumed(self, directive_id: str) -> None:
        p = self.directives_dir / f"{directive_id}.json"
        if p.exists():
            t = DirectiveTicket.model_validate_json(p.read_text(encoding="utf-8"))
            t.status = "consumed"
            p.write_text(t.model_dump_json(indent=2), encoding="utf-8")

    def abort_running_node(
        self,
        node_id: str,
        inject_replacement_spec: Optional[str] = None,
        fallback_parent_id: Optional[str] = None,
        replacement_spec: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sends SIGTERM to an active worker PID and optionally queues a priority replacement spec."""
        eff_spec = inject_replacement_spec or replacement_spec
        pid_file = self.active_dir / f"{node_id}.pid"
        aborted_pid: Optional[int] = None
        if pid_file.exists():
            aborted_pid = int(pid_file.read_text(encoding="utf-8").strip())
            (self.active_dir / f"{node_id}.aborted").write_text("1", encoding="utf-8")
            try:
                os.kill(aborted_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        ticket: Optional[DirectiveTicket] = None
        if eff_spec:
            parent_id = fallback_parent_id or self.incumbent_node_id
            if node_id in self.nodes and self.nodes[node_id].parent_id:
                parent_id = self.nodes[node_id].parent_id  # type: ignore[assignment]
            ticket = self.enqueue_directive(
                parent_node_id=parent_id,
                hypothesis_spec=eff_spec,
                priority=0,
                author="abort_preemption",
            )
        return {
            "aborted_node_id": node_id,
            "signaled_pid": aborted_pid,
            "queued_directive_id": ticket.directive_id if ticket else None,
        }

    inject_directive = enqueue_directive

    def get_steps_since_last_promotion(self) -> int:
        """Returns the number of evaluated non-root nodes since the last PROMOTED_DISCOVERY."""
        non_root = [
            n for n in self.nodes.values()
            if n.node_id != "node_0" and n.status not in (NodeStatus.ROOT, NodeStatus.ROOT_BASELINE, NodeStatus.ABORTED)
        ]
        # Sort chronologically by creation timestamp or node index
        non_root.sort(
            key=lambda x: (
                int(x.node_id.split("_")[1]) if "_" in x.node_id and x.node_id.split("_")[1].isdigit() else 0,
                getattr(x, "created_at", ""),
            )
        )
        stagnant = 0
        for n in reversed(non_root):
            if n.promoted or n.status == NodeStatus.PROMOTED_DISCOVERY:
                break
            stagnant += 1
        return stagnant

    def select_next_task(
        self, rope_epsilon: float = 0.5, c_uct: Optional[float] = None
    ) -> Tuple[str, OperatorRole, Optional[DirectiveTicket]]:
        """Selects (parent_node_id, operator_role, directive_ticket) with Superseded-Ancestor Retirement."""
        pending = self.list_directives(unconsumed_only=True)
        if pending:
            top = pending[0]
            return top.parent_node_id, top.operator_role, top

        if rope_epsilon == 0.5 and (self.bench_dir / "bench_spec.json").is_file():
            try:
                spec_data = json.loads((self.bench_dir / "bench_spec.json").read_text(encoding="utf-8"))
                rope_epsilon = float(spec_data.get("rope_epsilon", rope_epsilon))
            except Exception:
                pass

        incumbent = self.nodes.get(self.incumbent_node_id)
        inc_score = (
            (incumbent.pilot_primary_iqm if incumbent.pilot_primary_iqm is not None else incumbent.pilot_primary_mean)
            if incumbent else None
        )

        c_base = c_uct if c_uct is not None else 1.4142
        c_t = c_base * (0.95 ** (self.total_steps // 10))
        best_node_id = self.incumbent_node_id
        best_uct = -math.inf

        def _get_visits(n: Any) -> int:
            return int(getattr(n, "visits_n", getattr(n, "visits", 1)))

        def _get_mean_q(n: Any) -> float:
            v = max(1, _get_visits(n))
            q = float(getattr(n, "value_q", getattr(n, "total_reward", 0.0)))
            return q / v

        total_n = max(1, sum(_get_visits(n) for n in self.nodes.values()))

        for nid, node in self.nodes.items():
            status_str = node.status.value if hasattr(node.status, "value") else str(node.status)
            if status_str not in ("root", "root_baseline", "promoted_discovery"):
                continue

            # Check if this node has already spawned a promoted child AND trails Incumbent M* by > 2 * ROPE
            has_promoted_child = any(
                cid in self.nodes and (
                    self.nodes[cid].promoted
                    or (
                        self.nodes[cid].status.value
                        if hasattr(self.nodes[cid].status, "value")
                        else str(self.nodes[cid].status)
                    ) == "promoted_discovery"
                )
                for cid in node.children_ids
            )
            node_score = node.pilot_primary_iqm if node.pilot_primary_iqm is not None else node.pilot_primary_mean
            if (
                nid != self.incumbent_node_id
                and has_promoted_child
                and inc_score is not None
                and node_score is not None
                and abs(float(inc_score) - float(node_score)) > 2.0 * rope_epsilon
            ):
                # Retire superseded ancestor so search focuses on the active frontier
                continue

            # Count consecutive failed children on this node
            failed_children = sum(
                1 for cid in node.children_ids
                if cid in self.nodes and not self.nodes[cid].promoted
            )
            stagnation_penalty = 0.45 * max(0, failed_children - 1)
            if failed_children >= 4:
                # Full operator cycle exhausted without promotion
                stagnation_penalty += 2.5

            q_bar = _get_mean_q(node)
            explore = c_t * math.sqrt(math.log(1.0 + total_n) / (1.0 + _get_visits(node)))
            uct = q_bar + explore - stagnation_penalty
            if uct > best_uct:
                best_uct = uct
                best_node_id = nid

        if best_node_id not in self.nodes:
            if "node_0" in self.nodes:
                best_node_id = "node_0"
            elif self.nodes:
                best_node_id = next(iter(self.nodes))
            else:
                return "node_0", OPERATOR_CYCLE[0], None

        chosen_node = self.nodes[best_node_id]
        role_idx = len(chosen_node.children_ids) % len(OPERATOR_CYCLE)
        return best_node_id, OPERATOR_CYCLE[role_idx], None
