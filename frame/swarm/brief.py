"""Context Firewall Generator (`frame brief`) & Asymmetric Swarm Spec Exporter (`frame swarm`)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from frame.brain.tree import CampaignTree
from frame.eval.funnel import FunnelEvaluator
from frame.schemas import BenchSpec, DirectiveTicket, OperatorRole


OPERATOR_INSTRUCTIONS: Dict[OperatorRole, str] = {
    OperatorRole.ERROR_SLICE: (
        "MECHANICAL OPERATOR: ERROR-SLICE TARGETING\n"
        "- Inspect the `Dev-Diagnostic Error Telemetry` section below.\n"
        "- Identify the specific feature regime, nonlinear interaction, or state disturbance with the highest residual error.\n"
        "- Implement a targeted feature transform, state basis expansion, or regime-conditional branch that directly addresses that worst-performing slice."
    ),
    OperatorRole.ABLATION_SIMPLIFY: (
        "MECHANICAL OPERATOR: ABLATION & COMPLEXITY PRUNING\n"
        "- Strip out dead code, redundant features, or over-parameterized layers from the parent code.\n"
        "- Reduce complexity or regularization overhead while improving or preserving the primary metric and tightening guardrail margins."
    ),
    OperatorRole.HYPERPARAM_SCHEDULE: (
        "MECHANICAL OPERATOR: OPTIMIZATION & REGULARIZATION TUNING\n"
        "- Keep the parent's structural representation fixed.\n"
        "- Mutate the optimization dynamics, regularization penalties (L2/shrinkage/clipping), tree depth/learning rate, or control gain scaling."
    ),
    OperatorRole.SPEC_IMPLEMENTATION: (
        "MECHANICAL OPERATOR: STRUCTURAL ALGORITHM SHIFT\n"
        "- If a Priority Human/Frontier Directive is attached below, implement its exact specification.\n"
        "- Otherwise, replace the core estimator/controller family in the parent code with a stronger model class suited to the task structure."
    ),
}


def generate_context_brief(
    workspace_root: Path,
    topic: str,
    parent_node_id: Optional[str] = None,
    operator_role: Optional[OperatorRole] = None,
) -> Dict[str, Any]:
    """Generates a leak-free Context Firewall Packet for a subagent worker."""
    # Ensure FunnelEvaluator has initialized node_0 if needed
    FunnelEvaluator(workspace_root, topic)
    tree = CampaignTree(workspace_root, topic)
    spec = BenchSpec.model_validate_json(
        (tree.bench_dir / "bench_spec.json").read_text(encoding="utf-8")
    )

    directive: Optional[DirectiveTicket] = None
    if parent_node_id is None or operator_role is None:
        sel_parent, sel_role, directive = tree.select_next_task()
        parent_node_id = parent_node_id or sel_parent
        operator_role = operator_role or sel_role

    parent_node = tree.nodes[parent_node_id]
    parent_code = tree.get_node_code_path(parent_node_id).read_text(encoding="utf-8")

    rejected_attempts: List[str] = []
    for cid in parent_node.children_ids:
        child = tree.nodes.get(cid)
        if child is None:
            continue
        rejected_attempts.append(
            f"- `{cid}` ({child.operator_role.value}): status=`{child.status.value}`, "
            f"pilot_delta=`{child.pilot_delta_vs_parent}`, title='{child.title}'"
        )

    guardrail_lines = "\n".join(
        f"- `{g.name} {g.op} {g.threshold}` ({g.description})" for g in spec.guardrails
    )

    directive_block = ""
    if directive is not None:
        directive_block = (
            f"\n## PRIORITY DIRECTIVE (`{directive.directive_id}` from `{directive.author}`)\n"
            f"**Mandatory Specification**: {directive.hypothesis_spec}\n"
            f"**When running `frame step`, pass**: `--directive {directive.directive_id}`\n"
        )

    markdown_brief = f"""# FRAME-AR v2 Context Firewall Brief (`{spec.benchmark_id}`)
- **Parent Node**: `{parent_node_id}` (Incumbent Single Best `M*`: `{tree.incumbent_node_id}`)
- **Assigned Mechanical Operator**: `{operator_role.value}`
- **Primary Metric**: `{spec.primary_metric_name}` (`{spec.direction.value}`, ROPE = `{spec.rope_epsilon}`)
{directive_block}
## 1. Operator Assignment
{OPERATOR_INSTRUCTIONS[operator_role]}

## 2. Hard Guardrails (Must Pass All {len(spec.guardrails)})
{guardrail_lines}

## 3. Dev-Diagnostic Error Telemetry (From Parent `{parent_node_id}`)
```json
{json.dumps(parent_node.dev_diagnostics, indent=2)}
```

## 4. Prior Attempts from `{parent_node_id}` (Do NOT Repeat Falsified Mutations)
{chr(10).join(rejected_attempts) if rejected_attempts else "- None yet."}

## 5. Working Parent Code (`candidate.py` from `{parent_node_id}`)
```python
{parent_code}
```

## 6. Mandatory Jail Worker Instructions
1. Modify `candidate.py` IN-PLACE in the current working directory to implement the `{operator_role.value}` mutation.
2. Preserve the exact function signature:
   `def run_experiment(task: dict, seed: int, mode: str, artifact_dir: Path) -> dict:`
3. Do NOT create new files (do NOT write `cand_{operator_role.value}.py`).
4. Do NOT run `python3`, `pytest`, or `frame step` via shell/command tools inside this directory — the host hypervisor will automatically execute Gate 0 (`smoke-check`), Gate 1, and Gate 2 on `candidate.py` the instant you exit.
5. Do NOT access any paths outside the current working directory.
"""
    return {
        "topic": tree.topic,
        "parent_node_id": parent_node_id,
        "operator_role": operator_role.value,
        "directive_id": directive.directive_id if directive else None,
        "brief_markdown": markdown_brief,
    }


def export_swarm_specs(workspace_root: Path, topic: str) -> List[Dict[str, Any]]:
    """Exports 4 context-isolated subagent specs across the 4 mechanical operators."""
    FunnelEvaluator(workspace_root, topic)
    tree = CampaignTree(workspace_root, topic)
    parent_id = tree.incumbent_node_id
    packets: List[Dict[str, Any]] = []
    for role in OperatorRole:
        packets.append(
            generate_context_brief(
                workspace_root=workspace_root,
                topic=topic,
                parent_node_id=parent_id,
                operator_role=role,
            )
        )
    out_file = tree.campaign_dir / "swarm_packets.json"
    out_file.write_text(json.dumps(packets, indent=2), encoding="utf-8")
    return packets
