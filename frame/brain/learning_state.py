"""Persistent Swarm Learning State Node Engine for FRAME-AR (Milestone M3).

Provides queryable state node artifact tracking hypothesis quality,
epistemic entropy, and test-time compute scaling across recursive MCTS generations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from frame.schemas.learning_state import SwarmLearningStateNode, TrajectoryPoint

DEFAULT_LEARNING_STATE_PATH: str = os.path.join(".frame", "learning_state.json")


def save_learning_state(
    node: SwarmLearningStateNode,
    filepath: Optional[Union[str, Path]] = None,
) -> str:
    """Persists SwarmLearningStateNode to disk."""
    return node.save_to_file(filepath or DEFAULT_LEARNING_STATE_PATH)


def load_learning_state(
    filepath: Optional[Union[str, Path]] = None,
) -> SwarmLearningStateNode:
    """Loads SwarmLearningStateNode from disk."""
    return SwarmLearningStateNode.load_from_file(filepath or DEFAULT_LEARNING_STATE_PATH)


def save_to_file(
    node: SwarmLearningStateNode,
    filepath: Union[str, Path],
) -> str:
    """Saves node to specified filepath."""
    return node.save_to_file(filepath)


def load_from_file(
    filepath: Union[str, Path],
) -> SwarmLearningStateNode:
    """Loads node from specified filepath."""
    return SwarmLearningStateNode.load_from_file(filepath)


def export_scaling_data(
    node: SwarmLearningStateNode,
) -> Dict[str, Any]:
    """Exports test-time compute scaling curve data and telemetry."""
    is_mono, gain = node.verify_monotonic_progression()
    return {
        "node_id": node.node_id,
        "total_budget_n": node.total_budget_n,
        "upper_envelope_curve": node.get_upper_envelope_curve(),
        "is_monotonic": is_mono,
        "net_gain": gain,
        "epistemic_entropy": node.compute_epistemic_entropy(),
        "trajectories": [t.model_dump(mode="json") for t in node.trajectories],
    }


__all__ = [
    "TrajectoryPoint",
    "SwarmLearningStateNode",
    "DEFAULT_LEARNING_STATE_PATH",
    "save_learning_state",
    "load_learning_state",
    "save_to_file",
    "load_from_file",
    "export_scaling_data",
]
