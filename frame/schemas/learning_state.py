"""Data contracts and Pydantic schemas for Swarm Learning State (Milestone M3).

Provides queryable persistent artifacts tracking hypothesis quality,
epistemic entropy, and test-time compute scaling across recursive MCTS generations.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrajectoryPoint(BaseModel):
    """Atomic evaluation record in the recursive MCTS learning trajectory."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True, populate_by_name=True)

    hypothesis_id: str = Field(..., description="Hypothesis identifier")
    step: int = Field(..., ge=1, description="1-indexed search budget step N")
    quality_score: float = Field(..., description="Empirical quality metric / reward")
    lane: str = Field(..., description="Cognitive lane: architecture, objectives, optimization, features")
    depth: int = Field(default=0, ge=0, description="MCTS tree recursion depth")
    upper_envelope_quality: float = Field(..., description="Running maximum of quality scores up to step")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic telemetry")

    @property
    def score(self) -> float:
        """Property alias for quality_score."""
        return self.quality_score

    @model_validator(mode="before")
    @classmethod
    def _normalize_score_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "score" in data and "quality_score" not in data:
                data["quality_score"] = data["score"]
        return data


class SwarmLearningStateNode(BaseModel):
    """Persistent Swarm Learning State Node tracking compounding compute scaling."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True, populate_by_name=True)

    node_id: str = Field(..., description="Unique state node identifier")
    total_budget_n: int = Field(default=0, ge=0, description="Total evaluations executed N")
    trajectories: List[TrajectoryPoint] = Field(default_factory=list, description="Chronological trajectory")
    active_branches: List[str] = Field(default_factory=list, description="Active MCTS branch identifiers")
    governor_event_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Information Governor audit log")
    executioner_event_logs: List[Dict[str, Any]] = Field(default_factory=list, description="Executioner event log")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Campaign metadata")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC creation timestamp",
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC last update timestamp",
    )

    def append_governor_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Appends a structured audit event log entry to governor_event_logs."""
        entry = {
            "event_type": str(event_type),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            **(dict(payload) if payload else {}),
        }
        self.governor_event_logs.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return entry

    def append_executioner_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Appends a structured watchdog/executioner event log entry to executioner_event_logs."""
        entry = {
            "event_type": str(event_type),
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            **(dict(payload) if payload else {}),
        }
        self.executioner_event_logs.append(entry)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return entry

    def log_governor_event(self, event_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Convenience keyword-argument helper for appending governor events."""
        return self.append_governor_event(event_type=event_type, payload=kwargs)

    def log_executioner_event(self, event_type: str, **kwargs: Any) -> Dict[str, Any]:
        """Convenience keyword-argument helper for appending executioner events."""
        return self.append_executioner_event(event_type=event_type, payload=kwargs)

    def record_evaluation(
        self,
        hypothesis_id: str,
        score: float,
        lane: str,
        depth: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        log_event: bool = False,
    ) -> TrajectoryPoint:
        """Records a candidate hypothesis evaluation, updates cumulative upper envelope, and increments budget."""
        self.total_budget_n += 1
        step = self.total_budget_n
        score_val = float(score)

        if self.trajectories:
            prev_env = self.trajectories[-1].upper_envelope_quality
            curr_env = max(prev_env, score_val)
        else:
            curr_env = score_val

        pt = TrajectoryPoint(
            hypothesis_id=str(hypothesis_id),
            step=step,
            quality_score=score_val,
            lane=str(lane),
            depth=int(depth),
            upper_envelope_quality=curr_env,
            metadata=dict(metadata) if metadata else {},
        )
        self.trajectories.append(pt)
        if log_event:
            self.append_executioner_event(
                "upper_envelope_update",
                {
                    "hypothesis_id": str(hypothesis_id),
                    "step": step,
                    "score": score_val,
                    "upper_envelope": curr_env,
                    "lane": str(lane),
                    "depth": int(depth),
                },
            )
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return pt

    def get_upper_envelope_curve(self) -> List[Tuple[int, float]]:
        """Returns upper envelope curve as list of (step, upper_envelope_quality) tuples."""
        return [(pt.step, pt.upper_envelope_quality) for pt in self.trajectories]

    def verify_monotonic_progression(
        self,
        tolerance: float = 1e-9,
        log_event: bool = False,
    ) -> Tuple[bool, float]:
        """Verifies monotonic non-decreasing progression of upper envelope and computes net gain.

        Monotonicity condition: E(i) >= E(i-1) - tolerance for all i in [2, N].
        Incremental net gain: E(N) - E(1) (returns 0.0 if N <= 1).
        """
        if len(self.trajectories) <= 1:
            if log_event:
                self.append_executioner_event(
                    "watchdog_monotonicity",
                    {
                        "is_monotonic": True,
                        "net_gain": 0.0,
                        "tolerance": tolerance,
                        "step_count": len(self.trajectories),
                    },
                )
            return True, 0.0

        curve = self.get_upper_envelope_curve()
        is_monotonic = True
        for i in range(1, len(curve)):
            if curve[i][1] < curve[i - 1][1] - tolerance:
                is_monotonic = False
                break

        gain = float(curve[-1][1] - curve[0][1])
        if math.isclose(gain, 0.0, abs_tol=tolerance):
            gain = 0.0
        else:
            gain = round(gain, 12)

        if log_event:
            self.append_executioner_event(
                "watchdog_monotonicity",
                {
                    "is_monotonic": is_monotonic,
                    "net_gain": gain,
                    "tolerance": tolerance,
                    "step_count": len(curve),
                },
            )

        return is_monotonic, gain


    def compute_epistemic_entropy(self, num_bins: int = 5, normalized: bool = True) -> float:
        """Calculates Shannon entropy across hypothesis quality scores.

        Partitions scores into num_bins distribution bins.
        Returns 0.0 for empty trajectories, single point, or all identical scores.
        Bounded in [0.0, 1.0] when normalized=True.
        """
        if len(self.trajectories) <= 1:
            return 0.0

        scores = [pt.quality_score for pt in self.trajectories]
        min_s = min(scores)
        max_s = max(scores)

        # Degenerate: all scores identical within numerical tolerance
        if math.isclose(min_s, max_s, abs_tol=1e-9):
            return 0.0

        if num_bins <= 1:
            return 0.0

        # Standard unit-interval domain if scores are within [0.0, 1.0],
        # otherwise dynamic range [min_s, max_s] to prevent index overflow
        if 0.0 <= min_s and max_s <= 1.0:
            low, high = 0.0, 1.0
        else:
            low, high = min_s, max_s

        width = (high - low) / num_bins
        if width <= 0.0:
            return 0.0

        counts = [0] * num_bins
        for s in scores:
            bin_idx = int((s - low) / width)
            if bin_idx >= num_bins:
                bin_idx = num_bins - 1
            elif bin_idx < 0:
                bin_idx = 0
            counts[bin_idx] += 1

        # Check if all counts fell into a single bin
        non_zero_bins = [c for c in counts if c > 0]
        if len(non_zero_bins) <= 1:
            return 0.0

        total = len(scores)
        h = 0.0
        for c in non_zero_bins:
            p = c / total
            h -= p * math.log(p)

        if normalized:
            max_h = math.log(num_bins)
            norm_val = h / max_h if max_h > 0 else 0.0
            return float(max(0.0, min(1.0, norm_val)))

        return float(max(0.0, h))

    def save_to_json(self, filepath: Union[str, Path]) -> str:
        """Atomically serializes state node to a JSON file."""
        path = Path(filepath).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".tmp.{os.getpid()}")

        data = self.model_dump(mode="json")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
        return str(path)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> "SwarmLearningStateNode":
        """Loads state node from a JSON file or raw JSON string."""
        if isinstance(filepath, str) and filepath.strip().startswith("{"):
            return cls.model_validate_json(filepath)
        path = Path(filepath).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"SwarmLearningStateNode file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save_to_file(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Saves state node to file (defaults to .frame/learning_state.json)."""
        target = filepath or os.path.join(".frame", "learning_state.json")
        return self.save_to_json(target)

    @classmethod
    def load_from_file(cls, filepath: Optional[Union[str, Path]] = None) -> "SwarmLearningStateNode":
        """Loads state node from file (defaults to .frame/learning_state.json)."""
        target = filepath or os.path.join(".frame", "learning_state.json")
        return cls.load_from_json(target)

    def export_summary(self) -> Dict[str, Any]:
        """Exports high-level observability metrics for CLI and reporting."""
        is_mono, gain = self.verify_monotonic_progression()
        return {
            "node_id": self.node_id,
            "total_budget_n": self.total_budget_n,
            "trajectory_count": len(self.trajectories),
            "upper_envelope_peak": self.trajectories[-1].upper_envelope_quality if self.trajectories else 0.0,
            "monotonic_progression": is_mono,
            "monotonic_gain": gain,
            "epistemic_entropy": self.compute_epistemic_entropy(),
            "active_branches_count": len(self.active_branches),
            "governor_events_count": len(self.governor_event_logs),
            "executioner_events_count": len(self.executioner_event_logs),
        }
