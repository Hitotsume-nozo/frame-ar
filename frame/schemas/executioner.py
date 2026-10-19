"""Data schemas and contracts for The Executioner Watchdog (Milestone M4).

Defines stagnation triggers and structured executioner event records tracking
branch pruning and orthogonal warm restarts across recursive MCTS generations.
"""

from enum import Enum
import time
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class StagnationTrigger(str, Enum):
    """Enumeration of causes triggering Executioner intervention."""

    NONE = "NONE"
    VANISHING_GRADIENTS = "VANISHING_GRADIENTS"
    LOSS_PLATEAU = "LOSS_PLATEAU"
    CIRCULAR_AST = "CIRCULAR_AST"


class ExecutionerEvent(BaseModel):
    """Structured record of an Executioner audit, pruning, or warm-restart action."""

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )

    event_type: str = Field(
        default="ORTHOGONAL_WARM_RESTART",
        description="Type of executioner event: ORTHOGONAL_WARM_RESTART, PRUNED, EXEMPT",
    )
    branch_id: str = Field(
        ...,
        description="Unique branch identifier associated with event",
    )
    target_lane: Optional[str] = Field(
        default=None,
        description="Target cognitive lane after warm restart",
    )
    source_lane: Optional[str] = Field(
        default=None,
        description="Source cognitive lane prior to restart",
    )
    trigger: Union[StagnationTrigger, str] = Field(
        default=StagnationTrigger.NONE,
        description="Stagnation cause triggering the action",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Telemetry and diagnostic details",
    )
    injected_priors: List[str] = Field(
        default_factory=list,
        description="Orthogonal lane priors injected into restarted branch",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp (seconds) of event creation",
    )
