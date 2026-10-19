"""MCTS State Schema and Search-Level Accounting Ledger for FRAME-AR.

Implements:
1. SearchStage lifecycle enum.
2. MCTSNode schema with visit accounting, value Q, and resource tracking.
3. SearchAccountingLedger tracking stage-by-stage throughput, pass rates, and GPU hours.
"""

import json
import os
from pathlib import Path
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

DEFAULT_ACCOUNTING_LEDGER_PATH: str = os.path.join(".frame", "accounting_ledger.json")


class SearchStage(str, Enum):
    IDEA_PROPOSAL = "idea_proposal"
    POT_VERIFICATION = "pot_verification"
    EXPERIMENTAL_DESIGN = "experimental_design"
    RED_TEAM_AUDIT = "red_team_audit"
    MICRO_PASS = "micro_pass"
    STAGE1_FUTILITY = "stage1_futility"
    STAGE2_CONFIRMATORY = "stage2_confirmatory"
    REPLAY_VERIFICATION = "replay_verification"
    LORD_VERIFICATION = "lord_verification"
    CONFIRMED_DISCOVERY = "confirmed_discovery"
    REJECTED = "rejected"


class MCTSNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    hypothesis_id: Optional[str] = None
    stage: SearchStage = SearchStage.IDEA_PROPOSAL
    visits_n: int = 0
    value_q: float = 0.0  # Cumulative reward
    ast_fingerprint: Optional[str] = None
    gpu_hours_spent: float = 0.0
    cpu_hours_spent: float = 0.0
    rebuttal: Optional[str] = None
    depth: int = 0
    branch_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def mean_q(self) -> float:
        """Returns empirical mean value Q / N."""
        return (self.value_q / self.visits_n) if self.visits_n > 0 else 0.0


class StageAccounting(BaseModel):
    stage_name: str
    hypotheses_entered: int = 0
    hypotheses_passed: int = 0
    hypotheses_rejected: int = 0
    total_gpu_hours: float = 0.0
    total_cpu_hours: float = 0.0
    rejection_reasons: Dict[str, int] = Field(default_factory=dict)

    def pass_rate(self) -> float:
        total = self.hypotheses_passed + self.hypotheses_rejected
        return (self.hypotheses_passed / total) if total > 0 else 0.0


class SearchAccountingLedger(BaseModel):
    """Tracks hypotheses entered, passed, and rejected across all stages of exploration."""
    
    stages: Dict[str, StageAccounting] = Field(default_factory=dict)
    total_hypotheses: int = 0
    total_discoveries: int = 0

    def _get_or_create_stage(self, stage: Union[SearchStage, str]) -> StageAccounting:
        s_name = stage.value if isinstance(stage, SearchStage) else str(stage)
        if s_name not in self.stages:
            self.stages[s_name] = StageAccounting(stage_name=s_name)
        return self.stages[s_name]

    def record_entry(self, stage: Union[SearchStage, str]) -> None:
        st = self._get_or_create_stage(stage)
        st.hypotheses_entered += 1
        s_name = stage.value if isinstance(stage, SearchStage) else str(stage)
        if s_name == SearchStage.IDEA_PROPOSAL.value:
            self.total_hypotheses += 1

    def record_pass(
        self,
        stage: Union[SearchStage, str],
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
    ) -> None:
        st = self._get_or_create_stage(stage)
        st.hypotheses_passed += 1
        st.total_gpu_hours += gpu_hours
        st.total_cpu_hours += cpu_hours

    def record_rejection(
        self,
        stage: Union[SearchStage, str],
        reason: str,
        gpu_hours: float = 0.0,
        cpu_hours: float = 0.0,
    ) -> None:
        st = self._get_or_create_stage(stage)
        st.hypotheses_rejected += 1
        st.total_gpu_hours += gpu_hours
        st.total_cpu_hours += cpu_hours
        st.rejection_reasons[reason] = st.rejection_reasons.get(reason, 0) + 1

    def record_discovery(self) -> None:
        self.total_discoveries += 1
        self.record_entry(SearchStage.CONFIRMED_DISCOVERY)
        self.record_pass(SearchStage.CONFIRMED_DISCOVERY)

    def get_discovery_yield(self) -> float:
        """Calculates discovery yield: discoveries / total hypotheses entered."""
        return (self.total_discoveries / self.total_hypotheses) if self.total_hypotheses > 0 else 0.0

    def get_total_gpu_hours(self) -> float:
        return sum(st.total_gpu_hours for st in self.stages.values())

    def get_total_cpu_hours(self) -> float:
        return sum(st.total_cpu_hours for st in self.stages.values())

    def get_mean_gpu_hours_between_discoveries(self) -> float:
        """Calculates mean GPU hours expended per confirmed discovery."""
        tot_gpu = self.get_total_gpu_hours()
        return (tot_gpu / self.total_discoveries) if self.total_discoveries > 0 else tot_gpu

    def summary(self) -> Dict[str, Any]:
        return {
            "total_hypotheses": self.total_hypotheses,
            "total_discoveries": self.total_discoveries,
            "discovery_yield": self.get_discovery_yield(),
            "total_gpu_hours": self.get_total_gpu_hours(),
            "total_cpu_hours": self.get_total_cpu_hours(),
            "mean_gpu_hours_per_discovery": self.get_mean_gpu_hours_between_discoveries(),
            "stage_pass_rates": {s_name: st.pass_rate() for s_name, st in self.stages.items()},
        }

    def save_to_file(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """Persists accounting ledger to disk."""
        target = Path(filepath or DEFAULT_ACCOUNTING_LEDGER_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return str(target)

    @classmethod
    def load_from_file(cls, filepath: Optional[Union[str, Path]] = None) -> "SearchAccountingLedger":
        """Loads accounting ledger from disk."""
        target = Path(filepath or DEFAULT_ACCOUNTING_LEDGER_PATH)
        if not target.exists():
            return cls()
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)
