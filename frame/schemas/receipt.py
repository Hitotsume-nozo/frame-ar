import hashlib
from typing import Dict, List, Optional
from pydantic import BaseModel
from frame.schemas.compute import ExecutionTier
from frame.utils.canonical_json import canonical_json_dumps


class HardwareTelemetry(BaseModel):
    device_name: str
    cuda_available: bool
    cuda_version: Optional[str] = None
    pytorch_version: str
    peak_vram_mb: float
    cpu_threads: int


class ExecutionReceipt(BaseModel):
    receipt_id: str
    hypothesis_id: str
    tier: ExecutionTier
    baseline_seed_scores: Dict[str, float]
    candidate_seed_scores: Dict[str, float]
    replay_micro_step_losses: List[float]
    replay_seed: int
    wall_clock_seconds: float
    hardware: HardwareTelemetry
    stdout_hash: str
    code_hash: str
    hypothesis_hash: Optional[str] = None
    receipt_hash: Optional[str] = None

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"receipt_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()
