import hashlib
from typing import List, Optional
from pydantic import BaseModel
from frame.utils.canonical_json import canonical_json_dumps


class CapsuleManifest(BaseModel):
    capsule_id: str
    hypothesis_id: str
    code_hash: str
    deterministic_seeds: List[int]
    micro_replay_steps: int = 3
    replay_tolerance: float = 1e-4
    docker_image: str = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime"
    manifest_hash: Optional[str] = None

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"manifest_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()


class ReplayVerificationResult(BaseModel):
    hypothesis_id: str
    passed: bool
    max_loss_delta: float
    tolerance: float = 1e-4
    recorded_losses: List[float]
    replayed_losses: List[float]
    verification_hash: Optional[str] = None

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"verification_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()
