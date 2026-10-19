from enum import Enum
from pydantic import BaseModel, Field


class ExecutionTier(str, Enum):
    TIER_1_LOCAL = "tier_1_local"
    TIER_2_KAGGLE = "tier_2_kaggle"
    TIER_3_MANUAL = "tier_3_manual"


class ComputeProfile(BaseModel):
    tier: ExecutionTier
    parameter_count: int
    forward_flops_per_sample: int
    backward_flops_per_sample: int
    step_flops_total: int
    total_training_flops: int
    peak_vram_mb: int
    estimated_runtime_sec: float
    seed_count: int = Field(default=5, ge=3, le=20)
    epochs: int = Field(default=10, ge=1)
    batches_per_epoch: int = Field(default=100, ge=1)
    requires_gpu: bool = False
