"""Production Data Contracts and Schemas for FRAME-AR."""

from frame.schemas.compute import ExecutionTier, ComputeProfile
from frame.schemas.hypothesis import Hypothesis, TargetDirection
from frame.schemas.receipt import ExecutionReceipt, HardwareTelemetry
from frame.schemas.verification import StatisticalVerificationReport, PairedSeedMetrics
from frame.schemas.capsule import CapsuleManifest, ReplayVerificationResult
from frame.schemas.novelty import NoveltyVerdict, NoveltyContractResult
from frame.schemas.learning_state import TrajectoryPoint, SwarmLearningStateNode
from frame.schemas.executioner import StagnationTrigger, ExecutionerEvent

__all__ = [
    "ExecutionTier",
    "ComputeProfile",
    "Hypothesis",
    "TargetDirection",
    "ExecutionReceipt",
    "HardwareTelemetry",
    "StatisticalVerificationReport",
    "PairedSeedMetrics",
    "CapsuleManifest",
    "ReplayVerificationResult",
    "NoveltyVerdict",
    "NoveltyContractResult",
    "TrajectoryPoint",
    "SwarmLearningStateNode",
    "StagnationTrigger",
    "ExecutionerEvent",
]
