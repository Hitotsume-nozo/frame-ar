"""Program of Thought (PoT) Tracing & Compute Profiler.

Leverages PyTorch's FlopCounterMode to measure exact ATen-level FLOPs on a micro-batch
(B=1) and deterministically projects total experiment training compute, routing workloads
to Tier 1 (Host Local), Tier 2 (Kaggle Remote), or Tier 3 (Manual Air-Gapped Capsule).
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from frame.schemas.compute import ExecutionTier, ComputeProfile


def profile_module_flops(
    module: nn.Module,
    sample_input: torch.Tensor,
) -> Tuple[int, int, int]:
    """Profiles exact forward and backward FLOPs and parameter count for a given module.
    
    Returns:
        Tuple[forward_flops, backward_flops, total_parameters]
    """
    total_params = sum(p.numel() for p in module.parameters())
    
    # 1. Forward FLOP counting
    with FlopCounterMode(display=False) as fwd_counter:
        out = module(sample_input)
    fwd_flops = sum(fwd_counter.get_flop_counts()["Global"].values()) if "Global" in fwd_counter.get_flop_counts() else 0
    
    # 2. Backward FLOP counting
    grad_output = torch.randn_like(out)
    with FlopCounterMode(display=False) as bwd_counter:
        out.backward(grad_output, retain_graph=False)
    bwd_flops = sum(bwd_counter.get_flop_counts()["Global"].values()) if "Global" in bwd_counter.get_flop_counts() else 0
    
    # Reset gradients after profiling
    module.zero_grad(set_to_none=True)
    
    return int(fwd_flops), int(bwd_flops), int(total_params)


def route_compute_tier(
    fwd_flops: int,
    bwd_flops: int,
    param_count: int,
    seed_count: int = 5,
    epochs: int = 10,
    batches_per_epoch: int = 100,
) -> ComputeProfile:
    """Calculates total training FLOPs and routes to the appropriate execution tier."""
    step_flops = fwd_flops + bwd_flops
    total_training_flops = seed_count * epochs * batches_per_epoch * step_flops
    
    # Estimated peak VRAM: params (FP32: 4 bytes) + grads (4 bytes) + AdamW moments (8 bytes) + activations
    # Estimated activation multiplier: ~3x parameter memory for standard architectures
    vram_bytes = param_count * (4 + 4 + 8) + (param_count * 4 * 3)
    peak_vram_mb = int(vram_bytes / (1024 * 1024)) + 256  # Include PyTorch CUDA runtime overhead
    
    # Estimated runtime on host GPU (~1 TFLOP/s effective throughput)
    estimated_local_time_sec = total_training_flops / 1e12
    
    # Hardware Routing Matrix:
    # Tier 1 (Host Local): Total FLOPs <= 3e14 (~5 min) AND Peak VRAM <= 3000 MB
    # Tier 2 (Kaggle P100): 3e14 < Total FLOPs <= 1e17 AND Peak VRAM <= 14500 MB AND Time <= 8.5h
    # Tier 3 (Manual Air-Gapped Capsule): All larger or multi-GPU workloads
    
    if total_training_flops <= 3e14 and peak_vram_mb <= 3000:
        tier = ExecutionTier.TIER_1_LOCAL
    elif total_training_flops <= 1.0e17 and peak_vram_mb <= 14500 and (total_training_flops / 3.25e12) <= 30600:
        tier = ExecutionTier.TIER_2_KAGGLE
    else:
        tier = ExecutionTier.TIER_3_MANUAL
        
    return ComputeProfile(
        tier=tier,
        parameter_count=param_count,
        forward_flops_per_sample=fwd_flops,
        backward_flops_per_sample=bwd_flops,
        step_flops_total=step_flops,
        total_training_flops=total_training_flops,
        peak_vram_mb=peak_vram_mb,
        estimated_runtime_sec=estimated_local_time_sec,
        seed_count=seed_count,
        epochs=epochs,
        batches_per_epoch=batches_per_epoch,
        requires_gpu=(tier != ExecutionTier.TIER_1_LOCAL or peak_vram_mb > 500),
    )
