"""Manual Execution Capsule Packager & Replay Verifier.

Generates self-contained air-gapped run directories for massive compute workloads
(HPC, multi-GPU clusters, cloud pods) and performs deterministic micro-replay
verification on the local host when results are returned.
"""

import hashlib
import json
import math
import os
from typing import Dict, List, Tuple
from frame.schemas.capsule import CapsuleManifest, ReplayVerificationResult
from frame.utils.canonical_json import canonical_json_dumps


def package_manual_capsule(
    output_dir: str,
    capsule_id: str,
    hypothesis_id: str,
    code_content: str,
    deterministic_seeds: List[int],
    micro_replay_steps: int = 3,
    tolerance: float = 1e-4,
) -> CapsuleManifest:
    """Packages an air-gapped execution capsule containing scripts, configs, and container manifests."""
    os.makedirs(output_dir, exist_ok=True)
    
    code_hash = hashlib.sha256(code_content.encode("utf-8")).hexdigest()
    
    manifest = CapsuleManifest(
        capsule_id=capsule_id,
        hypothesis_id=hypothesis_id,
        code_hash=code_hash,
        deterministic_seeds=deterministic_seeds,
        micro_replay_steps=micro_replay_steps,
        replay_tolerance=tolerance,
    )
    manifest.manifest_hash = manifest.compute_hash()
    
    # 1. Write manifest
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        f.write(canonical_json_dumps(manifest.model_dump()))
        
    # 2. Write entrypoint script with strict TF32 precision disabling
    entrypoint_script = f'''"""Air-Gapped Autonomous Research Training Entrypoint."""
import os
import sys
import torch

# CRITICAL: Disable TensorFloat-32 to preserve IEEE 754 precision for host replay verification
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch, "use_deterministic_algorithms"):
    torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

{code_content}
'''
    with open(os.path.join(output_dir, "entrypoint.py"), "w", encoding="utf-8") as f:
        f.write(entrypoint_script)
        
    # 3. Write bootstrap shell script
    run_sh = '''#!/usr/bin/env bash
set -euo pipefail

echo "==> Bootstrapping Manual Execution Capsule..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch scipy numpy pydantic

echo "==> Executing Training Pipeline..."
python3 entrypoint.py

echo "==> Packaging results..."
tar -czf results.tar.gz receipt.json checkpoints/
echo "==> Completed successfully. Return results.tar.gz to host framework."
'''
    with open(os.path.join(output_dir, "run.sh"), "w", encoding="utf-8") as f:
        f.write(run_sh)
    os.chmod(os.path.join(output_dir, "run.sh"), 0o755)
    
    # 4. Write Dockerfile
    dockerfile = '''FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace
COPY . /workspace

RUN pip install --no-cache-dir scipy numpy pydantic

ENTRYPOINT ["bash", "run.sh"]
'''
    with open(os.path.join(output_dir, "Dockerfile"), "w", encoding="utf-8") as f:
        f.write(dockerfile)
        
    return manifest


def verify_micro_replay(
    hypothesis_id: str,
    recorded_losses: List[float],
    replayed_losses: List[float],
    tolerance: float = 1e-4,
) -> ReplayVerificationResult:
    """Verifies that replayed micro-step training losses match recorded losses within tolerance."""
    if len(recorded_losses) != len(replayed_losses):
        raise ValueError(
            f"Mismatched step counts: {len(recorded_losses)} recorded vs {len(replayed_losses)} replayed"
        )
        
    max_delta = 0.0
    for rec, rep in zip(recorded_losses, replayed_losses):
        delta = abs(rec - rep)
        if delta > max_delta:
            max_delta = delta
            
    passed = (max_delta <= tolerance)
    
    result = ReplayVerificationResult(
        hypothesis_id=hypothesis_id,
        passed=passed,
        max_loss_delta=max_delta,
        tolerance=tolerance,
        recorded_losses=recorded_losses,
        replayed_losses=replayed_losses,
    )
    result.verification_hash = result.compute_hash()
    return result
