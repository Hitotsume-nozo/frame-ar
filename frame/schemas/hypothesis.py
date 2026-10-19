from enum import Enum
import hashlib
import hmac
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from frame.utils.canonical_json import canonical_json_dumps


class TargetDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


def generate_sealed_seeds(verifier_secret_key: str, hypothesis_id: str, count: int = 5) -> List[int]:
    """Generates sealed deterministic seeds using HMAC-SHA256 at dispatch.
    
    s_i = HMAC-SHA256(K_verifier, hypothesis_id || i) mod (2^31 - 1)
    """
    key_bytes = verifier_secret_key.encode("utf-8")
    sealed_seeds: List[int] = []
    for i in range(count):
        msg = f"{hypothesis_id}:{i}".encode("utf-8")
        h = hmac.new(key_bytes, msg, hashlib.sha256).hexdigest()
        seed_val = int(h, 16) % (2**31 - 1)
        sealed_seeds.append(seed_val)
    return sealed_seeds


def create_seed_commitment(sealed_seeds: List[int], salt: str) -> str:
    """Creates a cryptographic HMAC-SHA256 commitment of the sealed seeds list."""
    payload = canonical_json_dumps({"seeds": sealed_seeds, "salt": salt})
    return hmac.new(salt.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_seed_commitment(commitment: str, sealed_seeds: List[int], salt: str) -> bool:
    """Verifies that the revealed sealed seeds match the earlier commitment."""
    expected = create_seed_commitment(sealed_seeds, salt)
    return hmac.compare_digest(commitment, expected)


class Hypothesis(BaseModel):
    hypothesis_id: str
    title: str
    description: str
    model_architecture: str
    hyperparameters: Dict[str, Any]
    metric_name: str
    target_direction: TargetDirection = TargetDirection.MAXIMIZE
    rope_epsilon: float = Field(default=0.005, gt=0.0)
    public_seeds: List[int] = Field(default_factory=lambda: [101, 102])
    deterministic_seeds: List[int] = Field(default_factory=lambda: [101, 102])
    sealed_seed_commitment: Optional[str] = None
    baseline_code_snippet: str
    candidate_code_snippet: str
    hypothesis_hash: Optional[str] = None

    def compute_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"hypothesis_hash"})
        return hashlib.sha256(canonical_json_dumps(data).encode("utf-8")).hexdigest()

