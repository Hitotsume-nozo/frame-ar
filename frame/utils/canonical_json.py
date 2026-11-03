"""RFC 8785 Canonical JSON serialization and SHA-256 cryptographic hashing."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _normalize_value(val: Any) -> Any:
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Non-finite float not permitted in canonical JSON: {val}")
        if val == 0.0:
            return 0.0
        return round(val, 10)
    if isinstance(val, dict):
        return {str(k): _normalize_value(v) for k, v in sorted(val.items(), key=lambda item: str(item[0]))}
    if isinstance(val, (list, tuple)):
        return [_normalize_value(v) for v in val]
    return val


def canonical_json_bytes(obj: Any) -> bytes:
    """Serializes a Python object into deterministic UTF-8 bytes."""
    normalized = _normalize_value(obj)
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text.encode("utf-8")


def compute_obj_sha256(obj: Any) -> str:
    """Computes the hex SHA-256 digest of a canonicalized Python object."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Computes the hex SHA-256 digest of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
