"""RFC 8785 Canonical JSON Serialization Scheme."""

import json
from typing import Any


def canonical_json_dumps(obj: Any) -> str:
    """Serializes a Python object to canonical JSON string according to RFC 8785 principles.
    
    Keys are sorted lexicographically by unicode code points, and separators have no whitespace.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
