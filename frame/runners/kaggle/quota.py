"""Local Rolling 7-Day Quota Ledger for Kaggle GPU Compute.

Tracks GPU hours against Kaggle's 30h rolling weekly limit.
Persists state in `.frame/kaggle_quota.json`.
"""

import json
import os
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

WEEKLY_GPU_HOURS_LIMIT: float = 30.0
ROLLING_WINDOW_DAYS: float = 7.0
ROLLING_WINDOW_SECONDS: float = ROLLING_WINDOW_DAYS * 24.0 * 3600.0  # 604,800 s
DEFAULT_QUOTA_RELPATH: str = os.path.join(".frame", "kaggle_quota.json")


class KaggleQuotaExceededError(RuntimeError):
    """Raised when an operation would exceed the rolling 7-day GPU quota."""
    pass


class QuotaUsageEntry(BaseModel):
    """Record of a single Kaggle execution and consumed GPU hours."""
    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    job_slug: str
    hypothesis_id: Optional[str] = None
    gpu_hours: float = Field(..., ge=0.0)
    timestamp: float = Field(default_factory=time.time)
    status: str = Field(default="completed")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuotaLedgerData(BaseModel):
    """Serializable structure for the rolling quota ledger."""
    weekly_limit_hours: float = WEEKLY_GPU_HOURS_LIMIT
    rolling_window_days: float = ROLLING_WINDOW_DAYS
    entries: List[QuotaUsageEntry] = Field(default_factory=list)


class KaggleQuotaLedger:
    """Manages the local rolling 7-day Kaggle GPU quota ledger."""

    def __init__(
        self,
        ledger_path: Optional[str] = None,
        weekly_limit_hours: float = WEEKLY_GPU_HOURS_LIMIT,
        rolling_window_days: float = ROLLING_WINDOW_DAYS,
    ):
        """Initializes the quota ledger.
        
        Args:
            ledger_path: Path to `.frame/kaggle_quota.json`. If None, resolves to `.frame/kaggle_quota.json`
                         relative to cwd or project root.
            weekly_limit_hours: Total GPU hours available per rolling week (default 30.0).
            rolling_window_days: Rolling window length in days (default 7.0).
        """
        if ledger_path is not None:
            self.ledger_path = os.path.abspath(ledger_path)
        else:
            self.ledger_path = os.path.abspath(DEFAULT_QUOTA_RELPATH)

        self.weekly_limit_hours = weekly_limit_hours
        self.rolling_window_days = rolling_window_days
        self.rolling_window_seconds = rolling_window_days * 24.0 * 3600.0

        self.data = QuotaLedgerData(
            weekly_limit_hours=self.weekly_limit_hours,
            rolling_window_days=self.rolling_window_days,
            entries=[],
        )
        self.load()

    def load(self) -> None:
        """Loads ledger from disk if it exists, initializing a fresh ledger otherwise."""
        if not os.path.isfile(self.ledger_path):
            return

        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
            loaded_data = QuotaLedgerData.model_validate(raw_json)
            # Merge loaded entries with in-memory entries deduplicating by entry_id
            known_ids = {e.entry_id for e in loaded_data.entries}
            merged = list(loaded_data.entries)
            for entry in self.data.entries:
                if entry.entry_id not in known_ids:
                    merged.append(entry)
                    known_ids.add(entry.entry_id)
            self.data = loaded_data
            self.data.entries = merged
        except Exception:
            # If ledger file is corrupted, preserve existing memory state
            pass

    def save(self) -> None:
        """Persists the quota ledger atomically to disk."""
        target_dir = os.path.dirname(self.ledger_path)
        os.makedirs(target_dir, exist_ok=True)

        data_dict = self.data.model_dump(mode="json")
        # Atomic write via temp file in same directory
        temp_fd, temp_path = tempfile.mkstemp(dir=target_dir, prefix="quota_", suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data_dict, f, indent=2)
                f.write("\n")
            os.replace(temp_path, self.ledger_path)
        except Exception:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise

    def get_entries(
        self,
        current_time: Optional[float] = None,
        rolling_only: bool = True,
    ) -> List[QuotaUsageEntry]:
        """Retrieves entries, optionally filtering to the active 7-day rolling window."""
        self.load()
        now = current_time if current_time is not None else time.time()
        if not rolling_only:
            return list(self.data.entries)

        cutoff = now - self.rolling_window_seconds
        return [entry for entry in self.data.entries if entry.timestamp >= cutoff]

    def get_used_gpu_hours(self, current_time: Optional[float] = None) -> float:
        """Calculates total GPU hours consumed within the active rolling 7-day window."""
        active_entries = self.get_entries(current_time=current_time, rolling_only=True)
        return round(sum(entry.gpu_hours for entry in active_entries), 4)

    def get_remaining_gpu_hours(self, current_time: Optional[float] = None) -> float:
        """Calculates remaining GPU hours available within the 30h weekly limit."""
        used = self.get_used_gpu_hours(current_time=current_time)
        return round(max(0.0, self.weekly_limit_hours - used), 4)

    def can_dispatch(self, gpu_hours: float, current_time: Optional[float] = None) -> bool:
        """Checks if a job requiring `gpu_hours` fits within the remaining rolling quota."""
        if gpu_hours < 0:
            return False
        used = self.get_used_gpu_hours(current_time=current_time)
        return (used + gpu_hours) <= self.weekly_limit_hours

    def assert_quota_available(self, gpu_hours: float, current_time: Optional[float] = None) -> None:
        """Raises KaggleQuotaExceededError if the required hours exceed available quota."""
        if not self.can_dispatch(gpu_hours, current_time=current_time):
            used = self.get_used_gpu_hours(current_time=current_time)
            remaining = self.get_remaining_gpu_hours(current_time=current_time)
            raise KaggleQuotaExceededError(
                f"Requested {gpu_hours:.2f}h GPU compute exceeds available rolling quota. "
                f"Used: {used:.2f}h / {self.weekly_limit_hours:.1f}h limit "
                f"({remaining:.2f}h remaining in 7-day window)."
            )

    def record_usage(
        self,
        job_slug: str,
        gpu_hours: float,
        hypothesis_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        status: str = "completed",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> QuotaUsageEntry:
        """Records a new execution run in the ledger and saves atomically."""
        self.load()
        ts = timestamp if timestamp is not None else time.time()
        entry = QuotaUsageEntry(
            job_slug=job_slug,
            hypothesis_id=hypothesis_id,
            gpu_hours=round(float(gpu_hours), 4),
            timestamp=ts,
            status=status,
            metadata=metadata or {},
        )
        self.data.entries.append(entry)
        self.save()
        return entry

    def prune_expired_entries(
        self,
        current_time: Optional[float] = None,
        retention_days: float = 30.0,
    ) -> int:
        """Prunes historical entries older than retention_days (default: 30 days).
        
        Returns:
            Number of pruned entries.
        """
        now = current_time if current_time is not None else time.time()
        retention_seconds = retention_days * 24.0 * 3600.0
        cutoff = now - retention_seconds

        initial_count = len(self.data.entries)
        self.data.entries = [entry for entry in self.data.entries if entry.timestamp >= cutoff]
        pruned_count = initial_count - len(self.data.entries)

        if pruned_count > 0:
            self.save()

        return pruned_count
