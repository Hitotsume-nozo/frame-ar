"""Kaggle runner package for FRAME-AR."""

from frame.runners.kaggle.bundler import (
    create_self_extracting_payload,
    bundle_framework_source,
    create_kaggle_bundle,
)
from frame.runners.kaggle.credentials import (
    KaggleCredentials,
    check_file_permissions,
    load_kaggle_credentials,
    save_kaggle_credentials,
)
from frame.runners.kaggle.quota import (
    KaggleQuotaLedger,
    QuotaUsageEntry,
    QuotaLedgerData,
    KaggleQuotaExceededError,
    WEEKLY_GPU_HOURS_LIMIT,
)
from frame.runners.kaggle.client import (
    KaggleClient,
    KaggleAPIError,
    KaggleWatchdogTimeoutError,
    KaggleExecutionError,
    WATCHDOG_SAFETY_CEILING_SEC,
)

__all__ = [
    "create_self_extracting_payload",
    "bundle_framework_source",
    "create_kaggle_bundle",
    "KaggleCredentials",
    "check_file_permissions",
    "load_kaggle_credentials",
    "save_kaggle_credentials",
    "KaggleQuotaLedger",
    "QuotaUsageEntry",
    "QuotaLedgerData",
    "KaggleQuotaExceededError",
    "WEEKLY_GPU_HOURS_LIMIT",
    "KaggleClient",
    "KaggleAPIError",
    "KaggleWatchdogTimeoutError",
    "KaggleExecutionError",
    "WATCHDOG_SAFETY_CEILING_SEC",
]
