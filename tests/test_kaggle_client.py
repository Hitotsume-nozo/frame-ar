"""Unit and Integration Tests for Kaggle REST Infrastructure & Credential Management.

Covers Phase 1 requirements:
1. Kaggle API credentials configuration:
   - Resolution order: Env vars (KAGGLE_USERNAME, KAGGLE_KEY) -> ~/.kaggle/kaggle.json -> .frame/kaggle.json.
   - Enforce POSIX 0600 permissions check.
2. Zero-dependency native REST client in `frame/runners/kaggle/client.py`:
   - Basic Auth header: `Authorization: Basic base64("{username}:{key}")`.
   - Kernel Push: `POST https://www.kaggle.com/api/v1/kernels/push`.
   - Kernel Status: `GET https://www.kaggle.com/api/v1/kernels/status`.
   - Kernel Output Download: `GET https://www.kaggle.com/api/v1/kernels/output`.
   - Active execution watchdog (8.5h safety ceiling) to prevent timeout without checkpoint saving.
3. Local rolling 7-day quota ledger in `frame/runners/kaggle/quota.py` tracking GPU hours (30h limit).
"""

import base64
import io
import json
import os
import stat
import tempfile
import urllib.error
from unittest import mock

import pytest

from frame.runners.kaggle.credentials import (
    KaggleCredentials,
    check_file_permissions,
    load_kaggle_credentials,
    save_kaggle_credentials,
)
from frame.runners.kaggle.quota import (
    KaggleQuotaLedger,
    KaggleQuotaExceededError,
    QuotaUsageEntry,
    WEEKLY_GPU_HOURS_LIMIT,
)
from frame.runners.kaggle.client import (
    KaggleClient,
    KaggleAPIError,
    KaggleWatchdogTimeoutError,
    KaggleExecutionError,
    WATCHDOG_SAFETY_CEILING_SEC,
)


# =====================================================================
# 1. Credentials Configuration & Permissions Tests
# =====================================================================

def test_credentials_basic_auth_header():
    """Verifies RFC 7617 Basic Authentication base64 encoding."""
    creds = KaggleCredentials(username="alice", key="secret123")
    expected_token = base64.b64encode(b"alice:secret123").decode("ascii")
    assert creds.get_basic_auth_header() == f"Basic {expected_token}"


def test_credentials_resolution_env_vars(monkeypatch):
    """Asserts environment variables take highest precedence in resolution hierarchy."""
    monkeypatch.setenv("KAGGLE_USERNAME", "env_user")
    monkeypatch.setenv("KAGGLE_KEY", "env_key_secret")

    creds = load_kaggle_credentials()
    assert creds.username == "env_user"
    assert creds.key == "env_key_secret"
    assert creds.source == "env"


def test_credentials_resolution_user_file(monkeypatch):
    """Asserts ~/.kaggle/kaggle.json is used when environment variables are unset."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmp_home:
        kaggle_dir = os.path.join(tmp_home, ".kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)
        cred_file = os.path.join(kaggle_dir, "kaggle.json")
        with open(cred_file, "w", encoding="utf-8") as f:
            json.dump({"username": "user_file_user", "key": "key_file_secret"}, f)
        os.chmod(cred_file, 0o600)

        monkeypatch.setenv("HOME", tmp_home)

        creds = load_kaggle_credentials(project_dir="/nonexistent")
        assert creds.username == "user_file_user"
        assert creds.key == "key_file_secret"
        assert creds.source == cred_file


def test_credentials_resolution_project_file(monkeypatch):
    """Asserts .frame/kaggle.json is used when neither env nor user file exists."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmp_home, tempfile.TemporaryDirectory() as tmp_proj:
        monkeypatch.setenv("HOME", tmp_home)

        frame_dir = os.path.join(tmp_proj, ".frame")
        os.makedirs(frame_dir, exist_ok=True)
        proj_cred = os.path.join(frame_dir, "kaggle.json")
        with open(proj_cred, "w", encoding="utf-8") as f:
            json.dump({"username": "proj_user", "key": "proj_key"}, f)
        os.chmod(proj_cred, 0o600)

        creds = load_kaggle_credentials(project_dir=tmp_proj)
        assert creds.username == "proj_user"
        assert creds.key == "proj_key"
        assert creds.source == proj_cred


def test_credentials_enforce_posix_0600_permissions(monkeypatch):
    """Asserts that files with permissions more open than 0600 raise PermissionError."""
    if os.name != "posix":
        pytest.skip("POSIX permission check only applicable on POSIX systems")

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmp_home, tempfile.TemporaryDirectory() as tmp_proj:
        monkeypatch.setenv("HOME", tmp_home)
        frame_dir = os.path.join(tmp_proj, ".frame")
        os.makedirs(frame_dir, exist_ok=True)
        insecure_cred = os.path.join(frame_dir, "kaggle.json")
        with open(insecure_cred, "w", encoding="utf-8") as f:
            json.dump({"username": "insecure_user", "key": "insecure_key"}, f)

        # Make file group and world readable (0644)
        os.chmod(insecure_cred, 0o644)

        with pytest.raises(PermissionError) as exc_info:
            load_kaggle_credentials(project_dir=tmp_proj, enforce_permissions=True)
        assert "expected 0o600" in str(exc_info.value)
        assert "chmod 600" in str(exc_info.value)

        # Fix permissions to 0600 and assert it now succeeds
        os.chmod(insecure_cred, 0o600)
        creds = load_kaggle_credentials(project_dir=tmp_proj, enforce_permissions=True)
        assert creds.username == "insecure_user"


def test_save_kaggle_credentials_sets_0600():
    """Verifies save_kaggle_credentials creates file with POSIX 0600 mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        target = os.path.join(tmp_dir, "test_kaggle.json")
        creds = KaggleCredentials(username="saved_user", key="saved_key")
        saved_path = save_kaggle_credentials(creds, filepath=target)

        assert os.path.isfile(saved_path)
        if os.name == "posix":
            mode = stat.S_IMODE(os.stat(saved_path).st_mode)
            assert mode == 0o600


def test_credentials_missing_error(monkeypatch):
    """Verifies FileNotFoundError raised when no credentials exist anywhere."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as empty_home, tempfile.TemporaryDirectory() as empty_proj:
        monkeypatch.setenv("HOME", empty_home)
        with pytest.raises(FileNotFoundError):
            load_kaggle_credentials(project_dir=empty_proj)


# =====================================================================
# 2. Rolling 7-Day Quota Ledger Tests
# =====================================================================

def test_quota_ledger_initial_state():
    """Asserts new ledger starts with 0 used and 30h remaining."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        ledger = KaggleQuotaLedger(ledger_path=ledger_file)

        assert ledger.weekly_limit_hours == 30.0
        assert ledger.get_used_gpu_hours() == 0.0
        assert ledger.get_remaining_gpu_hours() == 30.0
        assert ledger.can_dispatch(30.0) is True
        assert ledger.can_dispatch(30.1) is False


def test_quota_ledger_recording_and_reloading():
    """Asserts recording runs decrements available quota and persists to disk."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        ledger = KaggleQuotaLedger(ledger_path=ledger_file)

        entry = ledger.record_usage(
            job_slug="kernel-run-1",
            gpu_hours=4.5,
            hypothesis_id="hyp_101",
        )
        assert entry.gpu_hours == 4.5
        assert ledger.get_used_gpu_hours() == 4.5
        assert ledger.get_remaining_gpu_hours() == 25.5

        # Reload from disk in fresh instance
        reloaded = KaggleQuotaLedger(ledger_path=ledger_file)
        assert reloaded.get_used_gpu_hours() == 4.5
        assert reloaded.get_remaining_gpu_hours() == 25.5
        assert len(reloaded.data.entries) == 1
        assert reloaded.data.entries[0].job_slug == "kernel-run-1"


def test_quota_ledger_rolling_7day_window_expiry():
    """Asserts jobs older than 7 days fall off the rolling window sum."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        ledger = KaggleQuotaLedger(ledger_path=ledger_file)

        t_now = 10000000.0
        day = 86400.0

        # Run 1: 8 days ago (expired, > 7 days)
        ledger.record_usage("job-old", 10.0, timestamp=t_now - (8 * day))
        # Run 2: 6 days ago (active, < 7 days)
        ledger.record_usage("job-mid", 8.0, timestamp=t_now - (6 * day))
        # Run 3: 2 days ago (active, < 7 days)
        ledger.record_usage("job-recent", 4.0, timestamp=t_now - (2 * day))

        # Sum of active runs should be 8.0 + 4.0 = 12.0 (10.0 is expired)
        assert ledger.get_used_gpu_hours(current_time=t_now) == 12.0
        assert ledger.get_remaining_gpu_hours(current_time=t_now) == 18.0

        # Advance time by another 2 days: job-mid (originally 6 days ago) is now 8 days old
        t_future = t_now + (2 * day)
        assert ledger.get_used_gpu_hours(current_time=t_future) == 4.0
        assert ledger.get_remaining_gpu_hours(current_time=t_future) == 26.0


def test_quota_ledger_exceeded_error():
    """Asserts assert_quota_available raises KaggleQuotaExceededError when over limit."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        ledger = KaggleQuotaLedger(ledger_path=ledger_file)

        ledger.record_usage("job-large", 28.0)
        assert ledger.get_remaining_gpu_hours() == 2.0

        # 1.5h should pass
        ledger.assert_quota_available(1.5)

        # 2.5h should raise
        with pytest.raises(KaggleQuotaExceededError) as exc_info:
            ledger.assert_quota_available(2.5)
        assert "exceeds available rolling quota" in str(exc_info.value)


# =====================================================================
# 3. Native REST Client Tests (urllib Mocking)
# =====================================================================

def _mock_http_response(status_code: int = 200, json_data: dict = None, text_data: str = None):
    """Helper to mock urllib.request.urlopen responses."""
    resp = mock.MagicMock()
    resp.status = status_code
    if json_data is not None:
        raw = json.dumps(json_data).encode("utf-8")
    elif text_data is not None:
        raw = text_data.encode("utf-8")
    else:
        raw = b"{}"

    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


def test_client_push_kernel():
    """Verifies POST /api/v1/kernels/push payload format and headers."""
    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    mock_resp_data = {"ref": "researcher/frame-exp-1", "url": "https://kaggle.com/...", "versionNumber": 1}

    with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(200, mock_resp_data)) as mock_open:
        res = client.push_kernel(
            slug="frame-exp-1",
            text="print('Kaggle Autonomous Execution')",
            new_title="Frame Exp 1",
            enable_gpu=True,
            enable_internet=True,
            is_private=True,
        )

        assert res["ref"] == "researcher/frame-exp-1"
        assert mock_open.called
        call_args = mock_open.call_args[0]
        req = call_args[0]

        assert req.method == "POST"
        assert req.full_url == "https://www.kaggle.com/api/v1/kernels/push"

        # Check Basic Auth header
        auth_header = req.get_header("Authorization")
        expected_b64 = base64.b64encode(b"researcher:tokenxyz").decode("ascii")
        assert auth_header == f"Basic {expected_b64}"

        # Check payload fields
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["slug"] == "frame-exp-1"
        assert payload["newTitle"] == "Frame Exp 1"
        assert payload["text"] == "print('Kaggle Autonomous Execution')"
        assert payload["enableGpu"] is True
        assert payload["enableInternet"] is True
        assert payload["isPrivate"] is True


def test_client_get_kernel_status():
    """Verifies GET /api/v1/kernels/status query parameters and status parsing."""
    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    mock_resp_data = {"status": "running"}

    with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(200, mock_resp_data)) as mock_open:
        res = client.get_kernel_status("frame-exp-1")
        assert res["status"] == "running"

        call_args = mock_open.call_args[0]
        req = call_args[0]
        assert req.method == "GET"
        assert "userName=researcher" in req.full_url
        assert "kernelSlug=frame-exp-1" in req.full_url


def test_client_get_and_download_kernel_output():
    """Verifies GET /api/v1/kernels/output and file artifact saving."""
    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    mock_resp_data = {
        "log": "[FRAME-AR] Loss: 0.125\nTraining completed.",
        "files": [
            {"fileName": "metrics.json", "content": '{"loss": 0.125}'},
        ],
    }

    with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(200, mock_resp_data)):
        with tempfile.TemporaryDirectory() as tmp_out:
            saved = client.download_kernel_output("frame-exp-1", destination_dir=tmp_out)

            assert "stdout.log" in saved
            assert "metrics.json" in saved
            assert os.path.isfile(saved["stdout.log"])
            assert os.path.isfile(saved["metrics.json"])

            with open(saved["stdout.log"], "r", encoding="utf-8") as f:
                assert "Loss: 0.125" in f.read()

            with open(saved["metrics.json"], "r", encoding="utf-8") as f:
                assert json.load(f) == {"loss": 0.125}


def test_client_http_error_handling():
    """Asserts KaggleAPIError is raised with status code on HTTP error."""
    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    error_response = io.BytesIO(b'{"code": 403, "message": "Permission denied"}')
    http_error = urllib.error.HTTPError(
        url="https://www.kaggle.com/api/v1/kernels/push",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=error_response,
    )

    with mock.patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(KaggleAPIError) as exc_info:
            client.push_kernel("frame-exp", "text")
        assert exc_info.value.status_code == 403
        assert "Permission denied" in str(exc_info.value)


# =====================================================================
# 4. Watchdog & Polling Tests
# =====================================================================

def test_watchdog_successful_execution():
    """Asserts watchdog tracks queued -> running -> complete and records quota."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        # Mock status responses in sequence: queued -> running -> complete
        status_responses = [
            {"status": "queued"},
            {"status": "running"},
            {"status": "complete"},
        ]

        time_counter = [1000.0]
        def mock_time():
            return time_counter[0]

        def mock_sleep(sec):
            time_counter[0] += 30.0  # advance time by 30 seconds per poll

        with mock.patch.object(client, "get_kernel_status", side_effect=status_responses):
            res = client.watch_execution(
                slug="test-slug",
                poll_interval_sec=30.0,
                time_fn=mock_time,
                sleep_fn=mock_sleep,
            )

            assert res["status"] == "complete"
            # 2 sleep cycles = 60 seconds = 60 / 3600 hours
            expected_hours = 60.0 / 3600.0
            assert math_isclose(ledger.get_used_gpu_hours(), expected_hours)


def test_watchdog_8_5h_safety_ceiling_timeout():
    """Asserts watchdog aborts when runtime reaches 8.5h (30,600s) to prevent uncheckpointed kill."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        time_counter = [0.0]
        def mock_time():
            return time_counter[0]

        def mock_sleep(sec):
            time_counter[0] += 10000.0  # large jumps

        # Kernel stuck in 'running'
        with mock.patch.object(client, "get_kernel_status", return_value={"status": "running"}):
            with pytest.raises(KaggleWatchdogTimeoutError) as exc_info:
                client.watch_execution(
                    slug="long-running-kernel",
                    timeout_sec=WATCHDOG_SAFETY_CEILING_SEC,
                    poll_interval_sec=10.0,
                    time_fn=mock_time,
                    sleep_fn=mock_sleep,
                )
            assert "exceeded active execution watchdog safety ceiling" in str(exc_info.value)
            # Quota ledger should reflect consumed time under timeout_watchdog
            assert ledger.get_used_gpu_hours() > 8.0


def test_watchdog_error_status_raises_execution_error():
    """Asserts watchdog aborts and raises KaggleExecutionError when status is 'error'."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        status_responses = [
            {"status": "running"},
            {"status": "error", "failureMessage": "CUDA out of memory error"},
        ]

        time_val = [0.0]
        with mock.patch.object(client, "get_kernel_status", side_effect=status_responses):
            with pytest.raises(KaggleExecutionError) as exc_info:
                client.watch_execution(
                    slug="oom-kernel",
                    time_fn=lambda: time_val[0],
                    sleep_fn=lambda s: None,
                )
            assert "CUDA out of memory error" in str(exc_info.value)


# =====================================================================
# 5. High-Level Bundle and Dispatch Integration Test
# =====================================================================

def test_bundle_and_dispatch_pipeline():
    """Asserts end-to-end bundling, quota assertion, and push call."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        files = {"train.py": "print('train')", "config.json": "{}"}
        main_script = "import train\n"

        with mock.patch.object(client, "push_kernel", return_value={"ref": "user/test-bundle"}) as mock_push:
            res = client.bundle_and_dispatch(
                slug="test-bundle",
                files=files,
                main_execution_script=main_script,
                estimated_gpu_hours=2.0,
            )

            assert res["ref"] == "user/test-bundle"
            assert mock_push.called
            call_kwargs = mock_push.call_args[1]
            assert call_kwargs["slug"] == "test-bundle"
            # Text should be self-extracting script containing b64 archive
            assert "PAYLOAD_B64" in call_kwargs["text"]
            assert "import train" in call_kwargs["text"]


def test_credentials_partial_env_vars(monkeypatch):
    """Asserts partial env vars (only username or only key) falls through to file resolution."""
    monkeypatch.setenv("KAGGLE_USERNAME", "only_user")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmp_home, tempfile.TemporaryDirectory() as tmp_proj:
        monkeypatch.setenv("HOME", tmp_home)
        frame_dir = os.path.join(tmp_proj, ".frame")
        os.makedirs(frame_dir, exist_ok=True)
        cred = os.path.join(frame_dir, "kaggle.json")
        with open(cred, "w", encoding="utf-8") as f:
            json.dump({"username": "fallback_user", "key": "fallback_key"}, f)
        os.chmod(cred, 0o600)

        creds = load_kaggle_credentials(project_dir=tmp_proj)
        assert creds.username == "fallback_user"
        assert creds.key == "fallback_key"


def test_credentials_malformed_json(monkeypatch):
    """Asserts malformed json or missing required fields raises ValueError."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    with tempfile.TemporaryDirectory() as tmp_home, tempfile.TemporaryDirectory() as tmp_proj:
        monkeypatch.setenv("HOME", tmp_home)
        frame_dir = os.path.join(tmp_proj, ".frame")
        os.makedirs(frame_dir, exist_ok=True)
        cred = os.path.join(frame_dir, "kaggle.json")
        with open(cred, "w", encoding="utf-8") as f:
            f.write("not valid json at all")
        os.chmod(cred, 0o600)

        with pytest.raises(ValueError):
            load_kaggle_credentials(project_dir=tmp_proj)


def test_client_slug_with_username_prefix():
    """Asserts slug in format 'username/kernel-slug' is correctly split."""
    creds = KaggleCredentials(username="default_user", key="key123")
    client = KaggleClient(credentials=creds)

    mock_resp = {"status": "complete"}
    with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(200, mock_resp)) as mock_open:
        res = client.get_kernel_status("custom_user/my-custom-kernel")
        assert res["status"] == "complete"

        call_args = mock_open.call_args[0]
        req = call_args[0]
        assert "userName=custom_user" in req.full_url
        assert "kernelSlug=my-custom-kernel" in req.full_url


def test_quota_boundary_conditions():
    """Asserts exact boundary values (0.0h, 30.0h, negative hours) for quota ledger."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))

        assert ledger.can_dispatch(0.0) is True
        assert ledger.can_dispatch(-1.0) is False
        assert ledger.can_dispatch(30.0) is True
        assert ledger.can_dispatch(30.0001) is False

        # Exact fill
        ledger.record_usage("job-exact", 30.0)
        assert ledger.get_used_gpu_hours() == 30.0
        assert ledger.get_remaining_gpu_hours() == 0.0
        assert ledger.can_dispatch(0.0) is True
        assert ledger.can_dispatch(0.0001) is False


def test_quota_corrupted_file_recovery():
    """Asserts that corrupted JSON on disk doesn't crash the quota ledger."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        with open(ledger_file, "w", encoding="utf-8") as f:
            f.write("{invalid json content")

        ledger = KaggleQuotaLedger(ledger_path=ledger_file)
        assert ledger.get_used_gpu_hours() == 0.0
        assert ledger.get_remaining_gpu_hours() == 30.0


def test_real_user_credentials_file_validation():
    """Asserts that the user's actual ~/.kaggle/kaggle.json file is valid if present."""
    user_cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.isfile(user_cred_path):
        pytest.skip("No ~/.kaggle/kaggle.json present on host")

    creds = load_kaggle_credentials(enforce_permissions=True)
    assert len(creds.username) > 0
    assert len(creds.key) > 0
    assert creds.source == user_cred_path


def math_isclose(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# =====================================================================
# 6. Hardening & Deep Verification Tests (URLs, Bundling, Retries)
# =====================================================================

def test_download_kernel_output_with_signed_urls():
    """Verifies download_kernel_output handles remote signed URLs with spaces and subdirs."""
    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    mock_resp_data = {
        "log": "[FRAME-AR] Training finished successfully.",
        "files": [
            {
                "fileName": "checkpoints/sub folder/model_weights.pt",
                "url": "https://storage.googleapis.com/kaggle-fake-bucket/checkpoints/sub%20folder/model_weights.pt",
            },
            {
                "fileName": "metrics.json",
                "url": "https://storage.googleapis.com/kaggle-fake-bucket/metrics.json",
            }
        ]
    }

    pt_bytes = b"\x80\x02\x8a\nFakeTorchWeightsData"
    json_bytes = b'{"val_loss": 0.042}'

    def mock_urlopen_router(req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if "api/v1/kernels/output" in url:
            return _mock_http_response(200, mock_resp_data)
        elif "model_weights.pt" in url:
            return _mock_http_response(200, text_data=None, json_data=None) or mock.MagicMock(read=lambda: pt_bytes)
        elif "metrics.json" in url:
            m = mock.MagicMock()
            m.read.return_value = json_bytes
            m.__enter__.return_value = m
            m.__exit__.return_value = None
            return m
        else:
            m = mock.MagicMock()
            m.read.return_value = b"OK"
            m.__enter__.return_value = m
            m.__exit__.return_value = None
            return m

    # Specifically mock urlopen to handle both API call and binary downloads
    mock_resp = mock.MagicMock()
    mock_resp.read.side_effect = [
        json.dumps(mock_resp_data).encode("utf-8"),  # output API call
        pt_bytes,                                    # first URL download
        json_bytes,                                  # second URL download
    ]
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with mock.patch("urllib.request.urlopen", return_value=mock_resp):
        with tempfile.TemporaryDirectory() as tmp_out:
            saved = client.download_kernel_output("frame-test", destination_dir=tmp_out)

            assert "stdout.log" in saved
            assert "checkpoints/sub folder/model_weights.pt" in saved
            assert "metrics.json" in saved

            with open(saved["checkpoints/sub folder/model_weights.pt"], "rb") as f:
                assert f.read() == pt_bytes

            with open(saved["metrics.json"], "rb") as f:
                assert f.read() == json_bytes


def test_download_kernel_output_direct_zip_archive():
    """Verifies download_kernel_output handles direct binary zip archive responses."""
    import zipfile

    creds = KaggleCredentials(username="researcher", key="tokenxyz")
    client = KaggleClient(credentials=creds)

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("receipt.json", '{"status": "PASS"}')
        zf.writestr("stdout.log", "Zip execution log.")

    zip_bytes = zip_buffer.getvalue()

    with mock.patch.object(client, "get_kernel_output", return_value=zip_bytes):
        with tempfile.TemporaryDirectory() as tmp_out:
            saved = client.download_kernel_output("frame-test", destination_dir=tmp_out)

            assert "receipt.json" in saved
            assert "stdout.log" in saved
            with open(saved["receipt.json"], "r") as f:
                assert json.load(f) == {"status": "PASS"}


def test_bundle_framework_source_and_create_kaggle_bundle():
    """Verifies bundle_framework_source packages frame modules and creates runnable payload."""
    import subprocess
    import sys
    from frame.runners.kaggle.bundler import bundle_framework_source, create_kaggle_bundle

    source_tree = bundle_framework_source()
    assert len(source_tree) >= 30
    assert any("frame/__init__.py" in k for k in source_tree)
    assert any("frame/runners/kaggle/client.py" in k for k in source_tree)
    # Ensure __pycache__ and pyc files are excluded
    assert not any("__pycache__" in k for k in source_tree)
    assert not any(k.endswith(".pyc") for k in source_tree)

    # Test creating self-extracting payload and running in clean subprocess
    main_script = (
        "import frame\n"
        "from frame.runners.kaggle.client import KaggleClient\n"
        "print('FRAME_LOADED_OK')\n"
    )
    bundle_code = create_kaggle_bundle(main_script)
    assert "PAYLOAD_B64" in bundle_code
    assert "FRAME_LOADED_OK" in bundle_code

    with tempfile.TemporaryDirectory() as tmp_exec:
        run_file = os.path.join(tmp_exec, "run_entrypoint.py")
        with open(run_file, "w", encoding="utf-8") as f:
            f.write(bundle_code)

        proc = subprocess.run([sys.executable, run_file], cwd=tmp_exec, capture_output=True, text=True)
        assert proc.returncode == 0, f"Subprocess failed:\n{proc.stderr}"
        assert "FRAME_LOADED_OK" in proc.stdout


def test_watchdog_transient_error_recovery():
    """Verifies that transient errors (HTTP 503, URLError) do not abort the watchdog."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        # Sequence: running -> transient 503 -> transient URLError -> complete
        http_503 = KaggleAPIError("HTTP 503: Service Unavailable", status_code=503)
        net_err = KaggleAPIError("Network error: connection reset", status_code=None)
        status_sequence = [
            {"status": "running"},
            http_503,
            net_err,
            {"status": "complete"},
        ]

        time_val = [100.0]
        def mock_time():
            return time_val[0]

        def mock_sleep(sec):
            time_val[0] += 15.0

        with mock.patch.object(client, "get_kernel_status", side_effect=status_sequence):
            res = client.watch_execution(
                slug="transient-slug",
                poll_interval_sec=15.0,
                time_fn=mock_time,
                sleep_fn=mock_sleep,
            )
            assert res["status"] == "complete"
            assert ledger.get_used_gpu_hours() > 0.0


def test_watchdog_cancelled_status_handling():
    """Verifies watchdog terminates immediately on 'cancelled' or 'cancelAck' status."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger = KaggleQuotaLedger(ledger_path=os.path.join(tmp_dir, "quota.json"))
        client = KaggleClient(
            credentials=KaggleCredentials(username="user", key="key"),
            quota_ledger=ledger,
        )

        status_responses = [
            {"status": "running"},
            {"status": "cancelled", "failureMessage": "Kernel cancelled by user."},
        ]

        with mock.patch.object(client, "get_kernel_status", side_effect=status_responses):
            with pytest.raises(KaggleExecutionError) as exc_info:
                client.watch_execution(
                    slug="cancelled-slug",
                    sleep_fn=lambda s: None,
                )
            assert "cancelled" in str(exc_info.value).lower()


def test_quota_ledger_concurrent_merge():
    """Verifies that separate ledger instances do not clobber each other's records."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ledger_file = os.path.join(tmp_dir, "quota.json")
        ledger_a = KaggleQuotaLedger(ledger_path=ledger_file)
        ledger_b = KaggleQuotaLedger(ledger_path=ledger_file)

        ledger_a.record_usage("run-from-a", 2.5)
        ledger_b.record_usage("run-from-b", 3.5)

        # Both records must be present
        reloaded = KaggleQuotaLedger(ledger_path=ledger_file)
        assert reloaded.get_used_gpu_hours() == 6.0
        assert len(reloaded.data.entries) == 2


def test_client_check_connection_and_list_kernels():
    """Verifies client.check_connection() and client.list_kernels()."""
    creds = KaggleCredentials(username="test_researcher", key="tok")
    client = KaggleClient(credentials=creds)

    with mock.patch.object(client, "_request") as mock_req:
        mock_req.return_value = [{"ref": "test_researcher/k1", "title": "K1"}]

        conn = client.check_connection()
        assert conn["authenticated"] is True
        assert conn["username"] == "test_researcher"

        kernels = client.list_kernels(page_size=5)
        assert len(kernels) == 1
        assert kernels[0]["ref"] == "test_researcher/k1"
