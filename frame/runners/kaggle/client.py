"""Zero-Dependency Native Kaggle REST Client for FRAME-AR.

Communicates with Kaggle's REST API (https://www.kaggle.com/api/v1) using Python standard
library `urllib.request` with HTTP Basic Authentication. Requires no external CLI binaries.

Features:
- Basic Auth header: `Authorization: Basic base64("{username}:{key}")`
- Kernel Push: `POST /api/v1/kernels/push`
- Kernel Status: `GET /api/v1/kernels/status?userName={username}&kernelSlug={slug}`
- Kernel Output Download: `GET /api/v1/kernels/output?userName={username}&kernelSlug={slug}`
- Active execution watchdog (8.5h safety ceiling) to prevent timeout without checkpoint saving.
"""

import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from frame.runners.kaggle.credentials import KaggleCredentials, load_kaggle_credentials
from frame.runners.kaggle.quota import KaggleQuotaLedger, KaggleQuotaExceededError
from frame.runners.kaggle.bundler import create_self_extracting_payload, bundle_framework_source

DEFAULT_KAGGLE_API_BASE: str = "https://www.kaggle.com/api/v1"
WATCHDOG_SAFETY_CEILING_SEC: float = 8.5 * 3600.0  # 30,600 s (8.5 hours)


class KaggleAPIError(RuntimeError):
    """Base exception for Kaggle REST API failures."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class KaggleWatchdogTimeoutError(KaggleAPIError):
    """Raised when execution duration hits the 8.5h safety ceiling without completing."""
    pass


class KaggleExecutionError(KaggleAPIError):
    """Raised when Kaggle reports that kernel execution ended in 'error' status."""
    pass


class KaggleClient:
    """Native zero-dependency Kaggle REST client using urllib.request."""

    def __init__(
        self,
        credentials: Optional[KaggleCredentials] = None,
        base_url: str = DEFAULT_KAGGLE_API_BASE,
        timeout_seconds: float = 30.0,
        quota_ledger: Optional[KaggleQuotaLedger] = None,
    ):
        """Initializes the Kaggle REST client.
        
        Args:
            credentials: KaggleCredentials instance. If None, loaded via `load_kaggle_credentials()`.
            base_url: Kaggle API base URL (default: 'https://www.kaggle.com/api/v1').
            timeout_seconds: Network socket timeout for HTTP calls.
            quota_ledger: KaggleQuotaLedger instance for tracking GPU hours. If None, initialized
                          from default location.
        """
        self.credentials = credentials if credentials is not None else load_kaggle_credentials()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.quota_ledger = quota_ledger if quota_ledger is not None else KaggleQuotaLedger()

    def _extract_username(self, slug: str) -> Optional[str]:
        """Extracts username if slug is in 'username/slug' format."""
        if "/" in slug:
            return slug.split("/", 1)[0]
        return None

    def _clean_slug(self, slug: str) -> str:
        """Strips username prefix if slug is in 'username/slug' format."""
        if "/" in slug:
            return slug.split("/", 1)[1]
        return slug.strip()

    def _request(
        self,
        method: str,
        endpoint: str,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        raw_response: bool = False,
    ) -> Any:
        """Executes an authenticated HTTP request against the Kaggle REST API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query_params:
            filtered_params = {k: v for k, v in query_params.items() if v is not None}
            url = f"{url}?{urllib.parse.urlencode(filtered_params)}"

        headers = {
            "Authorization": self.credentials.get_basic_auth_header(),
            "User-Agent": "FRAME-AR-Native-Kaggle-Client/1.0",
            "Accept": "application/json, */*",
        }

        data_bytes = None
        if body is not None:
            data_bytes = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(
            url=url,
            data=data_bytes,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as response:
                content = response.read()
                if raw_response:
                    return content
                # If content is a zip archive or gzip binary payload, return raw bytes
                if content.startswith(b"PK\x03\x04") or content.startswith(b"\x1f\x8b"):
                    return content
                try:
                    return json.loads(content.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return content.decode("utf-8", errors="replace")

        except urllib.error.HTTPError as e:
            raw_err = e.read().decode("utf-8", errors="replace")
            err_msg = f"HTTP {e.code}: {e.reason}"
            try:
                err_json = json.loads(raw_err)
                if isinstance(err_json, dict) and "message" in err_json:
                    err_msg = f"HTTP {e.code}: {err_json['message']}"
            except Exception:
                pass
            raise KaggleAPIError(
                f"Kaggle API request failed: {err_msg}",
                status_code=e.code,
                response_body=raw_err,
            ) from e

        except urllib.error.URLError as e:
            raise KaggleAPIError(f"Network error communicating with Kaggle API: {e.reason}") from e

    def push_kernel(
        self,
        slug: str,
        text: str,
        new_title: Optional[str] = None,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        is_private: bool = True,
        language: str = "python",
        kernel_type: str = "script",
    ) -> Dict[str, Any]:
        """Pushes kernel code to Kaggle via POST /api/v1/kernels/push.
        
        Args:
            slug: Target kernel slug (e.g. 'frame-opt-attention').
            text: Raw Python script content to execute.
            new_title: Title displayed on Kaggle. Defaults to title-cased slug.
            enable_gpu: Whether to attach GPU (Tesla P100 / T4). Defaults to True.
            enable_internet: Whether to enable outbound internet. Defaults to True.
            is_private: Whether the kernel is private. Defaults to True.
            language: Programming language ('python').
            kernel_type: Kernel type ('script').
            
        Returns:
            JSON response from Kaggle API containing kernel reference and URL.
        """
        clean_slug = self._clean_slug(slug)
        title = new_title or clean_slug.replace("-", " ").title()

        payload = {
            "slug": clean_slug,
            "newTitle": title,
            "text": text,
            "enableGpu": enable_gpu,
            "enableInternet": enable_internet,
            "isPrivate": is_private,
            "language": language,
            "kernelType": kernel_type,
        }

        res = self._request("POST", "kernels/push", body=payload)
        return res if isinstance(res, dict) else {"response": res}

    def get_kernel_status(
        self,
        slug: str,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Queries kernel execution status via GET /api/v1/kernels/status.
        
        Args:
            slug: Target kernel slug.
            username: Kaggle username. If None, derived from slug or credentials.
            
        Returns:
            Dict containing 'status' ('queued' | 'running' | 'complete' | 'error'),
            and any failure messages.
        """
        user = username or self._extract_username(slug) or self.credentials.username
        clean_slug = self._clean_slug(slug)

        params = {
            "userName": user,
            "kernelSlug": clean_slug,
        }
        res = self._request("GET", "kernels/status", query_params=params)
        return res if isinstance(res, dict) else {"status": str(res)}

    def get_kernel_output(
        self,
        slug: str,
        username: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieves kernel outputs and stdout via GET /api/v1/kernels/output.
        
        Args:
            slug: Target kernel slug.
            username: Kaggle username. If None, derived from slug or credentials.
            
        Returns:
            Dict containing output files, log content, and metadata.
        """
        user = username or self._extract_username(slug) or self.credentials.username
        clean_slug = self._clean_slug(slug)

        params = {
            "userName": user,
            "kernelSlug": clean_slug,
        }
        res = self._request("GET", "kernels/output", query_params=params)
        return res if isinstance(res, dict) else {"output": res}

    def check_connection(self) -> Dict[str, Any]:
        """Probes Kaggle REST API connectivity and authentication using authentic endpoints.
        
        Returns:
            Dict containing connection verification details.
        """
        # Call authentic endpoint: datasets/list?pageSize=1
        res = self._request("GET", "datasets/list", query_params={"pageSize": 1})
        return {
            "authenticated": True,
            "username": self.credentials.username,
            "api_base": self.base_url,
            "status": "connected",
        }

    def list_kernels(
        self,
        username: Optional[str] = None,
        page_size: int = 20,
    ) -> List[Dict[str, Any]]:
        """Lists kernels owned by the user or specified username.
        
        Args:
            username: Kaggle username to query. Defaults to authenticated username.
            page_size: Maximum number of kernels to return.
            
        Returns:
            List of kernel metadata dictionaries.
        """
        user = username or self.credentials.username
        res = self._request("GET", "kernels/list", query_params={"user": user, "pageSize": page_size})
        return res if isinstance(res, list) else []

    def download_kernel_output(
        self,
        slug: str,
        destination_dir: str,
        username: Optional[str] = None,
    ) -> Dict[str, str]:
        """Downloads kernel output artifacts and stdout log to local destination_dir.
        
        Handles both direct content payloads, remote download URLs (with character quoting),
        and raw binary zip archives.
        
        Args:
            slug: Target kernel slug.
            destination_dir: Local directory path to write downloaded artifacts.
            username: Optional Kaggle username.
            
        Returns:
            Dictionary mapping artifact filenames to local absolute paths.
        """
        os.makedirs(destination_dir, exist_ok=True)
        output_data = self.get_kernel_output(slug, username=username)
        saved_paths: Dict[str, str] = {}

        # 1. Handle direct binary archive response (zip)
        if isinstance(output_data, bytes) or (isinstance(output_data, str) and output_data.startswith("PK\x03\x04")):
            raw_bytes = output_data if isinstance(output_data, bytes) else output_data.encode("latin-1")
            try:
                with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
                    zf.extractall(destination_dir)
                    for info in zf.infolist():
                        saved_paths[info.filename] = os.path.join(destination_dir, info.filename)
                return saved_paths
            except zipfile.BadZipFile:
                pass

        if not isinstance(output_data, dict):
            return saved_paths

        # 2. Save execution log
        log_content = output_data.get("log")
        if log_content:
            log_path = os.path.join(destination_dir, "stdout.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(str(log_content))
            saved_paths["stdout.log"] = log_path

        # 3. Save raw output files or download from URLs
        files = output_data.get("files", [])
        if isinstance(files, list):
            for file_entry in files:
                if not isinstance(file_entry, dict):
                    continue
                fname = file_entry.get("fileName") or file_entry.get("name")
                if not fname:
                    continue
                fpath = os.path.join(destination_dir, fname)
                os.makedirs(os.path.dirname(fpath), exist_ok=True)

                content = file_entry.get("content") or file_entry.get("data")
                file_url = file_entry.get("url") or file_entry.get("urlNullable")

                if content is not None:
                    if isinstance(content, str):
                        with open(fpath, "w", encoding="utf-8") as f:
                            f.write(content)
                    else:
                        with open(fpath, "wb") as f:
                            f.write(bytes(content))
                    saved_paths[fname] = fpath

                elif file_url:
                    # Download signed URL, safely quoting spaces and special characters in path
                    parts = urllib.parse.urlsplit(str(file_url))
                    quoted_path = urllib.parse.quote(parts.path)
                    safe_url = urllib.parse.urlunsplit((parts.scheme, parts.netloc, quoted_path, parts.query, parts.fragment))
                    dl_req = urllib.request.Request(
                        url=safe_url,
                        headers={"User-Agent": "FRAME-AR-Native-Kaggle-Client/1.0"},
                    )
                    with urllib.request.urlopen(dl_req, timeout=self.timeout_seconds) as dl_resp:
                        with open(fpath, "wb") as f:
                            f.write(dl_resp.read())
                    saved_paths[fname] = fpath

        return saved_paths

    def watch_execution(
        self,
        slug: str,
        username: Optional[str] = None,
        poll_interval_sec: float = 15.0,
        timeout_sec: float = WATCHDOG_SAFETY_CEILING_SEC,
        status_callback: Optional[Callable[[str, float], None]] = None,
        record_quota: bool = True,
        hypothesis_id: Optional[str] = None,
        time_fn: Callable[[], float] = time.time,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> Dict[str, Any]:
        """Monitors Kaggle kernel execution with an 8.5h active watchdog.
        
        The watchdog asserts that runtime does not exceed `timeout_sec` (8.5h safety ceiling).
        This ensures execution is aborted or handled before Kaggle's 9.0h hard timeout,
        protecting checkpoints and intermediate state.
        
        Args:
            slug: Target kernel slug.
            username: Kaggle username.
            poll_interval_sec: Seconds between status polls (default 15s).
            timeout_sec: Maximum allowable execution seconds (default 8.5h = 30600s).
            status_callback: Optional callback `(status_str, elapsed_sec) -> None`.
            record_quota: When True, records consumed GPU hours in local quota ledger.
            hypothesis_id: Optional research hypothesis ID to attach to quota record.
            time_fn: Injectable time provider (for deterministic testing).
            sleep_fn: Injectable sleep provider (for deterministic testing).
            
        Returns:
            Dict containing terminal status response.
            
        Raises:
            KaggleWatchdogTimeoutError: If execution exceeds 8.5h safety ceiling.
            KaggleExecutionError: If kernel status transitions to 'error' or 'cancelled'.
        """
        start_time = time_fn()
        clean_slug = self._clean_slug(slug)
        consecutive_transient_errors = 0
        max_transient_retries = 6

        while True:
            now = time_fn()
            elapsed_sec = now - start_time

            if elapsed_sec > timeout_sec:
                hours = elapsed_sec / 3600.0
                if record_quota and self.quota_ledger:
                    self.quota_ledger.record_usage(
                        job_slug=clean_slug,
                        gpu_hours=hours,
                        hypothesis_id=hypothesis_id,
                        status="timeout_watchdog",
                    )
                raise KaggleWatchdogTimeoutError(
                    f"Kernel '{clean_slug}' exceeded active execution watchdog safety ceiling "
                    f"of {timeout_sec / 3600.0:.2f}h ({timeout_sec:.0f}s). Aborting to preserve checkpoints.",
                    response_body=f"Elapsed: {elapsed_sec:.1f}s",
                )

            try:
                status_dict = self.get_kernel_status(clean_slug, username=username)
                consecutive_transient_errors = 0
            except KaggleAPIError as e:
                is_transient = (e.status_code in (429, 500, 502, 503, 504) or e.status_code is None)
                if is_transient and consecutive_transient_errors < max_transient_retries:
                    consecutive_transient_errors += 1
                    sleep_fn(poll_interval_sec)
                    continue
                raise

            status = str(status_dict.get("status", "")).lower()

            if status_callback is not None:
                status_callback(status, elapsed_sec)

            if status in ("complete", "completed"):
                if record_quota and self.quota_ledger:
                    self.quota_ledger.record_usage(
                        job_slug=clean_slug,
                        gpu_hours=elapsed_sec / 3600.0,
                        hypothesis_id=hypothesis_id,
                        status="complete",
                    )
                status_dict["wall_clock_seconds"] = elapsed_sec
                return status_dict

            if status in ("error", "failed", "cancelled", "cancelack"):
                if record_quota and self.quota_ledger:
                    self.quota_ledger.record_usage(
                        job_slug=clean_slug,
                        gpu_hours=elapsed_sec / 3600.0,
                        hypothesis_id=hypothesis_id,
                        status=status,
                    )
                msg = status_dict.get("failureMessage") or f"Kaggle kernel execution terminated with status '{status}'."
                raise KaggleExecutionError(
                    f"Execution failed for kernel '{clean_slug}': {msg}",
                    response_body=json.dumps(status_dict),
                )

            sleep_fn(poll_interval_sec)

    def bundle_and_dispatch(
        self,
        slug: str,
        files: Optional[Union[Dict[str, Any], str]] = None,
        main_execution_script: Optional[str] = None,
        title: Optional[str] = None,
        estimated_gpu_hours: float = 1.0,
        enable_gpu: bool = True,
        enable_internet: bool = True,
        is_private: bool = True,
        hypothesis_id: Optional[str] = None,
        auto_watch: bool = False,
        include_framework_source: bool = True,
    ) -> Dict[str, Any]:
        """Packages an experiment into a self-extracting payload, checks quota, and pushes to Kaggle.
        
        Args:
            slug: Kernel slug.
            files: Dictionary mapping relative file paths to code contents, or main script if single arg.
            main_execution_script: Python code to execute after extraction.
            title: Title for the Kaggle kernel.
            estimated_gpu_hours: Expected GPU hours, checked against the rolling 7-day quota ledger.
            enable_gpu: Whether to request GPU runtime.
            enable_internet: Whether to enable internet access.
            is_private: Whether to keep the kernel private.
            hypothesis_id: Associated research hypothesis ID.
            auto_watch: If True, blocks and monitors execution under the 8.5h safety watchdog.
            include_framework_source: If True, bundles the entire FRAME-AR source package into the payload.
            
        Returns:
            Push response or terminal execution status if auto_watch is True.
        """
        # Handle flexible argument signatures: (slug, script) or (slug, files, script)
        if isinstance(files, str) and main_execution_script is None:
            actual_script = files
            files_dict: Dict[str, Any] = {}
        else:
            actual_script = main_execution_script or ""
            files_dict = dict(files) if files else {}

        # 1. Assert local rolling quota is available
        if enable_gpu and self.quota_ledger:
            self.quota_ledger.assert_quota_available(estimated_gpu_hours)

        # 2. Package self-extracting payload, optionally including full framework source
        final_files: Dict[str, Any] = {}
        if include_framework_source:
            final_files.update(bundle_framework_source())
        final_files.update(files_dict)

        payload_text = create_self_extracting_payload(
            files=final_files,
            main_execution_script=actual_script,
        )

        # 3. Push to Kaggle
        push_result = self.push_kernel(
            slug=slug,
            text=payload_text,
            new_title=title,
            enable_gpu=enable_gpu,
            enable_internet=enable_internet,
            is_private=is_private,
        )

        # 4. Optional active execution watchdog
        if auto_watch:
            return self.watch_execution(
                slug=slug,
                hypothesis_id=hypothesis_id,
                record_quota=enable_gpu,
            )

        return push_result
