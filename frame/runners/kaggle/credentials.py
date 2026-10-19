"""Kaggle API Credentials Configuration and POSIX Permission Enforcement.

Resolution order:
1. Environment variables: `KAGGLE_USERNAME` & `KAGGLE_KEY`
2. User credential file: `~/.kaggle/kaggle.json`
3. Local framework credential file: `.frame/kaggle.json`

Enforces POSIX 0600 file permissions on credential files to guarantee secret isolation.
"""

import base64
import json
import os
import stat
from typing import Optional
from pydantic import BaseModel, Field


class KaggleCredentials(BaseModel):
    """Container for validated Kaggle API credentials."""
    username: str = Field(..., min_length=1, description="Kaggle user account identifier")
    key: str = Field(..., min_length=1, description="Kaggle API key / token")
    source: str = Field(default="unknown", description="Source of credentials (env or filepath)")

    def get_basic_auth_header(self) -> str:
        """Generates RFC 7617 HTTP Basic Authentication header string.
        
        Format: 'Basic base64("{username}:{key}")'
        """
        raw_token = f"{self.username}:{self.key}".encode("utf-8")
        b64_token = base64.b64encode(raw_token).decode("ascii")
        return f"Basic {b64_token}"


def check_file_permissions(filepath: str, expected_mode: int = 0o600) -> None:
    """Verifies that a credential file strictly matches expected POSIX permissions (0600 or 0400).
    
    Args:
        filepath: Absolute or relative path to the credential file.
        expected_mode: Expected octal permission mode (default 0o600: read/write by owner only).
        
    Raises:
        FileNotFoundError: If the credential file does not exist.
        PermissionError: If file permissions are more open than expected_mode on POSIX systems.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Credential file not found at: {filepath}")

    if os.name == "posix":
        st = os.stat(filepath)
        actual_mode = stat.S_IMODE(st.st_mode)
        # Check if group or other have any permissions (insecure) or permissions are broader than expected
        if (actual_mode & 0o077 != 0) or (actual_mode not in (expected_mode, 0o400)):
            raise PermissionError(
                f"Insecure permissions on credential file '{filepath}': "
                f"expected {oct(expected_mode)} (rw-------), got {oct(actual_mode)}. "
                f"Run 'chmod 600 {filepath}' to secure credentials."
            )


def load_kaggle_credentials(
    project_dir: Optional[str] = None,
    enforce_permissions: bool = True,
) -> KaggleCredentials:
    """Resolves and loads Kaggle API credentials according to the specification hierarchy.
    
    Resolution order:
    1. Environment variables: `KAGGLE_USERNAME` & `KAGGLE_KEY`
    2. User configuration: `~/.kaggle/kaggle.json`
    3. Project configuration: `<project_dir>/.frame/kaggle.json`
    
    Args:
        project_dir: Optional custom project directory. Defaults to current working directory.
        enforce_permissions: When True, checks that credential files have strict POSIX 0600 mode.
        
    Returns:
        KaggleCredentials containing username, API key, and source metadata.
        
    Raises:
        FileNotFoundError: If no valid credentials are found across all resolution targets.
        PermissionError: If a credential file violates POSIX 0600 permissions.
        ValueError: If a credential file exists but contains invalid or empty credentials.
    """
    # 1. Environment variables
    env_user = os.environ.get("KAGGLE_USERNAME", "").strip()
    env_key = os.environ.get("KAGGLE_KEY", "").strip()
    if env_user and env_key:
        return KaggleCredentials(username=env_user, key=env_key, source="env")

    # 2. Custom config dir from KAGGLE_CONFIG_DIR env var
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR", "").strip()
    if config_dir:
        config_path = os.path.join(os.path.expanduser(config_dir), "kaggle.json")
        if os.path.isfile(config_path):
            if enforce_permissions:
                check_file_permissions(config_path, expected_mode=0o600)
            return _parse_credential_file(config_path)

    # 3. User directory ~/.kaggle/kaggle.json
    user_cred_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.isfile(user_cred_path):
        if enforce_permissions:
            check_file_permissions(user_cred_path, expected_mode=0o600)
        return _parse_credential_file(user_cred_path)

    # 4. Project directory .frame/kaggle.json
    base_dir = project_dir or os.getcwd()
    proj_cred_path = os.path.join(base_dir, ".frame", "kaggle.json")
    if os.path.isfile(proj_cred_path):
        if enforce_permissions:
            check_file_permissions(proj_cred_path, expected_mode=0o600)
        return _parse_credential_file(proj_cred_path)

    raise FileNotFoundError(
        "Kaggle API credentials not found. Resolution hierarchy checked:\n"
        "  1. Environment variables: KAGGLE_USERNAME & KAGGLE_KEY\n"
        f"  2. Custom config dir: {config_dir if config_dir else 'unset'}\n"
        f"  3. User credential file: {user_cred_path}\n"
        f"  4. Project credential file: {proj_cred_path}\n"
        "Please provide credentials via environment or 'kaggle.json' with permissions 0600."
    )


def save_kaggle_credentials(
    credentials: KaggleCredentials,
    filepath: Optional[str] = None,
    mode: int = 0o600,
) -> str:
    """Saves Kaggle credentials to disk and strictly sets POSIX 0600 permissions.
    
    Args:
        credentials: Validated KaggleCredentials instance.
        filepath: Target file path. Defaults to `.frame/kaggle.json`.
        mode: POSIX file mode (defaults to 0o600).
        
    Returns:
        Absolute path to the saved credential file.
    """
    target = filepath or os.path.join(os.getcwd(), ".frame", "kaggle.json")
    target_abs = os.path.abspath(target)
    parent_dir = os.path.dirname(target_abs)
    os.makedirs(parent_dir, exist_ok=True)

    content = json.dumps(
        {"username": credentials.username, "key": credentials.key},
        indent=2,
    )
    with open(target_abs, "w", encoding="utf-8") as f:
        f.write(content + "\n")

    if os.name == "posix":
        os.chmod(target_abs, mode)

    return target_abs


def _parse_credential_file(filepath: str) -> KaggleCredentials:
    """Helper to parse and validate a kaggle.json file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to read/parse Kaggle JSON file at '{filepath}': {e}") from e

    username = str(data.get("username", "")).strip()
    key = str(data.get("key", "")).strip()

    if not username or not key:
        raise ValueError(
            f"Invalid Kaggle credentials in '{filepath}': both 'username' and 'key' fields are required."
        )

    return KaggleCredentials(username=username, key=key, source=filepath)
