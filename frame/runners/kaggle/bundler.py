"""Kaggle Single-File Self-Extracting Payload Bundler.

Kaggle's API (/api/v1/kernels/push) only transmits the single code_file string and ignores
all auxiliary directory files. This bundler compresses all necessary modules and data into an
in-memory Base64 archive string embedded inside a standalone Python wrapper script.
"""

import base64
import io
import os
import sys
import tarfile
from typing import Dict, Optional, Union


def bundle_framework_source(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Walks the FRAME-AR repository and collects all Python source files for remote execution.
    
    Args:
        base_dir: Directory containing the 'frame' package. Defaults to the repository root.
        
    Returns:
        Dictionary mapping relative file paths (e.g. 'frame/runners/kaggle/client.py') to their text contents.
    """
    if base_dir is None:
        # Default: traverse up from this file until we find the parent of 'frame'
        curr = os.path.abspath(os.path.dirname(__file__))
        while curr and curr != "/":
            parent = os.path.dirname(curr)
            if os.path.isdir(os.path.join(curr, "frame")):
                base_dir = curr
                break
            curr = parent
        if base_dir is None:
            base_dir = os.getcwd()

    frame_root = os.path.join(base_dir, "frame")
    if not os.path.isdir(frame_root):
        # Fallback to base_dir if already pointing to frame directory
        if os.path.basename(base_dir) == "frame":
            frame_root = base_dir
            base_dir = os.path.dirname(base_dir)
        else:
            frame_root = os.path.abspath("frame")
            base_dir = os.path.dirname(frame_root)

    bundled: Dict[str, str] = {}
    for root, dirs, files in os.walk(frame_root):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".pytest_cache", ".git")]
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, base_dir)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        bundled[rel_path] = f.read()
                except UnicodeDecodeError:
                    pass

    return bundled


def create_self_extracting_payload(
    files: Dict[str, Union[str, bytes]],
    main_execution_script: str,
) -> str:
    """Packages a dictionary of {rel_path: content} into a self-extracting Python script.
    
    Supports both text (str) and binary (bytes) file payloads.
    
    Args:
        files: Dictionary mapping relative file paths to string or bytes contents.
        main_execution_script: Python code to execute after the files are extracted.
        
    Returns:
        A standalone Python script string suitable for Kaggle kernels push.
    """
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        for rel_path, content in files.items():
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = bytes(content)
            tarinfo = tarfile.TarInfo(name=rel_path)
            tarinfo.size = len(content_bytes)
            tarinfo.mtime = 1700000000
            tar.addfile(tarinfo, io.BytesIO(content_bytes))
            
    tar_bytes = tar_buffer.getvalue()
    b64_payload = base64.b64encode(tar_bytes).decode("ascii")

    bootstrap_template = f'''"""Kaggle Autonomous Research Self-Extracting Container Entrypoint."""
import base64
import io
import os
import sys
import tarfile

# 1. IEEE 754 Precision Protection (Disable TF32 for deterministic host verification)
try:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
except ImportError:
    pass

# 2. Unpack embedded payload archive
WORKING_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else os.path.abspath("./working")
os.makedirs(WORKING_DIR, exist_ok=True)
SRC_DIR = os.path.join(WORKING_DIR, "frame_src")
os.makedirs(SRC_DIR, exist_ok=True)

PAYLOAD_B64 = "{b64_payload}"

payload_bytes = base64.b64decode(PAYLOAD_B64)
tar_buffer = io.BytesIO(payload_bytes)
with tarfile.open(fileobj=tar_buffer, mode="r:gz") as tar:
    try:
        tar.extractall(path=SRC_DIR, filter="data")
    except TypeError:
        tar.extractall(path=SRC_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# 3. Execute main experiment payload
{main_execution_script}
'''
    return bootstrap_template


def create_kaggle_bundle(
    main_execution_script: str,
    extra_files: Optional[Dict[str, Union[str, bytes]]] = None,
    base_dir: Optional[str] = None,
) -> str:
    """Bundles the entire FRAME-AR source framework together with the main execution script.
    
    Args:
        main_execution_script: The Python script to execute on Kaggle.
        extra_files: Optional auxiliary code, data, or config files to include.
        base_dir: Optional base directory to find the 'frame' package.
        
    Returns:
        Self-extracting Python container script.
    """
    bundled_files = bundle_framework_source(base_dir=base_dir)
    if extra_files:
        bundled_files.update(extra_files)
    return create_self_extracting_payload(bundled_files, main_execution_script)
