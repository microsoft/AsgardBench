"""
Utilities for reading JSON files, especially on blobfuse-mounted storage.

Provides robust JSON reading with retry logic and fallback to Azure direct download.
"""

import json
import os
import subprocess
import tempfile
import time
from typing import Any


class FileReadError(Exception):
    """Raised when a file cannot be read or parsed correctly."""

    pass


def read_json_file(file_path: str, max_retries: int = 3) -> Any:
    """Read and parse a JSON file with retry logic for blobfuse issues.

    Args:
        file_path: Path to the JSON file.
        max_retries: Number of retries for truncated files (blobfuse workaround).

    Returns:
        Parsed JSON data.

    Raises:
        FileReadError: If the file cannot be read, is incomplete, or has invalid JSON.
    """

    def invalidate_blobfuse_cache(fpath: str):
        """Try to invalidate blobfuse cache for a file."""
        try:
            # Delete the cached file to force re-fetch
            cache_base = "/tmp/blobfuse2_cache"
            # The cache path mirrors the mount structure
            if fpath.startswith("/mnt/magmathor/"):
                relative_path = fpath[len("/mnt/magmathor/") :]
                cache_file = os.path.join(cache_base, relative_path)
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    return True
        except Exception:
            pass
        return False

    def read_with_cat(fpath: str) -> str:
        """Use cat subprocess to read file, bypassing Python's buffering."""
        result = subprocess.run(["cat", fpath], capture_output=True, timeout=60)
        if result.returncode != 0:
            raise OSError(f"cat failed: {result.stderr.decode()}")
        return result.stdout.decode("utf-8")

    def download_from_azure(fpath: str) -> str:
        """Download file directly from Azure, bypassing blobfuse."""
        if not fpath.startswith("/mnt/magmathor/"):
            raise OSError("Not a blobfuse path")
        blob_name = fpath[len("/mnt/magmathor/") :]
        # Normalize path - remove double slashes
        while "//" in blob_name:
            blob_name = blob_name.replace("//", "/")
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [
                    "az",
                    "storage",
                    "blob",
                    "download",
                    "--account-name",
                    "magmardata",
                    "--container-name",
                    "magmathor",
                    "--name",
                    blob_name,
                    "--file",
                    tmp_path,
                    "--auth-mode",
                    "login",
                    "--no-progress",
                ],
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise OSError(f"az download failed: {result.stderr.decode()}")
            with open(tmp_path, "r", encoding="utf-8") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def check_content(content: str, fpath: str) -> None:
        """Check if content appears complete, raise FileReadError if not."""
        if not content:
            raise FileReadError(f"File is empty: {fpath}")
        content_stripped = content.strip()
        if not content_stripped:
            raise FileReadError(f"File contains only whitespace: {fpath}")
        if content_stripped.startswith("{") and not content_stripped.endswith("}"):
            raise FileReadError(
                f"File appears truncated (JSON object not closed): {fpath}"
            )
        if content_stripped.startswith("[") and not content_stripped.endswith("]"):
            raise FileReadError(
                f"File appears truncated (JSON array not closed): {fpath}"
            )

    last_error = None
    for attempt in range(max_retries):
        try:
            # On retry, try to invalidate cache first
            if attempt > 0:
                invalidate_blobfuse_cache(file_path)
                time.sleep(1)  # Give blobfuse time to notice

            # Use cat to bypass Python file caching
            content = read_with_cat(file_path)
            check_content(content, file_path)
            data = json.loads(content)
            return data

        except FileReadError as e:
            last_error = e
            if attempt < max_retries - 1:
                # Wait and retry - blobfuse may need time to fetch full file
                wait_time = (attempt + 1) * 2  # 2s, 4s, 6s
                print(
                    f"  Retrying {file_path} in {wait_time}s (attempt {attempt + 2}/{max_retries})..."
                )
                time.sleep(wait_time)
            else:
                # Last resort: download directly from Azure
                if file_path.startswith("/mnt/magmathor/"):
                    print(
                        f"  Blobfuse cache issue, downloading directly from Azure: {file_path}"
                    )
                    try:
                        content = download_from_azure(file_path)
                        check_content(content, file_path)
                        data = json.loads(content)
                        return data
                    except Exception as az_err:
                        raise FileReadError(
                            f"Failed via blobfuse and Azure direct: {file_path}. "
                            f"Blobfuse error: {last_error}. Azure error: {az_err}"
                        ) from az_err
                raise last_error

        except json.JSONDecodeError as e:
            raise FileReadError(f"Invalid JSON in {file_path}: {e}") from e
        except OSError as e:
            raise FileReadError(f"Cannot read file {file_path}: {e}") from e

    # Should not reach here, but just in case
    raise last_error if last_error else FileReadError(f"Failed to read {file_path}")
