#!/usr/bin/env python3
"""
Utility to move directories matching specific configurations from source to destination.

Usage:
    python scripts/move_configs.py

This provides a simple UI to:
1. Enter configuration patterns to match
2. Specify source and destination directories
3. Preview matching directories before moving

Uses Azure native copy APIs for server-side copies when available (much faster).
"""

import json
import os
import shutil
import subprocess
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import List, Optional, Tuple

# Load .env file if present
try:
    from dotenv import load_dotenv

    # Look for .env in the project root (parent of scripts/)
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Also try current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)
except ImportError:
    pass  # python-dotenv not installed, rely on environment variables

# Try to import Azure SDK
try:
    from azure.core.exceptions import AzureError
    from azure.storage.blob import BlobClient, BlobServiceClient

    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

# Base path for all operations
BASE_PATH = "/mnt/magmathor"

# Cache file for storing found matches
CACHE_FILE = Path(__file__).resolve().parent / ".move_configs_cache.json"

# Azure configuration - update these for your environment
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_CONTAINER_NAME = "magmathor"  # The blob container name


def is_azcopy_available() -> bool:
    """Check if azcopy is installed and available."""
    try:
        result = subprocess.run(
            ["azcopy", "--version"], capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_storage_account_info() -> Tuple[Optional[str], Optional[str]]:
    """Extract storage account name and key from connection string."""
    if not AZURE_STORAGE_CONNECTION_STRING:
        return None, None

    # Parse connection string - be careful with values that contain '=' (like base64 keys)
    parts = {}
    for part in AZURE_STORAGE_CONNECTION_STRING.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            parts[key] = value

    account_name = parts.get("AccountName")
    account_key = parts.get("AccountKey")
    return account_name, account_key


AZCOPY_AVAILABLE = is_azcopy_available()


def get_blob_service_client() -> Optional["BlobServiceClient"]:
    """Get Azure Blob Service client if available and configured."""
    if not AZURE_AVAILABLE:
        return None
    if not AZURE_STORAGE_CONNECTION_STRING:
        return None
    try:
        return BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    except Exception:
        return None


def path_to_blob_name(full_path: str) -> str:
    """Convert a local path to Azure blob name (relative to container)."""
    # Remove the base mount path to get the blob name
    if full_path.startswith(BASE_PATH):
        return full_path[len(BASE_PATH) :].lstrip("/")
    return full_path


def copy_blob_directory(
    blob_service: "BlobServiceClient",
    source_prefix: str,
    dest_prefix: str,
    progress_callback: callable = None,
) -> Tuple[int, List[str]]:
    """
    Copy all blobs under source_prefix to dest_prefix using server-side copy.

    Returns tuple of (success_count, error_messages)
    """
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    success_count = 0
    errors = []

    # List all blobs under source prefix
    blobs = list(container_client.list_blobs(name_starts_with=source_prefix))
    total = len(blobs)

    if total == 0:
        return 0, [f"No blobs found under {source_prefix}"]

    # Copy each blob
    for i, blob in enumerate(blobs):
        source_blob_name = blob.name
        # Replace source prefix with dest prefix
        dest_blob_name = dest_prefix + source_blob_name[len(source_prefix) :]

        if progress_callback and i % 10 == 0:
            progress_callback(i + 1, total, source_blob_name.split("/")[-1])

        try:
            # Get source blob URL
            source_blob = container_client.get_blob_client(source_blob_name)
            dest_blob = container_client.get_blob_client(dest_blob_name)

            # Start server-side copy
            dest_blob.start_copy_from_url(source_blob.url)
            success_count += 1
        except Exception as e:
            errors.append(f"Failed to copy {source_blob_name}: {e}")

    return success_count, errors


def delete_blob_directory(
    blob_service: "BlobServiceClient",
    prefix: str,
) -> Tuple[int, List[str]]:
    """Delete all blobs under prefix."""
    container_client = blob_service.get_container_client(AZURE_CONTAINER_NAME)
    success_count = 0
    errors = []

    blobs = list(container_client.list_blobs(name_starts_with=prefix))

    for blob in blobs:
        try:
            container_client.delete_blob(blob.name)
            success_count += 1
        except Exception as e:
            errors.append(f"Failed to delete {blob.name}: {e}")

    return success_count, errors


def find_matching_directories(
    source_dir: str,
    configs: List[str],
    progress_callback: callable = None,
) -> List[Tuple[str, str]]:
    """
    Find all directories in source that match any of the given configurations.

    Returns list of tuples: (relative_path, full_source_path)
    """
    matches = []
    source_path = Path(BASE_PATH) / source_dir

    if not source_path.exists():
        return matches

    dirs_scanned = 0
    # Walk through all subdirectories
    for root, dirs, _ in os.walk(source_path):
        # Track which directories to skip descending into
        matched_dirs = []

        for dir_name in dirs:
            dirs_scanned += 1
            # Update progress periodically
            if progress_callback and dirs_scanned % 50 == 0:
                progress_callback(dirs_scanned, len(matches))

            # Check if directory name contains any of the config patterns
            for config in configs:
                if config in dir_name:
                    full_path = Path(root) / dir_name
                    # Get path relative to source
                    rel_path = full_path.relative_to(source_path)
                    matches.append((str(rel_path), str(full_path)))
                    matched_dirs.append(dir_name)
                    break  # Don't add same directory multiple times

        # Don't descend into matched directories (no need to scan Plans/, Images/, etc.)
        for matched in matched_dirs:
            dirs.remove(matched)

    # Final progress update
    if progress_callback:
        progress_callback(dirs_scanned, len(matches), done=True)

    return sorted(matches)


def move_directories(
    source_dir: str,
    dest_dir: str,
    directories: List[Tuple[str, str]],
    progress_callback: callable = None,
    error_callback: callable = None,
    mode: str = "local",  # "local", "azure_sdk", "azcopy"
) -> Tuple[int, List[str]]:
    """
    Move directories from source to destination.

    Modes:
    - "local": Use shutil.move via filesystem (works with blobfuse mount)
    - "azure_sdk": Use Azure SDK server-side copy
    - "azcopy": Use azcopy sync + remove (fastest for large operations)

    Returns tuple of (success_count, error_messages)
    """
    if mode == "azcopy":
        # Double-check azcopy is available at runtime
        if not is_azcopy_available():
            if error_callback:
                error_callback("azcopy not installed - falling back to local mode")
            return _move_directories_local(
                source_dir, dest_dir, directories, progress_callback, error_callback
            )
        return _move_directories_azcopy(
            source_dir, dest_dir, directories, progress_callback, error_callback
        )
    elif mode == "azure_sdk":
        blob_service = get_blob_service_client()
        if blob_service:
            return _move_directories_azure(
                blob_service,
                source_dir,
                dest_dir,
                directories,
                progress_callback,
                error_callback,
            )
        # Fall back to local if Azure SDK not available

    return _move_directories_local(
        source_dir, dest_dir, directories, progress_callback, error_callback
    )


def _move_directories_azure(
    blob_service: "BlobServiceClient",
    source_dir: str,
    dest_dir: str,
    directories: List[Tuple[str, str]],
    progress_callback: callable = None,
    error_callback: callable = None,
) -> Tuple[int, List[str]]:
    """Move directories using Azure server-side copy."""
    success_count = 0
    errors = []
    total = len(directories)

    for i, (rel_path, full_source) in enumerate(directories):
        if progress_callback:
            should_continue = progress_callback(i + 1, total, f"[Azure] {rel_path}")
            if should_continue is False:
                break  # Stop requested

        # Convert paths to blob prefixes
        source_prefix = f"{source_dir}/{rel_path}"
        dest_prefix = f"{dest_dir}/{rel_path}"

        try:
            # Copy all blobs in directory
            copy_count, copy_errors = copy_blob_directory(
                blob_service, source_prefix, dest_prefix
            )

            if copy_errors:
                errors.extend(copy_errors)
                # Report errors as they happen
                if error_callback:
                    for err in copy_errors:
                        error_callback(err)
            elif copy_count > 0:
                # Delete source blobs after successful copy
                del_count, del_errors = delete_blob_directory(
                    blob_service, source_prefix
                )
                if del_errors:
                    errors.extend(del_errors)
                    if error_callback:
                        for err in del_errors:
                            error_callback(err)
                success_count += 1
            else:
                err_msg = f"No files copied for {rel_path}"
                errors.append(err_msg)
                if error_callback:
                    error_callback(err_msg)
        except Exception as e:
            err_msg = f"Failed to move {rel_path}: {e}"
            errors.append(err_msg)
            if error_callback:
                error_callback(err_msg)

    return success_count, errors


def _move_directories_local(
    source_dir: str,
    dest_dir: str,
    directories: List[Tuple[str, str]],
    progress_callback: callable = None,
    error_callback: callable = None,
) -> Tuple[int, List[str]]:
    """Move directories using local filesystem operations (shutil.move)."""
    dest_base = Path(BASE_PATH) / dest_dir
    success_count = 0
    errors = []
    total = len(directories)

    for i, (rel_path, full_source) in enumerate(directories):
        dest_path = dest_base / rel_path

        # Update progress
        if progress_callback:
            should_continue = progress_callback(i + 1, total, rel_path)
            if should_continue is False:
                break  # Stop requested

        try:
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Move the directory
            shutil.move(full_source, dest_path)
            success_count += 1
        except Exception as e:
            err_msg = f"Failed to move {rel_path}: {e}"
            errors.append(err_msg)
            if error_callback:
                error_callback(err_msg)

    return success_count, errors


def _move_directories_azcopy(
    source_dir: str,
    dest_dir: str,
    directories: List[Tuple[str, str]],
    progress_callback: callable = None,
    error_callback: callable = None,
) -> Tuple[int, List[str]]:
    """Move directories using azcopy sync + remove (much faster for Azure blob storage).

    Requires user to run 'azcopy login' first for Azure AD authentication.
    """
    account_name, _ = get_storage_account_info()
    if not account_name:
        # Try to get account name from connection string even without key
        account_name = "magmardata"  # Fallback to known account

    success_count = 0
    errors = []
    total = len(directories)

    # Build base URL (no SAS token - uses azcopy login credentials)
    base_url = f"https://{account_name}.blob.core.windows.net/{AZURE_CONTAINER_NAME}"

    for i, (rel_path, full_source) in enumerate(directories):
        # Update progress
        if progress_callback:
            should_continue = progress_callback(i + 1, total, f"[azcopy] {rel_path}")
            if should_continue is False:
                break  # Stop requested

        source_url = f"{base_url}/{source_dir}/{rel_path}/*"
        dest_url = f"{base_url}/{dest_dir}/{rel_path}/"

        # Environment with auto-login using Azure CLI credentials
        env = os.environ.copy()
        env["AZCOPY_AUTO_LOGIN_TYPE"] = "AZCLI"

        try:
            # Step 1: Copy with azcopy
            copy_result = subprocess.run(
                ["azcopy", "copy", source_url, dest_url, "--recursive"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per directory
                env=env,
            )

            if copy_result.returncode != 0:
                err_msg = (
                    f"azcopy copy failed for {rel_path}: {copy_result.stderr[:200]}"
                )
                errors.append(err_msg)
                # Print full error to console for debugging
                print(f"\n=== AZCOPY COPY ERROR for {rel_path} ===")
                print(f"Return code: {copy_result.returncode}")
                print(f"STDERR:\n{copy_result.stderr}")
                print(f"STDOUT:\n{copy_result.stdout}")
                print("=" * 50)
                if error_callback:
                    error_callback(err_msg)
                break  # Stop on first error

            # Step 2: Remove source with azcopy
            remove_url = f"{base_url}/{source_dir}/{rel_path}/*"
            remove_result = subprocess.run(
                ["azcopy", "remove", remove_url, "--recursive"],
                capture_output=True,
                text=True,
                timeout=300,
                env=env,
            )

            if remove_result.returncode != 0:
                err_msg = f"azcopy remove failed for {rel_path} (files copied but not deleted): {remove_result.stderr[:200]}"
                errors.append(err_msg)
                # Print full error to console for debugging
                print(f"\n=== AZCOPY REMOVE ERROR for {rel_path} ===")
                print(f"Return code: {remove_result.returncode}")
                print(f"STDERR:\n{remove_result.stderr}")
                print(f"STDOUT:\n{remove_result.stdout}")
                print("=" * 50)
                if error_callback:
                    error_callback(err_msg)
                break  # Stop on first error
            else:
                success_count += 1

        except subprocess.TimeoutExpired:
            err_msg = f"Timeout moving {rel_path}"
            errors.append(err_msg)
            if error_callback:
                error_callback(err_msg)
            break  # Stop on timeout
        except Exception as e:
            err_msg = f"Failed to move {rel_path}: {e}"
            errors.append(err_msg)
            if error_callback:
                error_callback(err_msg)
            break  # Stop on exception

    return success_count, errors


def load_cache() -> Tuple[List[Tuple[str, str]], Optional[str], str, str]:
    """Load cached matches from file.

    Returns tuple of (matches, cache_date, source_dir, configs_text)
    """
    if not CACHE_FILE.exists():
        return [], None, "", ""
    try:
        with open(CACHE_FILE, "r") as f:
            data = json.load(f)
        matches = [(m[0], m[1]) for m in data.get("matches", [])]
        cache_date = data.get("date")
        source_dir = data.get("source_dir", "")
        configs_text = data.get("configs", "")
        return matches, cache_date, source_dir, configs_text
    except Exception:
        return [], None, "", ""


def save_cache(matches: List[Tuple[str, str]], source_dir: str, configs_text: str):
    """Save matches to cache file."""
    data = {
        "matches": matches,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_dir": source_dir,
        "configs": configs_text,
    }
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Failed to save cache: {e}")


def clear_cache():
    """Delete the cache file."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()


class MoveConfigsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Move Configuration Directories")
        self.root.geometry("900x700")

        self.matches: List[Tuple[str, str]] = []
        self.azure_available = get_blob_service_client() is not None
        self.cache_date: Optional[str] = None

        self._create_widgets()
        self._load_cached_results()

    def _create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        row = 0

        # Source directory
        ttk.Label(main_frame, text="Source Directory:").grid(
            row=row, column=0, sticky="w", pady=5
        )
        self.source_var = tk.StringVar(value="20260115_Test")
        source_entry = ttk.Entry(main_frame, textvariable=self.source_var, width=50)
        source_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=5)

        row += 1

        # Destination directory
        ttk.Label(main_frame, text="Destination Directory:").grid(
            row=row, column=0, sticky="w", pady=5
        )
        self.dest_var = tk.StringVar(value="20260115_Test_Unused")
        dest_entry = ttk.Entry(main_frame, textvariable=self.dest_var, width=50)
        dest_entry.grid(row=row, column=1, sticky="ew", pady=5, padx=5)

        row += 1

        # Base path and transfer mode selection
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=5, pady=5)

        ttk.Label(status_frame, text=f"Base path: {BASE_PATH}", foreground="gray").pack(
            side=tk.LEFT
        )

        ttk.Label(status_frame, text="   Transfer mode:", foreground="gray").pack(
            side=tk.LEFT, padx=(20, 5)
        )

        # Radio buttons for transfer mode
        self.transfer_mode_var = tk.StringVar(
            value=(
                "azcopy"
                if AZCOPY_AVAILABLE
                else ("azure_sdk" if self.azure_available else "local")
            )
        )

        # Local filesystem option
        ttk.Radiobutton(
            status_frame,
            text="Local (blobfuse)",
            variable=self.transfer_mode_var,
            value="local",
        ).pack(side=tk.LEFT, padx=2)

        # Azure SDK option
        azure_sdk_state = "normal" if self.azure_available else "disabled"
        ttk.Radiobutton(
            status_frame,
            text="Azure SDK",
            variable=self.transfer_mode_var,
            value="azure_sdk",
            state=azure_sdk_state,
        ).pack(side=tk.LEFT, padx=2)

        # AzCopy option (recommended)
        azcopy_state = "normal" if AZCOPY_AVAILABLE else "disabled"
        azcopy_text = (
            "AzCopy (fastest)" if AZCOPY_AVAILABLE else "AzCopy (not installed)"
        )
        ttk.Radiobutton(
            status_frame,
            text=azcopy_text,
            variable=self.transfer_mode_var,
            value="azcopy",
            state=azcopy_state,
        ).pack(side=tk.LEFT, padx=2)

        # Status indicators
        status_indicators = []
        if AZCOPY_AVAILABLE:
            status_indicators.append("✓ azcopy")
        if self.azure_available:
            status_indicators.append("✓ SDK")
        if status_indicators:
            ttk.Label(
                status_frame,
                text=f"  ({', '.join(status_indicators)})",
                foreground="green",
            ).pack(side=tk.LEFT, padx=5)

        row += 1

        # AzCopy login hint
        if AZCOPY_AVAILABLE:
            hint_frame = ttk.Frame(main_frame)
            hint_frame.grid(row=row, column=0, columnspan=2, sticky="w", padx=5)
            ttk.Label(
                hint_frame,
                text="Note: AzCopy requires Azure login. Run 'azcopy login' in terminal first.",
                foreground="gray",
            ).pack(side=tk.LEFT)
            row += 1

        # Configurations input
        ttk.Label(main_frame, text="Configurations (one per line):").grid(
            row=row, column=0, sticky="nw", pady=5
        )
        self.configs_text = scrolledtext.ScrolledText(
            main_frame, width=60, height=6, wrap=tk.NONE
        )
        self.configs_text.grid(row=row, column=1, sticky="ew", pady=5, padx=5)
        # Insert example configs
        self.configs_text.insert(
            tk.END,
            "T0_Fd_H60_C0_P2_I0_R1_S1_E0_M4096\n"
            "T0_Fn_H60_C0_P2_I0_R1_S1_E0_M4096\n"
            "T0_Fs_H00_C0_P2_I0_R1_S1_E0_M4096\n"
            "T0_Fs_H60_C0_P2_I0_R0_S1_E0_M4096\n"
            "T0_Fs_H60_C0_P2_I0_R1_S0_E0_M4096",
        )

        row += 1

        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=10)

        ttk.Button(
            btn_frame, text="Find Matching Directories", command=self._find
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame, text="Move Selected", command=self._move, style="Accent.TButton"
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear Results", command=self._clear_results).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Clear Cache", command=self._clear_cache).pack(
            side=tk.LEFT, padx=5
        )

        row += 1

        # Results label with cache date
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 5))

        self.results_label = ttk.Label(results_frame, text="Matching directories:")
        self.results_label.pack(side=tk.LEFT)

        self.cache_label = ttk.Label(results_frame, text="", foreground="gray")
        self.cache_label.pack(side=tk.LEFT, padx=(20, 0))

        row += 1

        # Results list with checkboxes
        list_frame = ttk.Frame(main_frame)
        list_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=5)
        main_frame.rowconfigure(row, weight=1)

        # Treeview for directory list
        columns = ("select", "directory")
        self.tree = ttk.Treeview(
            list_frame, columns=columns, show="headings", selectmode="extended"
        )
        self.tree.heading("select", text="✓")
        self.tree.heading("directory", text="Directory Path")
        self.tree.column("select", width=30, anchor="center")
        self.tree.column("directory", width=800)

        # Scrollbars
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(list_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        # Bind click to toggle selection
        self.tree.bind("<Button-1>", self._on_click)

        row += 1

        # Select all / deselect all buttons
        select_frame = ttk.Frame(main_frame)
        select_frame.grid(row=row, column=0, columnspan=2, pady=5)

        ttk.Button(select_frame, text="Select All", command=self._select_all).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(select_frame, text="Deselect All", command=self._deselect_all).pack(
            side=tk.LEFT, padx=5
        )
        self.stop_button = ttk.Button(
            select_frame, text="Stop", command=self._stop_move, state="disabled"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        row += 1

        # Errors display area
        ttk.Label(main_frame, text="Errors (shown in real-time):").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(10, 2)
        )

        row += 1

        self.errors_text = scrolledtext.ScrolledText(
            main_frame,
            width=60,
            height=6,
            wrap=tk.WORD,
            state="disabled",
            background="#fff0f0",
        )
        self.errors_text.grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=2, padx=5
        )

        # Track selected items
        self.selected = set()
        self.stop_requested = False

    def _get_configs(self) -> List[str]:
        """Get list of configurations from text input."""
        text = self.configs_text.get("1.0", tk.END)
        configs = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return configs

    def _update_progress(
        self, dirs_scanned: int, matches_found: int, done: bool = False
    ):
        """Update progress label during search."""
        if done:
            status = f"Scan complete: {dirs_scanned} directories scanned"
        else:
            status = f"Scanning... {dirs_scanned} directories scanned, {matches_found} matches found"
        self.results_label.config(text=status)
        self.root.update_idletasks()

    def _find(self):
        """Find matching directories."""
        configs = self._get_configs()
        if not configs:
            messagebox.showwarning(
                "Warning", "Please enter at least one configuration."
            )
            return

        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning("Warning", "Please enter a source directory.")
            return

        source_path = Path(BASE_PATH) / source
        if not source_path.exists():
            messagebox.showerror(
                "Error", f"Source directory does not exist:\n{source_path}"
            )
            return

        # Show scanning status
        self.results_label.config(text="Scanning...")
        self.root.update_idletasks()

        self.matches = find_matching_directories(source, configs, self._update_progress)
        self._update_tree()

        # Save to cache
        self._save_to_cache()

        self.results_label.config(
            text=f"Matching directories: {len(self.matches)} found"
        )

    def _update_tree(self):
        """Update the treeview with matches."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.selected.clear()

        # Add matches
        for i, (rel_path, full_path) in enumerate(self.matches):
            item_id = self.tree.insert("", "end", values=("☑", rel_path))
            self.selected.add(item_id)

    def _on_click(self, event):
        """Handle click on treeview to toggle selection."""
        region = self.tree.identify("region", event.x, event.y)
        if region == "cell":
            column = self.tree.identify_column(event.x)
            item = self.tree.identify_row(event.y)
            if column == "#1" and item:  # Click on checkbox column
                self._toggle_selection(item)

    def _toggle_selection(self, item):
        """Toggle selection for an item."""
        if item in self.selected:
            self.selected.remove(item)
            values = self.tree.item(item, "values")
            self.tree.item(item, values=("☐", values[1]))
        else:
            self.selected.add(item)
            values = self.tree.item(item, "values")
            self.tree.item(item, values=("☑", values[1]))

    def _select_all(self):
        """Select all items."""
        for item in self.tree.get_children():
            if item not in self.selected:
                self.selected.add(item)
                values = self.tree.item(item, "values")
                self.tree.item(item, values=("☑", values[1]))

    def _deselect_all(self):
        """Deselect all items."""
        for item in self.tree.get_children():
            if item in self.selected:
                self.selected.remove(item)
                values = self.tree.item(item, "values")
                self.tree.item(item, values=("☐", values[1]))

    def _clear_results(self):
        """Clear the results list."""
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.matches = []
        self.selected.clear()
        self.results_label.config(text="Matching directories:")
        self.cache_label.config(text="")
        # Also clear errors
        self.errors_text.config(state="normal")
        self.errors_text.delete("1.0", tk.END)
        self.errors_text.config(state="disabled")

    def _clear_cache(self):
        """Clear the cache file and results."""
        clear_cache()
        self._clear_results()
        self.cache_date = None
        messagebox.showinfo("Cache Cleared", "Cache has been cleared.")

    def _load_cached_results(self):
        """Load cached results on startup."""
        matches, cache_date, source_dir, configs_text = load_cache()
        if matches and cache_date:
            self.matches = matches
            self.cache_date = cache_date
            # Restore source dir and configs if they were saved
            if source_dir:
                self.source_var.set(source_dir)
            if configs_text:
                self.configs_text.delete("1.0", tk.END)
                self.configs_text.insert(tk.END, configs_text)
            self._update_tree()
            self.results_label.config(
                text=f"Matching directories: {len(self.matches)} found"
            )
            self.cache_label.config(text=f"(cached: {cache_date})")

    def _save_to_cache(self):
        """Save current matches to cache."""
        source = self.source_var.get().strip()
        configs_text = self.configs_text.get("1.0", tk.END).strip()
        save_cache(self.matches, source, configs_text)
        self.cache_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cache_label.config(text=f"(cached: {self.cache_date})")

    def _remove_from_cache(self, rel_paths: List[str]):
        """Remove moved items from cache."""
        if not self.matches:
            return
        # Filter out the moved items
        self.matches = [
            (rel, full) for rel, full in self.matches if rel not in rel_paths
        ]
        # Save updated cache
        if self.matches:
            self._save_to_cache()
        else:
            clear_cache()
            self.cache_label.config(text="")

    def _stop_move(self):
        """Request stop of current move operation."""
        self.stop_requested = True
        self.results_label.config(
            text="Stopping... (will stop after current directory)"
        )
        self.root.update_idletasks()

    def _add_error(self, error_msg: str):
        """Add an error to the errors display in real-time."""
        self.errors_text.config(state="normal")
        self.errors_text.insert(tk.END, f"❌ {error_msg}\n")
        self.errors_text.see(tk.END)  # Scroll to bottom
        self.errors_text.config(state="disabled")
        self.root.update_idletasks()

    def _move(self):
        """Move selected directories."""
        if not self.selected:
            messagebox.showwarning("Warning", "No directories selected.")
            return

        dest = self.dest_var.get().strip()
        if not dest:
            messagebox.showwarning("Warning", "Please enter a destination directory.")
            return

        # Get selected directories
        selected_dirs = []
        for item in self.selected:
            values = self.tree.item(item, "values")
            rel_path = values[1]
            # Find the full path from matches
            for match_rel, match_full in self.matches:
                if match_rel == rel_path:
                    selected_dirs.append((match_rel, match_full))
                    break

        # Confirm
        source = self.source_var.get().strip()
        transfer_mode = self.transfer_mode_var.get()
        mode_names = {
            "local": "local filesystem",
            "azure_sdk": "Azure SDK",
            "azcopy": "AzCopy",
        }
        mode_str = f" ({mode_names.get(transfer_mode, transfer_mode)})"
        msg = (
            f"Move {len(selected_dirs)} directories?{mode_str}\n\n"
            f"From: {BASE_PATH}/{source}\n"
            f"To: {BASE_PATH}/{dest}"
        )
        if not messagebox.askyesno("Confirm Move", msg):
            return

        # Clear errors display and reset stop flag
        self.errors_text.config(state="normal")
        self.errors_text.delete("1.0", tk.END)
        self.errors_text.config(state="disabled")
        self.stop_requested = False
        self.stop_button.config(state="normal")
        self.error_count = 0

        # Progress callback
        def move_progress(current: int, total: int, current_path: str):
            if self.stop_requested:
                return False  # Signal to stop
            status = f"Moving {current}/{total}: {current_path[:60]}..."
            if self.error_count > 0:
                status += f" ({self.error_count} errors)"
            self.results_label.config(text=status)
            self.root.update_idletasks()
            return True  # Continue

        # Error callback for real-time display
        def error_callback(error_msg: str):
            self.error_count += 1
            self._add_error(error_msg)

        mode_label = f"Moving ({mode_names.get(transfer_mode, transfer_mode)})..."
        self.results_label.config(text=mode_label)
        self.root.update_idletasks()

        success, errors = move_directories(
            source,
            dest,
            selected_dirs,
            move_progress,
            mode=transfer_mode,
            error_callback=error_callback,
        )

        # Disable stop button
        self.stop_button.config(state="disabled")

        # Show results
        total_errors = len(errors)
        if self.stop_requested:
            messagebox.showinfo(
                "Stopped",
                f"Operation stopped by user.\n\n"
                f"Moved {success} directories before stopping.\n"
                f"Errors: {total_errors}",
            )
        elif total_errors > 0:
            messagebox.showwarning(
                "Completed with Errors",
                f"Moved {success} directories.\n"
                f"Errors: {total_errors}\n\n"
                f"See errors panel for details.",
            )
        else:
            messagebox.showinfo("Success", f"Successfully moved {success} directories.")

        # Remove successfully moved items from cache
        moved_paths = [rel_path for rel_path, _ in selected_dirs[:success]]
        self._remove_from_cache(moved_paths)

        # Refresh the list from cache (don't re-scan)
        self._update_tree()
        self.results_label.config(
            text=f"Matching directories: {len(self.matches)} found"
        )


def main():
    """Main entry point."""
    # Check if base path exists
    if not os.path.exists(BASE_PATH):
        print(f"Warning: Base path {BASE_PATH} does not exist.")
        print("The tool will still run but won't find any directories.")

    root = tk.Tk()
    app = MoveConfigsApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
