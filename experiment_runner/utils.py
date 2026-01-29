"""Utility functions for process management, Xvfb, GPU detection, etc."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import zip_longest
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment_runner.models import ExperimentConfig

# --- Xvfb Management ---


@dataclass
class XvfbManager:
    """Manages a virtual X display via Xvfb."""

    display_num: int = 99
    process: subprocess.Popen | None = None

    def setup(self) -> int:
        """Start Xvfb and return the display number."""
        print("[XVFB] Setting up virtual display...")

        self.display_num = self._find_available_display()
        print(f"[XVFB] Using display :{self.display_num}")

        xvfb_args = f":{self.display_num} -screen 0 1024x1024x24 +extension GLX +render -noreset -ac"
        print(f"[XVFB] Starting: Xvfb {xvfb_args}")

        self.process = subprocess.Popen(
            ["Xvfb"] + xvfb_args.split(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(2)  # Wait for Xvfb to start

        if self.process.poll() is not None:
            raise RuntimeError("[ERROR] Xvfb failed to start")

        print(f"[XVFB] Xvfb started successfully (PID: {self.process.pid})")
        os.environ["DISPLAY"] = f":{self.display_num}"

        return self.display_num

    def cleanup(self) -> None:
        """Stop Xvfb process."""
        if self.process and self.process.poll() is None:
            print(f"[CLEANUP] Stopping Xvfb (PID: {self.process.pid})")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception as e:
                print(f"[CLEANUP] Error stopping Xvfb: {e}")
                self.process.kill()

    def _find_available_display(self) -> int:
        """Find an available display number."""
        for n in range(99, 199):
            if not Path(f"/tmp/.X{n}-lock").exists():
                return n
        raise RuntimeError("Could not find available display number")


# --- GPU Detection ---


def detect_gpus() -> str:
    """Detect available GPUs via nvidia-smi and return CUDA_VISIBLE_DEVICES string."""
    print("[GPU] Detecting available GPUs...")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            gpu_ids = [
                s.strip()
                for s in result.stdout.strip().split("\n")
                if s.strip().isdigit()
            ]
            if gpu_ids:
                cuda_devices = ",".join(gpu_ids)
                print(
                    f"[GPU] Found {len(gpu_ids)} GPU(s): CUDA_VISIBLE_DEVICES={cuda_devices}"
                )
                return cuda_devices
            else:
                print("[GPU] No GPUs detected from nvidia-smi output")
        else:
            print("[GPU] nvidia-smi returned no GPU information")
    except FileNotFoundError:
        print("[GPU] nvidia-smi not found - continuing without GPU configuration")
    except subprocess.TimeoutExpired:
        print("[GPU] nvidia-smi timed out")
    except Exception as e:
        print(f"[GPU] Could not detect GPUs: {e}")

    return ""


# --- Test Data Download ---


async def download_test_data() -> None:
    """Download test data from Azure Blob Storage if not present."""
    if Path("Generated").exists():
        print("[DATA] Test data already exists, skipping download")
        return

    print("[DATA] Test data not found, downloading from Azure...")

    if not shutil.which("azcopy"):
        raise RuntimeError("[ERROR] azcopy not found. Please install it first.")

    proc = await asyncio.create_subprocess_exec(
        "azcopy",
        "copy",
        "https://magmardata.blob.core.windows.net/magmathor/Test",
        "./Generated",
        "--overwrite=prompt",
        "--check-md5",
        "FailIfDifferent",
        "--from-to=BlobLocal",
        "--recursive",
        "--log-level=INFO",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await proc.communicate()

    if proc.returncode != 0:
        msg = f"[ERROR] Failed to download test data. Exit code: {proc.returncode}"
        if stderr:
            msg += f"\n[ERROR] azcopy stderr:\n{stderr.decode().strip()}"
        raise RuntimeError(msg)

    print("[DATA] Test data downloaded successfully")


# --- Process Helpers ---


def start_process_with_log_redirect(
    command: list[str],
    log_file_path: str | Path,
    display_num: int,
    cuda_devices: str = "",
) -> subprocess.Popen:
    """Start a process with stdout/stderr redirected to a log file."""
    log_file_path = Path(log_file_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    env["DETAILED_LOGGING"] = "1"
    env["DISPLAY"] = f":{display_num}"

    if cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices

    # Open log file for writing
    log_file = open(log_file_path, "w")

    # Write the command being run at the top of the log
    import shlex

    timestamp = datetime.now().strftime("%H:%M:%S")
    command_str = shlex.join(command)
    log_file.write(f"[{timestamp}] Running command:\n")
    log_file.write(f"[{timestamp}] {command_str}\n")
    log_file.write(f"[{timestamp}] {'=' * 60}\n")
    log_file.flush()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )

    # Start a thread to write output to log file
    import threading

    def log_output():
        try:
            for line in process.stdout:
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_file.write(f"[{timestamp}] {line}")
                log_file.flush()
        except Exception:
            pass
        finally:
            # Write exit code when process completes
            process.wait()
            timestamp = datetime.now().strftime("%H:%M:%S")
            log_file.write(
                f"[{timestamp}] Process exited with code: {process.returncode}\n"
            )
            log_file.close()

    thread = threading.Thread(target=log_output, daemon=True)
    thread.start()

    return process


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def kill_process_tree(pid: int) -> None:
    """Kill a process and all its children."""
    import signal

    try:
        # Try to kill the process group
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            # Fallback to just killing the process
            os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


# --- Log Parsing ---


def parse_eta_from_log(log_path: str | Path, is_running: bool) -> tuple[str, str, str]:
    """Parse progress and ETA from log file.

    Returns: (progress, elapsed, eta)
    """
    log_path = Path(log_path)

    if not log_path.exists():
        return ("--", "--", "[yellow]Starting...[/]" if is_running else "--")

    try:
        lines = log_path.read_text().splitlines()

        # Search from end for progress line
        for line in reversed(lines):
            match = re.search(
                r"\[(\d+)/(\d+)\].*Elapsed:\s*([\d.]+)m.*Est\. Remaining:\s*([\d.]+)m",
                line,
            )
            if match:
                current = int(match.group(1))
                total = int(match.group(2))
                elapsed_min = float(match.group(3))
                remaining_min = float(match.group(4))

                progress = f"{current}/{total}"
                elapsed = _format_minutes(elapsed_min)
                eta = _format_minutes(remaining_min)

                return (progress, elapsed, eta)

        if lines:
            return ("0/?", "--", "[yellow]Initializing...[/]" if is_running else "--")

    except Exception:
        pass

    return ("--", "--", "[yellow]Starting...[/]" if is_running else "--")


def _format_minutes(minutes: float) -> str:
    """Format minutes as human-readable string."""
    if minutes < 1:
        return f"{minutes * 60:.0f}s"
    elif minutes < 60:
        return f"{minutes:.1f}m"
    else:
        hours = int(minutes / 60)
        mins = minutes % 60
        return f"{hours}h {mins:.0f}m"


def is_log_file_successfully_completed(log_file_path: str | Path) -> bool:
    """Check if log file indicates successful completion."""
    log_path = Path(log_file_path)

    if not log_path.exists():
        return False

    try:
        # Read last 1KB of file
        with open(log_path, "rb") as f:
            f.seek(max(0, log_path.stat().st_size - 1024))
            tail = f.read().decode("utf-8", errors="ignore")
        return "Process exited with code: 0" in tail
    except Exception:
        return False


# --- Port Checking ---


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    try:
        result = subprocess.run(
            ["lsof", f"-iTCP:{port}", "-sTCP:LISTEN", "-Pn"],
            capture_output=True,
            timeout=5,
        )
        # lsof returns lines for each process using the port
        return len(result.stdout.decode().strip().splitlines()) > 1
    except Exception:
        return False


async def wait_for_url_reachable(
    url: str, max_attempts: int = 1000, delay_seconds: int = 10
) -> bool:
    """Wait for a URL to become reachable."""
    import httpx

    async with httpx.AsyncClient(timeout=5.0) as client:
        for i in range(max_attempts):
            try:
                response = await client.get(f"{url}/v1/models")
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            if i < max_attempts - 1:
                print(
                    f"  Attempt {i + 1}/{max_attempts} failed, retrying in {delay_seconds}s..."
                )
                await asyncio.sleep(delay_seconds)

    return False


# --- Statistics ---


def compute_mean_stddev(values: list[float]) -> tuple[float, float]:
    """Compute mean and standard deviation of values."""
    if not values:
        return (0.0, 0.0)

    mean = sum(values) / len(values)

    if len(values) == 1:
        return (mean, 0.0)

    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    stddev = variance**0.5

    return (mean, stddev)


def format_mean_stddev(stats: tuple[float, float]) -> str:
    """Format mean ± stddev as percentage string."""
    mean, stddev = stats
    return f"{mean:.1%} ± {stddev:.1%}"


# --- Test Results Completion Check ---


def is_test_results_complete(
    test_results_path: str | Path,
) -> tuple[bool, int, int]:
    """Check if test_results.json indicates completion.

    Reads the test_results.json file and compares the number of completed
    test results to the expected_num_plans value.

    Args:
        test_results_path: Path to the test_results.json file.

    Returns:
        Tuple of (is_complete, completed_count, expected_count).
        Returns (False, 0, 0) if file doesn't exist or can't be parsed.
    """
    path = Path(test_results_path)
    if not path.exists():
        return (False, 0, 0)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected = data.get("expected_num_plans", 0)
        results = data.get("test_results", [])
        completed = len(results)

        is_complete = expected > 0 and completed >= expected
        return (is_complete, completed, expected)
    except (json.JSONDecodeError, IOError, OSError):
        return (False, 0, 0)


# --- Experiment Interleaving ---


def interleave_by_model(
    experiments: list["ExperimentConfig"],
) -> list["ExperimentConfig"]:
    """Interleave experiments using round-robin grouped by model.

    Groups experiments by their model setting, then interleaves them
    so that consecutive experiments use different models. This helps
    distribute load across model endpoints and avoid rate limiting.

    Args:
        experiments: List of experiment configurations to interleave.

    Returns:
        New list with experiments interleaved by model in round-robin order.
        If an experiment has no settings or model, it's grouped under "".
    """
    # Group experiments by model
    groups: dict[str, list[ExperimentConfig]] = defaultdict(list)
    for exp in experiments:
        model = exp.settings.model if exp.settings else ""
        groups[model].append(exp)

    # Sort groups by model name for deterministic ordering
    sorted_groups = [groups[model] for model in sorted(groups.keys())]

    # Round-robin interleave using zip_longest
    result: list[ExperimentConfig] = []
    for items in zip_longest(*sorted_groups, fillvalue=None):
        for item in items:
            if item is not None:
                result.append(item)

    return result
