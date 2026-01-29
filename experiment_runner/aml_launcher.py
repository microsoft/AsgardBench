"""AML job launcher for running experiments via Amulet.

This module provides functionality to:
- Submit experiments to AML via Amulet CLI
- Track submitted jobs and their status
- Generate Amulet YAML configurations dynamically
- Check for output path conflicts with local runs
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from experiment_runner.models import ExperimentConfig, ExperimentSettings

# Default AML configuration - can be overridden in aml_config.yaml
DEFAULT_AML_CONFIG = {
    "target": {
        "service": "sing",
        "name": "palisades-cpu",  # Default CPU cluster
        "workspace_name": "lalidenamlws",
    },
    "sku": "1x 10C3",  # CPU-only SKU
    "priority": "high",
    "sla_tier": "Premium",
    "keyvault_name": "",  # Must be configured
    "managed_identity": "",  # Must be configured
    "experiment_prefix": "magmathor",  # Amulet experiment name prefix
}

# Path for tracking AML jobs
AML_JOBS_FILE = Path("experiment_runner/aml_jobs.json")


@dataclass
class AmlJobInfo:
    """Information about a submitted AML job."""

    job_id: str  # Amulet job ID (experiment_name:job_name)
    experiment_key: str  # Local experiment key (experiment_name--test_suite)
    test_suite: str
    model_name: str
    submit_time: str  # ISO format
    status: str = "Submitted"  # Submitted, Running, Completed, Failed, Canceled
    amulet_experiment: str = ""  # Amulet experiment name

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "experiment_key": self.experiment_key,
            "test_suite": self.test_suite,
            "model_name": self.model_name,
            "submit_time": self.submit_time,
            "status": self.status,
            "amulet_experiment": self.amulet_experiment,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AmlJobInfo:
        return cls(
            job_id=data.get("job_id", ""),
            experiment_key=data.get("experiment_key", ""),
            test_suite=data.get("test_suite", ""),
            model_name=data.get("model_name", ""),
            submit_time=data.get("submit_time", ""),
            status=data.get("status", "Submitted"),
            amulet_experiment=data.get("amulet_experiment", ""),
        )


@dataclass
class AmlLauncher:
    """Manages AML job submission and tracking via Amulet."""

    config_path: str = "experiment_runner/aml_config.yaml"
    jobs_file: Path = field(default_factory=lambda: AML_JOBS_FILE)
    config: dict[str, Any] = field(default_factory=dict)
    jobs: dict[str, AmlJobInfo] = field(default_factory=dict)  # key -> job_info

    def __post_init__(self) -> None:
        """Load configuration and job history."""
        self._load_config()
        self._load_jobs()

    def _load_config(self) -> None:
        """Load AML configuration from file or use defaults."""
        config_path = Path(self.config_path)
        if config_path.exists():
            with open(config_path) as f:
                self.config = {**DEFAULT_AML_CONFIG, **yaml.safe_load(f)}
        else:
            self.config = dict(DEFAULT_AML_CONFIG)

    def _load_jobs(self) -> None:
        """Load submitted jobs from tracking file."""
        if self.jobs_file.exists():
            with open(self.jobs_file) as f:
                data = json.load(f)
                self.jobs = {
                    k: AmlJobInfo.from_dict(v) for k, v in data.get("jobs", {}).items()
                }

    def _save_jobs(self) -> None:
        """Save submitted jobs to tracking file."""
        self.jobs_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.jobs_file, "w") as f:
            json.dump(
                {"jobs": {k: v.to_dict() for k, v in self.jobs.items()}},
                f,
                indent=2,
            )

    def is_configured(self) -> bool:
        """Check if AML is properly configured."""
        required = ["keyvault_name", "managed_identity"]
        return all(self.config.get(k) for k in required)

    def get_missing_config(self) -> list[str]:
        """Get list of missing required configuration."""
        required = ["keyvault_name", "managed_identity"]
        return [k for k in required if not self.config.get(k)]

    def check_mounted_storage_conflict(self) -> tuple[bool, str]:
        """Check if MOUNTED_STORAGE_PATH exists and may conflict with AML jobs.

        NOTE: This check is currently disabled because recursive glob on
        mounted blob storage is too slow. The user should ensure no local
        experiments are writing to the same paths as AML jobs.

        Returns:
            Tuple of (has_conflict, message)
        """
        # Disabled - recursive glob on blob storage hangs
        # TODO: Implement a faster check by only looking at specific experiment folders
        return False, ""

    def is_job_submitted(self, experiment_key: str) -> bool:
        """Check if a job has already been submitted for this experiment."""
        return experiment_key in self.jobs

    def get_job_status(self, experiment_key: str) -> str | None:
        """Get the status of a submitted job."""
        job = self.jobs.get(experiment_key)
        return job.status if job else None

    def _generate_yaml(
        self,
        experiments: list[tuple[ExperimentConfig, str]],  # (config, test_suite)
    ) -> str:
        """Generate Amulet YAML configuration for the given experiments.

        Args:
            experiments: List of (ExperimentConfig, test_suite) tuples

        Returns:
            YAML content as string
        """
        target = self.config.get("target", DEFAULT_AML_CONFIG["target"])
        keyvault = self.config.get("keyvault_name", "")
        managed_identity = self.config.get("managed_identity", "")

        jobs = []
        for exp_config, test_suite in experiments:
            settings = exp_config.settings
            if not settings:
                continue

            # Extract model name, config string, and rep number from experiment name
            # Format: {model}--{config_string}--rep{n}
            # e.g., "gpt-4o--T0_Fs_H60_C0_P2_I0_R1_S1_E0_M4096--rep1"
            parts = exp_config.name.split("--")
            model_name = parts[0] if parts else "unknown"
            config_string = parts[1] if len(parts) > 1 else ""
            rep_match = re.search(r"rep(\d+)$", exp_config.name)
            rep_number = int(rep_match.group(1)) if rep_match else 1

            # Build job name with config string for clarity
            # Format: {model}-{config}-{test_suite}-rep{n}
            if config_string:
                job_name = f"{model_name}-{config_string}-{test_suite}-rep{rep_number}"
            else:
                job_name = f"{model_name}-{test_suite}-rep{rep_number}"
            # Sanitize job name for Amulet (no special chars except _ and -)
            job_name = re.sub(r"[^a-zA-Z0-9_-]", "-", job_name)

            # Build command line arguments
            args = settings.build_command_args(test_suite, exp_config.name)
            # Convert from uv format to direct uv run call
            # args starts with ["run", "python", "Magmathor/Model/model_tester.py", ...]
            # We keep the full command but use uv run
            python_args = args[1:]  # Skip just "run", keep "python" and rest

            # Build the log file path matching local runner convention:
            # {TEST_DIR}/{test_suite}/{experiment_name}/logs.txt
            # TEST_DIR on AML with mounted storage = /mnt/magmathor/{TEST_FOLDER_NAME}
            from Magmathor.constants import TEST_FOLDER_NAME

            log_dir = (
                f"/mnt/magmathor/{TEST_FOLDER_NAME}/{test_suite}/{exp_config.name}"
            )
            log_file = f"{log_dir}/logs.txt"

            # Build command that:
            # 1. Sets up PATH for uv (setup env doesn't persist to job command)
            # 2. Creates the output directory
            # 3. Runs the experiment and pipes output to logs.txt (like local runner)
            # Note: uv is installed in setup but PATH needs to be set again
            uv_path_setup = 'export PATH="$$HOME/.local/bin:$$PATH"'
            python_cmd = " ".join(["uv", "run", *python_args])

            job = {
                "name": job_name,
                "sku": self.config.get("sku", DEFAULT_AML_CONFIG["sku"]),
                "command": [
                    "set -ex",
                    # Set up PATH for uv (installed in setup but env doesn't persist)
                    uv_path_setup,
                    # Start Xvfb virtual display for AI2-THOR
                    "Xvfb :90 -screen 0 1024x1024x24 -ac +extension GLX +render -noreset &",
                    "sleep 2",
                    # Export DISPLAY for AI2-THOR (critical!)
                    "export DISPLAY=:90",
                    # Create output directory (matching local runner behavior)
                    f"mkdir -p {log_dir}",
                    "export MOUNTED_STORAGE_PATH=/mnt/magmathor",
                    # Run the command and tee output to both stdout and the experiment's log file
                    f"{python_cmd} 2>&1 | tee -a {log_file}",
                ],
            }
            jobs.append(job)

        yaml_config = {
            "description": f"Magmathor experiments submitted at {datetime.now().isoformat()}",
            "target": target,
            "environment": {
                "image": "amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04",
                "setup": [
                    "set -ex",
                    # Install uv and add to PATH
                    "curl -LsSf https://astral.sh/uv/install.sh | sh",
                    'export PATH="$$HOME/.local/bin:$$PATH"',
                    # Sync dependencies with uv BEFORE create_display.sh (needs ai2thor)
                    "uv sync",
                    "source .venv/bin/activate",
                    # Xvfb setup for AI2-THOR
                    "sudo apt-get update && sudo apt-get install -y vulkan-tools xvfb mesa-utils",
                    "sudo mkdir -p /tmp/.X11-unix",
                    "sudo chmod 1777 /tmp/.X11-unix",
                ],
            },
            "storage": {
                "magmathor": {
                    "storage_account_name": "magmardata",
                    "container_name": "magmathor",
                },
            },
            "code": {
                "local_dir": ".",
                "ignore": [".venv", ".github", "__pycache__", "*.pyc", ".git", "Test"],
            },
            "jobs": jobs,
        }

        # Add submit_args with env vars for Key Vault and Azure OpenAI auth
        for job in yaml_config["jobs"]:
            job["submit_args"] = {
                "env": {
                    # Key Vault access
                    "AZURE_KEYVAULT_NAME": keyvault,
                    "OPENROUTER_SECRET_NAME": "openrouter-api-key",
                    # Managed identity for Azure OpenAI and Key Vault
                    "_AZUREML_SINGULARITY_JOB_UAI": managed_identity,
                    # Misc
                    "MKL_THREADING_LAYER": "GNU",
                }
            }
            job["identity"] = "managed"
            job["priority"] = self.config.get("priority", "high")
            job["sla_tier"] = self.config.get("sla_tier", "Premium")
            job["process_count_per_node"] = 1

        return yaml.dump(yaml_config, default_flow_style=False, sort_keys=False)

    async def submit_experiments(
        self,
        experiments: list[tuple[ExperimentConfig, str]],  # (config, test_suite)
        experiment_name: str | None = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        """Submit experiments to AML via Amulet.

        Args:
            experiments: List of (ExperimentConfig, test_suite) tuples to submit
            experiment_name: Amulet experiment name (auto-generated if None)
            force: Skip conflict checks and resubmit even if already submitted
            dry_run: Generate YAML but don't actually submit

        Returns:
            List of submitted job IDs
        """
        if not self.is_configured():
            missing = self.get_missing_config()
            raise ValueError(
                f"AML not configured. Missing: {', '.join(missing)}\n"
                f"Please configure in {self.config_path}"
            )

        # Filter out already-submitted experiments (unless force)
        # Allow resubmission of failed/cancelled/killed jobs
        RESUBMITTABLE_STATUSES = {"Failed", "Canceled", "Cancelled", "Error", "Killed"}
        if not force:
            to_submit = []
            for exp_config, test_suite in experiments:
                key = f"{exp_config.name}--{test_suite}"
                if not self.is_job_submitted(key):
                    to_submit.append((exp_config, test_suite))
                else:
                    job = self.jobs.get(key)
                    if job and job.status in RESUBMITTABLE_STATUSES:
                        print(
                            f"[AML] Resubmitting {key} (previous status: {job.status})"
                        )
                        to_submit.append((exp_config, test_suite))
                    else:
                        status = job.status if job else "unknown"
                        print(
                            f"[AML] Skipping {key} - already submitted (status: {status})"
                        )
            experiments = to_submit

        if not experiments:
            print("[AML] No new experiments to submit")
            return []

        # Check for mounted storage conflicts (if not forced)
        # Note: Callers (like REPL) should handle conflict prompts in an async-safe way
        # and pass force=True after user confirms
        if not force:
            has_conflict, conflict_msg = self.check_mounted_storage_conflict()
            if has_conflict:
                print(f"[AML] WARNING: {conflict_msg}")
                print("[AML] Use force=True or handle conflict in caller to proceed")
                return []

        # Generate YAML
        yaml_content = self._generate_yaml(experiments)

        if dry_run:
            print("[AML] Dry run - generated YAML:")
            print("-" * 60)
            print(yaml_content)
            print("-" * 60)
            return []

        # Write to temp file and submit
        prefix = self.config.get("experiment_prefix", "magmathor")
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{prefix}_{timestamp}"

        submitted_jobs = []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as tmp_file:
            tmp_file.write(yaml_content)
            tmp_path = tmp_file.name

        try:
            # Submit via Amulet CLI
            # --yes: Skip confirmation prompts
            # -y: Alternative flag for confirmation (some versions)
            cmd = [
                "uv",
                "run",
                "amlt",
                "run",
                tmp_path,
                experiment_name,
                "--yes",
                "--description",
                "'Magmathor experiment submission'",
            ]
            print(
                f"[AML] Submitting {len(experiments)} job(s) to experiment '{experiment_name}'..."
            )
            print(f"[AML] Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # Increased timeout for submission
                stdin=subprocess.DEVNULL,  # Don't wait for stdin
            )

            if result.stdout:
                print(f"[AML] stdout:\n{result.stdout}")
            if result.stderr:
                print(f"[AML] stderr:\n{result.stderr}")

            if result.returncode != 0:
                print(f"[AML] Submission failed with exit code {result.returncode}")
                return []

            print(f"[AML] Submission completed successfully")

            # Track submitted jobs
            for exp_config, test_suite in experiments:
                key = f"{exp_config.name}--{test_suite}"
                parts = exp_config.name.split("--")
                model_name = parts[0] if parts else "unknown"

                job_info = AmlJobInfo(
                    job_id=f"{experiment_name}:{key}",
                    experiment_key=key,
                    test_suite=test_suite,
                    model_name=model_name,
                    submit_time=datetime.now().isoformat(),
                    status="Submitted",
                    amulet_experiment=experiment_name,
                )
                self.jobs[key] = job_info
                submitted_jobs.append(key)

            self._save_jobs()
            print(f"[AML] Successfully submitted {len(submitted_jobs)} job(s)")

        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

        return submitted_jobs

    async def update_job_statuses(self) -> dict[str, str]:
        """Update statuses for all tracked jobs via Amulet CLI.

        Returns:
            Dict mapping experiment_key to new status
        """
        if not self.jobs:
            return {}

        # Group jobs by Amulet experiment
        experiments: dict[str, list[str]] = {}
        for key, job in self.jobs.items():
            exp_name = job.amulet_experiment
            if exp_name not in experiments:
                experiments[exp_name] = []
            experiments[exp_name].append(key)

        updated = {}

        for exp_name, job_keys in experiments.items():
            try:
                # Get status from Amulet
                # Use uv run to ensure we use the project's amlt
                result = subprocess.run(
                    ["uv", "run", "amlt", "status", exp_name, "--hide-urls"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "SHELL": "/bin/bash"},  # Avoid shell issues
                )

                if result.returncode != 0:
                    print(f"[AML] Failed to get status for {exp_name}: {result.stderr}")
                    continue

                # Parse the amlt status output
                # Format: ":index  :job-name  duration  STATUS  size  submitted  flags  url"
                # Job names look like: :gpt-5-2-medium-T0_Fs_H60_C0_P2_I1_R1_S1_E0_M4096-magt_benchmark_p3-rep2
                # Status values: pass, fail, killed, running, queued, etc.
                job_statuses = self._parse_amlt_status_output(result.stdout)

                # Match job statuses to our tracked jobs
                for key in job_keys:
                    job = self.jobs[key]
                    # Build the expected job name format that amlt uses
                    # Our key: "gpt-5.2-medium--T0_Fs_H60_C0_P2_I1_R1_S1_E0_M4096--rep2--magt_benchmark_p3"
                    # AMLT job: "gpt-5-2-medium-T0_Fs_H60_C0_P2_I1_R1_S1_E0_M4096-magt_benchmark_p3-rep2"
                    amlt_job_name = self._key_to_amlt_job_name(key)

                    if amlt_job_name in job_statuses:
                        new_status = job_statuses[amlt_job_name]
                        # Normalize status to capitalized form
                        new_status = self._normalize_status(new_status)
                        if new_status != job.status:
                            job.status = new_status
                            updated[key] = new_status

            except subprocess.TimeoutExpired:
                print(f"[AML] Timeout getting status for {exp_name}")
            except Exception as e:
                print(f"[AML] Error getting status for {exp_name}: {e}")

        if updated:
            self._save_jobs()

        return updated

    def _parse_amlt_status_output(self, output: str) -> dict[str, str]:
        """Parse amlt status output to extract job names and their statuses.

        Args:
            output: Raw output from `amlt status` command

        Returns:
            Dict mapping job names to their status (e.g., "pass", "fail", "running")
        """
        job_statuses = {}

        for line in output.splitlines():
            line = line.strip()
            # Skip empty lines, header lines, and summary lines
            if not line or line.startswith("#") or line.startswith("---"):
                continue
            if line.startswith("EXPERIMENT_NAME") or line.startswith("II "):
                continue

            # Job lines start with ":index" followed by ":job-name"
            # Format: ":0   :job-name  duration  STATUS  size  submitted  flags  url"
            if not line.startswith(":"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            # parts[0] = ":index", parts[1] = ":job-name", parts[2] = duration, parts[3] = status
            job_name_raw = parts[1]
            status = parts[3]

            # Remove leading colon from job name
            job_name = job_name_raw.lstrip(":")

            job_statuses[job_name] = status

        return job_statuses

    def _key_to_amlt_job_name(self, key: str) -> str:
        """Convert our experiment key format to AMLT job name format.

        Our key format: "model-name--config-string--repN--test_suite"
        AMLT job name format: "model-name-config-string-test_suite-repN"

        Note: AMLT sanitizes job names by replacing certain chars with dashes.
        """
        # Split by "--" to get parts
        parts = key.split("--")
        if len(parts) < 4:
            # Fallback: just sanitize the key
            return re.sub(r"[^a-zA-Z0-9_-]", "-", key)

        model_name = parts[0]
        config_string = parts[1]
        rep_part = parts[2]  # e.g., "rep1"
        test_suite = parts[3]

        # Build AMLT job name format: model-config-test_suite-repN
        # Also sanitize (replace dots, underscores in certain places, etc.)
        amlt_name = f"{model_name}-{config_string}-{test_suite}-{rep_part}"

        # AMLT sanitizes job names: replace special chars with dashes
        amlt_name = re.sub(r"[^a-zA-Z0-9_-]", "-", amlt_name)

        return amlt_name

    def _normalize_status(self, status: str) -> str:
        """Normalize amlt status values to our internal format.

        Args:
            status: Raw status from amlt (e.g., "pass", "fail", "running")

        Returns:
            Normalized status string
        """
        status_map = {
            "pass": "Completed",
            "passed": "Completed",
            "fail": "Failed",
            "failed": "Failed",
            "killed": "Killed",
            "running": "Running",
            "queued": "Queued",
            "pending": "Pending",
            "preparing": "Preparing",
            "canceled": "Cancelled",
            "cancelled": "Cancelled",
        }
        return status_map.get(status.lower(), status.capitalize())

    async def cancel_job(self, experiment_key: str) -> bool:
        """Cancel a submitted AML job.

        Args:
            experiment_key: The experiment key to cancel

        Returns:
            True if cancelled successfully
        """
        job = self.jobs.get(experiment_key)
        if not job:
            print(f"[AML] No job found for {experiment_key}")
            return False

        try:
            # Cancel via Amulet (syntax may vary)
            result = subprocess.run(
                ["amlt", "cancel", job.amulet_experiment, job.job_id.split(":")[-1]],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                job.status = "Canceled"
                self._save_jobs()
                print(f"[AML] Cancelled {experiment_key}")
                return True
            else:
                print(f"[AML] Failed to cancel: {result.stderr}")
                return False

        except Exception as e:
            print(f"[AML] Error cancelling job: {e}")
            return False

    def get_submitted_jobs_summary(self) -> dict[str, int]:
        """Get summary of submitted jobs by status.

        Returns:
            Dict mapping status to count
        """
        summary: dict[str, int] = {}
        for job in self.jobs.values():
            summary[job.status] = summary.get(job.status, 0) + 1
        return summary

    def clear_completed_jobs(self) -> int:
        """Remove completed/failed/cancelled jobs from tracking.

        Returns:
            Number of jobs cleared
        """
        terminal_statuses = {"Completed", "Failed", "Canceled", "Cancelled"}
        to_remove = [
            key for key, job in self.jobs.items() if job.status in terminal_statuses
        ]

        for key in to_remove:
            del self.jobs[key]

        if to_remove:
            self._save_jobs()

        return len(to_remove)

    def set_job_status(self, experiment_key: str, status: str) -> bool:
        """Manually set the status of a tracked job.

        Useful for marking jobs as Failed/Cancelled when auto-detection fails.

        Args:
            experiment_key: The experiment key (supports partial matching)
            status: New status to set

        Returns:
            True if status was updated
        """
        # Support partial matching
        matching = [k for k in self.jobs.keys() if experiment_key.lower() in k.lower()]

        if not matching:
            print(f"[AML] No job found matching '{experiment_key}'")
            return False

        if len(matching) > 1:
            print(f"[AML] Multiple matches found:")
            for key in matching:
                print(f"  - {key}")
            print("[AML] Please be more specific")
            return False

        key = matching[0]
        old_status = self.jobs[key].status
        self.jobs[key].status = status
        self._save_jobs()
        print(f"[AML] Updated {key}: {old_status} -> {status}")
        return True

    def remove_job(self, experiment_key: str) -> bool:
        """Remove a specific job from tracking (allows resubmission).

        Args:
            experiment_key: The experiment key (supports partial matching)

        Returns:
            True if job was removed
        """
        # Support partial matching
        matching = [k for k in self.jobs.keys() if experiment_key.lower() in k.lower()]

        if not matching:
            print(f"[AML] No job found matching '{experiment_key}'")
            return False

        if len(matching) > 1:
            print(f"[AML] Multiple matches found:")
            for key in matching:
                print(f"  - {key}")
            print("[AML] Please be more specific, or use 'aml remove-all <pattern>'")
            return False

        key = matching[0]
        del self.jobs[key]
        self._save_jobs()
        print(f"[AML] Removed {key} from tracking")
        return True

    def remove_jobs_matching(self, pattern: str) -> int:
        """Remove all jobs matching a pattern from tracking.

        Args:
            pattern: Pattern to match against job keys (case-insensitive)

        Returns:
            Number of jobs removed
        """
        matching = [k for k in self.jobs.keys() if pattern.lower() in k.lower()]

        if not matching:
            print(f"[AML] No jobs found matching '{pattern}'")
            return 0

        for key in matching:
            del self.jobs[key]

        self._save_jobs()
        print(f"[AML] Removed {len(matching)} job(s) matching '{pattern}'")
        return len(matching)
