"""Main TestRunner orchestration class."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from experiment_runner.aml_launcher import AmlLauncher
from experiment_runner.amlt import AmltManager
from experiment_runner.models import (
    ExperimentCatalogue,
    ExperimentConfig,
    ExperimentSettings,
    ExperimentState,
    RunningExperiment,
    TestResult,
    load_experiment_catalogue,
)
from experiment_runner.repl import Repl
from experiment_runner.results import report_final_results
from experiment_runner.utils import (
    XvfbManager,
    detect_gpus,
    download_test_data,
    interleave_by_model,
    is_log_file_successfully_completed,
    is_process_running,
    is_test_results_complete,
    kill_process_tree,
    start_process_with_log_redirect,
)
from Magmathor.constants import TEST_DIR


@dataclass
class TestRunner:
    """Main experiment runner orchestration."""

    # Configuration
    catalogue_path: str = "experiment_runner/experiment_catalogue.yaml"
    state_file_path: str = "experiment_runner/experiment_state.json"
    test_output_dir: str = str(TEST_DIR)
    max_concurrent_jobs: int = 0  # 0 means no limit
    test_suites: list[str] = field(
        default_factory=lambda: [
            # "test_new_positions_1",
            # "test_new_positions_2",
            # "test_new_rooms",
            # "test_new_tasks",
            # "benchmark",
            # "mt_benchmark",
            # "magt_benchmark",
            # "prompt_creation_set",
            "magt_benchmark_p1",
            "magt_benchmark_p2",
            "magt_benchmark_p3",
            "magt_benchmark_p4",
            "magt_benchmark_p5",
            "magt_benchmark_p6",
        ]
    )

    # Managers
    xvfb_manager: XvfbManager = field(default_factory=XvfbManager)
    amlt_manager: AmltManager = field(default_factory=AmltManager)
    aml_launcher: AmlLauncher = field(default_factory=AmlLauncher)

    # State
    catalogue: ExperimentCatalogue = field(default_factory=ExperimentCatalogue)
    running_experiments: dict[str, RunningExperiment] = field(default_factory=dict)
    completed_results: list[TestResult] = field(default_factory=list)
    display_num: int = 99
    cuda_devices: str = ""
    _launcher_task: asyncio.Task[int] | None = field(default=None, repr=False)

    async def run_interactive(self) -> int:
        """Run the interactive experiment runner."""
        try:
            # Setup environment
            self.display_num = self.xvfb_manager.setup()
            self.cuda_devices = detect_gpus()

            # Download test data if needed
            await download_test_data()

            # Load experiment catalogue
            self.catalogue = load_experiment_catalogue(self.catalogue_path)

            # Setup SSH tunnels for AMLT jobs
            await self._setup_amlt_tunnels()

            # Cleanup stale experiments from previous runs
            self._cleanup_stale_experiments()

            # Start experiment launching in background (allows REPL to run while waiting for slots)
            self._launcher_task = asyncio.create_task(self._start_new_experiments())

            # Run REPL (concurrently with experiment launching)
            repl = Repl(self)
            await repl.run()

            # Cancel launcher task if still running
            if self._launcher_task and not self._launcher_task.done():
                self._launcher_task.cancel()
                try:
                    await self._launcher_task
                except asyncio.CancelledError:
                    pass

            # Wait for remaining experiments
            await self.wait_for_all_experiments()

            # Report final results
            report_final_results(self.completed_results)

            return 0 if all(r.success for r in self.completed_results) else 1

        finally:
            self.xvfb_manager.cleanup()

    async def _setup_amlt_tunnels(self) -> None:
        """Setup SSH tunnels and populate vLLM endpoints."""
        endpoints_by_experiment = await self.amlt_manager.setup_ssh_tunnels(
            self.catalogue.experiments
        )

        # Populate vllm_endpoints on experiments
        for exp_name, endpoints in endpoints_by_experiment.items():
            for exp in self.catalogue.experiments:
                if exp.name == exp_name:
                    if exp.settings:
                        exp.settings.vllm_endpoints = endpoints
                    break

            print(f"[AMLT] Set {len(endpoints)} vLLM endpoint(s) for '{exp_name}'")

    def _cleanup_stale_experiments(self) -> None:
        """Clean up experiments from previous runs that are no longer running."""
        states = self._load_state()
        active_states = []

        if not states:
            print("[STATE] No active experiments found")
        else:
            for state in states:
                if is_process_running(state.pid):
                    print(
                        f"[STATE] Experiment '{state.experiment_name}' ({state.test_suite}) "
                        f"is still running (PID: {state.pid})"
                    )
                    active_states.append(state)
                else:
                    print(
                        f"[STATE] Process {state.pid} for '{state.experiment_name}' "
                        f"({state.test_suite}) no longer exists"
                    )

        self._save_state(active_states)

    def _has_available_slot(self) -> bool:
        """Check if there's an available slot for a new experiment."""
        if self.max_concurrent_jobs <= 0:
            return True  # No limit
        return len(self.running_experiments) < self.max_concurrent_jobs

    async def _wait_for_available_slot(self) -> None:
        """Wait until a slot becomes available for a new experiment."""
        while not self._has_available_slot():
            self.check_completed_experiments()
            if not self._has_available_slot():
                await asyncio.sleep(30)

    async def _start_new_experiments(self) -> int:
        """Start new experiments from the catalogue. Returns count of started experiments."""
        from experiment_runner.models import RunMode

        started_count = 0

        # Filter to local experiments only and interleave by model
        local_experiments = [
            exp
            for exp in self.catalogue.experiments
            if not exp.completed and exp.run_mode != RunMode.AML and exp.settings
        ]
        interleaved_experiments = interleave_by_model(local_experiments)

        print(
            f"[EXPERIMENT] {len(interleaved_experiments)} local experiment(s) to run "
            f"(interleaved by model)"
        )

        for experiment in interleaved_experiments:
            settings = experiment.settings
            if settings is None:
                continue

            # Settings are already fully merged during catalogue expansion
            # (base_settings + experiment overrides were merged in _expand_experiment)

            # experiment.name already includes --rep{n} suffix from expansion
            experiment_name = experiment.name

            for test_suite in self.test_suites:
                key = f"{experiment_name}--{test_suite}"

                # Skip if already running
                if key in self.running_experiments:
                    continue

                # Skip if already completed
                if any(r.test_name == key for r in self.completed_results):
                    continue

                # Skip if running from previous session
                if self._is_experiment_running(experiment_name, test_suite):
                    print(
                        f"[EXPERIMENT] Skipping {key} - already running from previous session"
                    )
                    continue

                # Skip if test_results.json shows successful completion
                test_results_path = self._get_test_results_path(
                    test_suite, experiment_name
                )
                is_complete, completed, expected = is_test_results_complete(
                    test_results_path
                )
                if is_complete:
                    print(
                        f"[EXPERIMENT] Skipping {key} - test_results.json shows "
                        f"{completed}/{expected} tasks completed"
                    )
                    continue

                # Wait for an available slot before starting
                await self._wait_for_available_slot()

                # Start the experiment
                running_exp = self._start_experiment(
                    test_suite, experiment_name, settings
                )

                time.sleep(0.5)  # slight delay to avoid resource contention

                self.running_experiments[key] = running_exp
                started_count += 1

                print(
                    f"[EXPERIMENT] Started {key} "
                    f"(PID: {running_exp.process.pid if running_exp.process else 'N/A'}, "
                    f"logs: {running_exp.log_file_path})"
                )

        return started_count

    def _start_experiment(
        self,
        test_suite: str,
        experiment_name: str,
        settings: ExperimentSettings,
    ) -> RunningExperiment:
        """Start a single experiment process."""
        log_file_path = self._get_log_file_path(test_suite, experiment_name)
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Build command
        args = settings.build_command_args(test_suite, experiment_name)
        command = ["uv"] + args

        # Start process
        process = start_process_with_log_redirect(
            command,
            log_file_path,
            self.display_num,
            self.cuda_devices,
        )

        # Register in state file
        self._register_experiment(experiment_name, test_suite, process.pid)

        return RunningExperiment(
            experiment_name=experiment_name,
            test_suite=test_suite,
            process=process,
            start_time=time.time(),
            log_file_path=log_file_path,
        )

    def check_completed_experiments(self) -> None:
        """Check for and handle completed experiments."""
        completed_keys = []

        for key, exp in self.running_experiments.items():
            if exp.process and exp.process.poll() is not None:
                completed_keys.append(key)

        for key in completed_keys:
            exp = self.running_experiments.pop(key)

            result = TestResult(
                test_name=key,
                success=exp.process.returncode == 0,
                exit_code=exp.process.returncode,
                output=f"Logs written to: {exp.log_file_path}",
                errors="",
            )
            self.completed_results.append(result)

            self._unregister_experiment(exp.experiment_name, exp.test_suite)

            status = "✓ COMPLETED" if result.success else "✗ FAILED"
            print(f"\n[EXPERIMENT] {status}: {key} (exit code: {result.exit_code})")
            print(f"             Logs: {exp.log_file_path}")

    async def wait_for_all_experiments(self) -> None:
        """Wait for all running experiments to complete."""
        while self.running_experiments:
            self.check_completed_experiments()
            if self.running_experiments:
                await asyncio.sleep(1)

    def kill_all_experiments(self) -> None:
        """Kill all running experiments."""
        print("[QUIT] Killing all running experiments...")

        for key, exp in list(self.running_experiments.items()):
            try:
                if exp.process and exp.process.poll() is None:
                    print(f"[QUIT] Killing {key} (PID: {exp.process.pid})")
                    kill_process_tree(exp.process.pid)
            except Exception as e:
                print(f"[QUIT] Error killing {key}: {e}")

            self._unregister_experiment(exp.experiment_name, exp.test_suite)

        self.running_experiments.clear()
        print("[QUIT] All experiments killed. Goodbye!")

    async def reload_catalogue(self) -> None:
        """Reload the experiment catalogue and start new experiments."""
        print("[RELOAD] Reloading experiment catalogue...")

        try:
            # Cancel any existing launcher task
            if self._launcher_task and not self._launcher_task.done():
                self._launcher_task.cancel()
                try:
                    await self._launcher_task
                except asyncio.CancelledError:
                    pass

            self.catalogue = load_experiment_catalogue(self.catalogue_path)
            print(
                f"[RELOAD] Loaded {len(self.catalogue.experiments)} experiments from {self.catalogue_path}"
            )

            # Setup SSH tunnels for any new AMLT jobs
            await self._setup_amlt_tunnels()

            # Start launching in background
            self._launcher_task = asyncio.create_task(self._start_new_experiments())
            print("[RELOAD] Experiment launching started in background.")

        except Exception as e:
            print(f"[RELOAD] Error reloading catalogue: {e}")

    def get_all_experiment_keys(self) -> list[str]:
        """Get all experiment keys (running and completed)."""
        running = list(self.running_experiments.keys())
        completed = [r.test_name for r in self.completed_results]
        return list(set(running + completed))

    def _get_log_file_path(self, test_suite: str, experiment_name: str) -> str:
        """Get the log file path for an experiment."""
        return str(
            Path(self.test_output_dir) / test_suite / experiment_name / "logs.txt"
        )

    def _get_test_results_path(self, test_suite: str, experiment_name: str) -> str:
        """Get the test_results.json path for an experiment."""
        return str(
            Path(self.test_output_dir)
            / test_suite
            / experiment_name
            / "test_results.json"
        )

    # --- State Persistence ---

    def _load_state(self) -> list[ExperimentState]:
        """Load experiment state from file."""
        if not Path(self.state_file_path).exists():
            return []

        try:
            with open(self.state_file_path) as f:
                data = json.load(f)
            return [ExperimentState.from_dict(d) for d in data]
        except Exception as e:
            print(f"[STATE] Error loading state file: {e}. Starting fresh.")
            return []

    def _save_state(self, states: list[ExperimentState]) -> None:
        """Save experiment state to file."""
        try:
            with open(self.state_file_path, "w") as f:
                json.dump([s.to_dict() for s in states], f, indent=2)
        except Exception as e:
            print(f"[STATE] Error saving state file: {e}")

    def _is_experiment_running(self, experiment_name: str, test_suite: str) -> bool:
        """Check if an experiment is running according to state file."""
        states = self._load_state()
        return any(
            s.experiment_name == experiment_name and s.test_suite == test_suite
            for s in states
        )

    def _register_experiment(
        self, experiment_name: str, test_suite: str, pid: int
    ) -> None:
        """Register an experiment in the state file."""
        states = self._load_state()
        states.append(
            ExperimentState(
                experiment_name=experiment_name,
                test_suite=test_suite,
                pid=pid,
                start_time=datetime.utcnow().isoformat(),
            )
        )
        self._save_state(states)

    def _unregister_experiment(self, experiment_name: str, test_suite: str) -> None:
        """Unregister an experiment from the state file."""
        states = self._load_state()
        states = [
            s
            for s in states
            if not (s.experiment_name == experiment_name and s.test_suite == test_suite)
        ]
        self._save_state(states)
