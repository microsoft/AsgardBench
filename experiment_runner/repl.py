"""REPL (Read-Eval-Print Loop) for interactive experiment management."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table

from experiment_runner.results import display_results_summary, recompute_results
from experiment_runner.utils import parse_eta_from_log

if TYPE_CHECKING:
    from experiment_runner.runner import TestRunner

console = Console()

COMMANDS = [
    "help",
    "h",
    "?",
    "list",
    "ls",
    "l",
    "reload",
    "r",
    "status",
    "s",
    "logs",
    "tail",
    "t",
    "eta",
    "e",
    "results",
    "wait",
    "w",
    "quit",
    "q",
    "exit",
    "ssh",
    "pause",
    "resume",
    "rename",
    # AML commands
    "aml",
]


class Repl:
    """Interactive REPL for experiment management."""

    def __init__(self, runner: TestRunner):
        self.runner = runner
        self.session: PromptSession | None = None

    def _get_completer(self) -> WordCompleter:
        """Get word completer with commands and experiment names."""
        words = list(COMMANDS)
        words.extend(self.runner.get_all_experiment_keys())
        return WordCompleter(words, ignore_case=True)

    async def run(self) -> bool:
        """Run the REPL loop. Returns True if should exit gracefully."""
        self.session = PromptSession(
            history=InMemoryHistory(),
            completer=self._get_completer(),
        )

        self._print_help()
        print("\n[REPL] Ready for commands. Type 'help' for available commands.")
        print("[REPL] Tab completion is available for commands and experiment names.\n")

        while True:
            try:
                # Update completer with current experiment names
                self.session.completer = self._get_completer()

                user_input = await self.session.prompt_async("> ")

                if not user_input or not user_input.strip():
                    continue

                parts = user_input.strip().split()
                command = parts[0].lower()
                args = parts[1:]

                should_exit = await self._handle_command(command, args)
                if should_exit:
                    return True

                # Check for completed experiments after each command
                self.runner.check_completed_experiments()

            except KeyboardInterrupt:
                print("\n[REPL] Use 'quit' or 'exit' to exit.")
            except EOFError:
                return True

    async def _handle_command(self, command: str, args: list[str]) -> bool:
        """Handle a REPL command. Returns True if should exit."""
        if command in ("help", "h", "?"):
            self._print_help()

        elif command in ("list", "ls", "l"):
            self._list_running_experiments()

        elif command in ("reload", "r"):
            await self.runner.reload_catalogue()

        elif command in ("status", "s"):
            self._show_status()

        elif command == "logs":
            self._show_logs(args)

        elif command in ("tail", "t"):
            await self._tail_logs(args)

        elif command in ("eta", "e"):
            self._show_eta()

        elif command == "results":
            await recompute_results(self.runner.test_output_dir)

        elif command in ("wait", "w"):
            print("[REPL] Waiting for all experiments to complete...")
            await self.runner.wait_for_all_experiments()
            print("[REPL] All experiments completed!")
            return True

        elif command in ("quit", "q", "exit"):
            if await self._confirm_quit():
                self.runner.kill_all_experiments()
                return True

        elif command == "ssh":
            await self._ssh_command(args)

        elif command == "pause":
            await self._pause_command(args)

        elif command == "resume":
            await self._resume_command(args)

        elif command == "rename":
            await self._rename_command(args)

        elif command == "aml":
            await self._aml_command(args)

        else:
            print(
                f"[REPL] Unknown command: {command}. Type 'help' for available commands."
            )

        return False

    def _print_help(self) -> None:
        """Print help message."""
        table = Table(title="EXPERIMENT RUNNER - COMMANDS", title_style="cyan bold")
        table.add_column("Command", justify="center")
        table.add_column("Description")

        table.add_row("[green]help[/green], h, ?", "Show this help message")
        table.add_row("[green]list[/green], ls, l", "List all running experiments")
        table.add_row(
            "[green]reload[/green], r", "Reload catalogue and start new experiments"
        )
        table.add_row(
            "[green]status[/green], s", "Show summary status (running/completed/failed)"
        )
        table.add_row(
            "[green]logs[/green] <name>", "Show log file path for an experiment"
        )
        table.add_row(
            "[green]tail[/green], t <name>",
            "Tail the last 20 lines of an experiment's log",
        )
        table.add_row("[green]eta[/green], e", "Show ETA for all experiments")
        table.add_row(
            "[green]results[/green]",
            "Recompute the results.csv file from all test results",
        )
        table.add_row(
            "[green]wait[/green], w", "Wait for all experiments to complete, then exit"
        )
        table.add_row(
            "[green]quit[/green], q, exit",
            "Exit REPL (experiments continue in background)",
        )
        table.add_row("", "")
        table.add_row("[blue]--- AMLT Commands ---[/blue]", "")
        table.add_row(
            "[green]ssh[/green] <job> {port}",
            "Create SSH tunnel to AMLT job (default port: 43289)",
        )
        table.add_row("[green]pause[/green] <job>", "Pause an AMLT job")
        table.add_row("[green]resume[/green] <job>", "Resume a paused AMLT job")
        table.add_row("[green]rename[/green] <job> <new-name>", "Rename an AMLT job")
        table.add_row("", "")
        table.add_row("[blue]--- AML Job Submission ---[/blue]", "")
        table.add_row("[green]aml list[/green]", "List experiments with run_mode: aml")
        table.add_row(
            "[green]aml submit[/green]",
            "Submit all run_mode: aml experiments",
        )
        table.add_row(
            "[green]aml submit[/green] <name>",
            "Submit specific experiment (any run_mode)",
        )
        table.add_row("[green]aml status[/green]", "Show status of submitted AML jobs")
        table.add_row("[green]aml cancel[/green] <name>", "Cancel a submitted AML job")
        table.add_row(
            "[green]aml clear[/green]", "Clear completed/failed jobs from tracking"
        )
        table.add_row(
            "[green]aml dry-run[/green]",
            "Preview YAML without submitting",
        )
        table.add_row("", "")
        table.add_row(
            "[yellow]TAB[/yellow]", "Autocomplete commands and experiment names"
        )

        console.print()
        console.print(table)

    def _list_running_experiments(self) -> None:
        """List all running experiments."""
        running = list(self.runner.running_experiments.values())

        if not running:
            console.print("[yellow]No experiments currently running.[/yellow]")
            return

        table = Table(
            title=f"{len(running)} Running Experiment(s)", title_style="cyan bold"
        )
        table.add_column("Experiment", no_wrap=True)
        table.add_column("PID", justify="center")
        table.add_column("Duration", justify="center")

        for exp in sorted(running, key=lambda e: e.start_time):
            duration = datetime.now().timestamp() - exp.start_time
            duration_str = str(timedelta(seconds=int(duration)))
            pid = str(exp.process.pid) if exp.process else "N/A"
            table.add_row(exp.key, pid, duration_str)

        console.print()
        console.print(table)
        console.print()

    def _show_status(self) -> None:
        """Show status summary."""
        running = len(self.runner.running_experiments)
        completed = sum(1 for r in self.runner.completed_results if r.success)
        failed = sum(1 for r in self.runner.completed_results if not r.success)
        total = running + len(self.runner.completed_results)
        max_jobs = self.runner.max_concurrent_jobs

        table = Table(title="STATUS", title_style="cyan bold")
        table.add_column("Metric", justify="right")
        table.add_column("Value")

        if max_jobs > 0:
            table.add_row("Running", f"[blue]{running}[/blue] / {max_jobs}")
        else:
            table.add_row("Running", f"[blue]{running}[/blue]")
        table.add_row("Completed", f"[green]{completed}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else str(failed))
        table.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")

        console.print()
        console.print(table)
        console.print()

    def _show_logs(self, args: list[str]) -> None:
        """Show log file paths."""
        if not args:
            print("\n[LOGS] Log file locations:")
            for exp in sorted(
                self.runner.running_experiments.values(), key=lambda e: e.key
            ):
                print(f"  {exp.key}:")
                print(f"    {exp.log_file_path}")
            print()
            return

        search_term = args[0]
        matching = [
            e
            for e in self.runner.running_experiments.values()
            if search_term.lower() in e.key.lower()
        ]

        if not matching:
            print(f"[LOGS] No experiment matching '{search_term}' found.")
            return

        for exp in matching:
            print(f"[LOGS] {exp.key}:")
            print(f"       {exp.log_file_path}")

    async def _tail_logs(self, args: list[str]) -> None:
        """Tail last 20 lines of a log file."""
        if not args:
            print("[TAIL] Usage: tail <experiment-name-fragment>")
            return

        search_term = args[0]
        matching = [
            e
            for e in self.runner.running_experiments.values()
            if search_term.lower() in e.key.lower()
        ]

        if not matching:
            print(f"[TAIL] No experiment matching '{search_term}' found.")
            return

        if len(matching) > 1:
            print(f"[TAIL] Multiple matches found. Please be more specific:")
            for e in matching:
                print(f"  - {e.key}")
            return

        exp = matching[0]
        log_path = Path(exp.log_file_path)

        if not log_path.exists():
            print(f"[TAIL] Log file not yet created: {exp.log_file_path}")
            return

        lines = log_path.read_text().splitlines()
        last_lines = lines[-20:] if len(lines) > 20 else lines

        print(f"\n[TAIL] Last 20 lines of {exp.key}:")
        print("-" * 80)
        for line in last_lines:
            print(line)
        print("-" * 80 + "\n")

    def _show_eta(self) -> None:
        """Show ETA for running experiments."""
        running_experiments = list(self.runner.running_experiments.values())

        if not running_experiments:
            console.print("[yellow]No experiments currently running.[/yellow]")
            return

        table = Table(title="EXPERIMENT ETA", title_style="cyan bold")
        table.add_column("Experiment", no_wrap=True)
        table.add_column("Progress", justify="center")
        table.add_column("Elapsed", justify="center")
        table.add_column("ETA", justify="center")

        running_experiments.sort(key=lambda x: x.key)

        for exp in running_experiments:
            progress, elapsed, eta = parse_eta_from_log(exp.log_file_path, True)
            table.add_row(exp.key, progress, elapsed, eta)

        console.print()
        console.print(table)
        console.print()

    async def _confirm_quit(self) -> bool:
        """Confirm quit when experiments are running."""
        running_count = len(self.runner.running_experiments)

        if running_count == 0:
            print("[QUIT] No experiments running. Exiting.")
            return True

        print(f"[QUIT] There are {running_count} experiment(s) still running.")
        response = input(
            "[QUIT] Are you sure you want to kill them all and exit? (y/N): "
        )
        return response.strip().lower() in ("y", "yes")

    async def _ssh_command(self, args: list[str]) -> None:
        """Handle SSH tunnel command."""
        if not args:
            print("[SSH] Usage: ssh <amlt-job-name> [local-port]")
            print("[SSH] Example: ssh my-job-name 43289")
            return

        job_name = args[0]
        local_port = int(args[1]) if len(args) > 1 else 43289
        await self.runner.amlt_manager.start_ssh_tunnel(job_name, local_port)

    async def _pause_command(self, args: list[str]) -> None:
        """Handle pause AMLT job command."""
        if not args:
            print("[PAUSE] Usage: pause <amlt-job-name>")
            return

        await self.runner.amlt_manager.pause_job(args[0])

    async def _resume_command(self, args: list[str]) -> None:
        """Handle resume AMLT job command."""
        if not args:
            print("[RESUME] Usage: resume <amlt-job-name>")
            return

        await self.runner.amlt_manager.resume_job(args[0])

    async def _rename_command(self, args: list[str]) -> None:
        """Handle rename AMLT job command."""
        if len(args) < 2:
            print("[RENAME] Usage: rename <amlt-job-name> <new-name>")
            return

        await self.runner.amlt_manager.rename_job(args[0], args[1])

    async def _aml_command(self, args: list[str]) -> None:
        """Handle AML job submission commands."""
        if not args:
            self._print_aml_help()
            return

        subcommand = args[0].lower()
        subargs = args[1:]

        if subcommand == "submit":
            await self._aml_submit(subargs)
        elif subcommand == "status":
            await self._aml_status()
        elif subcommand == "cancel":
            await self._aml_cancel(subargs)
        elif subcommand == "clear":
            self._aml_clear()
        elif subcommand in ("dry-run", "dryrun", "preview"):
            await self._aml_submit(subargs, dry_run=True)
        elif subcommand == "config":
            self._aml_show_config()
        elif subcommand in ("list", "ls"):
            self._aml_list_experiments()
        elif subcommand in ("set-status", "setstatus"):
            self._aml_set_status(subargs)
        elif subcommand == "remove":
            self._aml_remove(subargs)
        elif subcommand == "remove-all":
            self._aml_remove_all(subargs)
        else:
            print(f"[AML] Unknown subcommand: {subcommand}")
            self._print_aml_help()

    def _print_aml_help(self) -> None:
        """Print AML command help."""
        print("\n[AML] Available subcommands:")
        print(
            "  aml list               List experiments marked for AML (run_mode: aml)"
        )
        print("  aml submit             Submit all experiments with run_mode: aml")
        print(
            "  aml submit <name>      Submit specific experiment by name (any run_mode)"
        )
        print("  aml status             Fetch & show status of submitted AML jobs")
        print("  aml cancel <name>      Cancel a submitted AML job")
        print("  aml clear              Clear completed/failed jobs from tracking")
        print("  aml dry-run            Preview YAML for run_mode: aml experiments")
        print("  aml config             Show current AML configuration")
        print()
        print("  Job management (for manual overrides if auto-detection fails):")
        print("  aml set-status <name> <status>  Manually set job status")
        print(
            "  aml remove <name>      Remove a job from tracking (allows resubmission)"
        )
        print("  aml remove-all <pattern>  Remove all jobs matching pattern")
        print()
        print(
            "  To mark an experiment for AML, add 'run_mode: aml' in the catalogue YAML."
        )
        print("  Jobs with Failed/Killed/Cancelled status will be auto-resubmitted.")
        print()

    async def _aml_submit(self, args: list[str], dry_run: bool = False) -> None:
        """Submit experiments to AML."""
        from experiment_runner.models import RunMode

        aml_launcher = self.runner.aml_launcher

        if not aml_launcher.is_configured():
            missing = aml_launcher.get_missing_config()
            print(f"[AML] Not configured. Missing: {', '.join(missing)}")
            print(f"[AML] Please edit: {aml_launcher.config_path}")
            return

        # Determine which experiments to submit
        experiments_to_submit: list[tuple] = []

        if not args or args[0] == "--all":
            # Submit all experiments marked with run_mode: aml
            for exp in self.runner.catalogue.experiments:
                if exp.completed or not exp.settings:
                    continue
                # Only submit experiments marked for AML
                if exp.run_mode != RunMode.AML:
                    continue
                for test_suite in self.runner.test_suites:
                    key = f"{exp.name}--{test_suite}"
                    # Skip if already running locally
                    if key in self.runner.running_experiments:
                        continue
                    # Skip if already submitted to AML
                    if aml_launcher.is_job_submitted(key):
                        continue
                    experiments_to_submit.append((exp, test_suite))
        else:
            # Submit specific experiment by name (partial match) - regardless of run_mode
            search_term = args[0].lower()
            for exp in self.runner.catalogue.experiments:
                if search_term in exp.name.lower():
                    if exp.settings:
                        for test_suite in self.runner.test_suites:
                            key = f"{exp.name}--{test_suite}"
                            if not aml_launcher.is_job_submitted(key):
                                experiments_to_submit.append((exp, test_suite))

        if not experiments_to_submit:
            print("[AML] No experiments to submit")
            return

        print(f"[AML] Found {len(experiments_to_submit)} experiment(s) to submit")

        # Check for mounted storage conflicts before submitting
        if not dry_run:
            has_conflict, conflict_msg = aml_launcher.check_mounted_storage_conflict()
            if has_conflict:
                print(f"[AML] WARNING: {conflict_msg}")
                if self.session:
                    response = await self.session.prompt_async(
                        "[AML] Continue anyway? (y/N): "
                    )
                else:
                    response = input("[AML] Continue anyway? (y/N): ")
                if response.strip().lower() not in ("y", "yes"):
                    print("[AML] Submission cancelled")
                    return

        if dry_run:
            print("[AML] Dry run mode - generating YAML preview...")

        try:
            submitted = await aml_launcher.submit_experiments(
                experiments_to_submit,
                dry_run=dry_run,
                force=True,  # Skip conflict check in launcher (handled above)
            )
            if submitted and not dry_run:
                print(f"[AML] Successfully submitted {len(submitted)} job(s)")
        except ValueError as e:
            print(f"[AML] Error: {e}")

    async def _aml_status(self) -> None:
        """Show status of submitted AML jobs."""
        aml_launcher = self.runner.aml_launcher

        # Update statuses from Amulet
        print("[AML] Fetching job statuses from AML...")
        updated = await aml_launcher.update_job_statuses()

        if updated:
            print(f"[AML] Updated {len(updated)} job status(es)")

        if not aml_launcher.jobs:
            print("[AML] No submitted jobs tracked")
            return

        # Display summary
        summary = aml_launcher.get_submitted_jobs_summary()
        table = Table(title="AML JOB STATUS", title_style="cyan bold")
        table.add_column("Experiment", no_wrap=True)
        table.add_column("Test Suite")
        table.add_column("Model")
        table.add_column("Status", justify="center")
        table.add_column("Submitted", justify="center")

        for key, job in sorted(aml_launcher.jobs.items()):
            status_style = {
                "Running": "blue",
                "Queued": "cyan",
                "Pending": "cyan",
                "Preparing": "cyan",
                "Completed": "green",
                "Failed": "red",
                "Killed": "red",
                "Canceled": "yellow",
                "Cancelled": "yellow",
            }.get(job.status, "white")

            table.add_row(
                job.experiment_key.split("--")[0][:40],
                job.test_suite,
                job.model_name,
                f"[{status_style}]{job.status}[/{status_style}]",
                job.submit_time[:16],
            )

        console.print()
        console.print(table)

        # Summary line
        summary_parts = [
            f"{status}: {count}" for status, count in sorted(summary.items())
        ]
        print(f"\n[AML] Summary: {', '.join(summary_parts)}")

        # Show resubmittable count
        resubmittable = {"Failed", "Killed", "Cancelled", "Canceled", "Error"}
        resubmittable_count = sum(
            1 for job in aml_launcher.jobs.values() if job.status in resubmittable
        )
        if resubmittable_count > 0:
            print(
                f"[AML] {resubmittable_count} job(s) can be resubmitted (run 'aml submit' to resubmit)"
            )
        print()

    async def _aml_cancel(self, args: list[str]) -> None:
        """Cancel a submitted AML job."""
        if not args:
            print("[AML] Usage: aml cancel <experiment-name>")
            return

        search_term = args[0].lower()
        aml_launcher = self.runner.aml_launcher

        # Find matching jobs
        matching = [
            key for key in aml_launcher.jobs.keys() if search_term in key.lower()
        ]

        if not matching:
            print(f"[AML] No jobs found matching '{args[0]}'")
            return

        if len(matching) > 1:
            print(f"[AML] Multiple matches found:")
            for key in matching:
                print(f"  - {key}")
            print("[AML] Please be more specific")
            return

        key = matching[0]
        await aml_launcher.cancel_job(key)

    def _aml_clear(self) -> None:
        """Clear completed/failed jobs from tracking."""
        aml_launcher = self.runner.aml_launcher
        cleared = aml_launcher.clear_completed_jobs()
        print(f"[AML] Cleared {cleared} completed/failed job(s) from tracking")

    def _aml_set_status(self, args: list[str]) -> None:
        """Manually set the status of a tracked job."""
        if len(args) < 2:
            print("[AML] Usage: aml set-status <experiment-name> <status>")
            print(
                "[AML] Valid statuses: Submitted, Running, Completed, Failed, Cancelled"
            )
            return

        experiment_key = args[0]
        status = args[1]

        # Normalize common status aliases
        status_map = {
            "failed": "Failed",
            "cancelled": "Cancelled",
            "canceled": "Cancelled",
            "completed": "Completed",
            "running": "Running",
            "submitted": "Submitted",
            "error": "Error",
        }
        status = status_map.get(status.lower(), status)

        aml_launcher = self.runner.aml_launcher
        aml_launcher.set_job_status(experiment_key, status)

    def _aml_remove(self, args: list[str]) -> None:
        """Remove a specific job from tracking."""
        if not args:
            print("[AML] Usage: aml remove <experiment-name>")
            return

        aml_launcher = self.runner.aml_launcher
        aml_launcher.remove_job(args[0])

    def _aml_remove_all(self, args: list[str]) -> None:
        """Remove all jobs matching a pattern from tracking."""
        if not args:
            print("[AML] Usage: aml remove-all <pattern>")
            print("[AML] Example: aml remove-all gpt-5.2-medium")
            return

        aml_launcher = self.runner.aml_launcher
        aml_launcher.remove_jobs_matching(args[0])

    def _aml_show_config(self) -> None:
        """Show current AML configuration."""
        aml_launcher = self.runner.aml_launcher
        print("\n[AML] Current configuration:")
        for key, value in aml_launcher.config.items():
            # Mask sensitive values
            if "identity" in key.lower() or "key" in key.lower():
                display_value = "***" if value else "(not set)"
            else:
                display_value = value if value else "(not set)"
            print(f"  {key}: {display_value}")
        print(f"\n[AML] Config file: {aml_launcher.config_path}")
        print()

    def _aml_list_experiments(self) -> None:
        """List experiments marked for AML execution."""
        from experiment_runner.models import RunMode

        aml_experiments = [
            exp
            for exp in self.runner.catalogue.experiments
            if exp.run_mode == RunMode.AML and not exp.completed
        ]
        local_experiments = [
            exp
            for exp in self.runner.catalogue.experiments
            if exp.run_mode == RunMode.LOCAL and not exp.completed
        ]

        if not aml_experiments:
            print("\n[AML] No experiments marked for AML (run_mode: aml)")
            print(
                f"[AML] {len(local_experiments)} experiment(s) marked for local execution"
            )
            print("\n[AML] To mark an experiment for AML, add to the catalogue:")
            print("      - name: my-experiment")
            print("        run_mode: aml")
            print("        settings:")
            print("          model: gpt-4o")
            print()
            return

        table = Table(title="EXPERIMENTS MARKED FOR AML", title_style="cyan bold")
        table.add_column("Experiment", no_wrap=True)
        table.add_column("Model")
        table.add_column("Replicates", justify="center")
        table.add_column("Submitted", justify="center")

        aml_launcher = self.runner.aml_launcher

        for exp in aml_experiments:
            model = exp.settings.model if exp.settings else "?"
            reps = exp.settings.replicates if exp.settings else 1

            # Check if any test suite for this experiment is already submitted
            submitted_count = sum(
                1
                for ts in self.runner.test_suites
                if aml_launcher.is_job_submitted(f"{exp.name}--{ts}")
            )
            submitted_str = (
                f"[green]{submitted_count}/{len(self.runner.test_suites)}[/green]"
                if submitted_count > 0
                else "-"
            )

            table.add_row(
                exp.name[:60],
                model,
                str(reps),
                submitted_str,
            )

        console.print()
        console.print(table)
        print(
            f"\n[AML] {len(aml_experiments)} experiment(s) for AML, {len(local_experiments)} for local"
        )
        print()
