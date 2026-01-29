"""AMLT (Azure ML) job management and SSH tunnel setup."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field

from experiment_runner.utils import is_port_in_use, wait_for_url_reachable


@dataclass
class AmltManager:
    """Manages AMLT jobs and SSH tunnels."""

    port_assignments: list[tuple[str, str, int]] = field(
        default_factory=list
    )  # (experiment_name, job_name, port)
    base_port: int = 43289

    def sanitize_tmux_name(self, name: str) -> str:
        """Sanitize a name for use as tmux window name."""
        return re.sub(r"[:./ ]", lambda m: "-" if m.group() != " " else "_", name)

    async def setup_ssh_tunnels(
        self,
        experiments: list,  # list of ExperimentConfig
    ) -> dict[str, list[str]]:
        """Set up SSH tunnels for all experiments with AMLT job names.

        Returns: dict mapping experiment name to list of endpoint URLs.
        """
        # Collect experiments with AMLT jobs
        experiments_with_jobs = [
            {"name": e.name, "jobs": e.settings.amlt_job_names}
            for e in experiments
            if e.settings and e.settings.amlt_job_names
        ]

        if not experiments_with_jobs:
            print("[AMLT] No AMLT job names configured in experiments.")
            return {}

        total_jobs = sum(len(e["jobs"]) for e in experiments_with_jobs)
        print(
            f"[cyan]Found {total_jobs} AMLT job(s) across {len(experiments_with_jobs)} experiment(s). Setting up SSH tunnels...[/]"
        )

        # First, resume all paused jobs
        all_job_names = list({job for e in experiments_with_jobs for job in e["jobs"]})
        for job_name in all_job_names:
            print(f"[cyan]Resuming AMLT job '{job_name}' (if paused)...[/]")
            await self._run_amlt_command("resume", job_name)

        # Track port assignments
        current_port = self.base_port
        self.port_assignments = []

        # Process each experiment
        for exp_group in experiments_with_jobs:
            experiment_name = exp_group["name"]
            jobs = exp_group["jobs"]
            window_name = self.sanitize_tmux_name(experiment_name)

            print(
                f"\n[cyan]Setting up SSH tunnels for experiment '{experiment_name}' ({len(jobs)} job(s))...[/]"
            )

            ports_for_experiment = []
            for job_name in jobs:
                port = current_port
                current_port += 1
                ports_for_experiment.append((job_name, port))
                self.port_assignments.append((experiment_name, job_name, port))

            # Check which ports are available
            jobs_to_start = []
            for job_name, port in ports_for_experiment:
                if is_port_in_use(port):
                    print(
                        f"[yellow]  Port {port} already in use - tunnel for '{job_name}' may already be active[/]"
                    )
                else:
                    jobs_to_start.append((job_name, port))

            # Start SSH tunnels
            if jobs_to_start:
                await self._start_ssh_tunnels_for_experiment(window_name, jobs_to_start)
            else:
                print(
                    f"[AMLT] All ports for '{experiment_name}' already in use - skipping tunnel creation"
                )

        # Wait for tunnels to establish
        print("\n[AMLT] Waiting for all tunnels to establish...")
        await asyncio.sleep(10)

        # Verify all URLs are reachable
        all_reachable = True
        for experiment_name, job_name, port in self.port_assignments:
            url = f"http://localhost:{port}"
            print(f"[AMLT] Verifying {url} for job '{job_name}' ({experiment_name})...")

            reachable = await wait_for_url_reachable(
                url, max_attempts=1000, delay_seconds=10
            )

            if reachable:
                print(f"[green]✓ {job_name} is reachable at {url}[/]")
            else:
                print(f"[red]✗ {job_name} not reachable at {url}[/]")
                all_reachable = False

        if all_reachable:
            print("\n[green]All AMLT SSH tunnels established successfully![/]")
        else:
            print(
                "\n[yellow]Some tunnels may not be ready. You can manually retry with 'ssh <job-name>'[/]"
            )

        # Build endpoint map
        endpoints_by_experiment: dict[str, list[str]] = {}
        for experiment_name, _job_name, port in self.port_assignments:
            if experiment_name not in endpoints_by_experiment:
                endpoints_by_experiment[experiment_name] = []
            endpoints_by_experiment[experiment_name].append(
                f"http://localhost:{port}/v1"
            )

        # Set REMOTE_URL environment variable
        if len(self.port_assignments) == 1:
            url = f"http://localhost:{self.port_assignments[0][2]}/v1"
            os.environ["REMOTE_URL"] = url
            print(f"[AMLT] Set REMOTE_URL={url}")
        elif len(self.port_assignments) > 1:
            urls = ",".join(
                f"http://localhost:{p[2]}/v1" for p in self.port_assignments
            )
            os.environ["REMOTE_URL"] = urls
            print(f"[AMLT] Set REMOTE_URL={urls}")

        return endpoints_by_experiment

    async def _start_ssh_tunnels_for_experiment(
        self,
        window_name: str,
        jobs: list[tuple[str, int]],  # (job_name, port)
    ) -> None:
        """Start SSH tunnels in tmux/byobu windows."""
        # Check if we're in tmux
        tmux_env = os.environ.get("TMUX")
        if not tmux_env:
            print(f"[AMLT] Not in tmux/byobu session - starting tunnels in background")
            for job_name, port in jobs:
                await self._start_ssh_tunnel_background(job_name, port)
            return

        is_first_pane = True
        window_index = None
        remote_port = 43289

        for job_name, port in jobs:
            try:
                if is_first_pane:
                    # Create new window
                    result = await self._run_tmux_command(
                        f"new-window -a -n {window_name} -P -F '#{{window_index}}'"
                    )
                    if result.returncode != 0:
                        print(
                            f"[AMLT] Failed to create tmux window for '{window_name}'"
                        )
                        await self._start_ssh_tunnel_background(job_name, port)
                        continue

                    # Parse window index
                    window_output = (
                        result.stdout.strip().strip("'\"").split(":")[0].strip()
                    )
                    try:
                        window_index = int(window_output)
                    except ValueError:
                        print(
                            f"[AMLT] Could not parse window index from: '{window_output}'"
                        )

                    await asyncio.sleep(0.3)

                    # Send SSH command
                    ssh_command = f"uv run amlt ssh {job_name} -o StrictHostKeyChecking=no -o '-4 -L {port}:localhost:{remote_port}'"
                    await self._run_tmux_command(
                        f'send-keys -t {window_index} "{ssh_command}" Enter'
                    )

                    print(
                        f"[green]  Created window '{window_name}' with tunnel for '{job_name}' on port {port}[/]"
                    )
                    is_first_pane = False
                else:
                    if window_index is None:
                        print(
                            f"[AMLT] No window index available, falling back to background"
                        )
                        await self._start_ssh_tunnel_background(job_name, port)
                        continue

                    # Split window for additional tunnels
                    result = await self._run_tmux_command(
                        f"split-window -t {window_index} -h -P -F '#{{pane_id}}'"
                    )
                    if result.returncode != 0:
                        # Try vertical split
                        result = await self._run_tmux_command(
                            f"split-window -t {window_index} -v -P -F '#{{pane_id}}'"
                        )
                        if result.returncode != 0:
                            print(f"[AMLT] Split failed")
                            await self._start_ssh_tunnel_background(job_name, port)
                            continue

                    pane_id = result.stdout.strip().strip("'\"")
                    await asyncio.sleep(0.3)

                    # Send SSH command
                    ssh_command = f"uv run amlt ssh {job_name} -o StrictHostKeyChecking=no -o '-4 -L {port}:localhost:{remote_port}'"
                    await self._run_tmux_command(
                        f'send-keys -t {pane_id} "{ssh_command}" Enter'
                    )

                    print(f"[green]  Added pane for '{job_name}' on port {port}[/]")

                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"[AMLT] Error creating pane for '{job_name}': {e}")
                await self._start_ssh_tunnel_background(job_name, port)

        # Rebalance panes
        if len(jobs) > 1 and window_index is not None:
            await self._run_tmux_command(f"select-layout -t {window_index} tiled")

    async def _start_ssh_tunnel_background(
        self, job_name: str, local_port: int
    ) -> None:
        """Start SSH tunnel as background process."""
        remote_port = 43289
        ssh_args = f'ssh "{job_name}" -o "StrictHostKeyChecking=no" -o "-4 -L {local_port}:localhost:{remote_port}"'

        print(f"[green]Starting SSH tunnel for '{job_name}' on port {local_port}...[/]")
        print(f"  Command: amlt {ssh_args}")

        try:
            proc = await asyncio.create_subprocess_shell(
                f"uv run amlt {ssh_args}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            print(f"  Started in background (PID: {proc.pid})")
            await asyncio.sleep(5)
        except Exception as e:
            print(f"[red]Error starting SSH tunnel for '{job_name}': {e}[/]")

    async def start_ssh_tunnel(self, job_name: str, local_port: int = 43289) -> None:
        """Manually start SSH tunnel (REPL command)."""
        remote_port = 43289

        if is_port_in_use(local_port):
            print(f"[red]Error: Local port {local_port} is already in use.[/]")
            print(
                f"You can check what's using it with: lsof -iTCP:{local_port} -sTCP:LISTEN -Pn"
            )
            return

        print(f"[green]Starting SSH tunnel to AMLT job '{job_name}'...[/]")
        print(f"  Local port: {local_port} -> Remote port: {remote_port}")

        ssh_args = f'ssh "{job_name}" -o "StrictHostKeyChecking=no" -o "-4 -L {local_port}:localhost:{remote_port}"'
        print(f"  Command: amlt {ssh_args}")

        # Try byobu/tmux first
        if os.environ.get("TMUX"):
            await self._start_in_byobu(job_name, local_port, ssh_args)
        else:
            await self._start_ssh_tunnel_background(job_name, local_port)

        print("\nWaiting for tunnel to establish...")
        await asyncio.sleep(5)

        url = f"http://localhost:{local_port}"
        print(f"Checking if {url} is reachable...")

        reachable = await wait_for_url_reachable(
            url, max_attempts=1000, delay_seconds=5
        )

        if reachable:
            print(f"[green]✓ SSH tunnel is active and {url} is reachable![/]")
        else:
            print(
                f"[yellow]⚠ Tunnel may be establishing. URL {url} not yet reachable.[/]"
            )
            print("  The tunnel might still be starting. Try accessing it manually.")

    async def _start_in_byobu(
        self, job_name: str, local_port: int, amlt_args: str
    ) -> bool:
        """Start SSH tunnel in byobu/tmux window."""
        window_name = "ssh-tunnels"
        pane_title = f"{job_name}:{local_port}"
        command = f"echo 'SSH Tunnel: {job_name} -> localhost:{local_port}'; echo 'Command: uv run amlt {amlt_args}'; echo ''; uv run amlt {amlt_args}"

        # Check if window exists
        result = subprocess.run(
            [
                "bash",
                "-c",
                f"tmux list-windows -F '#{{window_name}}' | grep -q '^{window_name}$'",
            ],
            capture_output=True,
        )
        window_exists = result.returncode == 0

        if not window_exists:
            # Create new window
            await self._run_tmux_command(f'new-window -n "{window_name}" "{command}"')
            await self._run_tmux_command(f'select-pane -T "{pane_title}"')
            print(f"[green]SSH tunnel started in new byobu window '{window_name}'[/]")
        else:
            # Split existing window
            result = await self._run_tmux_command(
                f'split-window -t "{window_name}:" -h "{command}"'
            )
            if result.returncode != 0:
                await self._run_tmux_command(
                    f'split-window -t "{window_name}:" -v "{command}"'
                )

            await self._run_tmux_command(f'select-pane -T "{pane_title}"')
            await self._run_tmux_command(f'select-layout -t "{window_name}:" tiled')
            print(
                f"[green]SSH tunnel started in new pane in byobu window '{window_name}'[/]"
            )

        return True

    async def pause_job(self, job_name: str) -> None:
        """Pause an AMLT job."""
        print(f"[yellow]Pausing AMLT job '{job_name}'...[/]")
        result = await self._run_amlt_command("pause", job_name)

        if result.returncode == 0:
            print(f"[green]✓ AMLT job '{job_name}' paused successfully.[/]")
            if result.stdout.strip():
                print(result.stdout.strip())
        else:
            print(
                f"[red]✗ Failed to pause AMLT job '{job_name}' (exit code: {result.returncode})[/]"
            )
            if result.stderr.strip():
                print(f"  Error: {result.stderr.strip()}")

    async def resume_job(self, job_name: str) -> None:
        """Resume an AMLT job."""
        print(f"[cyan]Resuming AMLT job '{job_name}'...[/]")
        result = await self._run_amlt_command("resume", job_name)

        if result.returncode == 0:
            print(f"[green]✓ AMLT job '{job_name}' resumed successfully.[/]")
            if result.stdout.strip():
                print(result.stdout.strip())
        else:
            print(
                f"[red]✗ Failed to resume AMLT job '{job_name}' (exit code: {result.returncode})[/]"
            )
            if result.stderr.strip():
                print(f"  Error: {result.stderr.strip()}")

    async def rename_job(self, job_name: str, new_name: str) -> None:
        """Rename an AMLT job."""
        print(f"[cyan]Renaming AMLT job '{job_name}' to '{new_name}'...[/]")

        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "amlt",
            "rename",
            job_name,
            new_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode == 0:
            print(f"[green]✓ AMLT job renamed from '{job_name}' to '{new_name}'.[/]")
            if stdout:
                print(stdout.decode().strip())
        else:
            print(f"[red]✗ Failed to rename AMLT job (exit code: {proc.returncode})[/]")
            if stderr:
                print(f"  Error: {stderr.decode().strip()}")

    async def _run_amlt_command(
        self, command: str, job_name: str
    ) -> subprocess.CompletedProcess:
        """Run an amlt command."""
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "amlt",
            command,
            job_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return subprocess.CompletedProcess(
            args=[],
            returncode=proc.returncode or 0,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
        )

    async def _run_tmux_command(self, args: str) -> subprocess.CompletedProcess:
        """Run a tmux command."""
        proc = await asyncio.create_subprocess_shell(
            f"tmux {args}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return subprocess.CompletedProcess(
            args=[],
            returncode=proc.returncode or 0,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else "",
        )
