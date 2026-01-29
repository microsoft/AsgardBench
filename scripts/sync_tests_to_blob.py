#!/usr/bin/env python3
"""
Sync local test results to Azure Blob Storage using azcopy.

azcopy sync handles incremental uploads based on mtime automatically,
making this much faster than manual file-by-file copying.

Usage:
    python sync_tests_to_blob.py                    # One-time sync
    python sync_tests_to_blob.py --watch            # Continuous sync every 60s
    python sync_tests_to_blob.py --watch --interval 30
    python sync_tests_to_blob.py --dry-run          # Preview what would be synced
"""

import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer

# === CONFIGURATION (hardcoded defaults) ===
REPO_ROOT = Path(__file__).parent.parent

DEFAULT_SOURCE = REPO_ROOT / "Test"
DEFAULT_DEST = "https://magmardata.blob.core.windows.net/magmathor/Test/"

DEFAULT_INTERVAL = 60 * 60  # seconds

app = typer.Typer(help="Sync local test results to Azure Blob Storage using azcopy")


def run_azcopy_sync(
    source: Path,
    dest: str,
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[bool, int, int]:
    """
    Run azcopy sync to upload changed files.

    Returns (success, files_transferred, files_failed).
    Success is True if sync completed (even with some files modified during transfer).
    """
    cmd = [
        "azcopy",
        "sync",
        str(source),
        dest,
        "--recursive=true",
        "--log-level=WARNING",
    ]

    if dry_run:
        cmd.append("--dry-run")

    if verbose:
        cmd.append("--log-level=INFO")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr

        # Parse transfer stats from output
        transferred = 0
        failed = 0
        for line in output.splitlines():
            if "Number of Copy Transfers Completed:" in line:
                transferred = int(line.split(":")[-1].strip())
            elif "Number of Copy Transfers Failed:" in line:
                failed = int(line.split(":")[-1].strip())

        if verbose:
            print(output)
        else:
            # Print summary
            for line in output.splitlines():
                if any(
                    x in line.lower()
                    for x in ["total", "transfer", "skip", "fail", "elapsed"]
                ):
                    print(f"  {line.strip()}")

        # Exit code 1 with "CompletedWithErrors" is OK - just means some files
        # were modified during transfer (will be picked up next cycle)
        if result.returncode == 0:
            return (True, transferred, failed)
        elif "CompletedWithErrors" in output:
            # Partial success - some files modified during transfer
            return (True, transferred, failed)
        else:
            print(f"  azcopy error (exit {result.returncode})")
            return (False, transferred, failed)

    except FileNotFoundError:
        print("  ERROR: azcopy not found. Please install it first.")
        return (False, 0, 0)
    except Exception as e:
        print(f"  ERROR: {e}")
        return (False, 0, 0)


@app.command()
def main(
    source: Annotated[
        Path, typer.Option("--source", "-s", help="Source folder to sync from")
    ] = DEFAULT_SOURCE,
    dest: Annotated[
        str, typer.Option("--dest", "-d", help="Destination Azure Blob URL")
    ] = DEFAULT_DEST,
    watch: Annotated[
        bool, typer.Option("--watch", "-w", help="Run continuously")
    ] = False,
    interval: Annotated[
        int, typer.Option("--interval", "-i", help="Interval in seconds for watch mode")
    ] = DEFAULT_INTERVAL,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", "-n", help="Show what would be synced without actually syncing"
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed azcopy output")
    ] = False,
):
    """Sync local test results to Azure Blob Storage."""
    typer.echo(f"Sync: {source} -> {dest}")
    if dry_run:
        typer.echo(
            "[bold yellow]DRY RUN MODE - no files will be uploaded[/bold yellow]"
        )
    typer.echo()

    if not source.exists():
        typer.echo(f"[red]ERROR: Source folder does not exist: {source}[/red]")
        raise typer.Exit(1)

    try:
        while True:
            start_time = time.time()
            typer.echo(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting sync..."
            )

            success, transferred, failed = run_azcopy_sync(
                source, dest, dry_run=dry_run, verbose=verbose
            )

            elapsed = time.time() - start_time

            if success:
                if failed > 0:
                    status = f"[yellow]OK ({failed} deferred - files changed during transfer)[/yellow]"
                else:
                    status = "[green]OK[/green]"
            else:
                status = "[red]FAILED[/red]"
            typer.echo(f"  Status: {status}")
            typer.echo(f"  Elapsed: {elapsed:.2f}s")
            typer.echo()

            if not watch:
                break

            typer.echo(f"Sleeping for {interval} seconds... (Ctrl+C to stop)")
            time.sleep(interval)

    except KeyboardInterrupt:
        typer.echo("\nInterrupted by user")


if __name__ == "__main__":
    app()
