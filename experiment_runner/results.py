"""Results display and reporting."""

from __future__ import annotations

import asyncio
import csv
import re
from pathlib import Path

from rich.console import Console
from rich.table import Table

from experiment_runner.models import TestResult
from experiment_runner.utils import compute_mean_stddev, format_mean_stddev

console = Console()


async def recompute_results(test_output_dir: str = "Test") -> None:
    """Recompute results.csv by running results_printer.py."""
    print("[RESULTS] Recomputing results.csv...")

    try:
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "run",
            "python",
            "Magmathor/Model/results_printer.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if stdout:
            print(stdout.decode())

        if proc.returncode == 0:
            print("[RESULTS] results.csv recomputed successfully.")
            await asyncio.sleep(1.5)
            display_results_summary(test_output_dir)
        else:
            print(f"[RESULTS] Command failed with exit code {proc.returncode}")
            if stderr:
                print(f"[RESULTS] Errors: {stderr.decode()}")

    except Exception as e:
        print(f"[RESULTS] Error running command: {e}")


def display_results_summary(
    test_output_dir: str = "Test",
    test_suites: list[str] | None = None,
) -> None:
    """Display summary table of results from results.csv."""
    results_path = Path(test_output_dir) / "results.csv"

    if not results_path.exists():
        print("[RESULTS] results.csv not found.")
        return

    try:
        with open(results_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            print("[RESULTS] results.csv is empty or has no data rows.")
            return

        # Get required columns
        required_columns = [
            "model_name",
            "test_set_name",
            "test_completed",
            "success_percentage",
        ]
        for col in required_columns:
            if col not in rows[0]:
                print(f"[RESULTS] Required column '{col}' not found in results.csv.")
                return

        # Determine aggregated test set name
        if test_suites:
            aggregated_test_set = " / ".join(test_suites)
        else:
            aggregated_test_set = "magt_benchmark"  # Default

        # Build display table
        table = Table(title="AGGREGATED RESULTS SUMMARY", title_style="cyan bold")
        table.add_column("Model", no_wrap=True)
        table.add_column("Success %", justify="center")
        table.add_column("Goals %", justify="center")
        table.add_column("Undoable %", justify="center")
        table.add_column("Unparsable %", justify="center")

        markdown_rows: list[tuple[str, str, str, str, str]] = []
        raw_data_by_model: dict[str, list[tuple[float, float, float, float]]] = {}

        for row in rows:
            test_set_name = row.get("test_set_name", "")
            test_completed = row.get("test_completed", "").lower() == "true"

            if test_set_name != aggregated_test_set or not test_completed:
                continue

            model_name = row.get("model_name", "")
            success_pct = row.get("success_percentage", "N/A")
            goals_pct = row.get("goals_reached_percent", "N/A")
            undoable_ratio = row.get("undoable_ratio", "N/A")
            unparsable_ratio = row.get("unparsable_ratio", "N/A")

            # Parse values for aggregation
            try:
                success_val = float(success_pct) if success_pct != "N/A" else 0
                goals_val = float(goals_pct) if goals_pct != "N/A" else 0
                undoable_val = float(undoable_ratio) if undoable_ratio != "N/A" else 0
                unparsable_val = (
                    float(unparsable_ratio) if unparsable_ratio != "N/A" else 0
                )
            except ValueError:
                success_val = goals_val = undoable_val = unparsable_val = 0

            # Group by base model name (remove --repN suffix)
            base_model = re.sub(r"--rep\d+$", "", model_name)
            if base_model not in raw_data_by_model:
                raw_data_by_model[base_model] = []
            raw_data_by_model[base_model].append(
                (success_val, goals_val, undoable_val, unparsable_val)
            )

            # Format for display
            success_str = f"{success_val:.1%}" if success_pct != "N/A" else "N/A"
            goals_str = f"{goals_val:.1%}" if goals_pct != "N/A" else "N/A"
            undoable_str = f"{undoable_val:.1%}" if undoable_ratio != "N/A" else "N/A"
            unparsable_str = (
                f"{unparsable_val:.1%}" if unparsable_ratio != "N/A" else "N/A"
            )

            table.add_row(
                model_name,
                f"[green]{success_str}[/green]",
                f"[cyan]{goals_str}[/cyan]",
                f"[yellow]{undoable_str}[/yellow]",
                f"[red]{unparsable_str}[/red]",
            )

            markdown_rows.append(
                (model_name, success_str, goals_str, undoable_str, unparsable_str)
            )

        if not markdown_rows:
            print("[RESULTS] No completed aggregated results found.")
            return

        console.print()
        console.print(table)
        console.print()

        # Save markdown summary
        _save_markdown_summary(test_output_dir, markdown_rows, raw_data_by_model)

    except Exception as e:
        print(f"[RESULTS] Error reading results.csv: {e}")


def _save_markdown_summary(
    test_output_dir: str,
    markdown_rows: list[tuple[str, str, str, str, str]],
    raw_data_by_model: dict[str, list[tuple[float, float, float, float]]],
) -> None:
    """Save results summary as markdown file."""
    markdown_path = Path(test_output_dir) / "summary_table.md"

    lines = []

    # Individual runs table
    lines.append("## Individual Runs")
    lines.append("")
    lines.append("| Model | Success % | Goals % | Undoable % | Unparsable % |")
    lines.append("|-------|-----------|---------|------------|--------------|")
    for model, success, goals, undoable, unparsable in markdown_rows:
        lines.append(f"| {model} | {success} | {goals} | {undoable} | {unparsable} |")

    # Summary stats table (mean ± stdev)
    lines.append("")
    lines.append("## Summary (Mean ± StdDev)")
    lines.append("")
    lines.append("| Model | Success % | Goals % | Undoable % | Unparsable % |")
    lines.append("|-------|-----------|---------|------------|--------------|")

    for base_model in sorted(raw_data_by_model.keys()):
        data = raw_data_by_model[base_model]

        success_stats = compute_mean_stddev([d[0] for d in data])
        goals_stats = compute_mean_stddev([d[1] for d in data])
        undoable_stats = compute_mean_stddev([d[2] for d in data])
        unparsable_stats = compute_mean_stddev([d[3] for d in data])

        lines.append(
            f"| {base_model} | {format_mean_stddev(success_stats)} | {format_mean_stddev(goals_stats)} | "
            f"{format_mean_stddev(undoable_stats)} | {format_mean_stddev(unparsable_stats)} |"
        )

    markdown_path.write_text("\n".join(lines))
    print(f"[RESULTS] Markdown table saved to {markdown_path}")


def report_final_results(results: list[TestResult]) -> None:
    """Display final test run summary."""
    table = Table(title="TEST RUN SUMMARY", title_style="cyan bold")
    table.add_column("Status", justify="center")
    table.add_column("Test Name")
    table.add_column("Exit Code", justify="center")

    for result in results:
        status = "[green]✓ PASSED[/green]" if result.success else "[red]✗ FAILED[/red]"
        exit_code = (
            str(result.exit_code)
            if result.success
            else f"[red]{result.exit_code}[/red]"
        )
        table.add_row(status, result.test_name, exit_code)

    console.print()
    console.print(table)
    console.print()
