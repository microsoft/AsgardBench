#!/usr/bin/env python3
"""Rename experiment folders to match their config.json settings.

This script iterates over experiment folders, computes the expected folder name
based on the EvaluationConfig, and offers to rename folders that don't match.
"""

import json
import re
from pathlib import Path

import typer

from Magmathor.constants import FeedbackType, PromptVersion
from Magmathor.Utils.config_utils import EvaluationConfig


def convert_config_enums(config_data: dict) -> dict:
    """Convert string values in config_data to their proper Enum types.

    Args:
        config_data: Raw config data with string values.

    Returns:
        Config data with enum strings converted to Enum instances.
    """
    result = config_data.copy()

    if "feedback_type" in result and isinstance(result["feedback_type"], str):
        result["feedback_type"] = FeedbackType(result["feedback_type"])

    if "prompt_version" in result and isinstance(result["prompt_version"], str):
        result["prompt_version"] = PromptVersion(result["prompt_version"])

    return result


def extract_rep_number(folder_name: str) -> int | None:
    """Extract the repetition number from a folder name.

    Args:
        folder_name: The folder name to parse (e.g., "gpt-4o--rep2_T0_Fs...")

    Returns:
        The rep number if found, None otherwise.
    """
    match = re.search(r"--rep(\d+)", folder_name)
    return int(match.group(1)) if match else None


def compute_expected_name(config_data: dict) -> str:
    """Compute the expected folder name from config data.

    Args:
        config_data: The parsed config.json data.

    Returns:
        The expected folder name in format: {model_path}--rep{N}_{config_suffix}
    """
    # Extract EvaluationConfig fields from config_data
    eval_config_fields = {
        "text_only",
        "feedback_type",
        "hand_transparency",
        "include_common_sense",
        "prompt_version",
        "include_previous_image",
        "use_memory",
        "temperature",
        "max_completion_tokens",
    }

    eval_config_data = {k: v for k, v in config_data.items() if k in eval_config_fields}
    eval_config_data = convert_config_enums(eval_config_data)
    eval_config = EvaluationConfig.from_dict(eval_config_data)
    config_suffix = eval_config.get_output_suffix()

    model_path = config_data["model_path"]
    model_name = config_data["model_name"]

    # Extract rep number from model_name
    rep_number = extract_rep_number(model_name)
    if rep_number is None:
        rep_number = 1  # Default to rep1 if not found

    return f"{model_path}--{config_suffix}--rep{rep_number}"


def find_available_rep_number(
    base_path: Path,
    model_path: str,
    config_suffix: str,
    current_folder: str,
    planned_names: set[str],
) -> int:
    """Find the next available rep number for a given config.

    Args:
        base_path: The parent directory containing all experiments.
        model_path: The model path prefix.
        config_suffix: The config suffix for the folder name.
        current_folder: The current folder name (excluded from conflict check).
        planned_names: Set of names that will be taken by planned renames.

    Returns:
        The next available rep number.
    """
    rep = 1
    while True:
        candidate_name = f"{model_path}--{config_suffix}--rep{rep}"
        candidate_path = base_path / candidate_name
        is_current = candidate_name == current_folder
        exists_on_disk = candidate_path.exists()
        is_planned = candidate_name in planned_names
        if is_current or (not exists_on_disk and not is_planned):
            return rep
        rep += 1


def process_experiment(exp_dir: Path, dry_run: bool, planned_names: set[str]) -> None:
    """Process a single experiment directory.

    Args:
        exp_dir: Path to the experiment directory.
        dry_run: If True, only show what would be done without making changes.
        planned_names: Set of names that will be taken by planned/executed renames.
    """
    config_path = exp_dir / "config.json"

    if not config_path.exists():
        typer.echo(f"⚠️  Skipping {exp_dir.name}: no config.json found")
        return

    with open(config_path) as f:
        config_data = json.load(f)

    # Compute what the name should be
    expected_name = compute_expected_name(config_data)
    current_name = exp_dir.name

    if expected_name == current_name:
        typer.echo(f"✓  {current_name}: name is correct")
        return

    # Check if expected name already exists or is planned for another rename
    expected_path = exp_dir.parent / expected_name
    name_taken = (expected_path.exists() and expected_path != exp_dir) or (
        expected_name in planned_names
    )

    if name_taken:
        # Need to find next available rep number
        eval_config_fields = {
            "text_only",
            "feedback_type",
            "hand_transparency",
            "include_common_sense",
            "prompt_version",
            "include_previous_image",
            "use_memory",
            "temperature",
            "max_completion_tokens",
        }
        eval_config_data = {
            k: v for k, v in config_data.items() if k in eval_config_fields
        }
        eval_config_data = convert_config_enums(eval_config_data)
        eval_config = EvaluationConfig.from_dict(eval_config_data)
        config_suffix = eval_config.get_output_suffix()
        model_path = config_data["model_path"]

        available_rep = find_available_rep_number(
            exp_dir.parent, model_path, config_suffix, current_name, planned_names
        )
        expected_name = f"{model_path}--{config_suffix}--rep{available_rep}"
        typer.echo(f"⚠️  Name conflict detected, using rep{available_rep} instead")

    typer.echo(f"\n📁 Current:  {current_name}")
    typer.echo(f"   Expected: {expected_name}")

    if dry_run:
        typer.echo("   [DRY RUN] Would rename")
        planned_names.add(expected_name)
        return

    if typer.confirm("   Rename this folder?", default=True):
        new_path = exp_dir.parent / expected_name
        exp_dir.rename(new_path)
        planned_names.add(expected_name)
        typer.echo(f"   ✓ Renamed to {expected_name}")
    else:
        typer.echo("   Skipped")


def main(
    experiment_dir: Path = typer.Argument(
        Path("Test/magt_benchmark"),
        help="Directory containing experiment folders to process",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be done without making changes"
    ),
) -> None:
    """Rename experiment folders to match their config.json settings.

    Iterates over all subdirectories in the experiment directory, computes
    the expected folder name based on EvaluationConfig, and offers to rename
    folders that don't match.
    """
    if not experiment_dir.exists():
        typer.echo(f"Error: Directory {experiment_dir} does not exist", err=True)
        raise typer.Exit(1)

    if not experiment_dir.is_dir():
        typer.echo(f"Error: {experiment_dir} is not a directory", err=True)
        raise typer.Exit(1)

    # Sort directories for consistent processing order
    exp_dirs = sorted(d for d in experiment_dir.iterdir() if d.is_dir())

    if not exp_dirs:
        typer.echo(f"No experiment directories found in {experiment_dir}")
        raise typer.Exit(0)

    typer.echo(f"Processing {len(exp_dirs)} experiment(s) in {experiment_dir}\n")

    if dry_run:
        typer.echo("🔍 DRY RUN MODE - no changes will be made\n")

    # Track names that have been claimed by renames (for conflict detection)
    planned_names: set[str] = set()

    for exp_dir in exp_dirs:
        process_experiment(exp_dir, dry_run, planned_names)


if __name__ == "__main__":
    typer.run(main)
