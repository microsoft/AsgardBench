#!/usr/bin/env python3
"""
Extract unique [MODEL ERROR] messages from plan.json files.

This script scans plan.json files under the Test directory whose paths match
a given regex pattern, extracts all [MODEL ERROR] messages from the step logs,
and outputs the unique errors to a text file.

Usage:
    python scripts/extract_model_errors.py "google__" --output errors_google.txt
    python scripts/extract_model_errors.py "qwen__" --output errors_qwen.txt
    python scripts/extract_model_errors.py ".*" --output all_errors.txt
"""

import argparse
import json
import os
import re
from pathlib import Path


def find_plan_json_files(test_dir: Path, pattern: str) -> list[Path]:
    """
    Find all plan.json files under test_dir whose path matches the given regex pattern.

    Args:
        test_dir: The Test directory to search in
        pattern: Regex pattern to match against the full path

    Returns:
        List of matching plan.json file paths
    """
    regex = re.compile(pattern)
    plan_files = []

    # Walk through the Test directory structure
    # Structure: Test/<benchmark>/<config>/Plans/<task>/plan.json
    for benchmark_dir in test_dir.iterdir():
        if not benchmark_dir.is_dir():
            continue

        for config_dir in benchmark_dir.iterdir():
            if not config_dir.is_dir():
                continue

            # Check if the config path matches the pattern
            config_path_str = str(config_dir)
            if not regex.search(config_path_str):
                continue

            plans_dir = config_dir / "Plans"
            if not plans_dir.exists():
                continue

            for task_dir in plans_dir.iterdir():
                if not task_dir.is_dir():
                    continue

                plan_json = task_dir / "plan.json"
                if plan_json.exists():
                    plan_files.append(plan_json)

    return plan_files


def extract_model_errors(plan_json_path: Path) -> list[str]:
    """
    Extract all [MODEL ERROR] messages from a plan.json file.

    Args:
        plan_json_path: Path to the plan.json file

    Returns:
        List of error messages found
    """
    errors = []

    try:
        with open(plan_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not read {plan_json_path}: {e}")
        return errors

    # Check steps for [MODEL ERROR] in the log field
    for step in data.get("steps", []):
        log = step.get("log", "")
        if isinstance(log, str) and "[MODEL ERROR]" in log:
            # Extract the error message - it typically spans multiple lines
            # Look for the pattern [MODEL ERROR] followed by the error text
            for match in re.finditer(
                r"\[MODEL ERROR\]\s*(.+?)(?=\n-|\n\[|$)", log, re.DOTALL
            ):
                error_msg = match.group(1).strip()
                if error_msg:
                    errors.append(error_msg)

    # Also check failed_steps if present
    for step in data.get("failed_steps", []):
        log = step.get("log", "")
        if isinstance(log, str) and "[MODEL ERROR]" in log:
            for match in re.finditer(
                r"\[MODEL ERROR\]\s*(.+?)(?=\n-|\n\[|$)", log, re.DOTALL
            ):
                error_msg = match.group(1).strip()
                if error_msg:
                    errors.append(error_msg)

    return errors


def normalize_error(error: str) -> str:
    """
    Normalize an error message for deduplication.

    This strips variable parts like timestamps, request IDs, etc.
    to group similar errors together.
    """
    # Remove common variable parts
    # Remove request IDs (hex strings)
    normalized = re.sub(r"[a-f0-9]{8,}", "<ID>", error)
    # Remove timestamps
    normalized = re.sub(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}", "<TIMESTAMP>", normalized
    )
    # Remove line numbers in stack traces
    normalized = re.sub(r", line \d+", ", line <N>", normalized)
    # Remove specific file paths but keep the filename
    normalized = re.sub(r'File "[^"]+/([^"]+)"', r'File "<PATH>/\1"', normalized)

    return normalized


def main():
    parser = argparse.ArgumentParser(
        description="Extract unique [MODEL ERROR] messages from plan.json files"
    )
    parser.add_argument(
        "pattern",
        help="Regex pattern to match against plan.json file paths (e.g., 'google__' or 'qwen__')",
    )
    parser.add_argument(
        "--test-dir", default="Test", help="Path to the Test directory (default: Test)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="model_errors.txt",
        help="Output file path (default: model_errors.txt)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize errors to group similar ones (removes IDs, timestamps, etc.)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print progress information"
    )

    args = parser.parse_args()

    # Resolve test directory path
    test_dir = Path(args.test_dir)
    if not test_dir.is_absolute():
        # Try relative to current directory or script location
        if not test_dir.exists():
            script_dir = Path(__file__).parent.parent
            test_dir = script_dir / args.test_dir

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return 1

    if args.verbose:
        print(f"Searching in: {test_dir}")
        print(f"Pattern: {args.pattern}")

    # Find matching plan.json files
    plan_files = find_plan_json_files(test_dir, args.pattern)

    if args.verbose:
        print(f"Found {len(plan_files)} matching plan.json files")

    if not plan_files:
        print(f"No plan.json files found matching pattern: {args.pattern}")
        return 1

    # Extract errors from all files
    all_errors: list[str] = []
    error_sources: dict[str, list[str]] = {}  # error -> list of source files

    for i, plan_file in enumerate(plan_files):
        if args.verbose and (i + 1) % 100 == 0:
            print(f"Processing file {i + 1}/{len(plan_files)}...")

        errors = extract_model_errors(plan_file)
        for error in errors:
            all_errors.append(error)

            # Track source for each error
            key = normalize_error(error) if args.normalize else error
            if key not in error_sources:
                error_sources[key] = []
            error_sources[key].append(str(plan_file))

    # Get unique errors
    if args.normalize:
        # Group by normalized form but keep one original example
        unique_errors: dict[str, str] = {}
        for error in all_errors:
            normalized = normalize_error(error)
            if normalized not in unique_errors:
                unique_errors[normalized] = error
        unique_error_list = list(unique_errors.values())
    else:
        unique_error_list = list(set(all_errors))

    # Sort for consistent output
    unique_error_list.sort()

    # Write output
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Unique [MODEL ERROR] messages\n")
        f.write(f"# Pattern: {args.pattern}\n")
        f.write(f"# Total errors found: {len(all_errors)}\n")
        f.write(f"# Unique errors: {len(unique_error_list)}\n")
        f.write(f"# Files searched: {len(plan_files)}\n")
        f.write("\n" + "=" * 80 + "\n\n")

        for i, error in enumerate(unique_error_list, 1):
            f.write(f"--- Error {i} ---\n")
            f.write(error)
            f.write("\n\n")

            # Optionally include source information
            key = normalize_error(error) if args.normalize else error
            sources = error_sources.get(key, [])
            if sources:
                f.write(f"Occurred in {len(sources)} file(s)\n")
                # Show up to 3 example sources
                for src in sources[:3]:
                    f.write(f"  - {src}\n")
                if len(sources) > 3:
                    f.write(f"  ... and {len(sources) - 3} more\n")
            f.write("\n" + "-" * 40 + "\n\n")

    print(f"Total errors found: {len(all_errors)}")
    print(f"Unique errors: {len(unique_error_list)}")
    print(f"Output written to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
