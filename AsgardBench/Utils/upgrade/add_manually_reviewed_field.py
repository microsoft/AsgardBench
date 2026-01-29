#!/usr/bin/env python3
"""
Script to add the 'manually_reviewed' field to all test result entries in test_results.json files.

This script:
1. Recursively searches for test_results.json files in Test/Test/
2. Creates a backup of each file (test_results.json.bak)
3. Adds 'manually_reviewed': false to each test result that doesn't have the field

Usage:
    python add_manually_reviewed_field.py [--apply]

By default, runs in preview mode. Use --apply to make actual changes.
"""

import json
import os
import shutil
import sys

from AsgardBench.constants import TEST_DIR


def find_test_results_files(base_dir: str) -> list[str]:
    """Recursively find all test_results.json files."""
    results = []
    for root, dirs, files in os.walk(base_dir):
        if "test_results.json" in files:
            results.append(os.path.join(root, "test_results.json"))
    return results


def process_file(file_path: str, dry_run: bool = True) -> dict:
    """
    Process a single test_results.json file.

    Returns a dict with:
        - 'path': file path
        - 'total_results': number of test results in file
        - 'missing_field': number of results missing the field
        - 'error': error message if any
    """
    result = {"path": file_path, "total_results": 0, "missing_field": 0, "error": None}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        result["error"] = f"JSON parse error: {e}"
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

    # Support both "results" and "test_results" keys
    results_key = "test_results" if "test_results" in data else "results"
    test_results = data.get(results_key, [])

    result["total_results"] = len(test_results)

    # Count and update entries missing the field
    modified = False
    for entry in test_results:
        if "manually_reviewed" not in entry:
            result["missing_field"] += 1
            if not dry_run:
                entry["manually_reviewed"] = False
                modified = True

    # Save changes if not dry run and there were modifications
    if not dry_run and modified:
        # Create backup
        backup_path = file_path + ".bak"
        shutil.copy2(file_path, backup_path)

        # Write updated file
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return result


def main():
    apply_changes = "--apply" in sys.argv
    base_dir = str(TEST_DIR)

    if not os.path.isdir(base_dir):
        print(f"Error: Directory not found: {base_dir}")
        sys.exit(1)

    print(f"Searching for test_results.json files in: {base_dir}")
    print()

    files = find_test_results_files(base_dir)

    if not files:
        print("No test_results.json files found.")
        return

    print(f"Found {len(files)} test_results.json file(s)")
    print()

    if apply_changes:
        print("=== APPLYING CHANGES ===")
    else:
        print("=== PREVIEW MODE (use --apply to make changes) ===")
    print()

    total_files = 0
    total_results = 0
    total_missing = 0
    errors = []

    for file_path in sorted(files):
        result = process_file(file_path, dry_run=not apply_changes)

        if result["error"]:
            errors.append((file_path, result["error"]))
            continue

        total_files += 1
        total_results += result["total_results"]
        total_missing += result["missing_field"]

        # Only show files that need updates
        if result["missing_field"] > 0:
            rel_path = os.path.relpath(file_path, base_dir)
            action = "Updated" if apply_changes else "Would update"
            print(f"  {action}: {rel_path}")
            print(
                f"           {result['missing_field']}/{result['total_results']} entries need 'manually_reviewed' field"
            )

    print()
    print("=== SUMMARY ===")
    print(f"  Files processed: {total_files}")
    print(f"  Total test results: {total_results}")
    print(f"  Entries missing 'manually_reviewed': {total_missing}")

    if errors:
        print()
        print("=== ERRORS ===")
        for path, error in errors:
            print(f"  {path}: {error}")

    if not apply_changes and total_missing > 0:
        print()
        print("Run with --apply to make these changes.")


if __name__ == "__main__":
    main()
