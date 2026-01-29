#!/usr/bin/env python3
"""
Code to update old test_results.json files with candidate_poses_errors counts.
Only needed to be run once on old files, to update them to new format
Script to add 'candidate_poses_errors' count to test_results.json files.

This counts the number of step_errors that contain pose errors
in their error_msg field.

Usage:
    python update_test_results_candidate_poses.py <input_file>

This will:
1. Create a backup at _test_results.json
2. Update the original test_results.json with the new fields
"""

import json
import shutil
import sys
from pathlib import Path

import Magmathor.constants as c


def count_candidate_poses_errors(test_result: dict) -> int:
    """Count errors containing pose errors in a single test result."""
    count = 0
    step_errors = test_result.get("step_errors") or []
    for error in step_errors:
        error_msg = error.get("error_msg", "")
        if c.POSES_ERROR in error_msg:
            count += 1
    return count


def update_test_results_file(input_path: str) -> dict:
    """
    Read a test_results.json file, add candidate_poses_errors counts,
    create a backup, and update the original file.

    Returns summary stats about the update.
    """
    input_path = Path(input_path)
    backup_path = input_path.parent / "_test_results.json"

    # Read original file
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if already processed (skip if field already exists at top level)
    if "total_candidate_poses_errors" in data:
        return {
            "input_path": str(input_path),
            "skipped": True,
            "reason": "Already has total_candidate_poses_errors field",
        }

    total_candidate_poses_errors = 0
    results_with_errors = 0

    # Process each test result
    test_results = data.get("test_results", [])
    for test_result in test_results:
        count = count_candidate_poses_errors(test_result)
        test_result["candidate_poses_errors"] = count
        total_candidate_poses_errors += count
        if count > 0:
            results_with_errors += 1

    # Add summary count at the top level too
    data["total_candidate_poses_errors"] = total_candidate_poses_errors

    # Create backup
    shutil.copy2(input_path, backup_path)

    # Save updated file to original location
    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {
        "input_path": str(input_path),
        "backup_path": str(backup_path),
        "skipped": False,
        "num_test_results": len(test_results),
        "results_with_candidate_poses_errors": results_with_errors,
        "total_candidate_poses_errors": total_candidate_poses_errors,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python update_test_results_candidate_poses.py <input_file>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Processing: {input_path}")
    stats = update_test_results_file(input_path)

    if stats.get("skipped"):
        print(f"  SKIPPED: {stats['reason']}")
    else:
        print(f"  Backup saved to: {stats['backup_path']}")
        print(f"  Test results: {stats['num_test_results']}")
        print(
            f"  Results with candidate poses errors: {stats['results_with_candidate_poses_errors']}"
        )
        print(
            f"  Total candidate poses errors: {stats['total_candidate_poses_errors']}"
        )
        print(f"  Updated: {stats['input_path']}")


if __name__ == "__main__":
    main()
