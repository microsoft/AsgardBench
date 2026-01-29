#!/usr/bin/env python3
"""
Script to update task descriptions in plan.json files.
"""

import json
import os
from pathlib import Path

from Magmathor.constants import DATASET_DIR


def update_task_descriptions(base_dir: str, dry_run: bool = False) -> tuple[int, int]:
    """
    Go through base_dir and subdirectories, find plan.json files,
    and replace old task description with new one.

    Args:
        base_dir: The base directory to search
        dry_run: If True, only report what would be changed without making changes

    Returns:
        Tuple of (files_found, files_modified)
    """
    old_text = "Fry an egg and serve it on a plate"
    new_text = "Cook an egg in a pan and serve it on a plate."

    files_found = 0
    files_modified = 0

    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename == "plan.json":
                files_found += 1
                filepath = os.path.join(root, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()

                    if old_text in content:
                        new_content = content.replace(old_text, new_text)
                        files_modified += 1

                        if dry_run:
                            print(f"Would update: {filepath}")
                        else:
                            with open(filepath, "w", encoding="utf-8") as f:
                                f.write(new_content)
                            print(f"Updated: {filepath}")

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return files_found, files_modified


if __name__ == "__main__":
    base_directory = str(Path(DATASET_DIR) / "magt_benchmark")

    print(f"Searching in: {base_directory}")
    print()

    # First do a dry run to see what would be changed
    print("=== DRY RUN ===")
    found, modified = update_task_descriptions(base_directory, dry_run=True)
    print(f"\nFound {found} plan.json files, {modified} would be modified")
    print()

    if modified > 0:
        response = input("Proceed with actual update? (y/n): ")
        if response.lower() == "y":
            print("\n=== APPLYING CHANGES ===")
            found, modified = update_task_descriptions(base_directory, dry_run=False)
            print(f"\nDone! Modified {modified} files.")
        else:
            print("Aborted.")
    else:
        print("No files need to be modified.")
