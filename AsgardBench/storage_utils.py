"""
Utility functions for handling file storage.
"""

import os
from pathlib import Path


def get_persistent_path(relative_path: str) -> str:
    """
    Get a path for persistent storage.

    Args:
        relative_path: The relative path from the storage root

    Returns:
        Full path
    """
    return relative_path


def ensure_dir_exists(file_path: str) -> None:
    """
    Ensure the directory for the given file path exists.

    Args:
        file_path: Full path to a file
    """
    dir_path = Path(file_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)


def save_json_results(data: dict, relative_path: str) -> str:
    """
    Save JSON data to storage.

    Args:
        data: Dictionary to save as JSON
        relative_path: Relative path from storage root

    Returns:
        Full path where file was saved
    """
    import json

    full_path = get_persistent_path(relative_path)
    ensure_dir_exists(full_path)

    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved results to: {full_path}")
    return full_path


def save_csv_results(csv_content: str, relative_path: str) -> str:
    """
    Save CSV data to storage.

    Args:
        csv_content: CSV content as string
        relative_path: Relative path from storage root

    Returns:
        Full path where file was saved
    """
    full_path = get_persistent_path(relative_path)
    ensure_dir_exists(full_path)

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(csv_content)

    print(f"Saved CSV to: {full_path}")
    return full_path
