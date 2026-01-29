"""
Utility functions for handling file storage in both local and Azure ML environments.
"""

import os
from pathlib import Path

from Magmathor.constants import MOUNTED_STORAGE_PATH, USING_MOUNTED_STORAGE


def get_persistent_path(relative_path: str) -> str:
    """
    Get a path that will persist in Azure ML or work locally.

    Args:
        relative_path: The relative path from the storage root

    Returns:
        Full path that works in both local and Azure ML environments
    """
    if USING_MOUNTED_STORAGE:
        # Running in Azure ML - use mounted storage
        return str((MOUNTED_STORAGE_PATH / relative_path).resolve())
    else:
        # Running locally - use local path
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
    Save JSON data to persistent storage.

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
    Save CSV data to persistent storage.

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


def get_azure_ml_info() -> dict:
    """
    Get information about the Azure ML environment.

    Returns:
        Dictionary with environment information
    """
    info = {
        "is_azure_ml": USING_MOUNTED_STORAGE,
        "working_dir": os.getcwd(),
        "mounted_storage": (
            str(MOUNTED_STORAGE_PATH) if USING_MOUNTED_STORAGE else None
        ),
        "job_id": os.environ.get("AZUREML_RUN_ID", "unknown"),
        "experiment_name": os.environ.get("AZUREML_EXPERIMENT_NAME", "unknown"),
    }
    return info
