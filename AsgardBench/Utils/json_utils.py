"""
Utilities for reading JSON files.
"""

import json
from typing import Any


class FileReadError(Exception):
    """Raised when a file cannot be read or parsed correctly."""

    pass


def read_json_file(file_path: str, max_retries: int = 3) -> Any:
    """Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file.
        max_retries: Ignored (kept for API compatibility).

    Returns:
        Parsed JSON data.

    Raises:
        FileReadError: If the file cannot be read or has invalid JSON.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            raise FileReadError(f"File is empty: {file_path}")

        return json.loads(content)

    except json.JSONDecodeError as e:
        raise FileReadError(f"Invalid JSON in {file_path}: {e}") from e
    except OSError as e:
        raise FileReadError(f"Cannot read file {file_path}: {e}") from e
