#!/usr/bin/env python3
"""
Script to rename subdirectories by replacing "_V1 " with "_V10 " and "_V1R" with "_V10R" in directory names.

Usage: python rename_directories.py /path/to/directory
"""

import os
import sys


def rename_v1_to_v10_directories(root_directory):
    """
    Given a directory name, look at all subdirectories and replace all instances
    of "_V1 " with "_V10 " and "_V1R" with "_V10R" in the directory names.

    Args:
        root_directory (str): The root directory to search for subdirectories

    Returns:
        tuple: (renamed_count, error_count) - number of successfully renamed directories and errors
    """
    if not os.path.exists(root_directory):
        print(f"Error: Directory '{root_directory}' does not exist.")
        return 0, 1

    if not os.path.isdir(root_directory):
        print(f"Error: '{root_directory}' is not a directory.")
        return 0, 1

    renamed_count = 0
    error_count = 0

    try:
        # Get all items in the root directory
        items = os.listdir(root_directory)

        for item in items:
            item_path = os.path.join(root_directory, item)

            # Only process directories
            if os.path.isdir(item_path):
                # Check if the directory name contains "_V1 " or "_V1R"
                new_name = item
                renamed = False

                if "_V1 " in item:
                    new_name = new_name.replace("_V1 ", "_V10 ")
                    renamed = True

                if "_V1R" in item:
                    new_name = new_name.replace("_V1R", "_V10R")
                    renamed = True

                if renamed:
                    new_path = os.path.join(root_directory, new_name)

                    try:
                        # Rename the directory
                        os.rename(item_path, new_path)
                        print(f"Renamed: '{item}' -> '{new_name}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"Error renaming '{item}': {e}")
                        error_count += 1
                else:
                    print(f"Skipped: '{item}' (no '_V1 ' or '_V1R' found)")
            else:
                print(f"Skipped: '{item}' (not a directory)")

    except OSError as e:
        print(f"Error reading directory '{root_directory}': {e}")
        error_count += 1

    return renamed_count, error_count


def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python rename_directories.py <directory_path>")
        print("Example: python rename_directories.py /path/to/your/directory")
        sys.exit(1)

    directory_path = sys.argv[1]

    print(f"Processing directory: {directory_path}")
    print("-" * 50)

    renamed_count, error_count = rename_v1_to_v10_directories(directory_path)

    print("-" * 50)
    print(f"Summary:")
    print(f"  Directories renamed: {renamed_count}")
    print(f"  Errors encountered: {error_count}")

    if error_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
