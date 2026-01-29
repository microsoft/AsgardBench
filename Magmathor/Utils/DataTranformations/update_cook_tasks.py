#!/usr/bin/env python3
"""
Script to update task descriptions in plan.json files for directories starting with 'cook_'.

Usage: python update_cook_tasks.py /path/to/directory
"""

import json
import os
import sys


def update_cook_task_descriptions(sub_path):
    """
    Updates task descriptions in plan.json files for directories starting with 'cook_'.

    Args:
        base_path (str): The base directory path to search in

    Returns:
        dict: Summary of the operation with counts and any errors
    """
    # Dictionary of task description replacements
    replacements = {
        "Cook Bread": "Make a slice of toast and serve it on a plate",
        "Cook Egg": "Fry an egg and serve it on a plate",
        "Cook Potato": "Microwave a potato and serve it in a bowl",
    }

    total_dirs = 0
    processed_files = 0
    updated_files = 0
    errors = []

    base_path = os.path.join(os.getcwd(), sub_path)
    print(f"Processing {base_path}...")

    try:
        # Get all subdirectories
        subdirs = [
            d
            for d in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, d)) and d.startswith("cook_")
        ]

        for subdir in subdirs:
            total_dirs += 1
            subdir_path = os.path.join(base_path, subdir)
            plan_json_path = os.path.join(subdir_path, "plan.json")

            # Check if plan.json exists
            if not os.path.exists(plan_json_path):
                errors.append(f"plan.json not found in {subdir}")
                continue

            try:
                # Read the plan.json file
                with open(plan_json_path, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)

                processed_files += 1
                file_updated = False

                # Update task descriptions in steps
                if "steps" in plan_data:
                    for step in plan_data["steps"]:
                        if "task_description" in step:
                            original_desc = step["task_description"]
                            for old_text, new_text in replacements.items():
                                if old_text in original_desc:
                                    step["task_description"] = original_desc.replace(
                                        old_text, new_text
                                    )
                                    file_updated = True
                                    print(
                                        f"Updated step task description in {subdir}: '{original_desc}' -> '{step['task_description']}'"
                                    )
                                    break

                # Update task description at the plan level
                if "task_description" in plan_data:
                    original_desc = plan_data["task_description"]
                    for old_text, new_text in replacements.items():
                        if old_text in original_desc:
                            plan_data["task_description"] = original_desc.replace(
                                old_text, new_text
                            )
                            file_updated = True
                            print(
                                f"Updated plan task description in {subdir}: '{original_desc}' -> '{plan_data['task_description']}'"
                            )
                            break

                # Save the updated file if changes were made
                if file_updated:
                    with open(plan_json_path, "w", encoding="utf-8") as f:
                        json.dump(plan_data, f, indent=2, ensure_ascii=False)
                    updated_files += 1

            except json.JSONDecodeError as e:
                errors.append(f"JSON decode error in {subdir}/plan.json: {str(e)}")
            except Exception as e:
                errors.append(f"Error processing {subdir}/plan.json: {str(e)}")

    except FileNotFoundError:
        errors.append(f"Base directory not found: {base_path}")
    except Exception as e:
        errors.append(f"Error accessing base directory: {str(e)}")

    return {
        "total_cook_dirs": total_dirs,
        "processed_files": processed_files,
        "updated_files": updated_files,
        "errors": errors,
    }


def main():
    """Main function for command line usage."""
    if len(sys.argv) != 2:
        print("Usage: python update_cook_tasks.py <directory_path>")
        print("Example: python update_cook_tasks.py /path/to/your/directory")
        sys.exit(1)

    directory_path = sys.argv[1]

    print(f"Processing cook task descriptions in: {directory_path}")
    print("-" * 50)

    result = update_cook_task_descriptions(directory_path)

    print("-" * 50)
    print(f"Summary:")
    print(f"  Total cook_ directories found: {result['total_cook_dirs']}")
    print(f"  Plan files processed: {result['processed_files']}")
    print(f"  Files updated: {result['updated_files']}")

    if result["errors"]:
        print(f"  Errors encountered: {len(result['errors'])}")
        for error in result["errors"]:
            print(f"    - {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
