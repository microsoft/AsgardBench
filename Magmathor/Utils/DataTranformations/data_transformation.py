import json
import os
import shutil
from typing import List

from Magmathor import constants as c

"""
Files for manipulating plan.json files after creation
Handy if they contain errors that nee fixing
"""


def get_kitchen_directories() -> List[str]:
    """
    Returns a list of directory names from GENERATION_DIR/NEW_PLANS_DIR
    that start with "cook", "distribute", "coffee" or "put_away".

    Returns:
        List[str]: List of directory names (not full paths) that start with the specified prefixes
    """
    kitchen_prefixes = ["cook", "distribute", "coffee", "put_away"]
    kitchen_dirs = []

    try:
        new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"

        if not os.path.exists(new_plans_dir):
            print(f"Directory does not exist: {new_plans_dir}")
            return []

        # Get all items in the NEW_PLANS_DIR
        all_items = os.listdir(new_plans_dir)

        # Filter for directories only and those starting with kitchen prefixes
        for item in all_items:
            full_path = os.path.join(new_plans_dir, item)
            if os.path.isdir(full_path):
                # Check if directory name starts with any of the kitchen prefixes
                for prefix in kitchen_prefixes:
                    if item.lower().startswith(prefix.lower()):
                        kitchen_dirs.append(item)
                        break  # Found a match, no need to check other prefixes

        return sorted(kitchen_dirs)  # Return sorted for consistency

    except Exception as e:
        print(f"Error reading directories from {new_plans_dir}: {e}")
        return []


def get_dirty_directories() -> List[str]:
    """
    Fix error where isDirty items weren't properly set
    Read all directories under NEW_PLANS_DIR and return a list of those that have "_dirty_" in their name.

    Returns:
        List[str]: List of directory names (not full paths) that contain "_dirty_"
    """

    new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"

    if not os.path.exists(new_plans_dir):
        return []

    try:
        # Get all items in the NEW_PLANS_DIR
        all_items = os.listdir(new_plans_dir)

        # Filter for directories only and those containing "_dirty_"
        dirty_dirs = []
        for item in all_items:
            full_path = os.path.join(new_plans_dir, item)
            if os.path.isdir(full_path) and "_dirty_" in item:
                dirty_dirs.append(item)

        return sorted(dirty_dirs)  # Return sorted for consistency

    except Exception as e:
        print(f"Error reading directories from {c.NEW_PLANS_DIR}: {e}")
        return []


def backup_plan_file(directory_name: str) -> bool:
    """
    Create a backup of plan.json as plan_backup.json for a given directory.

    Args:
        directory_name: Directory name to backup plan.json from

    Returns:
        bool: True if backup was created or already exists, False if failed
    """
    try:
        new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"
        full_dir_path = os.path.join(new_plans_dir, directory_name)

        if not (os.path.exists(full_dir_path) and os.path.isdir(full_dir_path)):
            return False

        plan_json_path = os.path.join(full_dir_path, "plan.json")
        plan_backup_path = os.path.join(full_dir_path, "plan_backup.json")

        if not os.path.exists(plan_json_path):
            return False

        if os.path.exists(plan_backup_path):
            # Backup already exists
            return True

        shutil.copy2(plan_json_path, plan_backup_path)
        print(f"Created backup: {plan_backup_path}")
        return True

    except Exception as backup_error:
        print(f"Error creating backup for {directory_name}: {backup_error}")
        return False


def get_dirty_items_from_directory(directory_name: str) -> List[str]:
    """
    Extract dirty items from a directory name after "_dirty_".
    Also creates a backup of plan.json as plan_backup.json for dirty directories.

    Args:
        directory_name: Directory name containing "_dirty_" pattern

    Returns:
        List[str]: List of dirty items (capitalized)

    Examples:
        "distribute__Lettuce_dirty_plate_FloorPlan16_V5R1_Pickup Not Visible-Bowl Pot Plate [50]" -> ["Plate"]
        "cook_Egg_dirty_pan_and_plate_FloorPlan26_V2 [50]" -> ["Pan", "Plate"]
        "coffee_Mug_dirty_FloorPlan3_V1 [54]" -> ["Mug"]
    """
    if "_dirty_" not in directory_name:
        return []

    try:
        # Create backup of plan.json if directory exists
        backup_plan_file(directory_name)

        # Find the "_dirty_" part
        dirty_start = directory_name.find("_dirty_")
        if dirty_start == -1:
            return []

        # Special case for Mug - if "Mug" appears before "_dirty_", return ["Mug"]
        if "Mug" in directory_name[:dirty_start]:
            return ["Mug"]

        # Extract everything after "_dirty_"
        after_dirty = directory_name[dirty_start + len("_dirty_") :]

        # Find where the dirty items end (before FloorPlan or other suffixes)
        # Split by common separators that indicate end of dirty items
        end_markers = ["_FloorPlan", "_V1", "_V2", "_V3", "_V4", "_V5", " ["]

        dirty_section = after_dirty
        for marker in end_markers:
            if marker in dirty_section:
                dirty_section = dirty_section[: dirty_section.find(marker)]
                break

        # Handle "and" separator and split items
        if "_and_" in dirty_section:
            items = dirty_section.split("_and_")
        else:
            # Single item case
            items = [dirty_section] if dirty_section else []

        # Clean up items and convert to capitalized format
        cleaned_items = []
        for item in items:
            # Remove any remaining underscores and clean up
            cleaned = item.strip("_").strip()
            if cleaned:
                cleaned_items.append(cleaned.capitalize())

        return cleaned_items

    except Exception as e:
        print(f"Error parsing directory name '{directory_name}': {e}")
        return []


def print_dirty_directories_and_items():
    """
    Print all dirty directories and their extracted dirty items.
    Uses get_dirty_directories() and get_dirty_items_from_directory() functions.
    """
    dirty_dirs = get_dirty_directories()

    if not dirty_dirs:
        print("No dirty directories found.")
        return

    print(f"Found {len(dirty_dirs)} dirty directories:\n")

    for directory in dirty_dirs:
        dirty_items = get_dirty_items_from_directory(directory)
        items_str = ", ".join(dirty_items) if dirty_items else "No items extracted"
        print(f"Directory: {directory:<30}")
        print(f"Dirty items: [{items_str}]\n")


def update_dirty_state_goals(directory_name: str, dirty_items: List[str]) -> bool:
    """
    Update the state_goals in a plan.json file to match the provided dirty items.
    Removes existing "isDirty" state_goals and adds new ones for the dirty items.

    Args:
        directory_name: Directory name containing the plan.json file
        dirty_items: List of item names that should have isDirty state goals

    Returns:
        bool: True if update was successful, False if failed

    Example:
        update_dirty_state_goals("cook_Egg_dirty_pan_and_plate_FloorPlan1_V1", ["Pan", "Plate"])
    """
    try:
        # Construct path to plan.json
        new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"
        full_dir_path = os.path.join(new_plans_dir, directory_name)
        plan_json_path = os.path.join(full_dir_path, "plan.json")

        # Check if directory and plan.json exist
        if not (os.path.exists(full_dir_path) and os.path.isdir(full_dir_path)):
            print(f"Directory does not exist: {full_dir_path}")
            return False

        if not os.path.exists(plan_json_path):
            print(f"plan.json does not exist: {plan_json_path}")
            return False

        # Load the plan.json file
        with open(plan_json_path, "r") as f:
            plan_data = json.load(f)

        # Check if goal exists
        if "goal" not in plan_data:
            print(f"No 'goal' found in plan.json for {directory_name}")
            return False

        goal_data = plan_data["goal"]

        # Initialize state_goals if it doesn't exist
        if "state_goals" not in goal_data:
            goal_data["state_goals"] = []

        state_goals = goal_data["state_goals"]

        # Remove existing isDirty state goals
        updated_state_goals = []
        for state_goal in state_goals:
            # Keep state goals that are not isDirty
            if state_goal.get("state") != "isDirty":
                updated_state_goals.append(state_goal)
            else:
                print(
                    f"Removed isDirty state goal for {state_goal.get('object_type', 'unknown')}"
                )

        # Add new isDirty state goals for dirty items
        for dirty_item in dirty_items:
            new_state_goal = {
                "object_type": dirty_item,
                "state": "isDirty",
                "value": False,  # Goal is to make it not dirty (clean it)
                "outcome": "success",
            }
            updated_state_goals.append(new_state_goal)
            print(f"Added isDirty state goal for {dirty_item}")

        # Update the state_goals in the plan data
        goal_data["state_goals"] = updated_state_goals

        # Write the updated plan back to the file
        with open(plan_json_path, "w") as f:
            json.dump(plan_data, f, indent=2)

        print(f"Successfully updated state_goals in {plan_json_path}")
        return True

    except Exception as e:
        print(f"Error updating dirty state goals for {directory_name}: {e}")
        return False


def update_all_dirty_plan_files() -> None:
    """
    Get all dirty directories, extract dirty items from each directory name,
    and update the state_goals in each plan.json file to match the dirty items.

    This function combines:
    - get_dirty_directories() to find all directories with "_dirty_" in the name
    - get_dirty_items_from_directory() to extract dirty items from directory names
    - update_dirty_state_goals() to update the plan.json files
    """
    print("Starting update of all dirty plan files...\n")

    # Get all dirty directories
    dirty_dirs = get_dirty_directories()

    if not dirty_dirs:
        print("No dirty directories found.")
        return

    print(f"Found {len(dirty_dirs)} dirty directories to process.\n")

    success_count = 0
    failure_count = 0

    # Process each dirty directory
    for directory in dirty_dirs:
        print(f"Processing: {directory}")

        # Extract dirty items from directory name
        dirty_items = get_dirty_items_from_directory(directory)

        if not dirty_items:
            print(f"  No dirty items extracted from directory name")
            failure_count += 1
            continue

        print(f"  Dirty items found: {dirty_items}")

        # Update the plan.json file with the correct dirty state goals
        success = update_dirty_state_goals(directory, dirty_items)

        if success:
            success_count += 1
            print(f"  ✓ Successfully updated plan.json")
        else:
            failure_count += 1
            print(f"  ✗ Failed to update plan.json")

        print()  # Add blank line between directories

    # Print summary
    print("=" * 50)
    print(f"Update complete!")
    print(f"Successful updates: {success_count}")
    print(f"Failed updates: {failure_count}")
    print(f"Total directories processed: {len(dirty_dirs)}")

    if failure_count > 0:
        print(
            f"\nNote: {failure_count} directories failed to update. Check the error messages above for details."
        )


def delete_all_plan_backups() -> None:
    """
    Delete all plan_backup.json files in the NEW_PLANS_DIR directory.
    This will recursively search through all subdirectories and remove any plan_backup.json files found.
    """
    new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"

    if not os.path.exists(new_plans_dir):
        print(f"Directory does not exist: {new_plans_dir}")
        return

    print(f"Searching for plan_backup.json files in: {new_plans_dir}")

    deleted_count = 0
    error_count = 0

    try:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(new_plans_dir):
            for file in files:
                if file == "plan_backup.json":
                    backup_file_path = os.path.join(root, file)
                    try:
                        os.remove(backup_file_path)
                        deleted_count += 1
                        print(f"Deleted: {backup_file_path}")
                    except Exception as e:
                        error_count += 1
                        print(f"Error deleting {backup_file_path}: {e}")

    except Exception as e:
        print(f"Error walking through directory {new_plans_dir}: {e}")
        return

    # Print summary
    print("\n" + "=" * 50)
    print(f"Deletion complete!")
    print(f"Files deleted: {deleted_count}")
    print(f"Errors encountered: {error_count}")

    if deleted_count == 0:
        print("No plan_backup.json files were found.")
    elif error_count > 0:
        print(
            f"\nNote: {error_count} files could not be deleted. Check the error messages above for details."
        )


def get_directories_with_underscore_files() -> List[str]:
    """
    Look in GENERATION_DIR, find all directories (excluding 'best_of' and 'failures')
    that contain files starting with underscore, and return a list of unique directory names.
    Adds index suffixes to prevent duplicate names.

    Returns:
        List[str]: List of directory names, with index suffixes for duplicates

    Example:
        If two directories both have "_clean_mirror_FloorPlan423_V1 [7]" files,
        they would be returned as:
        ["_clean_mirror_FloorPlan423_V1 [7]", "_clean_mirror_FloorPlan423_V1 [7] (1)"]
    """
    generation_dir = c.DATASET_DIR

    if not os.path.exists(generation_dir):
        print(f"Generation directory does not exist: {generation_dir}")
        return []

    try:
        # Get all items in GENERATION_DIR
        all_items = os.listdir(generation_dir)

        # Filter for directories only, excluding 'best_of' and 'failures'
        excluded_dirs = {"best_of", "failures"}
        target_dirs = []

        for item in all_items:
            full_path = os.path.join(generation_dir, item)
            if os.path.isdir(full_path) and item not in excluded_dirs:
                target_dirs.append(item)

        # Find directories that contain files starting with underscore
        underscore_file_dirs = []

        for dir_name in target_dirs:
            dir_path = os.path.join(generation_dir, dir_name)

            try:
                files_in_dir = os.listdir(dir_path)

                # Check if any file starts with underscore
                has_underscore_file = any(file.startswith("_") for file in files_in_dir)

                if has_underscore_file:
                    # Find the first underscore file to use as the name
                    underscore_files = [f for f in files_in_dir if f.startswith("_")]
                    if underscore_files:
                        underscore_file_dirs.extend(underscore_files)

            except Exception as e:
                print(f"Error reading directory {dir_path}: {e}")
                continue

        # Handle duplicates by adding index suffixes
        unique_names = []
        name_counts = {}

        for name in underscore_file_dirs:
            if name not in name_counts:
                name_counts[name] = 0
                unique_names.append(name)
            else:
                name_counts[name] += 1
                indexed_name = f"{name} ({name_counts[name]})"
                unique_names.append(indexed_name)

        return sorted(unique_names)  # Return sorted for consistency

    except Exception as e:
        print(f"Error processing generation directory {generation_dir}: {e}")
        return []


def find_misplaced_coffee_goals() -> List[str]:
    """
    Find directories that contain "Drink Coffee" action goal but don't start with "coffee".

    Returns:
        List[str]: Directory names that have "Drink Coffee" action goal but don't start with "coffee"
    """
    misplaced_directories = []

    try:
        best_of_path = os.path.join(c.DATASET_DIR, c.NEW_PLANS_DIR)

        if not os.path.exists(best_of_path):
            print(f"Best of directory does not exist: {best_of_path}")
            return misplaced_directories

        # Get all subdirectories
        for dir_name in os.listdir(best_of_path):
            dir_path = os.path.join(best_of_path, dir_name)

            if not os.path.isdir(dir_path):
                continue

            plan_file = os.path.join(dir_path, "plan.json")

            if not os.path.exists(plan_file):
                continue

            try:
                # Read and parse the plan.json file
                with open(plan_file, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)

                # Check if the plan has a goal with action_goals containing "Drink Coffee"
                if "goal" in plan_data and "action_goals" in plan_data["goal"]:
                    action_goals = plan_data["goal"]["action_goals"]

                    # Check if "Drink Coffee" action goal exists and is true
                    if action_goals.get("Drink Coffee", False):
                        # Check if directory name does NOT start with "coffee"
                        if not dir_name.lower().startswith("coffee"):
                            misplaced_directories.append(dir_name)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading plan.json in {dir_name}: {e}")
                continue

    except Exception as e:
        print(f"Error processing best_of directory: {e}")

    return sorted(misplaced_directories)


def copy_plans_to_temp(directories: List[str]) -> bool:
    """
    Create a TEMP directory and copy plan.json files from misplaced coffee goal directories.

    Args:
        directories: List of directory names

    Returns:
        bool: True if all operations successful, False if any errors occurred
    """
    if not directories:
        print("No directories provided to copy.")
        return True

    try:
        # Create paths
        best_of_path = os.path.join(c.DATASET_DIR, c.NEW_PLANS_DIR)
        temp_path = os.path.join(c.DATASET_DIR, "TEMP")

        # Check if source directory exists
        if not os.path.exists(best_of_path):
            print(f"Source directory does not exist: {best_of_path}")
            return False

        # Create TEMP directory if it doesn't exist
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
            print(f"Created TEMP directory: {temp_path}")
        else:
            print(f"TEMP directory already exists: {temp_path}")

        print(f"\nCopying {len(directories)} directories to TEMP...\n")

        success_count = 0
        failure_count = 0

        for dir_name in directories:
            print(f"Processing: {dir_name}")

            # Source directory and plan.json file
            source_dir_path = os.path.join(best_of_path, dir_name)
            source_plan_file = os.path.join(source_dir_path, "plan.json")

            # Destination directory and plan.json file
            dest_dir_path = os.path.join(temp_path, dir_name)
            dest_plan_file = os.path.join(dest_dir_path, "plan.json")

            try:
                # Check if source directory and plan.json exist
                if not os.path.exists(source_dir_path):
                    print(f"  ✗ Source directory not found: {source_dir_path}")
                    failure_count += 1
                    continue

                if not os.path.exists(source_plan_file):
                    print(f"  ✗ plan.json not found in source: {source_plan_file}")
                    failure_count += 1
                    continue

                # Create destination directory
                if not os.path.exists(dest_dir_path):
                    os.makedirs(dest_dir_path)
                    print(f"  Created directory: {dest_dir_path}")
                else:
                    print(f"  Directory already exists: {dest_dir_path}")

                # Copy plan.json file
                shutil.copy2(source_plan_file, dest_plan_file)
                print(f"  ✓ Copied plan.json to: {dest_plan_file}")
                success_count += 1

            except Exception as e:
                print(f"  ✗ Error processing {dir_name}: {e}")
                failure_count += 1

        # Print summary
        print("\n" + "=" * 60)
        print("Copy operation complete!")
        print(f"Successful copies: {success_count}")
        print(f"Failed copies: {failure_count}")
        print(f"Total directories processed: {len(directories)}")
        print(f"TEMP directory location: {temp_path}")

        if failure_count > 0:
            print(
                f"\nNote: {failure_count} directories failed to copy. Check the error messages above for details."
            )

        return failure_count == 0

    except Exception as e:
        print(f"Error in copy_misplaced_coffee_goals_to_temp: {e}")
        return False


def correct_coffee_goals_in_temp() -> bool:
    """
    Correct plan.json files in the TEMP directory by:
    1. Removing "Drink Coffee" action goal from goal object
    2. Removing "ActionDrinkCoffee completed successfully" from all updated_memory lists

    Returns:
        bool: True if all operations successful, False if any errors occurred
    """
    try:
        temp_path = os.path.join(c.DATASET_DIR, "TEMP")

        if not os.path.exists(temp_path):
            print(f"TEMP directory does not exist: {temp_path}")
            return False

        # Get all directories in TEMP
        temp_directories = [
            d
            for d in os.listdir(temp_path)
            if os.path.isdir(os.path.join(temp_path, d))
        ]

        if not temp_directories:
            print("No directories found in TEMP directory.")
            return True

        print(
            f"Correcting {len(temp_directories)} plan.json files in TEMP directory...\n"
        )

        success_count = 0
        failure_count = 0

        for dir_name in temp_directories:
            print(f"Processing: {dir_name}")

            plan_file = os.path.join(temp_path, dir_name, "plan.json")

            if not os.path.exists(plan_file):
                print(f"  ✗ plan.json not found: {plan_file}")
                failure_count += 1
                continue

            try:
                # Read the plan.json file
                with open(plan_file, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)

                changes_made = False

                # 1. Remove "Drink Coffee" action goal
                if "goal" in plan_data and "action_goals" in plan_data["goal"]:
                    action_goals = plan_data["goal"]["action_goals"]
                    if "Drink Coffee" in action_goals:
                        del action_goals["Drink Coffee"]
                        print(f"  ✓ Removed 'Drink Coffee' action goal")
                        changes_made = True
                    else:
                        print(f"  - No 'Drink Coffee' action goal found")
                else:
                    print(f"  - No action_goals found in goal")

                # 2. Remove "Action Drink Coffee completed successfully." from all memory arrays
                memory_changes = 0

                def remove_coffee_from_memory(obj, path=""):
                    """Recursively search for memory arrays and remove coffee entries"""
                    nonlocal memory_changes

                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key

                            if key == "memory" and isinstance(value, list):
                                # Remove "Action Drink Coffee completed successfully." entries
                                original_length = len(value)
                                filtered_items = []

                                for item in value:
                                    if (
                                        isinstance(item, str)
                                        and "Action Drink Coffee completed successfully"
                                        in item
                                    ):
                                        # Skip this coffee-related item
                                        continue
                                    else:
                                        filtered_items.append(item)

                                obj[key] = filtered_items
                                removed_count = original_length - len(obj[key])
                                if removed_count > 0:
                                    memory_changes += removed_count
                                    print(
                                        f"  ✓ Removed {removed_count} coffee memory entries from {current_path}"
                                    )
                            elif key == "updated_memory" and isinstance(value, list):
                                # Also check updated_memory arrays
                                original_length = len(value)
                                filtered_items = []

                                for item in value:
                                    if (
                                        isinstance(item, str)
                                        and "Action Drink Coffee completed successfully"
                                        in item
                                    ):
                                        # Skip this coffee-related item
                                        continue
                                    else:
                                        filtered_items.append(item)

                                obj[key] = filtered_items
                                removed_count = original_length - len(obj[key])
                                if removed_count > 0:
                                    memory_changes += removed_count
                                    print(
                                        f"  ✓ Removed {removed_count} coffee memory entries from {current_path}"
                                    )
                            else:
                                remove_coffee_from_memory(value, current_path)

                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            remove_coffee_from_memory(item, f"{path}[{i}]")

                # Apply memory cleaning recursively
                remove_coffee_from_memory(plan_data)

                if memory_changes > 0:
                    changes_made = True
                else:
                    print(f"  - No coffee memory entries found")

                # Save the corrected plan.json file if changes were made
                if changes_made:
                    with open(plan_file, "w", encoding="utf-8") as f:
                        json.dump(plan_data, f, indent=2)
                    print(f"  ✓ Saved corrected plan.json")
                    success_count += 1
                else:
                    print(f"  - No changes needed")
                    success_count += 1

            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"  ✗ Error processing {plan_file}: {e}")
                failure_count += 1

        # Print summary
        print("\n" + "=" * 60)
        print("Correction operation complete!")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {failure_count}")
        print(f"Total directories processed: {len(temp_directories)}")

        if failure_count > 0:
            print(
                f"\nNote: {failure_count} files failed to process. Check the error messages above for details."
            )

        return failure_count == 0

    except Exception as e:
        print(f"Error in correct_coffee_goals_in_temp: {e}")
        return False


def add_kitchen_state_goals_to_plan(plan_json_path: str) -> bool:
    """
    Load a plan.json file and add standard kitchen state goals to it, then save it back.

    Adds the following state goals to the plan:
    - CoffeeMachine: isToggled = False
    - Toaster: isToggled = False
    - StoveKnob: isToggled = False
    - Microwave: isOpen = False
    - Fridge: isOpen = False
    - Faucet: isToggled = False

    Args:
        plan_json_path: Full path to the plan.json file to modify

    Returns:
        bool: True if successful, False if failed
    """
    try:
        # Check if file exists
        if not os.path.exists(plan_json_path):
            print(f"Plan file does not exist: {plan_json_path}")
            return False

        # Load the plan.json file
        with open(plan_json_path, "r", encoding="utf-8") as f:
            plan_data = json.load(f)

        # Check if goal exists
        if "goal" not in plan_data:
            print(f"No 'goal' found in plan.json: {plan_json_path}")
            return False

        goal_data = plan_data["goal"]

        # Initialize state_goals if it doesn't exist
        if "state_goals" not in goal_data:
            goal_data["state_goals"] = []

        state_goals = goal_data["state_goals"]

        # Define the kitchen state goals to add
        kitchen_goals = [
            {"object_type": "CoffeeMachine", "state": "isToggled", "value": False},
            {"object_type": "Toaster", "state": "isToggled", "value": False},
            {"object_type": "StoveKnob", "state": "isToggled", "value": False},
            {"object_type": "Microwave", "state": "isOpen", "value": False},
            {"object_type": "Fridge", "state": "isOpen", "value": False},
            {"object_type": "Faucet", "state": "isToggled", "value": False},
        ]

        goals_added = 0

        # Add each kitchen goal if it doesn't already exist
        for kitchen_goal in kitchen_goals:
            object_type = kitchen_goal["object_type"]
            state = kitchen_goal["state"]
            value = kitchen_goal["value"]

            # Check if this state goal already exists
            exists = False
            for existing_goal in state_goals:
                if (
                    existing_goal.get("object_type") == object_type
                    and existing_goal.get("state") == state
                ):
                    exists = True
                    break

            # Add the goal if it doesn't exist
            if not exists:
                new_state_goal = {
                    "object_type": object_type,
                    "state": state,
                    "value": value,
                    "outcome": "success",
                }
                state_goals.append(new_state_goal)
                goals_added += 1
                print(f"  Added state goal: {object_type} {state} = {value}")
            else:
                print(f"  Skipped existing state goal: {object_type} {state}")

        # Save the updated plan back to the file
        with open(plan_json_path, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, indent=2)

        print(f"Successfully updated {plan_json_path}")
        print(f"Goals added: {goals_added}")
        return True

    except (json.JSONDecodeError, KeyError, IOError) as e:
        print(f"Error processing {plan_json_path}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error processing {plan_json_path}: {e}")
        return False


def update_all_kitchen_plan_files() -> bool:
    """
    Complete workflow to backup, copy, and update all kitchen plan files with state goals.

    Steps performed:
    1. Backup all plan.json files in NEW_PLANS_DIR using backup_plan_file
    2. Get list of kitchen directories using get_kitchen_directories
    3. Copy plans to TEMP directory using copy_plans_to_temp
    4. Update each plan.json file with kitchen state goals using add_kitchen_state_goals_to_plan

    Returns:
        bool: True if all operations successful, False if any errors occurred
    """
    print("Starting complete kitchen plan files update workflow...\n")

    try:
        # Step 1: Backup all plan.json files in NEW_PLANS_DIR
        print("=" * 60)
        print("Step 1: Backing up all plan.json files in NEW_PLANS_DIR...")
        print("=" * 60)

        new_plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"

        if not os.path.exists(new_plans_dir):
            print(f"NEW_PLANS_DIR does not exist: {new_plans_dir}")
            return False

        # Get all directories in NEW_PLANS_DIR
        all_dirs = [
            d
            for d in os.listdir(new_plans_dir)
            if os.path.isdir(os.path.join(new_plans_dir, d))
        ]

        backup_success_count = 0
        backup_failure_count = 0

        for dir_name in all_dirs:
            success = backup_plan_file(dir_name)
            if success:
                backup_success_count += 1
            else:
                backup_failure_count += 1

        print(f"\nBackup summary:")
        print(f"Successful backups: {backup_success_count}")
        print(f"Failed backups: {backup_failure_count}")
        print(f"Total directories processed: {len(all_dirs)}\n")

        # Step 2: Get kitchen directories
        print("=" * 60)
        print("Step 2: Getting kitchen directories...")
        print("=" * 60)

        kitchen_dirs = get_kitchen_directories()

        if not kitchen_dirs:
            print("No kitchen directories found.")
            return False

        print(f"Found {len(kitchen_dirs)} kitchen directories:")
        for dir_name in kitchen_dirs:
            print(f"  - {dir_name}")
        print()

        # Step 3: Copy plans to TEMP directory
        print("=" * 60)
        print("Step 3: Copying kitchen plans to TEMP directory...")
        print("=" * 60)

        copy_success = copy_plans_to_temp(kitchen_dirs)

        if not copy_success:
            print("Failed to copy plans to TEMP directory.")
            return False

        print("Successfully copied all kitchen plans to TEMP directory.\n")

        # Step 4: Update each plan.json file with kitchen state goals
        print("=" * 60)
        print("Step 4: Adding kitchen state goals to plan files...")
        print("=" * 60)

        temp_path = os.path.join(c.DATASET_DIR, "TEMP")
        update_success_count = 0
        update_failure_count = 0

        for dir_name in kitchen_dirs:
            print(f"Updating: {dir_name}")
            plan_json_path = os.path.join(temp_path, dir_name, "plan.json")

            success = add_kitchen_state_goals_to_plan(plan_json_path)
            if success:
                update_success_count += 1
                print(f"  ✓ Successfully updated kitchen state goals")
            else:
                update_failure_count += 1
                print(f"  ✗ Failed to update kitchen state goals")
            print()

        # Final summary
        print("=" * 60)
        print("COMPLETE WORKFLOW SUMMARY")
        print("=" * 60)
        print(
            f"Step 1 - Backups: {backup_success_count} successful, {backup_failure_count} failed"
        )
        print(f"Step 2 - Kitchen directories found: {len(kitchen_dirs)}")
        print(f"Step 3 - Copy to TEMP: {'Success' if copy_success else 'Failed'}")
        print(
            f"Step 4 - State goals added: {update_success_count} successful, {update_failure_count} failed"
        )
        print()

        overall_success = (
            backup_failure_count == 0 and copy_success and update_failure_count == 0
        )

        if overall_success:
            print("✓ All operations completed successfully!")
            print(f"Updated plan files are available in: {temp_path}")
        else:
            print("✗ Some operations failed. Check the detailed logs above.")

        return overall_success

    except Exception as e:
        print(f"Error in update_all_kitchen_plan_files: {e}")
        return False


def add_cooking_location_goals_to_temp_plans() -> bool:
    """
    Add appropriate location goals to cooking plan files in the TEMP directory.

    Rules applied:
    - cook_Egg directories: add location goal "EggCracked" -> "Plate"
    - cook_Bread directories: add location goal "BreadSliced" -> "Plate"
    - cook_Potato directories: add location goal "Potato" -> "Bowl"

    Returns:
        bool: True if all operations successful, False if any errors occurred
    """
    print("Adding cooking location goals to TEMP plan files...\n")

    try:
        temp_path = os.path.join(c.DATASET_DIR, "TEMP")

        if not os.path.exists(temp_path):
            print(f"TEMP directory does not exist: {temp_path}")
            return False

        # Get all directories in TEMP
        all_dirs = [
            d
            for d in os.listdir(temp_path)
            if os.path.isdir(os.path.join(temp_path, d))
        ]

        # Define the cooking goal mappings
        cooking_goals = [
            {
                "prefix": "cook_Egg",
                "object_type": "EggCracked",
                "destination_type": "Plate",
            },
            {
                "prefix": "cook_Bread",
                "object_type": "BreadSliced",
                "destination_type": "Plate",
            },
            {
                "prefix": "cook_Potato",
                "object_type": "Potato",
                "destination_type": "Bowl",
            },
        ]

        success_count = 0
        failure_count = 0
        skipped_count = 0

        for dir_name in all_dirs:
            print(f"Processing: {dir_name}")

            # Check if this directory matches any cooking pattern
            matching_goal = None
            for goal_config in cooking_goals:
                if dir_name.startswith(goal_config["prefix"]):
                    matching_goal = goal_config
                    break

            if matching_goal is None:
                print(f"  - Skipped (not a cooking plan)")
                skipped_count += 1
                continue

            # Load and update the plan.json file
            plan_json_path = os.path.join(temp_path, dir_name, "plan.json")

            if not os.path.exists(plan_json_path):
                print(f"  ✗ plan.json not found: {plan_json_path}")
                failure_count += 1
                continue

            try:
                # Read the plan.json file
                with open(plan_json_path, "r", encoding="utf-8") as f:
                    plan_data = json.load(f)

                # Check if goal exists
                if "goal" not in plan_data:
                    print(f"  ✗ No 'goal' found in plan.json")
                    failure_count += 1
                    continue

                goal_data = plan_data["goal"]

                # Initialize location_goals if it doesn't exist
                if "location_goals" not in goal_data:
                    goal_data["location_goals"] = []

                location_goals = goal_data["location_goals"]

                # Check if this location goal already exists
                object_type = matching_goal["object_type"]
                destination_type = matching_goal["destination_type"]

                exists = False
                for existing_goal in location_goals:
                    if (
                        existing_goal.get("object_type") == object_type
                        and existing_goal.get("destination_type") == destination_type
                    ):
                        exists = True
                        break

                if exists:
                    print(
                        f"  - Location goal already exists: {object_type} -> {destination_type}"
                    )
                    success_count += 1
                    continue

                # Add the new location goal
                new_location_goal = {
                    "object_type": object_type,
                    "destination_type": destination_type,
                    "outcome": "success",
                }
                location_goals.append(new_location_goal)

                # Save the updated plan back to the file
                with open(plan_json_path, "w", encoding="utf-8") as f:
                    json.dump(plan_data, f, indent=2)

                print(f"  ✓ Added location goal: {object_type} -> {destination_type}")
                success_count += 1

            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"  ✗ Error processing {plan_json_path}: {e}")
                failure_count += 1

        # Print summary
        print("\n" + "=" * 60)
        print("Cooking location goals update complete!")
        print(f"Successfully updated: {success_count}")
        print(f"Failed to update: {failure_count}")
        print(f"Skipped (non-cooking): {skipped_count}")
        print(f"Total directories processed: {len(all_dirs)}")

        if failure_count > 0:
            print(
                f"\nNote: {failure_count} files failed to process. Check the error messages above for details."
            )

        return failure_count == 0

    except Exception as e:
        print(f"Error in add_cooking_location_goals_to_temp_plans: {e}")
        return False
