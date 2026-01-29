#!/usr/bin/env python3
"""
Script to remove test results from a test directory structure.

By default, runs in dry-run mode for safety. Use --execute (-x) to apply changes.

Usage:
    # Preview changes (dry-run is default):
    python remove_test_result.py --base <base_name> --test <test_name>

    # Actually apply changes:
    python remove_test_result.py --base <base_name> --test <test_name> --execute

    # Remove a specific plan folder:
    python remove_test_result.py <plan_folder_path> --execute

Examples:
    # Preview what would be removed:
    python remove_test_result.py -b Test -t "turn_on_tv__FloorPlan212_V1"

    # Actually remove all instances of a test:
    python remove_test_result.py -b Test -t "turn_on_tv__FloorPlan212_V1" -x

    # Remove a specific plan folder:
    python remove_test_result.py "Test/magt_benchmark/gpt-4o--T0_.../Plans/_coffee__Mug(d)_FloorPlan12_V1" -x
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


def extract_plan_name_from_folder(folder_name: str) -> str:
    """
    Extract the base plan name from a folder name.
    Handles underscore prefix and step count suffix like "[38]".

    Examples:
        "_coffee__Mug(d)_FloorPlan12_V1" -> "coffee__Mug(d)_FloorPlan12_V1"
        "_coffee__Mug(d)_FloorPlan12_V1 [38]" -> "coffee__Mug(d)_FloorPlan12_V1"
        "coffee__Mug(d)_FloorPlan12_V1[38]" -> "coffee__Mug(d)_FloorPlan12_V1"
    """
    name = folder_name

    # Remove leading underscore (indicates failed test)
    if name.startswith("_"):
        name = name[1:]

    # Remove step count suffix like " [38]" or "[38]"
    name = re.sub(r"\s*\[\d+\]$", "", name)

    return name


def parse_plan_path(plan_path: str) -> tuple[str, str, str]:
    """
    Parse a plan folder path to extract test_dir, plans_dir, and plan_name.

    Args:
        plan_path: Path like "Test/.../Plans/_plan_name" or absolute path

    Returns:
        Tuple of (test_dir, plan_folder, plan_name)
    """
    plan_path = plan_path.rstrip("/")

    # Get absolute path
    if not os.path.isabs(plan_path):
        plan_path = os.path.abspath(plan_path)

    # Extract folder name and parent
    plan_folder = plan_path
    folder_name = os.path.basename(plan_path)
    plans_dir = os.path.dirname(plan_path)

    # Verify we're in a Plans directory
    if os.path.basename(plans_dir) != "Plans":
        raise ValueError(
            f"Expected path to be inside a 'Plans' directory, got: {plans_dir}"
        )

    test_dir = os.path.dirname(plans_dir)
    plan_name = extract_plan_name_from_folder(folder_name)

    return test_dir, plan_folder, plan_name


def remove_test_result(plan_path: str, dry_run: bool = False) -> bool:
    """
    Remove a test result entry from test_results.json and delete the associated plan folder.

    Args:
        plan_path: Path to the plan folder (e.g., "Test/.../Plans/_coffee__Mug(d)_FloorPlan12_V1")
        dry_run: If True, only report what would be changed without making changes

    Returns:
        True if successful, False otherwise
    """
    try:
        test_dir, plan_folder, plan_name = parse_plan_path(plan_path)
    except ValueError as e:
        print(f"Error: {e}")
        return False

    print(f"Test directory: {test_dir}")
    print(f"Plan folder: {plan_folder}")
    print(f"Plan name: {plan_name}")
    print()

    # Path to test_results.json
    results_path = os.path.join(test_dir, "test_results.json")

    # Check if test_results.json exists
    if not os.path.exists(results_path):
        print(f"Error: test_results.json not found at {results_path}")
        return False

    # Check if plan folder exists
    folder_exists = os.path.exists(plan_folder) and os.path.isdir(plan_folder)
    if not folder_exists:
        print(f"Warning: Plan folder not found at {plan_folder}")

    # Load and modify test_results.json
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse test_results.json: {e}")
        return False

    # Find and remove the entry
    # Support both "results" and "test_results" keys
    results_key = "test_results" if "test_results" in data else "results"
    results = data.get(results_key, [])
    original_count = len(results)

    # Filter out entries matching the plan name
    new_results = [r for r in results if r.get("task_name") != plan_name]
    removed_count = original_count - len(new_results)

    if removed_count == 0:
        print(f"Warning: No entry found for '{plan_name}' in test_results.json")
    else:
        print(
            f"Found {removed_count} entry/entries for '{plan_name}' in test_results.json"
        )

    # Summary
    print()
    print("=== Summary ===")
    if removed_count > 0:
        print(f"  Will remove {removed_count} entry/entries from test_results.json")
    if folder_exists:
        print(f"  Will delete folder: {plan_folder}")

    if removed_count == 0 and not folder_exists:
        print("  Nothing to do.")
        return True

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return True

    # Apply changes
    print()

    # Update test_results.json
    if removed_count > 0:
        data[results_key] = new_results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Updated test_results.json (removed {removed_count} entries)")

    # Delete plan folder
    if folder_exists:
        shutil.rmtree(plan_folder)
        print(f"Deleted folder: {plan_folder}")

    print("\nDone!")
    return True


def remove_test_from_base(
    base_name: str, test_name: str, dry_run: bool = False
) -> bool:
    """
    Remove all instances of a test from a base directory structure.

    Finds all folders matching the test_name in {base_name}/**/Plans/ directories,
    moves them to {base_name}_Removed/ preserving directory structure,
    and removes entries from all test_results.json files.

    Args:
        base_name: Base directory name (e.g., "Test")
        test_name: Test name to remove (e.g., "turn_on_tv__FloorPlan212_V1")
        dry_run: If True, only report what would be changed without making changes

    Returns:
        True if successful, False otherwise
    """
    base_path = Path(base_name)
    removed_base_path = Path(f"{base_name}_Removed")

    if not base_path.exists():
        print(f"Error: Base directory '{base_name}' not found")
        return False

    # Normalize test_name (remove leading underscore if present)
    normalized_test_name = test_name.lstrip("_")

    # Find all matching plan folders
    found_folders = []
    found_results_files = set()

    # Walk through the base directory looking for Plans folders
    for plans_dir in base_path.rglob("Plans"):
        if not plans_dir.is_dir():
            continue

        # Check for both with and without underscore prefix
        for prefix in ["", "_"]:
            plan_folder = plans_dir / f"{prefix}{normalized_test_name}"
            if plan_folder.exists() and plan_folder.is_dir():
                found_folders.append(plan_folder)
                # Track the test_results.json file for this test setup
                test_dir = plans_dir.parent
                results_file = test_dir / "test_results.json"
                if results_file.exists():
                    found_results_files.add(results_file)

    if not found_folders and not found_results_files:
        print(f"No instances of '{test_name}' found in {base_name}")
        return True

    # Summary
    print(f"=== Found {len(found_folders)} folder(s) matching '{test_name}' ===")
    for folder in found_folders:
        print(f"  {folder}")

    print(
        f"\n=== Found {len(found_results_files)} test_results.json file(s) to update ==="
    )
    for results_file in found_results_files:
        print(f"  {results_file}")

    # Process test_results.json files
    total_entries_removed = 0
    results_updates = []  # List of (file_path, original_data, new_data)

    for results_file in found_results_files:
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {results_file}: {e}")
            continue

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])
        original_count = len(results)

        # Filter out entries matching the test name
        new_results = [r for r in results if r.get("task_name") != normalized_test_name]
        removed_count = original_count - len(new_results)

        if removed_count > 0:
            total_entries_removed += removed_count
            new_data = data.copy()
            new_data[results_key] = new_results
            results_updates.append((results_file, data, new_data, results_key))
            print(f"  Will remove {removed_count} entry/entries from {results_file}")

    print(f"\n=== Summary ===")
    print(f"  Folders to move: {len(found_folders)}")
    print(f"  Entries to remove from test_results.json: {total_entries_removed}")

    if dry_run:
        print("\n[DRY RUN] No changes made.")
        return True

    # Apply changes
    print()

    # Move folders to _Removed directory
    for folder in found_folders:
        # Calculate relative path from base
        rel_path = folder.relative_to(base_path)
        dest_folder = removed_base_path / rel_path

        # Create parent directories
        dest_folder.parent.mkdir(parents=True, exist_ok=True)

        # Move the folder
        shutil.move(str(folder), str(dest_folder))
        print(f"Moved: {folder} -> {dest_folder}")

    # Update test_results.json files
    for results_file, original_data, new_data, results_key in results_updates:
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2)
        removed_count = len(original_data[results_key]) - len(new_data[results_key])
        print(f"Updated {results_file} (removed {removed_count} entries)")

    print("\nDone!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Remove test results from a test directory structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch the UI:
  python remove_test_result.py --ui
  python remove_test_result.py --ui --base Test/20260115_Test

  # Preview changes (default - dry run):
  python remove_test_result.py --base Test --test "turn_on_tv__FloorPlan212_V1"

  # Actually make the changes:
  python remove_test_result.py --base Test --test "turn_on_tv__FloorPlan212_V1" --execute
  python remove_test_result.py -b Test -t "turn_on_tv__FloorPlan212_V1" -x

  # Remove a specific plan folder:
  python remove_test_result.py "Test/magt_benchmark/gpt-4o--T0_.../Plans/_coffee__Mug(d)_FloorPlan12_V1" --execute
""",
    )

    parser.add_argument(
        "plan_path", nargs="?", help="Path to a specific plan folder to remove"
    )
    parser.add_argument("-b", "--base", help="Base directory name (e.g., 'Test')")
    parser.add_argument(
        "-t", "--test", help="Test name to remove (e.g., 'turn_on_tv__FloorPlan212_V1')"
    )
    parser.add_argument(
        "-x",
        "--execute",
        action="store_true",
        help="Actually execute the changes (default is dry-run for safety)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch the graphical user interface",
    )

    args = parser.parse_args()

    # Launch UI if requested
    if args.ui:
        run_ui(args.base)
        return

    # dry_run is True by default, False only if --execute is passed
    dry_run = not args.execute

    if dry_run:
        print("=== DRY RUN MODE (use --execute or -x to apply changes) ===\n")

    # Determine which mode to use
    if args.base and args.test:
        # Remove all instances of a test from base directory
        success = remove_test_from_base(args.base, args.test, dry_run=dry_run)
    elif args.plan_path:
        # Remove a specific plan folder (original behavior)
        success = remove_test_result(args.plan_path, dry_run=dry_run)
    else:
        parser.print_help()
        sys.exit(1)

    if not success:
        sys.exit(1)


# ============================================================================
# Graphical User Interface
# ============================================================================


def find_all_test_names(base_path: Path) -> dict[str, list[Path]]:
    """
    Find all unique test names and their locations in a base directory.

    Returns:
        Dict mapping test_name -> list of plan folder paths
    """
    test_names: dict[str, list[Path]] = {}

    # Walk through the base directory looking for Plans folders
    for plans_dir in base_path.rglob("Plans"):
        if not plans_dir.is_dir():
            continue

        # Look at all subdirectories in Plans
        try:
            for item in plans_dir.iterdir():
                if not item.is_dir():
                    continue

                # Extract test name from folder name
                plan_name = extract_plan_name_from_folder(item.name)
                if plan_name:
                    if plan_name not in test_names:
                        test_names[plan_name] = []
                    test_names[plan_name].append(item)
        except PermissionError:
            continue

    return test_names


def run_ui(initial_base: str | None = None):
    """Launch the graphical user interface."""
    import tkinter as tk
    from tkinter import messagebox, ttk

    class RemoveTestResultApp:
        def __init__(self, root: tk.Tk):
            self.root = root
            self.root.title("Remove Test Results")
            self.root.geometry("1000x700")

            self.test_names: dict[str, list[Path]] = {}
            self.is_scanning = False
            self.is_removing = False

            self._create_widgets()

            # Set initial base if provided
            if initial_base:
                self.dir_entry.delete(0, tk.END)
                self.dir_entry.insert(0, initial_base)

        def _create_widgets(self):
            # Main container
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky="nsew")
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=1)

            # Directory input section
            dir_frame = ttk.LabelFrame(main_frame, text="Base Directory", padding="5")
            dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            dir_frame.columnconfigure(1, weight=1)

            ttk.Label(dir_frame, text="Directory:").grid(
                row=0, column=0, sticky="w", padx=(0, 5)
            )
            self.dir_entry = ttk.Entry(dir_frame)
            self.dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
            self.dir_entry.insert(0, "Test")

            self.scan_btn = ttk.Button(dir_frame, text="Scan", command=self._start_scan)
            self.scan_btn.grid(row=0, column=2, padx=(5, 0))

            # Filter section
            filter_frame = ttk.Frame(main_frame)
            filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
            filter_frame.columnconfigure(1, weight=1)

            ttk.Label(filter_frame, text="Filter:").grid(
                row=0, column=0, sticky="w", padx=(0, 5)
            )
            self.filter_entry = ttk.Entry(filter_frame)
            self.filter_entry.grid(row=0, column=1, sticky="ew")
            self.filter_entry.bind("<KeyRelease>", self._apply_filter)

            # Status label
            self.status_label = ttk.Label(
                main_frame, text="Enter a directory and click Scan"
            )
            self.status_label.grid(row=2, column=0, sticky="w", pady=(0, 5))

            # Results section
            results_frame = ttk.LabelFrame(
                main_frame, text="Test Names Found", padding="5"
            )
            results_frame.grid(row=3, column=0, sticky="nsew", pady=(0, 10))
            results_frame.columnconfigure(0, weight=1)
            results_frame.rowconfigure(0, weight=1)
            main_frame.rowconfigure(3, weight=1)

            # Treeview with columns
            columns = ("test_name", "count", "locations")
            self.results_tree = ttk.Treeview(
                results_frame, columns=columns, show="headings", selectmode="extended"
            )
            self.results_tree.heading("test_name", text="Test Name", anchor="w")
            self.results_tree.heading("count", text="Count", anchor="center")
            self.results_tree.heading("locations", text="Sample Location", anchor="w")
            self.results_tree.column("test_name", width=350)
            self.results_tree.column("count", width=60, anchor="center")
            self.results_tree.column("locations", width=500)
            self.results_tree.grid(row=0, column=0, sticky="nsew")

            scrollbar_y = ttk.Scrollbar(
                results_frame, orient="vertical", command=self.results_tree.yview
            )
            scrollbar_y.grid(row=0, column=1, sticky="ns")
            self.results_tree.configure(yscrollcommand=scrollbar_y.set)

            scrollbar_x = ttk.Scrollbar(
                results_frame, orient="horizontal", command=self.results_tree.xview
            )
            scrollbar_x.grid(row=1, column=0, sticky="ew")
            self.results_tree.configure(xscrollcommand=scrollbar_x.set)

            # Count label
            self.count_label = ttk.Label(results_frame, text="")
            self.count_label.grid(row=2, column=0, sticky="w", pady=(5, 0))

            # Progress section
            progress_frame = ttk.Frame(main_frame)
            progress_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
            progress_frame.columnconfigure(0, weight=1)

            self.progress_var = tk.DoubleVar()
            self.progress_bar = ttk.Progressbar(
                progress_frame, variable=self.progress_var, maximum=100
            )
            self.progress_bar.grid(row=0, column=0, sticky="ew")

            self.progress_label = ttk.Label(progress_frame, text="")
            self.progress_label.grid(row=1, column=0, sticky="w")

            # Buttons section
            btn_frame = ttk.Frame(main_frame)
            btn_frame.grid(row=5, column=0, sticky="e")

            self.remove_btn = ttk.Button(
                btn_frame,
                text="Remove Selected Tests",
                command=self._start_remove,
                state="disabled",
            )
            self.remove_btn.grid(row=0, column=0, padx=(0, 5))

            self.cancel_btn = ttk.Button(
                btn_frame, text="Close", command=self.root.quit
            )
            self.cancel_btn.grid(row=0, column=1)

        def _start_scan(self):
            """Start scanning for test names."""
            if self.is_scanning or self.is_removing:
                return

            base_path = self.dir_entry.get().strip()
            if not base_path:
                messagebox.showerror("Error", "Please enter a directory path")
                return

            if not os.path.isdir(base_path):
                messagebox.showerror("Error", f"Directory not found: {base_path}")
                return

            self.is_scanning = True
            self.scan_btn.configure(state="disabled")
            self.remove_btn.configure(state="disabled")
            self.results_tree.delete(*self.results_tree.get_children())
            self.test_names = {}

            self.status_label.configure(text="Scanning for test names...")
            self.progress_label.configure(text="Scanning directories...")
            self.root.update()

            # Find all test names
            base = Path(base_path)
            self.test_names = find_all_test_names(base)

            # Populate treeview
            self._populate_tree()

            self.count_label.configure(
                text=f"Total: {len(self.test_names)} unique test names"
            )
            self.status_label.configure(
                text=f"Scan complete. Found {len(self.test_names)} unique test names."
            )
            self.progress_var.set(0)
            self.progress_label.configure(text="")

            self.is_scanning = False
            self.scan_btn.configure(state="normal")

            if self.test_names:
                self.remove_btn.configure(state="normal")

        def _populate_tree(self, filter_text: str = ""):
            """Populate the treeview with test names."""
            self.results_tree.delete(*self.results_tree.get_children())

            filter_lower = filter_text.lower()
            visible_count = 0

            for test_name in sorted(self.test_names.keys()):
                if filter_lower and filter_lower not in test_name.lower():
                    continue

                locations = self.test_names[test_name]
                count = len(locations)
                # Show first location as sample
                sample_location = str(locations[0]) if locations else ""

                self.results_tree.insert(
                    "", "end", values=(test_name, count, sample_location)
                )
                visible_count += 1

            if filter_text:
                self.count_label.configure(
                    text=f"Showing {visible_count} of {len(self.test_names)} test names"
                )
            else:
                self.count_label.configure(
                    text=f"Total: {len(self.test_names)} unique test names"
                )

        def _apply_filter(self, event=None):
            """Apply filter to the treeview."""
            filter_text = self.filter_entry.get().strip()
            self._populate_tree(filter_text)

        def _start_remove(self):
            """Start removing selected tests."""
            if self.is_removing:
                return

            selected_items = self.results_tree.selection()
            if not selected_items:
                messagebox.showwarning(
                    "Warning", "Please select at least one test to remove"
                )
                return

            # Get selected test names
            selected_tests = []
            total_folders = 0
            for item in selected_items:
                values = self.results_tree.item(item, "values")
                test_name = values[0]
                count = int(values[1])
                selected_tests.append(test_name)
                total_folders += count

            # Get base path for backup location
            base_path = self.dir_entry.get().strip()
            base_name = os.path.basename(base_path.rstrip("/"))
            backup_path = f"{base_path}_Removed"

            # Confirm with user
            if not messagebox.askyesno(
                "Confirm Removal",
                f"This will:\n"
                f"• Move {total_folders} plan folder(s) to:\n"
                f"  {backup_path}\n"
                f"• Remove entries from test_results.json files\n\n"
                f"Tests to remove ({len(selected_tests)}):\n"
                f"  {', '.join(selected_tests[:5])}"
                f"{'...' if len(selected_tests) > 5 else ''}\n\n"
                f"Continue?",
            ):
                return

            self.is_removing = True
            self.scan_btn.configure(state="disabled")
            self.remove_btn.configure(state="disabled")

            self.status_label.configure(text="Removing selected tests...")
            self.root.update()

            # Remove each selected test
            success_count = 0
            error_count = 0
            total = len(selected_tests)

            for i, test_name in enumerate(selected_tests):
                self.progress_var.set((i + 1) / total * 100)
                self.progress_label.configure(
                    text=f"[{i + 1}/{total}] Removing {test_name}..."
                )
                self.root.update()

                try:
                    if remove_test_from_base(base_path, test_name, dry_run=False):
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error removing {test_name}: {e}")
                    error_count += 1

            self.progress_var.set(0)
            self.progress_label.configure(text="")

            self.is_removing = False
            self.scan_btn.configure(state="normal")

            # Refresh the list
            self._start_scan()

            if error_count > 0:
                self.status_label.configure(
                    text=f"Completed with {error_count} error(s). {success_count} tests removed."
                )
            else:
                self.status_label.configure(
                    text=f"Successfully removed {success_count} test(s)."
                )
                messagebox.showinfo(
                    "Complete", f"Successfully removed {success_count} test(s)."
                )

    root = tk.Tk()
    app = RemoveTestResultApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
