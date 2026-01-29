#!/usr/bin/env python3
"""
UI utility to find and remove test results with API failures.

Recursively searches for directories ending in --rep{n}, reads their test_results.json,
finds entries with "fail_reason": "API_Failure", and allows the user to remove them.

Removed plan folders are moved to a backup location, not permanently deleted.
"""

import json
import os
import re
import shutil
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Tuple

# Default base path
DEFAULT_BASE_PATH = "/mnt/magmathor"

# Pattern to match --rep{n} directories
REP_PATTERN = re.compile(r"--rep\d+$")


@dataclass
class APIFailureEntry:
    """Represents a test result with API failure."""

    rep_dir: Path  # The --rep{n} directory
    task_name: str  # The plan/task name
    plan_folder: Path  # Path to the plan folder in Plans/
    results_file: Path  # Path to test_results.json


def find_rep_directories(base_path: Path, progress_callback=None) -> List[Path]:
    """Find all directories ending in --rep{n} pattern."""
    rep_dirs = []
    dirs_scanned = 0

    for root, dirs, files in os.walk(base_path):
        dirs_scanned += 1
        if progress_callback and dirs_scanned % 50 == 0:
            progress_callback(
                f"Scanning directories... ({dirs_scanned} checked, {len(rep_dirs)} found)"
            )

        # Check which dirs match the pattern and remove them from dirs to avoid descending into them
        matched = []
        for d in dirs:
            if REP_PATTERN.search(d):
                rep_dirs.append(Path(root) / d)
                matched.append(d)
                print(f"Found rep dir: {Path(root) / d}")

        # Remove matched dirs from the list to prevent os.walk from descending into them
        for d in matched:
            dirs.remove(d)

        # Also skip _Removed directories to avoid scanning backups
        dirs[:] = [d for d in dirs if not d.endswith("_Removed")]

    return rep_dirs


def find_api_failures(rep_dir: Path) -> List[APIFailureEntry]:
    """Find all API failure entries in a --rep{n} directory."""
    failures = []
    results_file = rep_dir / "test_results.json"

    if not results_file.exists():
        print(f"  No test_results.json in {rep_dir}")
        return failures

    try:
        with open(results_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to read {results_file}: {e}")
        return failures

    # Support both "results" and "test_results" keys
    results_key = "test_results" if "test_results" in data else "results"
    results = data.get(results_key, [])

    # Debug: show all fail_reasons in this file
    fail_reasons = set(r.get("fail_reason") for r in results if r.get("fail_reason"))
    api_failure_count = sum(1 for r in results if r.get("fail_reason") == "API_Failure")
    print(
        f"  {rep_dir.name}: {len(results)} results, fail_reasons={fail_reasons}, API_Failure count={api_failure_count}"
    )

    plans_dir = rep_dir / "Plans"

    for entry in results:
        if entry.get("fail_reason") == "API_Failure":
            task_name = entry.get("task_name", "")
            if task_name:
                # Find the plan folder - could be with or without underscore prefix
                plan_folder = None
                for prefix in ["_", ""]:
                    candidate = plans_dir / f"{prefix}{task_name}"
                    if candidate.exists():
                        plan_folder = candidate
                        break
                    # Also check with step count suffix like "[38]"
                    for item in plans_dir.glob(f"{prefix}{task_name}*"):
                        if item.is_dir():
                            plan_folder = item
                            break
                    if plan_folder:
                        break

                failures.append(
                    APIFailureEntry(
                        rep_dir=rep_dir,
                        task_name=task_name,
                        plan_folder=plan_folder,
                        results_file=results_file,
                    )
                )

    return failures


def remove_api_failures(
    failures: List[APIFailureEntry], progress_callback=None, error_callback=None
) -> Tuple[int, List[str]]:
    """
    Remove API failure entries and move their plan folders to backup.

    Returns:
        Tuple of (success_count, list of error messages)
    """
    success_count = 0
    errors = []
    total = len(failures)

    # Group failures by results file for batch processing
    failures_by_file: Dict[Path, List[APIFailureEntry]] = {}
    for f in failures:
        if f.results_file not in failures_by_file:
            failures_by_file[f.results_file] = []
        failures_by_file[f.results_file].append(f)

    processed = 0

    for results_file, file_failures in failures_by_file.items():
        # Update test_results.json
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            results_key = "test_results" if "test_results" in data else "results"
            results = data.get(results_key, [])

            # Get task names to remove
            task_names_to_remove = {f.task_name for f in file_failures}

            # Filter out API failure entries
            new_results = [
                r for r in results if r.get("task_name") not in task_names_to_remove
            ]

            removed_count = len(results) - len(new_results)

            if removed_count > 0:
                data[results_key] = new_results
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                print(f"Updated {results_file}: removed {removed_count} entries")

        except Exception as e:
            err_msg = f"Failed to update {results_file}: {e}"
            errors.append(err_msg)
            if error_callback:
                error_callback(err_msg)
            continue

        # Move plan folders to backup
        for failure in file_failures:
            processed += 1

            if progress_callback:
                should_continue = progress_callback(
                    processed, total, f"Processing {failure.task_name}"
                )
                if should_continue is False:
                    break

            if failure.plan_folder and failure.plan_folder.exists():
                # Create backup path
                # Use the first directory after /mnt/magmathor/ as the root, append _API_Failures
                # e.g., /mnt/magmathor/20260115_Test/benchmark/model/task--rep1/Plans/plan
                #    -> /mnt/magmathor/20260115_Test_API_Failures/benchmark/model/task--rep1/Plans/plan
                rep_dir = failure.rep_dir

                # Find the magmathor mount point and use the next directory as the root
                backup_folder = None
                parts = rep_dir.parts
                for i, part in enumerate(parts):
                    if part == "magmathor" and i + 1 < len(parts):
                        # Next part is the test root directory
                        test_root_name = parts[i + 1]
                        test_root = Path(*parts[: i + 2])
                        backup_root = (
                            test_root.parent / f"{test_root_name}_API_Failures"
                        )
                        # Preserve relative path from test root
                        rel_path = rep_dir.relative_to(test_root)
                        backup_plans_dir = backup_root / rel_path / "Plans"
                        backup_folder = backup_plans_dir / failure.plan_folder.name
                        break

                # Fallback to old behavior if magmathor not found in path
                if backup_folder is None:
                    backup_rep_dir = rep_dir.parent / f"{rep_dir.name}_Removed"
                    backup_plans_dir = backup_rep_dir / "Plans"
                    backup_folder = backup_plans_dir / failure.plan_folder.name

                try:
                    backup_plans_dir.mkdir(parents=True, exist_ok=True)

                    # If backup already exists, remove it first
                    if backup_folder.exists():
                        shutil.rmtree(backup_folder)

                    shutil.move(str(failure.plan_folder), str(backup_folder))
                    print(f"Moved {failure.plan_folder} -> {backup_folder}")
                    success_count += 1

                except Exception as e:
                    err_msg = f"Failed to move {failure.plan_folder}: {e}"
                    errors.append(err_msg)
                    if error_callback:
                        error_callback(err_msg)
            else:
                # No folder to move, but entry was removed from JSON
                success_count += 1

    return success_count, errors


class RemoveAPIFailuresApp:
    """Tkinter application for removing API failure test results."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Remove API Failures")
        self.root.geometry("900x700")

        self.failures: List[APIFailureEntry] = []
        self.is_searching = False
        self.is_removing = False

        self._create_widgets()

    def _create_widgets(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Directory input section
        dir_frame = ttk.LabelFrame(main_frame, text="Search Directory", padding="5")
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Directory:").grid(
            row=0, column=0, sticky="w", padx=(0, 5)
        )
        self.dir_entry = ttk.Entry(dir_frame)
        self.dir_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.dir_entry.insert(0, DEFAULT_BASE_PATH)

        ttk.Label(dir_frame, text="Model Filter:").grid(
            row=1, column=0, sticky="w", padx=(0, 5), pady=(5, 0)
        )
        self.model_filter_entry = ttk.Entry(dir_frame)
        self.model_filter_entry.grid(
            row=1, column=1, sticky="ew", padx=(0, 5), pady=(5, 0)
        )
        self.model_filter_entry.insert(0, "")

        filter_hint = ttk.Label(
            dir_frame,
            text="(e.g., google__gemini-3-pro-preview-high)",
            foreground="gray",
        )
        filter_hint.grid(row=2, column=1, sticky="w", pady=(0, 5))

        self.search_btn = ttk.Button(
            dir_frame, text="Search", command=self._start_search
        )
        self.search_btn.grid(row=0, column=2, rowspan=2, padx=(5, 0))

        # Status label
        self.status_label = ttk.Label(
            main_frame, text="Enter a directory and click Search"
        )
        self.status_label.grid(row=1, column=0, sticky="w", pady=(0, 5))

        # Results section
        results_frame = ttk.LabelFrame(
            main_frame, text="API Failures Found", padding="5"
        )
        results_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Listbox with scrollbar
        list_frame = ttk.Frame(results_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        self.results_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, font=("Courier", 10)
        )
        self.results_listbox.grid(row=0, column=0, sticky="nsew")

        scrollbar_y = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.results_listbox.yview
        )
        scrollbar_y.grid(row=0, column=1, sticky="ns")
        self.results_listbox.configure(yscrollcommand=scrollbar_y.set)

        scrollbar_x = ttk.Scrollbar(
            list_frame, orient="horizontal", command=self.results_listbox.xview
        )
        scrollbar_x.grid(row=1, column=0, sticky="ew")
        self.results_listbox.configure(xscrollcommand=scrollbar_x.set)

        # Count label
        self.count_label = ttk.Label(results_frame, text="")
        self.count_label.grid(row=1, column=0, sticky="w", pady=(5, 0))

        # Progress section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            progress_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew")

        self.progress_label = ttk.Label(progress_frame, text="")
        self.progress_label.grid(row=1, column=0, sticky="w")

        # Errors section
        errors_frame = ttk.LabelFrame(main_frame, text="Errors", padding="5")
        errors_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        errors_frame.columnconfigure(0, weight=1)

        self.errors_text = tk.Text(
            errors_frame, height=4, state="disabled", wrap="word"
        )
        self.errors_text.grid(row=0, column=0, sticky="ew")

        errors_scrollbar = ttk.Scrollbar(
            errors_frame, orient="vertical", command=self.errors_text.yview
        )
        errors_scrollbar.grid(row=0, column=1, sticky="ns")
        self.errors_text.configure(yscrollcommand=errors_scrollbar.set)

        # Buttons section
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=5, column=0, sticky="e")

        self.remove_btn = ttk.Button(
            btn_frame,
            text="Move Selected Failures",
            command=self._start_remove,
            state="disabled",
        )
        self.remove_btn.grid(row=0, column=0, padx=(0, 5))

        self.cancel_btn = ttk.Button(btn_frame, text="Close", command=self.root.quit)
        self.cancel_btn.grid(row=0, column=1)

    def _start_search(self):
        """Start searching for API failures."""
        if self.is_searching or self.is_removing:
            return

        base_path = self.dir_entry.get().strip()
        if not base_path:
            messagebox.showerror("Error", "Please enter a directory path")
            return

        if not os.path.isdir(base_path):
            messagebox.showerror("Error", f"Directory not found: {base_path}")
            return

        model_filter = self.model_filter_entry.get().strip()

        self.is_searching = True
        self.search_btn.configure(state="disabled")
        self.remove_btn.configure(state="disabled")
        self.results_listbox.delete(0, tk.END)
        self.failures = []
        self._clear_errors()

        self.status_label.configure(text="Searching for --rep directories...")
        self.progress_label.configure(text="Scanning directories...")
        self.root.update()

        # Find rep directories with progress feedback
        base = Path(base_path)

        def dir_progress_callback(message: str):
            self.progress_label.configure(text=message)
            self.root.update()

        rep_dirs = find_rep_directories(base, progress_callback=dir_progress_callback)

        # Apply model filter if specified
        if model_filter:
            rep_dirs = [d for d in rep_dirs if model_filter in d.name]
            self.status_label.configure(
                text=f"Found {len(rep_dirs)} --rep directories matching '{model_filter}'. Scanning for API failures..."
            )
        else:
            self.status_label.configure(
                text=f"Found {len(rep_dirs)} --rep directories. Scanning for API failures..."
            )
        self.root.update()

        # Find API failures in each
        total_dirs = len(rep_dirs)
        for i, rep_dir in enumerate(rep_dirs):
            self.progress_var.set((i + 1) / total_dirs * 100 if total_dirs > 0 else 0)
            self.progress_label.configure(
                text=f"[{i + 1}/{total_dirs}] Scanning {rep_dir.name}... ({len(self.failures)} failures found)"
            )
            self.root.update()

            failures = find_api_failures(rep_dir)
            self.failures.extend(failures)

        # Populate listbox
        self.progress_label.configure(text="Populating results list...")
        self.root.update()

        for failure in self.failures:
            # Show task name with rep directory for context
            rel_rep = (
                failure.rep_dir.relative_to(base)
                if failure.rep_dir.is_relative_to(base)
                else failure.rep_dir
            )
            display = f"{failure.task_name}  [{rel_rep}]"
            self.results_listbox.insert(tk.END, display)

        self.count_label.configure(
            text=f"Total: {len(self.failures)} API failures found"
        )
        self.status_label.configure(
            text=f"Search complete. Found {len(self.failures)} API failures."
        )
        self.progress_var.set(0)
        self.progress_label.configure(text="")

        self.is_searching = False
        self.search_btn.configure(state="normal")

        if self.failures:
            self.remove_btn.configure(state="normal")

    def _start_remove(self):
        """Start moving API failures to backup."""
        if self.is_removing or not self.failures:
            return

        # Determine destination path for display
        dest_path = "(backup location)"
        if self.failures:
            sample_dir = self.failures[0].rep_dir
            parts = sample_dir.parts
            for i, part in enumerate(parts):
                if part == "magmathor" and i + 1 < len(parts):
                    test_root_name = parts[i + 1]
                    dest_path = f"/mnt/magmathor/{test_root_name}_API_Failures/"
                    break

        # Confirm with user
        count = len(self.failures)
        if not messagebox.askyesno(
            "Confirm Move",
            f"This will:\n"
            f"• Move {count} plan folder(s) to:\n"
            f"  {dest_path}\n"
            f"• Remove {count} entries from test_results.json files\n\n"
            f"Continue?",
        ):
            return

        self.is_removing = True
        self.search_btn.configure(state="disabled")
        self.remove_btn.configure(state="disabled")
        self._clear_errors()

        self.status_label.configure(text="Moving API failures to backup...")
        self.root.update()

        def progress_callback(current, total, message):
            self.progress_var.set(current / total * 100)
            self.progress_label.configure(text=f"[{current}/{total}] {message}")
            self.root.update()
            return True

        def error_callback(error_msg):
            self._add_error(error_msg)
            self.root.update()

        success_count, errors = remove_api_failures(
            self.failures,
            progress_callback=progress_callback,
            error_callback=error_callback,
        )

        self.progress_var.set(0)
        self.progress_label.configure(text="")

        self.is_removing = False
        self.search_btn.configure(state="normal")

        # Clear the list since items have been processed
        self.results_listbox.delete(0, tk.END)
        self.failures = []
        self.count_label.configure(text="")

        if errors:
            self.status_label.configure(
                text=f"Completed with {len(errors)} error(s). {success_count} items moved."
            )
        else:
            self.status_label.configure(
                text=f"Successfully moved {success_count} API failure(s)."
            )
            messagebox.showinfo(
                "Complete", f"Successfully moved {success_count} API failure(s)."
            )

    def _add_error(self, error_msg: str):
        """Add an error message to the errors text widget."""
        self.errors_text.configure(state="normal")
        self.errors_text.insert(tk.END, error_msg + "\n")
        self.errors_text.see(tk.END)
        self.errors_text.configure(state="disabled")

    def _clear_errors(self):
        """Clear the errors text widget."""
        self.errors_text.configure(state="normal")
        self.errors_text.delete(1.0, tk.END)
        self.errors_text.configure(state="disabled")


def main():
    root = tk.Tk()
    app = RemoveAPIFailuresApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
