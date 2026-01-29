#!/usr/bin/env python3
"""
Script to verify that for every plan mentioned in test_results.json files,
the corresponding plan directory exists in the Plans folder.
"""

import argparse
import json
import os
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List, Optional, Tuple


@dataclass
class MissingPlan:
    """Represents a missing plan directory."""

    test_results_path: str
    task_name: str
    expected_plan_path: str
    has_underscore_prefix: bool = False  # True if directory exists with _ prefix


@dataclass
class ScanResult:
    """Result of scanning a single test_results.json file."""

    test_results_path: str
    total_plans: int
    missing_plans: List[MissingPlan]
    prefixed_plans: List[MissingPlan]  # Plans that exist with _ prefix
    missing_plan_json: List[MissingPlan] = field(
        default_factory=list
    )  # Dirs exist but no plan.json
    error: str = None


class ProgressUI:
    """Simple terminal-based progress UI."""

    def __init__(self):
        self.start_time = time.time()
        self.files_scanned = 0
        self.total_files = 0
        self.missing_count = 0
        self.prefixed_count = 0
        self.missing_json_count = 0
        self.current_file = ""
        self.errors = []

    def set_total(self, total: int):
        self.total_files = total

    def update(
        self,
        current_file: str,
        missing: int = 0,
        prefixed: int = 0,
        missing_json: int = 0,
    ):
        self.files_scanned += 1
        self.current_file = current_file
        self.missing_count += missing
        self.prefixed_count += prefixed
        self.missing_json_count += missing_json
        self._render()

    def add_error(self, error: str):
        self.errors.append(error)

    def _render(self):
        elapsed = time.time() - self.start_time
        if self.files_scanned > 0:
            rate = self.files_scanned / elapsed
            eta = (self.total_files - self.files_scanned) / rate if rate > 0 else 0
        else:
            eta = 0

        # Calculate progress bar
        if self.total_files > 0:
            progress = self.files_scanned / self.total_files
            bar_width = 40
            filled = int(bar_width * progress)
            bar = "█" * filled + "░" * (bar_width - filled)
        else:
            bar = "░" * 40
            progress = 0

        # Clear line and print progress
        status_line = f"\r[{bar}] {progress*100:5.1f}% | {self.files_scanned}/{self.total_files} | Missing: {self.missing_count} | NoPlanJson: {self.missing_json_count} | Prefixed: {self.prefixed_count} | ETA: {eta:.0f}s"

        # Truncate current file for display
        max_path_len = 60
        display_path = self.current_file
        if len(display_path) > max_path_len:
            display_path = "..." + display_path[-(max_path_len - 3) :]

        print(status_line, end="", flush=True)

    def finish(self):
        elapsed = time.time() - self.start_time
        print()  # New line after progress bar
        print(f"\n{'='*70}")
        print(f"Scan Complete!")
        print(f"{'='*70}")
        print(f"  Files scanned:      {self.files_scanned}")
        print(f"  Missing plans:      {self.missing_count}")
        print(f"  Missing plan.json:  {self.missing_json_count}")
        print(f"  _Prefixed plans:    {self.prefixed_count}")
        print(f"  Time elapsed:       {elapsed:.1f}s")
        if self.errors:
            print(f"  Errors:             {len(self.errors)}")


def find_test_results_files(root_dir: str, progress: ProgressUI) -> List[str]:
    """Find all test_results.json files under the root directory."""
    print(f"Scanning for test_results.json files in: {root_dir}")
    test_results_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip Plans directories - they don't contain test_results.json
        if "Plans" in dirnames:
            dirnames.remove("Plans")

        if "test_results.json" in filenames:
            test_results_files.append(os.path.join(dirpath, "test_results.json"))
            # Don't recurse into subdirectories once we find test_results.json
            dirnames.clear()

    print(f"Found {len(test_results_files)} test_results.json files")
    return test_results_files


def check_test_results_file(test_results_path: str) -> ScanResult:
    """Check a single test_results.json file for missing plan directories."""
    try:
        with open(test_results_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return ScanResult(
            test_results_path=test_results_path,
            total_plans=0,
            missing_plans=[],
            prefixed_plans=[],
            error=f"JSON parse error: {e}",
        )
    except Exception as e:
        return ScanResult(
            test_results_path=test_results_path,
            total_plans=0,
            missing_plans=[],
            prefixed_plans=[],
            error=str(e),
        )

    # Get the Plans directory path
    parent_dir = os.path.dirname(test_results_path)
    plans_dir = os.path.join(parent_dir, "Plans")

    test_results = data.get("test_results", [])
    missing_plans = []
    prefixed_plans = []
    missing_plan_json = []

    for result in test_results:
        task_name = result.get("task_name")
        if not task_name:
            continue

        # Check if the plan directory exists
        plan_path = os.path.join(plans_dir, task_name)
        if not os.path.isdir(plan_path):
            # Check if it exists with underscore prefix
            prefixed_plan_path = os.path.join(plans_dir, "_" + task_name)
            if os.path.isdir(prefixed_plan_path):
                # Check if plan.json exists in prefixed directory
                if not os.path.isfile(os.path.join(prefixed_plan_path, "plan.json")):
                    missing_plan_json.append(
                        MissingPlan(
                            test_results_path=test_results_path,
                            task_name="_" + task_name,
                            expected_plan_path=prefixed_plan_path,
                            has_underscore_prefix=True,
                        )
                    )
                else:
                    prefixed_plans.append(
                        MissingPlan(
                            test_results_path=test_results_path,
                            task_name=task_name,
                            expected_plan_path=plan_path,
                            has_underscore_prefix=True,
                        )
                    )
            else:
                missing_plans.append(
                    MissingPlan(
                        test_results_path=test_results_path,
                        task_name=task_name,
                        expected_plan_path=plan_path,
                        has_underscore_prefix=False,
                    )
                )
        else:
            # Directory exists, check if plan.json is present
            if not os.path.isfile(os.path.join(plan_path, "plan.json")):
                missing_plan_json.append(
                    MissingPlan(
                        test_results_path=test_results_path,
                        task_name=task_name,
                        expected_plan_path=plan_path,
                        has_underscore_prefix=False,
                    )
                )

    return ScanResult(
        test_results_path=test_results_path,
        total_plans=len(test_results),
        missing_plans=missing_plans,
        prefixed_plans=prefixed_plans,
        missing_plan_json=missing_plan_json,
    )


def scan_directories(
    root_dir: str, max_workers: int = 8
) -> Tuple[List[ScanResult], ProgressUI]:
    """Scan all test_results.json files and check for missing plan directories."""
    progress = ProgressUI()

    # Find all test_results.json files
    test_results_files = find_test_results_files(root_dir, progress)
    progress.set_total(len(test_results_files))

    all_results = []

    print(f"\nVerifying plan directories...")
    print()

    # Process files with thread pool for I/O parallelism
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(check_test_results_file, f): f for f in test_results_files
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                all_results.append(result)

                if result.error:
                    progress.add_error(f"{file_path}: {result.error}")

                progress.update(
                    file_path,
                    len(result.missing_plans),
                    len(result.prefixed_plans),
                    len(result.missing_plan_json),
                )

            except Exception as e:
                progress.add_error(f"{file_path}: {e}")
                progress.update(file_path)

    progress.finish()
    return all_results, progress


def print_detailed_report(results: List[ScanResult], output_file: str = None):
    """Print a detailed report of missing plans."""
    missing_plans = []
    prefixed_plans = []
    missing_plan_json = []
    errors = []
    total_plans_checked = 0

    for result in results:
        total_plans_checked += result.total_plans
        missing_plans.extend(result.missing_plans)
        prefixed_plans.extend(result.prefixed_plans)
        missing_plan_json.extend(result.missing_plan_json)
        if result.error:
            errors.append((result.test_results_path, result.error))

    output_lines = []

    output_lines.append("\n" + "=" * 70)
    output_lines.append("DETAILED REPORT")
    output_lines.append("=" * 70)
    output_lines.append(f"\nTotal plans checked: {total_plans_checked}")
    output_lines.append(f"Missing plan directories: {len(missing_plans)}")
    output_lines.append(f"Missing plan.json files: {len(missing_plan_json)}")
    output_lines.append(f"Plans with _ prefix: {len(prefixed_plans)}")

    if errors:
        output_lines.append(f"\n{'='*70}")
        output_lines.append("ERRORS")
        output_lines.append("=" * 70)
        for path, error in errors:
            output_lines.append(f"\n  File: {path}")
            output_lines.append(f"  Error: {error}")

    if prefixed_plans:
        output_lines.append(f"\n{'='*70}")
        output_lines.append("PLANS WITH _ PREFIX (exist but renamed)")
        output_lines.append("=" * 70)

        # Group by test_results.json file
        by_file: Dict[str, List[MissingPlan]] = {}
        for mp in prefixed_plans:
            if mp.test_results_path not in by_file:
                by_file[mp.test_results_path] = []
            by_file[mp.test_results_path].append(mp)

        for test_results_path, plans in sorted(by_file.items()):
            output_lines.append(f"\n  Test Results: {test_results_path}")
            output_lines.append(f"  Prefixed Plans ({len(plans)}):")
            for plan in plans:
                output_lines.append(f"    - {plan.task_name}")

    if missing_plan_json:
        output_lines.append(f"\n{'='*70}")
        output_lines.append("DIRECTORIES MISSING plan.json FILE")
        output_lines.append("=" * 70)

        # Group by test_results.json file
        by_file: Dict[str, List[MissingPlan]] = {}
        for mp in missing_plan_json:
            if mp.test_results_path not in by_file:
                by_file[mp.test_results_path] = []
            by_file[mp.test_results_path].append(mp)

        for test_results_path, plans in sorted(by_file.items()):
            output_lines.append(f"\n  Test Results: {test_results_path}")
            output_lines.append(f"  Missing plan.json ({len(plans)}):")
            for plan in plans:
                output_lines.append(f"    - {plan.task_name}")

    if missing_plans:
        output_lines.append(f"\n{'='*70}")
        output_lines.append("TRULY MISSING PLAN DIRECTORIES")
        output_lines.append("=" * 70)

        # Group by test_results.json file
        by_file: Dict[str, List[MissingPlan]] = {}
        for mp in missing_plans:
            if mp.test_results_path not in by_file:
                by_file[mp.test_results_path] = []
            by_file[mp.test_results_path].append(mp)

        for test_results_path, plans in sorted(by_file.items()):
            output_lines.append(f"\n  Test Results: {test_results_path}")
            output_lines.append(f"  Missing Plans ({len(plans)}):")
            for plan in plans:
                output_lines.append(f"    - {plan.task_name}")

    if not missing_plans and not missing_plan_json:
        output_lines.append(
            "\n✓ All plan directories exist with plan.json (or have _ prefix)!"
        )

    # Print to console
    for line in output_lines:
        print(line)

    # Write to file if specified
    if output_file:
        with open(output_file, "w") as f:
            f.write("\n".join(output_lines))
        print(f"\nReport saved to: {output_file}")


class VerifyPlanDirectoriesGUI:
    """GUI for verifying plan directories."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Verify Plan Directories")
        self.root.geometry("900x700")

        self.scanning = False
        self.results: List[ScanResult] = []

        self._create_widgets()

    def _create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # Directory selection frame
        dir_frame = ttk.LabelFrame(main_frame, text="Directory", padding="5")
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Root Directory:").grid(row=0, column=0, padx=(0, 5))
        self.dir_var = tk.StringVar(value="Test")
        self.dir_entry = ttk.Entry(dir_frame, textvariable=self.dir_var)
        self.dir_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.browse_btn = ttk.Button(
            dir_frame, text="Browse...", command=self._browse_directory
        )
        self.browse_btn.grid(row=0, column=2)

        # Filters frame
        filter_frame = ttk.LabelFrame(
            main_frame,
            text="Filters (case-insensitive, leave blank for all)",
            padding="5",
        )
        filter_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        filter_frame.columnconfigure(1, weight=1)
        filter_frame.columnconfigure(3, weight=1)

        ttk.Label(filter_frame, text="Model:").grid(row=0, column=0, padx=(0, 5))
        self.model_filter_var = tk.StringVar()
        self.model_filter_entry = ttk.Entry(
            filter_frame, textvariable=self.model_filter_var
        )
        self.model_filter_entry.grid(row=0, column=1, sticky="ew", padx=(0, 20))

        ttk.Label(filter_frame, text="Config:").grid(row=0, column=2, padx=(0, 5))
        self.config_filter_var = tk.StringVar()
        self.config_filter_entry = ttk.Entry(
            filter_frame, textvariable=self.config_filter_var
        )
        self.config_filter_entry.grid(row=0, column=3, sticky="ew")

        # Example labels
        ttk.Label(
            filter_frame, text="e.g. gemini, gpt-4o, glm", foreground="gray"
        ).grid(row=1, column=1, sticky="w")
        ttk.Label(
            filter_frame, text="e.g. T1_Fs_H00_C0_P2_I0_R1_S1_E0", foreground="gray"
        ).grid(row=1, column=3, sticky="w")

        # Controls frame
        ctrl_frame = ttk.Frame(main_frame)
        ctrl_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        self.scan_btn = ttk.Button(
            ctrl_frame, text="Start Scan", command=self._start_scan
        )
        self.scan_btn.pack(side="left")

        self.stop_btn = ttk.Button(
            ctrl_frame, text="Stop", command=self._stop_scan, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        ttk.Label(ctrl_frame, text="Workers:").pack(side="left", padx=(20, 5))
        self.workers_var = tk.IntVar(value=8)
        self.workers_spin = ttk.Spinbox(
            ctrl_frame, from_=1, to=32, textvariable=self.workers_var, width=5
        )
        self.workers_spin.pack(side="left")

        # Progress frame
        prog_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        prog_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        prog_frame.columnconfigure(0, weight=1)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            prog_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", pady=(0, 5))

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(prog_frame, textvariable=self.status_var)
        self.status_label.grid(row=1, column=0, sticky="w")

        # Stats frame
        stats_frame = ttk.Frame(prog_frame)
        stats_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))

        self.files_var = tk.StringVar(value="Files: 0/0")
        ttk.Label(stats_frame, textvariable=self.files_var).pack(
            side="left", padx=(0, 20)
        )

        self.missing_var = tk.StringVar(value="Missing: 0")
        self.missing_label = ttk.Label(stats_frame, textvariable=self.missing_var)
        self.missing_label.pack(side="left", padx=(0, 20))

        self.missing_json_var = tk.StringVar(value="No plan.json: 0")
        self.missing_json_label = ttk.Label(
            stats_frame, textvariable=self.missing_json_var
        )
        self.missing_json_label.pack(side="left", padx=(0, 20))

        self.prefixed_var = tk.StringVar(value="Prefixed: 0")
        ttk.Label(stats_frame, textvariable=self.prefixed_var).pack(
            side="left", padx=(0, 20)
        )

        self.time_var = tk.StringVar(value="Time: 0.0s")
        ttk.Label(stats_frame, textvariable=self.time_var).pack(side="left")

        # Results notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=4, column=0, sticky="nsew", pady=(0, 10))
        main_frame.rowconfigure(4, weight=1)

        # Missing plans tab
        missing_frame = ttk.Frame(self.notebook)
        self.notebook.add(missing_frame, text="Missing Plans (0)")
        missing_frame.columnconfigure(0, weight=1)
        missing_frame.rowconfigure(0, weight=1)

        self.missing_tree = ttk.Treeview(
            missing_frame, columns=("task", "path"), show="headings"
        )
        self.missing_tree.heading("task", text="Task Name")
        self.missing_tree.heading("path", text="Test Results Path")
        self.missing_tree.column("task", width=300)
        self.missing_tree.column("path", width=500)
        self.missing_tree.grid(row=0, column=0, sticky="nsew")

        missing_scroll = ttk.Scrollbar(
            missing_frame, orient="vertical", command=self.missing_tree.yview
        )
        missing_scroll.grid(row=0, column=1, sticky="ns")
        self.missing_tree.configure(yscrollcommand=missing_scroll.set)

        # Missing plan.json tab
        missing_json_frame = ttk.Frame(self.notebook)
        self.notebook.add(missing_json_frame, text="No plan.json (0)")
        missing_json_frame.columnconfigure(0, weight=1)
        missing_json_frame.rowconfigure(0, weight=1)

        self.missing_json_tree = ttk.Treeview(
            missing_json_frame, columns=("task", "path"), show="headings"
        )
        self.missing_json_tree.heading("task", text="Task Name")
        self.missing_json_tree.heading("path", text="Test Results Path")
        self.missing_json_tree.column("task", width=300)
        self.missing_json_tree.column("path", width=500)
        self.missing_json_tree.grid(row=0, column=0, sticky="nsew")

        missing_json_scroll = ttk.Scrollbar(
            missing_json_frame, orient="vertical", command=self.missing_json_tree.yview
        )
        missing_json_scroll.grid(row=0, column=1, sticky="ns")
        self.missing_json_tree.configure(yscrollcommand=missing_json_scroll.set)

        # Prefixed plans tab
        prefixed_frame = ttk.Frame(self.notebook)
        self.notebook.add(prefixed_frame, text="Prefixed Plans (0)")
        prefixed_frame.columnconfigure(0, weight=1)
        prefixed_frame.rowconfigure(0, weight=1)

        self.prefixed_tree = ttk.Treeview(
            prefixed_frame, columns=("task", "path"), show="headings"
        )
        self.prefixed_tree.heading("task", text="Task Name")
        self.prefixed_tree.heading("path", text="Test Results Path")
        self.prefixed_tree.column("task", width=300)
        self.prefixed_tree.column("path", width=500)
        self.prefixed_tree.grid(row=0, column=0, sticky="nsew")

        prefixed_scroll = ttk.Scrollbar(
            prefixed_frame, orient="vertical", command=self.prefixed_tree.yview
        )
        prefixed_scroll.grid(row=0, column=1, sticky="ns")
        self.prefixed_tree.configure(yscrollcommand=prefixed_scroll.set)

        # Errors tab
        errors_frame = ttk.Frame(self.notebook)
        self.notebook.add(errors_frame, text="Errors (0)")
        errors_frame.columnconfigure(0, weight=1)
        errors_frame.rowconfigure(0, weight=1)

        self.errors_text = scrolledtext.ScrolledText(errors_frame, height=10)
        self.errors_text.grid(row=0, column=0, sticky="nsew")

        # Export button
        export_frame = ttk.Frame(main_frame)
        export_frame.grid(row=5, column=0, sticky="e")

        self.export_btn = ttk.Button(
            export_frame,
            text="Export to JSON...",
            command=self._export_json,
            state="disabled",
        )
        self.export_btn.pack(side="right")

    def _browse_directory(self):
        directory = filedialog.askdirectory(initialdir=self.dir_var.get())
        if directory:
            self.dir_var.set(directory)

    def _matches_filters(self, path: str) -> bool:
        """Check if a path matches the model and config filters."""
        model_filter = self.model_filter_var.get().strip().lower()
        config_filter = self.config_filter_var.get().strip().lower()

        # If no filters, match everything
        if not model_filter and not config_filter:
            return True

        path_lower = path.lower()

        # Check model filter
        if model_filter and model_filter not in path_lower:
            return False

        # Check config filter
        if config_filter and config_filter not in path_lower:
            return False

        return True

    def _start_scan(self):
        root_dir = self.dir_var.get()
        if not os.path.isdir(root_dir):
            messagebox.showerror("Error", f"'{root_dir}' is not a valid directory")
            return

        self.scanning = True
        self.scan_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.export_btn.configure(state="disabled")

        # Set progress bar to indeterminate mode during discovery
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(10)

        # Clear previous results
        for item in self.missing_tree.get_children():
            self.missing_tree.delete(item)
        for item in self.missing_json_tree.get_children():
            self.missing_json_tree.delete(item)
        for item in self.prefixed_tree.get_children():
            self.prefixed_tree.delete(item)
        self.errors_text.delete("1.0", tk.END)
        self.results = []

        # Start scan in background thread
        thread = threading.Thread(target=self._run_scan, args=(root_dir,), daemon=True)
        thread.start()

    def _stop_scan(self):
        self.scanning = False

    def _run_scan(self, root_dir: str):
        start_time = time.time()

        # Get filter values (capture them now to avoid thread issues)
        model_filter = self.model_filter_var.get().strip().lower()
        config_filter = self.config_filter_var.get().strip().lower()

        filter_desc = []
        if model_filter:
            filter_desc.append(f"model='{model_filter}'")
        if config_filter:
            filter_desc.append(f"config='{config_filter}'")
        filter_str = f" (filters: {', '.join(filter_desc)})" if filter_desc else ""

        # Find all test_results.json files
        self._update_status(f"Scanning for test_results.json files{filter_str}...")
        test_results_files = []
        dirs_scanned = 0
        last_update_time = time.time()

        for dirpath, dirnames, filenames in os.walk(root_dir):
            if not self.scanning:
                break

            # Always skip Plans directories
            if "Plans" in dirnames:
                dirnames.remove("Plans")

            dirs_scanned += 1
            # Update status every 0.5 seconds to show we're still working
            current_time = time.time()
            if current_time - last_update_time > 0.5:
                elapsed = current_time - start_time
                self._update_status(
                    f"Scanning directories... {dirs_scanned} dirs, {len(test_results_files)} files found ({elapsed:.1f}s){filter_str}"
                )
                last_update_time = current_time

            if "test_results.json" in filenames:
                full_path = os.path.join(dirpath, "test_results.json")
                # Apply filters
                path_lower = full_path.lower()
                if model_filter and model_filter not in path_lower:
                    # Don't recurse further even if filtered out
                    dirnames.clear()
                    continue
                if config_filter and config_filter not in path_lower:
                    # Don't recurse further even if filtered out
                    dirnames.clear()
                    continue
                test_results_files.append(full_path)
                # Don't recurse into subdirectories once we find test_results.json
                dirnames.clear()

        if not self.scanning:
            self._scan_finished(cancelled=True)
            return

        total_files = len(test_results_files)
        self._update_status(
            f"Found {total_files} matching test_results.json files{filter_str}. Verifying..."
        )

        # Switch progress bar to determinate mode
        self.root.after(0, lambda: self._switch_to_determinate_progress())

        files_scanned = 0
        total_missing = 0
        total_missing_json = 0
        total_prefixed = 0
        errors = []

        max_workers = self.workers_var.get()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(check_test_results_file, f): f
                for f in test_results_files
            }

            for future in as_completed(future_to_file):
                if not self.scanning:
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                file_path = future_to_file[future]
                try:
                    result = future.result()
                    self.results.append(result)

                    if result.error:
                        errors.append(f"{file_path}: {result.error}")

                    # Add to trees
                    for mp in result.missing_plans:
                        self.root.after(
                            0,
                            lambda t=mp.task_name, p=mp.test_results_path: self.missing_tree.insert(
                                "", "end", values=(t, p)
                            ),
                        )
                    for mp in result.missing_plan_json:
                        self.root.after(
                            0,
                            lambda t=mp.task_name, p=mp.test_results_path: self.missing_json_tree.insert(
                                "", "end", values=(t, p)
                            ),
                        )
                    for mp in result.prefixed_plans:
                        self.root.after(
                            0,
                            lambda t=mp.task_name, p=mp.test_results_path: self.prefixed_tree.insert(
                                "", "end", values=(t, p)
                            ),
                        )

                    total_missing += len(result.missing_plans)
                    total_missing_json += len(result.missing_plan_json)
                    total_prefixed += len(result.prefixed_plans)

                except Exception as e:
                    errors.append(f"{file_path}: {e}")

                files_scanned += 1
                elapsed = time.time() - start_time
                progress = (files_scanned / total_files) * 100 if total_files > 0 else 0

                self.root.after(
                    0,
                    lambda p=progress, f=files_scanned, t=total_files, m=total_missing, mj=total_missing_json, pf=total_prefixed, e=elapsed: self._update_progress(
                        p, f, t, m, mj, pf, e
                    ),
                )

        # Update errors
        if errors:
            self.root.after(
                0, lambda: self.errors_text.insert("1.0", "\n".join(errors))
            )

        self._scan_finished(cancelled=not self.scanning)

    def _update_status(self, status: str):
        self.root.after(0, lambda: self.status_var.set(status))

    def _update_progress(
        self,
        progress: float,
        files: int,
        total: int,
        missing: int,
        missing_json: int,
        prefixed: int,
        elapsed: float,
    ):
        self.progress_var.set(progress)
        self.files_var.set(f"Files: {files}/{total}")
        self.missing_var.set(f"Missing: {missing}")
        self.missing_json_var.set(f"No plan.json: {missing_json}")
        self.prefixed_var.set(f"Prefixed: {prefixed}")
        self.time_var.set(f"Time: {elapsed:.1f}s")

        # Update tab labels
        self.notebook.tab(0, text=f"Missing Plans ({missing})")
        self.notebook.tab(1, text=f"No plan.json ({missing_json})")
        self.notebook.tab(2, text=f"Prefixed Plans ({prefixed})")

        # Highlight missing if > 0
        if missing > 0:
            self.missing_label.configure(foreground="red")
        else:
            self.missing_label.configure(foreground="")

        # Highlight missing_json if > 0
        if missing_json > 0:
            self.missing_json_label.configure(foreground="orange")
        else:
            self.missing_json_label.configure(foreground="")

    def _switch_to_determinate_progress(self):
        """Switch progress bar from indeterminate to determinate mode."""
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.progress_var.set(0)

    def _scan_finished(self, cancelled: bool = False):
        self.scanning = False

        def finish():
            # Stop indeterminate animation if still running
            self.progress_bar.stop()
            self.progress_bar.configure(mode="determinate")

            self.scan_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

            if self.results:
                self.export_btn.configure(state="normal")

            # Update errors tab
            error_count = sum(1 for r in self.results if r.error)
            self.notebook.tab(3, text=f"Errors ({error_count})")

            if cancelled:
                self.status_var.set("Scan cancelled")
            else:
                total_missing = sum(len(r.missing_plans) for r in self.results)
                total_missing_json = sum(len(r.missing_plan_json) for r in self.results)
                total_prefixed = sum(len(r.prefixed_plans) for r in self.results)
                total_plans = sum(r.total_plans for r in self.results)

                if total_missing == 0 and total_missing_json == 0:
                    self.status_var.set(
                        f"✓ Complete! All {total_plans} plans verified ({total_prefixed} with _ prefix)"
                    )
                else:
                    issues = []
                    if total_missing > 0:
                        issues.append(f"{total_missing} missing dirs")
                    if total_missing_json > 0:
                        issues.append(f"{total_missing_json} missing plan.json")
                    self.status_var.set(
                        f"⚠ Complete! {', '.join(issues)} out of {total_plans} plans"
                    )

        self.root.after(0, finish)

    def _export_json(self):
        if not self.results:
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="missing_plans.json",
        )

        if not filepath:
            return

        missing_data = {
            "missing_plans": [],
            "missing_plan_json": [],
            "prefixed_plans": [],
        }

        for result in self.results:
            for mp in result.missing_plans:
                missing_data["missing_plans"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )
            for mp in result.missing_plan_json:
                missing_data["missing_plan_json"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )
            for mp in result.prefixed_plans:
                missing_data["prefixed_plans"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )

        try:
            with open(filepath, "w") as f:
                json.dump(missing_data, f, indent=2)
            messagebox.showinfo("Export Complete", f"Results exported to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export: {e}")


def run_gui():
    """Launch the GUI."""
    root = tk.Tk()
    app = VerifyPlanDirectoriesGUI(root)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Verify that plan directories exist for all test results."
    )
    parser.add_argument(
        "root_dir", nargs="?", help="Root directory to scan for test_results.json files"
    )
    parser.add_argument("-o", "--output", help="Output file for the report (optional)")
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads (default: 8)",
    )
    parser.add_argument("--json-output", help="Output missing plans to a JSON file")
    parser.add_argument("--gui", action="store_true", help="Launch GUI mode")

    args = parser.parse_args()

    # If no root_dir and no --gui, launch GUI by default
    if args.root_dir is None and not args.gui:
        run_gui()
        return

    if args.gui:
        run_gui()
        return

    if not os.path.isdir(args.root_dir):
        print(f"Error: {args.root_dir} is not a valid directory")
        sys.exit(1)

    results, progress = scan_directories(args.root_dir, args.workers)
    print_detailed_report(results, args.output)

    # Export to JSON if requested
    if args.json_output:
        missing_data = {
            "missing_plans": [],
            "missing_plan_json": [],
            "prefixed_plans": [],
        }
        for result in results:
            for mp in result.missing_plans:
                missing_data["missing_plans"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )
            for mp in result.missing_plan_json:
                missing_data["missing_plan_json"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )
            for mp in result.prefixed_plans:
                missing_data["prefixed_plans"].append(
                    {
                        "test_results_path": mp.test_results_path,
                        "task_name": mp.task_name,
                        "expected_plan_path": mp.expected_plan_path,
                    }
                )

        with open(args.json_output, "w") as f:
            json.dump(missing_data, f, indent=2)
        print(f"JSON report saved to: {args.json_output}")

    # Exit with error code if missing plans or missing plan.json found
    total_missing = sum(len(r.missing_plans) for r in results)
    total_missing_json = sum(len(r.missing_plan_json) for r in results)
    sys.exit(1 if (total_missing > 0 or total_missing_json > 0) else 0)


if __name__ == "__main__":
    main()
