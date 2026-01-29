#!/usr/bin/env python3
"""Script to search for retry-related warnings in Qwen/GLM job logs."""

import os
import re
from pathlib import Path

# Base directory for test results
TEST_BASE = "/home/andreatupini/mnt/magmardata_magmathor/20260115_Test"

# Model name filters (case-insensitive substring match)
MODEL_FILTERS = ["qwen", "glm", "google", "anthropic"]


def discover_log_files() -> list[str]:
    """Discover all qwen/glm log files by listing benchmark directories."""
    log_files = []

    # List benchmark directories (p1-p6)
    try:
        benchmark_dirs = sorted(
            [
                d
                for d in os.listdir(TEST_BASE)
                if d.startswith("magt_benchmark_p")
                and os.path.isdir(os.path.join(TEST_BASE, d))
            ]
        )
    except OSError as e:
        print(f"Error listing {TEST_BASE}: {e}")
        return []

    for benchmark in benchmark_dirs:
        benchmark_path = os.path.join(TEST_BASE, benchmark)
        try:
            # List run directories in this benchmark
            run_dirs = os.listdir(benchmark_path)
        except OSError as e:
            print(f"Error listing {benchmark_path}: {e}")
            continue

        for run_dir in run_dirs:
            # Check if this is a qwen or glm run
            run_lower = run_dir.lower()
            if any(model in run_lower for model in MODEL_FILTERS):
                log_path = os.path.join(benchmark_path, run_dir, "logs.txt")
                log_files.append(log_path)

    return sorted(log_files)


# Patterns from openrouter_actor.py retry logic
RETRY_PATTERNS = [
    r"Rate limited \(429\)",
    r"Transient API error",
    r"Policy violation.*retrying",
    r"Model returned empty response",
    r"no 'choices'",
    r"invalid 'message' structure",
    r"malformed response.*retrying",
    r"Timeout error.*retrying",
    r"Connection error.*retrying",
    r"Chunked encoding error",
    r"JSON decode error.*retrying",
    r"Unexpected error during API request",
    r"ModelEmptyResponseError",
    r"Model produced no output",
]

# Compile patterns for efficiency
COMBINED_PATTERN = re.compile("|".join(RETRY_PATTERNS), re.IGNORECASE)


def extract_run_info(path: str) -> tuple[str, str]:
    """Extract benchmark (p1, p2, etc.) and run name from path."""
    benchmark_match = re.search(r"magt_benchmark_p(\d)", path)
    benchmark = f"p{benchmark_match.group(1)}" if benchmark_match else "unknown"

    # Extract run directory name
    parts = path.split("/")
    run_name = parts[-2] if len(parts) >= 2 else "unknown"

    # Shorten run name: extract model and config
    # e.g. "qwen__qwen3-vl-235b-a22b-thinking--T0_Fs_H60_C0_P2_I1_R1_S1_E0_M16384--rep1"
    # -> "qwen3-vl-235b (T0_Fs_H60...)"
    run_match = re.match(r"([^_]+)__([^-]+(?:-[^-]+)*?)--([^-]+)", run_name)
    if run_match:
        model = run_match.group(2)
        config = run_match.group(3)
        short_name = f"{model} ({config})"
    else:
        short_name = run_name[:50]

    return benchmark, short_name


def check_log_file(log_path: str) -> dict:
    """Check a log file for retry warnings."""
    benchmark, run_name = extract_run_info(log_path)
    result = {
        "path": log_path,
        "benchmark": benchmark,
        "run_name": run_name,
        "exists": False,
        "lines": 0,
        "matches": [],
        "last_line": "",
    }

    if not os.path.exists(log_path):
        return result

    result["exists"] = True

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    result["lines"] = len(lines)
    if lines:
        result["last_line"] = lines[-1].strip()[:100]

    for i, line in enumerate(lines, 1):
        if COMBINED_PATTERN.search(line):
            result["matches"].append((i, line.strip()[:200]))

    return result


def main():
    print("=" * 70)
    print("Searching for OpenRouter retry warnings in Qwen/GLM logs")
    print("=" * 70)
    print()

    # Discover all log files dynamically
    log_files = discover_log_files()
    print(f"Found {len(log_files)} qwen/glm log files to check")
    print()

    total_matches = 0
    files_with_warnings = 0
    files_not_found = 0

    for log_path in log_files:
        result = check_log_file(log_path)
        benchmark = result["benchmark"]
        run_name = result["run_name"]

        if not result["exists"]:
            files_not_found += 1
            # Only show if you want to see missing files
            # print(f"[{benchmark}] {run_name} - File not found")
            continue

        match_count = len(result["matches"])
        total_matches += match_count

        # Only show files with warnings or very short logs (may indicate issues)
        if match_count > 0:
            files_with_warnings += 1
            status = f"⚠ {match_count} warnings"
            print(f"[{benchmark}] {run_name}")
            print(f"         {result['lines']:,} lines | {status}")
            print(f"         Last: {result['last_line']}...")
            print("         Matches:")
            for line_num, text in result["matches"][:5]:  # Show first 5
                print(f"           L{line_num}: {text}")
            if len(result["matches"]) > 5:
                print(f"           ... and {len(result['matches']) - 5} more")
            print()

    print("=" * 70)
    print(
        f"Checked: {len(log_files) - files_not_found} files ({files_not_found} not found/empty)"
    )
    if total_matches == 0:
        print("✓ No retry warnings found across all log files!")
    else:
        print(
            f"⚠ Found {total_matches} total retry warnings in {files_with_warnings} files"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
