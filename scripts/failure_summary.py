#!/usr/bin/env python3
"""Quick summary of failure patterns across all benchmarks."""

import json
from collections import Counter, defaultdict
from pathlib import Path

TEST_BASE = Path(__file__).parent.parent / "Test"
BENCHMARKS = [
    "magt_benchmark_p1",
    "magt_benchmark_p2",
    "magt_benchmark_p3",
    "magt_benchmark_p4",
    "magt_benchmark_p5",
    "magt_benchmark_p6",
]


def main():
    # Collect stats
    stats = defaultdict(
        lambda: {
            "pass": 0,
            "fail": 0,
            "api_fail": 0,
            "fail_reasons": Counter(),
            "total": 0,
            "configs": set(),
        }
    )
    overall = {
        "pass": 0,
        "fail": 0,
        "api_fail": 0,
        "total": 0,
        "fail_reasons": Counter(),
        "json_errors": 0,
    }
    per_benchmark = defaultdict(lambda: {"pass": 0, "fail": 0, "total": 0})

    for benchmark in BENCHMARKS:
        bdir = TEST_BASE / benchmark
        if not bdir.exists():
            continue
        for run_dir in sorted(bdir.iterdir()):
            if not run_dir.is_dir():
                continue
            results_file = run_dir / "test_results.json"
            if not results_file.exists():
                continue
            try:
                with open(results_file) as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                overall["json_errors"] += 1
                continue
            except IOError:
                continue

            # Parse model name and config from run_dir name
            parts = run_dir.name.split("--")
            model_name = parts[0]
            config = parts[1] if len(parts) > 1 else "unknown"

            stats[model_name]["configs"].add(config)

            for result in data.get("test_results", []):
                stats[model_name]["total"] += 1
                overall["total"] += 1
                per_benchmark[benchmark]["total"] += 1

                if result.get("passed"):
                    stats[model_name]["pass"] += 1
                    overall["pass"] += 1
                    per_benchmark[benchmark]["pass"] += 1
                else:
                    stats[model_name]["fail"] += 1
                    overall["fail"] += 1
                    per_benchmark[benchmark]["fail"] += 1
                    reason = result.get("fail_reason", "Unknown")
                    stats[model_name]["fail_reasons"][reason] += 1
                    overall["fail_reasons"][reason] += 1
                    if reason == "API_Failure":
                        stats[model_name]["api_fail"] += 1
                        overall["api_fail"] += 1

    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total tasks evaluated: {overall['total']}")
    print(
        f"Passed: {overall['pass']} ({100*overall['pass']/max(1,overall['total']):.1f}%)"
    )
    print(
        f"Failed: {overall['fail']} ({100*overall['fail']/max(1,overall['total']):.1f}%)"
    )
    print(f"  - API Failures: {overall['api_fail']}")
    print(f"JSON parse errors (corrupted files): {overall['json_errors']}")
    print()

    print("Failure reasons breakdown:")
    for reason, count in overall["fail_reasons"].most_common():
        pct = 100 * count / max(1, overall["fail"])
        print(f"  {reason}: {count} ({pct:.1f}% of failures)")

    print()
    print("=" * 70)
    print("PER-BENCHMARK SUMMARY")
    print("=" * 70)
    for benchmark in BENCHMARKS:
        s = per_benchmark[benchmark]
        if s["total"] == 0:
            continue
        pass_rate = 100 * s["pass"] / max(1, s["total"])
        print(f"{benchmark}: {s['pass']}/{s['total']} ({pass_rate:.1f}%)")

    print()
    print("=" * 70)
    print("PER-MODEL SUMMARY (sorted by pass rate)")
    print("=" * 70)
    model_stats = [(m, s) for m, s in stats.items()]
    model_stats.sort(key=lambda x: x[1]["pass"] / max(1, x[1]["total"]), reverse=True)

    for model, s in model_stats:
        if s["total"] == 0:
            continue
        pass_rate = 100 * s["pass"] / max(1, s["total"])
        api_info = f" [API_Fail: {s['api_fail']}]" if s["api_fail"] > 0 else ""
        num_configs = len(s["configs"])
        print(
            f"{model}: {s['pass']}/{s['total']} ({pass_rate:.1f}%) - {num_configs} configs{api_info}"
        )

        # Show top failure reasons for this model
        if s["fail_reasons"]:
            top_reasons = s["fail_reasons"].most_common(3)
            reasons_str = ", ".join([f"{r}: {c}" for r, c in top_reasons])
            print(f"    Top failures: {reasons_str}")


if __name__ == "__main__":
    main()
