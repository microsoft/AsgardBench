"""Entry point for the experiment runner."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment_runner.runner import TestRunner

# Global reference to the runner for cleanup on interrupt
_runner: TestRunner | None = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments with optional concurrency limits."
    )
    parser.add_argument(
        "--max-jobs",
        "-j",
        type=int,
        default=0,
        help="Maximum number of concurrent jobs (0 = no limit, default: 0)",
    )
    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    global _runner
    from experiment_runner.runner import TestRunner

    args = parse_args()
    _runner = TestRunner(max_concurrent_jobs=args.max_jobs)
    return await _runner.run_interactive()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")
        if _runner is not None:
            _runner.kill_all_experiments()
        sys.exit(1)
