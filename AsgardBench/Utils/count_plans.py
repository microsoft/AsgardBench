import json
import os
import re
from dataclasses import dataclass

import AsgardBench.utils as Utils
from AsgardBench import constants as c
from AsgardBench.objects import GenerationResults

# File to persist status between plan generator and monitor
STATUS_FILE = os.path.join(c.DATASET_DIR, "status.json")


def get_tracking_name(plan_name: str) -> str:
    """Extract tracking name from plan name by removing scene and version info."""
    return plan_name.split("_FloorPlan")[0].strip()


def _load_status() -> dict:
    """Load status from the JSON file."""
    if not os.path.exists(STATUS_FILE):
        return {}
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_status(data: dict) -> None:
    """Save status to the JSON file."""
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def set_current_task(task_name: str) -> None:
    """Set the current task being worked on."""
    status = _load_status()
    status["current_task"] = task_name
    _save_status(status)


def get_current_task() -> str | None:
    """Get the current task being worked on."""
    status = _load_status()
    return status.get("current_task")


def save_failures(
    failed_scene_dict: dict[str, set[str]], failures_dict: dict[str, list[str]]
) -> None:
    """Save failures to the status file."""
    status = _load_status()
    status["failed_scenes"] = {k: list(v) for k, v in failed_scene_dict.items()}
    status["failures"] = {k: list(v) for k, v in failures_dict.items()}
    _save_status(status)


def load_failures() -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """Load failures from the status file."""
    status = _load_status()
    failed_scene_dict = {k: set(v) for k, v in status.get("failed_scenes", {}).items()}
    failures_dict = {k: list(v) for k, v in status.get("failures", {}).items()}
    return failed_scene_dict, failures_dict


def clear_status() -> None:
    """Clear the status file."""
    if os.path.exists(STATUS_FILE):
        os.remove(STATUS_FILE)


# Keep for backward compatibility
def clear_failures() -> None:
    """Clear the status file (alias for clear_status)."""
    clear_status()


def get_family_name(plan_name: str) -> str | None:
    """
    Extract family name from plan name for family-based limiting.

    For example:
    - "distribute__Lettuce_Plate_FloorPlan1_V1" -> "distribute__Lettuce"
    - "distribute__Apple_Bowl(d)_FloorPlan2_V1" -> "distribute__Apple"
    - "cook_Egg_FloorPlan1_V1" -> None (no family limiting)

    Returns:
        Family name if applicable, None otherwise.
    """
    if plan_name.startswith("distribute__"):
        # Extract "distribute__<slicable_type>"
        parts = plan_name.split("_")
        # parts[0] = "distribute", parts[1] = "", parts[2] = slicable_type
        if len(parts) >= 3:
            return f"distribute__{parts[2]}"
    return None


@dataclass
class PlanStats:
    """Container for plan counting results."""

    results_dict: dict[str, GenerationResults]
    scene_dict: dict[str, set[str]]
    failed_scene_dict: dict[str, set[str]]
    steps_dict: dict[str, list[int]]
    total_steps: int

    def plan_exists(self, plan_name: str) -> bool:
        """
        Check if a plan already exists on disk.

        Args:
            plan_name: The plan name to check (e.g., "cook_Egg_FloorPlan1_V1").

        Returns:
            True if the plan exists (at least non-injection version), False otherwise.
        """
        # Only keep part of name before the dash
        plan_root = plan_name.split("-")[0]
        tracking_name = get_tracking_name(plan_name)

        if tracking_name not in self.results_dict:
            return False

        results = self.results_dict[tracking_name]

        # Check if any existing directory starts with this plan root
        for directory in results.existing + results.existing_inj:
            if directory.startswith(plan_root):
                return True

        return False

    def num_plans(self, plan_name: str) -> int:
        """
        Get the number of successful plans for a tracking name.

        Args:
            plan_name: The plan name to check.

        Returns:
            Number of successful plans (non-injection only) plus existing on disk.
        """
        tracking_name = get_tracking_name(plan_name)

        if tracking_name not in self.results_dict:
            return 0

        results = self.results_dict[tracking_name]
        return len(results.success) + len(results.existing)

    def num_family_plans(self, plan_name: str) -> int:
        """
        Get the total number of plans in a family.

        A family groups related plans together, e.g., all "distribute__Lettuce" plans
        regardless of serving types or conditions.

        Args:
            plan_name: The plan name to check.

        Returns:
            Total number of plans in the family, or 0 if no family.
        """
        family_name = get_family_name(plan_name)
        if family_name is None:
            return 0

        total = 0
        for tracking_name, results in self.results_dict.items():
            if tracking_name.startswith(family_name):
                total += len(results.success) + len(results.existing)
        return total

    def _ensure_tracking_name(self, tracking_name: str) -> None:
        """Ensure a tracking name exists in results_dict and scene_dict."""
        if tracking_name not in self.results_dict:
            self.results_dict[tracking_name] = GenerationResults()
        if tracking_name not in self.scene_dict:
            self.scene_dict[tracking_name] = set()
        if tracking_name not in self.failed_scene_dict:
            self.failed_scene_dict[tracking_name] = set()
        if tracking_name not in self.steps_dict:
            self.steps_dict[tracking_name] = []

    def _add_scene(self, tracking_name: str, scene: str) -> None:
        """Extract and add scene number from scene name."""
        scene_match = re.search(r"FloorPlan(\d+)", scene)
        if scene_match:
            self.scene_dict[tracking_name].add(scene_match.group(1))

    def _add_failed_scene(self, tracking_name: str, scene: str) -> None:
        """Extract and add scene number from scene name to failed scenes."""
        scene_match = re.search(r"FloorPlan(\d+)", scene)
        if scene_match:
            self.failed_scene_dict[tracking_name].add(scene_match.group(1))

    def _update_step_stats(
        self, tracking_name: str, steps: int, is_injection: bool
    ) -> None:
        """Update step statistics for a tracking name."""
        if steps > 0:
            self.steps_dict[tracking_name].append(steps)
            self.total_steps += steps

            # Recalculate min/max/avg for this task
            results = self.results_dict[tracking_name]
            all_steps = self.steps_dict[tracking_name]
            results.max_steps = max(all_steps)
            results.avg_steps = sum(all_steps) / len(all_steps)
            # Only update min_steps from non-injection plans
            if not is_injection:
                if results.min_steps == 0:
                    results.min_steps = steps
                else:
                    results.min_steps = min(results.min_steps, steps)

    def add_failure(self, plan_name: str, scene: str, steps: int = 0) -> None:
        """Add a failed plan to the results and persist to disk."""
        tracking_name = get_tracking_name(plan_name)
        self._ensure_tracking_name(tracking_name)
        self.results_dict[tracking_name].failures.append(scene)
        self._add_failed_scene(tracking_name, scene)
        self._update_step_stats(tracking_name, steps, is_injection=False)
        # Persist failures to disk
        failures_dict = {
            k: v.failures for k, v in self.results_dict.items() if v.failures
        }
        save_failures(self.failed_scene_dict, failures_dict)

    def add_success(self, plan_name: str, scene: str, steps: int = 0) -> None:
        """Add a successful plan to the results."""
        tracking_name = get_tracking_name(plan_name)
        self._ensure_tracking_name(tracking_name)
        self.results_dict[tracking_name].success.append(scene)
        self._add_scene(tracking_name, scene)
        self._update_step_stats(tracking_name, steps, is_injection=False)

    def add_success_injection(self, plan_name: str, scene: str, steps: int = 0) -> None:
        """Add a successful injection plan to the results."""
        tracking_name = get_tracking_name(plan_name)
        self._ensure_tracking_name(tracking_name)
        self.results_dict[tracking_name].success_inj.append(scene)
        self._add_scene(tracking_name, scene)
        self._update_step_stats(tracking_name, steps, is_injection=True)

    def print_stats(self, file=None, max_samples: int | None = None) -> dict:
        """
        Print formatted plan statistics combining generation and disk info.

        Args:
            file: Optional file to write to. If None, prints to stdout in color.
            max_samples: Optional max samples threshold. If provided and a task has
                        enough plans (success + existing >= max_samples), prints in dark green.

        Returns:
            Dictionary with totals for success, success_inj, failures, existing, existing_inj.
        """

        def print_line(text: str, color: c.Color = c.Color.BLUE):
            if file is not None:
                file.write(f"{text}\r\n")
            else:
                Utils.print_color(color, text)

        totals = {
            "success": 0,
            "success_inj": 0,
            "failures": 0,
            "existing": 0,
            "existing_inj": 0,
        }

        # Sort results_dict alphabetically by task name
        keys = sorted(self.results_dict.keys())

        for task in keys:
            results = self.results_dict[task]

            # Count totals
            num_success = len(results.success)
            num_success_inj = len(results.success_inj)
            num_failures = len(results.failures)
            num_existing = len(results.existing)
            num_existing_inj = len(results.existing_inj)

            totals["success"] += num_success
            totals["success_inj"] += num_success_inj
            totals["failures"] += num_failures
            totals["existing"] += num_existing
            totals["existing_inj"] += num_existing_inj

            # Format counts
            total_normal = num_success + num_existing
            total_inj = num_success_inj + num_existing_inj
            total_str = f"{total_normal}+{total_inj}"
            success_str = f"{num_success}+{num_success_inj}"
            existing_str = f"{num_existing}+{num_existing_inj}"

            # Determine line color based on whether we have enough samples
            # For distribute__ plans, check family limit (3x max_samples)
            # For other plans, check individual limit (max_samples)
            line_color = c.Color.BLUE
            if max_samples is not None:
                family_name = get_family_name(task)
                if family_name is not None:
                    # Family-based limiting: check if family has 3x max_samples
                    family_total = self.num_family_plans(task)
                    if family_total >= max_samples * 3:
                        line_color = c.Color.DARK_GREEN
                elif (num_success + num_existing) >= max_samples:
                    line_color = c.Color.DARK_GREEN

            # Colorize success and failure counts (reset back to line color after)
            color_reset = line_color.value
            if num_success + num_success_inj > 0:
                success_part = f"{c.Color.GREEN.value}S:{success_str:>5}{color_reset}"
            else:
                success_part = f"S:{success_str:>5}"

            if num_failures > 0:
                failure_part = f"{c.Color.RED.value}F:{num_failures:>2}{color_reset}"
            else:
                failure_part = f"F:{num_failures:>2}"

            # Get scene info if available - successful in green, failed in red
            scene_str = ""
            has_success_scenes = task in self.scene_dict and self.scene_dict[task]
            has_failed_scenes = (
                task in self.failed_scene_dict and self.failed_scene_dict[task]
            )

            if has_success_scenes or has_failed_scenes:
                scene_str = "  Scenes: "
                if has_success_scenes:
                    scene_numbers = [
                        str(num) for num in sorted(self.scene_dict[task], key=int)
                    ]
                    scene_str += (
                        f"{c.Color.GREEN.value}{', '.join(scene_numbers)}{color_reset}"
                    )
                if has_failed_scenes:
                    failed_numbers = [
                        str(num)
                        for num in sorted(self.failed_scene_dict[task], key=int)
                    ]
                    if has_success_scenes:
                        scene_str += " "
                    scene_str += (
                        f"{c.Color.RED.value}{', '.join(failed_numbers)}{color_reset}"
                    )

            # Get step stats if available - format with fixed widths for alignment
            step_str = ""
            if results.min_steps > 0 or results.max_steps > 0:
                step_str = f"  Steps: {results.min_steps:>3}-{results.max_steps:<3} ({results.avg_steps:>5.1f})"

            print_line(
                f"{task:<40}  T:{total_str:>5}  {success_part}  {failure_part}  E:{existing_str:>5}{step_str}{scene_str}",
                line_color,
            )

        # Print summary
        print_line("=" * 80)
        total_normal = totals["success"] + totals["existing"]
        total_inj = totals["success_inj"] + totals["existing_inj"]
        total_all = total_normal + total_inj
        print_line(
            f"Total: {total_all} ({total_normal}+{total_inj})  "
            f"Success: {totals['success']}+{totals['success_inj']}  "
            f"Failures: {totals['failures']}  "
            f"Existing: {totals['existing']}+{totals['existing_inj']}  "
            f"Steps: {self.total_steps}"
        )

        return totals


def count_plans(plans_dir: str | None = None) -> PlanStats:
    """
    Count and analyze plans in the given directory.

    Args:
        plans_dir: Path to the plans directory. Defaults to the standard new plans dir.

    Returns:
        PlanStats object containing results (empty if directory doesn't exist).
    """
    if plans_dir is None:
        plans_dir = f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}"

    if not os.path.exists(plans_dir):
        return PlanStats(
            results_dict={},
            scene_dict={},
            failed_scene_dict={},
            steps_dict={},
            total_steps=0,
        )

    results_dict: dict[str, GenerationResults] = {}
    scene_dict: dict[str, set[str]] = {}
    steps_dict: dict[str, list[int]] = {}
    steps_dict_inj: dict[str, list[int]] = {}  # Steps for injection plans only
    total_steps = 0

    all_directories = os.listdir(plans_dir)
    for directory in all_directories:
        # Extract text before "FloorPlan" to get task name
        task_name = get_tracking_name(directory)

        if task_name not in results_dict:
            results_dict[task_name] = GenerationResults()

        if task_name not in steps_dict:
            steps_dict[task_name] = []
            steps_dict_inj[task_name] = []
            scene_dict[task_name] = set()

        injection_pattern = re.compile(r"[Rr]\d+")
        is_injection = bool(re.search(injection_pattern, directory))

        if not is_injection:
            injection_pattern = re.compile(r"[Fr]\d+")
            is_injection = bool(re.search(injection_pattern, directory))

        is_injection = Utils.is_error_recovery_plan(directory)

        # If it ends with "_R" plus a number, its an injection
        if is_injection:
            results_dict[task_name].existing_inj.append(directory)
        else:
            results_dict[task_name].existing.append(directory)

        # Extract scene number {n} from "{dect}_FloorPlan{n}_{info}"
        scene_match = re.search(r"FloorPlan(\d+)", directory)
        if scene_match:
            scene_dict[task_name].add(scene_match.group(1))

        # Get number of steps from the directory name
        bracket_match = re.search(r"\[(\d+)\]$", directory)
        if bracket_match:
            steps_count = int(bracket_match.group(1))
            if is_injection:
                steps_dict_inj[task_name].append(steps_count)
            else:
                steps_dict[task_name].append(steps_count)
            total_steps += steps_count
        else:
            raise ValueError(f"Could not find steps in directory name: {directory}")

    # Get stats on number steps (min from non-injection only, max/avg from all)
    for task_name in steps_dict.keys():
        regular_steps = steps_dict[task_name]
        injection_steps = steps_dict_inj[task_name]
        all_steps = regular_steps + injection_steps

        if all_steps:
            # Min steps only from regular (non-injection) plans
            min_steps = min(regular_steps) if regular_steps else min(injection_steps)
            max_steps = max(all_steps)
            avg_steps = sum(all_steps) / len(all_steps)
            results_dict[task_name].min_steps = min_steps
            results_dict[task_name].max_steps = max_steps
            results_dict[task_name].avg_steps = avg_steps
        else:
            results_dict[task_name].min_steps = 0
            results_dict[task_name].max_steps = 0
            results_dict[task_name].avg_steps = 0

    # Load persisted failures
    failed_scene_dict, failures_dict = load_failures()
    for task_name, failures in failures_dict.items():
        if task_name not in results_dict:
            results_dict[task_name] = GenerationResults()
        results_dict[task_name].failures = failures
        if task_name not in scene_dict:
            scene_dict[task_name] = set()

    return PlanStats(
        results_dict=results_dict,
        scene_dict=scene_dict,
        failed_scene_dict=failed_scene_dict,
        steps_dict=steps_dict,
        total_steps=total_steps,
    )


def print_plan_stats(stats: PlanStats) -> None:
    """
    Print formatted plan statistics.

    Args:
        stats: PlanStats object from count_plans().
    """
    stats.print_stats()


if __name__ == "__main__":
    plan_stats = count_plans()
    if plan_stats is None:
        print("No plans found.")
    else:
        print_plan_stats(plan_stats)
