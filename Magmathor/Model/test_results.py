from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import os
from enum import Enum
from typing import Dict, List

import Magmathor.constants as c
import Magmathor.utils as Utils
from Magmathor.goal import Goal
from Magmathor.objects import StepError, StepErrorType


class FailType(str, Enum):
    MAX_REPEATS = "Max_Repeats"
    MAX_STEPS = "Max_Steps"
    MAX_FAILURES = "Max_Failures"
    API_FAILURE = "API_Failure"


class StepExtension(str, Enum):
    NONE = "None"
    EXTENDED = "Extended"
    HIT_HARD_LIMIT = "Hit_Hard_Limit"


class TestResult:
    def __init__(
        self,
        task_name: str,
        goal: Goal,
        task_failed: bool,
        fail_reason: FailType | None,
        orig_step_count: int,
        test_step_count: int,
        invalid_actions: List[str],
        invalid_objects: List[str],
        step_errors: List[StepError] | None,
        manually_reviewed: bool = False,
        step_extension: StepExtension = StepExtension.NONE,
        candidate_poses_errors: int | None = None,
    ):
        self.task_name = task_name
        self.goal = goal
        self.task_failed = task_failed
        self.fail_reason = fail_reason
        self.manually_reviewed = manually_reviewed
        self.step_extension = step_extension
        self.orig_step_count = orig_step_count
        self.test_step_count = test_step_count
        self.invalid_actions = invalid_actions
        self.invalid_objects = invalid_objects
        self.step_errors = step_errors

        # Count pose errors
        # If passed in (from JSON), use that value; otherwise compute from step_errors
        if candidate_poses_errors is not None:
            self.candidate_poses_errors = candidate_poses_errors
        else:
            self.candidate_poses_errors = 0
            if step_errors:
                for steperror in step_errors:
                    if c.POSES_ERROR in (steperror.error_msg or ""):
                        self.candidate_poses_errors += 1

        if self.step_errors != None:

            num_step_errors = len(self.step_errors)
            num_invalid_objects = len(
                [
                    steperror
                    for steperror in step_errors
                    if steperror.error_type == StepErrorType.INVALID_OBJECT
                ]
            )
            num_invalid_actions = len(
                [
                    steperror
                    for steperror in step_errors
                    if steperror.error_type == StepErrorType.INVALID_ACTION
                ]
            )
            num_invalid_responses = len(
                [
                    steperror
                    for steperror in step_errors
                    if steperror.error_type == StepErrorType.INVALID_RESPONSE
                ]
            )
            num_undoable = len(
                [
                    steperror
                    for steperror in step_errors
                    if steperror.error_type == StepErrorType.UNDOABLE
                ]
            )
            num_valid = self.test_step_count - num_step_errors

            total = (
                num_invalid_objects
                + num_invalid_actions
                + num_invalid_responses
                + num_undoable
                + num_valid
            )
            if total != self.test_step_count:
                print("Checksum failure")
        else:
            num_step_errors = 0
            num_invalid_objects = 0
            num_invalid_actions = 0
            num_invalid_responses = 0
            num_undoable = 0
            num_valid = test_step_count

        self.invalid_object_ratio = num_invalid_objects / self.test_step_count
        self.invalid_action_ratio = num_invalid_actions / self.test_step_count
        self.invalid_response_ratio = num_invalid_responses / self.test_step_count
        self.undoable_ratio = num_undoable / self.test_step_count
        self.valid_ratio = num_valid / self.test_step_count
        self.checksum = (
            self.invalid_action_ratio
            + self.invalid_object_ratio
            + self.invalid_response_ratio
            + self.undoable_ratio
            + self.valid_ratio
        )

        self.percent_goals_reached = goal.percent_goals_reached()

        if (self.checksum - 1.0) > 0.001:
            print("Error!")

    def to_dict(self):
        return {
            "task_name": self.task_name,
            "task_failed": self.task_failed,
            "fail_reason": self.fail_reason.value if self.fail_reason else None,
            "orig_step_count": self.orig_step_count,
            "test_step_count": self.test_step_count,
            "goal": self.goal.to_dict() if self.goal else None,
            "invalid_actions": self.invalid_actions,
            "invalid_objects": self.invalid_objects,
            "step_errors": (
                [error.to_dict() for error in self.step_errors]
                if self.step_errors
                else None
            ),
            "manually_reviewed": self.manually_reviewed,
            "step_extension": self.step_extension.value,
            "candidate_poses_errors": self.candidate_poses_errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestResult:
        return cls(
            task_name=data["task_name"],
            task_failed=data.get("task_failed", False),
            goal=Goal.from_dict(data["goal"]) if data.get("goal") else None,
            fail_reason=(
                FailType(data["fail_reason"]) if data.get("fail_reason") else None
            ),
            orig_step_count=data.get("orig_step_count", 0),
            test_step_count=data.get("test_step_count", 0),
            invalid_actions=data.get("invalid_actions", None),
            invalid_objects=data.get("invalid_objects", None),
            step_errors=(
                [StepError.from_dict(item) for item in data.get("step_errors", [])]
                if data.get("step_errors")
                else None
            ),
            manually_reviewed=data.get("manually_reviewed", False),
            step_extension=(
                StepExtension(data["step_extension"])
                if data.get("step_extension")
                else StepExtension.NONE
            ),
            candidate_poses_errors=data.get("candidate_poses_errors"),
        )


class ResultStat:
    def __init__(self):
        self.fail_reason_counts: Dict[str, int] = {}
        for fail_reason in FailType:
            self.fail_reason_counts[fail_reason] = 0
        self.step_extension_counts: Dict[StepExtension, int] = {}
        for step_ext in StepExtension:
            self.step_extension_counts[step_ext] = 0
        self.success_count = 0
        self.fail_count = 0
        self.step_ratio = 0
        self.invalid_action_ratio = 0
        self.invalid_object_ratio = 0
        self.invalid_response_ratio = 0
        self.valid_ratio = 0
        self.undoable_ratio = 0
        self.percent_goals_reached = 0
        self.checksum = 0

    def add_result(self, test_result: TestResult):

        # Update ratios that are counted for both success and failure
        total_count = self.success_count + self.fail_count
        self.invalid_action_ratio = (
            self.invalid_action_ratio * total_count + test_result.invalid_action_ratio
        ) / (total_count + 1)
        self.invalid_object_ratio = (
            self.invalid_object_ratio * total_count + test_result.invalid_object_ratio
        ) / (total_count + 1)
        self.invalid_response_ratio = (
            self.invalid_response_ratio * total_count
            + test_result.invalid_response_ratio
        ) / (total_count + 1)

        self.undoable_ratio = (
            self.undoable_ratio * total_count + test_result.undoable_ratio
        ) / (total_count + 1)
        self.valid_ratio = (
            self.valid_ratio * total_count + test_result.valid_ratio
        ) / (total_count + 1)
        self.percent_goals_reached = (
            self.percent_goals_reached * total_count + test_result.percent_goals_reached
        ) / (total_count + 1)

        if test_result.task_failed:
            self.fail_count += 1
            if test_result.fail_reason:
                self.fail_reason_counts[test_result.fail_reason] += 1
        else:
            # Update ratio for number of steps to scucess
            step_ratio = test_result.test_step_count / test_result.orig_step_count
            self.step_ratio = (self.step_ratio * self.success_count + step_ratio) / (
                self.success_count + 1
            )
            self.success_count += 1

        # Track step extension counts
        self.step_extension_counts[test_result.step_extension] += 1

        self.checksum = (
            self.invalid_action_ratio
            + self.invalid_object_ratio
            + self.invalid_response_ratio
            + self.undoable_ratio
            + self.valid_ratio
        )
        if (self.checksum - 1.0) > 0.001:
            print("Checksum error!")

    def to_percent(self, number: float) -> str:
        return f"{int(round(number * 100)):>3}%"

    def print(self, title):
        output = f"{title:<30}"
        total = self.success_count + self.fail_count
        per_success = self.success_count / total if total > 0 else 0
        per_fail = self.fail_count / total if total > 0 else 0
        # Format percentages as right-aligned, 4 characters wide, e.g. '100%' or '  0%'
        success_pct = f"{int(round(per_success * 100)):>3}%"
        fail_pct = f"{int(round(per_fail * 100)):>3}%"
        output += f"Pass: {self.success_count:<2} ({success_pct}), Fail: {self.fail_count:<2} ({fail_pct})"

        for fail_reason, count in self.fail_reason_counts.items():
            output += f"  {fail_reason:<10}: {count:<2}"

        output += f" Step_Ratio:{self.to_percent(self.step_ratio)}"
        output += f" Valid:{self.to_percent(self.valid_ratio)}"
        output += f" Bad_Action:{self.to_percent(self.invalid_action_ratio)}"
        output += f" Bad_Object:{self.to_percent(self.invalid_object_ratio)}"
        output += f" Undoable:{self.to_percent(self.undoable_ratio)}"
        output += f" Unparsable:{self.to_percent(self.invalid_response_ratio)}"
        output += f" Goals_Reached:{self.to_percent(self.percent_goals_reached)}"

        print(output)

    def print_as_csv(
        self,
        test_results: TestResults,
        file,
        config_data: Dict | None = None,
        config_keys: List[str] | None = None,
    ):
        """Print results as CSV line to file with model_name first, then other config values, then metrics"""
        total = self.success_count + self.fail_count
        per_success = self.success_count / total if total > 0 else 0
        per_fail = self.fail_count / total if total > 0 else 0

        # Get fail reason counts
        max_repeats = self.fail_reason_counts.get(FailType.MAX_REPEATS, 0)
        max_steps = self.fail_reason_counts.get(FailType.MAX_STEPS, 0)
        max_failures = self.fail_reason_counts.get(FailType.MAX_FAILURES, 0)
        fail_api = self.fail_reason_counts.get(FailType.API_FAILURE, 0)

        test_completed = test_results.expected_num_plans == total

        # Start with model_name from config (always first column)
        csv_parts = []
        model_name = ""
        if config_data:
            model_name = config_data.get("model_name", "")
        csv_parts.append(str(model_name))

        # Add test_set_name as second column
        csv_parts.append(test_results.test_set_name)

        # Add other config values (excluding model_name, test_set_name, test_name, git_commit)
        if config_data and config_keys:
            for key in config_keys:
                if key in ("model_name", "test_set_name", "test_name", "git_commit"):
                    continue  # Already added or will be added at end
                value = config_data.get(key, "")
                # Escape commas in string values
                if isinstance(value, str) and "," in value:
                    value = f'"{value}"'
                csv_parts.append(str(value))

        # Add metrics
        csv_parts.extend(
            [
                str(test_results.expected_num_plans),
                str(test_completed),
                str(self.success_count),
                f"{per_success:.3f}",
                str(self.fail_count),
                f"{per_fail:.3f}",
                str(max_repeats),
                str(max_steps),
                str(max_failures),
                str(fail_api),
                f"{self.step_ratio:.3f}",
                f"{self.valid_ratio:.3f}",
                f"{self.invalid_action_ratio:.3f}",
                f"{self.invalid_object_ratio:.3f}",
                f"{self.undoable_ratio:.3f}",
                f"{self.invalid_response_ratio:.3f}",
                f"{self.percent_goals_reached:.3f}",
            ]
        )

        # Add git_commit as last column
        if config_data and config_keys and "git_commit" in config_keys:
            value = config_data.get("git_commit", "")
            if isinstance(value, str) and "," in value:
                value = f'"{value}"'
            csv_parts.append(str(value))

        csv_line = ",".join(csv_parts) + "\n"
        file.write(csv_line)

    @staticmethod
    def add_header_to_csv(file, config_keys: List[str] | None = None):
        """Add CSV header row to file with model_name first, then test_set_name, then other config columns"""
        # Start with model_name (always first column) and test_set_name (second column)
        header_parts = ["model_name", "test_set_name"]

        # Add other config column headers (excluding model_name, test_set_name, test_name, git_commit)
        if config_keys:
            for key in config_keys:
                if key in ("model_name", "test_set_name", "test_name", "git_commit"):
                    continue  # Already added or will be added at end
                header_parts.append(key)

        # Add metric column headers
        header_parts.extend(
            [
                "expected_num_plans",
                "test_completed",
                "success_count",
                "success_percentage",
                "fail_count",
                "fail_percentage",
                "max_repeats",
                "max_steps",
                "max_failures",
                "fail_api",
                "step_ratio",
                "valid_ratio",
                "invalid_action_ratio",
                "invalid_object_ratio",
                "undoable_ratio",
                "unparsable_ratio",
                "goals_reached_percent",
            ]
        )

        # Add git_commit as last column
        if config_keys and "git_commit" in config_keys:
            header_parts.append("git_commit")

        header = ",".join(header_parts) + "\n"
        file.write(header)


class TestResults:
    def __init__(
        self,
        test_name: str,
        model_name: str,
        model_path: str = "",
        temperature: float = -1,
        expected_num_plans: int = None,
    ):
        self.test_set_name = test_name
        self.model_name = model_name
        self.model_path = model_path
        self.temperature = temperature
        # Always define expected_num_plans to avoid AttributeError during aggregation.
        if expected_num_plans is not None:
            # Use the provided value (caller may be aggregating multiple test sets or starting from 0)
            self.expected_num_plans = expected_num_plans
        else:
            # Attempt to derive from directory listing; fall back to 0 if unavailable (e.g. combined sets)
            data_dir = f"{c.DATASET_DIR}/{test_name}"
            if os.path.isdir(data_dir):
                try:
                    self.expected_num_plans = len(os.listdir(data_dir))
                except Exception:
                    self.expected_num_plans = 0
            else:
                self.expected_num_plans = 0
        self.test_results: List[TestResult] = []

    def to_dict(self):
        return {
            "test_set_name": self.test_set_name,
            "model_name": self.model_name,
            "model_path": self.model_path,
            "temperature": self.temperature,
            "expected_num_plans": self.expected_num_plans,
            "test_results": [result.to_dict() for result in self.test_results],
        }

    @classmethod
    def from_dict(cls, data: dict) -> TestResults:
        instance = cls(
            data.get("test_set_name", ""),
            data.get("model_name", ""),
            data.get("model_path", ""),
            data.get("temperature", 0.1),
            data.get("expected_num_plans"),
        )
        instance.test_results = [
            TestResult.from_dict(item) for item in data.get("test_results", [])
        ]
        # Backward compatibility: older JSON may not have expected_num_plans; ensure attribute exists
        if not hasattr(instance, "expected_num_plans"):
            instance.expected_num_plans = 0
        return instance

    def get_result(self, task_name: str) -> TestResult | None:
        for result in self.test_results:
            if result.task_name == task_name:
                return result
        return None

    def print(self):

        total_results = ResultStat()
        task_type_results: Dict[str, ResultStat] = {}
        for test_result in self.test_results:
            task_type = test_result.task_name.split("_FloorPlan")[0].strip()
            if task_type not in task_type_results:
                task_type_results[task_type] = ResultStat()

            total_results.add_result(test_result)
            task_type_results[task_type].add_result(test_result)

        Utils.print_color(c.Color.GREY, "\n==========================")
        Utils.print_color(
            c.Color.LIGHT_BLUE, f"{self.test_set_name} - {self.model_name}"
        )
        print("--------------------------")
        for task_type, stats in task_type_results.items():
            stats.print(task_type)
        print("--------------------------")
        total_results.print("Total:")
        Utils.print_color(c.Color.GREY, "==========================\n")

    def add_to_csv(
        self,
        file,
        config_data: Dict | None = None,
        config_keys: List[str] | None = None,
    ):

        total_results = ResultStat()
        for test_result in self.test_results:
            total_results.add_result(test_result)

        # Write total as csv to file if not None
        if file is not None:
            total_results.print_as_csv(self, file, config_data, config_keys)
