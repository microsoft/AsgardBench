# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, List

from dotenv import load_dotenv

load_dotenv()
from PIL import Image

import AsgardBench.Model.prompt_templates as Prompts
import AsgardBench.utils as Utils
from AsgardBench import constants as c
from AsgardBench.cache.item_cache import ItemCache

# Check if verbose output should be disabled
_QUIET_MODE = os.environ.get("ASGARDBENCH_QUIET", "").lower() in ("1", "true")

from AsgardBench.Model.openai_actor import OpenAIActor
from AsgardBench.Model.prompt_templates import PromptParams, render_prompt
from AsgardBench.Model.test_results import (
    FailType,
    StepExtension,
    TestResult,
    TestResults,
)
from AsgardBench.objects import AgentPolicyError, ModelEmptyResponseError, StepError
from AsgardBench.plan import Plan, PlanType
from AsgardBench.player import Player
from AsgardBench.scenario import Scenario
from AsgardBench.step_log import clear_log_buffer, log_print, set_current_step
from AsgardBench.storage_utils import ensure_dir_exists
from AsgardBench.Utils.config_utils import EvaluationConfig

# Plans that I'm testing
PLAN_FOLDER = "Plans"

MAX_ACTION_REPEATS = 8
MAX_CONSECUTIVE_FAILURES = 10

# How many extra steps to allow over completed plan
# After EXTRA_STEP_RATIO_SOFT, we check for object novelty before continuing
# After EXTRA_STEP_RATIO_HARD, we always fail
EXTRA_STEP_RATIO_SOFT = 1.5
EXTRA_STEP_RATIO_HARD = 2.0

# Number of recent steps to check for object novelty
OBJECT_NOVELTY_WINDOW = 10


def extract_partial_prompt(
    prompt: str,
    current_image: str | None,
    previous_image: str | None,
) -> str:
    """Extract the SCENE STATE section and image info from a prompt.

    This extracts just the dynamic parts of the prompt that change each step:
    - Image filenames
    - The SCENE STATE section (history, memory, suggested plan)

    Args:
        prompt: The full prompt string
        current_image: Path to current image file (or None for text-only)
        previous_image: Path to previous image file (or None)

    Returns:
        A string containing the partial prompt info
    """
    parts = []

    # Add image info
    parts.append("=== IMAGES ===")
    if previous_image:
        # Just the filename, not full path
        parts.append(f"Previous: {os.path.basename(previous_image)}")
    if current_image:
        parts.append(f"Current: {os.path.basename(current_image)}")
    if not current_image and not previous_image:
        parts.append("(text-only mode)")
    parts.append("")

    # Extract SCENE STATE section
    scene_state_marker = "SECTION: SCENE STATE"
    if scene_state_marker in prompt:
        # Find the start of SCENE STATE section
        start_idx = prompt.find(scene_state_marker)
        # Find the next section or end of prompt
        # Look for "SECTION:" after the scene state marker
        after_scene_state = prompt[start_idx + len(scene_state_marker) :]
        next_section_idx = after_scene_state.find("SECTION:")

        if next_section_idx != -1:
            scene_state_content = after_scene_state[:next_section_idx]
        else:
            # No next section, take until end but try to stop at examples or answer format
            # Look for common endings
            end_markers = ["<example>", "## Answer format", "----"]
            scene_state_content = after_scene_state
            for marker in end_markers:
                marker_idx = scene_state_content.find(marker)
                if marker_idx != -1:
                    scene_state_content = scene_state_content[:marker_idx]
                    break

        parts.append("=== SCENE STATE ===")
        # Filter out NOTE about proposed plan and dash lines
        filtered_lines = []
        for line in scene_state_content.strip().split("\n"):
            # Skip dash separator lines
            if line.strip().startswith("---"):
                continue
            # Skip the NOTE about proposed plan
            if "NOTE: This was YOUR proposed plan" in line:
                continue
            if "Do NOT follow it blindly" in line:
                continue
            filtered_lines.append(line)
        parts.append("\n".join(filtered_lines))

    return "\n".join(parts)


import filecmp


def verify_image_filename_consistency(
    temp_current_image: str | None,
    temp_previous_image: str | None,
    step_current_image: str | None,
    step_previous_image: str | None,
    save_dir: str,
) -> None:
    """Verify that step image files have the same content as temp files used in prompt.

    This sanity check ensures that the descriptive filenames from step objects
    (e.g., '1_put Mug.png') contain the same image data as the temp files that were
    actually sent to the model.

    Args:
        temp_current_image: Path to the temp current image (cur_image.png)
        temp_previous_image: Path to the temp previous image (prev_image.png)
        step_current_image: The descriptive filename from current step
        step_previous_image: The descriptive filename from previous step
        save_dir: Directory where step images are saved

    Raises:
        AssertionError: If image contents don't match
    """
    # Compare current image contents
    if temp_current_image is not None and step_current_image is not None:
        step_current_path = os.path.join(save_dir, step_current_image)
        if os.path.exists(temp_current_image) and os.path.exists(step_current_path):
            assert filecmp.cmp(
                temp_current_image, step_current_path, shallow=False
            ), f"Current image mismatch: {temp_current_image} != {step_current_path}"

    # Compare previous image contents
    if temp_previous_image is not None and step_previous_image is not None:
        step_previous_path = os.path.join(save_dir, step_previous_image)
        if os.path.exists(temp_previous_image) and os.path.exists(step_previous_path):
            assert filecmp.cmp(
                temp_previous_image, step_previous_path, shallow=False
            ), f"Previous image mismatch: {temp_previous_image} != {step_previous_path}"


def get_git_commit() -> str | None:
    """Get the current git commit hash.

    Returns:
        The git commit hash, or None if git is not available
        (e.g., on remote cluster nodes without .git folder).
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        # git command failed, not installed, or .git folder not present
        pass
    return None


class ModelTester:
    """
    Class to handle manual control of the environment.
    """

    def __init__(
        self,
        test_name: str,
        model: str,
        config: EvaluationConfig,
        rep_number: int = 1,
    ):

        print(f"-== Testing model: {model} at temperature {config.temperature}")

        self.test_name = test_name
        self.model_id = model
        self.config = config
        self.rep_number = rep_number

        mode_str = "TEXT ONLY" if self.config.text_only else "IMAGE"
        print(f"-== Model: {self.model_id} ({mode_str} mode)")
        self.replay = "replay" in model
        print(f"-== Replay mode: {self.replay}")

        # Where I can find plans to test on
        self.test_plan_dir = f"{c.DATASET_DIR}/{test_name}"

        suffix = self.config.get_output_suffix()
        if suffix:
            self.output_folder = f"{c.TEST_DIR}/{test_name}/{self.model_id}--{suffix}"
        else:
            self.output_folder = f"{c.TEST_DIR}/{test_name}/{self.model_id}"

        # Append rep number
        self.output_folder = f"{self.output_folder}--rep{rep_number}"

        self.test_results_file = f"{self.output_folder}/test_results.json"

        self.config_file = f"{self.output_folder}/config.json"

        # Where I put the plans generated during testing
        self.plan_folder = f"{self.output_folder}/{PLAN_FOLDER}"

        # Ensure output directories exist using storage-aware paths
        ensure_dir_exists(self.test_results_file)
        ensure_dir_exists(f"{self.plan_folder}/dummy")  # Ensure plan folder exists

        if not os.path.exists(self.test_plan_dir):
            print("No plans found.")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            self.test_results = TestResults(
                self.test_name,
                self.model_id,
                self.model_id,
                self.config.temperature,
            )
        elif not os.path.exists(self.test_results_file):
            self.test_results = TestResults(
                self.test_name,
                self.model_id,
                self.model_id,
                self.config.temperature,
            )
        else:
            print(f"Loading existing test results from {self.test_results_file}")
            with open(self.test_results_file, "r", encoding="utf-8") as f:
                results_json = json.load(f)
            self.test_results = TestResults.from_dict(results_json)
            self.test_results.print()

        # Save config for reproducibility
        config_data = self.config.to_dict()
        config_data["test_name"] = self.test_name
        config_data["model"] = self.model_id
        config_data["git_commit"] = get_git_commit()

        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"Saved config to {self.config_file}")

        self.fail_reason: FailType | None = None
        self.max_steps = 0
        self.max_steps_hard = 0
        self.start_time: float | None = None  # Will be set when run() is called

        self.model: Any = None

        # Actions names that were invalid names
        self.invalid_actions: List[str] = []

        # Object names that were invalid
        self.invalid_objects: List[str] = []

        # Action that could not be taken
        self.step_errors: List[StepError] | None = None

    def print_elapsed_time(self, completed_tests=None, total_tests=None):
        """Print the elapsed time since testing started and estimated time remaining"""
        if self.start_time is not None:
            elapsed = time.time() - self.start_time

            # Format elapsed time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            if hours > 0:
                elapsed_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                elapsed_str = f"{minutes:02d}:{seconds:02d}"

            # Calculate estimated time remaining if progress info is available
            if (
                completed_tests is not None
                and total_tests is not None
                and completed_tests > 0
            ):
                avg_time_per_test = elapsed / completed_tests
                remaining_tests = total_tests - completed_tests
                estimated_remaining = avg_time_per_test * remaining_tests

                # Format estimated remaining time
                rem_hours = int(estimated_remaining // 3600)
                rem_minutes = int((estimated_remaining % 3600) // 60)
                rem_seconds = int(estimated_remaining % 60)

                if rem_hours > 0:
                    remaining_str = (
                        f"{rem_hours:02d}:{rem_minutes:02d}:{rem_seconds:02d}"
                    )
                else:
                    remaining_str = f"{rem_minutes:02d}:{rem_seconds:02d}"

                print(
                    f"Elapsed: {elapsed_str} | Est. remaining: {remaining_str} ({completed_tests}/{total_tests} tests)"
                )
            else:
                print(f"Elapsed time: {elapsed_str}")

    def run(self):

        self.start_time = time.time()
        all_directories = os.listdir(self.test_plan_dir)

        # Count only testable plans (exclude already tested, error recovery, and missing plan.json)
        testable_count = 0
        for directory in all_directories:
            plan_path = os.path.join(self.test_plan_dir, directory, "plan.json")
            if not os.path.exists(plan_path):
                continue
            if Utils.is_error_recovery_plan(directory):
                continue
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    plan_json = json.load(f)
                test_plan = Plan.from_dict(plan_json)
                if self.test_results.get_result(test_plan.name) is not None:
                    continue
                testable_count += 1
            except Exception:
                continue

        total_dirs = len(all_directories)
        completed_tests = 0

        print(f"--= Found {total_dirs} directories, {testable_count} plans to test.")
        for i, directory in enumerate(all_directories):

            # Load the plan from the directory
            plan_path = os.path.join(self.test_plan_dir, directory, "plan.json")
            if not os.path.exists(plan_path):
                print(
                    f"{i:<4}/{total_dirs:<4} Skipping {directory} - plan.json not found."
                )
                continue

            with open(plan_path, "r", encoding="utf-8") as f:
                plan_json = json.load(f)

            test_plan = Plan.from_dict(plan_json)
            self.max_steps_soft = max(
                int(test_plan.step_count * EXTRA_STEP_RATIO_SOFT), 15
            )
            self.max_steps_hard = max(
                int(test_plan.step_count * EXTRA_STEP_RATIO_HARD), 20
            )
            self.step_extension = StepExtension.NONE

            # DEBUG a particular plan
            # i f "mirror" not in test_plan.name:
            #   continue

            if self.test_results.get_result(test_plan.name) is not None:
                Utils.print_color(
                    c.Color.LIGHT_BLUE,
                    f"{i:<4}/{total_dirs:<4} Skipping {directory} - already tested.",
                )
                continue

            if Utils.is_error_recovery_plan(directory):
                Utils.print_color(
                    c.Color.LIGHT_BLUE,
                    f"{i:<4}/{total_dirs:<4} Skipping {directory} - skip error recovery plan.",
                )
                continue

            # Set up model unless doing a replay
            if not self.replay and self.model is None:
                try:
                    print(
                        f"--= Loading model: {self.model_id} at temperature {self.config.temperature}"
                    )

                    self.model = OpenAIActor(
                        self.model_id,
                        self.config.temperature,
                        self.config.max_completion_tokens,
                    )

                except Exception as e:  # pylint: disable=broad-except
                    print(f"--= Exception initializing model: {e}")
                    Utils.print_color(
                        c.Color.RED, f"!! Failed to initialize model: {self.model_id}"
                    )
                    sys.exit(1)

            # Clear out goal states
            test_plan.goal.reset_goals()

            # Delete temp file if it exists
            if os.path.exists(f"{self.plan_folder}/{test_plan.name}"):
                shutil.rmtree(
                    f"{self.plan_folder}/{test_plan.name}"
                )  # Changed from os.rmdir

            Utils.print_color(c.Color.BLUE, "-----------------------------------------")
            Utils.print_color(
                c.Color.BLUE,
                f"Testing: [{i}/{total_dirs}] {self.model_id} {self.test_plan_dir} - {test_plan.name} - {self.config.temperature}",
            )
            Utils.print_color(c.Color.BLUE, "-----------------------------------------")
            completed_tests += 1

            try:
                new_plan = self.test_plan(
                    test_plan, i, total_dirs, self.start_time, completed_tests
                )
            except AgentPolicyError as ape:
                Utils.print_color(
                    c.Color.RED,
                    f"!! Test aborted due to responsible AI policy violation: {ape}",
                )
                new_plan = None

            # If test failed continue to next plan
            if new_plan is None:
                print(
                    f"{i:<4}/{total_dirs:<4} Finished {directory} - test failed due to no response."
                )
                continue

            # Save json version of action plan
            filename = f"{self.plan_folder}/{test_plan.name}/plan.json"
            with open(filename, "w", encoding="utf-8") as file:
                json.dump(new_plan.to_dict(), file, indent=2)

            test_result = TestResult(
                task_name=test_plan.name,
                task_failed=new_plan.task_failed,
                goal=new_plan.goal,
                fail_reason=self.fail_reason,
                orig_step_count=test_plan.step_count,
                test_step_count=len(new_plan.steps),
                invalid_actions=self.invalid_actions,
                invalid_objects=self.invalid_objects,
                step_errors=self.step_errors,
                step_extension=self.step_extension,
            )
            self.test_results.test_results.append(test_result)

            # Save test results with proper encoding and Azure ML awareness
            with open(self.test_results_file, "w", encoding="utf-8") as file:
                json.dump(self.test_results.to_dict(), file, indent=2)

            # Log Azure ML info for debugging
            azure_info = get_azure_ml_info()
            if azure_info["is_azure_ml"]:
                print(f"Azure ML: Saved results to {self.test_results_file}")
                print(f"Azure ML Job ID: {azure_info['job_id']}")
                print(f"Azure ML Experiment: {azure_info['experiment_name']}")

            if new_plan.task_failed:
                Utils.print_color(
                    c.Color.RED,
                    f"{i:<4}/{total_dirs:<4} Finished {directory} - test failed.",
                )

                # Add underscore to note as failed
                old_name = f"{self.plan_folder}/{test_plan.name}"
                fail_name = f"{self.plan_folder}/_{test_plan.name}"

                if os.path.exists(fail_name):
                    shutil.rmtree(fail_name)

                os.rename(old_name, fail_name)

            else:
                Utils.print_color(
                    c.Color.GREEN,
                    f"{i:<4}/{total_dirs:<4} Finished {directory} - tested successfully.",
                )

            self.print_elapsed_time(completed_tests, testable_count)
            self.test_results.print()

        self.print_elapsed_time(testable_count, testable_count)
        self.test_results.print()

    def check_completion(self) -> bool:
        """Return True if all non-recovery plans in test set have results recorded.

        Mirrors the skipping logic in run():
        - skip directories without plan.json
        - skip error recovery plans (Utils.is_error_recovery_plan)
        - consider a plan completed if test_results.get_result(plan.name) is not None
        """
        if not os.path.exists(self.test_plan_dir):
            print("No plans directory found; treating as incomplete.")
            return False

        all_directories = os.listdir(self.test_plan_dir)
        total_candidates = 0
        completed = 0
        for directory in all_directories:
            plan_path = os.path.join(self.test_plan_dir, directory, "plan.json")
            if not os.path.exists(plan_path):
                continue
            if Utils.is_error_recovery_plan(directory):
                continue
            try:
                with open(plan_path, "r", encoding="utf-8") as f:
                    plan_json = json.load(f)
                test_plan = Plan.from_dict(plan_json)
            except Exception as e:  # If a plan is unreadable treat as incomplete
                print(f"Error reading plan {directory}: {e}")
                continue
            total_candidates += 1
            if self.test_results.get_result(test_plan.name) is not None:
                completed += 1
        print(f"Completion status: {completed}/{total_candidates} plans have results.")
        return total_candidates > 0 and completed == total_candidates

    def get_action_by_value(self, action_name: str):

        try:
            action = c.Action[action_name.upper()]
            return action
        except KeyError:
            for action in c.Action:
                if action.value == action_name.lower():
                    return action

        raise KeyError(f"No Action member found for value '{action_name}'")

    def num_consecutive_failures(self, step_error_msgs: List[str]) -> int:
        """
        Count the number of consecutive failures at the end of the list of step error messages.
        """
        count = 0
        for msg in reversed(step_error_msgs):
            if msg is not None:
                count += 1
            else:
                break
        return count

    def check_step_limit(
        self,
        step_count: int,
        object_type: str | None,
        step_history: List[str],
        scenario: Scenario,
    ) -> bool:
        """
        Two-tier step limit check.
        Returns True if execution should stop (limit exceeded).
        """
        if step_count > self.max_steps_hard:
            # Hard limit: always fail
            self.fail_reason = FailType.MAX_STEPS
            self.step_extension = StepExtension.HIT_HARD_LIMIT
            return True
        elif step_count > self.max_steps_soft:
            # Check if new subgoal reached, if so extend the limit
            if scenario.reached_new_subgoal:
                scenario.reached_new_subgoal = False  # Reset flag
                self.max_steps_soft += 10  # Give some extra steps to continue
                log_print(
                    f"Extending soft step limit due to new subgoal reached. New soft limit: {self.max_steps_soft}"
                )
                return False

            # Soft limit: check if we're making progress by targeting new objects
            # Compare objects in last 10 steps vs objects in steps -10 to -20
            # If there are novel objects (in last 10 but not in -10 to -20), extend the limit

            def get_objects_from_history(history_slice: List[str]) -> set:
                """Extract object names from a slice of action history."""
                objects = set()
                for action in history_slice:
                    parts = action.split(" ", 1)
                    if len(parts) > 1:
                        objects.add(parts[1])
                return objects

            # Get objects from last 10 steps
            recent_objects = get_objects_from_history(
                step_history[-OBJECT_NOVELTY_WINDOW:]
            )

            # Get objects from steps -10 to -20 (the 10 steps before the last 10)
            older_objects = get_objects_from_history(
                step_history[-2 * OBJECT_NOVELTY_WINDOW : -OBJECT_NOVELTY_WINDOW]
            )

            # Check if there are any novel objects (in recent but not in older)
            novel_objects = recent_objects - older_objects

            if novel_objects:
                # Making progress - targeting new objects
                log_print(
                    f"Extending soft step limit due to novel objects: {novel_objects}"
                )
                self.max_steps_soft += 10  # Give some extra steps to continue
                self.step_extension = StepExtension.EXTENDED
            else:
                # Not making progress - targeting same objects as before
                self.fail_reason = FailType.MAX_STEPS
                return True
        return False

    def print_history(self, history: List[str], errors: List[str]) -> None:
        """
        Print the history of actions taken along with any associated errors.
        """
        for i, action_object in enumerate(history):
            error = errors[i]
            if error is None:
                Utils.print_color(c.Color.GREEN, f"{i:<2}) {action_object:<30}")
            else:
                Utils.print_color(c.Color.RED, f"{i:<2}) {action_object:<30} > {error}")

        print(f"Consecutive failures: {self.num_consecutive_failures(errors)}")

    def test_plan(
        self,
        test_plan: Plan,
        current_index: int,
        total: int,
        start_time: float,
        processed_count: int,
    ) -> Plan | None:

        player = Player(
            test_plan,
            plan_type=PlanType.REPLAY,
            config=self.config,
            save_directory=self.plan_folder,
        )

        # Get initial pose from plan (works with both full and stripped formats)
        initial_pose = test_plan.initial_pose
        if initial_pose:
            initial_pose = initial_pose.copy()  # Don't modify the original
            initial_pose["standing"] = initial_pose.get("isStanding", True)

        if not os.path.exists(f"{self.plan_folder}/{test_plan.name}"):
            os.makedirs(f"{self.plan_folder}/{test_plan.name}")

        cur_image = player.start()
        image_filename: str | None = (
            f"{self.plan_folder}/{test_plan.name}/0_cur_image.png"
        )
        previous_image_filename: str | None = None  # No previous image on first step
        cur_image.save(image_filename)

        if self.config.text_only:
            print("TEXT ONLY MODE")
            image_filename = None

        task_description = test_plan.task_description

        objects_in_scene_list = sorted(ItemCache.get_scene_types(test_plan.scene))

        # Remove PepperShaker (as it looks too much like saltshaker. Also done in scenario.py)
        if "PepperShaker" in objects_in_scene_list:
            objects_in_scene_list.remove("PepperShaker")

        # Remove StoveKnow as agent can toggle on/off stove burner directly
        if "StoveKnob" in objects_in_scene_list:
            objects_in_scene_list.remove("StoveKnob")

        objects_in_scene_str = ", ".join(objects_in_scene_list)
        self.fail_reason = None
        self.invalid_actions = []
        self.invalid_objects = []
        self.step_errors = None
        step_count = 0
        step_history: List[str] = []
        step_error_msgs: List[str | None] = []
        memory = ""
        suggested_plan_sequence = ""  # For full_steps mode
        response = None  # Model response for current step
        # instruction = instructions.get("clean_mirror", "")
        while True:
            # Clear current step and log buffer at start of each iteration
            # to prevent logs from being attributed to the previous step
            set_current_step(None)
            clear_log_buffer()

            first_action = step_count == 0
            mode = "text" if self.config.text_only else "image"

            prompt_params = PromptParams(
                mode=mode,
                first_action=first_action,
                task_description=task_description,
                objects_in_scene=objects_in_scene_str,
                feedback_type=self.config.feedback_type.value,
                include_common_sense=self.config.include_common_sense,
                include_simulation=True,
                hand_transparency=self.config.hand_transparency,
                previous_image=self.config.previous_image.value,
                use_memory=self.config.use_memory,
                full_steps=self.config.full_steps,
                suggested_plan_sequence=suggested_plan_sequence,
            )

            if not first_action:
                history_prompt = Prompts.history_to_prompt(
                    step_history,
                    step_error_msgs,
                    feedback_type=self.config.feedback_type.value,
                )
                prompt_params.action_history = history_prompt
                prompt_params.memories = memory

            prompt = render_prompt(prompt_params)

            if not _QUIET_MODE:
                self.print_history(step_history, step_error_msgs)
                if self.config.use_memory:
                    print(f"Memory: {memory}")

            # Calculate time elapsed and estimated time remaining
            current_time = time.time()
            elapsed_time = current_time - start_time
            elapsed_minutes = elapsed_time / 60

            if processed_count > 0:
                avg_time_per_item = elapsed_time / processed_count
                remaining_items = total - current_index - 1
                estimated_remaining_time = avg_time_per_item * remaining_items
                remaining_minutes = estimated_remaining_time / 60
                time_info = f"Elapsed: {elapsed_minutes:.1f}m, Est. Remaining: {remaining_minutes:.1f}m"
            else:
                time_info = (
                    f"Elapsed: {elapsed_minutes:.1f}m, Est. Remaining: calculating..."
                )

            config_info = self.config.get_output_suffix().split("__")
            config_str = f"({', '.join(config_info)})" if config_info else ""

            Utils.print_color(
                c.Color.YELLOW,
                f"[{current_index}/{total}] {self.test_plan_dir} : {test_plan.name} : {self.model_id}: {self.config.temperature} : {config_str} | {time_info}",
            )

            step_count += 1

            # Save a sample prompt on the 5th step for debugging/sanity checking
            if step_count == 5 and not self.replay:
                sample_prompt_dir = f"{self.plan_folder}/{test_plan.name}/sample_prompt"
                os.makedirs(sample_prompt_dir, exist_ok=True)

                sample_prompt_file = f"{sample_prompt_dir}/sample_prompt.txt"
                with open(sample_prompt_file, "w", encoding="utf-8") as f:
                    # Include image filenames at the front
                    f.write("=== IMAGES INCLUDED IN PROMPT ===\n")
                    if (
                        self.config.previous_image != c.PreviousImageType.NONE
                        and previous_image_filename
                    ):
                        f.write("Previous Image: previous_image.png\n")
                    if image_filename:
                        f.write("Current Image: current_image.png\n")
                    else:
                        f.write("No images (text-only mode)\n")
                    f.write("==================================\n\n")
                    f.write(prompt)

                # Copy image files to sample_prompt directory
                if (
                    self.config.previous_image != c.PreviousImageType.NONE
                    and previous_image_filename
                    and os.path.exists(previous_image_filename)
                ):
                    shutil.copy(
                        previous_image_filename,
                        f"{sample_prompt_dir}/previous_image.png",
                    )
                if image_filename and os.path.exists(image_filename):
                    shutil.copy(
                        image_filename, f"{sample_prompt_dir}/current_image.png"
                    )

            if self.replay:
                # Replay mode requires full plan with steps array
                if not test_plan.steps:
                    raise ValueError(
                        "Replay mode requires full plan.json with steps array. "
                        "Stripped benchmark plans cannot be used for replay."
                    )
                if (step_count - 1) >= len(test_plan.steps):
                    self.fail_reason = FailType.MAX_STEPS
                    break

                action_name = test_plan.steps[step_count - 1].action
                action = self.get_action_by_value(action_name)
                object_type = Utils.short_name(test_plan.steps[step_count - 1].object)
                memory = ""
            else:
                if image_filename is None:
                    print("Text Only: No image used for response generation.")

                # Determine previous image path based on previous_image config
                prev_image_to_send = (
                    previous_image_filename
                    if self.config.previous_image != c.PreviousImageType.NONE
                    else None
                )

                # ===================
                # Get model response
                # ===================
                action = None  # Will be set below or on exception
                try:
                    response = self.model.get_response(
                        image_filename, prev_image_to_send, prompt
                    )
                except ModelEmptyResponseError as e:
                    # Model didn't produce output - treat as invalid response and continue
                    log_print(f"[ERROR] Model empty response: {e}")
                    response = ""
                    action_name = "empty_response"
                    object_type = ""
                    action = c.Action.INVALID_RESPONSE
                except Exception as e:  # pylint: disable=broad-except
                    import traceback

                    stack_trace = traceback.format_exc()
                    log_print(
                        f"[ERROR] Exception getting model response: {e}\n{stack_trace}"
                    )
                    self.fail_reason = FailType.API_FAILURE
                    if player.plan.steps:
                        player.plan.steps[-1].log += f"[MODEL ERROR] {e}\n{stack_trace}"
                    break

                # AI model failed to return a response
                if response is None:
                    return None

                if action is None:  # Normal path - parse the response
                    action_name, object_type, err = Prompts.extract_action_object(
                        response, objects_in_scene_list
                    )

                    if err is None:
                        try:
                            action = self.get_action_by_value(action_name)
                        except KeyError:
                            log_print("[ERROR] Invalid action:", action_name)
                            self.invalid_actions.append(action_name)
                            action = c.Action.INVALID_ACTION

                        # If not a valid object, the action will be invalid
                        if (
                            object_type not in objects_in_scene_list
                            and action != c.Action.INVALID_ACTION
                        ):
                            log_print("[ERROR] Invalid object:", object_type)
                            self.invalid_objects.append(object_type)
                            action = c.Action.INVALID_OBJECT

                    else:
                        log_print(f"[ERROR] {err}")
                        action = c.Action.INVALID_RESPONSE

                # Only extract memories if memory feature is enabled
                if self.config.use_memory:
                    memory = Prompts.extract_memories(response)

                # Extract and format suggested plan sequence if full_steps is enabled
                if self.config.full_steps:
                    # Extract the raw suggested plan sequence as-is (preserving all text)
                    suggested_plan_sequence = (
                        Prompts.extract_suggested_plan_sequence_raw(response)
                    )

            step_history.append(f"{action_name} {object_type}")

            cur_image = player.step(action, object_type)

            # Sanity check: verify the image used in prompt matches the previous step's saved image
            if len(player.plan.steps) >= 2 and image_filename is not None:
                prev_step_image_path = os.path.join(
                    player.scenario.save_dir, player.plan.steps[-1].image_filename
                )
                if os.path.exists(prev_step_image_path):
                    with (
                        open(image_filename, "rb") as f1,
                        open(prev_step_image_path, "rb") as f2,
                    ):
                        if f1.read() != f2.read():
                            log_print(
                                f"[WARNING] Image mismatch: prompt used {image_filename} but step has {prev_step_image_path}"
                            )

            # Set the current step for logging (flushes any buffered log messages)
            set_current_step(player.plan.steps[-1])

            # Store the model's response and partial prompt on the step (only in non-replay mode)
            if not self.replay and response is not None:
                player.plan.steps[-1].model_response = response
                # Extract and store partial prompt (scene state + image info)
                # Use actual saved filenames from step objects for clarity
                current_step_image = player.plan.steps[-1].image_filename
                # Get previous step's image filename if previous_image is enabled
                if (
                    self.config.previous_image != c.PreviousImageType.NONE
                    and len(player.plan.steps) >= 2
                ):
                    prev_step_image = player.plan.steps[-2].image_filename
                else:
                    prev_step_image = None

                # Sanity check: verify step image contents match temp files used in prompt
                # verify_image_filename_consistency(
                #     image_filename,
                #     previous_image_filename,
                #     current_step_image,
                #     prev_step_image,
                #     player.scenario.save_dir,
                # )

                partial_prompt = extract_partial_prompt(
                    prompt, current_step_image, prev_step_image
                )
                player.plan.steps[-1].partial_prompt = partial_prompt

                if os.getenv("DETAILED_LOGGING", "0") == "0":
                    # slimmer DEBUG prompt printing
                    print("== PARTIAL PROMPT ==")
                    print(partial_prompt)
                    print("== RESPONSE ==")
                    print(response)
                    print("== END RESPONSE ==")

            if player.scenario.step_error is not None:
                step_error_msgs.append(player.scenario.step_error.error_msg)
                log_print(f"[STEP ERROR] {player.scenario.step_error.error_msg}")
            else:
                step_error_msgs.append(None)

            # Save previous image before saving current (for previous_image feature)
            if (
                self.config.previous_image != c.PreviousImageType.NONE
                and image_filename is not None
            ):
                previous_image_filename = (
                    f"{self.plan_folder}/{player.plan.name}/{step_count}_prev_image.png"
                )
                # Copy current to previous before overwriting
                if os.path.exists(image_filename):
                    if self.config.previous_image == c.PreviousImageType.GRAYSCALE:
                        # Convert to grayscale before saving
                        img = Image.open(image_filename)
                        gray_img = img.convert("L").convert(
                            "RGB"
                        )  # L=grayscale, back to RGB for consistency
                        gray_img.save(previous_image_filename)
                    else:
                        shutil.copy(image_filename, previous_image_filename)
            else:
                previous_image_filename = None

            # Save image
            image_filename = (
                f"{self.plan_folder}/{player.plan.name}/{step_count}_cur_image.png"
            )
            cur_image.save(image_filename)

            if self.config.text_only:
                image_filename = None
                previous_image_filename = None

            goals_completed = player.scenario.raw_plan.goal.evaluate_goals(
                player.scenario
            )
            if goals_completed:
                break

            # Check for looping patterns in the action history
            if Utils.is_sequence_looping(
                step_history, min_repetitions=MAX_ACTION_REPEATS
            ):
                self.fail_reason = FailType.MAX_REPEATS
                break

            # Check for too many consecutive failures in the step error messages
            num_recent_failures = self.num_consecutive_failures(step_error_msgs)
            if num_recent_failures >= MAX_CONSECUTIVE_FAILURES:
                self.fail_reason = FailType.MAX_FAILURES
                break

            # Two-tier step limit check
            if self.check_step_limit(
                step_count, object_type, step_history, player.scenario
            ):
                break

        # Clear current step for logging since we're done with steps
        set_current_step(None)

        # Delete temp image file
        if image_filename is not None and os.path.exists(image_filename):
            os.remove(image_filename)

        player.complete()

        self.step_errors = player.scenario.plan_errors

        if self.fail_reason is not None:
            player.plan.task_failed = True
            print("---------")
            Utils.print_color(c.Color.RED, self.fail_reason)
            # Log the failure reason to the last step if there is one
            if player.plan.steps:
                player.plan.steps[-1].log += f"[FAIL] {self.fail_reason}\n"
            print("---------")

        return player.plan


def run_tests(
    test_name: str,
    model: str,
    rep_number: int = 1,
    config: EvaluationConfig | None = None,
    print_results: bool = False,
    print_all_results_auto: bool = False,
    check_completion: bool = False,
) -> bool | None:
    """Programmatic entrypoint to run tests without CLI parsing.

    Example:
        from AsgardBench.Model.model_tester import run_tests
        run_tests(
            test_name="magt_benchmark",
            model="gpt-4o",
            config=EvaluationConfig(temperature=0.0),
        )
    """
    print("--= Running tests =--")

    assert config is not None, "config must be provided"

    if check_completion:
        tester = ModelTester(
            test_name,
            model,
            config=config,
            rep_number=rep_number,
        )
        is_complete = tester.check_completion()

        print("True" if is_complete else "False")
        return is_complete

    model_tester = ModelTester(
        test_name,
        model,
        config=config,
        rep_number=rep_number,
    )
    model_tester.run()
    return None


def main():
    print("--= Parsing AsgardBench Model Tester =--")
    parser = argparse.ArgumentParser(description="Run AI2Thor model testing")
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Test set directory name (e.g., 'magt_benchmark')",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet')",
    )
    parser.add_argument(
        "--rep",
        type=int,
        default=1,
        help="Repetition number for multiple runs of the same configuration (default: 1)",
    )

    parser.add_argument(
        "--print-results",
        action="store_true",
        help="Only print results without running tests",
    )
    parser.add_argument(
        "--print-all-results-auto",
        action="store_true",
        help="Auto-discover all tests and models under TEST_DIR and print aggregated results",
    )

    parser.add_argument(
        "--check-completion",
        action="store_true",
        help="Check whether all eligible plans for this test/model are completed (prints True/False and exits)",
    )

    # Let EvaluationConfig add its own arguments
    EvaluationConfig.add_argparse_args(parser)

    args = parser.parse_args()
    config = EvaluationConfig.from_args(args)

    print(f"Parsed arguments: {args}")

    completion = run_tests(
        test_name=args.test,
        model=args.model,
        rep_number=args.rep,
        config=config,
        print_results=args.print_results,
        print_all_results_auto=args.print_all_results_auto,
        check_completion=args.check_completion,
    )
    if args.check_completion and completion is not None:
        sys.exit(0 if completion else 1)


if __name__ == "__main__":
    main()
