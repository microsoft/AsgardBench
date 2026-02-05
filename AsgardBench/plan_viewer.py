import json
import os
import random
from dataclasses import dataclass, field
from typing import Callable, Generator

import streamlit as st
from PIL import Image, ImageDraw

from AsgardBench import constants as c
from AsgardBench.Model.test_results import FailType, StepExtension

DEFAULT_OUTPUT_DIR = f"{c.DATASET_DIR}"

# Debug flag to enable pose error analysis UI features
PLAN_DEBUG = False

# Status filter options: "All", "Success", and each FailType value
STATUS_FILTER_OPTIONS = ["All", "Success"] + [ft.value for ft in FailType]

# Step extension filter options: "All" and each StepExtension value
STEP_EXTENSION_FILTER_OPTIONS = ["All"] + [se.value for se in StepExtension]


def resolve_image_path(
    base_dir: str, plan_dir: str, step_idx: int, original_filename: str
) -> tuple[str, str]:
    """
    Resolve the image path, preferring step-based images if they exist.

    Args:
        base_dir: The output directory (e.g., Test/Test/...)
        plan_dir: The plan subdirectory name
        step_idx: The step index (0, 1, 2, ...)
        original_filename: The original image filename from plan.json

    Returns:
        Tuple of (image_path, display_filename) - the actual path to use and the filename to display
    """
    # Try step-based image first
    step_image_filename = f"{step_idx}_cur_image.png"
    step_image_path = os.path.join(base_dir, plan_dir, step_image_filename)

    if os.path.exists(step_image_path):
        return step_image_path, step_image_filename

    # Fall back to original filename
    if original_filename:
        original_path = os.path.join(base_dir, plan_dir, original_filename)
        return original_path, original_filename

    return "", ""


def resolve_prev_image_path(
    base_dir: str, plan_dir: str, step_idx: int
) -> tuple[str, str] | None:
    """
    Resolve the previous image path if it exists.

    Args:
        base_dir: The output directory (e.g., Test/Test/...)
        plan_dir: The plan subdirectory name
        step_idx: The step index (0, 1, 2, ...)

    Returns:
        Tuple of (image_path, display_filename) if exists, None otherwise
    """
    prev_image_filename = f"{step_idx}_prev_image.png"
    prev_image_path = os.path.join(base_dir, plan_dir, prev_image_filename)

    if os.path.exists(prev_image_path):
        return prev_image_path, prev_image_filename

    return None


def get_plan_status(output_dir: str, plan_dir: str) -> str | None:
    """
    Get the status of a plan from test_results.json.
    Returns: "Success", a FailType value string, or None if not found.
    """
    test_results_path = get_test_results_path(output_dir)
    if not test_results_path:
        return None

    try:
        with open(test_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        # Plan directory may have leading underscore for failed plans
        task_name = plan_dir.lstrip("_")

        for result in results:
            if result.get("task_name") == task_name:
                if result.get("task_failed", False):
                    fail_reason = result.get("fail_reason")
                    if fail_reason:
                        return fail_reason
                    return "Failed"  # Generic failed if no specific reason
                return "Success"

        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_test_results_path(output_dir: str) -> str | None:
    """
    Get the path to test_results.json for the given output directory.
    The test_results.json is in the parent directory of the Plans folder.
    """
    # Check if output_dir ends with "Plans" - if so, go to parent
    if os.path.basename(output_dir.rstrip("/")) == "Plans":
        parent_dir = os.path.dirname(output_dir.rstrip("/"))
    else:
        parent_dir = output_dir

    test_results_path = os.path.join(parent_dir, "test_results.json")
    if os.path.exists(test_results_path):
        return test_results_path
    return None


def get_manually_reviewed(output_dir: str, plan_dir: str) -> bool | None:
    """
    Get the manually_reviewed value for a plan from test_results.json.
    Returns True/False if found, None if not found.
    """
    test_results_path = get_test_results_path(output_dir)
    if not test_results_path:
        return None

    try:
        with open(test_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        # Plan directory may have leading underscore for failed plans
        task_name = plan_dir.lstrip("_")

        for result in results:
            if result.get("task_name") == task_name:
                return result.get("manually_reviewed", False)

        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_step_extension(output_dir: str, plan_dir: str) -> str | None:
    """
    Get the step_extension value for a plan from test_results.json.
    Returns the StepExtension value string if found, None if not found.
    """
    test_results_path = get_test_results_path(output_dir)
    if not test_results_path:
        return None

    try:
        with open(test_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        # Plan directory may have leading underscore for failed plans
        task_name = plan_dir.lstrip("_")

        for result in results:
            if result.get("task_name") == task_name:
                return result.get("step_extension", StepExtension.NONE.value)

        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def get_candidate_poses_errors(output_dir: str, plan_dir: str) -> int | None:
    """
    Get the candidate_poses_errors value for a plan from test_results.json.
    Returns the count if found, None if not found.
    """
    test_results_path = get_test_results_path(output_dir)
    if not test_results_path:
        return None

    try:
        with open(test_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        # Plan directory may have leading underscore for failed plans
        task_name = plan_dir.lstrip("_")

        for result in results:
            if result.get("task_name") == task_name:
                return result.get("candidate_poses_errors", 0)

        return None
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def set_manually_reviewed(output_dir: str, plan_dir: str, value: bool) -> bool:
    """
    Set the manually_reviewed value for a plan in test_results.json.
    Returns True if successful, False otherwise.
    """
    test_results_path = get_test_results_path(output_dir)
    if not test_results_path:
        return False

    try:
        with open(test_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        # Plan directory may have leading underscore for failed plans
        task_name = plan_dir.lstrip("_")

        # Find and update the result
        found = False
        for result in results:
            if result.get("task_name") == task_name:
                result["manually_reviewed"] = value
                found = True
                break

        if not found:
            return False

        # Write back to file
        with open(test_results_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return True
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return False


@st.dialog("Raw Plan JSON", width="large")
def _show_raw_json_dialog():
    """Display the raw plan.json content in a dialog."""
    plan_path = st.session_state.get("raw_json_path", "")
    if plan_path and os.path.exists(plan_path):
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                plan_data = json.load(f)
            # Display as formatted JSON
            st.json(plan_data, expanded=False)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            st.error(f"Error loading JSON: {e}")
    else:
        st.error("Plan file not found")

    if st.button("Close", key="close_raw_json"):
        st.session_state.show_raw_json = False
        st.rerun()


@st.dialog("Delete Plan", width="small")
def _show_delete_confirm_dialog():
    """Display a confirmation dialog for deleting a plan directory."""
    import shutil

    plan_name = st.session_state.get("delete_plan_name", "")
    plan_path = st.session_state.get("delete_plan_path", "")
    filtered_dirs = st.session_state.get("delete_filtered_dirs", [])
    current_idx = st.session_state.get("delete_current_idx", 0)

    st.markdown(f"### Are you sure you want to delete this plan?")
    st.markdown(f"**{plan_name}**")
    st.warning(
        "This action cannot be undone. The entire plan directory will be permanently deleted."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", key="cancel_delete", width="stretch"):
            st.session_state.show_delete_confirm = False
            st.rerun()
    with col2:
        if st.button("🗑️ Delete", key="confirm_delete", type="primary", width="stretch"):
            if plan_path and os.path.isdir(plan_path):
                try:
                    shutil.rmtree(plan_path)
                    st.session_state.show_delete_confirm = False

                    # Navigate to next item, or previous if at end
                    if len(filtered_dirs) > 1:
                        if current_idx < len(filtered_dirs) - 1:
                            # Go to next item
                            next_dir = filtered_dirs[current_idx + 1]
                        else:
                            # At end, go to previous item
                            next_dir = filtered_dirs[current_idx - 1]
                        st.query_params["selected_dir"] = next_dir
                    else:
                        # No more items, clear selection
                        st.query_params.clear()

                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting plan: {e}")
            else:
                st.error("Plan directory not found")


@st.dialog("Filter Plans", width="small")
def _show_filter_dialog():
    """Display the filter options in a dialog."""
    st.markdown("### Filter Options")

    # Status filter dropdown
    current_status = st.session_state.get("status_filter_value", "All")
    status_index = (
        STATUS_FILTER_OPTIONS.index(current_status)
        if current_status in STATUS_FILTER_OPTIONS
        else 0
    )
    new_status = st.selectbox(
        "Status filter:",
        options=STATUS_FILTER_OPTIONS,
        index=status_index,
        key="filter_dialog_status",
        help="Filter plans by success or specific failure type",
    )

    # Search filter for plans
    current_search = st.session_state.get("plan_search_filter_value", "")
    new_search = st.text_input(
        "Search plan directories:",
        value=current_search,
        key="filter_dialog_search",
        placeholder="Type to filter...",
        help="Filter plan directories by name",
    )

    # Unreviewed filter toggle
    current_unreviewed = st.session_state.get("filter_unreviewed", False)
    new_unreviewed = st.checkbox(
        "Show only unreviewed plans",
        value=current_unreviewed,
        key="filter_dialog_unreviewed",
        help="Only show plans that haven't been manually reviewed",
    )

    # Failed plans filter toggle
    current_failed_only = st.session_state.get("filter_failed_only", False)
    new_failed_only = st.checkbox(
        "Show only failed plans",
        value=current_failed_only,
        key="filter_dialog_failed_only",
        help="Only show plans where task_failed is True",
    )

    # Candidate poses errors filter toggle (only shown when PLAN_DEBUG is True)
    new_candidate_poses = False
    if PLAN_DEBUG:
        current_candidate_poses = st.session_state.get(
            "filter_candidate_poses_errors", False
        )
        new_candidate_poses = st.checkbox(
            "Show only candidate_poses_errors",
            value=current_candidate_poses,
            key="filter_dialog_candidate_poses",
            help="Only show plans with candidate_poses_errors > 0",
        )

    # Step extension filter dropdown
    current_step_ext = st.session_state.get("step_extension_filter_value", "All")
    step_ext_index = (
        STEP_EXTENSION_FILTER_OPTIONS.index(current_step_ext)
        if current_step_ext in STEP_EXTENSION_FILTER_OPTIONS
        else 0
    )
    new_step_ext = st.selectbox(
        "Step extension filter:",
        options=STEP_EXTENSION_FILTER_OPTIONS,
        index=step_ext_index,
        key="filter_dialog_step_ext",
        help="Filter plans by step extension status (None, Extended, Hit_Hard_Limit)",
    )

    st.divider()

    # Apply and Clear buttons
    btn_cols = st.columns(2)
    with btn_cols[0]:
        if st.button("Apply", type="primary", width="stretch"):
            st.session_state.status_filter_value = new_status
            st.session_state.plan_search_filter_value = new_search
            st.session_state.filter_unreviewed = new_unreviewed
            st.session_state.filter_failed_only = new_failed_only
            if PLAN_DEBUG:
                st.session_state.filter_candidate_poses_errors = new_candidate_poses
            st.session_state.step_extension_filter_value = new_step_ext
            st.session_state.show_filter_dialog = False
            st.rerun()

    with btn_cols[1]:
        if st.button("Clear All", width="stretch"):
            st.session_state.status_filter_value = "All"
            st.session_state.plan_search_filter_value = ""
            st.session_state.filter_unreviewed = False
            st.session_state.filter_failed_only = False
            if PLAN_DEBUG:
                st.session_state.filter_candidate_poses_errors = False
            st.session_state.step_extension_filter_value = "All"
            st.session_state.show_filter_dialog = False
            st.rerun()


# Cache for test_results.json data: {test_results_path: {task_name: [list of error messages]}}
# This avoids re-reading test_results.json files during repeated searches.
# Since test_results.json files are immutable once created, this cache never needs clearing.
# Cache is persisted to disk for faster searches across app restarts.
_CACHE_FILE_PATH = os.path.join(os.path.dirname(__file__), "cache", "search_cache.json")
_test_results_cache: dict[str, dict[str, list[str]]] = {}
_cache_loaded = False
_cache_timestamp: float = 0.0  # Unix timestamp of last cache save
_CACHE_FRESHNESS_SECONDS = 30 * 60  # 30 minutes


def _load_cache_from_disk() -> None:
    """Load the search cache from disk if it exists."""
    global _test_results_cache, _cache_loaded, _cache_timestamp
    if _cache_loaded:
        return
    _cache_loaded = True

    if os.path.exists(_CACHE_FILE_PATH):
        try:
            with open(_CACHE_FILE_PATH, "r") as f:
                data = json.load(f)
            # Handle both old format (just cache dict) and new format (with timestamp)
            if isinstance(data, dict) and "_timestamp" in data:
                _cache_timestamp = data.get("_timestamp", 0.0)
                _test_results_cache = data.get("cache", {})
            else:
                # Old format - just the cache dict
                _test_results_cache = data
                _cache_timestamp = 0.0
        except (json.JSONDecodeError, IOError):
            _test_results_cache = {}
            _cache_timestamp = 0.0


def _save_cache_to_disk() -> None:
    """Save the search cache to disk with timestamp."""
    global _cache_timestamp
    import time

    _cache_timestamp = time.time()
    try:
        os.makedirs(os.path.dirname(_CACHE_FILE_PATH), exist_ok=True)
        with open(_CACHE_FILE_PATH, "w") as f:
            json.dump({"_timestamp": _cache_timestamp, "cache": _test_results_cache}, f)
    except IOError:
        pass  # Silently fail if we can't write the cache


def _is_cache_fresh() -> bool:
    """Check if the cache is fresh (within the freshness window)."""
    import time

    _load_cache_from_disk()
    if _cache_timestamp == 0.0:
        return False
    return (time.time() - _cache_timestamp) < _CACHE_FRESHNESS_SECONDS


def get_test_results_errors_cached(test_results_path: str) -> dict[str, list[str]]:
    """
    Get error messages from a test_results.json file, using cache if available.

    Args:
        test_results_path: Path to the test_results.json file

    Returns:
        Dict mapping task_name to list of error messages from step_errors
    """
    global _test_results_cache

    # Ensure cache is loaded from disk
    _load_cache_from_disk()

    # Check cache first
    if test_results_path in _test_results_cache:
        return _test_results_cache[test_results_path]

    # Read from file
    task_errors: dict[str, list[str]] = {}
    try:
        with open(test_results_path, "r") as f:
            data = json.load(f)

        # Support both "results" and "test_results" keys
        results_key = "test_results" if "test_results" in data else "results"
        results = data.get(results_key, [])

        for result in results:
            task_name = result.get("task_name", "")
            if not task_name:
                continue

            error_messages = []
            seen_errors: set[str] = set()
            step_errors = result.get("step_errors") or []  # Handle None case
            for step_error in step_errors:
                error_msg = step_error.get("error_msg", "")
                if error_msg and error_msg not in seen_errors:
                    error_messages.append(error_msg)
                    seen_errors.add(error_msg)

            task_errors[task_name] = error_messages

    except (json.JSONDecodeError, IOError):
        # Skip files that can't be read, cache empty dict
        pass

    # Store in cache and persist to disk
    _test_results_cache[test_results_path] = task_errors
    _save_cache_to_disk()

    return task_errors


@dataclass
class SearchProgress:
    """Progress information for search operation."""

    directories_searched: int = 0
    test_results_found: int = 0
    plans_found: int = 0
    model_matches: int = 0
    error_matches: int = 0
    name_matches: int = 0
    config_matches: int = 0
    all_filters_match: int = 0
    matching_plans: list = field(default_factory=list)
    is_complete: bool = False
    was_abandoned: bool = False
    cache_hits: int = 0  # Track how many files were served from cache
    current_directory: str = ""  # Currently being examined
    used_cache_only: bool = False  # True if search used only cached data


def _search_cache_only(
    root_dir: str,
    error_text: str,
    model_type: str = "",
    plan_name: str = "",
    config: str = "",
) -> Generator[SearchProgress, None, None]:
    """
    Fast search using only cached data, no filesystem walk.
    Only searches test_results.json files that are already in the cache.
    """
    _load_cache_from_disk()

    progress = SearchProgress()
    progress.used_cache_only = True
    error_text_lower = error_text.lower() if error_text else ""
    model_type_lower = model_type.lower().strip() if model_type else ""
    plan_name_lower = plan_name.lower().strip() if plan_name else ""

    # Filter cache entries that are under root_dir
    for test_results_path, task_errors in _test_results_cache.items():
        if not test_results_path.startswith(root_dir):
            continue

        progress.test_results_found += 1
        progress.cache_hits += 1
        progress.current_directory = os.path.dirname(test_results_path)

        # Check config filter from directory path
        if config and config not in test_results_path:
            continue

        # Track config matches (only if filter is active)
        if config:
            progress.config_matches += 1

        # Check model type filter from directory path
        model_match = True
        if model_type_lower:
            model_match = False
            path_parts = test_results_path.split(os.sep)
            for part in path_parts:
                if "--" in part:
                    dir_model = part.split("--")[0].lower()
                    if model_type_lower in dir_model:
                        model_match = True
                        break

        if not model_match:
            continue

        # Determine the Plans directory path
        dirpath = os.path.dirname(test_results_path)
        plans_dir = os.path.join(dirpath, "Plans")

        # Process each task in the cached test results
        for task_name, error_messages in task_errors.items():
            progress.plans_found += 1

            # Check plan name filter
            name_match = not plan_name_lower or plan_name_lower in task_name.lower()
            if not name_match:
                continue

            # Track name matches (only if filter is active)
            if plan_name_lower and name_match:
                progress.name_matches += 1

            if model_match and model_type_lower:
                progress.model_matches += 1

            # Build the plan directory path
            plan_dir = os.path.join(plans_dir, task_name)
            if not os.path.exists(plan_dir):
                plan_dir = os.path.join(plans_dir, "_" + task_name)

            # If no error filter, match based on model and plan name
            if not error_text_lower:
                if model_match:
                    progress.all_filters_match += 1
                    if os.path.exists(plan_dir):
                        progress.matching_plans.append(plan_dir)
            else:
                # Check error messages for match
                error_match = False
                for error_msg in error_messages:
                    if error_text_lower in error_msg.lower():
                        error_match = True
                        progress.error_matches += 1
                        break

                if error_match and (model_match or not model_type_lower):
                    progress.all_filters_match += 1
                    if os.path.exists(plan_dir):
                        progress.matching_plans.append(plan_dir)

        # Yield progress periodically
        if progress.test_results_found % 10 == 0:
            yield progress

    progress.is_complete = True
    yield progress


def search_for_plans_generator(
    root_dir: str,
    error_text: str,
    model_type: str = "",
    plan_name: str = "",
    config: str = "",
    check_abandon: Callable[[], bool] | None = None,
) -> Generator[SearchProgress, None, None]:
    """
    Generator that searches for plans and yields progress updates.
    Searches test_results.json files for efficiency.

    If the cache is fresh (within 30 minutes), uses only cached data for fast results.

    Args:
        root_dir: Root directory to start searching from
        error_text: Error text to search for (case-insensitive partial match)
        model_type: Optional model type filter (e.g., 'gpt-4o')
        plan_name: Optional plan name filter (case-insensitive partial match)
        config: Optional config filter (e.g., 'T0_Fd_H60_C0_P2_I1_R1_S1_E0_M4096')
        check_abandon: Optional callable that returns True if search should be abandoned

    Yields:
        SearchProgress objects with current search state
    """
    # If cache is fresh, use cache-only search for fast results
    if _is_cache_fresh():
        yield from _search_cache_only(
            root_dir, error_text, model_type, plan_name, config
        )
        return

    progress = SearchProgress()
    error_text_lower = error_text.lower() if error_text else ""
    model_type_lower = model_type.lower().strip() if model_type else ""
    plan_name_lower = plan_name.lower().strip() if plan_name else ""

    # Walk through all directories looking for test_results.json files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Check for abandon request
        if check_abandon and check_abandon():
            progress.was_abandoned = True
            yield progress
            return

        progress.directories_searched += 1

        # Don't descend into Plans directories - they never contain test_results.json
        # This significantly speeds up the search
        if "Plans" in dirnames:
            dirnames.remove("Plans")

        # Check if this directory has a test_results.json
        if "test_results.json" in filenames:
            # Only update current_directory for directories with test_results.json
            progress.current_directory = dirpath
            progress.test_results_found += 1

            # Check config filter from directory path
            if config and config not in dirpath:
                # Yield progress periodically
                if progress.directories_searched % 20 == 0:
                    yield progress
                continue

            # Track config matches (only if filter is active)
            if config:
                progress.config_matches += 1

            # Check model type filter from directory path
            model_match = True  # Default to True if no filter
            if model_type_lower:
                model_match = False
                # Model type is the part before the first '--' in the directory name
                # e.g., 'gpt-4o--T0_Fs_H60_C0_P2_I1_R0_S1_E0_M4096--rep1'
                path_parts = dirpath.split(os.sep)
                for part in path_parts:
                    if "--" in part:
                        dir_model = part.split("--")[0].lower()
                        if model_type_lower in dir_model:
                            model_match = True
                            break

            # If model doesn't match and we have a model filter, skip this test_results
            if not model_match:
                # Yield progress periodically
                if progress.directories_searched % 20 == 0:
                    yield progress
                continue

            test_results_path = os.path.join(dirpath, "test_results.json")

            # Check if this file is already in cache
            was_cached = test_results_path in _test_results_cache
            if was_cached:
                progress.cache_hits += 1

            # Get errors from test_results.json (from cache or file)
            task_errors = get_test_results_errors_cached(test_results_path)

            # Determine the Plans directory path
            plans_dir = os.path.join(dirpath, "Plans")

            # Process each task in the test results
            for task_name, error_messages in task_errors.items():
                progress.plans_found += 1

                # Check plan name filter
                name_match = not plan_name_lower or plan_name_lower in task_name.lower()
                if not name_match:
                    continue

                # Track name matches (only if filter is active)
                if plan_name_lower and name_match:
                    progress.name_matches += 1

                if model_match and model_type_lower:
                    progress.model_matches += 1

                # Build the plan directory path
                # Plans can have underscore prefix for failed ones
                plan_dir = os.path.join(plans_dir, task_name)
                if not os.path.exists(plan_dir):
                    plan_dir = os.path.join(plans_dir, "_" + task_name)

                # If no error filter, match based on model and plan name
                if not error_text_lower:
                    if model_match:
                        progress.all_filters_match += 1
                        if os.path.exists(plan_dir):
                            progress.matching_plans.append(plan_dir)
                else:
                    # Check error messages for match
                    error_match = False
                    for error_msg in error_messages:
                        if error_text_lower in error_msg.lower():
                            error_match = True
                            progress.error_matches += 1
                            break

                    # Track combined matches
                    if error_match and (model_match or not model_type_lower):
                        progress.all_filters_match += 1
                        if os.path.exists(plan_dir):
                            progress.matching_plans.append(plan_dir)

            # Yield progress after processing each test_results.json
            yield progress

        # Also yield periodically during directory traversal
        elif progress.directories_searched % 100 == 0:
            yield progress

    progress.is_complete = True
    yield progress


def search_for_plans_with_error(
    root_dir: str,
    error_text: str,
    model_type: str = "",
    plan_name: str = "",
    config: str = "",
) -> list[str]:
    """
    Search for plans containing the specified error text (non-generator version).

    Args:
        root_dir: Root directory to start searching from
        error_text: Error text to search for (case-insensitive partial match)
        model_type: Optional model type filter (e.g., 'gpt-4o')
        plan_name: Optional plan name filter (case-insensitive partial match)
        config: Optional config filter (e.g., 'T0_Fd_H60_C0_P2_I1_R1_S1_E0_M4096')

    Returns:
        List of plan directory paths that contain the error
    """
    # Use the generator but just get the final result
    result = SearchProgress()
    for progress in search_for_plans_generator(
        root_dir, error_text, model_type, plan_name, config
    ):
        result = progress
    return result.matching_plans


def _show_search_form_inline():
    """Display the search form inline (replaces all other UI in sidebar)."""
    # Set a page marker that other code can check to avoid rendering
    st.session_state._current_page = "search_form"

    st.markdown("### 🔎 Search Plans")
    st.markdown("Search across directories for plans containing specific errors.")

    # Check for pending quick-select value BEFORE rendering the widget
    if "search_root_pending" in st.session_state:
        pending_value = st.session_state.search_root_pending
        del st.session_state.search_root_pending
        # Store in the persistent key
        st.session_state.search_root_dir = pending_value
        # Increment counter to force widget refresh
        st.session_state.search_root_counter = (
            st.session_state.get("search_root_counter", 0) + 1
        )

    # Get initial value
    initial_value = st.session_state.get("search_root_dir", "")

    # Use a dynamic key that changes when we need to refresh the widget
    widget_key = (
        f"search_form_root_input_{st.session_state.get('search_root_counter', 0)}"
    )

    # Search root directory
    new_search_root = st.text_input(
        "Search root directory (required):",
        value=initial_value,
        key=widget_key,
        placeholder="Enter the root directory to search from...",
        help="The starting directory for the search. All subdirectories will be searched.",
    )

    # Quick select buttons for common directories
    quick_cols = st.columns(3)
    with quick_cols[0]:
        if st.button("📁 Generated", key="search_generated", use_container_width=True):
            generated_path = os.path.abspath(os.path.join(os.getcwd(), c.DATASET_DIR))
            if os.path.exists(generated_path):
                st.session_state.search_root_pending = generated_path
                st.rerun()
    with quick_cols[1]:
        if st.button("📁 Test", key="search_test", use_container_width=True):
            test_path = os.path.abspath(os.path.join(os.getcwd(), c.TEST_DIR))
            if os.path.exists(test_path):
                st.session_state.search_root_pending = test_path
                st.rerun()
    with quick_cols[2]:
        blob_mount_path = "Test"
        if os.path.exists(blob_mount_path):
            if st.button("☁️ Blob", key="search_blob", use_container_width=True):
                st.session_state.search_root_pending = blob_mount_path
                st.rerun()

    st.divider()
    st.markdown("### Search Filters")

    # Simple approach: one session state key per widget, initialized once
    # The checkbox controls enabled state, the text input holds the value
    # We read values AFTER widgets render (via their return values)

    def filter_row(
        label: str, checkbox_key: str, text_key: str, placeholder: str, help_text: str
    ):
        """Render a filter row with checkbox and text input. Returns (enabled, value)."""
        # Initialize keys on first access
        if checkbox_key not in st.session_state:
            st.session_state[checkbox_key] = False
        if text_key not in st.session_state:
            st.session_state[text_key] = ""

        cols = st.columns([0.5, 5])
        with cols[0]:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            enabled = st.checkbox("", key=checkbox_key, label_visibility="collapsed")
        with cols[1]:
            value = st.text_input(
                label,
                key=text_key,
                placeholder=placeholder,
                help=help_text,
                disabled=not enabled,
            )
        return enabled, value

    model_enabled, model_text = filter_row(
        "Model type (partial match, case-insensitive):",
        "sf_model_chk",
        "sf_model_txt",
        'e.g., "gpt-4o", "claude", "gemini"',
        "Filter by model type from directory name (part before '--')",
    )

    error_enabled, error_text = filter_row(
        "Error text (partial match, case-insensitive):",
        "sf_error_chk",
        "sf_error_txt",
        'e.g., "No valid positions to place object found"',
        "Search for plans containing this error message text",
    )

    plan_enabled, plan_text = filter_row(
        "Plan name (partial match, case-insensitive):",
        "sf_plan_chk",
        "sf_plan_txt",
        'e.g., "cook__Egg_Pan(d)_FloorPlan10"',
        "Filter by plan/task name",
    )

    config_enabled, config_text = filter_row(
        "Config (partial match):",
        "sf_config_chk",
        "sf_config_txt",
        'e.g., "T0_Fd_H60_C0_P2_I1_R1_S1_E0_M4096"',
        "Filter by config string in directory path",
    )

    st.divider()

    # Buttons - validation uses the values returned by widgets above
    btn_cols = st.columns(2)
    with btn_cols[0]:
        # A filter is valid if enabled AND has text
        has_model = model_enabled and model_text
        has_error = error_enabled and error_text
        has_plan = plan_enabled and plan_text
        has_config = config_enabled and config_text

        # An enabled filter without text is invalid
        model_invalid = model_enabled and not model_text
        error_invalid = error_enabled and not error_text
        plan_invalid = plan_enabled and not plan_text
        config_invalid = config_enabled and not config_text

        has_valid_filter = has_model or has_error or has_plan or has_config
        has_invalid_filter = (
            model_invalid or error_invalid or plan_invalid or config_invalid
        )

        search_disabled = (
            not new_search_root or not has_valid_filter or has_invalid_filter
        )
        if st.button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
            disabled=search_disabled,
        ):
            if not os.path.exists(new_search_root):
                st.error(f"Directory not found: {new_search_root}")
            elif not os.path.isdir(new_search_root):
                st.error(f"Not a directory: {new_search_root}")
            else:
                # Save search parameters for the search functions
                st.session_state.search_root_dir = new_search_root
                st.session_state.search_error_text = error_text if error_enabled else ""
                st.session_state.search_model_type = model_text if model_enabled else ""
                st.session_state.search_plan_name = plan_text if plan_enabled else ""
                st.session_state.search_config = config_text if config_enabled else ""
                # Save the filter text for display even when disabled
                st.session_state.search_model_text_display = model_text
                st.session_state.search_error_text_display = error_text
                st.session_state.search_plan_text_display = plan_text
                st.session_state.search_config_text_display = config_text
                # Save enabled states for display
                st.session_state.search_model_enabled = model_enabled
                st.session_state.search_error_enabled = error_enabled
                st.session_state.search_plan_enabled = plan_enabled
                st.session_state.search_config_enabled = config_enabled
                st.session_state.show_search_dialog = False
                st.session_state.show_search_progress = True
                st.session_state.browse_search_results = (
                    False  # Exit browse mode to allow new search
                )
                st.session_state.search_abandoned = False
                st.session_state.search_completed = False  # Reset to trigger new search
                st.rerun()
                return  # Stop rendering to prevent ghost widgets

    with btn_cols[1]:
        if st.button("Close", use_container_width=True):
            st.session_state.show_search_dialog = False
            st.rerun()
            return  # Stop rendering

    # Show cache info
    cache_size = len(_test_results_cache)
    if cache_size > 0:
        st.caption(
            f"📦 {cache_size} test_results.json files cached for faster searches"
        )

    # Stop all further rendering to prevent ghost widgets
    st.stop()


def _show_search_progress_inline():
    """Display the search progress inline (replaces all other UI)."""
    # If user already clicked "Browse Results", don't show progress at all
    if st.session_state.get("browse_search_results", False):
        return  # Let the caller continue with normal UI

    # Two-phase rendering to clear ghost widgets:
    # Phase 1: Render empty page to clear old widgets, then rerun
    # Phase 2: Render the actual progress content

    if not st.session_state.get("_search_progress_cleared", False):
        # Phase 1: Mark as clearing, render minimal content, rerun
        st.session_state._search_progress_cleared = True
        st.markdown("### Starting search...")
        st.rerun()
        return

    # Phase 2: Clear the flag and render actual content
    st.session_state._search_progress_cleared = False

    main_container = st.container()
    with main_container:
        _render_search_progress()
    # Stop all further rendering
    st.stop()


def _render_search_progress():
    """Render search progress content (called from within the search dialog)."""
    # Set a page marker
    st.session_state._current_page = "search_progress"

    # Check if user already clicked "Browse Results" - if so, force close by rerunning
    # This prevents the greyed-out dialog from lingering
    if st.session_state.get("browse_search_results", False):
        st.session_state.show_search_progress = False
        st.rerun()
        return

    # Clear any form widget state to prevent ghost widgets
    form_keys = [
        "sf_model_chk",
        "sf_model_txt",
        "sf_error_chk",
        "sf_error_txt",
        "sf_plan_chk",
        "sf_plan_txt",
        "sf_config_chk",
        "sf_config_txt",
        "search_form_root_input_0",
        "search_form_root_input_1",
        "search_form_root_input_2",
    ]
    for key in form_keys:
        if key in st.session_state:
            del st.session_state[key]

    # Check if we already have search results (don't re-run search)
    existing_results = st.session_state.get("search_results", [])

    # Get search parameters
    root_dir = st.session_state.get("search_root_dir", "")
    error_text = st.session_state.get("search_error_text", "")
    model_type = st.session_state.get("search_model_type", "")
    plan_name = st.session_state.get("search_plan_name", "")
    config = st.session_state.get("search_config", "")

    st.markdown("### Search in Progress")
    st.markdown(f"**Root:** `{root_dir[:70]}{'...' if len(root_dir) > 70 else ''}`")

    st.divider()

    # Check if search already completed (results exist and we haven't started a new search)
    if existing_results and st.session_state.get("search_completed", False):
        # Show results from previous search, don't re-run
        final_progress = SearchProgress(
            is_complete=True,
            matching_plans=existing_results,
            all_filters_match=len(existing_results),
        )
    else:
        # Mark that we're running a new search
        st.session_state.search_completed = False

        # Run the search with progress updates
        # Create placeholders for live updates
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        progress_bar = st.progress(0, text="Starting search...")

        final_progress = None
        plan_name = st.session_state.get("search_plan_name", "")

        for progress in search_for_plans_generator(
            root_dir, error_text, model_type, plan_name, config
        ):
            final_progress = progress

            # Update status
            if progress.is_complete:
                if progress.used_cache_only:
                    status_placeholder.success("✅ Search complete (using cached data)")
                else:
                    status_placeholder.success("✅ Search complete!")
                progress_bar.progress(100, text="Complete")
            else:
                # Show current directory being examined (truncate if too long)
                current_dir = progress.current_directory
                if len(current_dir) > 100:
                    current_dir = "..." + current_dir[-97:]
                if progress.used_cache_only:
                    status_placeholder.info(f"⚡ Searching cache: `{current_dir}`")
                else:
                    status_placeholder.info(f"🔍 Searching: `{current_dir}`")
                # Progress bar is indeterminate since we don't know total
                pct = min(99, progress.test_results_found % 100)
                progress_bar.progress(
                    pct, text=f"Found {progress.test_results_found} test results..."
                )

            # Update metrics - three columns: title, filter value, count
            with metrics_placeholder.container():
                # Get enabled states for display
                model_enabled = st.session_state.get(
                    "search_model_enabled", bool(model_type)
                )
                error_enabled = st.session_state.get(
                    "search_error_enabled", bool(error_text)
                )
                plan_enabled_state = st.session_state.get(
                    "search_plan_enabled", bool(plan_name)
                )
                config_enabled_state = st.session_state.get(
                    "search_config_enabled", bool(config)
                )
                # Get display text (may have text even when disabled)
                model_text_display = st.session_state.get(
                    "search_model_text_display", model_type
                )
                error_text_display = st.session_state.get(
                    "search_error_text_display", error_text
                )
                plan_text_display = st.session_state.get(
                    "search_plan_text_display", plan_name
                )
                config_text_display = st.session_state.get(
                    "search_config_text_display", config
                )

                # Helper function to display a metric row (with optional greyed out style)
                def metric_row(
                    title: str, filter_val: str, count: int | str, enabled: bool = True
                ):
                    cols = st.columns([2, 2, 1])
                    if enabled:
                        with cols[0]:
                            st.markdown(f"**{title}**")
                        with cols[1]:
                            if filter_val:
                                st.markdown(f"`{filter_val}`")
                        with cols[2]:
                            st.markdown(f"**{count}**")
                    else:
                        # Greyed out style for disabled filters
                        with cols[0]:
                            st.markdown(
                                f"<span style='color: #999;'>{title}</span>",
                                unsafe_allow_html=True,
                            )
                        with cols[1]:
                            if filter_val:
                                st.markdown(
                                    f"<span style='color: #999;'>{filter_val}</span>",
                                    unsafe_allow_html=True,
                                )
                        with cols[2]:
                            st.markdown(
                                f"<span style='color: #999;'>—</span>",
                                unsafe_allow_html=True,
                            )

                metric_row("Test Results", "", progress.test_results_found)
                metric_row("Plans Found", "", progress.plans_found)
                # Show all filters - greyed out if disabled
                model_display = (
                    (model_text_display[:25] + "...")
                    if model_text_display and len(model_text_display) > 25
                    else (model_text_display or "")
                )
                metric_row(
                    "Model Matches",
                    model_display,
                    progress.model_matches if model_enabled else "—",
                    enabled=model_enabled,
                )
                error_display = (
                    (error_text_display[:30] + "...")
                    if error_text_display and len(error_text_display) > 30
                    else (error_text_display or "")
                )
                metric_row(
                    "Error Matches",
                    error_display,
                    progress.error_matches if error_enabled else "—",
                    enabled=error_enabled,
                )
                name_display = (
                    (plan_text_display[:25] + "...")
                    if plan_text_display and len(plan_text_display) > 25
                    else (plan_text_display or "")
                )
                metric_row(
                    "Name Matches",
                    name_display,
                    progress.name_matches if plan_enabled_state else "—",
                    enabled=plan_enabled_state,
                )
                config_display = (
                    (config_text_display[:25] + "...")
                    if config_text_display and len(config_text_display) > 25
                    else (config_text_display or "")
                )
                metric_row(
                    "Config Matches",
                    config_display,
                    progress.config_matches if config_enabled_state else "—",
                    enabled=config_enabled_state,
                )
                metric_row("✅ Matches All", "", progress.all_filters_match)

        # Store results and mark search as completed
        if final_progress:
            st.session_state.search_results = final_progress.matching_plans
            st.session_state.search_completed = True

    st.divider()

    # Show completion options
    if final_progress and final_progress.is_complete:
        result_count = len(final_progress.matching_plans)

        if result_count > 0:
            st.success(f"Found **{result_count}** matching plans")
        else:
            st.warning("No matching plans found")

        def on_browse_results_click():
            """Callback for Browse Results button."""
            st.session_state.browse_search_results = True
            st.session_state.show_search_progress = False
            st.session_state._search_progress_cleared = False
            # Reset toggle states
            for key in [
                "toggle_history",
                "toggle_memory",
                "toggle_observations",
                "toggle_updated_memory",
                "toggle_reasoning",
                "toggle_user_prompt",
                "toggle_agent_response",
                "toggle_thinking",
                "toggle_model_response",
                "toggle_partial_prompt",
                "toggle_log",
                "toggle_prev_image",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            # Select the first result
            if "search_results" in st.session_state and st.session_state.search_results:
                st.session_state.search_result_current_path = (
                    st.session_state.search_results[0]
                )

        def on_close_click():
            """Callback for Close button."""
            st.session_state.show_search_progress = False

        done_cols = st.columns(2)
        with done_cols[0]:
            st.button(
                "📊 Browse Results",
                type="primary",
                width="stretch",
                disabled=result_count == 0,
                key="search_browse_results_btn",
                on_click=on_browse_results_click,
            )
        with done_cols[1]:
            st.button(
                "Close",
                width="stretch",
                key="search_close_btn",
                on_click=on_close_click,
            )

    # Ensure nothing else renders after this
    st.stop()


def has_active_filters() -> bool:
    """Check if any filters are currently active."""
    status_filter = st.session_state.get("status_filter_value", "All")
    search_filter = st.session_state.get("plan_search_filter_value", "")
    unreviewed_filter = st.session_state.get("filter_unreviewed", False)
    failed_only_filter = st.session_state.get("filter_failed_only", False)
    candidate_poses_filter = (
        st.session_state.get("filter_candidate_poses_errors", False)
        if PLAN_DEBUG
        else False
    )
    step_ext_filter = st.session_state.get("step_extension_filter_value", "All")
    return (
        status_filter != "All"
        or search_filter != ""
        or unreviewed_filter
        or failed_only_filter
        or candidate_poses_filter
        or step_ext_filter != "All"
    )


def directory_picker_ui(
    current_output_dir, browsing_search_results=False, plan_count=0
):
    """Create a UI for directory picking with st-file-browser"""
    # Initialize session state for file browser visibility
    if "show_file_browser" not in st.session_state:
        st.session_state.show_file_browser = False

    st.text("")
    # Folder row - same layout as Filters and Search rows
    folder_row_cols = st.columns([2, 3])
    with folder_row_cols[0]:
        if st.button(
            "📁 Folder",
            help="Browse for directory",
            key="folder_picker",
            disabled=browsing_search_results,
        ):
            st.session_state.show_file_browser = True
            # Reset browser path to current output directory when opening (use absolute path)
            st.session_state.current_browser_path = os.path.abspath(current_output_dir)
            st.rerun()
    with folder_row_cols[1]:
        # Show plan count (not when browsing search results)
        if not browsing_search_results:
            st.markdown(f"{plan_count} plans")

    return current_output_dir, False  # False indicates no directory change


def show_file_browser(current_output_dir):
    """Show the st-file-browser for directory selection"""
    st.markdown("### Select Directory")

    # Add CSS for custom directory browser styling
    st.markdown(
        """
    <style>
    /* Style for directory buttons */
    div[data-testid="column"] button[data-testid="baseButton-secondary"] {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px;
        margin: 2px;
        text-align: left;
        background-color: #f8f9fa;
        transition: background-color 0.2s;
    }

    div[data-testid="column"] button[data-testid="baseButton-secondary"]:hover {
        background-color: #e9ecef;
        border-color: #007bff;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Initialize current_browser_path if not exists and ensure it's absolute
    if "current_browser_path" not in st.session_state:
        st.session_state.current_browser_path = os.path.abspath(current_output_dir)

    # Ensure current path is absolute to prevent relative path confusion
    current_path = os.path.abspath(st.session_state.current_browser_path)
    st.session_state.current_browser_path = current_path

    # Show current path as a black label
    browser_path = os.path.abspath(
        st.session_state.get("current_browser_path", current_output_dir)
    )
    st.markdown("**Current path:**")
    st.markdown(
        f"<div style='color: black; padding: 5px 0px; font-family: monospace;'>{browser_path}</div>",
        unsafe_allow_html=True,
    )

    # Direct path entry with Go button
    go_cols = st.columns([5, 1])
    with go_cols[0]:
        direct_path = st.text_input(
            "Direct path",
            key="direct_path_input",
            placeholder="Enter full path to a plan directory...",
            label_visibility="collapsed",
        )
    with go_cols[1]:
        if st.button("🚀 Go", key="go_direct_path", width="stretch"):
            if direct_path:
                # Clean up path (remove trailing slash if present)
                direct_path = direct_path.rstrip("/")

                # Check if path exists, if not try adding underscore prefix to last component
                if not os.path.exists(direct_path):
                    # Try adding underscore to the last path component
                    parent_dir = os.path.dirname(direct_path)
                    last_component = os.path.basename(direct_path)
                    underscore_path = os.path.join(parent_dir, "_" + last_component)
                    if os.path.exists(underscore_path):
                        direct_path = underscore_path

                if os.path.exists(direct_path) and os.path.isdir(direct_path):
                    # Check if it's a plan directory (contains plan.json)
                    if os.path.exists(os.path.join(direct_path, "plan.json")):
                        # Go directly to the plan
                        # Set output_dir to the parent directory (e.g., .../Plans/)
                        # and set selected_dir via query params to the plan folder name
                        parent_dir = os.path.dirname(direct_path)
                        plan_folder_name = os.path.basename(direct_path)
                        st.session_state.custom_output_dir = parent_dir
                        st.session_state.use_custom_dir = True
                        st.session_state.show_file_browser = False
                        if "current_browser_path" in st.session_state:
                            del st.session_state.current_browser_path
                        # Set query params to select this specific plan
                        st.query_params["selected_dir"] = plan_folder_name
                        st.rerun()
                    else:
                        # Navigate to the directory in the browser
                        st.session_state.current_browser_path = direct_path
                        st.rerun()
                else:
                    st.error(f"Path does not exist: {direct_path}")

    # Check if blob storage is mounted
    blob_mount_path = "Test"
    blob_mounted = os.path.exists(blob_mount_path) and os.path.isdir(blob_mount_path)

    # Navigation controls: Ok, Up, Generated, Test, Blob (if mounted), Cancel buttons
    main_cols = st.columns([1, 1])

    with main_cols[0]:
        # Create sub-columns for all the buttons in the left column
        # Add extra column if blob is mounted
        if blob_mounted:
            nav_cols = st.columns([1, 1, 1, 1, 1, 1], gap="small")
        else:
            nav_cols = st.columns([1, 1, 1, 1, 1], gap="small")

        with nav_cols[0]:
            # Ok button on the far left
            if st.button("✅ Ok", key="ok_browser", width="stretch"):
                selected_dir = os.path.abspath(
                    st.session_state.get("current_browser_path", current_output_dir)
                )
                st.session_state.custom_output_dir = selected_dir
                st.session_state.use_custom_dir = True
                st.session_state.show_file_browser = False
                # Reset browser path when closing
                if "current_browser_path" in st.session_state:
                    del st.session_state.current_browser_path
                st.success(f"Selected directory: {selected_dir}")
                st.rerun()

        with nav_cols[1]:
            # Define navigation boundary - don't allow going beyond a reasonable project root
            # Use current working directory or find a reasonable project root
            repo_root = os.path.abspath(os.getcwd())

            # If we're in a subdirectory, try to find a reasonable project boundary
            # Look for common project root indicators
            potential_roots = []
            temp_path = current_path
            while temp_path != os.path.dirname(
                temp_path
            ):  # Until we reach filesystem root
                if any(
                    os.path.exists(os.path.join(temp_path, indicator))
                    for indicator in [
                        ".git",
                        "setup.py",
                        "pyproject.toml",
                        "package.json",
                        ".project",
                    ]
                ):
                    potential_roots.append(temp_path)
                temp_path = os.path.dirname(temp_path)

            # Use the deepest (most specific) project root, or fall back to cwd
            if potential_roots:
                repo_root = potential_roots[0]

            # Check if we can go up (not at filesystem root and directory exists)
            parent_dir = os.path.dirname(current_path)
            can_go_up = (
                parent_dir != current_path  # Not at filesystem root
                and os.path.exists(parent_dir)  # Parent directory exists
                and os.path.isdir(parent_dir)  # Parent is actually a directory
            )

            if st.button(
                "⬆️ Up",
                key="up_browser",
                disabled=not can_go_up,
                width="stretch",
            ):
                if can_go_up:
                    st.session_state.current_browser_path = parent_dir
                    st.rerun()

        with nav_cols[2]:
            # Generated button - sets path to Generated
            if st.button("📁 Generated", key="generated_browser", width="stretch"):
                generated_path = os.path.abspath(
                    os.path.join(os.getcwd(), c.DATASET_DIR)
                )
                if os.path.exists(generated_path) and os.path.isdir(generated_path):
                    st.session_state.current_browser_path = generated_path
                    st.rerun()
                else:
                    st.warning(f"Generated directory not found: {generated_path}")

        with nav_cols[3]:
            # Test button - sets path to Test directory
            if st.button("📁 Test", key="test_browser", width="stretch"):
                test_path = os.path.abspath(os.path.join(os.getcwd(), c.TEST_DIR))
                if os.path.exists(test_path) and os.path.isdir(test_path):
                    st.session_state.current_browser_path = test_path
                    st.rerun()
                else:
                    st.warning(f"Test directory not found: {test_path}")

        # Blob button - only shown if blob storage is mounted
        if blob_mounted:
            with nav_cols[4]:
                if st.button("☁️ Blob", key="blob_browser", width="stretch"):
                    st.session_state.current_browser_path = blob_mount_path
                    st.rerun()

            with nav_cols[5]:
                # Cancel button
                if st.button("❌ Cancel", key="cancel_browser", width="stretch"):
                    st.session_state.show_file_browser = False
                    st.rerun()
        else:
            with nav_cols[4]:
                # Cancel button on the far right
                if st.button("❌ Cancel", key="cancel_browser", width="stretch"):
                    st.session_state.show_file_browser = False
                    st.rerun()

    # Right column is empty for layout

    st.markdown("</div>", unsafe_allow_html=True)

    # Use st-file-browser configured to show directories for navigation
    browser_path = os.path.abspath(
        st.session_state.get("current_browser_path", current_output_dir)
    )

    # Validate that the browser path exists and is a directory before using it
    if not os.path.exists(browser_path) or not os.path.isdir(browser_path):
        # Fallback to the current output directory if path is invalid
        browser_path = os.path.abspath(current_output_dir)
        st.session_state.current_browser_path = browser_path
        st.warning(f"Invalid path detected, falling back to: {browser_path}")

    # Use a hash of the path to create a unique key that changes when path changes
    path_hash = hash(browser_path) % 10000  # Keep it short to avoid key length issues

    # Check if we're in a /Test/{sub_directory} path (model results directory)
    # In this case, only show directories that contain a "Plans" subdirectory
    # We detect this by checking if the path contains /Test/ and has at least one more directory after it
    def is_in_test_subdir(path):
        """Check if path is like /something/Test/subdir (not just /something/Test)"""
        parts = path.rstrip("/").split("/")
        try:
            test_idx = parts.index("Test")
            # We're in a test subdir if there's at least one directory after "Test"
            return test_idx < len(parts) - 1
        except ValueError:
            return False

    is_test_subdir = is_in_test_subdir(browser_path)

    # Custom directory browser - show only directories in current path
    try:
        # Get all items in the current directory
        items = os.listdir(browser_path)

        # Filter to only directories
        directories = []
        for item in items:
            item_path = os.path.join(browser_path, item)
            if os.path.isdir(item_path):
                # If we're in a test subdirectory, only show dirs with Plans folder
                if is_test_subdir:
                    plans_path = os.path.join(item_path, "Plans")
                    has_plans = os.path.exists(plans_path) and os.path.isdir(plans_path)
                    if has_plans:
                        directories.append(item)
                else:
                    directories.append(item)

        # Sort directories alphabetically
        directories.sort()

        if directories:
            st.markdown("**Directories in current path:**")

            # Create a container for the directory list
            with st.container():
                # Show directories in a grid layout
                cols_per_row = 3
                for i in range(0, len(directories), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        idx = i + j
                        if idx < len(directories):
                            dir_name = directories[idx]
                            with cols[j]:
                                # Create clickable button for each directory
                                if st.button(
                                    f"📁 {dir_name}",
                                    key=f"dir_{path_hash}_{idx}",
                                    width="stretch",
                                ):
                                    # Navigate to this directory
                                    new_path = os.path.abspath(
                                        os.path.join(browser_path, dir_name)
                                    )
                                    if os.path.exists(new_path) and os.path.isdir(
                                        new_path
                                    ):
                                        # Check if this directory contains a "Plans" subdirectory
                                        plans_path = os.path.join(new_path, "Plans")
                                        if os.path.exists(plans_path) and os.path.isdir(
                                            plans_path
                                        ):
                                            # Auto-select the Plans directory
                                            st.session_state.custom_output_dir = (
                                                plans_path
                                            )
                                            st.session_state.use_custom_dir = True
                                            st.session_state.show_file_browser = False
                                            st.rerun()
                                        elif not is_test_subdir:
                                            # Only navigate into directories without Plans if not in a test subdir
                                            st.session_state.current_browser_path = (
                                                new_path
                                            )
                                            st.rerun()
                                        # If is_test_subdir and no Plans, do nothing (shouldn't happen due to filtering)
        else:
            st.info("No subdirectories found in current path.")

    except PermissionError:
        st.error("Permission denied: Cannot access this directory.")
    except Exception as e:
        st.error(f"Error reading directory: {e}")


def double_spaces(n: int) -> str:
    return "&nbsp;" * (n * 2)


# First, modify load_plan_data to extract both memory and updated_memory separately:
def load_plan_data(plan_path):
    """Load the plan data from the new format where steps are already aggregated."""
    with open(plan_path, "r") as f:
        plan = json.load(f)

    # Look for steps instead of actions
    steps = plan.get("steps", [])
    lines = []
    image_files = {}

    for idx, step in enumerate(steps):
        # Updated property names based on the Step class
        action_desc = step.get("action_desc", "")
        goal = step.get("goal", "")
        image_filename = step.get("image_filename", "")
        reasoning = step.get("reasoning", [])
        observations = step.get("observations", [])
        history = step.get("history", [])
        # Extract memory and updated_memory separately
        memory = step.get("memory", {})
        updated_memory = step.get("updated_memory", {})

        # Extract formatted prompt data
        formatted = step.get("formatted") or {}
        user_prompt = formatted.get("user_part", "")
        agent_response = formatted.get("agent_part", "")
        thinking = formatted.get("natural_language_agent_part", "")

        # Extract model response (from model testing)
        model_response = step.get("model_response", "")

        # Extract log data
        log = step.get("log", "")

        # Extract partial prompt (scene state + image info from model testing)
        partial_prompt = step.get("partial_prompt", "")

        # If updated_memory is empty but memory exists, use memory as updated_memory
        if not updated_memory and memory:
            updated_memory = memory

        # Add both to the tuple including the new prompt data fields
        lines.append(
            (
                action_desc,
                image_filename,
                reasoning,
                observations,
                goal,
                history,
                memory,
                updated_memory,
                user_prompt,
                agent_response,
                thinking,
                model_response,
                log,
                partial_prompt,
            )
        )

        # Keep track of image files
        if image_filename:
            image_files[image_filename] = image_filename

    return plan, lines, image_files


# Add this function to draw bounding boxes on images
def draw_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes on an image and return the new image with boxes"""
    if not bounding_boxes:
        return image

    # Create a copy of the image to draw on
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # Generate a consistent color for each object name
    colors = {}

    # Draw each bounding box
    for obj_name, box in bounding_boxes.items():
        if obj_name not in colors:
            # Create a consistent color based on the object name
            random.seed(hash(obj_name))
            colors[obj_name] = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255),
            )

        # Extract coordinates
        ulx, uly, lrx, lry = box

        # Draw rectangle
        draw.rectangle([ulx, uly, lrx, lry], outline=colors[obj_name], width=3)

        # Draw label at the top of the box
        draw.rectangle(
            [ulx, uly - 20, ulx + len(obj_name) * 8, uly], fill=colors[obj_name]
        )
        draw.text((ulx + 5, uly - 18), obj_name, fill=(255, 255, 255))

    return img_draw


def display(
    selected_dir,
    output_dir=None,
    show_history=True,
    show_memory=True,
    show_observations=True,
    show_updated_memory=True,
    show_reasoning=True,
    show_user_prompt=True,
    show_agent_response=True,
    show_thinking=True,
    show_model_response=True,
    show_partial_prompt=False,
    show_log=False,
    show_prev_image=False,
):
    """Display the alternative gallery view with steps grouped by image."""
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    plan_path = os.path.join(output_dir, selected_dir, "plan.json")

    if not os.path.exists(plan_path):
        st.error(f"No plan.json found in {selected_dir}")
        return

    plan, lines, _ = load_plan_data(plan_path)

    # Group lines by image filename
    grouped_lines = []
    current_group = []
    current_image = None

    # Update the grouping logic with the new tuple structure:
    for idx, (
        task,
        image_filename,
        reasoning,
        observations,
        goal,
        history,
        memory,
        updated_memory,
        user_prompt,
        agent_response,
        thinking,
        model_response,
        log,
        partial_prompt,
    ) in enumerate(lines):
        if image_filename != current_image:
            if current_group:
                grouped_lines.append((current_group, current_image))
            current_group = [
                (
                    idx,
                    task,
                    reasoning,
                    observations,
                    goal,
                    history,
                    memory,
                    updated_memory,
                    user_prompt,
                    agent_response,
                    thinking,
                    model_response,
                    log,
                    partial_prompt,
                )
            ]
            current_image = image_filename
        else:
            current_group.append(
                (
                    idx,
                    task,
                    reasoning,
                    observations,
                    goal,
                    history,
                    memory,
                    updated_memory,
                    user_prompt,
                    agent_response,
                    thinking,
                    model_response,
                    log,
                    partial_prompt,
                )
            )

    if current_group:  # Add the last group
        grouped_lines.append((current_group, current_image))

    # Add custom CSS for gallery view
    st.markdown(
        """
        <style>
        .gallery-group {
            margin-bottom: 20px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .gallery-task {
            padding: 4px 10px;
            line-height: 1.5;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Track if we've found the first matching error log for auto-scrolling
    first_error_match_found = False
    search_error_text_for_scroll = ""
    if st.session_state.get("browse_search_results", False):
        search_error_text_for_scroll = st.session_state.get("search_error_text", "")

    # Display groups with their images
    for group_idx, (group, image_filename) in enumerate(grouped_lines):
        # Get selected_idx from query params to highlight the selected step
        try:
            selected_idx = int(st.query_params.get("selected_idx", 0))
        except Exception:
            selected_idx = 0

        with st.container():
            # Calculate visible columns and their widths dynamically
            visible_columns = []
            column_widths = []
            column_indices = {}  # Maps column type to actual column index

            current_idx = 0

            # History column
            if show_history:
                visible_columns.append("history")
                column_widths.append(3)
                column_indices["history"] = current_idx
                current_idx += 1

            # Memory column
            if show_memory:
                visible_columns.append("memory")
                column_widths.append(2.5)
                column_indices["memory"] = current_idx
                current_idx += 1

            # Previous image column (optional, shown to the left of current image)
            if show_prev_image:
                visible_columns.append("prev_image")
                column_widths.append(4)
                column_indices["prev_image"] = current_idx
                current_idx += 1

            # Image column (always visible)
            visible_columns.append("image")
            column_widths.append(4)
            column_indices["image"] = current_idx
            current_idx += 1

            # Arrow column (always visible)
            visible_columns.append("arrow")
            column_widths.append(0.75)
            column_indices["arrow"] = current_idx
            current_idx += 1

            # Observations column
            if show_observations:
                visible_columns.append("observations")
                column_widths.append(3)
                column_indices["observations"] = current_idx
                current_idx += 1

            # Updated Memory column
            if show_updated_memory:
                visible_columns.append("updated_memory")
                column_widths.append(3)
                column_indices["updated_memory"] = current_idx
                current_idx += 1

            # Only show prompt columns if prompt_data is present for the step
            has_prompt_data = False
            for step in plan.get("steps", []):
                formatted = step.get("formatted") or {}
                if (
                    formatted.get("user_part")
                    or formatted.get("agent_part")
                    or formatted.get("natural_language_agent_part")
                ):
                    has_prompt_data = True
                    break

            if has_prompt_data:
                if show_user_prompt:
                    visible_columns.append("user_prompt")
                    column_widths.append(4)
                    column_indices["user_prompt"] = current_idx
                    current_idx += 1

                if show_agent_response:
                    visible_columns.append("agent_response")
                    column_widths.append(4)
                    column_indices["agent_response"] = current_idx
                    current_idx += 1

                if show_thinking:
                    visible_columns.append("thinking")
                    column_widths.append(4)
                    column_indices["thinking"] = current_idx
                    current_idx += 1

            # Check if any step has partial_prompt
            has_partial_prompt = False
            for step in plan.get("steps", []):
                if step.get("partial_prompt"):
                    has_partial_prompt = True
                    break

            if has_partial_prompt and show_partial_prompt:
                visible_columns.append("partial_prompt")
                column_widths.append(4)
                column_indices["partial_prompt"] = current_idx
                current_idx += 1

            # Check if any step has model_response
            has_model_response = False
            for step in plan.get("steps", []):
                if step.get("model_response"):
                    has_model_response = True
                    break

            if has_model_response and show_model_response:
                visible_columns.append("model_response")
                column_widths.append(4)
                column_indices["model_response"] = current_idx
                current_idx += 1

            # Check if any step has log
            has_log = False
            for step in plan.get("steps", []):
                if step.get("log"):
                    has_log = True
                    break

            if has_log and show_log:
                visible_columns.append("log")
                column_widths.append(4)
                column_indices["log"] = current_idx
                current_idx += 1

            # Reasoning column (after log)
            if show_reasoning:
                visible_columns.append("reasoning")
                column_widths.append(3)
                column_indices["reasoning"] = current_idx
                current_idx += 1

            cols = st.columns(column_widths)

            # Get the last task in the group
            (
                last_idx,
                last_action_desc,  # Changed from last_task
                last_reasoning,
                last_observations,
                last_goal,
                last_history,
                last_memory,
                last_updated_memory,
                last_user_prompt,
                last_agent_response,
                last_thinking,
                last_model_response,
                last_log,
                last_partial_prompt,
            ) = group[-1]

            # History column
            if show_history and "history" in column_indices:
                with cols[column_indices["history"]]:
                    if last_history and len(last_history) > 0:
                        st.markdown(
                            f"<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"<div style='font-weight:bold; margin-bottom:5px; color:#444;'>History:</div>",
                            unsafe_allow_html=True,
                        )

                        # Calculate how many columns to use based on history length
                        history_len = len(last_history)

                        if history_len > 24:
                            # Use 3 columns for longer history lists
                            history_cols = st.columns(3)

                            # Calculate items per column (distributed evenly)
                            items_per_col = (
                                history_len + 2
                            ) // 3  # Round up for even distribution

                            # Display items in first column
                            with history_cols[0]:
                                for i in range(0, items_per_col):
                                    if i < history_len:
                                        st.markdown(
                                            f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{last_history[i]}</div>",
                                            unsafe_allow_html=True,
                                        )

                            # Display items in second column
                            with history_cols[1]:
                                for i in range(items_per_col, items_per_col * 2):
                                    if i < history_len:
                                        st.markdown(
                                            f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{last_history[i]}</div>",
                                            unsafe_allow_html=True,
                                        )

                            # Display items in third column
                            with history_cols[2]:
                                for i in range(items_per_col * 2, history_len):
                                    st.markdown(
                                        f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{last_history[i]}</div>",
                                        unsafe_allow_html=True,
                                    )
                        else:
                            # Use 2 columns for shorter history lists
                            history_cols = st.columns(2)
                            items_in_first_col = min(12, history_len)

                            # Display items in first column (up to 12)
                            with history_cols[0]:
                                for i in range(items_in_first_col):
                                    st.markdown(
                                        f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{last_history[i]}</div>",
                                        unsafe_allow_html=True,
                                    )

                            # Display remaining items in second column
                            with history_cols[1]:
                                for i in range(items_in_first_col, history_len):
                                    st.markdown(
                                        f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{last_history[i]}</div>",
                                        unsafe_allow_html=True,
                                    )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Memory column
            if show_memory and "memory" in column_indices:
                with cols[column_indices["memory"]]:
                    # Create a container for memory list
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        # Display memory if available - UPDATED TO USE STEP'S MEMORY
                        memory_items = []
                        for step in plan.get("steps", []):
                            if (
                                step.get("image_filename") == image_filename
                                and step.get("action_desc") == last_action_desc
                            ):
                                memory_items = step.get("memory", [])
                                break

                        if memory_items:
                            st.markdown(
                                "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Memory:</div>",
                                unsafe_allow_html=True,
                            )

                            # Display each memory item
                            for memory_item in memory_items:
                                st.markdown(
                                    f"<div style='padding-left:10px; color:#666; font-size:0.9em; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;'>{memory_item}</div>",
                                    unsafe_allow_html=True,
                                )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Previous image column (optional)
            if show_prev_image and "prev_image" in column_indices:
                with cols[column_indices["prev_image"]]:
                    prev_result = resolve_prev_image_path(
                        output_dir, selected_dir, last_idx
                    )
                    if prev_result:
                        prev_image_path, prev_display_filename = prev_result
                        try:
                            prev_img = Image.open(prev_image_path)
                            st.image(prev_img, width="stretch")
                            st.markdown(
                                f"<div style='text-align:center; font-size:0.8em; color:#888; margin-top:2px;'>{prev_display_filename}</div>",
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            st.error(f"Error loading prev image: {e}")
                    else:
                        st.markdown(
                            "<div style='text-align:center; color:#888; margin-top:150px;'>No prev image</div>",
                            unsafe_allow_html=True,
                        )

            # Image column (always visible)
            with cols[column_indices["image"]]:
                if image_filename and image_filename != "--":
                    # Try step-based image first, fall back to original
                    image_path, display_filename = resolve_image_path(
                        output_dir, selected_dir, last_idx, image_filename
                    )
                    if image_path and os.path.exists(image_path):
                        # Get bounding boxes and action_success from the plan for this image
                        bounding_boxes = {}
                        action_success = None
                        for step in plan.get("steps", []):
                            if step.get("image_filename") == image_filename:
                                bounding_boxes = step.get("object_bounding_boxes", {})
                                action_success = step.get("action_success")
                                break

                        # Load and display the image
                        try:
                            img = Image.open(image_path)
                            # Apply bounding boxes if enabled
                            if st.session_state.toggle_boxes and bounding_boxes:
                                img = draw_bounding_boxes(img, bounding_boxes)
                            # Add red border if action failed (action_success is explicitly False)
                            if action_success is False:
                                border_width = 8
                                draw = ImageDraw.Draw(img)
                                width, height = img.size
                                # Draw red rectangle border
                                for i in range(border_width):
                                    draw.rectangle(
                                        [i, i, width - 1 - i, height - 1 - i],
                                        outline="red",
                                    )
                            st.image(img, width="stretch")
                            # Show image filename - red if action failed
                            if action_success is False:
                                st.markdown(
                                    f"<div style='text-align:center; font-size:0.8em; color:#ff0000; font-weight:bold; margin-top:2px;'>{display_filename}</div>",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.markdown(
                                    f"<div style='text-align:center; font-size:0.8em; color:#888; margin-top:2px;'>{display_filename}</div>",
                                    unsafe_allow_html=True,
                                )
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    else:
                        st.warning(f"Image not found: {image_filename}")
                else:
                    st.info("No image for this group")

            # Arrow column (always visible)
            with cols[column_indices["arrow"]]:
                st.markdown(
                    """
                    <div style="display:flex; height:100%; align-items:center; justify-content:center; margin-top:150px;">
                        <span style="font-size:24px; color:#666;">➡️</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Observations column
            if show_observations and "observations" in column_indices:
                with cols[column_indices["observations"]]:
                    # Create a container for observations only
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        # Display observations for this group if available
                        has_observations = any(
                            observations
                            for _, _, _, observations, _, _, _, _, _, _, _, _, _, _ in group
                            if observations
                        )
                        if has_observations:
                            st.markdown(
                                "<div style='font-weight:bold; margin-top:10px; margin-bottom:5px; color:#444;'>Observations:</div>",
                                unsafe_allow_html=True,
                            )

                            for (
                                idx,
                                task,
                                reasoning,
                                observations,
                                goal,
                                history,
                                memory,
                                updated_memory,
                                user_prompt,
                                agent_response,
                                thinking,
                                model_response,
                                log,
                                partial_prompt,
                            ) in group:
                                if observations:
                                    for observation in observations:
                                        st.markdown(
                                            f"<div style='padding-left:20px; font-style:italic; color:#666; font-size:0.9em;'>{observation}</div>",
                                            unsafe_allow_html=True,
                                        )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Updated Memory column
            if show_updated_memory and "updated_memory" in column_indices:
                with cols[column_indices["updated_memory"]]:
                    # Create a container for memory
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        # Show updated memory content
                        st.markdown(
                            "<div style='font-weight:bold; margin-top:10px; margin-bottom:5px; color:#444;'>Updated Memory:</div>",
                            unsafe_allow_html=True,
                        )

                        # Check if updated_memory is a dictionary or list
                        if isinstance(last_updated_memory, dict) and isinstance(
                            last_memory, dict
                        ):
                            # First, show items that are in memory but not in updated_memory (removed items)
                            for obj_name, memory_data in last_memory.items():
                                if obj_name not in last_updated_memory:
                                    # Display crossed out object that was removed
                                    st.markdown(
                                        f"<div style='padding-left:20px; font-weight:bold; color:#999; font-size:0.9em; text-decoration:line-through;'>{obj_name}:</div>",
                                        unsafe_allow_html=True,
                                    )

                                    # Display crossed out properties
                                    if isinstance(memory_data, dict):
                                        for prop, value in memory_data.items():
                                            if prop != "name" and value is not None:
                                                st.markdown(
                                                    f"<div style='padding-left:30px; color:#999; font-size:0.85em; text-decoration:line-through;'>{prop}: {value}</div>",
                                                    unsafe_allow_html=True,
                                                )

                            # For each object in updated_memory
                            for obj_name, memory_data in last_updated_memory.items():
                                # Check if this object exists in original memory
                                # If last_memory is None/empty or doesn't contain this object, mark as modified
                                if not last_memory or obj_name not in last_memory:
                                    is_modified = True
                                else:
                                    original_obj_data = last_memory.get(obj_name, {})
                                    # Mark as modified if object is new or has different data
                                    is_modified = original_obj_data != memory_data

                                # Display object name (in red if modified)
                                if is_modified:
                                    st.markdown(
                                        f"<div style='padding-left:20px; font-weight:bold; color:#e63946; font-size:0.9em;'>{obj_name}:</div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<div style='padding-left:20px; font-weight:bold; color:#555; font-size:0.9em;'>{obj_name}:</div>",
                                        unsafe_allow_html=True,
                                    )

                                # Display memory properties
                                if isinstance(memory_data, dict):
                                    # First check for properties in original that aren't in updated (removed properties)
                                    if obj_name in last_memory and isinstance(
                                        last_memory[obj_name], dict
                                    ):
                                        original_obj_data = last_memory[obj_name]
                                        for prop, value in original_obj_data.items():
                                            if (
                                                prop != "name"
                                                and prop not in memory_data
                                            ):
                                                # Show crossed out removed property
                                                st.markdown(
                                                    f"<div style='padding-left:30px; color:#999; font-size:0.85em; text-decoration:line-through;'>{prop}: {value}</div>",
                                                    unsafe_allow_html=True,
                                                )

                                    # Then show current properties
                                    for prop, value in memory_data.items():
                                        if (
                                            prop != "name" and value is not None
                                        ):  # Skip name as we already show it
                                            # Check if this property changed
                                            original_value = (
                                                original_obj_data.get(prop, None)
                                                if obj_name in last_memory
                                                and isinstance(
                                                    last_memory[obj_name], dict
                                                )
                                                else None
                                            )
                                            has_changed = original_value != value

                                            # Display in red if changed
                                            if has_changed:
                                                st.markdown(
                                                    f"<div style='padding-left:30px; color:#e63946; font-size:0.85em;'>{prop}: {value}</div>",
                                                    unsafe_allow_html=True,
                                                )
                                            else:
                                                st.markdown(
                                                    f"<div style='padding-left:30px; color:#666; font-size:0.85em;'>{prop}: {value}</div>",
                                                    unsafe_allow_html=True,
                                                )
                                else:
                                    # Handle case where memory_data is not a dictionary
                                    st.markdown(
                                        f"<div style='padding-left:30px; color:#666; font-size:0.85em;'>{memory_data}</div>",
                                        unsafe_allow_html=True,
                                    )
                        elif isinstance(last_memory, list) and isinstance(
                            last_updated_memory, list
                        ):
                            # First show items that were removed (in memory but not in updated_memory)
                            for item in last_memory:
                                if item not in last_updated_memory:
                                    st.markdown(
                                        f"<div style='padding-left:20px; color:#999; font-size:0.9em; text-decoration:line-through;'>{item}</div>",
                                        unsafe_allow_html=True,
                                    )

                            # Then show current items
                            for item in last_updated_memory:
                                # Consider item new if it wasn't in original memory
                                is_new = (
                                    item not in last_memory if last_memory else True
                                )

                                if is_new:
                                    st.markdown(
                                        f"<div style='padding-left:20px; color:#e63946; font-size:0.9em;'>{item}</div>",
                                        unsafe_allow_html=True,
                                    )
                                else:
                                    st.markdown(
                                        f"<div style='padding-left:20px; color:#666; font-size:0.9em;'>{item}</div>",
                                        unsafe_allow_html=True,
                                    )
                        elif last_updated_memory:  # Handle any other case
                            # Just display updated_memory normally
                            if isinstance(last_updated_memory, dict):
                                for (
                                    obj_name,
                                    memory_data,
                                ) in last_updated_memory.items():
                                    st.markdown(
                                        f"<div style='padding-left:20px; font-weight:bold; color:#555; font-size:0.9em;'>{obj_name}:</div>",
                                        unsafe_allow_html=True,
                                    )
                                    # Display properties
                                    if isinstance(memory_data, dict):
                                        for prop, value in memory_data.items():
                                            if prop != "name" and value is not None:
                                                st.markdown(
                                                    f"<div style='padding-left:30px; color:#666; font-size:0.85em;'>{prop}: {value}</div>",
                                                    unsafe_allow_html=True,
                                                )
                            elif isinstance(last_updated_memory, list):
                                for item in last_updated_memory:
                                    st.markdown(
                                        f"<div style='padding-left:20px; color:#666; font-size:0.9em;'>{item}</div>",
                                        unsafe_allow_html=True,
                                    )

                        st.markdown("</div>", unsafe_allow_html=True)

            # User Prompt column
            if show_user_prompt and "user_prompt" in column_indices:
                with cols[column_indices["user_prompt"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>User Prompt:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_user_prompt:
                            # Display user prompt in a scrollable text area
                            st.text_area(
                                "User Prompt",
                                value=last_user_prompt,
                                height=400,
                                key=f"user_prompt_{selected_dir}_{last_idx}",
                                label_visibility="collapsed",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Agent Response column
            if show_agent_response and "agent_response" in column_indices:
                with cols[column_indices["agent_response"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Agent Response:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_agent_response:
                            # Display agent response in a scrollable text area
                            st.text_area(
                                "Agent Response",
                                value=last_agent_response,
                                height=400,
                                key=f"agent_response_{selected_dir}_{last_idx}",
                                label_visibility="collapsed",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Thinking column
            if show_thinking and "thinking" in column_indices:
                with cols[column_indices["thinking"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Thinking:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_thinking:
                            # Display thinking content in a scrollable text area
                            st.text_area(
                                "Thinking",
                                value=last_thinking,
                                height=400,
                                key=f"thinking_{selected_dir}_{last_idx}",
                                label_visibility="collapsed",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Partial Prompt column
            if show_partial_prompt and "partial_prompt" in column_indices:
                with cols[column_indices["partial_prompt"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Partial Prompt:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_partial_prompt:
                            # Display partial prompt content in a scrollable text area
                            st.text_area(
                                "Partial Prompt",
                                value=last_partial_prompt,
                                height=400,
                                key=f"partial_prompt_{selected_dir}_{last_idx}",
                                label_visibility="collapsed",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Model Response column
            if show_model_response and "model_response" in column_indices:
                with cols[column_indices["model_response"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Model Response:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_model_response:
                            # Display model response content in a scrollable text area
                            st.text_area(
                                "Model Response",
                                value=last_model_response,
                                height=400,
                                key=f"model_response_{selected_dir}_{last_idx}",
                                label_visibility="collapsed",
                            )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Log column
            if show_log and "log" in column_indices:
                with cols[column_indices["log"]]:
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            "<div style='font-weight:bold; margin-bottom:5px; color:#444;'>Log:</div>",
                            unsafe_allow_html=True,
                        )

                        if last_log:
                            # Check for search error text match
                            search_error_text = st.session_state.get(
                                "search_error_text", ""
                            )
                            browsing_results = st.session_state.get(
                                "browse_search_results", False
                            )
                            has_search_error_match = (
                                browsing_results
                                and search_error_text
                                and search_error_text.lower() in last_log.lower()
                            )

                            # Check if this is the first matching error for auto-scroll
                            is_first_error_match = (
                                not first_error_match_found and has_search_error_match
                            )
                            if is_first_error_match:
                                first_error_match_found = True

                            if has_search_error_match:
                                log_key = f"log_{selected_dir}_{last_idx}"
                                bg_color = "#ffcccc"

                                # Display log in a styled div instead of text_area
                                import html

                                escaped_log = html.escape(last_log)
                                # Add anchor ID for first matching error
                                anchor_id = (
                                    ' id="first-error-match"'
                                    if is_first_error_match
                                    else ""
                                )
                                st.markdown(
                                    f"""<div{anchor_id} style='background-color: {bg_color}; padding: 10px;
                                        border-radius: 5px; height: 400px; overflow-y: auto;
                                        font-family: monospace; font-size: 14px; white-space: pre-wrap;'>{escaped_log}</div>""",
                                    unsafe_allow_html=True,
                                )
                            else:
                                st.text_area(
                                    "Log",
                                    value=last_log,
                                    height=400,
                                    key=f"log_{selected_dir}_{last_idx}",
                                    label_visibility="collapsed",
                                )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Reasoning column (after log)
            if show_reasoning and "reasoning" in column_indices:
                with cols[column_indices["reasoning"]]:
                    # Create a container for reasoning and action
                    with st.container():
                        st.markdown(
                            "<div style='padding:10px;border-radius:5px;'>",
                            unsafe_allow_html=True,
                        )

                        # Display reasoning for this group if available
                        has_reasoning = any(
                            reasoning
                            for _, _, reasoning, _, _, _, _, _, _, _, _, _, _, _ in group
                            if reasoning
                        )
                        if has_reasoning:
                            st.markdown(
                                "<div style='font-weight:bold; margin-top:10px; margin-bottom:5px; color:#444;'>Reasoning:</div>",
                                unsafe_allow_html=True,
                            )

                            for (
                                idx,
                                task,
                                reasoning,
                                observations,
                                goal,
                                history,
                                memory,
                                updated_memory,
                                user_prompt,
                                agent_response,
                                thinking,
                                model_response,
                                log,
                                partial_prompt,
                            ) in group:
                                if reasoning:
                                    for reason in reasoning:
                                        st.markdown(
                                            f"<div style='padding-left:20px; font-style:italic; color:#666; font-size:0.9em;'>{reason}</div>",
                                            unsafe_allow_html=True,
                                        )

                        # Always show the Action section regardless of training mode
                        st.markdown(
                            "<div style='font-weight:bold; margin-top:10px; margin-bottom:5px; color:#444;'>Action:</div>",
                            unsafe_allow_html=True,
                        )

                        # Display the task as plain text but still make it clickable
                        st.markdown(
                            f"<div style='padding-left:10px; color:#000; font-size:1.05em;'>"
                            f"<a href='#' onclick=\"window.location.search='?selected_dir={selected_dir}&selected_idx={last_idx}&view_mode=detail'; return false;\" "
                            f"style='text-decoration:none; color:inherit;'>{last_action_desc}</a>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        st.markdown("</div>", unsafe_allow_html=True)

            # Add some space between groups
            st.markdown(
                "<div style='margin-bottom:15px;'></div>", unsafe_allow_html=True
            )

    # After all groups are rendered, inject JavaScript to scroll to first error match
    if first_error_match_found and search_error_text_for_scroll:
        import streamlit.components.v1 as components

        components.html(
            """
            <script>
                // Wait for DOM to be ready, then scroll to first error match
                setTimeout(function() {
                    var element = parent.document.getElementById('first-error-match');
                    if (element) {
                        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }, 200);
            </script>
            """,
            height=0,
        )


def main():
    # Initialize session state for custom directory
    if "custom_output_dir" not in st.session_state:
        st.session_state.custom_output_dir = DEFAULT_OUTPUT_DIR
    if "use_custom_dir" not in st.session_state:
        st.session_state.use_custom_dir = False
    # Always start on the select directory page
    if "show_file_browser" not in st.session_state:
        st.session_state.show_file_browser = True

    # Set wider page layout
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1800px !important;
            padding-top: 2rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 0.5rem;
        }
        .stApp {
            max-width: 95%;
            margin: 0 auto;
        }
        /* Reduce vertical spacing between elements */
        .stMarkdown, .stSelectbox, .stCaption {
            margin-bottom: 0 !important;
        }
        h3 {
            margin-top: 0 !important;
            margin-bottom: 0.25rem !important;
        }
        .stBlock {
            display: block;
        }
        /* Style toggle buttons container to be horizontal */
        .stToggle {
            display: inline-block;
            margin-right: 20px;
        }
        /* Reduce bullet point spacing */
        .stMarkdown ul {
            margin-top: 0px;
            margin-bottom: 0px;
            padding-left: 1rem;
        }
        .stMarkdown li {
            margin-bottom: 2px !important;
            line-height: 1.2 !important;
        }
        /* Goal bullet points with reduced line height */
        .goals-section p {
            line-height: 1.0 !important;
            margin-bottom: 2px !important;
        }

        /* Style for the third column toggle buttons */
        [data-testid="column"]:nth-child(3) {
            padding-left: 20px;
        }

        /* More aggressive approach - target third column and reduce all spacing */
        [data-testid="column"]:nth-child(3) > div > div {
            margin-bottom: 0px !important;
            margin-top: 0px !important;
            padding-bottom: 2px !important;
            padding-top: 2px !important;
        }

        /* Target toggle elements specifically */
        [data-testid="column"]:nth-child(3) .stCheckbox,
        [data-testid="column"]:nth-child(3) [data-testid="stCheckbox"] {
            margin-bottom: 0px !important;
            margin-top: 0px !important;
            padding-bottom: 2px !important;
            padding-top: 2px !important;
        }

        /* Try to override any vertical spacing in the column */
        [data-testid="column"]:nth-child(3) * {
            line-height: 1.2 !important;
        }

        [data-testid="column"]:nth-child(3) [data-testid="stToggle"] label {
            font-weight: 500;
            font-size: 0.9em;
        }

        </style>
    """,
        unsafe_allow_html=True,
    )

    # Get current output directory
    current_output_dir = (
        st.session_state.custom_output_dir
        if st.session_state.use_custom_dir
        else DEFAULT_OUTPUT_DIR
    )

    # Check if file browser should be shown
    if st.session_state.get("show_file_browser", False):
        show_file_browser(current_output_dir)
        return  # Exit early, hiding all other UI

    # List dirs in alphabetical order (ignoring leading underscores)
    if os.path.exists(current_output_dir):
        dirs = [
            d
            for d in os.listdir(current_output_dir)
            if os.path.isdir(os.path.join(current_output_dir, d))
        ]
        dirs = sorted(dirs, key=lambda d: d.lstrip("_").lower())
    else:
        dirs = []

    if not dirs:
        st.warning(f"No output directories found in {current_output_dir}.")
        # Show directory picker UI so user can select another directory
        new_output_dir, dir_changed = directory_picker_ui(current_output_dir)
        if dir_changed:
            st.session_state.custom_output_dir = new_output_dir
            st.session_state.use_custom_dir = True
            st.rerun()
        return

    # Show search progress full-width if requested - check EARLY before any columns
    browsing_search_results = st.session_state.get("browse_search_results", False)
    show_search_progress = st.session_state.get("show_search_progress", False)

    if show_search_progress and not browsing_search_results:
        _show_search_progress_inline()
        return  # Don't show any other UI while search is in progress

    # Show search form full-width if requested - check EARLY before any columns
    if st.session_state.get("show_search_dialog", False):
        _show_search_form_inline()
        return  # Don't show any other UI while search form is displayed

    # Create a two-column layout for the top section
    top_cols = st.columns([1, 1.5])

    # Left column: Toggles, item selector, and navigation buttons
    with top_cols[0]:
        # Item selector
        # Fix: Handle query params more robustly
        query_selected_dir = st.query_params.get("selected_dir")

        # Get filter values from session state
        status_filter = st.session_state.get("status_filter_value", "All")
        search_term = st.session_state.get("plan_search_filter_value", "")

        # Filter directories based on search term
        if search_term:
            filtered_dirs = [d for d in dirs if search_term.lower() in d.lower()]
        else:
            filtered_dirs = dirs

        # Filter directories based on status filter
        if status_filter != "All":
            status_filtered = []
            for d in filtered_dirs:
                plan_status = get_plan_status(current_output_dir, d)
                if plan_status == status_filter:
                    status_filtered.append(d)
            filtered_dirs = status_filtered

        # Filter to show only unreviewed plans if filter is active
        if st.session_state.get("filter_unreviewed", False):
            unreviewed_filtered = []
            for d in filtered_dirs:
                manually_reviewed = get_manually_reviewed(current_output_dir, d)
                if manually_reviewed is False:
                    unreviewed_filtered.append(d)
            filtered_dirs = unreviewed_filtered

        # Filter to show only failed plans if filter is active
        if st.session_state.get("filter_failed_only", False):
            failed_filtered = []
            for d in filtered_dirs:
                plan_path = os.path.join(current_output_dir, d, "plan.json")
                if os.path.exists(plan_path):
                    try:
                        with open(plan_path, "r") as f:
                            plan_data = json.load(f)
                        if plan_data.get("task_failed", False):
                            failed_filtered.append(d)
                    except (json.JSONDecodeError, IOError):
                        pass
            filtered_dirs = failed_filtered

        # Filter by step extension if filter is active
        step_ext_filter = st.session_state.get("step_extension_filter_value", "All")
        if step_ext_filter != "All":
            step_ext_filtered = []
            for d in filtered_dirs:
                step_ext = get_step_extension(current_output_dir, d)
                if step_ext == step_ext_filter:
                    step_ext_filtered.append(d)
            filtered_dirs = step_ext_filtered

        # Check if we're browsing search results (already checked above but need value in this scope)
        browsing_search_results = st.session_state.get("browse_search_results", False)
        search_results_plans = []
        search_results_plans_unfiltered = []
        if browsing_search_results and "search_results" in st.session_state:
            search_results_plans_unfiltered = st.session_state.search_results
            search_results_plans = list(
                search_results_plans_unfiltered
            )  # Copy for filtering

            # Apply failed-only filter to search results
            if st.session_state.get("filter_failed_only", False):
                failed_filtered = []
                for full_path in search_results_plans:
                    plan_path = os.path.join(full_path, "plan.json")
                    if os.path.exists(plan_path):
                        try:
                            with open(plan_path, "r") as f:
                                plan_data = json.load(f)
                            if plan_data.get("task_failed", False):
                                failed_filtered.append(full_path)
                        except (json.JSONDecodeError, IOError):
                            pass
                search_results_plans = failed_filtered

        # Use the new directory picker UI (after browsing_search_results is determined)
        new_output_dir, dir_changed = directory_picker_ui(
            current_output_dir, browsing_search_results, len(dirs)
        )

        # If directory changed, update session state and rerun
        if dir_changed:
            st.session_state.custom_output_dir = new_output_dir
            st.session_state.use_custom_dir = True
            st.rerun()

        # Search row
        search_row_cols = st.columns([2, 3])
        with search_row_cols[0]:
            # Build search tooltip showing active search filters
            search_filters = []
            model_type_search = st.session_state.get("search_model_type", "")
            error_text_search = st.session_state.get("search_error_text", "")
            plan_name_search = st.session_state.get("search_plan_name", "")
            config_search = st.session_state.get("search_config", "")
            if model_type_search:
                search_filters.append(f"Model: {model_type_search}")
            if error_text_search:
                search_filters.append(f"Error: {error_text_search}")
            if plan_name_search:
                search_filters.append(f"Plan: {plan_name_search}")
            if config_search:
                search_filters.append(f"Config: {config_search}")
            search_tooltip = (
                ", ".join(search_filters) if search_filters else "Search for plans"
            )

            # Put Search button and X button in same column using inner columns
            if browsing_search_results:
                inner_cols = st.columns([3, 1])
                with inner_cols[0]:
                    if st.button(
                        "🔎 Search", key="open_search_dialog", help=search_tooltip
                    ):
                        st.session_state.show_search_dialog = True
                        st.rerun()
                with inner_cols[1]:
                    if st.button(
                        "❌", key="exit_search_results", help="Exit search results"
                    ):
                        st.session_state.browse_search_results = False
                        # Reset toggle states so they re-detect based on normal directory data
                        for key in [
                            "toggle_history",
                            "toggle_memory",
                            "toggle_observations",
                            "toggle_updated_memory",
                            "toggle_reasoning",
                            "toggle_user_prompt",
                            "toggle_agent_response",
                            "toggle_thinking",
                            "toggle_model_response",
                            "toggle_partial_prompt",
                            "toggle_log",
                            "toggle_prev_image",
                        ]:
                            if key in st.session_state:
                                del st.session_state[key]
                        st.rerun()
            else:
                if st.button(
                    "🔎 Search", key="open_search_dialog", help=search_tooltip
                ):
                    st.session_state.show_search_dialog = True
                    st.rerun()

        with search_row_cols[1]:
            if browsing_search_results:
                st.markdown(f"**{len(search_results_plans)}** results")

        # Thin line to separate search from filters (minimal vertical space)
        st.markdown(
            "<hr style='margin: 0.5rem 0; border: none; border-top: 1px solid #ddd;'>",
            unsafe_allow_html=True,
        )

        # Build filter tooltip showing active filters
        active_filters = []
        if status_filter != "All":
            active_filters.append(f"Status: {status_filter}")
        if search_term:
            active_filters.append(f"Search: '{search_term}'")
        if st.session_state.get("filter_unreviewed", False):
            active_filters.append("Unreviewed only")
        if st.session_state.get("filter_failed_only", False):
            active_filters.append("Failed only")
        if step_ext_filter != "All":
            active_filters.append(f"Step ext: {step_ext_filter}")
        filter_tooltip = ", ".join(active_filters) if active_filters else "Filter plans"
        has_active = has_active_filters()

        # Filters row
        filter_row_cols = st.columns([2, 3])
        with filter_row_cols[0]:
            # Put Filters button and X button in same column using inner columns
            if has_active:
                inner_cols = st.columns([3, 1])
                with inner_cols[0]:
                    if st.button(
                        "🔍 Filters", key="open_filter_dialog", help=filter_tooltip
                    ):
                        st.session_state.show_filter_dialog = True
                        st.rerun()
                with inner_cols[1]:
                    if st.button("❌", key="clear_filters", help="Clear all filters"):
                        # Reset all filter values
                        st.session_state.status_filter_value = "All"
                        st.session_state.plan_search_filter_value = ""
                        st.session_state.filter_unreviewed = False
                        st.session_state.filter_failed_only = False
                        st.session_state.filter_candidate_poses_errors = False
                        st.session_state.step_extension_filter_value = "All"
                        st.rerun()
            else:
                if st.button(
                    "🔍 Filters", key="open_filter_dialog", help=filter_tooltip
                ):
                    st.session_state.show_filter_dialog = True
                    st.rerun()

        with filter_row_cols[1]:
            if has_active:
                # Show filtered count as fraction of folder or search results
                if browsing_search_results:
                    total_count = len(search_results_plans_unfiltered)
                    filtered_count = len(search_results_plans)
                else:
                    total_count = len(dirs)
                    filtered_count = len(filtered_dirs)
                st.markdown(f"**{filtered_count}** of {total_count}")

        # Show filter dialog if requested
        if st.session_state.get("show_filter_dialog", False):
            st.session_state.show_filter_dialog = (
                False  # Reset immediately to prevent reopening on rerun
            )
            _show_filter_dialog()

        # Handle search results browsing mode vs normal mode
        if browsing_search_results and search_results_plans:
            # In search results mode, use full paths and override filtered_dirs
            # Create display names (just the plan name) for the dropdown
            display_to_path = {}
            for full_path in search_results_plans:
                # Show just the plan name (last path component)
                display_name = os.path.basename(full_path)
                # Handle duplicates by adding parent directory
                if display_name in display_to_path:
                    # Add parent directory to disambiguate
                    parent = os.path.basename(os.path.dirname(full_path))
                    display_name = f"{parent}/{display_name}"
                    # If still duplicate, add more path context
                    while display_name in display_to_path:
                        idx = full_path.rfind(display_name)
                        if idx > 0:
                            display_name = full_path[max(0, idx - 20) :]
                        else:
                            display_name = full_path
                            break
                display_to_path[display_name] = full_path

            display_names = list(display_to_path.keys())

            # Find current selection
            current_full_path = st.session_state.get("search_result_current_path", "")
            if current_full_path and current_full_path in search_results_plans:
                # Find the display name for current path
                current_display = None
                for dn, fp in display_to_path.items():
                    if fp == current_full_path:
                        current_display = dn
                        break
                current_idx = (
                    display_names.index(current_display) if current_display else 0
                )
            else:
                current_idx = 0
                if search_results_plans:
                    st.session_state.search_result_current_path = search_results_plans[
                        0
                    ]

            # Show dropdown with search results
            st.markdown(
                """
                <style>
                div[data-testid="column"] > div > div[data-testid="stSelectbox"] {
                    margin-top: -15px !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            selected_display = st.selectbox(
                "Select search result",
                display_names,
                index=current_idx,
                key=f"search_result_selector_{current_idx}",
                label_visibility="collapsed",
            )

            # Get the full path and extract dir info
            # Check if selection changed via dropdown
            selected_full_path = display_to_path.get(selected_display, "")
            if selected_full_path and selected_full_path != current_full_path:
                # User changed selection via dropdown
                st.session_state.search_result_current_path = selected_full_path

            if selected_full_path:
                selected_dir = os.path.basename(selected_full_path)
                current_output_dir = os.path.dirname(selected_full_path)
            else:
                selected_dir = None

            # Navigation buttons for search results
            nav_cols = st.columns([1, 1, 1])
            with nav_cols[0]:
                if st.button(
                    "◀️ Prev",
                    disabled=current_idx == 0,
                    key="prev_search_result",
                ):
                    prev_idx = max(0, current_idx - 1)
                    prev_path = display_to_path[display_names[prev_idx]]
                    st.session_state.search_result_current_path = prev_path
                    st.rerun()
            with nav_cols[1]:
                # Show count (1-indexed)
                st.markdown(
                    f"<div style='text-align: center; padding-top: 5px;'><b>{current_idx + 1}</b> / {len(display_names)}</div>",
                    unsafe_allow_html=True,
                )
            with nav_cols[2]:
                if st.button(
                    "Next ▶️",
                    disabled=current_idx == len(display_names) - 1,
                    key="next_search_result",
                ):
                    next_idx = min(len(display_names) - 1, current_idx + 1)
                    next_path = display_to_path[display_names[next_idx]]
                    st.session_state.search_result_current_path = next_path
                    st.rerun()
        else:
            # Normal browsing mode
            # Ensure we have a valid directory name, not an index
            if query_selected_dir and query_selected_dir in filtered_dirs:
                initial_selected_dir = query_selected_dir
            elif query_selected_dir and query_selected_dir in dirs:
                # If current selection is filtered out, clear it and use first filtered item
                initial_selected_dir = filtered_dirs[0] if filtered_dirs else None
            else:
                initial_selected_dir = filtered_dirs[0] if filtered_dirs else None

            # Find current idx in the filtered dirs list
            current_idx = (
                filtered_dirs.index(initial_selected_dir)
                if initial_selected_dir in filtered_dirs
                else 0
            )

            # Add directory dropdown with a key that forces re-render when query params change
            # Apply negative margin to reduce whitespace
            st.markdown(
                """
                <style>
                div[data-testid="column"] > div > div[data-testid="stSelectbox"] {
                    margin-top: -15px !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            if filtered_dirs:
                # Use a key that includes the query param to force widget recreation when nav buttons change it
                selected_dir = st.selectbox(
                    "Select plan directory",
                    filtered_dirs,
                    index=current_idx,
                    key=f"dir_selector_{query_selected_dir}",
                    label_visibility="collapsed",
                    on_change=lambda: st.query_params.update(
                        {
                            "selected_dir": st.session_state[
                                f"dir_selector_{query_selected_dir}"
                            ]
                        }
                    ),
                )
            else:
                st.warning("No plans match the search filter.")
                selected_dir = None

            # Add navigation buttons below dropdown
            nav_cols = st.columns([1, 1, 1])
            with nav_cols[0]:
                if st.button(
                    "◀️ Prev",
                    disabled=current_idx == 0 or not filtered_dirs,
                    key="prev_dir",
                ):
                    prev_idx = max(0, current_idx - 1)
                    st.query_params["selected_dir"] = filtered_dirs[prev_idx]
                    st.rerun()
            with nav_cols[1]:
                # Show count (1-indexed)
                if filtered_dirs:
                    st.markdown(
                        f"<div style='text-align: center; padding-top: 5px;'><b>{current_idx + 1}</b> / {len(filtered_dirs)}</div>",
                        unsafe_allow_html=True,
                    )
            with nav_cols[2]:
                if st.button(
                    "Next ▶️",
                    disabled=current_idx == len(filtered_dirs) - 1 or not filtered_dirs,
                    key="next_dir",
                ):
                    next_idx = min(len(filtered_dirs) - 1, current_idx + 1)
                    st.query_params["selected_dir"] = filtered_dirs[next_idx]
                    st.rerun()

    # Right column: Plan title, goals, config, and display options
    with top_cols[1]:
        # Check if prompt data exists for the selected plan (needed for toggle states)
        has_prompt_data = False
        has_log_data = False
        has_model_response = False
        has_partial_prompt = False
        has_log = False
        has_prev_images = False

        if selected_dir:
            plan_path = os.path.join(current_output_dir, selected_dir, "plan.json")
            if os.path.exists(plan_path):
                try:
                    with open(plan_path, "r", encoding="utf-8") as f:
                        plan_data = json.load(f)

                    for step in plan_data.get("steps", []):
                        formatted = step.get("formatted") or {}
                        if (
                            formatted.get("user_part")
                            or formatted.get("agent_part")
                            or formatted.get("natural_language_agent_part")
                        ):
                            has_prompt_data = True
                        if step.get("log"):
                            has_log_data = True
                            has_log = True
                        if step.get("model_response"):
                            has_model_response = True
                        if step.get("partial_prompt"):
                            has_partial_prompt = True
                except (json.JSONDecodeError, FileNotFoundError):
                    pass

            # Check for prev_image files
            plan_dir_path = os.path.join(current_output_dir, selected_dir)
            if os.path.isdir(plan_dir_path):
                for filename in os.listdir(plan_dir_path):
                    if filename.endswith("_prev_image.png") and filename[0].isdigit():
                        has_prev_images = True
                        break

        if selected_dir:
            # Load plan data for right column
            plan_path = os.path.join(current_output_dir, selected_dir, "plan.json")
            if os.path.exists(plan_path):
                with open(plan_path, "r") as f:
                    plan = json.load(f)
                task_description = plan.get(
                    "task_description", "No task description found"
                )
            else:
                task_description = "No plan.json found"
                plan = {}

            # Display task description (plan title), path, and plan name
            st.markdown(f"### {task_description}")
            # Show path/plan_name on same line in caption style
            st.caption(f"{current_output_dir} / {selected_dir}")

            # Create three sub-columns: config, goals, display options
            config_col, goals_col, display_col = st.columns([1, 1, 0.6])

            # Left sub-column: Display config values
            with config_col:
                config_path = os.path.join(
                    os.path.dirname(current_output_dir), "config.json"
                )
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as f:
                            config_data = json.load(f)
                        # Exclude metadata and less important fields
                        excluded_keys = {
                            "test_name",
                            "model_path",
                            "git_commit",
                            "max_completion_tokens",
                            "temperature",
                            "implementation",
                            "prompt_version",
                            "model_name",
                        }
                        config_items = [
                            (k, v)
                            for k, v in config_data.items()
                            if k not in excluded_keys
                        ]

                        # Start with model name at top in larger font (use model_path for clean name)
                        config_html = '<div style="font-size: 0.9em;">'
                        model_path_value = config_data.get("model_path", "")
                        if model_path_value:
                            config_html += f'<div style="font-size: 1.2em; margin-bottom: 4px;"><b>{model_path_value}</b></div>'

                        for key, value in config_items:
                            # Handle boolean values with checkmarks
                            if isinstance(value, bool):
                                icon = "✅" if value else "❌"
                                config_html += f"<div>{icon} <b>{key}</b></div>"
                            # Handle hand_transparency specially
                            elif key == "hand_transparency":
                                icon = "❌" if value == 0 else "✅"
                                config_html += (
                                    f"<div>{icon} <b>{key}</b>: {value}</div>"
                                )
                            # Handle feedback_type with matching icons from generate_reports.py
                            elif key == "feedback_type":
                                if isinstance(value, str):
                                    if value.lower() == "simple":
                                        icon = "🟡"  # Yellow circle for simple
                                    elif value.lower() == "detailed":
                                        icon = "🟢"  # Green circle for detailed
                                    else:  # none
                                        icon = "❌"  # Red X for none
                                    config_html += (
                                        f"<div>{icon} <b>{key}</b>: {value}</div>"
                                    )
                                else:
                                    config_html += f"<div><b>{key}</b>: {value}</div>"
                            else:
                                config_html += f"<div><b>{key}</b>: {value}</div>"
                        config_html += "</div>"
                        st.markdown(config_html, unsafe_allow_html=True)
                    except (json.JSONDecodeError, FileNotFoundError):
                        pass

            # Middle sub-column: Display goals
            with goals_col:
                if os.path.exists(plan_path):
                    goal_data = plan.get("goal", {})
                    location_goals = goal_data.get("location_goals", [])
                    state_goals = goal_data.get("state_goals", [])
                    put_away_goals = goal_data.get("put_away_goals", [])
                    action_goals = goal_data.get("action_goals", [])
                    contents_goals = goal_data.get("contents_goals", [])

                    # Check if any goals exist
                    has_goals = (
                        location_goals
                        or state_goals
                        or put_away_goals
                        or action_goals
                        or contents_goals
                    )

                    if has_goals:
                        st.markdown(
                            '<div class="stBlock">Goals:</div>', unsafe_allow_html=True
                        )

                        # Build the goals HTML as a single string
                        goals_html = '<div class="goals-section">'

                        # Display location goals
                        for location_goal in location_goals:
                            object_type = location_goal.get("object_type", "")
                            destination_type = location_goal.get("destination_type", "")
                            outcome = location_goal.get("outcome", None)

                            # Set text color based on outcome
                            if outcome == "success" or outcome is True:
                                text_color = "green"
                            elif outcome == "failure" or outcome is False:
                                text_color = "red"
                            else:
                                text_color = "gray"

                            goals_html += f"<p><span style='color: {text_color};'>• {object_type} in {destination_type}</span></p>"

                        # Display state goals
                        for state_goal in state_goals:
                            object_type = state_goal.get("object_type", "")
                            state = state_goal.get("state", "")
                            value = state_goal.get("value", False)
                            outcome = state_goal.get("outcome", None)
                            state_text = "True" if value else "False"

                            # Set text color based on outcome
                            if outcome == "success" or outcome is True:
                                text_color = "green"
                            elif outcome == "failure" or outcome is False:
                                text_color = "red"
                            else:
                                text_color = "gray"

                            goals_html += f"<p><span style='color: {text_color};'>• {object_type} {state} {state_text}</span></p>"

                        # Display put away goals
                        for put_away_goal in put_away_goals:
                            object_type = put_away_goal.get("object_type", "")
                            outcome = put_away_goal.get("outcome", None)

                            # Set text color based on outcome
                            if outcome == "success" or outcome is True:
                                text_color = "green"
                            elif outcome == "failure" or outcome is False:
                                text_color = "red"
                            else:
                                text_color = "gray"

                            goals_html += f"<p><span style='color: {text_color};'>• {object_type} put away</span></p>"

                        # Display action goals
                        for action_description, outcome in action_goals.items():
                            # Set text color based on outcome
                            if outcome == "success" or outcome is True:
                                text_color = "green"
                            elif outcome == "failure" or outcome is False:
                                text_color = "red"
                            else:
                                text_color = "gray"

                            goals_html += f"<p><span style='color: {text_color};'>• {action_description}</span></p>"

                        # Display contents goals
                        for contents_goal in contents_goals:
                            container_type = contents_goal.get("container_type", "")
                            contents = contents_goal.get("contents", [])
                            outcome = contents_goal.get("outcome", None)

                            # Format the contents list as a readable string
                            contents_text = ", ".join(contents) if contents else ""

                            # Set text color based on outcome
                            if outcome == "success" or outcome is True:
                                text_color = "green"
                            elif outcome == "failure" or outcome is False:
                                text_color = "red"
                            else:
                                text_color = "gray"

                            goals_html += f"<p><span style='color: {text_color};'>• {container_type} contains {contents_text}</span></p>"

                        # Close the goals div and render as single markdown
                        goals_html += "</div>"
                        st.markdown(goals_html, unsafe_allow_html=True)

            # Right sub-column: Display options popover
            with display_col:
                # Show status (Success/Failed) inline
                if plan:
                    if plan.get("task_failed", False):
                        # Extract fail reason from last step's log
                        fail_reason = None
                        steps = plan.get("steps", [])
                        if steps:
                            last_log = steps[-1].get("log", "")
                            if "[FAIL]" in last_log:
                                for line in last_log.split("\n"):
                                    if "[FAIL]" in line:
                                        fail_reason = line.split("[FAIL]")[-1].strip()
                                        break
                        if fail_reason:
                            st.markdown(
                                f"<div style='color: red; font-weight: bold; font-size: 0.85em;'>❌ {fail_reason}</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<div style='color: red; font-weight: bold; font-size: 0.85em;'>❌ Failed</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.markdown(
                            "<div style='color: green; font-weight: bold; font-size: 0.85em;'>✅ Success</div>",
                            unsafe_allow_html=True,
                        )

                # Manually reviewed checkbox
                if selected_dir:
                    current_reviewed = get_manually_reviewed(
                        current_output_dir, selected_dir
                    )
                    reviewed_key = f"manually_reviewed_{selected_dir}"
                    new_reviewed = st.checkbox(
                        "Manually Reviewed",
                        value=(
                            current_reviewed if current_reviewed is not None else False
                        ),
                        key=reviewed_key,
                    )
                    # Update if changed
                    if new_reviewed != current_reviewed:
                        set_manually_reviewed(
                            current_output_dir, selected_dir, new_reviewed
                        )
                        # If checked (not unchecked), advance to next plan
                        if new_reviewed and current_idx < len(filtered_dirs) - 1:
                            next_idx = current_idx + 1
                            st.query_params["selected_dir"] = filtered_dirs[next_idx]
                            st.rerun()

                with st.popover("⚙️ Display", use_container_width=True):
                    st.markdown("**Show/Hide Sections**")

                    col1, col2 = st.columns(2)
                    with col1:
                        show_history = st.toggle(
                            "History", value=False, key="toggle_history"
                        )
                        show_memory = st.toggle(
                            "Memory", value=False, key="toggle_memory"
                        )
                        show_observations = st.toggle(
                            "Observations",
                            value=not has_log_data,
                            key="toggle_observations",
                        )
                        show_updated_memory = st.toggle(
                            "Upd Memory",
                            value=not has_log_data,
                            key="toggle_updated_memory",
                        )
                        show_reasoning = st.toggle(
                            "Reasoning", value=True, key="toggle_reasoning"
                        )
                        show_bounding_boxes = st.toggle(
                            "Boxes", value=False, key="toggle_boxes"
                        )

                    with col2:
                        show_user_prompt = st.toggle(
                            "User Prompt",
                            value=False,
                            disabled=not has_prompt_data,
                            key="toggle_user_prompt",
                        )
                        show_agent_response = st.toggle(
                            "Agent Resp",
                            value=False,
                            disabled=not has_prompt_data,
                            key="toggle_agent_response",
                        )
                        show_thinking = st.toggle(
                            "Thinking",
                            value=True if has_prompt_data else False,
                            disabled=not has_prompt_data,
                            key="toggle_thinking",
                        )
                        show_model_response = st.toggle(
                            "Model Resp",
                            value=True if has_model_response else False,
                            disabled=not has_model_response,
                            key="toggle_model_response",
                        )
                        show_partial_prompt = st.toggle(
                            "Partial Prompt",
                            value=True if has_partial_prompt else False,
                            disabled=not has_partial_prompt,
                            key="toggle_partial_prompt",
                        )
                        show_log = st.toggle(
                            "Log",
                            value=True if has_log else False,
                            disabled=not has_log,
                            key="toggle_log",
                        )

                    show_prev_image = st.toggle(
                        "Prev Image",
                        value=True,
                        disabled=not has_prev_images,
                        key="toggle_prev_image",
                    )

                    st.divider()

                    # Button to view raw JSON
                    if selected_dir:
                        plan_path = os.path.join(
                            current_output_dir, selected_dir, "plan.json"
                        )
                        if os.path.exists(plan_path):
                            if st.button(
                                "📄 View Raw JSON",
                                key="view_raw_json",
                                use_container_width=True,
                            ):
                                st.session_state.show_raw_json = True
                                st.session_state.raw_json_path = plan_path

                    # Delete button with confirmation
                    if selected_dir:
                        plan_dir_path = os.path.join(current_output_dir, selected_dir)
                        if os.path.isdir(plan_dir_path):
                            if st.button(
                                "🗑️ Delete Plan",
                                key="delete_plan",
                                type="secondary",
                                use_container_width=True,
                            ):
                                st.session_state.show_delete_confirm = True
                                st.session_state.delete_plan_path = plan_dir_path
                                st.session_state.delete_plan_name = selected_dir
                                st.session_state.delete_filtered_dirs = filtered_dirs
                                st.session_state.delete_current_idx = current_idx

    # Show delete confirmation dialog if triggered
    if st.session_state.get("show_delete_confirm", False):
        _show_delete_confirm_dialog()

    # Show raw JSON dialog if triggered
    if st.session_state.get("show_raw_json", False):
        _show_raw_json_dialog()

    # Create scrollable container for steps - CSS overrides height to fill viewport
    st.markdown(
        """
        <style>
        /* Target the scrollable container created by st.container(height=...) */
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            max-height: calc(100vh - 220px) !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"][style*="height"] {
            height: calc(100vh - 220px) !important;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    steps_container = st.container(height=1000, border=True)

    # Always use the alternative view - but only if a directory is selected
    if selected_dir:
        with steps_container:
            display(
                selected_dir,
                current_output_dir,
                show_history=st.session_state.get("toggle_history", True),
                show_memory=st.session_state.get("toggle_memory", True),
                show_observations=st.session_state.get("toggle_observations", True),
                show_updated_memory=st.session_state.get("toggle_updated_memory", True),
                show_reasoning=st.session_state.get("toggle_reasoning", True),
                show_user_prompt=st.session_state.get("toggle_user_prompt", False),
                show_agent_response=st.session_state.get(
                    "toggle_agent_response", False
                ),
                show_thinking=st.session_state.get("toggle_thinking", False),
                show_model_response=st.session_state.get(
                    "toggle_model_response", False
                ),
                show_partial_prompt=st.session_state.get(
                    "toggle_partial_prompt", False
                ),
                show_log=st.session_state.get("toggle_log", False),
                show_prev_image=st.session_state.get("toggle_prev_image", False),
            )
    else:
        st.info("No plan directories available with the current filter.")


if __name__ == "__main__":
    main()
