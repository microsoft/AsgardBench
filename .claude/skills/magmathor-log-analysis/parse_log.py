#!/usr/bin/env python3
"""
Magmathor Log Parser

Parses Magmathor experiment logs into a structured JSON format for easier analysis.

Usage:
    # Single experiment mode (default):
    python parse_log.py <logs.txt> [--output <output.json>]

    # Multi-experiment mode (for combined logs from multiple partitions):
    python parse_log.py <combined_logs.txt> --base-dir <experiment_dir> --config-string <config>

    # Also output filtered.txt (prompts removed):
    python parse_log.py <logs.txt> --filter

Image Semantics:
    The log format records images AFTER each action is executed:

    Step N:
        Model receives: Previous image (from step N-1) + Current prompt
        Model outputs: Action
        Action executes
        Images logged: Previous=N-1_*.png, Current=N_*.png

    When analyzing step N:
        - "model_input" / "previous": What the model SAW when making decision N
        - "action_result" / "current": Result AFTER action N was executed

    Example for step 5:
        model_input = "4_pickup Mug.png"  (result from step 4, shown to model at step 5)
        action_result = "5_put Cabinet.png"  (result after step 5's PUT action)

Output JSON structure:
{
    "source_file": "path/to/logs.txt",
    "config": {
        "text_only": false,
        "feedback_type": "simple",
        ...
    },
    "config_natural_failures": ["list of failure types expected given config"],
    "summary": {
        "total_tasks": N,
        "passed": N,
        "failed": N,
        "pass_rate": "X%"
    },
    "tasks": [
        {
            "name": "task_name",
            "index": "X/Y",
            "passed": true/false,
            "steps": [
                {
                    "step_number": N,
                    "timestamp": "...",
                    "think": "model reasoning...",
                    "action": "action name",
                    "target": "target object",
                    "error": "[STEP ERROR] message" or null,
                    "error_category": "category name" or null,
                    "is_config_natural": true/false,
                    "images": {
                        "previous": "absolute path or null",
                        "current": "absolute path",
                        "model_input": "same as previous (semantic name)",
                        "action_result": "same as current (semantic name)"
                    }
                }
            ],
            "error_summary": ["list of errors"]
        }
    ],
    "errors_by_category": {
        "category_name": [
            {"task": "...", "step": N, "error": "...", "think": "...", "is_config_natural": bool}
        ]
    }
}
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# Words to exclude from object matching (common English words that look like objects)
# Copied from Magmathor/Model/generate_reports.py
EXCLUDED_WORDS = {
    "Cannot",
    "The",
    "This",
    "That",
    "There",
    "Here",
    "What",
    "When",
    "Where",
    "Which",
    "While",
    "With",
    "Without",
    "Could",
    "Would",
    "Should",
    "Must",
    "Will",
    "Shall",
    "Have",
    "Has",
    "Had",
    "Does",
    "Did",
    "Done",
    "Been",
    "Being",
    "Was",
    "Were",
    "Are",
    "Is",
    "Am",
    "Be",
    "Not",
    "But",
    "And",
    "For",
    "From",
    "Into",
    "Onto",
    "Over",
    "Under",
    "Above",
    "Below",
    "Between",
    "Among",
    "Through",
    "During",
    "Before",
    "After",
    "Since",
    "Until",
    "Already",
    "Also",
    "Always",
    "Never",
    "Ever",
    "Only",
    "Just",
    "Still",
    "Even",
    "Very",
    "Too",
    "Much",
    "Many",
    "Some",
    "Any",
    "All",
    "Each",
    "Every",
    "Both",
    "Either",
    "Neither",
    "Other",
    "Another",
    "Such",
    "Same",
    "Different",
    "PUT",
    "GET",
    "Ran",
    "Failed",
    "Error",
    "Invalid",
    "Missing",
    "Expected",
    "Unexpected",
    "Unable",
    "Slice",
    "Sliced",
    "Types",
    "Type",
    "Action",
    "Object",
    "Objects",
    "Specifier",
    "Candidate",
    "Candidates",
    "None",
    "True",
    "False",
    "Target",
    "Receptacle",
    "Clone",
}

# Pattern to match object names with suffixes (e.g., Bread_fe4bb3e3, Egg_Cracked_20)
OBJECT_WITH_SUFFIX_PATTERN = re.compile(
    r"[A-Z][a-zA-Z]*(?:_[a-zA-Z0-9]+)+(?:\(Clone\))?"
)
# Pattern to match simple CamelCase object names (e.g., WineBottle, SinkBasin)
SIMPLE_OBJECT_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b")
# Pattern to match single capitalized words (e.g., Pencil, Apple, Bread)
SINGLE_WORD_PATTERN = re.compile(r"\b([A-Z][a-z]{2,})\b")


def extract_object_type(obj_name: str) -> str:
    """Extract the base object type from a full object name.

    E.g., 'Bread_fe4bb3e3' -> 'Bread', 'Egg_Cracked_20(Clone)' -> 'Egg_Cracked'

    Copied from Magmathor/Model/generate_reports.py
    """
    # Remove (Clone) suffix if present
    name = obj_name.replace("(Clone)", "")
    # Split by underscore and take parts that are not hex/numeric
    parts = name.split("_")
    type_parts = []
    for part in parts:
        # Stop if we hit a hex-like or pure numeric suffix
        if re.match(r"^[a-f0-9]{6,}$", part, re.IGNORECASE):
            break
        if re.match(r"^\d+$", part):
            break
        type_parts.append(part)
    return "_".join(type_parts) if type_parts else parts[0]


def find_objects_in_message(msg: str) -> list:
    """Find all object references in an error message.

    Copied from Magmathor/Model/generate_reports.py
    """
    result = []
    # First find objects with suffixes (e.g., Bread_fe4bb3e3)
    for match in OBJECT_WITH_SUFFIX_PATTERN.findall(msg):
        if match not in result:
            result.append(match)
    # Find simple CamelCase objects (e.g., WineBottle, SinkBasin)
    for match in SIMPLE_OBJECT_PATTERN.findall(msg):
        if match not in EXCLUDED_WORDS and match not in result:
            # Check it's not already covered by a suffixed version
            if not any(m.startswith(match + "_") for m in result):
                result.append(match)
    # Also find single capitalized words that might be objects
    # (e.g., Pencil, Apple, Bread)
    for match in SINGLE_WORD_PATTERN.findall(msg):
        if match not in EXCLUDED_WORDS and match not in result:
            if not any(m.startswith(match + "_") for m in result):
                result.append(match)
    return result


def normalize_error_message(error_msg: str) -> tuple[str, dict[str, str]]:
    """Normalize an error message by replacing object names with placeholders.

    Returns:
        tuple: (normalized_message, object_types_dict)
            - normalized_message: Error with {X}, {Y}, {Z} placeholders
            - object_types_dict: Maps placeholder name to object type

    Example:
        Input: "Cannot pick up Bread_fe4bb3e3 while holding Knife_abc123"
        Output: ("Cannot pick up {X} while holding {Y}", {"X": "Bread", "Y": "Knife"})
    """
    # Find all unique object names in the error message
    matches = find_objects_in_message(error_msg)
    unique_objects = []
    for m in matches:
        if m not in unique_objects:
            unique_objects.append(m)

    # Replace each unique object with {X}, {Y}, {Z}, etc.
    placeholder_names = ["X", "Y", "Z", "W", "V"]
    normalized_msg = error_msg
    object_types = {}  # placeholder_name -> object_type

    for i, obj_name in enumerate(unique_objects):
        ph_name = placeholder_names[i] if i < len(placeholder_names) else f"O{i}"
        placeholder = "{" + ph_name + "}"
        normalized_msg = normalized_msg.replace(obj_name, placeholder)
        object_types[ph_name] = extract_object_type(obj_name)

    return normalized_msg, object_types


# Legacy error category patterns (kept for backward compatibility and high-level grouping)
ERROR_PATTERNS = {
    "openable_precondition": r"Target openable Receptacle is CLOSED|CLOSED, can't place",
    "not_visible_target": r"is not visible",
    "inventory_hand_state": r"(PUT: No object held|while holding|No object held)",
    "dirty_state_blocks": r"is dirty",
    "sink_basin_constraints": r"SinkBasin contains",
    "navigation_controller": r"Ran out of candidate poses",
    "already_state": r"is already (open|closed|on|off|sliced)",
    "not_pickupable": r"not pickupable",
    "invalid_response": r"Object name cannot be None|invalid format",
    "missing_tool": r"without holding a (knife|sponge|cloth|spray)",
    "placement_error": r"cannot be placed in|NOT a receptacle",
    "precondition_error": r"must be in the|not on a countertop|not filled with liquid|is not openable",
}

# Config-to-natural-failure mappings
# Maps config settings to error categories/patterns that are "natural" given that config
CONFIG_NATURAL_FAILURES = {
    "text_only": {
        True: ["not_visible_target", "navigation_controller", "spatial_confusion"],
        False: [],
    },
    "feedback_type": {
        "none": ["repetitive_failures"],
        "simple": [],
        "detailed": [],
    },
    "previous_image": {
        "none": ["state_tracking_failure"],
        "grayscale": [],
        "color": [],
    },
    "use_memory": {
        False: ["context_loss"],
        True: [],
    },
    "include_common_sense": {
        False: ["physics_violation", "prerequisite_skip", "affordance_error"],
        True: [],
    },
}


def get_config_natural_failures(config: dict) -> list[str]:
    """Get list of failure types that are natural given the config."""
    natural = []
    for field, mapping in CONFIG_NATURAL_FAILURES.items():
        if field in config:
            value = config[field]
            if value in mapping:
                natural.extend(mapping[value])
    return list(set(natural))


def is_error_config_natural(error_text: str, error_category: str, config: dict) -> bool:
    """Check if an error is expected given the config settings."""
    natural_failures = get_config_natural_failures(config)

    # Direct category match
    if error_category in natural_failures:
        return True

    # text_only specific: visibility/spatial errors are natural
    if config.get("text_only", False):
        if error_category in ["not_visible_target", "navigation_controller"]:
            return True
        # Spatial confusion patterns
        if re.search(r"(too far|not facing|cannot reach)", error_text, re.IGNORECASE):
            return True

    # feedback_type=none: repetitive errors are natural
    if config.get("feedback_type") == "none":
        if error_category == "repetitive_failures":
            return True

    # previous_image=none: state tracking issues are natural
    if config.get("previous_image") == "none":
        if re.search(
            r"(already open|already closed|already)", error_text, re.IGNORECASE
        ):
            return True

    # include_common_sense=False: domain errors are natural
    if not config.get("include_common_sense", True):
        if re.search(r"(not openable|doesn't fit|no knife)", error_text, re.IGNORECASE):
            return True

    return False


@dataclass
class Step:
    step_number: int
    timestamp: Optional[str] = None
    think: Optional[str] = None
    action: Optional[str] = None
    target: Optional[str] = None
    executing_action: Optional[str] = None
    low_level_calls: list = field(default_factory=list)
    error: Optional[str] = None
    error_category: Optional[str] = (
        None  # High-level category (e.g., "not_visible_target")
    )
    error_normalized: Optional[str] = (
        None  # Normalized error with placeholders (e.g., "Cannot pick up {X}")
    )
    error_object_types: dict = field(
        default_factory=dict
    )  # Placeholder -> object type (e.g., {"X": "Bread"})
    is_config_natural: bool = False
    # Images dict structure:
    # - "previous": The image from step N-1 (what model saw as "previous" at step N)
    # - "current": The image from step N (result after action N, shown as "current" at step N)
    #
    # When analyzing step N:
    # - "model_input": What the model SAW when making decision N = "previous" image
    # - "action_result": Result AFTER action N was executed = "current" image
    images: dict = field(
        default_factory=lambda: {
            "previous": None,  # Image the model saw (from step N-1)
            "current": None,  # Result after this step's action
            "model_input": None,  # Alias for previous (semantic name)
            "action_result": None,  # Alias for current (semantic name)
        }
    )
    raw_response: Optional[str] = None


@dataclass
class Task:
    name: str
    index: str
    passed: bool = False
    steps: list = field(default_factory=list)
    error_summary: list = field(default_factory=list)


@dataclass
class LogData:
    source_file: str
    config: dict = field(default_factory=dict)
    config_natural_failures: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    tasks: list = field(default_factory=list)
    errors_by_category: dict = field(default_factory=dict)  # High-level categories
    errors_by_normalized: dict = field(
        default_factory=dict
    )  # Detailed normalized errors


def categorize_error(error_text: str) -> Optional[str]:
    """Categorize an error based on known patterns (high-level categories)."""
    if not error_text:
        return None
    for category, pattern in ERROR_PATTERNS.items():
        if re.search(pattern, error_text, re.IGNORECASE):
            return category
    return "uncategorized"


def extract_timestamp(text: str) -> Optional[str]:
    """Extract the first timestamp from text in [HH:MM:SS] format."""
    match = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", text)
    if match:
        return match.group(1)
    return None


def strip_timestamps(text: str) -> str:
    """Remove [HH:MM:SS] timestamp prefixes from each line."""
    # Remove timestamp at start of lines: "[14:16:35] actual content" -> "actual content"
    return re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", text, flags=re.MULTILINE)


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI escape codes (color codes) from text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def clean_text(text: str) -> str:
    """Clean text by stripping timestamps and ANSI codes."""
    text = strip_timestamps(text)
    text = strip_ansi_codes(text)
    return text.strip()


def extract_think_content(response_text: str) -> Optional[str]:
    """Extract <think> content from a model response."""
    match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if match:
        return clean_text(match.group(1))
    return None


def extract_action_from_response(
    response_text: str,
) -> tuple[Optional[str], Optional[str]]:
    """Extract action and target from model response."""
    # Clean text first
    cleaned = clean_text(response_text)

    # Look for action patterns like: Action: pickup knife
    action_match = re.search(r"Action:\s*(\w+)\s+(.+?)(?:\n|$)", cleaned, re.IGNORECASE)
    if action_match:
        return action_match.group(1).strip(), action_match.group(2).strip()

    # Also try: Extracted action: / Extracted item:
    extracted = re.search(r"Extracted action:\s*(\w+)", cleaned, re.IGNORECASE)
    item = re.search(r"Extracted item:\s*(\w+)", cleaned, re.IGNORECASE)
    if extracted:
        return extracted.group(1).strip(), item.group(1).strip() if item else None

    return None, None


def extract_simple_task_name(full_task_name: str) -> str:
    """Extract simple task name from full task name.

    Full format: 'gpt-4o Generated/magt_benchmark_p1 - cook__Bread_Plate(d)_FloorPlan3_V1 - 0.6'
    Simple format: 'cook__Bread_Plate(d)_FloorPlan3_V1'
    """
    # Split by ' - ' and take the middle part (task name)
    parts = full_task_name.split(" - ")
    if len(parts) >= 2:
        return parts[1].strip()
    return full_task_name


def extract_partition_from_task_name(full_task_name: str) -> Optional[str]:
    """Extract partition name from full task name.

    Full format: 'gpt-4o Generated/magt_benchmark_p1 - cook__Bread_Plate(d)_FloorPlan3_V1 - 0.6'
    Returns: 'magt_benchmark_p1'
    """
    # Look for partition pattern: Generated/magt_benchmark_pN or just magt_benchmark_pN
    match = re.search(r"(magt_benchmark_p\d+)", full_task_name)
    if match:
        return match.group(1)
    return None


def extract_model_from_task_name(full_task_name: str) -> Optional[str]:
    """Extract model name from full task name.

    Full format: 'gpt-4o Generated/magt_benchmark_p1 - cook__Bread_Plate(d)_FloorPlan3_V1 - 0.6'
    Returns: 'gpt-4o'
    """
    # Model name is typically the first word before a space
    match = re.match(r"^(\S+)", full_task_name)
    if match:
        return match.group(1)
    return None


def parse_log(
    content: str,
    source_file: str,
    config: Optional[dict] = None,
    base_dir: Optional[str] = None,
    config_string: Optional[str] = None,
) -> LogData:
    """Parse a Magmathor log file into structured data.

    Args:
        content: The log file content
        source_file: Path to the source log file (used to build absolute image paths)
        config: Optional config dict from config.json
        base_dir: Optional base experiment directory (e.g., Test/ or 20251231__Test__...).
                  If provided, image paths are computed from base_dir + partition + config.
                  If not provided, uses source_file's parent directory.
        config_string: Optional config string (e.g., "T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096").
                       Required if base_dir is provided.
    """
    log_data = LogData(source_file=source_file)

    # Compute experiment directory from source_file for building image paths
    # source_file is like: Test/magt_benchmark_p1/gpt-4o--T0_Fs.../logs.txt
    source_path = Path(source_file).resolve()

    # If base_dir provided, we'll compute Plans paths per-task from the task name
    # Otherwise use the traditional single-experiment approach
    use_multi_experiment_mode = base_dir is not None
    base_path = Path(base_dir).resolve() if base_dir else None

    if not use_multi_experiment_mode:
        experiment_dir = source_path.parent  # Directory containing logs.txt
        plans_dir = experiment_dir / "Plans"
    else:
        plans_dir = None  # Will be computed per-task

    # Store config and compute natural failures
    if config:
        log_data.config = config
        log_data.config_natural_failures = get_config_natural_failures(config)

    # Split into tasks using "Testing: [X/Y]" markers
    task_splits = re.split(r"(?=Testing:\s*\[\d+/\d+\])", content)

    current_task = None
    current_task_plans_dir = None  # Absolute path to current task's Plans folder

    for section in task_splits:
        if not section.strip():
            continue

        # Check for task start
        task_match = re.match(r"Testing:\s*\[(\d+/\d+)\]\s*(.+?)(?:\n|$)", section)
        if task_match:
            if current_task:
                log_data.tasks.append(current_task)

            task_name = strip_ansi_codes(task_match.group(2).strip())
            current_task = Task(name=task_name, index=task_match.group(1))

            # Compute path to this task's Plans folder
            simple_task_name = extract_simple_task_name(task_name)

            if use_multi_experiment_mode and base_path and config_string:
                # Multi-experiment mode: compute plans_dir from task name
                partition = extract_partition_from_task_name(task_name)
                model = extract_model_from_task_name(task_name)
                if partition and model:
                    # Try to find the experiment directory with rep1 or rep2
                    for rep in ["rep1", "rep2"]:
                        exp_dir = (
                            base_path / partition / f"{model}--{config_string}--{rep}"
                        )
                        task_plans_dir = exp_dir / "Plans"
                        # Check both with and without underscore prefix (failed vs passed)
                        task_plans_dir_failed = task_plans_dir / f"_{simple_task_name}"
                        task_plans_dir_passed = task_plans_dir / simple_task_name
                        if task_plans_dir_failed.exists():
                            current_task_plans_dir = task_plans_dir_failed
                            break
                        elif task_plans_dir_passed.exists():
                            current_task_plans_dir = task_plans_dir_passed
                            break
                    else:
                        current_task_plans_dir = None
                else:
                    current_task_plans_dir = None
            else:
                # Single-experiment mode: use fixed plans_dir
                # Check both with and without underscore prefix (failed vs passed)
                task_plans_dir_failed = plans_dir / f"_{simple_task_name}"
                task_plans_dir_passed = plans_dir / simple_task_name
                if task_plans_dir_failed.exists():
                    current_task_plans_dir = task_plans_dir_failed
                elif task_plans_dir_passed.exists():
                    current_task_plans_dir = task_plans_dir_passed
                else:
                    current_task_plans_dir = None

            # Check if task passed or failed
            # Formats: "test passed", "tested successfully", "test failed"
            if re.search(r"(test passed|tested successfully)", section, re.IGNORECASE):
                current_task.passed = True
            elif re.search(r"test failed", section, re.IGNORECASE):
                current_task.passed = False

        if not current_task:
            continue

        # Parse steps by finding RESPONSE blocks and then finding associated
        # execution info, errors, and images that follow each response
        response_pattern = r"===============RESPONSE=================\s*(.*?)\s*=============END RESPONSE==============="
        response_matches = list(re.finditer(response_pattern, section, re.DOTALL))

        for step_idx, response_match in enumerate(response_matches):
            step_number = step_idx + 1
            step = Step(step_number=step_number)

            response_text = response_match.group(1)

            # Extract timestamp from the response (first occurrence)
            step.timestamp = extract_timestamp(response_text)

            # Extract think content (already cleaned by extract_think_content)
            step.think = extract_think_content(response_text)

            # Clean and truncate raw_response
            cleaned_response = clean_text(response_text)
            step.raw_response = (
                cleaned_response[:500] + "..."
                if len(cleaned_response) > 500
                else cleaned_response
            )

            # Extract action info from response
            step.action, step.target = extract_action_from_response(response_text)

            # Find the text between this response and the next (or end of section)
            start_pos = response_match.end()
            if step_idx + 1 < len(response_matches):
                end_pos = response_matches[step_idx + 1].start()
            else:
                end_pos = len(section)

            between_text = section[start_pos:end_pos]

            # Extract executing action for this step (clean ANSI codes)
            exec_match = re.search(r"Executing action:\s*(.+?)(?:\n|$)", between_text)
            if exec_match:
                step.executing_action = strip_ansi_codes(exec_match.group(1).strip())

            # Extract error for this step
            error_match = re.search(r"\[STEP ERROR\]\s*(.+?)(?:\n|$)", between_text)
            if error_match:
                error_text = error_match.group(1).strip()
                error_category = categorize_error(error_text)

                # Normalize the error message (replace object names with placeholders)
                normalized_msg, object_types = normalize_error_message(error_text)
                # Create action-prefixed normalized key (like generate_reports.py does)
                action_name = step.action or ""
                normalized_key = (
                    f"{action_name} : {normalized_msg}"
                    if action_name
                    else normalized_msg
                )

                step.error = f"[STEP ERROR] {error_text}"
                step.error_category = error_category
                step.error_normalized = normalized_key
                step.error_object_types = object_types
                current_task.error_summary.append(error_text)

                # Check if error is natural given config
                if config:
                    step.is_config_natural = is_error_config_natural(
                        error_text, error_category, config
                    )

                # Add to high-level category index
                if error_category not in log_data.errors_by_category:
                    log_data.errors_by_category[error_category] = []
                # Store index to update with images later
                error_entry = {
                    "task": current_task.name,
                    "step": step_number,
                    "error": error_text,
                    "error_normalized": normalized_key,
                    "error_object_types": object_types,
                    "think": step.think[:200] if step.think else None,
                    "is_config_natural": step.is_config_natural,
                    "images": None,  # Will be populated after image extraction
                }
                log_data.errors_by_category[error_category].append(error_entry)

                # Also add to normalized error index
                if normalized_key not in log_data.errors_by_normalized:
                    log_data.errors_by_normalized[normalized_key] = {
                        "count": 0,
                        "category": error_category,
                        "examples": [],
                        "object_type_counts": {},  # placeholder -> {type -> count}
                    }
                log_data.errors_by_normalized[normalized_key]["count"] += 1
                # Track object type distributions
                for ph, obj_type in object_types.items():
                    if (
                        ph
                        not in log_data.errors_by_normalized[normalized_key][
                            "object_type_counts"
                        ]
                    ):
                        log_data.errors_by_normalized[normalized_key][
                            "object_type_counts"
                        ][ph] = {}
                    type_counts = log_data.errors_by_normalized[normalized_key][
                        "object_type_counts"
                    ][ph]
                    type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
                # Store a few examples (limit to 3)
                if len(log_data.errors_by_normalized[normalized_key]["examples"]) < 3:
                    log_data.errors_by_normalized[normalized_key]["examples"].append(
                        {
                            "task": current_task.name,
                            "step": step_number,
                            "original_error": error_text,
                        }
                    )

                # Keep reference to update images later
                step._error_entry = error_entry

            # Extract images for this step
            # Handle timestamps in image lines: [HH:MM:SS] === IMAGES ===
            image_match = re.search(
                r"=== IMAGES ===.*?"
                r"(?:(?:\[\d{2}:\d{2}:\d{2}\]\s*)?Previous:\s*(.+?)(?:\n|$))?"
                r"(?:\[\d{2}:\d{2}:\d{2}\]\s*)?Current:\s*(.+?)(?:\n|$)",
                between_text,
                re.DOTALL,
            )
            if image_match:
                prev_img_name = (
                    image_match.group(1).strip() if image_match.group(1) else None
                )
                curr_img_name = (
                    image_match.group(2).strip() if image_match.group(2) else None
                )

                # Build absolute paths if we have the task's plans directory
                prev_img_path = None
                curr_img_path = None
                if current_task_plans_dir:
                    if prev_img_name:
                        prev_img_full = current_task_plans_dir / prev_img_name
                        if prev_img_full.exists():
                            prev_img_path = str(prev_img_full)
                    if curr_img_name:
                        curr_img_full = current_task_plans_dir / curr_img_name
                        if curr_img_full.exists():
                            curr_img_path = str(curr_img_full)

                # Image semantics:
                # - "previous" is the image from step N-1, shown to model at step N as "previous"
                # - "current" is the result after step N's action, saved as N_action.png
                #
                # When analyzing step N's decision:
                # - "model_input" = "previous" = what model saw when deciding
                # - "action_result" = "current" = result after the action
                step.images = {
                    "previous": prev_img_path,  # Raw log field
                    "current": curr_img_path,  # Raw log field
                    "model_input": prev_img_path,  # Semantic: what model saw when deciding
                    "action_result": curr_img_path,  # Semantic: result after action executed
                }

                # Update error entry with images if this step had an error
                if hasattr(step, "_error_entry") and step._error_entry:
                    step._error_entry["images"] = step.images

            # Extract low-level calls for this step
            low_level_calls = re.findall(
                r"\(\d+\)\s+(PickupObject|PutObject|OpenObject|CloseObject|ToggleObject|SliceObject|NavigateTo|MoveAhead|RotateRight|RotateLeft|LookUp|LookDown|Done)",
                between_text,
            )
            if low_level_calls:
                step.low_level_calls = low_level_calls

            current_task.steps.append(step)

    # Add last task
    if current_task:
        log_data.tasks.append(current_task)

    # Parse summary table if present
    summary_match = re.search(r"Total:\s*(\d+)\s+(\d+)\s+(\d+)", content)
    if summary_match:
        total = int(summary_match.group(1))
        passed = int(summary_match.group(2))
        failed = int(summary_match.group(3))
        log_data.summary = {
            "total_tasks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
        }
    else:
        # Calculate from parsed tasks
        total = len(log_data.tasks)
        passed = sum(1 for t in log_data.tasks if t.passed)
        failed = total - passed
        log_data.summary = {
            "total_tasks": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
        }

    # Detect repetitive failures (same action repeated 3+ times with errors)
    for task in log_data.tasks:
        if not task.passed:
            action_errors = {}
            for step in task.steps:
                if step.error and step.executing_action:
                    key = step.executing_action
                    action_errors[key] = action_errors.get(key, 0) + 1

            for action, count in action_errors.items():
                if count >= 3:
                    if "repetitive_failures" not in log_data.errors_by_category:
                        log_data.errors_by_category["repetitive_failures"] = []

                    # Repetitive failures are natural when feedback_type=none
                    is_natural = (
                        config.get("feedback_type") == "none" if config else False
                    )

                    log_data.errors_by_category["repetitive_failures"].append(
                        {
                            "task": task.name,
                            "action": action,
                            "repeat_count": count,
                            "is_config_natural": is_natural,
                        }
                    )

    return log_data


def filter_prompts(content: str) -> str:
    """Remove prompt blocks from log content."""
    return re.sub(
        r"===============PROMPT=================.*?=============END PROMPT===============\s*",
        "",
        content,
        flags=re.DOTALL,
    )


def to_dict(obj):
    """Convert dataclass to dict recursively."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def load_config(log_path: Path) -> Optional[dict]:
    """Try to load config.json from the same directory as logs.txt."""
    config_path = log_path.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config.json: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Parse Magmathor experiment logs into structured JSON"
    )
    parser.add_argument("logfile", help="Path to logs.txt file")
    parser.add_argument(
        "--output", "-o", help="Output JSON file path (default: <logfile>.json)"
    )
    parser.add_argument(
        "--filter",
        "-f",
        action="store_true",
        help="Also output filtered.txt with prompts removed",
    )
    parser.add_argument(
        "--pretty",
        "-p",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True)",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to config.json (default: auto-detect from log directory)",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        help="Base experiment directory for multi-experiment mode (e.g., 20251231__Test__...)",
    )
    parser.add_argument(
        "--config-string",
        "-s",
        help="Config string for multi-experiment mode (e.g., T0_Fs_H60_C1_P2_I1_R0_S1_E60_M4096)",
    )

    args = parser.parse_args()

    log_path = Path(args.logfile)
    if not log_path.exists():
        print(f"Error: File not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    content = log_path.read_text(encoding="utf-8", errors="replace")

    # Load config (from argument or auto-detect)
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"Loaded config from: {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}", file=sys.stderr)
    else:
        config = load_config(log_path)
        if config:
            print(f"Auto-loaded config from: {log_path.parent / 'config.json'}")

    # Parse the log
    log_data = parse_log(
        content,
        str(log_path),
        config,
        base_dir=args.base_dir,
        config_string=args.config_string,
    )

    # Output JSON
    output_path = Path(args.output) if args.output else log_path.with_suffix(".json")
    with open(output_path, "w") as f:
        json.dump(to_dict(log_data), f, indent=2 if args.pretty else None)
    print(f"Parsed log saved to: {output_path}")

    # Output filtered log if requested
    if args.filter:
        filtered_content = filter_prompts(content)
        filtered_path = log_path.parent / f"filtered_{log_path.name}"
        filtered_path.write_text(filtered_content)
        print(f"Filtered log saved to: {filtered_path}")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total tasks: {log_data.summary.get('total_tasks', 0)}")
    print(f"  Passed: {log_data.summary.get('passed', 0)}")
    print(f"  Failed: {log_data.summary.get('failed', 0)}")
    print(f"  Pass rate: {log_data.summary.get('pass_rate', '0%')}")

    if log_data.config:
        print(f"\nConfig:")
        for key in [
            "text_only",
            "feedback_type",
            "previous_image",
            "use_memory",
            "include_common_sense",
        ]:
            if key in log_data.config:
                print(f"  {key}: {log_data.config[key]}")
        if log_data.config_natural_failures:
            print(f"\nExpected failure types given config:")
            for ft in log_data.config_natural_failures:
                print(f"  - {ft}")

    if log_data.errors_by_category:
        print(f"\nErrors by high-level category:")
        for category, errors in sorted(
            log_data.errors_by_category.items(), key=lambda x: -len(x[1])
        ):
            natural_count = sum(1 for e in errors if e.get("is_config_natural", False))
            if natural_count > 0:
                print(f"  {category}: {len(errors)} ({natural_count} config-natural)")
            else:
                print(f"  {category}: {len(errors)}")

    if log_data.errors_by_normalized:
        print(f"\nTop 15 normalized error patterns:")
        sorted_errors = sorted(
            log_data.errors_by_normalized.items(), key=lambda x: -x[1]["count"]
        )
        for i, (pattern, info) in enumerate(sorted_errors[:15]):
            print(
                f"  {info['count']:4d}x  {pattern[:80]}{'...' if len(pattern) > 80 else ''}"
            )


if __name__ == "__main__":
    main()
