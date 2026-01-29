import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

from Magmathor.Model.prompt_dsl import PromptTemplate, parse_prompt_file
from Magmathor.step_log import log_print
from Magmathor.Utils.config_utils import (
    EvaluationConfig,
    FeedbackType,
    PreviousImageType,
)

# ----------------------------------------------------------------------
# DSL TEMPLATE LOADING
# ----------------------------------------------------------------------

_PROMPTS_DIR = Path(__file__).parent.parent / "Data" / "prompts"

_dsl_templates: dict[str, PromptTemplate] = {}


def _load_dsl_template(name: str) -> PromptTemplate:
    """Load and cache a DSL template."""
    if name not in _dsl_templates:
        path = _PROMPTS_DIR / name
        _dsl_templates[name] = parse_prompt_file(path)
    return _dsl_templates[name]


# Pre-load main template (sub-templates are loaded on-demand via {include:})
_main_prompt_template = _load_dsl_template("main_prompt.prompt")


@dataclass
class PromptParams:
    """
    All parameters for rendering a prompt.

    Contains both configuration (how to render) and context (what to render).
    """

    # Observation mode
    mode: Literal["image", "text"] = "image"
    feedback_type: Literal["none", "simple", "detailed"] = "none"

    # Content filtering
    include_common_sense: bool = True
    include_simulation: bool = True

    # Image generation
    hand_transparency: float = 0.0
    previous_image: Literal["none", "color", "grayscale"] = "none"

    # Action context
    first_action: bool = False

    # Runtime context
    task_description: str = ""
    objects_in_scene: str = ""
    action_history: str = ""
    memories: str = ""

    # Memory feature
    use_memory: bool = True

    # Full steps mode - generate complete action sequence
    full_steps: bool = False

    # Suggested plan sequence from previous full_steps response
    suggested_plan_sequence: str = ""


def _params_to_eval_config(params: PromptParams) -> EvaluationConfig:
    """
    Convert PromptParams to an EvaluationConfig for DSL template rendering.

    This bridges the old PromptParams interface with the new DSL system.
    """
    # Map previous_image to PreviousImageType
    previous_image_map = {
        "none": PreviousImageType.NONE,
        "color": PreviousImageType.COLOR,
        "grayscale": PreviousImageType.GRAYSCALE,
    }
    previous_image = previous_image_map.get(
        params.previous_image, PreviousImageType.NONE
    )

    # Map feedback_type to FeedbackType
    feedback_map = {
        "none": FeedbackType.NONE,
        "simple": FeedbackType.SIMPLE,
        "detailed": FeedbackType.DETAILED,
    }
    feedback_type = feedback_map.get(params.feedback_type, FeedbackType.NONE)

    # Mode is text_only
    text_only = params.mode == "text"

    return EvaluationConfig(
        text_only=text_only,
        feedback_type=feedback_type,
        include_common_sense=params.include_common_sense,
        hand_transparency=int(params.hand_transparency),
        previous_image=previous_image,
        use_memory=params.use_memory,
        full_steps=params.full_steps,
    )


def _render_dsl_with_includes(
    template: PromptTemplate,
    config: EvaluationConfig,
    variables: dict,
    extra_conditions: dict,
    max_depth: int = 5,
) -> str:
    """
    Render a DSL template, processing {include:filename.prompt} placeholders.

    This handles the sub-template inclusion pattern used in main_prompt.prompt.
    Includes are processed recursively up to max_depth levels.
    """
    # First render the main template
    rendered = template.render(config, variables, extra_conditions)

    # Process includes recursively
    include_pattern = r"\{include:([^}]+)\}"

    def replace_include(match):
        include_name = match.group(1)
        try:
            sub_template = _load_dsl_template(include_name)
            return sub_template.render(config, variables, extra_conditions)
        except FileNotFoundError:
            log_print(f"Warning: Include template not found: {include_name}")
            return ""

    # Process includes repeatedly until no more are found (up to max_depth)
    for _ in range(max_depth):
        new_result = re.sub(include_pattern, replace_include, rendered)
        if new_result == rendered:
            break  # No more includes to process
        rendered = new_result

    return rendered


def _render_prompt_dsl(params: PromptParams) -> str:
    """Render the main prompt using DSL templates."""
    # Convert params to EvaluationConfig
    config = _params_to_eval_config(params)

    # Compute img_ref based on previous_image
    if params.previous_image == "grayscale":
        img_ref = "the current (second, color) image"
    elif params.previous_image != "none":
        img_ref = "the current (second) image"
    else:
        img_ref = "the image"

    # Build variables dict from params
    variables = {
        "task_description": params.task_description,
        "objects_in_scene": params.objects_in_scene,
        "action_history": params.action_history,
        "memories": params.memories,
        "suggested_plan_sequence": params.suggested_plan_sequence,
        "img_ref": img_ref,
    }

    # Extra conditions not in EvaluationConfig
    # Use single-letter prefix as key, value as string "0" or "1"
    extra_conditions = {
        "A": "1" if params.first_action else "0",
    }

    # Render using main template with include processing
    rendered = _render_dsl_with_includes(
        _main_prompt_template, config, variables, extra_conditions
    )

    # Strip leading/trailing whitespace and normalize multiple blank lines
    return rendered.strip()


# Cache boundary marker used to split static vs dynamic content for prompt caching.
# Actors that support caching can split on this marker internally.
CACHE_BOUNDARY_MARKER = "<<CACHE_BOUNDARY>>"


def split_prompt_for_caching(prompt: str) -> tuple[str, str]:
    """
    Split a rendered prompt at the cache boundary marker.

    This is used by actors that support prompt caching (e.g., OpenRouter)
    to separate static content (cacheable) from dynamic content.

    Args:
        prompt: The rendered prompt string (may contain CACHE_BOUNDARY_MARKER)

    Returns:
        Tuple of (static_part, dynamic_part). If no marker is found,
        returns ("", prompt) so all content is treated as dynamic.
    """
    if CACHE_BOUNDARY_MARKER in prompt:
        static_part, dynamic_part = prompt.split(CACHE_BOUNDARY_MARKER, 1)
        return static_part.strip(), dynamic_part.strip()
    else:
        # No marker - treat entire prompt as dynamic (no static content to cache)
        return "", prompt.strip()


def strip_cache_marker(prompt: str) -> str:
    """
    Remove the cache boundary marker from a prompt.

    Use this in actors that don't support prompt caching to avoid
    the marker appearing in the model's input.

    Args:
        prompt: The rendered prompt string (may contain CACHE_BOUNDARY_MARKER)

    Returns:
        The prompt with the marker removed (no extra blank lines).
    """
    # Remove marker and any surrounding newlines to avoid extra blank lines
    # Try common patterns: marker on its own line, or inline
    result = prompt.replace(f"\n{CACHE_BOUNDARY_MARKER}\n", "\n")
    result = result.replace(CACHE_BOUNDARY_MARKER, "")
    return result.strip()


def render_prompt(params: PromptParams) -> str:
    """
    Render the main prompt template with the given parameters.

    The returned string may contain a <<CACHE_BOUNDARY>> marker that actors
    can use to split static vs dynamic content for prompt caching. Actors
    that don't support caching will simply include the marker as-is (harmless)
    or can strip it before use.

    Args:
        params: PromptParams with all rendering parameters and context

    Returns:
        Rendered prompt string (may contain cache boundary marker)

    Raises:
        ValueError: If both include_common_sense and include_simulation are False

    Example:
        params = PromptParams(
            mode="image",
            include_common_sense=True,
            task_description="Make coffee",
            objects_in_scene="Mug, CoffeeMachine",
        )
        prompt = render_prompt(params)

        # Actors can split for caching if needed:
        static, dynamic = split_prompt_for_caching(prompt)
    """
    if not params.include_common_sense and not params.include_simulation:
        raise ValueError(
            "At least one of include_common_sense or include_simulation must be True. "
            "Setting both to False would result in empty rules."
        )

    if not params.include_simulation:
        raise ValueError(
            "Simulation rules are currently required. "
            "Setting include_simulation to False is not supported yet."
        )

    rendered = _render_prompt_dsl(params)
    return rendered.strip()


# ----------------------------------------------------------------------
# RESPONSE PARSING
# ----------------------------------------------------------------------


def extract_suggested_plan_sequence_raw(response: str) -> str:
    """
    Extract the raw suggested plan sequence text from a full_steps model response.

    This captures the entire suggested plan section as-is, preserving the model's
    exact formatting including comments in parentheses.

    Args:
        response: Model response string

    Returns:
        Raw text of the suggested plan sequence section, or empty string if not found.
    """
    # Find the <answer> section
    answer_start = response.find("<answer>")
    answer_end = response.find("</answer>")

    if answer_start == -1:
        # No answer tag, search from beginning
        answer_section = response
    elif answer_end == -1:
        answer_section = response[answer_start:]
    else:
        answer_section = response[answer_start:answer_end]

    # Find "Suggested Plan Sequence:" section
    sequence_start = answer_section.lower().find("suggested plan sequence")
    if sequence_start == -1:
        return ""

    # Extract lines after the header
    sequence_section = answer_section[sequence_start:]

    # Skip the header line itself
    first_newline = sequence_section.find("\n")
    if first_newline != -1:
        sequence_section = sequence_section[first_newline + 1 :]
    else:
        return ""

    # Find where the sequence ends (at "Action:", "Things to remember", or end of section)
    end_markers = ["\naction:", "\nthings to remember", "\nthings i want to remember"]
    end_position = len(sequence_section)
    for marker in end_markers:
        marker_pos = sequence_section.lower().find(marker)
        if marker_pos != -1 and marker_pos < end_position:
            end_position = marker_pos
    sequence_section = sequence_section[:end_position]

    return sequence_section.strip()


def format_suggested_plan_sequence(sequence: List[Tuple[str, str]]) -> str:
    """
    Format a suggested plan sequence for inclusion in the prompt.

    Args:
        sequence: List of (action, object) tuples

    Returns:
        Formatted string with numbered actions
    """
    if not sequence:
        return ""

    lines = []
    for i, (action, obj) in enumerate(sequence, 1):
        lines.append(f"{i}. {action} {obj}")
    return "\n".join(lines)


def extract_memories(response: str) -> str:

    memories_for_this_round = []
    response_lower = response.lower()

    # Try new format first: "THINGS TO REMEMBER:"
    idx_things_to_remember = response_lower.find("things to remember")
    header_len = len("things to remember")

    # Fall back to old format: "Things I want to remember:"
    if idx_things_to_remember == -1:
        idx_things_to_remember = response_lower.find("things i want to remember")
        header_len = len("things i want to remember")

    if idx_things_to_remember != -1:
        this_memories = response[(idx_things_to_remember + header_len) :].strip(":\n*")
        this_memories = this_memories.split("\n\n")[0].strip()

        memories_for_this_round.append(this_memories)

    if len(memories_for_this_round) == 0:
        print("WARNING: No memories found in model response.")

    return "\n".join(memories_for_this_round)


def find_valid_string(text: str, valid_items: Iterable[str]) -> str | None:
    for valid_object in valid_items:
        if valid_object.lower() == text.lower():
            return valid_object
    return None


def extract_action_content(text: str) -> Optional[str]:
    """
    Extract the action content from the LAST 'Action: ACTION OBJECT' in the <answer> section.

    We specifically want the last occurrence because the model may output similar
    action patterns in its reasoning trace before the final action.

    Args:
        text: Input text containing Action: line

    Returns:
        str: The action and object (e.g., "FIND Mirror"), or None if not found

    Example:
        >>> extract_action_content("<answer>Action: FIND Mirror</answer>")
        "FIND Mirror"
    """
    import re

    # First, find the <answer> tag as sometimes action shows up in <thinking> part and should be ignored
    answer_start = text.find("<answer>")
    if answer_start == -1:
        # If <answer> tag is not found, start from the beginning
        answer_start = 0

    # Search only within the answer section
    answer_section = text[answer_start:]

    # Match "Action:" followed by the action and object
    # Use findall to get all matches, then take the last one
    pattern = r"Action:\s*([A-Z_]+\s+\w+)"
    matches = re.findall(pattern, answer_section)

    if matches:
        return matches[-1].strip()

    return None


def extract_action_object(
    model_response: str, valid_objects: List[str]
) -> Tuple[str | None, str | None, str | None]:
    """
    Parse model outputs and extract high-level AI2-THOR actions.

    Expected input format:
        <answer>
        Things I want to remember:
            - The mirror is dirty.
            - I have already sprayed the mirror successfully.

        Action: FIND Mirror
        </answer>

    Args:
        action: Model output string (in the format specified above)

    Returns:
        parsed_action: Action string (e.g., "PICKUP Potato")
        valids: Floats (1.0 for valid action, 0.0 for invalid)
    """

    # Define valid high-level action types (matching your team's format)
    valid_action_types = {
        "CLEAN",
        "CLOSE",
        "EMPTY",
        "DRINK",
        "FIND",
        "OPEN",
        "PICKUP",
        "PUT",
        "SLICE",
        "SPRAY",
        "TOGGLE_OFF",
        "TOGGLE_ON",
    }

    action_content = extract_action_content(model_response)
    if action_content is None:
        return None, None, "FAIL: Action content not found"

    log_print("-  Action content:", action_content)

    pieces = action_content.split()
    if len(pieces) > 1:
        action_name = find_valid_string(pieces[0], valid_action_types)
        object_name = find_valid_string(pieces[1], valid_objects)
        log_print(f"-  Extracted action: [{action_name}]")
        log_print(f"-  Extracted item: [{object_name}]")

        if action_name is None and object_name is None:
            return (
                pieces[0],
                pieces[1],
                f"-  FAIL: Invalid action and object extracted: [{pieces[0]}] [{pieces[1]}]",
            )
        elif action_name is None:
            return (
                pieces[0],
                object_name,
                f"-  FAIL: Invalid action extracted: [{pieces[0]}]",
            )
        elif object_name is None:
            return (
                action_name,
                pieces[1],
                f"-  FAIL: Invalid object extracted: [{pieces[1]}]",
            )
        else:
            return action_name, object_name, None

    return action_content, None, f"-  FAIL: Invalid format: [{action_content}]"


def history_to_prompt(
    history: List[str],
    error_msgs: List[str | None] | None = None,
    feedback_type: Literal["none", "simple", "detailed"] = "none",
) -> str:
    """Convert the history of actions into a prompt format.

    Args:
        history: List of action strings (e.g., "FIND Mug").
        error_msgs: List of error messages (None for success) for each action.
        feedback_type: How to format feedback:
            - "none": No feedback (just action list)
            - "simple": Append Success/Failure
            - "detailed": Append Success or Failure with reason

    Returns:
        Formatted history string for prompt.
    """
    # Filter out unparseable actions (None None) - they provide no useful information
    filtered = [
        (action, error_msgs[i] if error_msgs else None)
        for i, action in enumerate(history)
        if action != "None None"
    ]

    lines = []
    for i, (action, error_msg) in enumerate(filtered):
        if feedback_type == "none":
            lines.append(f"{i}. {action}")
        elif error_msg is None:
            lines.append(f"{i}. {action}  Success")
        elif feedback_type == "simple":
            lines.append(f"{i}. {action}  Failure")
        else:  # detailed
            cleaned = re.sub(r"(\w+)_[a-zA-Z0-9]+", r"\1", error_msg)
            lines.append(f"{i}. {action}  Failure: {cleaned}")

    return "\n".join(lines) + "\n" if lines else ""


if __name__ == "__main__":
    import argparse
    import itertools
    from pathlib import Path

    from Magmathor.Model.prompt_dsl import format_config_to_dsl_values

    parser = argparse.ArgumentParser(description="Preview prompt templates")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="PromptExamples",
        help="Directory to write template files (default: PromptExamples)",
    )
    parser.add_argument(
        "--verbose-names",
        action="store_true",
        help="Use verbose names instead of short DSL codes",
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)

    # remove if exists and recreate
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # Sample runtime context
    sample_context = {
        "task_description": "Make coffee and serve it",
        "objects_in_scene": "Mug, CoffeeMachine, CounterTop, SinkBasin, Faucet, DishSponge, Plate, Fork",
        "action_history": "1. FIND Mug\n2. PICKUP Mug",
        "memories": "- Mug was found on CounterTop\n- Currently holding Mug",
        "suggested_plan_sequence": "",
    }

    # v2 parameter options (DSL-style codes in comments)
    # T: text_only (0=image, 1=text)
    # F: feedback_type (n=none, s=simple, d=detailed)
    # H: hand_transparency (0=off, 1=on) - simplified for generation
    # C: include_common_sense (0=false, 1=true)
    # I: previous_image (0=none, 1=color, 2=grayscale)
    # R: use_memory (0=false, 1=true)
    # S: full_steps (0=false, 1=true)
    # A: first_action (0=false, 1=true) - extra condition

    v2_options = {
        "mode": ["image", "text"],  # T0, T1
        "feedback_type": ["none", "simple", "detailed"],  # Fn, Fs, Fd
        "first_action": [True, False],  # A1, A0
        "include_common_sense": [False],  # C1 only (C0 is invalid for v2)
        "hand_transparency": [0, 60],  # H0, H1 (any non-zero is H1)
        "previous_image": ["none", "color", "grayscale"],  # I0, I1, I2
        "use_memory": [True, False],  # R1, R0
        "full_steps": [True, False],  # S1, S0
    }

    configs_to_render: list[dict] = []

    # Generate combinations
    for values in itertools.product(*v2_options.values()):
        config = dict(zip(v2_options.keys(), values))
        configs_to_render.append(config)

    print(f"Generating {len(configs_to_render)} prompt combinations to {output_path}/")

    for config in configs_to_render:
        # Create params (config + sample context)
        params = PromptParams(**config, **sample_context)

        # Convert to EvaluationConfig for DSL code generation
        eval_config = _params_to_eval_config(params)
        dsl_values = format_config_to_dsl_values(eval_config)

        # Add first_action (A) which isn't in EvaluationConfig
        dsl_values["A"] = "1" if params.first_action else "0"

        # Build filename from DSL codes: T0_Fn_H0_C1_I0_R1_S0_A1.txt
        # Sort by a fixed order for consistency
        code_order = ["T", "F", "H", "C", "I", "R", "S", "A"]
        parts = [f"{prefix}{dsl_values[prefix]}" for prefix in code_order]
        base_name = "_".join(parts)

        # Mark specific configurations with __ prefix (primary test configs)
        primary_configs = {
            "T0_Fs_H1_C0_I2_R0_S1",
            "T0_Fs_H1_C0_I2_R1_S1",
            "T0_Fs_H1_C0_I1_R0_S1",
            "T0_Fs_H1_C0_I2_R0_S0",
        }
        # Check without the A suffix
        base_without_a = "_".join(parts[:-1])
        prefix = "__" if base_without_a in primary_configs else ""

        filename = prefix + base_name + ".txt"

        rendered = render_prompt(params)

        filepath = output_path / filename
        filepath.write_text(rendered)

    print(f"Generated {len(configs_to_render)} files in {output_path}/")
