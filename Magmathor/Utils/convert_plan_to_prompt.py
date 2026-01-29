#!/usr/bin/env python3
"""
Ultra-simplified Ray-based converter: Magmathor -> ShareGPT single-turn conversations.
- Assumes Ray is installed and available.
- No CLI; set constants below.
- Minimal helpers; Ray actors handle Ollama enhancement.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

INPUT_DIR = os.getenv("INPUT_DIR", "C:/Code/magma-r/magmathor/Generated/training")


def extract_memory_from_agent_response(agent_text: str) -> List[str]:
    items: List[str] = []
    lines = agent_text.splitlines()
    in_mem = False
    for line in lines:
        s = line.strip()
        if s.startswith("**Things I want to remember:**") or s.startswith(
            "Things I want to remember:"
        ):
            in_mem = True
            continue
        if in_mem and s.startswith("**"):  # next section starts
            break
        if in_mem and s.startswith("- "):
            items.append(s[2:].strip())
    return [x for x in items if x.strip().lstrip("-").strip()]


def action_str_from_step(step: dict[str, Any]) -> str:
    action = step["action"].upper()
    orig_obj_name = step["object"]

    if "_Slice_" in orig_obj_name or "_Sliced_" in orig_obj_name:
        # ex, orig_obj_name = Tomato_25_Slice_6
        what_is_sliced = orig_obj_name.split("_")[0]
        obj_name = f"{what_is_sliced}Sliced"

    elif "_Cracked_" in orig_obj_name:
        # ex, orig_obj_name = Egg_Cracked_26(Clone)
        what_is_cracked = orig_obj_name.split("_")[0]
        obj_name = f"{what_is_cracked}Cracked"
    else:
        obj_name = orig_obj_name.split("_")[0]  # Get the object name before any suffix

    return f"{action} {obj_name}"


def create_reasoning_trace(step: Dict[str, Any]) -> str:
    parts: List[str] = []
    if step.get("observations"):
        parts.append("**Observations:**")
        for obs in step["observations"]:
            parts.append(f"- {obs}")
        parts.append("")
    if step.get("reasoning"):
        parts.append("**Reasoning:**")
        for r in step["reasoning"]:
            if r.strip():
                parts.append(f"- {r}")
        parts.append("")
    if step.get("updated_memory"):
        parts.append("**Things I want to remember:**")
        for m in step["updated_memory"]:
            parts.append(f"- {m}")
        parts.append("")

    parts.append(f"**Action:** {action_str_from_step(step)}")
    return "\n".join(parts)


def create_formatted_parts(
    step: Dict[str, Any],
    task_description: str,
    has_image: bool,
    previous_memory: List[str],
    objects_in_scene: list[str],
    previous_actions: Optional[List[str]] = None,
) -> Dict[str, str]:
    # History
    # Prefer explicitly aggregated previous actions if provided; otherwise fallback
    # to whatever is present in the step (if any).
    history = (
        previous_actions if previous_actions is not None else step.get("history", [])
    )
    if history:
        hist_lines = ["**Previous Actions:**"] + [
            f"{i+1}. {t}" for i, t in enumerate(history)
        ]
        history_text = "\n".join(hist_lines)
    else:
        history_text = "This is the first step in the sequence."

    user_parts: List[str] = []
    if has_image:
        user_parts.append("<image>")
    user_parts += [
        f"You are acting in a simulation environment. Your goal is to perform the following task: {task_description}",
        "",
        history_text,
        "",
        "You are currently in a scene that may have the following objects:",
        f"**Available Objects**: {objects_in_scene}",
        "",
    ]
    mem = previous_memory or step.get("memory", [])
    if mem:
        user_parts.append("**Memory from previous steps:**")
        user_parts += [f"- {m}" for m in mem]
        user_parts.append("")
    user_parts += [
        "**Available Actions:**",
        "- CLEAN {object_name} - Clean {object_name}",
        "- CLOSE {object_name} - Close {object_name}",
        "- EMPTY {object_name} - Empty the contents of {object_name}",
        "- DRINK {object_name} - Drink the liquid from {object_name}",
        "- FIND {object_name} - Look around to find {object_name}",
        "- OPEN {object_name} - Open {object_name}",
        "- PICKUP {object_name} - Pickup {object_name} and hold it in your hand",
        "- PUT {object_name} - Put the object you are currently holding into {object_name}",
        "- SLICE {object_name} - Slice {object_name}, if object is an egg then crack the egg",
        "- SPRAY {object_name} - Spray {object_name} with the object you're currently holding",
        "- TOGGLE_OFF {object_name} - Turn off {object_name}",
        "- TOGGLE_ON {object_name} - Turn on {object_name}",
        "",
        "Your task: Observe the environment, reason about the current state, remember important information, and determine the next action to take.",
        "",
        "Please provide your response as natural, flowing text that explains your observations, reasoning, and decision-making process.  You MUST Include a section titled 'Things I want to remember:' with bulleted list items (each starting with '- ') for important information to carry forward. The last line MUST end with with 'Action: <action> <object>' on a new line where <action> MUST be from the <available actions> list and <object> MUST be from the <available objects> list.",
    ]
    user_part = "\n".join(user_parts)
    agent_part = create_reasoning_trace(step)
    return {"user_part": user_part, "agent_part": agent_part}


def load_plan(task_dir: str) -> Dict[str, Any]:
    with open(Path(task_dir) / "plan.json", "r") as f:
        return json.load(f)


def save_plan(task_dir: str, data: Dict[str, Any]) -> None:
    with open(Path(task_dir) / "plan.json", "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def process_all() -> None:

    input_path = Path(INPUT_DIR)
    task_dirs = [
        str(p)
        for p in input_path.iterdir()
        if p.is_dir() and (p / "plan.json").exists()
    ]

    print(f"Processing {len(task_dirs)} tasks from {input_path}.")

    items_cache_map = json.loads(
        (Path(__file__).parent.parent / "Data" / "item_cache.json").read_text(
            encoding="utf-8"
        )
    )

    # --- Parallelize enhancement across all steps in all tasks ---
    plans = []
    steps_per_task = []
    objects_in_scene_per_task = []
    task_descriptions = []
    task_names = []

    # Preload all plans and collect enhancement jobs
    for task_idx, td in enumerate(tqdm(task_dirs)):
        try:
            plan = load_plan(td)
        except Exception as e:
            print(f"Error loading {td}: {e}")
            continue

        steps = plan["steps"]
        plans.append(plan)
        steps_per_task.append(steps)
        task_description = plan["task_description"]
        task_descriptions.append(task_description)
        task_name = plan["name"]
        task_names.append(task_name)
        objects_in_scene = sorted(items_cache_map["types"][plan["scene"]])
        objects_in_scene_per_task.append(objects_in_scene)

        # Build formatted parts and schedule enhancements for THIS task only
        # Aggregate exact previous actions (not paraphrased) and carry forward memory
        current_task_idx = len(steps_per_task) - 1
        previous_memory: List[str] = []
        aggregated_prev_actions: List[str] = []
        for i, step in enumerate(steps):
            has_image = step.get("image_filename") is not None
            mem_for_this = previous_memory if i > 0 else []
            actions_for_this = aggregated_prev_actions.copy() if i > 0 else []

            # Rebuild formatted prompt components when creating or forcing recompute
            formatted = create_formatted_parts(
                step,
                task_description,
                has_image,
                mem_for_this,
                objects_in_scene,
                actions_for_this,
            )
            print(formatted["agent_part"])


if __name__ == "__main__":
    process_all()
