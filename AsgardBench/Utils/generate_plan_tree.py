"""
Generate a tree structure from multiple plan.json files showing divergence pattern.

Each node in the tree represents a unique action_desc.
Child nodes represent the next steps taken from that node across all plans.
Reasoning is tracked but does not cause nodes to split.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TreeNode:
    """A node in the plan tree."""

    action_desc: str
    reasoning: List[str]  # Reasoning from the first plan that created this node
    count: int = 0  # Number of plans that pass through this node
    plan_names: List[str] = field(
        default_factory=list
    )  # Names of plans passing through this node
    children: Dict[str, "TreeNode"] = field(
        default_factory=dict
    )  # Keyed by action_desc only

    def to_dict(self) -> dict:
        """Convert the node to a JSON-serializable dictionary."""
        return {
            "action_desc": self.action_desc,
            "reasoning": self.reasoning,
            "count": self.count,
            "plan_names": self.plan_names,
            "children": [child.to_dict() for child in self.children.values()],
        }


def load_plan(plan_path: str) -> Optional[tuple[List[dict], str, str]]:
    """Load a plan.json file and return its steps, plan name, and task description."""
    try:
        with open(plan_path, "r", encoding="utf-8") as f:
            plan = json.load(f)
        # Extract plan name from the directory name
        plan_name = os.path.basename(os.path.dirname(plan_path))
        task_description = plan.get("task_description", "")
        return plan.get("steps", []), plan_name, task_description
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load {plan_path}: {e}")
        return None


def build_tree(source_dir: str) -> Optional[tuple[TreeNode, str]]:
    """
    Build a tree from all plan.json files in the source directory.

    Args:
        source_dir: Directory containing subdirectories with plan.json files

    Returns:
        Tuple of (root node of the tree, task_description), or None if no valid plans found
    """
    # Find all plan.json files
    plan_files = []
    for entry in os.listdir(source_dir):
        plan_path = os.path.join(source_dir, entry, "plan.json")
        if os.path.isfile(plan_path):
            plan_files.append(plan_path)

    if not plan_files:
        print(f"No plan.json files found in {source_dir}")
        return None

    print(f"Found {len(plan_files)} plan files")

    # Load all plans
    all_plans: List[tuple[List[dict], str]] = []  # (steps, plan_name)
    task_description = ""
    for plan_path in plan_files:
        result = load_plan(plan_path)
        if result:
            steps, plan_name, task_desc = result
            if steps:
                all_plans.append((steps, plan_name))
                if not task_description and task_desc:
                    task_description = task_desc  # Use first non-empty task description

    if not all_plans:
        print("No valid plans loaded")
        return None

    print(f"Loaded {len(all_plans)} valid plans")

    # Create root node from the first step of the first plan
    # (assuming all plans start with the same first step)
    first_step = all_plans[0][0][0]
    root = TreeNode(
        action_desc=first_step.get("action_desc", ""),
        reasoning=first_step.get("reasoning", []),
    )

    # Process each plan
    for steps, plan_name in all_plans:
        current_node = root
        current_node.count += 1
        if plan_name not in current_node.plan_names:
            current_node.plan_names.append(plan_name)

        # Process each step (starting from index 1 since we used index 0 for root)
        for i in range(1, len(steps)):
            step = steps[i]
            action_desc = step.get("action_desc", "")
            reasoning = step.get("reasoning", [])

            # Key only on action_desc (reasoning no longer causes splits)
            key = action_desc

            # Check if this child already exists
            if key not in current_node.children:
                current_node.children[key] = TreeNode(
                    action_desc=action_desc,
                    reasoning=reasoning,  # Use reasoning from first plan that creates this node
                )

            # Move to child and increment count
            current_node = current_node.children[key]
            current_node.count += 1
            if plan_name not in current_node.plan_names:
                current_node.plan_names.append(plan_name)

    return root, task_description


def save_tree(root: TreeNode, output_path: str, task_description: str = "") -> None:
    """Save the tree to a JSON file."""
    tree_dict = root.to_dict()
    # Add task_description at the top level
    tree_dict["task_description"] = task_description

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=2)

    print(f"Tree saved to {output_path}")


def print_tree_stats(root: TreeNode, depth: int = 0, max_depth: int = 5) -> None:
    """Print statistics about the tree."""
    if depth == 0:
        print("\nTree Statistics:")
        print(f"  Root: {root.action_desc} (count: {root.count})")

    if depth < max_depth:
        for child in root.children.values():
            indent = "  " * (depth + 2)
            print(f"{indent}- {child.action_desc} (count: {child.count})")
            print_tree_stats(child, depth + 1, max_depth)


def generate_plan_tree(source_dir: str) -> None:
    """
    Main function to generate a plan tree from a directory of plans.

    Args:
        source_dir: Directory containing subdirectories with plan.json files
    """
    # Build the tree
    result = build_tree(source_dir)

    if result is None:
        return

    root, task_description = result

    # Save to output file
    output_path = os.path.join(source_dir, "plan_tree.json")
    save_tree(root, output_path, task_description)

    # Print some stats
    print_tree_stats(root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a tree structure from plan.json files showing divergence patterns."
    )
    parser.add_argument(
        "source_dir",
        type=str,
        help="Directory containing subdirectories with plan.json files",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        print(f"Error: {args.source_dir} is not a valid directory")
        return

    generate_plan_tree(args.source_dir)


if __name__ == "__main__":
    main()
