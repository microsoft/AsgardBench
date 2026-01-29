"""
Display a plan_tree.json file as an interactive graphical tree.

Usage:
    python -m AsgardBench.Utils.display_plan_tree <path_to_plan_tree.json>
    python -m AsgardBench.Utils.display_plan_tree <directory_containing_plan_tree.json>
"""

import argparse
import json
import os
from typing import Optional

from bokeh.io import save
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    Range1d,
    Slider,
    TapTool,
)
from bokeh.plotting import figure


def load_tree(tree_path: str) -> dict:
    """Load the plan tree from a JSON file."""
    with open(tree_path, "r") as f:
        return json.load(f)


def simplify_action(action_desc: str) -> str:
    """Simplify action descriptions by removing 'in Y' from 'Put X in Y' patterns."""
    import re

    # Match "Put <object> in <container>" and simplify to "Put <object>"
    return re.sub(r"^(Put \S+) in \S+$", r"\1", action_desc)


def flatten_tree(
    node: dict,
    parent_id: Optional[int] = None,
    node_id: int = 0,
    depth: int = 0,
) -> tuple[list[dict], list[dict], int]:
    """
    Flatten the tree into lists of nodes and edges.

    Returns:
        Tuple of (nodes, edges, next_available_id)
    """
    nodes = []
    edges = []

    # Get reasoning - store both last item and full chain
    reasoning_list = node.get("reasoning", [])
    # Filter out empty strings
    reasoning_list = [r for r in reasoning_list if r and r.strip()]
    last_reasoning = reasoning_list[-1] if reasoning_list else ""
    full_reasoning = " | ".join(reasoning_list) if reasoning_list else ""

    current_node = {
        "id": node_id,
        "action_desc": simplify_action(node.get("action_desc", "ROOT")),
        "last_reasoning": last_reasoning,
        "full_reasoning": full_reasoning,
        "count": node.get("count", 0),
        "plan_names": node.get("plan_names", []),
        "depth": depth,
        "parent_id": parent_id,
    }
    nodes.append(current_node)

    if parent_id is not None:
        edges.append({"source": parent_id, "target": node_id})

    next_id = node_id + 1

    for child in node.get("children", []):
        child_nodes, child_edges, next_id = flatten_tree(
            child, node_id, next_id, depth + 1
        )
        nodes.extend(child_nodes)
        edges.extend(child_edges)

    return nodes, edges, next_id


def collapse_single_child_chains(
    nodes: list[dict], edges: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Collapse chains of nodes where each node has only one child into a single node.
    Nodes immediately after a split (parent has multiple children) are not collapsed.

    Returns:
        Tuple of (collapsed_nodes, collapsed_edges)
    """
    # Build parent-child maps
    children_map: dict[int, list[int]] = {}
    parent_map: dict[int, int] = {}
    for e in edges:
        src, tgt = e["source"], e["target"]
        if src not in children_map:
            children_map[src] = []
        children_map[src].append(tgt)
        parent_map[tgt] = src

    # Find root
    all_targets = {e["target"] for e in edges}
    root_id = None
    for n in nodes:
        if n["id"] not in all_targets:
            root_id = n["id"]
            break

    if root_id is None and nodes:
        root_id = nodes[0]["id"]

    # Build node lookup
    node_lookup = {n["id"]: n for n in nodes}

    # Track which nodes are collapsed into which
    # collapsed_into[node_id] = representative_node_id
    collapsed_into: dict[int, int] = {}

    def is_after_split(nid: int) -> bool:
        """Check if this node is immediately after a split (parent has multiple children)."""
        if nid not in parent_map:
            return False  # Root node
        parent_id = parent_map[nid]
        return len(children_map.get(parent_id, [])) > 1

    def find_chain_end(start_id: int) -> tuple[int, list[int]]:
        """
        Follow a chain from start_id until we hit a node with != 1 child.
        Do not include leaf nodes (nodes with 0 children) in the chain - they should never be collapsed.
        Returns (end_node_id, list_of_all_node_ids_in_chain).
        """
        chain = [start_id]
        current = start_id
        while True:
            children = children_map.get(current, [])
            if len(children) != 1:
                # End of chain (0 or multiple children)
                break
            child = children[0]
            child_children = children_map.get(child, [])
            if len(child_children) == 0:
                # Child is a leaf node - don't include it in the chain
                break
            chain.append(child)
            current = child
        return current, chain

    # Process the tree, collapsing chains
    collapsed_nodes = []
    collapsed_edges = []
    processed = set()
    new_depth_map: dict[int, int] = {}  # Maps representative node id to new depth

    def process_node(nid: int, current_depth: int, can_start_chain: bool = True):
        if nid in processed:
            return
        processed.add(nid)

        node = node_lookup[nid]
        children = children_map.get(nid, [])

        # Only start a chain if:
        # 1. This node has exactly 1 child
        # 2. This node is allowed to start a chain (not immediately after a split)
        if len(children) == 1 and can_start_chain:
            # Start of a potential chain - find where it ends
            end_id, chain = find_chain_end(nid)

            # Mark all chain nodes as processed and collapsed into the first
            for cid in chain:
                processed.add(cid)
                collapsed_into[cid] = nid

            # Create a collapsed node
            chain_nodes = [node_lookup[cid] for cid in chain]
            chain_actions = [cn["action_desc"] for cn in chain_nodes]
            chain_reasonings = []
            for cn in chain_nodes:
                if cn["full_reasoning"]:
                    chain_reasonings.append(cn["full_reasoning"])

            # Use the last node's count and plan_names (should be same for chain)
            last_node = chain_nodes[-1]

            # original_depth is the depth of the last node in the chain (total uncollapsed nodes from root)
            original_depth = last_node["depth"]

            collapsed_node = {
                "id": nid,  # Use first node's id as representative
                "action_desc": " → ".join(chain_actions),
                "last_reasoning": last_node["last_reasoning"],
                "full_reasoning": " ||| ".join(
                    chain_reasonings
                ),  # Triple pipe to separate chain steps
                "count": last_node["count"],
                "plan_names": last_node["plan_names"],
                "depth": current_depth,
                "original_depth": original_depth,  # Total uncollapsed nodes from root
                "parent_id": node["parent_id"],
                "is_collapsed": len(chain) > 1,
                "chain_length": len(chain),
                "chain_actions": chain_actions,  # Store individual actions for details
            }
            collapsed_nodes.append(collapsed_node)
            new_depth_map[nid] = current_depth

            # Process children of the end node
            # These children are after a potential split, so check if end node has multiple children
            end_children = children_map.get(end_id, [])
            children_can_start_chain = (
                len(end_children) == 1
            )  # Only allow chain if single child
            for child_id in end_children:
                if child_id not in processed:
                    # Add edge from collapsed node to child
                    collapsed_edges.append({"source": nid, "target": child_id})
                    process_node(
                        child_id,
                        current_depth + 1,
                        can_start_chain=children_can_start_chain,
                    )
        else:
            # Not a chain start - regular node (either has !=1 children or can't start chain)
            collapsed_node = {
                "id": nid,
                "action_desc": node["action_desc"],
                "last_reasoning": node["last_reasoning"],
                "full_reasoning": node["full_reasoning"],
                "count": node["count"],
                "plan_names": node["plan_names"],
                "depth": current_depth,
                "original_depth": node["depth"],  # Total uncollapsed nodes from root
                "parent_id": node["parent_id"],
                "is_collapsed": False,
                "chain_length": 1,
                "chain_actions": [node["action_desc"]],
            }
            collapsed_nodes.append(collapsed_node)
            new_depth_map[nid] = current_depth

            # Process all children
            # If this node has multiple children, those children cannot start chains
            children_can_start_chain = len(children) == 1
            for child_id in children:
                if child_id not in processed:
                    collapsed_edges.append({"source": nid, "target": child_id})
                    process_node(
                        child_id,
                        current_depth + 1,
                        can_start_chain=children_can_start_chain,
                    )

    if root_id is not None:
        process_node(root_id, 0, can_start_chain=True)

    # Update depths in collapsed nodes based on new_depth_map
    for node in collapsed_nodes:
        node["depth"] = new_depth_map.get(node["id"], node["depth"])

    # Update edge targets to point to representative nodes
    final_edges = []
    for e in collapsed_edges:
        src = collapsed_into.get(e["source"], e["source"])
        tgt = collapsed_into.get(e["target"], e["target"])
        # Only add edge if both nodes are in our collapsed nodes
        if any(n["id"] == src for n in collapsed_nodes) and any(
            n["id"] == tgt for n in collapsed_nodes
        ):
            final_edges.append({"source": src, "target": tgt})

    # Mark final (leaf) nodes - nodes with no children in the collapsed tree
    nodes_with_children = {e["source"] for e in final_edges}
    for node in collapsed_nodes:
        node["is_final"] = node["id"] not in nodes_with_children

    # Build parent map for collapsed nodes
    collapsed_parent_map: dict[int, int] = {}
    for e in final_edges:
        collapsed_parent_map[e["target"]] = e["source"]

    # Build node lookup for collapsed nodes
    collapsed_node_lookup = {n["id"]: n for n in collapsed_nodes}

    # Compute full action path from root to each node
    def get_action_path(nid: int) -> list[str]:
        """Get the list of all actions from root to this node."""
        path = []
        current = nid
        while current is not None:
            node = collapsed_node_lookup.get(current)
            if node:
                # chain_actions contains all actions in this node (may be multiple if collapsed)
                path = node.get("chain_actions", [node["action_desc"]]) + path
            current = collapsed_parent_map.get(current)
        return path

    for node in collapsed_nodes:
        node["action_path"] = get_action_path(node["id"])

    return collapsed_nodes, final_edges


def compute_tree_layout(
    nodes: list[dict], edges: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Compute x, y positions for each node in a tree layout.
    x = depth (horizontal), y = vertical position within depth level.

    Uses proportional spacing based on subtree size to prevent overlapping.
    """
    # Build parent-child map
    children_map: dict[int, list[int]] = {}
    for e in edges:
        src = e["source"]
        if src not in children_map:
            children_map[src] = []
        children_map[src].append(e["target"])

    # Find root
    all_targets = {e["target"] for e in edges}
    root_id = None
    for n in nodes:
        if n["id"] not in all_targets:
            root_id = n["id"]
            break

    if root_id is None and nodes:
        root_id = nodes[0]["id"]

    # Cache leaf counts for each node (memoized)
    leaf_count_cache: dict[int, int] = {}

    def count_leaves(nid: int) -> int:
        if nid in leaf_count_cache:
            return leaf_count_cache[nid]
        children = children_map.get(nid, [])
        if not children:
            leaf_count_cache[nid] = 1
            return 1
        total = sum(count_leaves(c) for c in children)
        leaf_count_cache[nid] = total
        return total

    total_leaves = count_leaves(root_id) if root_id is not None else 1

    # Layout subtree recursively with proportional spacing
    node_positions: dict[int, tuple[float, float]] = {}

    def layout_subtree(nid: int, y_min: float, y_max: float):
        children = children_map.get(nid, [])
        y_center = (y_min + y_max) / 2

        # Find this node's depth
        node = next(n for n in nodes if n["id"] == nid)
        node_positions[nid] = (float(node["depth"]), y_center)

        if not children:
            return

        # Allocate vertical space proportionally based on each child's subtree size
        total_child_leaves = sum(count_leaves(c) for c in children)
        y_range = y_max - y_min

        current_y = y_min
        for child_id in children:
            child_leaves = count_leaves(child_id)
            # Proportional allocation based on number of leaves in subtree
            child_height = (child_leaves / total_child_leaves) * y_range
            child_y_min = current_y
            child_y_max = current_y + child_height
            layout_subtree(child_id, child_y_min, child_y_max)
            current_y = child_y_max

    if root_id is not None:
        layout_subtree(root_id, 0, total_leaves)

    # Update nodes with positions
    for node in nodes:
        if node["id"] in node_positions:
            node["x"], node["y"] = node_positions[node["id"]]
        else:
            node["x"], node["y"] = 0.0, 0.0

    # Update edges with coordinates
    for edge in edges:
        src_x, src_y = node_positions.get(edge["source"], (0, 0))
        tgt_x, tgt_y = node_positions.get(edge["target"], (0, 0))
        edge["x0"] = src_x
        edge["y0"] = src_y
        edge["x1"] = tgt_x
        edge["y1"] = tgt_y

    return nodes, edges


def create_tree_visualization(tree_path: str, max_depth: Optional[int] = None):
    """
    Create a Bokeh visualization of the plan tree.

    Args:
        tree_path: Path to plan_tree.json file
        max_depth: Maximum depth to display (None = all)

    Returns:
        Bokeh layout object
    """
    tree = load_tree(tree_path)

    # Extract task_description for the chart title
    task_description = tree.get("task_description", "Plan Tree")
    if not task_description:
        task_description = "Plan Tree"

    nodes, edges, _ = flatten_tree(tree)
    # Collapse single-child chains
    nodes, edges = collapse_single_child_chains(nodes, edges)
    nodes, edges = compute_tree_layout(nodes, edges)

    # Find the actual max depth in data
    actual_max_depth = max(n["depth"] for n in nodes) if nodes else 0
    default_depth = actual_max_depth  # Default to showing full tree

    # Create labels (action only, and with reasoning)
    for node in nodes:
        node["label_action_only"] = node["action_desc"]
        if node["last_reasoning"]:
            node["label_with_reasoning"] = (
                f"{node['action_desc']} → {node['last_reasoning']}"
            )
        else:
            node["label_with_reasoning"] = node["action_desc"]

    # Create data sources
    node_source = ColumnDataSource(
        data={
            "id": [n["id"] for n in nodes],
            "x": [n["x"] for n in nodes],
            "y": [n["y"] for n in nodes],
            "action_desc": [n["action_desc"] for n in nodes],
            "last_reasoning": [n["last_reasoning"] for n in nodes],
            "full_reasoning": [n["full_reasoning"] for n in nodes],
            "count": [n["count"] for n in nodes],
            "depth": [n["depth"] for n in nodes],
            "original_depth": [
                n["original_depth"] for n in nodes
            ],  # Total uncollapsed depth
            "plan_names": [
                "\n".join(n["plan_names"]) for n in nodes
            ],  # Join for display
            "plan_names_count": [len(n["plan_names"]) for n in nodes],
            "label_action_only": [n["label_action_only"] for n in nodes],
            "label_with_reasoning": [n["label_with_reasoning"] for n in nodes],
            # For display - will be updated by callbacks
            # Collapsed nodes don't show action labels, only regular nodes do
            "display_label": [
                n["label_action_only"] if not n.get("is_collapsed", False) else ""
                for n in nodes
            ],
            "node_alpha": [1.0 if n["depth"] <= default_depth else 0.0 for n in nodes],
            # Collapsed chain info
            "is_collapsed": [n.get("is_collapsed", False) for n in nodes],
            "chain_length": [n.get("chain_length", 1) for n in nodes],
            "chain_actions": [
                "\n".join(n.get("chain_actions", [n["action_desc"]])) for n in nodes
            ],
            # Full action path from root to this node
            "action_path": ["\n".join(n.get("action_path", [])) for n in nodes],
            # Final node flag
            "is_final": [n.get("is_final", False) for n in nodes],
            # Color based on node type: green for final, purple for collapsed, steelblue for regular
            "node_color": [
                (
                    "limegreen"
                    if n.get("is_final", False)
                    else "mediumorchid" if n.get("is_collapsed", False) else "steelblue"
                )
                for n in nodes
            ],
            "line_color": [
                (
                    "darkgreen"
                    if n.get("is_final", False)
                    else "purple" if n.get("is_collapsed", False) else "darkblue"
                )
                for n in nodes
            ],
        }
    )

    edge_source = ColumnDataSource(
        data={
            "x0": [e["x0"] for e in edges],
            "y0": [e["y0"] for e in edges],
            "x1": [e["x1"] for e in edges],
            "y1": [e["y1"] for e in edges],
            "source_depth": [
                next(n["depth"] for n in nodes if n["id"] == e["source"]) for e in edges
            ],
            "target_depth": [
                next(n["depth"] for n in nodes if n["id"] == e["target"]) for e in edges
            ],
            "edge_alpha": [
                (
                    0.5
                    if next(n["depth"] for n in nodes if n["id"] == e["target"])
                    <= default_depth
                    else 0.0
                )
                for e in edges
            ],
        }
    )

    # Calculate plot dimensions
    max_y = max(n["y"] for n in nodes) if nodes else 10
    max_x = actual_max_depth

    # Create figure with responsive sizing (no title - we'll add a clickable one)
    p = figure(
        title="",
        x_range=Range1d(-0.5, max_x + 0.7),
        y_range=Range1d(-1, max_y + 1),
        tools="pan,wheel_zoom,box_zoom,reset",
        active_scroll="wheel_zoom",
        sizing_mode="stretch_both",  # Responsive sizing
        min_height=400,
    )

    p.xaxis.visible = False
    p.yaxis.visible = False
    p.grid.visible = False

    # Draw edges
    p.segment(
        x0="x0",
        y0="y0",
        x1="x1",
        y1="y1",
        source=edge_source,
        line_color="gray",
        line_alpha="edge_alpha",
        line_width=1,
    )

    # Create separate data sources for regular, collapsed, and final nodes
    regular_indices = [
        i
        for i, n in enumerate(nodes)
        if not n.get("is_collapsed", False) and not n.get("is_final", False)
    ]
    collapsed_indices = [
        i
        for i, n in enumerate(nodes)
        if n.get("is_collapsed", False) and not n.get("is_final", False)
    ]
    final_indices = [i for i, n in enumerate(nodes) if n.get("is_final", False)]

    def get_subset(data_dict, indices):
        return {k: [v[i] for i in indices] for k, v in data_dict.items()}

    regular_source = ColumnDataSource(
        data=get_subset(node_source.data, regular_indices)
    )
    collapsed_source = ColumnDataSource(
        data=get_subset(node_source.data, collapsed_indices)
    )
    final_source = ColumnDataSource(data=get_subset(node_source.data, final_indices))

    # Use scatter with size (screen pixels) for consistent sizing
    node_size_px = 12  # size in pixels

    # For collapsed nodes, add the chain length as text to display inside the oval
    collapsed_data = collapsed_source.data
    collapsed_data["chain_length_text"] = [
        str(cl) for cl in collapsed_data["chain_length"]
    ]
    # Calculate oval dimensions in screen pixels to match scatter marker size
    # Height matches the circle size, width grows with chain length
    oval_height_px = node_size_px + 2  # Slightly taller than circle for text
    base_oval_width_px = node_size_px + 6  # Base width slightly wider than circle
    width_per_step_px = 3  # Additional width per step
    collapsed_data["oval_width_px"] = [
        base_oval_width_px
        + width_per_step_px * min(cl - 1, 9)  # -1 because base already accounts for 1
        for cl in collapsed_data["chain_length"]
    ]
    collapsed_data["oval_height_px"] = [oval_height_px] * len(collapsed_data["id"])
    collapsed_source = ColumnDataSource(data=collapsed_data)

    # Draw regular nodes as circles
    regular_renderer = p.scatter(
        x="x",
        y="y",
        source=regular_source,
        size=node_size_px,
        marker="circle",
        fill_color="node_color",
        fill_alpha="node_alpha",
        line_color="line_color",
        line_alpha="node_alpha",
        selection_fill_color="orange",
        selection_line_color="darkorange",
        nonselection_fill_alpha="node_alpha",
        nonselection_line_alpha="node_alpha",
    )

    # Draw collapsed nodes as rounded rectangles (pill shape) with chain length number inside
    # Using rect with screen units for consistent sizing like scatter markers
    collapsed_renderer = p.rect(
        x="x",
        y="y",
        width="oval_width_px",
        height="oval_height_px",
        width_units="screen",
        height_units="screen",
        source=collapsed_source,
        fill_color="steelblue",
        fill_alpha="node_alpha",
        line_color="darkblue",
        line_alpha="node_alpha",
        line_width=1.5,
        border_radius=6,  # Rounded corners to look like a pill/oval
    )

    # Draw the chain length number inside the oval
    p.text(
        x="x",
        y="y",
        source=collapsed_source,
        text="chain_length_text",
        text_font_size="8pt",
        text_alpha="node_alpha",
        text_color="white",
        text_font_style="bold",
        text_align="center",
        text_baseline="middle",
    )

    # Use the ellipse renderer directly for selection/hover (no hidden hit area needed)
    collapsed_hit_renderer = collapsed_renderer

    # Draw final (leaf) nodes as green triangles
    final_renderer = p.scatter(
        x="x",
        y="y",
        source=final_source,
        size=node_size_px,
        marker="triangle",  # Triangle marker for final nodes
        fill_color="node_color",
        fill_alpha="node_alpha",
        line_color="line_color",
        line_alpha="node_alpha",
        selection_fill_color="orange",
        selection_line_color="darkorange",
        nonselection_fill_alpha="node_alpha",
        nonselection_line_alpha="node_alpha",
    )

    # Draw labels - offset based on shape width
    p.text(
        x="x",
        y="y",
        source=regular_source,
        text="display_label",
        text_font_size="7pt",
        text_alpha="node_alpha",
        x_offset=10,
        y_offset=0,
        text_baseline="middle",
    )
    p.text(
        x="x",
        y="y",
        source=collapsed_source,
        text="display_label",
        text_font_size="7pt",
        text_alpha="node_alpha",
        x_offset=15,  # Larger offset for ellipse width
        y_offset=0,
        text_baseline="middle",
    )
    p.text(
        x="x",
        y="y",
        source=final_source,
        text="display_label",
        text_font_size="7pt",
        text_alpha="node_alpha",
        x_offset=10,
        y_offset=0,
        text_baseline="middle",
    )

    # Add hover tool for all renderers
    # Use HTML tooltips for vertical action list
    hover = HoverTool(
        renderers=[regular_renderer, collapsed_hit_renderer, final_renderer],
        tooltips="""
            <div style="max-width: 400px;">
                <div><b>Actions:</b></div>
                <div style="white-space: pre-wrap;">@chain_actions</div>
                <div style="margin-top: 5px;"><b>Steps:</b> @chain_length</div>
                <div><b>Count:</b> @count</div>
                <div><b>Total Depth:</b> @original_depth</div>
                <div><b>Reasoning:</b> @last_reasoning</div>
            </div>
        """,
    )
    p.add_tools(hover)

    # Add tap tool for selection on all renderers (use hit renderer for collapsed)
    tap = TapTool(renderers=[regular_renderer, collapsed_hit_renderer, final_renderer])
    p.add_tools(tap)

    # Create clickable title button
    title_button = Button(
        label=f"{task_description} ⚙️",
        button_type="light",
        width=None,
        styles={
            "font-size": "16px",
            "font-weight": "bold",
            "text-align": "center",
            "width": "100%",
        },
    )

    # Create widgets
    depth_slider = Slider(
        start=1,
        end=actual_max_depth,
        value=default_depth,
        step=1,
        title="Max Depth",
        width=250,
    )

    show_action_checkbox = CheckboxGroup(
        labels=["Show Action"],
        active=[0],  # Checked by default
        width=120,
    )

    show_reasoning_checkbox = CheckboxGroup(
        labels=["Show Reasoning"],
        active=[],  # Empty = unchecked
        width=140,
    )

    # Settings popup (hidden by default)
    settings_div = Div(
        text="",
        width=0,
        height=0,
        visible=False,
    )

    # Settings close button (not needed but kept for layout compatibility)
    settings_close_button = Div(
        text="",
        width=0,
        height=0,
        visible=False,
    )

    # Details panel as floating popup (hidden by default)
    details_div = Div(
        text="",
        width=400,
        styles={
            "display": "none",
            "position": "fixed",
            "top": "100px",
            "right": "20px",
            "padding": "15px",
            "padding-top": "45px",  # Make room for close button
            "background": "#e8f4f8",
            "border": "2px solid #2c5282",
            "border-radius": "8px",
            "overflow-y": "auto",
            "max-height": "70vh",
            "box-shadow": "0 4px 12px rgba(0,0,0,0.3)",
            "z-index": "1000",
        },
    )

    # Close button for popup
    close_button = Button(
        label="✕",
        button_type="danger",
        width=40,
        styles={
            "display": "none",
            "position": "fixed",
            "top": "108px",
            "right": "28px",
            "z-index": "1001",
        },
    )

    # Close button callback
    close_callback = CustomJS(
        args=dict(details_div=details_div, close_button=close_button),
        code="""
        details_div.styles = {...details_div.styles, display: 'none'};
        close_button.styles = {...close_button.styles, display: 'none'};
        """,
    )
    close_button.js_on_click(close_callback)

    # Settings close button callback (not needed, kept for compatibility)
    settings_close_callback = CustomJS(
        args=dict(
            settings_div=settings_div, settings_close_button=settings_close_button
        ),
        code="",
    )

    # Title click callback - toggle settings controls visibility
    # We'll set the actual callback after settings_controls is created

    # JavaScript callback for depth slider
    depth_callback = CustomJS(
        args=dict(
            regular_source=regular_source,
            collapsed_source=collapsed_source,
            final_source=final_source,
            edge_source=edge_source,
        ),
        code="""
        const max_depth = cb_obj.value;

        // Update regular node visibility
        const reg_data = regular_source.data;
        for (let i = 0; i < reg_data['depth'].length; i++) {
            reg_data['node_alpha'][i] = reg_data['depth'][i] <= max_depth ? 1.0 : 0.0;
        }

        // Update collapsed node visibility
        const col_data = collapsed_source.data;
        for (let i = 0; i < col_data['depth'].length; i++) {
            col_data['node_alpha'][i] = col_data['depth'][i] <= max_depth ? 1.0 : 0.0;
        }

        // Update final node visibility
        const fin_data = final_source.data;
        for (let i = 0; i < fin_data['depth'].length; i++) {
            fin_data['node_alpha'][i] = fin_data['depth'][i] <= max_depth ? 1.0 : 0.0;
        }

        // Update edge visibility
        const edge_data = edge_source.data;
        for (let i = 0; i < edge_data['target_depth'].length; i++) {
            edge_data['edge_alpha'][i] = edge_data['target_depth'][i] <= max_depth ? 0.5 : 0.0;
        }

        regular_source.change.emit();
        collapsed_source.change.emit();
        final_source.change.emit();
        edge_source.change.emit();
        """,
    )
    depth_slider.js_on_change("value", depth_callback)

    # JavaScript callback for action checkbox
    # Only regular (circle) and final (triangle) nodes show action labels, not collapsed (square) nodes
    action_callback = CustomJS(
        args=dict(
            regular_source=regular_source,
            collapsed_source=collapsed_source,
            final_source=final_source,
            show_reasoning_checkbox=show_reasoning_checkbox,
        ),
        code="""
        const show_action = cb_obj.active.includes(0);
        const show_reasoning = show_reasoning_checkbox.active.includes(0);

        // Update regular node labels (circles) - these show actions
        const reg_data = regular_source.data;
        for (let i = 0; i < reg_data['id'].length; i++) {
            let label = '';
            if (show_action) {
                label = reg_data['action_desc'][i];
            }
            if (show_reasoning && reg_data['last_reasoning'][i]) {
                if (label) {
                    label += ' → ' + reg_data['last_reasoning'][i];
                } else {
                    label = reg_data['last_reasoning'][i];
                }
            }
            reg_data['display_label'][i] = label;
        }

        // Collapsed nodes (squares) never show action labels, only reasoning if enabled
        const col_data = collapsed_source.data;
        for (let i = 0; i < col_data['id'].length; i++) {
            let label = '';
            if (show_reasoning && col_data['last_reasoning'][i]) {
                label = col_data['last_reasoning'][i];
            }
            col_data['display_label'][i] = label;
        }

        // Final nodes (triangles) show actions like regular nodes
        const fin_data = final_source.data;
        for (let i = 0; i < fin_data['id'].length; i++) {
            let label = '';
            if (show_action) {
                label = fin_data['action_desc'][i];
            }
            if (show_reasoning && fin_data['last_reasoning'][i]) {
                if (label) {
                    label += ' → ' + fin_data['last_reasoning'][i];
                } else {
                    label = fin_data['last_reasoning'][i];
                }
            }
            fin_data['display_label'][i] = label;
        }

        regular_source.change.emit();
        collapsed_source.change.emit();
        final_source.change.emit();
        """,
    )
    show_action_checkbox.js_on_change("active", action_callback)

    # JavaScript callback for reasoning checkbox
    # Only regular (circle) and final (triangle) nodes show action labels, not collapsed (square) nodes
    reasoning_callback = CustomJS(
        args=dict(
            regular_source=regular_source,
            collapsed_source=collapsed_source,
            final_source=final_source,
            show_action_checkbox=show_action_checkbox,
        ),
        code="""
        const show_reasoning = cb_obj.active.includes(0);
        const show_action = show_action_checkbox.active.includes(0);

        // Update regular node labels (circles) - these show actions
        const reg_data = regular_source.data;
        for (let i = 0; i < reg_data['id'].length; i++) {
            let label = '';
            if (show_action) {
                label = reg_data['action_desc'][i];
            }
            if (show_reasoning && reg_data['last_reasoning'][i]) {
                if (label) {
                    label += ' → ' + reg_data['last_reasoning'][i];
                } else {
                    label = reg_data['last_reasoning'][i];
                }
            }
            reg_data['display_label'][i] = label;
        }

        // Collapsed nodes (squares) never show action labels, only reasoning if enabled
        const col_data = collapsed_source.data;
        for (let i = 0; i < col_data['id'].length; i++) {
            let label = '';
            if (show_reasoning && col_data['last_reasoning'][i]) {
                label = col_data['last_reasoning'][i];
            }
            col_data['display_label'][i] = label;
        }

        // Final nodes (triangles) show actions like regular nodes
        const fin_data = final_source.data;
        for (let i = 0; i < fin_data['id'].length; i++) {
            let label = '';
            if (show_action) {
                label = fin_data['action_desc'][i];
            }
            if (show_reasoning && fin_data['last_reasoning'][i]) {
                if (label) {
                    label += ' → ' + fin_data['last_reasoning'][i];
                } else {
                    label = fin_data['last_reasoning'][i];
                }
            }
            fin_data['display_label'][i] = label;
        }

        regular_source.change.emit();
        collapsed_source.change.emit();
        final_source.change.emit();
        """,
    )
    show_reasoning_checkbox.js_on_change("active", reasoning_callback)

    # JavaScript callback for regular node selection (tap)
    regular_selection_callback = CustomJS(
        args=dict(
            source=regular_source,
            details_div=details_div,
            collapsed_source=collapsed_source,
            final_source=final_source,
            close_button=close_button,
        ),
        code="""
        const indices = source.selected.indices;
        if (indices.length === 0) {
            // Don't clear details when deselected - let the other source handle it
            return;
        }

        // Clear selection on other sources (without triggering their callbacks to clear details)
        if (collapsed_source.selected.indices.length > 0) {
            collapsed_source.selected.indices = [];
        }
        if (final_source.selected.indices.length > 0) {
            final_source.selected.indices = [];
        }

        const idx = indices[0];
        const data = source.data;

        const action = data['action_desc'][idx];
        const count = data['count'][idx];
        const depth = data['depth'][idx];
        const full_reasoning = data['full_reasoning'][idx];
        const plan_names = data['plan_names'][idx];

        let reasoning_html = "(none)";
        if (full_reasoning) {
            const items = full_reasoning.split(' | ');
            reasoning_html = "<ol style='margin: 5px 0 0 20px; padding: 0;'>";
            for (const item of items) {
                reasoning_html += "<li>" + item + "</li>";
            }
            reasoning_html += "</ol>";
        }

        let plans_html = "(none)";
        if (plan_names) {
            const items = plan_names.split('\\n');
            plans_html = "<ul style='margin: 5px 0 0 20px; padding: 0; list-style-type: disc;'>";
            for (const item of items) {
                plans_html += "<li style='margin: 2px 0;'>" + item + "</li>";
            }
            plans_html += "</ul>";
        }

        // Show popup and close button
        details_div.styles = {...details_div.styles, display: 'block'};
        close_button.styles = {...close_button.styles, display: 'block'};
        details_div.text = `
            <h3 style="margin: 0 0 10px 0; color: #2c5282;">Selected Node</h3>
            <div style="margin: 5px 0;"><b>Action:</b> ${action}</div>
            <div style="margin: 5px 0;"><b>Count:</b> ${count}</div>
            <div style="margin: 5px 0;"><b>Depth:</b> ${depth}</div>
            <div style="margin: 5px 0;"><b>Reasoning Chain:</b>${reasoning_html}</div>
            <div style="margin: 10px 0 5px 0;"><b>Plans (${count}):</b>${plans_html}</div>
        `;
        """,
    )
    regular_source.selected.js_on_change("indices", regular_selection_callback)

    # JavaScript callback for collapsed node selection (tap)
    collapsed_selection_callback = CustomJS(
        args=dict(
            source=collapsed_source,
            details_div=details_div,
            regular_source=regular_source,
            final_source=final_source,
            close_button=close_button,
        ),
        code="""
        const indices = source.selected.indices;
        if (indices.length === 0) {
            // Don't clear details when deselected - let the other source handle it
            return;
        }

        // Clear selection on other sources
        if (regular_source.selected.indices.length > 0) {
            regular_source.selected.indices = [];
        }
        if (final_source.selected.indices.length > 0) {
            final_source.selected.indices = [];
        }

        const idx = indices[0];
        const data = source.data;

        const action = data['action_desc'][idx];
        const count = data['count'][idx];
        const depth = data['depth'][idx];
        const full_reasoning = data['full_reasoning'][idx];
        const plan_names = data['plan_names'][idx];
        const chain_length = data['chain_length'][idx];
        const chain_actions = data['chain_actions'][idx];

        // Build actions display as list
        let actions_html = "<ol style='margin: 5px 0 0 20px; padding: 0;'>";
        if (chain_actions) {
            const items = chain_actions.split('\\n');
            for (const item of items) {
                actions_html += "<li>" + item + "</li>";
            }
        }
        actions_html += "</ol>";

        let reasoning_html = "(none)";
        if (full_reasoning) {
            const chain_parts = full_reasoning.split(' ||| ');
            if (chain_parts.length > 1) {
                reasoning_html = "<div style='margin: 5px 0 0 0;'>";
                for (let i = 0; i < chain_parts.length; i++) {
                    const items = chain_parts[i].split(' | ');
                    reasoning_html += "<div style='margin: 5px 0;'><b>Step " + (i+1) + ":</b><ol style='margin: 2px 0 0 20px; padding: 0;'>";
                    for (const item of items) {
                        if (item.trim()) {
                            reasoning_html += "<li>" + item + "</li>";
                        }
                    }
                    reasoning_html += "</ol></div>";
                }
                reasoning_html += "</div>";
            } else {
                const items = full_reasoning.split(' | ');
                reasoning_html = "<ol style='margin: 5px 0 0 20px; padding: 0;'>";
                for (const item of items) {
                    reasoning_html += "<li>" + item + "</li>";
                }
                reasoning_html += "</ol>";
            }
        }

        let plans_html = "(none)";
        if (plan_names) {
            const items = plan_names.split('\\n');
            plans_html = "<ul style='margin: 5px 0 0 20px; padding: 0; list-style-type: disc;'>";
            for (const item of items) {
                plans_html += "<li style='margin: 2px 0;'>" + item + "</li>";
            }
            plans_html += "</ul>";
        }

        // Show popup and close button
        details_div.styles = {...details_div.styles, display: 'block'};
        close_button.styles = {...close_button.styles, display: 'block'};
        details_div.text = `
            <h3 style="margin: 0 0 10px 0; color: #2c5282;">Selected Node <span style='color: steelblue; font-size: 0.9em;'>(collapsed chain of ${chain_length} steps)</span></h3>
            <div style="margin: 5px 0;"><b>Actions:</b>${actions_html}</div>
            <div style="margin: 5px 0;"><b>Count:</b> ${count}</div>
            <div style="margin: 5px 0;"><b>Collapsed Depth:</b> ${depth}</div>
            <div style="margin: 5px 0;"><b>Reasoning Chain:</b>${reasoning_html}</div>
            <div style="margin: 10px 0 5px 0;"><b>Plans (${count}):</b>${plans_html}</div>
        `;
        """,
    )
    collapsed_source.selected.js_on_change("indices", collapsed_selection_callback)

    # JavaScript callback for final node selection (tap)
    final_selection_callback = CustomJS(
        args=dict(
            source=final_source,
            details_div=details_div,
            regular_source=regular_source,
            collapsed_source=collapsed_source,
            close_button=close_button,
        ),
        code="""
        const indices = source.selected.indices;
        if (indices.length === 0) {
            // Don't clear details when deselected - let the other source handle it
            return;
        }

        // Clear selection on other sources
        if (regular_source.selected.indices.length > 0) {
            regular_source.selected.indices = [];
        }
        if (collapsed_source.selected.indices.length > 0) {
            collapsed_source.selected.indices = [];
        }

        const idx = indices[0];
        const data = source.data;

        const action = data['action_desc'][idx];
        const count = data['count'][idx];
        const depth = data['depth'][idx];
        const original_depth = data['original_depth'][idx];
        const full_reasoning = data['full_reasoning'][idx];
        const plan_names = data['plan_names'][idx];
        const action_path = data['action_path'][idx];

        // Build full action path display
        let action_path_html = "<ol style='margin: 5px 0 0 20px; padding: 0;'>";
        if (action_path) {
            const items = action_path.split('\\n');
            for (const item of items) {
                action_path_html += "<li>" + item + "</li>";
            }
        }
        action_path_html += "</ol>";

        let reasoning_html = "(none)";
        if (full_reasoning) {
            const items = full_reasoning.split(' | ');
            reasoning_html = "<ol style='margin: 5px 0 0 20px; padding: 0;'>";
            for (const item of items) {
                reasoning_html += "<li>" + item + "</li>";
            }
            reasoning_html += "</ol>";
        }

        let plans_html = "(none)";
        if (plan_names) {
            const items = plan_names.split('\\n');
            plans_html = "<ul style='margin: 5px 0 0 20px; padding: 0; list-style-type: disc;'>";
            for (const item of items) {
                plans_html += "<li style='margin: 2px 0;'>" + item + "</li>";
            }
            plans_html += "</ul>";
        }

        // Show popup and close button
        details_div.styles = {...details_div.styles, display: 'block'};
        close_button.styles = {...close_button.styles, display: 'block'};
        details_div.text = `
            <h3 style="margin: 0 0 10px 0; color: #2c5282;">Selected Node <span style='color: green; font-size: 0.9em;'>(final)</span></h3>
            <div style="margin: 5px 0;"><b>Full Action Path:</b>${action_path_html}</div>
            <div style="margin: 5px 0;"><b>Count:</b> ${count}</div>
            <div style="margin: 5px 0;"><b>Total Depth:</b> ${original_depth}</div>
            <div style="margin: 5px 0;"><b>Reasoning Chain:</b>${reasoning_html}</div>
            <div style="margin: 10px 0 5px 0;"><b>Plans (${count}):</b>${plans_html}</div>
        `;
        """,
    )
    final_source.selected.js_on_change("indices", final_selection_callback)

    # Layout: clickable title on top, chart takes full width, popups are floating
    # Settings controls in a column that's hidden by default
    settings_controls = column(
        depth_slider,
        show_action_checkbox,
        show_reasoning_checkbox,
        visible=False,  # Hidden by default
    )

    # Title click callback - toggle settings controls visibility
    title_click_callback = CustomJS(
        args=dict(settings_controls=settings_controls),
        code="""
        settings_controls.visible = !settings_controls.visible;
        """,
    )
    title_button.js_on_click(title_click_callback)

    layout = column(
        title_button,
        settings_controls,
        p,
        settings_div,
        settings_close_button,
        details_div,
        close_button,
        sizing_mode="stretch_both",
    )

    return layout


def display_plan_tree(
    tree_path: str,
    max_depth: Optional[int] = None,
    output_html: Optional[str] = None,
):
    """
    Display the plan tree.

    Args:
        tree_path: Path to plan_tree.json file
        max_depth: Maximum depth to display
        output_html: If provided, save to this HTML file; otherwise saves next to input
    """
    layout = create_tree_visualization(tree_path, max_depth)

    if output_html is None:
        # Default to saving next to the input file
        base_dir = os.path.dirname(tree_path)
        output_html = os.path.join(base_dir, "plan_tree.html")

    # Save as standalone HTML
    save(layout, filename=output_html, title="Plan Tree Visualization")

    abs_path = os.path.abspath(output_html)
    print(f"Chart saved to {abs_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Display a plan_tree.json file as an interactive graphical tree."
    )
    parser.add_argument(
        "tree_path",
        type=str,
        help="Path to plan_tree.json file (or directory containing it)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth to display (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML file path (default: plan_tree.html next to input)",
    )
    args = parser.parse_args()

    # Handle directory input
    tree_path = args.tree_path
    if os.path.isdir(tree_path):
        tree_path = os.path.join(tree_path, "plan_tree.json")

    if not os.path.isfile(tree_path):
        print(f"Error: {tree_path} not found")
        return

    display_plan_tree(tree_path, args.max_depth, args.output)


if __name__ == "__main__":
    main()
