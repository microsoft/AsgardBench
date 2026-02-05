"""
Image comparison tool for comparing source and test images side by side.

Usage:
    streamlit run AsgardBench/Utils/compare_images.py -- --benchmark <benchmark_name>

Example:
    streamlit run AsgardBench/Utils/compare_images.py -- --benchmark benchmark
"""

import argparse
import base64
import json
import os
import re

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


def compute_image_difference(image_path1: str, image_path2: str) -> int:
    """
    Compare two images and return a difference score from 0 to 100.

    0 = identical images
    100 = completely different images

    Uses downsampled mean absolute difference for speed.
    """
    try:
        # Open and convert to RGB
        img1 = Image.open(image_path1).convert("RGB")
        img2 = Image.open(image_path2).convert("RGB")

        # Downsample to small size for speed (32x32)
        size = (32, 32)
        img1_small = img1.resize(size, Image.Resampling.BILINEAR)
        img2_small = img2.resize(size, Image.Resampling.BILINEAR)

        # Get pixel data
        pixels1 = list(img1_small.getdata())
        pixels2 = list(img2_small.getdata())

        # Calculate mean absolute difference
        total_diff = 0
        for p1, p2 in zip(pixels1, pixels2):
            # Sum of absolute differences for R, G, B
            total_diff += abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) + abs(p1[2] - p2[2])

        # Normalize: max possible diff is 255*3 per pixel, times number of pixels
        max_diff = 255 * 3 * len(pixels1)
        score = int((total_diff / max_diff) * 100)

        return min(100, max(0, score))
    except Exception:
        return -1  # Error case


def load_observations_lookup(dir_path: str) -> dict[str, list[str]]:
    """
    Load plan.json and return a lookup dict mapping image_filename to observations.

    Returns:
        Dict mapping image filename (lowercase) to list of observation strings.
    """
    plan_path = os.path.join(dir_path, "plan.json")
    if not os.path.exists(plan_path):
        return {}

    try:
        with open(plan_path, "r") as f:
            plan_data = json.load(f)

        lookup = {}
        for step in plan_data.get("steps", []):
            image_filename = step.get("image_filename", "")
            observations = step.get("observations", [])
            if image_filename:
                # Use lowercase for case-insensitive matching
                lookup[image_filename.lower()] = observations
        return lookup
    except (json.JSONDecodeError, IOError):
        return {}


def strip_bracket_suffix(dir_name: str) -> str:
    """
    Remove the bracket suffix with image count from directory name.

    Example: 'cook_Egg_dirty_plate_FloorPlan22_V1 [50]' -> 'cook_Egg_dirty_plate_FloorPlan22_V1'
    """
    return re.sub(r"\s*\[\d+\]$", "", dir_name)


# Default paths (can be overridden)
GENERATED_DIR = "Generated"
TEST_DIR = "Test"


def get_source_dir(benchmark: str) -> str:
    """Get the source directory path."""
    return os.path.join(GENERATED_DIR, benchmark)


def get_compare_dir(benchmark: str) -> str:
    """Get the compare directory path."""
    return os.path.join(TEST_DIR, benchmark, "replay/Plans")


def get_matching_directories(benchmark: str) -> list[tuple[str, str, str]]:
    """
    Find directories in source that have matching underscore-prefixed directories in compare.

    Returns:
        List of tuples: (directory_name, source_path, compare_path)
    """
    source_dir = get_source_dir(benchmark)
    compare_dir = get_compare_dir(benchmark)

    if not os.path.exists(source_dir):
        st.error(f"Source directory not found: {os.path.abspath(source_dir)}")
        return []

    if not os.path.exists(compare_dir):
        st.error(f"Compare directory not found: {os.path.abspath(compare_dir)}")
        return []

    matching = []
    for dir_name in sorted(os.listdir(source_dir)):
        source_path = os.path.join(source_dir, dir_name)
        if not os.path.isdir(source_path):
            continue

        # Strip bracket suffix (e.g., " [50]") and add underscore prefix for compare directory
        # Source: cook_Egg_dirty_plate_FloorPlan22_V1 [50]
        # Compare: _cook_Egg_dirty_plate_FloorPlan22_V1
        base_name = strip_bracket_suffix(dir_name)
        compare_name = f"_{base_name}"
        compare_path = os.path.join(compare_dir, compare_name)

        if os.path.exists(compare_path) and os.path.isdir(compare_path):
            matching.append((dir_name, source_path, compare_path))

    return matching


def is_valid_png_filename(filename: str) -> bool:
    """Check if filename is a valid .png file (not a Windows alternate data stream)."""
    return filename.lower().endswith(".png") and ":" not in filename


def get_matching_images(source_path: str, compare_path: str) -> list[tuple[str, str]]:
    """
    Get list of .png files that exist in both source and compare directories.
    Matching is done by step number (the leading number in the filename).

    Returns:
        Sorted list of tuples: (source_filename, compare_filename)
    """
    source_files = {f for f in os.listdir(source_path) if is_valid_png_filename(f)}
    compare_files = {f for f in os.listdir(compare_path) if is_valid_png_filename(f)}

    def get_step_number(filename: str) -> int | None:
        """Extract the leading step number from a filename like '7_Turn on Faucet.png'."""
        try:
            return int(filename.split("_")[0])
        except (ValueError, IndexError):
            return None

    # Create lookup by step number for compare files
    compare_by_step = {}
    for f in compare_files:
        step_num = get_step_number(f)
        if step_num is not None:
            compare_by_step[step_num] = f

    # Find files that have matching step numbers
    matching_files = []
    for source_file in source_files:
        step_num = get_step_number(source_file)
        if step_num is not None and step_num in compare_by_step:
            compare_file = compare_by_step[step_num]
            matching_files.append((source_file, compare_file))

    # Sort by step number
    def sort_key(file_tuple):
        try:
            return int(file_tuple[0].split("_")[0])
        except (ValueError, IndexError):
            return file_tuple[0]

    return sorted(matching_files, key=sort_key)


def main():
    st.set_page_config(layout="wide", page_title="Image Comparison Tool")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compare source and test images")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="benchmark",
        help="Benchmark directory name",
    )

    # Streamlit passes args after '--'
    try:
        args = parser.parse_args()
        benchmark = args.benchmark
    except SystemExit:
        benchmark = "benchmark"

    # Get matching directories
    matching_dirs = get_matching_directories(benchmark)

    if not matching_dirs:
        st.warning("No matching directories found.")
        st.info(
            f"Looking for directories in `{os.path.abspath(get_source_dir(benchmark))}` "
            f"with underscore-prefixed matches in `{os.path.abspath(get_compare_dir(benchmark))}`"
        )
        return

    # Initialize session state
    if "dir_index" not in st.session_state:
        st.session_state.dir_index = 0
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    # Handle keyboard navigation via query params
    query_params = st.query_params
    if "nav" in query_params:
        nav_action = query_params["nav"]
        st.query_params.clear()

        if nav_action == "prev" and st.session_state.image_index > 0:
            st.session_state.image_index -= 1
            st.rerun()
        elif nav_action == "next":
            # We'll check bounds after getting images
            st.session_state._pending_next = True
        elif nav_action == "next_diff":
            # Jump to next image with diff > 5%
            st.session_state._pending_next_diff = True

    # Directory selector at top
    dir_names = [d[0] for d in matching_dirs]
    selected_dir = st.selectbox(
        "Select Directory",
        options=range(len(dir_names)),
        format_func=lambda i: f"{i + 1}. {dir_names[i]}",
        index=st.session_state.dir_index,
    )

    if selected_dir != st.session_state.dir_index:
        st.session_state.dir_index = selected_dir
        st.session_state.image_index = 0
        st.rerun()

    # Get current directory info
    dir_name, source_path, compare_path = matching_dirs[st.session_state.dir_index]

    # Get matching images
    images = get_matching_images(source_path, compare_path)

    if not images:
        st.warning(f"No matching images found in: {dir_name}")
        return

    # Handle pending next action (now that we know image count)
    if getattr(st.session_state, "_pending_next", False):
        st.session_state._pending_next = False
        if st.session_state.image_index < len(images) - 1:
            st.session_state.image_index += 1
            st.rerun()

    # Handle pending next_diff action - find next image with diff > 5%
    if getattr(st.session_state, "_pending_next_diff", False):
        st.session_state._pending_next_diff = False
        for i in range(st.session_state.image_index + 1, len(images)):
            src_img, cmp_img = images[i]
            src_path = os.path.join(source_path, src_img)
            cmp_path = os.path.join(compare_path, cmp_img)
            diff = compute_image_difference(src_path, cmp_path)
            if diff > 5:
                st.session_state.image_index = i
                st.rerun()
        # If no diff found, stay on current image

    # Load observations from plan.json files
    source_observations = load_observations_lookup(source_path)
    compare_observations = load_observations_lookup(compare_path)

    # Check if done with current directory
    is_last_image = st.session_state.image_index >= len(images) - 1

    # Get current image filenames (source and compare may have different casing)
    source_image, compare_image = images[st.session_state.image_index]

    # Get observations for current images
    source_obs = source_observations.get(source_image.lower(), [])
    compare_obs = compare_observations.get(compare_image.lower(), [])

    def format_observations_html(
        observations: list[str], other_observations: list[str]
    ) -> str:
        """Format observations as HTML bullet list, highlighting differences in red."""
        if not observations:
            return "<p style='color: gray; font-style: italic;'>No observations</p>"
        other_set = set(other_observations)
        items = []
        for obs in observations:
            if obs in other_set:
                items.append(f"<li>{obs}</li>")
            else:
                items.append(f"<li style='color: red;'>{obs}</li>")
        return f"<ul style='text-align: left; font-size: 14px;'>{''.join(items)}</ul>"

    # Display images side by side using HTML table
    source_image_path = os.path.join(source_path, source_image)
    compare_image_path = os.path.join(compare_path, compare_image)

    # Get previous image name if available
    prev_image_name = ""
    if st.session_state.image_index > 0:
        prev_source_image, _ = images[st.session_state.image_index - 1]
        # Remove .png suffix
        prev_image_name = prev_source_image.rsplit(".", 1)[0]

    # Current image name without .png suffix
    current_image_name = source_image.rsplit(".", 1)[0]

    # Format the image name line with arrow if there's a previous image
    if prev_image_name:
        image_name_html = f"<p style='font-size: 12px; color: gray;'>{prev_image_name} → {current_image_name}</p>"
    else:
        image_name_html = (
            f"<p style='font-size: 12px; color: gray;'>{current_image_name}</p>"
        )

    if os.path.exists(source_image_path) and os.path.exists(compare_image_path):
        with open(source_image_path, "rb") as f:
            source_data = base64.b64encode(f.read()).decode()
        with open(compare_image_path, "rb") as f:
            compare_data = base64.b64encode(f.read()).decode()

        source_obs_html = format_observations_html(source_obs, compare_obs)
        compare_obs_html = format_observations_html(compare_obs, source_obs)

        # Determine compare title color - red if directory name starts with underscore
        compare_dir_name = os.path.basename(compare_path)
        compare_title_style = "color: red;" if compare_dir_name.startswith("_") else ""

        # Compute image difference score
        diff_score = compute_image_difference(source_image_path, compare_image_path)
        if diff_score >= 0:
            # Red if diff > 5%, otherwise green
            if diff_score > 5:
                score_color = "#dc3545"  # red
            else:
                score_color = "#28a745"  # green
            score_html = f"<div id='diff-badge' style='position: absolute; bottom: 5px; right: 5px; background: {score_color}; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; font-size: 14px; cursor: pointer;'>Diff: {diff_score}%</div>"
        else:
            score_html = ""

        html = f"""
        <table style="width: 100%; table-layout: fixed;">
            <tr>
                <td style="width: 50%; text-align: center; vertical-align: top; padding: 10px;">
                    <h4>Source (Generated)</h4>
                    <img id="left-image" src="data:image/png;base64,{source_data}" style="max-width: 100%; height: auto; cursor: pointer;">
                    {image_name_html}
                </td>
                <td style="width: 50%; text-align: center; vertical-align: top; padding: 10px; position: relative;">
                    <h4 style="{compare_title_style}">Compare (Test/Replay)</h4>
                    <div style="position: relative; display: inline-block;">
                        <img id="right-image" src="data:image/png;base64,{compare_data}" style="max-width: 100%; height: auto; cursor: pointer;">
                        {score_html}
                    </div>
                    {image_name_html}
                </td>
            </tr>
        </table>
        """
        st.markdown(html, unsafe_allow_html=True)
    else:
        if not os.path.exists(source_image_path):
            st.error(f"Source image not found: {source_image_path}")
        if not os.path.exists(compare_image_path):
            st.error(f"Compare image not found: {compare_image_path}")

    # Navigation buttons - centered with small gaps
    # Order: prev dir, prev image, next diff, next image, next dir
    # Labels: ⏮️ Plan, ⬅️ Image, 🔴 Diff, ➡️ Image, ⏭️ Plan
    cols = st.columns([3.5, 1.2, 1.2, 1, 1.2, 1.2, 3.5])
    with cols[1]:
        prev_dir_clicked = st.button("⏮️ Plan", disabled=st.session_state.dir_index == 0)
    with cols[2]:
        prev_clicked = st.button("⬅️ Image", disabled=st.session_state.image_index == 0)
    with cols[3]:
        next_diff_clicked = st.button("🔴 Diff", key="next_diff_btn")
    with cols[4]:
        next_clicked = st.button(
            "➡️ Image", disabled=st.session_state.image_index >= len(images) - 1
        )
    with cols[5]:
        next_dir_clicked = st.button(
            "⏭️ Plan",
            disabled=st.session_state.dir_index >= len(matching_dirs) - 1,
        )

    # Image slider (below buttons, no label)
    image_idx = st.slider(
        label="Image",
        min_value=0,
        max_value=len(images) - 1,
        value=st.session_state.image_index,
        label_visibility="collapsed",
    )

    # Progress indicator
    st.caption(
        f"Image {st.session_state.image_index + 1} of {len(images)} | "
        f"Directory {st.session_state.dir_index + 1} of {len(matching_dirs)}"
    )

    # Show done message if on last image
    if is_last_image:
        st.success(
            f"✅ Done with **{dir_name}**! "
            f"Click 'Next Dir' to continue to the next directory."
        )

    # Display observations below the fixed navigation elements
    if os.path.exists(source_image_path) and os.path.exists(compare_image_path):
        obs_html = f"""
        <table style="width: 100%; table-layout: fixed;">
            <tr>
                <td style="width: 50%; text-align: center; vertical-align: top; padding: 10px;">
                    <strong>Observations:</strong>
                    {source_obs_html}
                </td>
                <td style="width: 50%; text-align: center; vertical-align: top; padding: 10px;">
                    <strong>Observations:</strong>
                    {compare_obs_html}
                </td>
            </tr>
        </table>
        """
        st.markdown(obs_html, unsafe_allow_html=True)

    st.divider()

    # Handle button clicks after all UI is rendered
    if prev_clicked:
        st.session_state.image_index -= 1
        st.rerun()
    if next_clicked:
        st.session_state.image_index += 1
        st.rerun()
    if prev_dir_clicked:
        st.session_state.dir_index -= 1
        st.session_state.image_index = 0
        st.rerun()
    if next_dir_clicked:
        st.session_state.dir_index += 1
        st.session_state.image_index = 0
        st.rerun()
    if next_diff_clicked:
        # Find next image with diff > 5%
        for i in range(st.session_state.image_index + 1, len(images)):
            src_img, cmp_img = images[i]
            src_path = os.path.join(source_path, src_img)
            cmp_path = os.path.join(compare_path, cmp_img)
            diff = compute_image_difference(src_path, cmp_path)
            if diff > 5:
                st.session_state.image_index = i
                st.rerun()
    if image_idx != st.session_state.image_index:
        st.session_state.image_index = image_idx
        st.rerun()

    # Show balloons if all directories reviewed
    if is_last_image:
        if st.session_state.dir_index >= len(matching_dirs) - 1:
            st.balloons()
            st.success("🎉 All directories reviewed!")

    # JavaScript for keyboard and click navigation
    components.html(
        """
        <script>
        const doc = window.parent.document;

        // Helper function to click a Streamlit button by text content
        function clickButton(buttonText) {
            const buttons = doc.querySelectorAll('button[kind="secondary"]');
            for (const btn of buttons) {
                if (btn.textContent.includes(buttonText)) {
                    btn.click();
                    return true;
                }
            }
            return false;
        }

        // Keyboard navigation
        doc.addEventListener('keydown', function(e) {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                clickButton('Prev') || clickButton('⬅️');
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                clickButton('Next') || clickButton('➡️');
            }
        });

        // Click on left image -> previous
        const leftImg = doc.getElementById('left-image');
        if (leftImg) {
            leftImg.addEventListener('click', function() {
                clickButton('Prev') || clickButton('⬅️');
            });
        }

        // Click on right image -> next
        const rightImg = doc.getElementById('right-image');
        if (rightImg) {
            rightImg.addEventListener('click', function() {
                clickButton('Next') || clickButton('➡️');
            });
        }

        // Click on diff badge -> next diff
        const diffBadge = doc.getElementById('diff-badge');
        if (diffBadge) {
            diffBadge.addEventListener('click', function() {
                clickButton('Next Diff') || clickButton('🔴');
            });
        }
        </script>
        """,
        height=0,
    )


if __name__ == "__main__":
    main()
