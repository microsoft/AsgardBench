import hashlib
import json
import random
import re
from typing import List

import numpy as np

import Magmathor.constants as c


def array_to_np(position):
    return np.array([position["x"], position["y"], position["z"]])


def np_to_array(position):
    return {"x": float(position[0]), "y": float(position[1]), "z": float(position[2])}


def horizontal_distance(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """
    Calculate the horizontal distance between two positions, ignoring height (y).
    """
    # Calculate using only x and z coordinates (indices 0 and 2)
    return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[2] - pos2[2]) ** 2)


def max_interaction_distance(obj_type):
    if obj_type in c.OBJECT_MAX_INTERACTION_DIST:
        return c.OBJECT_MAX_INTERACTION_DIST[obj_type]
    else:
        return c.DEFAULT_MAX_INTERACTION_DIST


def min_interaction_distance(obj_type):
    if obj_type in c.OBJECT_MIN_INTERACTION_DIST:
        return c.OBJECT_MIN_INTERACTION_DIST[obj_type]
    else:
        return c.DEFAULT_MIN_INTERACTION_DIST


def join_with_and(items: list[str]) -> str:
    """
    Join a list of strings with commas and 'and' before the last item.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"

    # For 3 or more items (no comma before 'and')
    return f"{', '.join(items[:-1])} and {items[-1]}"


def join_with_or(items: list[str]) -> str:
    """
    Join a list of strings with commas and 'or' before the last item.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} or {items[1]}"

    # For 3 or more items (no comma before 'or')
    return f"{', '.join(items[:-1])} or {items[-1]}"


def add_color(color: c.Color, text: str) -> str:
    """
    Add color to a string.
    """
    return f"{color.value}{text}{c.Color.RESET.value}"  # Reset color at the end


def print_color(color: c.Color, text: str) -> None:
    """
    Print colored text to the console, unless printing in AML log mode (then prints plain text).
    """
    if c.IN_AML is True:
        print(text)
        return

    colored_text = add_color(color, text)
    print(colored_text)


def print_start_overwrite():
    """
    Prepare for line overwriting in console.
    Call this once before you start a sequence of overwrite operations.
    """
    if c.IN_AML:
        return

    # Clear any previous output on the current line
    print("", end="\r", flush=True)


def print_overwrite(color: c.Color, text):
    """
    Print text that overwrites the current line.

    Args:
        text: Text to print
        color: Optional Color enum value to color the text
    """
    if c.IN_AML:
        print(text)
        return

    # Calculate terminal width (fallback to 80 if can't determine)
    try:
        import shutil

        terminal_width = shutil.get_terminal_size().columns
    except:
        terminal_width = 80

    # Truncate text if too long for terminal
    if len(text) > terminal_width:
        text = text[: terminal_width - 3] + "..."

    # Add padding to clear any previous longer text
    padded_text = text.ljust(terminal_width)

    # Apply color if specified
    if color:
        padded_text = f"{color.value}{padded_text}{c.Color.RESET.value}"

    # Print with carriage return to start of line
    print(padded_text, end="\r", flush=True)


def print_end_overwrite():
    """
    End line overwriting and move to the next line.
    Call this when you're done with overwrite operations.
    """
    print()  # Print empty line to move cursor to next line


def sort_by_angle(poses, reference_angle):
    """
    Sort poses by their angle relative to a reference angle.
    Handles the circular nature of angles (0° and 360° are the same).
    """
    if not poses:
        return []

    def angular_distance(angle1, angle2):
        """Calculate the minimum angular distance between two angles in degrees."""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    # Sort poses based on the angular distance to the reference angle
    sorted_poses = sorted(
        poses, key=lambda pose: angular_distance(pose["rotation"], reference_angle)
    )

    return sorted_poses


def select_diverse_poses(poses, n):
    """
    Select n pose indices that maximize spatial diversity.

    """
    if not poses:
        return []

    # Cap n to available poses
    n = min(n, len(poses))
    if n == len(poses):
        return poses

    # Define distance function between poses
    def calc_distance(pose1, pose2):
        # Only use x and z coordinates (ignore height)
        p1 = np.array([pose1["position"]["x"], pose1["position"]["z"]])
        p2 = np.array([pose2["position"]["x"], pose2["position"]["z"]])
        return np.linalg.norm(p1 - p2)

    # Start with a random pose
    selected_indices = [random.randint(0, len(poses) - 1)]

    # Iteratively add the pose that's furthest from already selected poses
    while len(selected_indices) < n:
        max_min_distance = -1
        max_index = -1

        # For each unselected pose
        for i in range(len(poses)):
            if i in selected_indices:
                continue

            # Find minimum distance to any already selected pose
            min_distance = float("inf")
            for j in selected_indices:
                dist = calc_distance(poses[i], poses[j])
                min_distance = min(min_distance, dist)

            # Keep track of the pose with maximum minimum distance
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                max_index = i

        selected_indices.append(max_index)

    # Return the actual poses rather than indices
    new_poses = [poses[i] for i in selected_indices]
    return new_poses


def average_angles(angles):
    """
    Calculate the average of a list of angles in degrees [0, 360).
    Takes into account the circular nature of angles.
    """
    if not angles:
        return 0

    # Convert angles to radians
    angles_rad = np.radians(angles)

    # Convert angles to unit vectors and sum them
    x_sum = 0
    y_sum = 0

    for angle in angles_rad:
        x_sum += np.cos(angle)
        y_sum += np.sin(angle)

    # Calculate the average vector
    x_avg = x_sum / len(angles)
    y_avg = y_sum / len(angles)

    # Handle case where vectors cancel out (e.g., average of 90 and 270)
    if np.isclose(x_avg, 0, atol=1e-10) and np.isclose(y_avg, 0, atol=1e-10):
        # No clear average direction, return 0 by convention
        return 0

    # Convert back to angle in degrees
    avg_angle_rad = np.arctan2(y_avg, x_avg)
    avg_angle_deg = np.degrees(avg_angle_rad)

    # Normalize to [0, 360)
    avg_angle_deg = avg_angle_deg % 360

    return avg_angle_deg


def action_name_from_action(action):
    """
    Extract the action name from a method string.
    """
    try:
        method_str = str(action)
        # Find the position after "Scenario.agent"
        start_marker = "Scenario.agent_"
        start_pos = method_str.find(start_marker)
        if start_pos == -1:
            return ""

        # Start position is right after "Scenario.agent"
        start_pos += len(start_marker)

        # Find " of" after start_pos
        end_marker = " of"
        end_pos = method_str.find(end_marker, start_pos)
        if end_pos == -1:
            return ""

        # Return the substring between start_pos and end_pos
        return method_str[start_pos:end_pos]
    except:
        return "UNKNOWN_ACTION"


def clear_directory(directory):
    """
    Clear all files in a directory.
    """
    import glob
    import os

    # Ensure the directory exists
    if not os.path.exists(directory):
        return

    # Get all files in the directory
    files = glob.glob(os.path.join(directory, "*"))

    # Remove each file
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error removing file {file}: {e}")


def short_name(name):
    if name is None:
        return ""

    if "Slice" in name:
        if "Lettuce" in name:
            return "LettuceSliced"
        if "Tomato" in name:
            return "TomatoSliced"
        if "Bread" in name:
            return "BreadSliced"
        if "Potato" in name:
            return "PotatoSliced"
        if "Apple" in name:
            return "AppleSliced"
        else:
            raise ValueError(f"Unknown sliced object: {name}")

    if "Cracked" in name:
        if "Egg" in name:
            return "EggCracked"
        else:
            raise ValueError(f"Unknown cracked object: {name}")

    # Find the position of the last underscore
    last_underscore = name.rfind("_")

    # If underscore found, return everything before it
    if last_underscore != -1:
        return name[:last_underscore]

    return name


def make_object_hash(obj, exclude_fields=None):
    exclude_fields = set(exclude_fields or [])

    def clean_dict(o):
        if hasattr(o, "__dict__"):
            d = o.__dict__.copy()
            for field in exclude_fields:
                d.pop(field, None)
            return {k: clean_dict(v) for k, v in d.items()}
        elif isinstance(o, list):
            return [clean_dict(i) for i in o]
        elif isinstance(o, dict):
            return {k: clean_dict(v) for k, v in o.items()}
        elif isinstance(o, (int, float, str, bool)) or o is None:
            return o
        else:
            return str(o)  # 👈 fallback for unrecognized types

    cleaned = clean_dict(obj)
    obj_str = json.dumps(cleaned, sort_keys=True)
    return hashlib.sha256(obj_str.encode("utf-8")).hexdigest()


def create_all_combinations(items: List[str]):
    """
    Create all combinations of items of length 2 to len(items)
    """
    from itertools import combinations

    all_combinations = []
    for i in range(2, len(items) + 1):
        all_combinations.extend(combinations(items, i))
    item_lists = [list(combo) for combo in all_combinations]

    # Shuffle the list to ensure randomness
    np.random.shuffle(item_lists)
    return item_lists


def get_names_by_type(type_name: str, item_names: List[str]) -> List[str]:
    """
    Get names of items that start with the given type name.
    """
    if not type_name or not item_names:
        return []

    # Filter item names that start with the type name
    filtered_names = [name for name in item_names if name.startswith(f"{type_name}_")]

    return filtered_names


def get_image_index(filename: str) -> int:
    """
    Extract the image index from a filename.
    Assumes the filename format is like '_2_image.png'.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    parts = filename.split("_")

    # Raw Plan file names in format _4_action.png
    if filename.startswith("_"):
        index_part = parts[1]

    # Plan file names in format of 4_action.png
    else:
        index_part = parts[0]

    try:
        return int(index_part)
    except ValueError:
        return -1  # Return -1 if conversion fails


def set_image_index(filename: str, index: int) -> str:
    """
    Set the image index in a filename.
    Assumes the filename format is like '_2_image.png' -> '{index}_image.png'.
    """
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Split by underscore and replace the first part with the new index
    parts = filename.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename '{filename}' does not follow expected format.")

    new_filename = f"{index}_{parts[2]}"
    return new_filename


def is_error_recovery_plan(plan_name: str) -> bool:
    """
    Check if a plan name indicates an error recovery plan.
    """
    injection_pattern = re.compile(r"[Rr]\d+")
    is_injection = bool(re.search(injection_pattern, plan_name))

    if not is_injection:
        injection_pattern = re.compile(r"[Fr]\d+")
        is_injection = bool(re.search(injection_pattern, plan_name))

    return is_injection


def is_object_inside(obj1, obj2) -> bool:
    """
    Determine if the first object is inside the second object using Object-Oriented Bounding Boxes.

    Criteria:
    - Horizontal (x, z): Center of obj1 must be inside the bounds of obj2
    - Vertical (y): Center of obj1 can be within a threshold of obj2, where
      threshold = max(height of obj1, height of obj2)

    Args:
        obj1: First object with objectOrientedBoundingBox data (cornerPoints)
        obj2: Second object (container) with objectOrientedBoundingBox data (cornerPoints)

    Returns:
        bool: True if obj1 is considered inside obj2, False otherwise
    """
    try:
        # Extract corner points from bounding boxes
        corners1 = obj1["objectOrientedBoundingBox"]["cornerPoints"]
        corners2 = obj2["objectOrientedBoundingBox"]["cornerPoints"]

        # Calculate center and dimensions from corner points
        def get_bbox_info(corners):
            # Convert corner points to numpy arrays for easier calculation
            points = np.array(corners)

            # Calculate center as the mean of all corner points
            center = np.mean(points, axis=0)

            # Calculate dimensions by finding the extents
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            size = max_coords - min_coords

            return center, size

        center1, size1 = get_bbox_info(corners1)
        center2, size2 = get_bbox_info(corners2)

        # Check horizontal containment using 2D projection to XZ plane

        # Create 2D version of obj1's center
        center1_2d = np.array([center1[0], center1[2]])  # X, Z coordinates

        # For 2D containment, use a simpler approach: point in polygon
        def point_in_polygon_2d(point, polygon):
            """Point in polygon test using ray casting algorithm"""
            x, z = point
            n = len(polygon)
            inside = False

            p1x, p1z = polygon[0]
            for i in range(1, n + 1):
                p2x, p2z = polygon[i % n]
                if z > min(p1z, p2z):
                    if z <= max(p1z, p2z):
                        if x <= max(p1x, p2x):
                            if p1z != p2z:
                                xinters = (z - p1z) * (p2x - p1x) / (p2z - p1z) + p1x
                            else:
                                xinters = p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1z = p2x, p2z

            return inside

        # Get unique XZ coordinates to form the 2D footprint
        unique_xz = []
        for corner in corners2:
            xz_point = [corner[0], corner[2]]
            if xz_point not in unique_xz:
                unique_xz.append(xz_point)

        # Sort points to form a proper polygon (convex hull)
        if len(unique_xz) >= 3:
            # Sort by angle from center to form a proper polygon
            center_xz = [center2[0], center2[2]]
            unique_xz.sort(
                key=lambda p: np.arctan2(p[1] - center_xz[1], p[0] - center_xz[0])
            )

            horizontal_inside = point_in_polygon_2d(center1_2d, unique_xz)
        else:
            horizontal_inside = False

        if not horizontal_inside:
            return False

        # Check vertical containment (y axis) with threshold
        # Threshold is the larger of the two objects' heights
        threshold = max(size1[1], size2[1])  # Y is index 1

        # Calculate obj2's vertical bounds with threshold
        obj2_y_min = center2[1] - size2[1] / 2 - threshold
        obj2_y_max = center2[1] + size2[1] / 2 + threshold

        # Check if obj1's center is within the vertical threshold
        y_inside = obj2_y_min <= center1[1] <= obj2_y_max

        return y_inside

    except (KeyError, TypeError, ValueError, IndexError) as e:
        # Handle missing keys or invalid data structure
        print(f"Error in is_object_inside: {e}")
        return False


def test_is_object_inside():
    """
    Test cases for the is_object_inside function
    """
    print("Testing is_object_inside function...")

    # Test Case 1: Simple case - small object clearly inside larger object
    obj1 = {
        "objectOrientedBoundingBox": {
            "cornerPoints": [
                [0.9, 0.9, 0.9],  # Small cube centered at (1,1,1)
                [1.1, 0.9, 0.9],
                [1.1, 0.9, 1.1],
                [0.9, 0.9, 1.1],
                [0.9, 1.1, 0.9],
                [1.1, 1.1, 0.9],
                [1.1, 1.1, 1.1],
                [0.9, 1.1, 1.1],
            ]
        }
    }

    obj2 = {
        "objectOrientedBoundingBox": {
            "cornerPoints": [
                [0.0, 0.0, 0.0],  # Large cube centered at (1,1,1)
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 2.0],
                [0.0, 0.0, 2.0],
                [0.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 2.0, 2.0],
                [0.0, 2.0, 2.0],
            ]
        }
    }

    result1 = is_object_inside(obj1, obj2)
    print(f"Test 1 - Small object inside large object: {result1} (Expected: True)")

    # Test Case 2: Object clearly outside
    obj3 = {
        "objectOrientedBoundingBox": {
            "cornerPoints": [
                [5.9, 0.9, 0.9],  # Small cube at (6,1,1) - outside obj2
                [6.1, 0.9, 0.9],
                [6.1, 0.9, 1.1],
                [5.9, 0.9, 1.1],
                [5.9, 1.1, 0.9],
                [6.1, 1.1, 0.9],
                [6.1, 1.1, 1.1],
                [5.9, 1.1, 1.1],
            ]
        }
    }

    result2 = is_object_inside(obj3, obj2)
    print(f"Test 2 - Object outside container: {result2} (Expected: False)")

    # Test Case 3: Object on the edge (should be considered inside due to threshold)
    obj4 = {
        "objectOrientedBoundingBox": {
            "cornerPoints": [
                [1.9, 1.9, 1.9],  # Small cube at edge of obj2
                [2.1, 1.9, 1.9],
                [2.1, 1.9, 2.1],
                [1.9, 1.9, 2.1],
                [1.9, 2.1, 1.9],
                [2.1, 2.1, 1.9],
                [2.1, 2.1, 2.1],
                [1.9, 2.1, 2.1],
            ]
        }
    }

    result3 = is_object_inside(obj4, obj2)
    print(f"Test 3 - Object on edge: {result3} (Expected: True due to threshold)")

    print("Test complete!\n")


def is_sequence_looping(sequence: list, min_repetitions: int = 3) -> bool:
    """
    Detects if a sequence is looping by finding repeating patterns.

    Args:
        sequence: List of strings to check for looping patterns
        min_repetitions: Minimum number of repetitions required to consider it a loop

    Returns:
        bool: True if a looping pattern is detected, False otherwise

    Examples:
        is_sequence_looping(['b', 'c', 'a', 'a', 'a', 'a']) -> True (pattern 'a' repeats 4 times)
        is_sequence_looping(['b', 'c', 'a', 'd', 'c', 'a', 'd', 'c', 'a', 'd', 'c']) -> True (pattern 'cad' repeats 3 times)
        is_sequence_looping(['g', 'h', 'h', 'g', 'h', 'h', 'g', 'h', 'h']) -> True (pattern 'ghh' repeats 3 times)
    """
    if len(sequence) < min_repetitions:
        return False

    # Check for single element loops (most common case)
    if len(sequence) >= min_repetitions:
        last_element = sequence[-1]
        count = 0
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] == last_element:
                count += 1
            else:
                break
        if count >= min_repetitions:
            return True

    # Check for multi-element patterns
    max_pattern_length = len(sequence) // min_repetitions

    for pattern_length in range(2, max_pattern_length + 1):
        # Extract the potential pattern from the end of the sequence
        if len(sequence) < pattern_length * min_repetitions:
            continue

        pattern = sequence[-pattern_length:]

        # Count how many times this pattern repeats at the end
        repetitions = 0
        for i in range(len(sequence) - pattern_length, -1, -pattern_length):
            if i + pattern_length <= len(sequence):
                current_segment = sequence[i : i + pattern_length]
                if current_segment == pattern:
                    repetitions += 1
                else:
                    break

        if repetitions >= min_repetitions:
            return True

    return False
