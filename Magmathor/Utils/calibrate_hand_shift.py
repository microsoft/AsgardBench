"""
Utility for calibrating hand overlay horizontal shifts for different asset types.
Iterates through scenes, picks up objects, and allows user to adjust the hand overlay position.
Uses assetId to differentiate between different geometries of the same object type.
"""

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image

from Magmathor.cache.item_cache import ItemCache
from Magmathor.plan import PlanType
from Magmathor.scenario import Scenario
from Magmathor.scenes import Scenes

HANDSHIFTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "Data", "handshifts.json"
)

# Look angles to calibrate (horizon is 0, negative is looking up, positive is looking down)
# AI2Thor limits: can't look up beyond 30 degrees (-30) or down beyond 60 degrees
LOOK_ANGLES = [-30, -20, -10, 0, 10, 20, 30, 40]

# Sliceable object types that need to be sliced for additional calibration
SLICEABLE_TYPES = {"Apple", "Bread", "Lettuce", "Potato", "Tomato"}


def load_handshifts() -> Dict[str, int]:
    """Load existing hand shifts from file."""
    if os.path.exists(HANDSHIFTS_FILE):
        with open(HANDSHIFTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_handshifts(shifts: Dict[str, int]) -> None:
    """Save hand shifts to file."""
    with open(HANDSHIFTS_FILE, "w", encoding="utf-8") as f:
        json.dump(shifts, f, indent=2, sort_keys=True)


def is_fully_calibrated(handshifts: dict, asset_id: str) -> bool:
    """Check if an asset has been calibrated (all angles are stored together)."""
    if asset_id not in handshifts:
        return False
    # Assets are only saved when all angles are complete
    return True


def compose_hand_overlay_with_shift(
    base_image: Image.Image,
    hand_image: Image.Image,
    shift_x: int,
    shift_y: int = 0,
    transparency: float = 1.0,
) -> Image.Image:
    """
    Compose the hand image on top of the base image with horizontal and vertical shifts.

    Args:
        base_image: The base image to overlay on
        hand_image: The hand image (RGBA)
        shift_x: Horizontal shift in pixels (negative = left, positive = right)
        shift_y: Vertical shift in pixels (negative = up, positive = down)
        transparency: Transparency of the hand overlay (0.0 to 1.0)

    Returns:
        Composited image
    """
    # Resize hand image to match base image size if needed
    hand = hand_image
    if hand.size != base_image.size:
        hand = hand.resize(base_image.size, Image.Resampling.LANCZOS)

    # Convert base image to RGBA if not already
    base_rgba = base_image.convert("RGBA")

    # Create a new transparent image to hold the shifted hand
    shifted_hand = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

    # Calculate paste position with shift
    paste_x = shift_x
    paste_y = shift_y

    # Paste the hand onto the shifted position
    shifted_hand.paste(hand, (paste_x, paste_y))

    # Apply transparency to the hand image's alpha channel
    if transparency < 1.0:
        alpha = shifted_hand.split()[3]
        alpha = alpha.point(lambda p: int(p * transparency))
        shifted_hand.putalpha(alpha)

    # Composite the hand on top of the base image
    composited = Image.alpha_composite(base_rgba, shifted_hand)

    # Convert back to RGB for consistency
    return composited.convert("RGB")


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV format (BGR)."""
    rgb = np.array(pil_image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def get_pickupable_assets_in_scene(scenario: Scenario) -> list:
    """Get list of unique pickupable asset IDs in the current scene."""
    pickupable_assets = set()
    for obj in scenario.all_objects():
        if obj.get("pickupable", False):
            pickupable_assets.add(obj["assetId"])
    return sorted(list(pickupable_assets))


def get_object_by_asset_id(scenario: Scenario, asset_id: str) -> Optional[dict]:
    """Get an object with the specified assetId that can be picked up."""
    for obj in scenario.all_objects():
        if obj["assetId"] == asset_id and obj.get("pickupable", False):
            return obj
    return None


def is_sliceable_object(asset_id: str) -> bool:
    """Check if an asset is a sliceable type (Apple, Bread, Lettuce, Potato, Tomato)."""
    for sliceable_type in SLICEABLE_TYPES:
        if sliceable_type in asset_id:
            return True
    return False


def get_sliced_objects(scenario: Scenario, original_asset_id: str) -> list:
    """Get all sliced objects from the original asset after slicing.

    Note: Sliced objects in AI2Thor don't have an assetId, so we construct one
    from the original asset ID + "_Sliced".
    """
    sliced_objects = []
    for obj in scenario.all_objects():
        # Sliced objects have the original assetId in their name and "Slice" in objectId
        if (
            original_asset_id in obj.get("name", "")
            and "Slice" in obj.get("objectId", "")
            and obj.get("pickupable", False)
        ):
            # Sliced objects don't have assetId, so we construct one
            # Use original_asset_id + "_Sliced" since all slices look similar
            obj["_constructed_asset_id"] = f"{original_asset_id}_Sliced"
            sliced_objects.append(obj)
    # Sort by size (smallest first for easier handling)
    sliced_objects.sort(
        key=lambda o: o.get("axisAlignedBoundingBox", {}).get("size", {}).get("x", 0)
        * o.get("axisAlignedBoundingBox", {}).get("size", {}).get("z", 0)
    )
    return sliced_objects


def calibrate_single_object(
    scenario: Scenario,
    asset_id: str,
    hand_image: Image.Image,
    handshifts: dict,
    calibrated_assets: set,
    session_stats: dict,
) -> tuple:
    """
    Calibrate hand positions for a single object across all angles.

    Args:
        scenario: The Scenario instance
        asset_id: The asset ID to use as the key in handshifts
        hand_image: The hand overlay image
        handshifts: Dictionary of existing hand shifts
        calibrated_assets: Set of already calibrated assets
        session_stats: Dict with 'labeled' and 'skipped' counters

    Returns:
        tuple: (quit_requested, skip_object)
    """
    quit_requested = False
    skip_object = False

    # Create window and position it in the center of the screen
    window_name = "Hand Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    screen_width = 1920
    screen_height = 1080

    # Reset camera to horizon (0) before starting angle iteration
    current_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
    reset_degrees = round(current_horizon, 1)
    if abs(reset_degrees) > 0.1:
        if reset_degrees > 0:
            result = scenario.controller.step(
                action="LookUp", degrees=abs(reset_degrees)
            )
        else:
            result = scenario.controller.step(
                action="LookDown", degrees=abs(reset_degrees)
            )
        if not result.metadata["lastActionSuccess"]:
            print(
                f"    WARNING: Failed to reset camera: {result.metadata.get('errorMessage', 'unknown')}"
            )
        new_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
        print(f"    Reset camera from {current_horizon:.1f}° to {new_horizon:.1f}°")

    # Now move to the first angle (-30)
    first_angle = LOOK_ANGLES[0]
    current_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
    angle_diff = round(first_angle - current_horizon, 1)
    if abs(angle_diff) > 0.1:
        if angle_diff > 0:
            result = scenario.controller.step(
                action="LookDown", degrees=abs(angle_diff)
            )
        else:
            result = scenario.controller.step(action="LookUp", degrees=abs(angle_diff))
        if not result.metadata["lastActionSuccess"]:
            print(
                f"    ERROR: Failed to set initial angle: {result.metadata.get('errorMessage', 'unknown')}"
            )
        new_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
        print(f"    Set initial camera angle to {new_horizon:.1f}°")

    # Collect all angle data in memory before saving
    angle_data = {}

    # Iterate through all look angles
    for angle in LOOK_ANGLES:
        if quit_requested or skip_object:
            break

        print(f"\n    Calibrating {asset_id} at angle {angle}°:")

        # Set the camera to the target angle
        current_horizon = scenario.controller.last_event.metadata["agent"][
            "cameraHorizon"
        ]
        target_horizon = angle
        angle_diff = target_horizon - current_horizon
        # Round to 1 decimal place to avoid floating point precision issues
        angle_diff = round(angle_diff, 1)

        if abs(angle_diff) > 0.1:
            if angle_diff > 0:
                # Look down
                result = scenario.controller.step(
                    action="LookDown", degrees=abs(angle_diff)
                )
            else:
                # Look up
                result = scenario.controller.step(
                    action="LookUp", degrees=abs(angle_diff)
                )

            if not result.metadata["lastActionSuccess"]:
                print(
                    f"    ERROR: Camera move failed: {result.metadata.get('errorMessage', 'unknown')}"
                )

        # Verify the angle changed
        new_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
        if abs(new_horizon - angle) > 1.0:
            print(
                f"    WARNING: Camera angle mismatch! Got {new_horizon:.1f}°, expected {angle}°"
            )
        else:
            print(f"    Camera horizon: {new_horizon:.1f}° (target: {angle}°)")

        # Capture fresh frame directly from the last event
        frame = Image.fromarray(scenario.controller.last_event.frame)

        # Position window
        window_x = (screen_width - frame.size[0]) // 2
        window_y = (screen_height - frame.size[1]) // 2
        cv2.moveWindow(window_name, max(0, window_x), max(0, window_y))

        # Get starting values: prefer existing calibration, then previous angle in session, then 0
        current_shift_x = 0
        current_shift_y = 0

        # First, check if there's existing calibration data for this asset and angle
        if asset_id in handshifts and str(angle) in handshifts[asset_id]:
            current_shift_x = handshifts[asset_id][str(angle)]["x"]
            current_shift_y = handshifts[asset_id][str(angle)]["y"]
            print(
                f"      Loaded existing calibration: X={current_shift_x}px, Y={current_shift_y}px"
            )
        else:
            # Fall back to previous angle's values from current session
            prev_angle_idx = LOOK_ANGLES.index(angle) - 1
            if prev_angle_idx >= 0:
                prev_angle = LOOK_ANGLES[prev_angle_idx]
                if str(prev_angle) in angle_data:
                    current_shift_x = angle_data[str(prev_angle)]["x"]
                    current_shift_y = angle_data[str(prev_angle)]["y"]

        print("    Horizontal: s/l=10, d/k=25, f/j=50, g/h=100 | Vert: r/v=25")
        print("    Keys: 'Enter' = save, 'z' = skip object, 'q' = quit")

        # Interactive calibration loop for this angle
        while True:
            # Compose the overlay with current shift
            composited = compose_hand_overlay_with_shift(
                frame, hand_image, current_shift_x, current_shift_y, transparency=1.0
            )

            # Convert to OpenCV format and display
            cv_image = pil_to_cv2(composited)

            # Add text overlay showing current shift and angle
            text = f"{asset_id} @ {angle}deg - X: {current_shift_x}px, Y: {current_shift_y}px"
            cv2.putText(
                cv_image, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
            )
            cv2.putText(
                cv_image,
                f"Session: {session_stats['labeled']} labeled, {session_stats['skipped']} skipped (prev labeled)",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                cv_image,
                "Horiz: s/l=10, d/k=25, f/j=50, g/h=100 | Vert: r/v=25",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                cv_image,
                "Enter=save, z=skip object, q=quit",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(window_name, cv_image)

            # Wait for key press
            key = cv2.waitKey(0) & 0xFF

            if key == ord("s"):
                current_shift_x -= 10
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("l"):
                current_shift_x += 10
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("d"):
                current_shift_x -= 25
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("k"):
                current_shift_x += 25
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("f"):
                current_shift_x -= 50
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("j"):
                current_shift_x += 50
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("g"):
                current_shift_x -= 100
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("h"):
                current_shift_x += 100
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("r"):
                current_shift_y -= 25
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == ord("v"):
                current_shift_y += 25
                print(f"      Shift X: {current_shift_x}px, Y: {current_shift_y}px")
            elif key == 13 or key == 10:  # Enter key
                # Store this angle in memory and continue to next angle
                angle_data[str(angle)] = {"x": current_shift_x, "y": current_shift_y}
                print(
                    f"      Recorded {asset_id} @ {angle}°: X={current_shift_x}px, Y={current_shift_y}px"
                )
                break
            elif key == ord("z"):
                # Skip this entire object (don't save any data)
                print(f"      Skipped {asset_id} - no data will be saved")
                skip_object = True
                break
            elif key == ord("q"):
                # Quit
                print("      Quit requested")
                quit_requested = True
                break

    # Only save if all angles were captured (not skipped or quit)
    if not skip_object and not quit_requested and len(angle_data) == len(LOOK_ANGLES):
        # Validate asset_id is not empty
        if not asset_id or asset_id.strip() == "":
            print(f"    ERROR: Cannot save with empty asset_id!")
        else:
            handshifts[asset_id] = angle_data
            save_handshifts(handshifts)
            calibrated_assets.add(asset_id)
            session_stats["labeled"] += 1
            print(
                f"    {asset_id} fully calibrated and saved! (Session total: {session_stats['labeled']})"
            )
    elif not skip_object and not quit_requested:
        print(
            f"    WARNING: Only {len(angle_data)}/{len(LOOK_ANGLES)} angles captured for {asset_id}, not saving"
        )

    cv2.destroyAllWindows()

    return quit_requested, skip_object


def scene_has_matching_objects(scene: str, asset_filter: Optional[str]) -> bool:
    """
    Check if a scene contains any objects matching the filter using ItemCache.
    This avoids expensive AI2Thor initialization for scenes that don't have
    any objects of the type we're looking for.

    Note: This only checks if objects exist, not if they're calibrated.
    The actual assetId (used in handshifts.json) is different from the object name
    and can only be determined after loading the scene.
    """
    if not asset_filter:
        # No filter means we need to check all scenes
        return True

    try:
        scene_names = ItemCache.get_scene_names(scene)
    except (KeyError, FileNotFoundError):
        # If cache doesn't have this scene, we need to check it
        return True

    # Check if any object name contains the filter string
    for name in scene_names:
        if asset_filter in name:
            return True

    return False


def calibrate_hand_shifts():
    """Main function to calibrate hand shifts for all asset types, including sliced objects."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate hand overlay horizontal shifts for different asset types."
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default=None,
        help="Filter to assets containing this string (e.g., 'Potato', 'Apple')",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Review mode: show already calibrated assets for review",
    )
    args = parser.parse_args()

    asset_filter = args.filter
    review_mode = args.review

    if asset_filter:
        print(f"Filtering to assets containing: '{asset_filter}'")
    if review_mode:
        print("Review mode: will show already calibrated assets for review")

    # Load existing shifts
    handshifts = load_handshifts()
    print(f"Loaded {len(handshifts)} existing hand shifts from {HANDSHIFTS_FILE}")

    # Load hand image
    hand_path = os.path.join(os.path.dirname(__file__), "..", "Data", "hand.png")
    hand_image = Image.open(hand_path).convert("RGBA")
    print(f"Loaded hand image from {hand_path}")

    # Get all scenes (excluding bedrooms)
    all_scenes = []
    all_scenes.extend(Scenes.get_kitchens())
    all_scenes.extend(Scenes.get_living_rooms())
    all_scenes.extend(Scenes.get_bathrooms())

    print(f"Found {len(all_scenes)} total scenes")

    # Track which assets we've already fully calibrated (all angles)
    calibrated_assets = set(
        a for a in handshifts.keys() if is_fully_calibrated(handshifts, a)
    )
    print(f"Already fully calibrated: {len(calibrated_assets)} assets")

    # Session stats (mutable dict so it can be updated by calibrate_single_object)
    session_stats = {"labeled": 0, "skipped": 0}

    quit_requested = False

    for scene_idx, scene in enumerate(all_scenes):
        if quit_requested:
            break

        # Pre-check using ItemCache to skip scenes that don't have matching objects
        if not scene_has_matching_objects(scene, asset_filter):
            print(
                f"Scene {scene_idx + 1}/{len(all_scenes)}: {scene} - skipping (no matching objects)"
            )
            continue

        print(f"\n{'='*60}")
        print(f"Scene {scene_idx + 1}/{len(all_scenes)}: {scene}")
        print(f"{'='*60}")

        try:
            # Create scenario for this scene
            scenario = Scenario(
                task="Hand calibration",
                scene=scene,
                name="calibration",
                plan_type=PlanType.MANUAL,
                data_folder=None,  # Don't save anything
            )

            # Get pickupable assets in this scene
            pickupable_assets = get_pickupable_assets_in_scene(scenario)

            # Apply command-line filter if provided
            if asset_filter:
                pickupable_assets = [a for a in pickupable_assets if asset_filter in a]

            # For living rooms and bathrooms, only focus on task-relevant items
            if scene.startswith("FloorPlan2") or scene.startswith("FloorPlan4"):
                # Living rooms (2xx) and Bathrooms (4xx) - only remote, cloth, spray bottle
                pickupable_assets = [
                    a
                    for a in pickupable_assets
                    if any(t in a for t in ["Remote", "Cloth", "Spray_Bottle"])
                ]
                if pickupable_assets:
                    print(f"  Filtered to task-relevant assets: {pickupable_assets}")

            # Check for uncalibrated originals
            uncalibrated = [a for a in pickupable_assets if a not in calibrated_assets]

            # Also check for uncalibrated slices of sliceable objects
            uncalibrated_slices = []
            for asset_id in pickupable_assets:
                if is_sliceable_object(asset_id):
                    slice_asset_id = f"{asset_id}_Sliced"
                    if slice_asset_id not in calibrated_assets:
                        uncalibrated_slices.append(slice_asset_id)

            if not review_mode and not uncalibrated and not uncalibrated_slices:
                print("  No uncalibrated assets or slices in this scene, skipping...")
                scenario.controller.stop()
                continue

            print(
                f"Found {len(pickupable_assets)} pickupable assets, {len(uncalibrated)} uncalibrated originals, {len(uncalibrated_slices)} uncalibrated slices"
            )

            for asset_id in pickupable_assets:
                if quit_requested:
                    break

                # Check if original is already fully calibrated
                original_calibrated = is_fully_calibrated(handshifts, asset_id)

                # Skip entirely only if calibrated AND (not sliceable OR slice already calibrated)
                # Unless in review mode, which shows everything
                if not review_mode and original_calibrated:
                    if not is_sliceable_object(asset_id):
                        print(f"  Skipping {asset_id} - already fully calibrated")
                        session_stats["skipped"] += 1
                        continue
                    else:
                        # Check if the slice is also calibrated
                        slice_asset_id = f"{asset_id}_Sliced"
                        if is_fully_calibrated(handshifts, slice_asset_id):
                            print(
                                f"  Skipping {asset_id} - original and slices already calibrated"
                            )
                            session_stats["skipped"] += 1
                            continue

                # Get an object with this asset ID
                obj = get_object_by_asset_id(scenario, asset_id)
                if obj is None:
                    print(f"  Skipping {asset_id} - no object found")
                    continue

                # If original needs calibration (or we're in review mode), do it
                if not original_calibrated or review_mode:
                    print(f"\n  Picking up {asset_id} ({obj['name']})...")

                    # Make sure we're not holding anything
                    if scenario.holding_obj_name() is not None:
                        held_name = scenario.holding_obj_name()
                        print(f"    Currently holding {held_name}, putting it down...")
                        try:
                            scenario.controller.step(
                                action="DropHandObject", forceAction=True
                            )
                        except Exception as e:
                            print(f"    Failed to drop object: {e}")
                            continue

                    # Try to pick up the object
                    try:
                        scenario.controller.step(
                            action="PickupObject",
                            objectId=obj["objectId"],
                            forceAction=True,
                        )
                    except Exception as e:
                        print(f"    Failed to pick up {asset_id}: {e}")
                        continue

                    # Check if we're actually holding the object
                    scenario.controller.step(action="Pass")  # Update state
                    held_obj = scenario.holding_obj_name()
                    if held_obj is None:
                        print(
                            f"    ERROR: Failed to pick up {asset_id} - not holding anything"
                        )
                        continue

                    print(f"    Successfully picked up {held_obj}")

                    # Calibrate the original (unsliced) object
                    quit_requested, skip_object = calibrate_single_object(
                        scenario=scenario,
                        asset_id=asset_id,
                        hand_image=hand_image,
                        handshifts=handshifts,
                        calibrated_assets=calibrated_assets,
                        session_stats=session_stats,
                    )

                    if quit_requested:
                        break

                    # Put down the object before slicing or moving to next
                    if scenario.holding_obj_name() is not None:
                        try:
                            scenario.controller.step(
                                action="DropHandObject", forceAction=True
                            )
                        except Exception as e:
                            print(f"    Warning: Failed to drop object: {e}")

                # If this is a sliceable object, slice it and calibrate the slices
                if not skip_object and is_sliceable_object(asset_id):
                    print(f"\n  Slicing {asset_id} to calibrate slices...")

                    # Need to find the dropped object to slice it
                    scenario.controller.step(action="Pass")  # Update state
                    obj_to_slice = get_object_by_asset_id(scenario, asset_id)

                    if obj_to_slice is None:
                        print(f"    ERROR: Cannot find {asset_id} to slice")
                        continue

                    # Slice the object using forceAction
                    try:
                        result = scenario.controller.step(
                            action="SliceObject",
                            objectId=obj_to_slice["objectId"],
                            forceAction=True,
                        )
                        if not result.metadata["lastActionSuccess"]:
                            print(
                                f"    ERROR: Failed to slice {asset_id}: "
                                f"{result.metadata.get('errorMessage', 'unknown')}"
                            )
                            continue
                        print(f"    Successfully sliced {asset_id}")
                    except Exception as e:
                        print(f"    ERROR: Exception slicing {asset_id}: {e}")
                        continue

                    # Update state and get the sliced objects
                    scenario.controller.step(action="Pass")
                    sliced_objects = get_sliced_objects(scenario, asset_id)

                    if not sliced_objects:
                        print(f"    WARNING: No sliced objects found for {asset_id}")
                        continue

                    # All slices from the same original look similar, so just calibrate one
                    # Use the constructed asset ID (original_asset_id + "_Sliced")
                    slice_asset_id = sliced_objects[0].get("_constructed_asset_id", "")

                    if not slice_asset_id:
                        print(
                            f"    ERROR: No constructed asset ID for slices of {asset_id}"
                        )
                        continue

                    # Skip if already calibrated (unless in review mode)
                    if not review_mode and is_fully_calibrated(
                        handshifts, slice_asset_id
                    ):
                        print(f"    Skipping {slice_asset_id} - already calibrated")
                        session_stats["skipped"] += 1
                        continue

                    print(
                        f"    Found {len(sliced_objects)} slices, calibrating as {slice_asset_id}"
                    )

                    # Just pick up the first slice to calibrate (they all look the same)
                    slice_obj = sliced_objects[0]

                    print(f"\n    Picking up slice: {slice_obj['name']}")

                    # Make sure we're not holding anything
                    if scenario.holding_obj_name() is not None:
                        try:
                            scenario.controller.step(
                                action="DropHandObject", forceAction=True
                            )
                        except Exception as e:
                            print(f"      Failed to drop object: {e}")
                            continue

                    # Pick up the slice
                    try:
                        scenario.controller.step(
                            action="PickupObject",
                            objectId=slice_obj["objectId"],
                            forceAction=True,
                        )
                    except Exception as e:
                        print(f"      Failed to pick up slice: {e}")
                        continue

                    # Check if we're actually holding the object
                    scenario.controller.step(action="Pass")
                    if scenario.holding_obj_name() is None:
                        print(
                            "      ERROR: Failed to pick up slice - not holding anything"
                        )
                        continue

                    print("      Successfully picked up slice")

                    # Calibrate this slice using the constructed asset ID
                    quit_requested, _ = calibrate_single_object(
                        scenario=scenario,
                        asset_id=slice_asset_id,
                        hand_image=hand_image,
                        handshifts=handshifts,
                        calibrated_assets=calibrated_assets,
                        session_stats=session_stats,
                    )

                    # Drop the slice
                    if scenario.holding_obj_name() is not None:
                        try:
                            scenario.controller.step(
                                action="DropHandObject", forceAction=True
                            )
                        except Exception as e:
                            print(f"      Warning: Failed to drop slice: {e}")

            # Clean up the scenario
            scenario.controller.stop()

        except Exception as e:
            print(f"  Error in scene {scene}: {e}")
            import traceback

            traceback.print_exc()
            continue

    cv2.destroyAllWindows()
    print(f"\n{'='*60}")
    print("Calibration complete!")
    print(
        f"Session: {session_stats['labeled']} labeled, {session_stats['skipped']} skipped"
    )
    print(f"Total saved: {len(handshifts)} hand shifts to {HANDSHIFTS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    calibrate_hand_shifts()
