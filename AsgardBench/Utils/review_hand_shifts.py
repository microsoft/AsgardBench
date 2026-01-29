"""
Utility for reviewing hand overlay calibrations.
Goes through scenes, captures base items and sliced items with hand overlay.
For sliced items, captures both the biggest and smallest slice.
Skips files that already exist on disk.
Saves images to local disk for review.
"""

import json
import os
from typing import Dict, Optional, Set

from PIL import Image

from AsgardBench.plan import PlanType
from AsgardBench.scenario import Scenario
from AsgardBench.scenes import Scenes

HANDSHIFTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "Data", "handshifts.json"
)

# Look angles to capture
LOOK_ANGLES = [-30, -20, -10, 0, 10, 20, 30, 40]

# Sliceable object types
SLICEABLE_TYPES = {"Apple", "Bread", "Lettuce", "Potato", "Tomato"}

# Default output folder
DEFAULT_OUTPUT_FOLDER = "Generated/hand_shift_review"


def load_handshifts() -> Dict[str, dict]:
    """Load existing hand shifts from file."""
    if os.path.exists(HANDSHIFTS_FILE):
        with open(HANDSHIFTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def compose_hand_overlay_with_shift(
    base_image: Image.Image,
    hand_image: Image.Image,
    shift_x: int,
    shift_y: int = 0,
    transparency: float = 1.0,
) -> Image.Image:
    """
    Compose the hand image on top of the base image with horizontal and vertical shifts.
    """
    hand = hand_image
    if hand.size != base_image.size:
        hand = hand.resize(base_image.size, Image.Resampling.LANCZOS)

    base_rgba = base_image.convert("RGBA")
    shifted_hand = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    shifted_hand.paste(hand, (shift_x, shift_y))

    if transparency < 1.0:
        alpha = shifted_hand.split()[3]
        alpha = alpha.point(lambda p: int(p * transparency))
        shifted_hand.putalpha(alpha)

    composited = Image.alpha_composite(base_rgba, shifted_hand)
    return composited.convert("RGB")


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
    """Check if an asset is a sliceable type."""
    for sliceable_type in SLICEABLE_TYPES:
        if sliceable_type in asset_id:
            return True
    return False


def get_slice_volume(obj: dict) -> float:
    """Calculate volume from bounding box for sorting slices by size."""
    bbox = obj.get("axisAlignedBoundingBox", {}).get("size", {})
    x = bbox.get("x", 0)
    y = bbox.get("y", 0)
    z = bbox.get("z", 0)
    return x * y * z


def get_sliced_objects_sorted_by_size(
    scenario: Scenario, original_asset_id: str
) -> list:
    """
    Get all sliced objects from the original asset after slicing.
    Returns list sorted by volume (smallest first).
    """
    sliced_objects = []
    for obj in scenario.all_objects():
        if (
            original_asset_id in obj.get("name", "")
            and "Slice" in obj.get("objectId", "")
            and obj.get("pickupable", False)
        ):
            sliced_objects.append(obj)

    # Sort by volume (smallest first)
    sliced_objects.sort(key=get_slice_volume)
    return sliced_objects


def get_sliceable_variants_to_review(handshifts: dict) -> dict:
    """
    For each sliceable type, find the smallest and largest variant numbers.
    Returns dict like: {"Apple": [1, 28], "Bread": [1, 28], ...}
    """
    variants_by_type = {}

    for sliceable_type in SLICEABLE_TYPES:
        # Find all variant numbers for this type (base versions, not sliced)
        variant_nums = set()
        for key in handshifts.keys():
            if key.startswith(f"{sliceable_type}_") and not key.endswith("_Sliced"):
                parts = key.split("_")
                if len(parts) >= 2:
                    try:
                        num = int(parts[1])
                        variant_nums.add(num)
                    except ValueError:
                        pass

        if variant_nums:
            variants_by_type[sliceable_type] = [min(variant_nums), max(variant_nums)]

    return variants_by_type


def all_files_exist(output_folder: str, asset_id: str, suffix: str = "") -> bool:
    """Check if all angle files for an asset already exist."""
    for angle in LOOK_ANGLES:
        if suffix:
            filename = f"{asset_id}_{suffix}_{angle}.png"
        else:
            filename = f"{asset_id}_{angle}.png"
        filepath = os.path.join(output_folder, filename)
        if not os.path.exists(filepath):
            return False
    return True


def reset_camera_to_first_angle(scenario: Scenario):
    """Reset camera to horizon then move to first angle."""
    current_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
    reset_degrees = round(current_horizon, 1)
    if abs(reset_degrees) > 0.1:
        if reset_degrees > 0:
            scenario.controller.step(action="LookUp", degrees=abs(reset_degrees))
        else:
            scenario.controller.step(action="LookDown", degrees=abs(reset_degrees))

    # Move to first angle
    first_angle = LOOK_ANGLES[0]
    current_horizon = scenario.controller.last_event.metadata["agent"]["cameraHorizon"]
    angle_diff = round(first_angle - current_horizon, 1)
    if abs(angle_diff) > 0.1:
        if angle_diff > 0:
            scenario.controller.step(action="LookDown", degrees=abs(angle_diff))
        else:
            scenario.controller.step(action="LookUp", degrees=abs(angle_diff))


def capture_at_angles(
    scenario: Scenario,
    handshift_key: str,
    filename_prefix: str,
    hand_image: Image.Image,
    handshifts: dict,
    output_folder: str,
) -> bool:
    """
    Capture images of currently held object at all angles with hand overlay.

    Args:
        handshift_key: Key to look up in handshifts (e.g., "Apple_1" or "Apple_1_Sliced")
        filename_prefix: Prefix for saved files (e.g., "Apple_1" or "Apple_1_Sliced_small")

    Returns True if successful.
    """
    if handshift_key not in handshifts:
        print(f"    WARNING: No calibration data for {handshift_key}, skipping")
        return False

    reset_camera_to_first_angle(scenario)

    # Capture at each angle
    for angle in LOOK_ANGLES:
        # Check if file already exists
        filename = f"{filename_prefix}_{angle}.png"
        filepath = os.path.join(output_folder, filename)
        if os.path.exists(filepath):
            print(f"    Skipped (exists): {filename}")
            # Still need to move camera for next angle
            current_horizon = scenario.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]
            next_angle_idx = LOOK_ANGLES.index(angle) + 1
            if next_angle_idx < len(LOOK_ANGLES):
                next_angle = LOOK_ANGLES[next_angle_idx]
                angle_diff = round(next_angle - current_horizon, 1)
                if abs(angle_diff) > 0.1:
                    if angle_diff > 0:
                        scenario.controller.step(
                            action="LookDown", degrees=abs(angle_diff)
                        )
                    else:
                        scenario.controller.step(
                            action="LookUp", degrees=abs(angle_diff)
                        )
            continue

        # Set the camera to the target angle
        current_horizon = scenario.controller.last_event.metadata["agent"][
            "cameraHorizon"
        ]
        angle_diff = round(angle - current_horizon, 1)

        if abs(angle_diff) > 0.1:
            if angle_diff > 0:
                scenario.controller.step(action="LookDown", degrees=abs(angle_diff))
            else:
                scenario.controller.step(action="LookUp", degrees=abs(angle_diff))

        # Capture frame
        frame = Image.fromarray(scenario.controller.last_event.frame)

        # Get shift values
        angle_str = str(angle)
        if angle_str in handshifts[handshift_key]:
            shift_x = handshifts[handshift_key][angle_str].get("x", 0)
            shift_y = handshifts[handshift_key][angle_str].get("y", 0)
        else:
            print(f"    WARNING: No data for angle {angle} in {handshift_key}")
            shift_x = 0
            shift_y = 0

        # Compose overlay
        composited = compose_hand_overlay_with_shift(
            frame, hand_image, shift_x, shift_y, transparency=1.0
        )

        # Save image
        composited.save(filepath)
        print(f"    Saved: {filename}")

    return True


def is_sliceable(asset_id: str) -> bool:
    """Check if an asset ID is for a sliceable object type."""
    for sliceable_type in SLICEABLE_TYPES:
        if asset_id.startswith(f"{sliceable_type}_"):
            return True
    return False


def review_hand_shifts():
    """Main function to review hand shifts by capturing images for ALL variants."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Review hand overlay calibrations by capturing images."
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for images (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--filter",
        "-f",
        default=None,
        help="Filter to specific object type (e.g., 'Apple', 'Mug')",
    )
    args = parser.parse_args()

    output_folder = args.output
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")

    # Load hand shifts and hand image
    handshifts = load_handshifts()
    print(f"Loaded {len(handshifts)} hand shifts from {HANDSHIFTS_FILE}")

    hand_path = os.path.join(os.path.dirname(__file__), "..", "Data", "hand.png")
    hand_image = Image.open(hand_path).convert("RGBA")
    print(f"Loaded hand image from {hand_path}")

    # Build set of all calibrated base asset IDs (not ending in _Sliced)
    all_calibrated_assets = set()
    for key in handshifts.keys():
        if not key.endswith("_Sliced"):
            # Extract object type (e.g., "Apple" from "Apple_1")
            obj_type = key.split("_")[0]
            if args.filter and args.filter != obj_type:
                continue
            all_calibrated_assets.add(key)

    print(f"Total calibrated assets to capture: {len(all_calibrated_assets)}")

    # Track which we've already processed (captured or skipped because files exist)
    processed_assets = set()

    # Get all scenes (kitchens, bedrooms, bathrooms - where pickupable items are)
    all_scenes = Scenes.get_kitchens() + Scenes.get_bedrooms() + Scenes.get_bathrooms()
    print(f"Found {len(all_scenes)} scenes")

    for scene_idx, scene in enumerate(all_scenes):
        # Check if we've captured everything
        remaining = all_calibrated_assets - processed_assets
        if not remaining:
            print("\nAll calibrated assets processed!")
            break

        print(f"\n{'='*60}")
        print(f"Scene {scene_idx + 1}/{len(all_scenes)}: {scene}")
        print(f"Remaining: {len(remaining)} assets to process")
        print(f"{'='*60}")

        try:
            scenario = Scenario(
                task="Hand shift review",
                scene=scene,
                name="review",
                plan_type=PlanType.MANUAL,
                data_folder=None,
            )

            # Get pickupable assets in this scene
            pickupable_assets = get_pickupable_assets_in_scene(scenario)

            # Find sliceable assets that we still need to process
            scene_targets = []
            for asset_id in pickupable_assets:
                if asset_id in processed_assets:
                    continue
                if asset_id not in all_calibrated_assets:
                    continue
                scene_targets.append(asset_id)

            if not scene_targets:
                print("  No unprocessed calibrated assets in this scene, skipping...")
                scenario.controller.stop()
                continue

            print(f"  Found {len(scene_targets)} assets to process: {scene_targets}")

            for asset_id in scene_targets:
                if asset_id in processed_assets:
                    continue

                obj = get_object_by_asset_id(scenario, asset_id)
                if obj is None:
                    print(f"  Could not find {asset_id}, skipping")
                    processed_assets.add(
                        asset_id
                    )  # Mark as processed to avoid retrying
                    continue

                # Make sure we're not holding anything
                if scenario.holding_obj_name() is not None:
                    scenario.controller.step(action="DropHandObject", forceAction=True)

                print(f"\n  Processing {asset_id}...")

                # ========== CAPTURE BASE (UNSLICED) ITEM ==========
                base_all_exist = all_files_exist(output_folder, asset_id)
                if base_all_exist:
                    print(f"    Base item: all files exist, skipping")
                else:
                    print(f"    Capturing base item...")
                    scenario.controller.step(
                        action="PickupObject",
                        objectId=obj["objectId"],
                        forceAction=True,
                    )
                    scenario.controller.step(action="Pass")

                    if scenario.holding_obj_name() is None:
                        print(f"    Failed to pick up {asset_id}")
                    else:
                        capture_at_angles(
                            scenario,
                            asset_id,
                            asset_id,
                            hand_image,
                            handshifts,
                            output_folder,
                        )
                        scenario.controller.step(
                            action="DropHandObject", forceAction=True
                        )

                # ========== SLICE AND CAPTURE SLICES (only for sliceable items) ==========
                if is_sliceable(asset_id):
                    sliced_key = f"{asset_id}_Sliced"
                    small_all_exist = all_files_exist(
                        output_folder, sliced_key, "small"
                    )
                    big_all_exist = all_files_exist(output_folder, sliced_key, "big")

                    if small_all_exist and big_all_exist:
                        print(f"    Sliced items: all files exist, skipping")
                    else:
                        # Need to slice - find a fresh object
                        obj_to_slice = get_object_by_asset_id(scenario, asset_id)
                        if obj_to_slice is None:
                            print(f"    Could not find {asset_id} to slice")
                        else:
                            print(f"    Slicing {asset_id}...")
                            result = scenario.controller.step(
                                action="SliceObject",
                                objectId=obj_to_slice["objectId"],
                                forceAction=True,
                            )

                            if not result.metadata["lastActionSuccess"]:
                                print(
                                    f"    Failed to slice: {result.metadata.get('errorMessage', 'unknown')}"
                                )
                            else:
                                scenario.controller.step(action="Pass")
                                sliced_objects = get_sliced_objects_sorted_by_size(
                                    scenario, asset_id
                                )

                                if len(sliced_objects) < 2:
                                    print(
                                        f"    Not enough slices (got {len(sliced_objects)})"
                                    )
                                else:
                                    print(f"    Found {len(sliced_objects)} slices")

                                    smallest_slice = sliced_objects[0]
                                    biggest_slice = sliced_objects[-1]

                                    # Capture smallest slice
                                    if not small_all_exist:
                                        print(f"    Picking up smallest slice...")
                                        scenario.controller.step(
                                            action="PickupObject",
                                            objectId=smallest_slice["objectId"],
                                            forceAction=True,
                                        )
                                        scenario.controller.step(action="Pass")

                                        if scenario.holding_obj_name() is None:
                                            print(
                                                f"    Failed to pick up smallest slice"
                                            )
                                        else:
                                            capture_at_angles(
                                                scenario,
                                                sliced_key,
                                                f"{sliced_key}_small",
                                                hand_image,
                                                handshifts,
                                                output_folder,
                                            )
                                            scenario.controller.step(
                                                action="DropHandObject",
                                                forceAction=True,
                                            )

                                    # Capture biggest slice
                                    if not big_all_exist:
                                        print(f"    Picking up biggest slice...")
                                        scenario.controller.step(
                                            action="PickupObject",
                                            objectId=biggest_slice["objectId"],
                                            forceAction=True,
                                        )
                                        scenario.controller.step(action="Pass")

                                        if scenario.holding_obj_name() is None:
                                            print(
                                                f"    Failed to pick up biggest slice"
                                            )
                                        else:
                                            capture_at_angles(
                                                scenario,
                                                sliced_key,
                                                f"{sliced_key}_big",
                                                hand_image,
                                                handshifts,
                                                output_folder,
                                            )
                                            scenario.controller.step(
                                                action="DropHandObject",
                                                forceAction=True,
                                            )

                # Mark this asset as processed
                processed_assets.add(asset_id)

            scenario.controller.stop()

        except Exception as e:
            print(f"  ERROR in scene {scene}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"Review complete!")
    print(
        f"Processed {len(processed_assets)} of {len(all_calibrated_assets)} calibrated assets"
    )
    print(f"Images saved to: {output_folder}")
    print(f"{'='*60}")

    # Report any assets we couldn't find in any scene
    missing = all_calibrated_assets - processed_assets
    if missing:
        print(f"\nWARNING: Could not find {len(missing)} assets in any scene:")
        for asset_id in sorted(missing):
            print(f"  - {asset_id}")


if __name__ == "__main__":
    review_hand_shifts()
