#!/usr/bin/env python3
"""
Manual test utility for testing _push_slices_apart functionality.
Teleports agent to sliceable objects in kitchen scenes and tests slice adjustment.

Usage:
    python test_push_slices_apart.py
"""

import numpy as np

import Magmathor.constants as c
from Magmathor.goal import Goal
from Magmathor.plan import PlanType
from Magmathor.scenario import Scenario


def reposition_agent_for_slices(scenario, obj):
    """
    Reposition agent to get better visibility of slices after they're created.
    Moves agent back 1.5m from the object and face it.
    """
    try:
        obj_pos = scenario.get_nppos(obj)
        agent_metadata = scenario.controller.last_event.metadata["agent"]
        agent_pos = np.array(
            [
                agent_metadata["position"]["x"],
                agent_metadata["position"]["y"],
                agent_metadata["position"]["z"],
            ]
        )

        # Calculate direction away from object
        direction = agent_pos - obj_pos
        direction[1] = 0  # Don't change vertical
        dist = np.linalg.norm(direction)

        if dist > 0.01:
            direction = direction / dist
        else:
            # If too close, move back along Z axis
            direction = np.array([0, 0, 1])

        # New position 1.5m away from object
        new_pos = obj_pos + direction * 1.5
        new_pos[1] = agent_pos[1]  # Keep agent at same height

        pose = {
            "position": {
                "x": float(new_pos[0]),
                "y": float(new_pos[1]),
                "z": float(new_pos[2]),
            },
            "rotation": agent_metadata["rotation"],
            "horizon": agent_metadata["cameraHorizon"],
            "standing": agent_metadata["isStanding"],
        }

        scenario.agent_teleport(pose, obj["name"])
        scenario.agent_face_object(obj["name"])
        scenario.pause()
    except Exception as e:
        print(f"    ⚠ Could not reposition agent: {e}")


def test_push_slices_apart():
    """
    Test _push_slices_apart by:
    1. Going through each kitchen scene
    2. Finding sliceable objects (bread, tomato, lettuce, apple, potato)
    3. Teleporting agent in front of each one
    4. Slicing the object
    5. Pausing for user to see the slice positions
    """

    # Kitchen scenes to test
    kitchen_scenes = [
        "FloorPlan1",
        "FloorPlan2",
        "FloorPlan3",
        "FloorPlan4",
        "FloorPlan5",
    ]

    sliceable_types = ["Bread", "Tomato", "Lettuce", "Apple", "Potato"]

    for scene in kitchen_scenes:
        print(f"\n{'='*60}")
        print(f"Testing scene: {scene}")
        print(f"{'='*60}")

        try:
            # Initialize scenario (don't save data for testing)
            scenario = Scenario(
                task="Test slicing",
                scene=scene,
                name=f"test_slice_{scene}",
                plan_type=PlanType.GENERATED,
                data_folder="",  # Empty = don't save
                goal=Goal(),
            )

            # Find all sliceable objects in scene
            for sliceable_type in sliceable_types:
                sliceable_objs = scenario.get_objs_by_types(
                    [sliceable_type], must_exist=False
                )

                if sliceable_objs is None or len(sliceable_objs) == 0:
                    continue

                for obj in sliceable_objs:
                    obj_name = obj["name"]
                    print(f"\n  Testing: {obj_name} ({sliceable_type})")

                    try:
                        # Teleport agent in front of object
                        pose = scenario.get_pose_in_front(obj, 1.0)
                        scenario.agent_teleport(pose, obj_name)

                        # Face the object so we can see it
                        scenario.agent_face_object(obj_name)

                        # Check if it's sliceable
                        if not scenario.is_sliceable(obj_name):
                            print(f"    ✗ Not sliceable")
                            continue

                        if scenario.is_sliced(obj_name):
                            print(f"    ✗ Already sliced")
                            continue

                        # Make sure it's on counter if needed
                        if sliceable_type in c.FOOD_SLICE_ON_COUNTER_TOP:
                            if not scenario.is_on_countertop(obj_name):
                                print(
                                    f"    ℹ Not on counter, picking up and placing on counter"
                                )
                                scenario.agent_pickup(obj_name)
                                scenario.agent_place(obj_name)

                        # Reposition agent to see slices better before slicing
                        reposition_agent_for_slices(scenario, obj)

                        # Slice the object
                        print(f"    → Slicing {obj_name}...")
                        slice_result = scenario.agent_slice(obj_name)

                        if slice_result:
                            print(f"    ✓ Sliced successfully")

                            # Show pause for user to see slice positions
                            input(
                                f"    Press ENTER to continue (examine slice positions)..."
                            )
                        else:
                            print(f"    ✗ Slicing failed")

                    except Exception as e:
                        print(f"    ✗ Error: {str(e)}")
                        continue

            # Clean up
            scenario.controller.stop()

        except Exception as e:
            print(f"  ✗ Scene initialization failed: {str(e)}")
            continue

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_push_slices_apart()
