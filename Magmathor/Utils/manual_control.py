import random
from datetime import datetime

import numpy as np
import readchar

import Magmathor.utils as Utils
from Magmathor import constants as c
from Magmathor.plan import PlanType
from Magmathor.randomization import Randomization
from Magmathor.scenario import Scenario
from Magmathor.scenes import Scenes
from Magmathor.setup import ObjectSetup
from Magmathor.specifier import Specifier


class ManualControl:
    """
    Class to handle manual control of the environment.
    """

    starttime = datetime.now()
    data_folder = f'Manual/{starttime.strftime("%Y-%m-%d_%H-%M-%S")}'

    def run(self):
        """
        Run the manual control environment.
        """
        old_positions = None

        # Initial scene selection
        scene = self.select_scene()
        randomization = Randomization(seed=random.randint(0, 10000))

        self.scenario = Scenario(
            task="Any",
            scene=scene,
            name="None",
            plan_type=PlanType.MANUAL,
            data_folder=self.data_folder,
            object_setup=ObjectSetup(),
            randomization=randomization,
        )

        while True:
            print(
                "Press arrow keys for movement, or type a command (or 'exit' to quit)"
            )
            print("Type 'room' to change to a different scene")  # Changed from 'switch'

            # Get the first character to check if it's a special key
            key = readchar.readkey()

            # Check for arrow keys
            if key == readchar.key.LEFT:
                action_string = "left"
                obj_name = None
                print(f"Action: {action_string}")
            elif key == readchar.key.RIGHT:
                action_string = "right"
                obj_name = None
                print(f"Action: {action_string}")
            elif key == readchar.key.UP:
                action_string = "forward"
                obj_name = None
                print(f"Action: {action_string}")
            elif key == readchar.key.DOWN:
                action_string = "backward"
                obj_name = None
                print(f"Action: {action_string}")
            # Add Page Up/Down for looking up/down
            elif key == readchar.key.PAGE_UP:
                action_string = "up"
                obj_name = None
                print(f"Action: Looking Up")
            elif key == readchar.key.PAGE_DOWN:
                action_string = "down"
                obj_name = None
                print(f"Action: Looking Down")
            else:
                # For regular typed commands, collect the full input
                if key == readchar.key.ENTER:
                    entry = ""
                    print("> ", end="", flush=True)
                else:
                    entry = key
                    print(f"> {key}", end="", flush=True)  # Display what user is typing

                    # Continue reading until Enter
                    while True:
                        next_char = readchar.readkey()
                        if next_char == readchar.key.ENTER:
                            print()  # Move to next line after Enter is pressed
                            break
                        elif (
                            next_char == readchar.key.BACKSPACE or next_char == "\x7f"
                        ):  # Handle backspace
                            if entry:
                                entry = entry[:-1]
                                # Clear character: move back, print space, move back again
                                print("\b \b", end="", flush=True)
                        elif next_char == readchar.key.TAB:
                            # If there's no space yet, try to autocomplete
                            if " " not in entry:
                                matches = self.get_action_matches(entry)
                                if matches:
                                    # Clear the current input display
                                    for _ in range(len(entry)):
                                        print("\b \b", end="", flush=True)

                                    # Replace with the first match
                                    entry = matches[0]
                                    print(entry, end="", flush=True)
                        else:
                            entry += next_char
                            print(
                                next_char, end="", flush=True
                            )  # Show character as it's typed

                            # Split the entry to handle action and object separately
                            parts = entry.split(maxsplit=1)

                            # Action autocomplete (if we're still typing the action)
                            if len(parts) == 1:
                                matches = self.get_action_matches(entry)
                                if len(matches) == 1 and matches[0] != entry:
                                    # Unique match - autocomplete fully
                                    # Clear the current input display
                                    for _ in range(len(entry)):
                                        print("\b \b", end="", flush=True)

                                    # Replace with the autocompleted action
                                    entry = matches[0]
                                    print(entry, end="", flush=True)
                                elif len(matches) > 1:
                                    # Multiple matches - autocomplete to common prefix
                                    common_prefix = self.get_common_prefix(matches)
                                    if len(common_prefix) > len(entry):
                                        # Clear the current input display
                                        for _ in range(len(entry)):
                                            print("\b \b", end="", flush=True)

                                        # Replace with the common prefix
                                        entry = common_prefix
                                        print(entry, end="", flush=True)

                            # Object autocomplete (if we're typing an object after the action)
                            elif len(parts) == 2:
                                action = parts[0]
                                partial_obj = parts[1]

                                # Only autocomplete if we've typed at least one character for the object
                                if partial_obj:
                                    matches = self.get_object_matches(partial_obj)
                                    if (
                                        len(matches) == 1
                                        and matches[0].lower() != partial_obj.lower()
                                    ):
                                        # Unique match - autocomplete fully
                                        # Clear the current object part display
                                        for _ in range(len(partial_obj)):
                                            print("\b \b", end="", flush=True)

                                        # Replace with the autocompleted object (preserve case)
                                        entry = f"{action} {matches[0]}"
                                        print(matches[0], end="", flush=True)
                                    elif len(matches) > 1:
                                        # Multiple matches - autocomplete to common prefix (case-insensitive)
                                        # Convert to lowercase for finding common prefix
                                        lowercase_matches = [m.lower() for m in matches]
                                        common_prefix = self.get_common_prefix(
                                            lowercase_matches
                                        )

                                        if len(common_prefix) > len(
                                            partial_obj.lower()
                                        ):
                                            # Clear the current object part display
                                            for _ in range(len(partial_obj)):
                                                print("\b \b", end="", flush=True)

                                            # Replace with the common prefix
                                            entry = f"{action} {common_prefix}"
                                            print(common_prefix, end="", flush=True)

                # Handle regular command input
                # Handle special commands
                if entry.lower() == "exit":
                    print("Exiting manual control.")
                    break
                elif entry.lower() == "room":  # Changed from 'switch'
                    print("Switching to a new scene...")
                    self.scenario.complete_manual()  # Complete without saving files
                    # Select a new scene
                    scene = self.select_scene()
                    starttime = datetime.now()
                    self.data_folder = (
                        f'Manual/{starttime.strftime("%Y-%m-%d_%H-%M-%S")}'
                    )
                    # Create a new scenario with the selected scene
                    randomization = Randomization(seed=random.randint(0, 10000))
                    self.scenario = Scenario(
                        task="Any",
                        scene=scene,
                        name="None",
                        plan_type=PlanType.MANUAL,
                        data_folder=self.data_folder,
                        object_setup=ObjectSetup(),
                        randomization=randomization,
                    )
                    continue

                elif entry.lower() == "x":
                    self.build_map()
                    continue
                    new_positions = self.scenario.controller.step(
                        action="GetReachablePositions"
                    ).metadata["actionReturn"]

                    # Track min and max positions
                    min_x = float("inf")
                    max_x = float("-inf")
                    min_z = float("inf")
                    max_z = float("-inf")

                    # Display first few positions
                    print(f"Found {len(new_positions)} reachable positions")
                    for i, pos in enumerate(new_positions):
                        print(f"Position {i}: {pos}")
                        if i > 10:
                            print(
                                f"... (plus {len(new_positions) - 11} more positions)"
                            )
                            break

                    # Process all positions to find boundaries
                    for pos in new_positions:
                        # Update min/max values
                        min_x = min(min_x, pos["x"])
                        max_x = max(max_x, pos["x"])
                        min_z = min(min_z, pos["z"])
                        max_z = max(max_z, pos["z"])

                    # Print boundary information
                    print("\nPosition Boundaries:")
                    print(
                        f"X range: {min_x:.2f} to {max_x:.2f} (width: {max_x - min_x:.2f})"
                    )
                    print(
                        f"Z range: {min_z:.2f} to {max_z:.2f} (depth: {max_z - min_z:.2f})"
                    )

                    # Check reachability of old positions
                    if old_positions is not None:
                        # Check if every old_position is in new_positions
                        for old_pos in old_positions:
                            if old_pos not in new_positions:
                                Utils.print_color(
                                    c.Color.RED,
                                    f"Old position {old_pos} not found in new positions",
                                )
                            else:
                                Utils.print_color(
                                    c.Color.GREEN,
                                    f"Old position {old_pos} is still reachable",
                                )
                    old_positions = new_positions
                    continue
                # Split input into action and object (object is optional)
                parts = entry.strip().split(maxsplit=1)
                if not parts:
                    continue

                action_string = parts[0].lower()
                obj_name = parts[1] if len(parts) > 1 else None

                if obj_name is not None:
                    obj_name = self.convert_sliced_objects(obj_name)
                    print(f"Object name: {obj_name}")

            # Execute the action (same for both arrow keys and typed commands)
            if action_string == "":
                continue

            if obj_name and obj_name not in self.scenario.compatible_recepticles:
                Utils.print_color(
                    c.Color.RED,
                    f"Object '{obj_name}' is not found in the current scene",
                )
                keys = self.scenario.compatible_recepticles.keys()
                Utils.print_color(c.Color.YELLOW, f"Available objects: {list(keys)}")
            elif action_string == "slice_nonudge":
                # Slice without nudging - directly call the controller
                if obj_name:
                    obj = self.scenario.get_obj_by_name(obj_name, must_exist=False)
                    if obj is None:
                        # Try to find by type
                        objs = self.scenario.get_objs_by_types(
                            [obj_name], must_exist=False
                        )
                        if objs:
                            obj = objs[0]
                    if obj:
                        # Check if holding a knife
                        holding_obj = self.scenario.holding_obj()
                        if (
                            holding_obj is None
                            or holding_obj.get("objectType") != "Knife"
                        ):
                            Utils.print_color(
                                c.Color.YELLOW,
                                "Not holding a knife - using forceAction=True",
                            )
                            force = True
                        else:
                            force = False
                        self.scenario.controller.step(
                            action="SliceObject",
                            objectId=obj["objectId"],
                            forceAction=force,
                        )
                        if self.scenario.controller.last_event.metadata[
                            "lastActionSuccess"
                        ]:
                            print(f"Sliced {obj['name']} (without nudge)")
                        else:
                            error = self.scenario.controller.last_event.metadata.get(
                                "errorMessage", ""
                            )
                            Utils.print_color(c.Color.RED, f"Slice failed: {error}")
                    else:
                        print(f"Object '{obj_name}' not found")
                else:
                    print("Usage: slice_nonudge <object>")
            elif action_string == "nudge":
                # Nudge sliced objects apart
                if obj_name:
                    self.nudge_slices(obj_name)
                else:
                    print("Usage: nudge <original_object_type> (e.g., nudge Bread)")
            elif action_string in [action.value for action in c.Action]:
                action = c.Action(action_string)
                # Handle movement actions directly without going through scenario.do()
                if action in [
                    c.Action.MOVE_FORWARD,
                    c.Action.MOVE_BACKWARD,
                    c.Action.TURN_LEFT,
                    c.Action.TURN_RIGHT,
                    c.Action.TURN_UP,
                    c.Action.TURN_DOWN,
                ]:
                    # Map action to controller move action
                    move_map = {
                        c.Action.MOVE_FORWARD: "MoveAhead",
                        c.Action.MOVE_BACKWARD: "MoveBack",
                        c.Action.TURN_LEFT: "RotateLeft",
                        c.Action.TURN_RIGHT: "RotateRight",
                        c.Action.TURN_UP: "LookUp",
                        c.Action.TURN_DOWN: "LookDown",
                    }
                    self.scenario.agent_move_action(move_map[action])
                elif obj_name is not None:
                    specifier = Specifier(types=[obj_name])
                    self.scenario.do(action, specifier, "manual_control")
                else:
                    print(f"Action '{action_string}' requires an object")
            else:
                print(
                    f"Invalid action: {action_string}. Available actions: {[action.value for action in c.Action]}"
                )

    def build_map(self):

        # First I need to get the valid y height of the room
        new_positions = self.scenario.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        y = new_positions[0]["y"]

        # Loop from -4 to 4 in both x and z directions by increments of 0.25
        x_range = np.arange(-4, 4.25, 0.25)
        z_range = np.arange(-4, 4.25, 0.25)
        positions = []
        for x in x_range:
            for z in z_range:
                positions.append({"x": x, "z": z, "y": y})

        # Now try to teleport to each position
        valid_positions = []
        Utils.print_start_overwrite()
        for pos in positions:
            controller = self.scenario.controller
            controller.step(
                action="Teleport",
                position=pos,
                rotation=dict(x=0, y=270, z=0),
                horizon=30,
                standing=True,
            )
            # See if it worked
            event = controller.last_event
            if event.metadata["lastActionSuccess"]:
                valid_positions.append(pos)
                Utils.print_end_overwrite()
                print(f"Valid position: {pos}")
                Utils.print_start_overwrite()
            else:
                if "out of scene bounds" in event.metadata["errorMessage"]:
                    Utils.print_overwrite(
                        c.Color.RED, f"Position {pos} is out of scene bounds, skipping."
                    )
                    continue
                elif "Collided with" in event.metadata["errorMessage"]:
                    Utils.print_overwrite(
                        c.Color.YELLOW,
                        f"Position {pos} is blocked by an object, skipping.",
                    )
                    continue
                elif "too large a" in event.metadata["errorMessage"]:
                    Utils.print_overwrite(
                        c.Color.ORANGE, f"Position {pos} has height issues, skipping."
                    )
                else:
                    print(f"Invalid position: {pos} - {event.metadata['errorMessage']}")

        Utils.print_end_overwrite()

        print(
            f"Found {len(valid_positions)} valid positions out of {len(positions)} total positions."
        )

    # Util to check contants. Not actively used
    """
    def check_constants(self):
        missing = set()
        for scene in Scenes.all:
            self.scenario = Scenario(
                task="Any",
                scene=scene,
                name="None",
                data_folder=self.data_folder,
                randomization=Randomization(seed=random.randint(0, 10000)),
            )

            # Check that evey object type has been defined in contants
            for obj in self.scenario.controller.last_event.metadata["objects"]:
                if obj["objectType"] not in c.COMPATIBLE_RECEPTACLES:
                    missing.add(obj["objectType"])
                    Utils.print_color(
                        c.Color.RED,
                        f"Object type {obj['objectType']} not defined in constants. Please add it.",
                    )

            self.scenario.complete()
            print(missing)

        print("done")
    """

    def convert_sliced_objects(self, object_name):
        keys = self.scenario.compatible_recepticles.keys()
        for key in keys:
            if key.lower() == object_name.lower():
                object_name = key

        all_object_types = self.scenario.all_object_types()
        if (
            "Bread" in object_name
            and "Sliced" not in object_name
            and "BreadSliced" in all_object_types
        ):
            return "BreadSliced"
        if (
            "Egg" in object_name
            and "Eggcracked" not in object_name
            and "EggCracked" in all_object_types
        ):
            return "EggCracked"
        return object_name

    def nudge_slices(self, object_type: str):
        """Nudge slices of the given object type apart.

        Args:
            object_type: The type of the original unsliced object (e.g., "Bread")
        """
        # Find all sliced objects of this type
        sliced_objs = []
        for obj in self.scenario.all_objects():
            if "Sliced" in obj["objectType"] and object_type in obj["objectType"]:
                sliced_objs.append(obj)

        if not sliced_objs:
            Utils.print_color(c.Color.YELLOW, f"No sliced {object_type} objects found")
            return False

        Utils.print_color(
            c.Color.LIGHT_BLUE, f"Found {len(sliced_objs)} sliced objects"
        )
        return self.scenario._push_slices_apart_nudge(sliced_objs)

    def select_scene(self):
        """Select a scene type and then a specific scene from that category."""
        print("Select scene type (press a number key):")
        print("1. Kitchen")
        print("2. Bedroom")
        print("3. Living Room")
        print("4. Bathroom")

        # Get a single keypress for selection
        while True:
            key = readchar.readkey()

            if key == "1":
                scene_list = Scenes.get_kitchens()
                scene_type = "Kitchen"
                break
            elif key == "2":
                scene_list = Scenes.get_bedrooms()
                scene_type = "Bedroom"
                break
            elif key == "3":
                scene_list = Scenes.get_living_rooms()
                scene_type = "Living Room"
                break
            elif key == "4":
                scene_list = Scenes.get_bathrooms()
                scene_type = "Bathroom"
                break
            # Ignore other keypresses

        # Print selection confirmation
        print(f"Selected: {scene_type}")

        # Show available scenes as a range
        scene_nums = sorted([int(s.replace("FloorPlan", "")) for s in scene_list])
        print(f"Available scenes: {scene_nums[0]}-{scene_nums[-1]} (or 'r' for random)")

        # Get scene selection
        while True:
            user_input = input("Scene: ").strip()
            if user_input.lower() == "r":
                scene = random.choice(scene_list)
                print(f"Randomly selected: {scene}")
                break
            else:
                # Try to match the input to a scene number
                target = f"FloorPlan{user_input}"
                if target in scene_list:
                    scene = target
                    print(f"Selected: {scene}")
                    break
                else:
                    print(f"Invalid selection. Enter a scene number or 'r' for random.")

        print(f"Loading {scene_type} scene: {scene}")
        return scene

    # Update get_action_matches to include special commands
    def get_action_matches(self, partial_text):
        """Find all actions that start with the given partial text."""
        # Get regular action values from enum
        actions = [action.value for action in c.Action]

        # Add special commands that should be autocompleted
        special_commands = ["room", "exit", "slice_nonudge", "nudge"]
        all_commands = actions + special_commands

        # Find matches
        matches = [
            action for action in all_commands if action.startswith(partial_text.lower())
        ]
        return matches

    def get_common_prefix(self, strings):
        """Find the longest common prefix of the given strings."""
        if not strings:
            return ""

        shortest = min(strings, key=len)
        for i, char in enumerate(shortest):
            for other in strings:
                if other[i] != char:
                    return shortest[:i]
        return shortest

    # Add a new helper function to match objects
    def get_object_matches(self, partial_text):
        """Find all objects that start with the given partial text."""
        objects = list(self.scenario.compatible_recepticles.keys())
        matches = [
            obj for obj in objects if obj.lower().startswith(partial_text.lower())
        ]
        return matches


manual_control = ManualControl()
manual_control.run()
