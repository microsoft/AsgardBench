# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import copy
import json
import os
import shutil
import traceback
from collections import Counter
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64
from PIL import Image

import Magmathor.constants as c
import Magmathor.utils as Utils
from Magmathor.cache.placement_cache import PlacementCache
from Magmathor.cache.put_cache import PutCache
from Magmathor.defective_containers import DefectiveContainers
from Magmathor.goal import ActionGoal, Goal
from Magmathor.jump_candidates import JumpCandidates
from Magmathor.memory import Memory
from Magmathor.objects import (
    AgentCantDo,
    AgentFailure,
    AgentFatal,
    ObjectMetadata,
    Pose,
    StepError,
    StepErrorType,
)
from Magmathor.plan import PlanType, RawPlan
from Magmathor.randomization import (
    INJECTIONTYPE_TO_ACTION,
    InjectionType,
    Randomization,
)
from Magmathor.setup import ObjectSetup, SetupAction
from Magmathor.specifier import Specifier
from Magmathor.step import RawStep, Step

DEBUG_SHOW_PLAN = False
DEBUG_IMAGES = False

MAX_TIME = 120
MAX_PLAN_STEPS = 175


class Scenario:
    def __init__(
        self,
        task: str,
        scene: str,
        name: str,
        plan_type: PlanType,
        data_folder: str,
        hand_transparency: int = 0,
        setup_actions: List[SetupAction] = None,
        object_setup: ObjectSetup = None,
        randomization: Randomization = None,
        goal: Goal = None,
        record_video=False,  # Add recording option
        bad_object_names: List[str] = None,
    ):
        # Set current task for monitoring
        from Magmathor.Utils.count_plans import set_current_task

        set_current_task(name)

        self.data_folder = data_folder
        self.record_video = record_video
        self.save_dir = f"{self.data_folder}/{name}"

        self.image_index = 0
        self.max_reach = 1.5
        self.rotation_speed_horiz = 5
        self.rotation_speed_vert = 5
        self.last_save = ""
        self.last_repeat = False

        self.cur_image = None
        self.cur_find_image = None
        self.memory: dict[str, Memory] = {}

        # Hand overlay transparency (0 = no overlay, 1.0 = fully opaque)
        self.hand_transparency = hand_transparency
        self._hand_image = None  # Cached hand image
        self._handshifts = None  # Cached hand shift data

        # Observations about objects
        self.cur_observations = None
        self.cur_find_observations = None

        # 2D bounding boxes of relevant visible objects
        self.cur_bounding_boxes = None
        self.cur_find_bounding_boxes = None

        # Pose when the action was executed
        self.cur_pose = None
        self.cur_find_pose = None

        # Last error message from action retries
        self.last_retry_error = ""

        # Number of times I've tried to execute the current action
        self.num_tries = 0
        self.candidate_poses = None

        # List of empty hand specifiers, to prevent loops
        self.empty_hand_containers = []

        # Track the last opened container per type for prioritization
        # Maps objectType -> objectName (e.g., {"Drawer": "Drawer_abc123"})
        self.last_opened_container: dict[str, str] = {}

        # When objects are picked up, AI2Thor loses containment relationship with
        # anything in the picked up object so need to keep track of it manually
        self.holding_contents = {}

        self.start_time = datetime.now()

        # Count for how often actions occur
        self.action_counts = {}

        # Am I wanting to inject random states into the scene?
        if randomization is not None and randomization.error_injection is not None:
            self.injection_state = c.InjectionState.PRE_INJECTION
        else:
            self.injection_state = c.InjectionState.POST_INJECTION

        # Video recording attributes
        self.video_frames = []
        self.frame_rate = 10  # Frames per second in final video

        self.log: List[str] = []
        self.plan_errors: List[StepError] = []  # Errors for entire plan
        self.step_error: StepError = None  # Error on current step

        # Used to determine when to terminate model (i.e. max steps reached)
        self.reached_new_subgoal = False

        # Hash for store put_cache so we can re-use counter placements
        self.hash = (
            f"{scene}_{Utils.make_object_hash(randomization, ['error_injection'])}"
        )

        self.raw_plan = RawPlan(
            task_description=task,
            name=name,
            first_step=RawStep(action_desc=task, action=None, obj=None),
            scene=scene,
            setup_actions=setup_actions if setup_actions is not None else [],
            object_setup=object_setup,
            randomization=randomization,
            plan_type=plan_type,
            goal=goal if goal is not None else Goal(),
        )

        self.cur_step = self.raw_plan.first_step

        self.object_metadata = {}
        self.object_attempt = 0

        print("Initializing AI2Thor Controller...")
        self.controller = Controller(
            agentMode="default",
            visibilityDistance=5,  # DEFAULT 1.5,
            scene=scene,
            # step sizes
            gridSize=0.25,
            snapToGrid=False,
            rotateStepDegrees=self.rotation_speed_horiz,
            # image modalities
            renderDepthImage=False,
            fieldOfView=60,
            # Increase timeout significantly for containerized environments
            server_timeout=300,  # 5 minutes
            width=1024,
            height=1024,
            quality="High",
            makeAgentsVisible=False,  # Keep agents invisible for cleaner images
            platform=Linux64,
            renderInstanceSegmentation=True,  # For bounding boxes
        )

        Specifier.get_obj_by_name = self.get_obj_by_name

        if bad_object_names is None:
            bad_object_names = []
        self.remove_bad_objects(bad_object_names)
        self.remove_duplicate_objects()

        self.compatible_recepticles = self.update_compatible_receptacles()

        self.controller.step(action="Pass")

        # If data_folder isn't set, we're not generating plans
        if not self.data_folder:
            return

        Specifier.clear_observations()

        # Check if output directory exists, remove it
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)
        os.makedirs(self.save_dir)
        os.makedirs(f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}", exist_ok=True)
        Utils.print_color(c.Color.GREEN, name)

        np.random.seed(self.raw_plan.randomization.seed)
        self.randomize()
        self.run_object_setup(self.raw_plan)
        self.run_setup_actions(self.raw_plan)

        # Do this at the very end to prevent issues with seed randomization in tests

        # Silverware can be hidden in sink and be hard to see so move to counters
        self.move_silverware_to_counters()

        # PepperShaker looks too similar to SaltShaker and causes confusion
        self.remove_bad_types(["PepperShaker"])

    def remove_bad_types(self, bad_type_names: List[str]):
        """
        Remove all objects of the given types from the scene
        """

        # Convert object types into object names
        bad_object_names = []
        for bad_type in bad_type_names:
            bad_objects = self.get_objs_by_types([bad_type], must_exist=False)
            if bad_objects is not None:
                for bad_obj in bad_objects:
                    bad_object_names.append(bad_obj["name"])

        self.remove_bad_objects(bad_object_names)

    def remove_duplicate_objects(self):
        """
        Some objects in Magmathor are duplicated, causing issues with object selection.
        Remove duplicates, keeping one of each type.
        """
        all_objects = self.all_objects()
        type_counts = Counter([obj["objectType"] for obj in all_objects])

        for obj_type, count in type_counts.items():
            if count > 1 and obj_type in c.ACTIONABLE_OBJECTS["Pickupable"]:
                # Get all objects of this type
                objs_of_type = self.get_objs_by_types([obj_type])
                # Keep the first one, remove the rest
                for obj in objs_of_type[1:]:
                    try:
                        print(f"Removing duplicate {obj['name']} of type {obj_type} ")
                        # Remove the object from the scene
                        self.controller.step(
                            action="DisableObject",
                            objectId=obj["objectId"],
                        )
                        Utils.print_color(
                            c.Color.LIGHT_BLUE,
                            f"Removed duplicate object {obj['name']} of type {obj_type} from scene {self.raw_plan.scene}",
                        )
                    except Exception as e:
                        Utils.print_color(
                            c.Color.RED,
                            f"Failed to remove duplicate object {obj['name']} of type {obj_type} from scene {self.raw_plan.scene}: {e}",
                        )
                        traceback.print_exc()

    def remove_bad_objects(self, bad_object_names: List[str]):
        """
        Some objects in Magmathor are buggy, meaning they fall through the
        surface they are placed on, or can never be manually placed.  Exclude these
        """
        problem_names = ["Cup_344dbbfb"]  # Had been removed from item cache

        problem_names.extend(bad_object_names)

        for problem_name in problem_names:
            problem_obj = self.get_obj_by_name(problem_name, must_exist=False)
            if problem_obj is not None:
                try:
                    print(f"Removing {problem_obj['name']} ")
                    # Remove the object from the scene
                    self.controller.step(
                        action="DisableObject",
                        objectId=problem_obj["objectId"],
                    )
                    Utils.print_color(
                        c.Color.LIGHT_BLUE,
                        f"Removed problem object {problem_name} from scene {self.raw_plan.scene}",
                    )
                except Exception as e:
                    Utils.print_color(
                        c.Color.RED,
                        f"Failed to remove problem object {problem_name} from scene {self.raw_plan.scene}: {e}",
                    )
                    traceback.print_exc()

    def move_silverware_to_counters(self):
        """
        Move silverware to counters so they don't end up in sink where can't be seen
        """
        # Get silverware types from the class definition
        silverware_types = c.CLASS_TO_TYPES.get("Silverware", [])

        # Check if any silverware exists in the scene
        silverware_objects = self.get_objs_by_types(silverware_types, must_exist=False)
        if silverware_objects is None or len(silverware_objects) == 0:
            return  # No silverware to move

        # Create specifier for silverware objects
        obj_specifier = Specifier(types=silverware_types, observe=False)

        # Create specifier for counter-type receptacles (excluding sinks)
        counter_types = [
            "CounterTop",
            "DiningTable",
            "CoffeeTable",
            "Desk",
            "SideTable",
        ]
        cont_specifier = Specifier(types=counter_types, observe=False)

        # Use randomize_objects to move silverware to counters
        try:
            self.randomize_objects(obj_specifier, cont_specifier)
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"Moved silverware to counters in scene {self.raw_plan.scene}",
            )
        except Exception as e:
            Utils.print_color(
                c.Color.YELLOW,
                f"Failed to move silverware to counters: {e}",
            )

    def randomize_objects(
        self, obj_specifier: Specifier, cont_specifier: Specifier = None
    ):
        """
        Randomize the objects
        """
        all_objects = self.all_objects()
        excluded_receptacles = cont_specifier.unspecified_types(all_objects)
        excluded_objects = obj_specifier.unspecified_objects(all_objects)
        excluded_objectids = [obj["objectId"] for obj in excluded_objects]

        # Some receptacles in AI2Thor are problematic, objects fall through them
        bad_containers_names = ["SinkBasin_ae09d571"]

        for problem_name in bad_containers_names:
            problem_obj = self.get_obj_by_name(problem_name, must_exist=False)
            if problem_obj is not None:
                excluded_receptacles.append(problem_obj["objectType"])

        self.controller.step(
            action="InitialRandomSpawn",
            randomSeed=self.raw_plan.randomization.seed,
            forceVisible=True,
            numPlacementAttempts=10,
            placeStationary=True,
            excludedReceptacles=excluded_receptacles,
            excludedObjectIds=excluded_objectids,
        )

        if self.controller.last_event.metadata["errorMessage"]:
            Utils.print_color(
                c.Color.RED,
                f"{self.controller.last_event.metadata['errorMessage']}",
            )
            raise Exception(
                f"Failed to spawn scene {self.raw_plan.scene} with randomization {obj_specifier}"
            )

    def randomize(self):

        # Face agent in a random position from 0 to 360 degrees
        rotation = np.random.randint(0, 360)
        pose = {
            "position": self.controller.last_event.metadata["agent"]["position"],
            "rotation": rotation,
            "standing": self.controller.last_event.metadata["agent"]["isStanding"],
            "horizon": self.controller.last_event.metadata["agent"]["cameraHorizon"],
        }
        self.agent_teleport(pose, "")

    def run_object_setup(self, plan: RawPlan):
        # Make requested object types visible
        if plan.object_setup is None:
            return
        if plan.object_setup.target_setup is not None:
            object_specifier = plan.object_setup.target_setup.object_specifier
            if plan.object_setup.target_setup.container_specifier is None:
                container_types = Scenario.nonclosing_container_types()
                container_specifier = Specifier(types=container_types, observe=False)
            else:
                container_specifier = plan.object_setup.target_setup.container_specifier

            self.randomize_objects(object_specifier, container_specifier)

        # Move distractors around
        for distractor_setup in plan.object_setup.distractor_setups:
            if distractor_setup.container_specifier is None:
                container_types = c.ACTIONABLE_OBJECTS["Receptacle"]
                container_specifier = Specifier(types=container_types, observe=False)
            else:
                container_specifier = distractor_setup.container_specifier
            self.randomize_objects(
                distractor_setup.object_specifier, container_specifier
            )

        # Make sure there is enough space on the counters to put things
        counters = self.get_objs_by_types(
            ["CounterTop", "DiningTable", "CoffeeTable", "Desk"]
        )
        if counters != None:
            all_objects = self.all_objects()
            exclude_types = (
                plan.object_setup.target_setup.object_specifier.get_specified_types(
                    all_objects
                )
            )
            for counter in counters:
                self.make_space(counter["name"], exclude_types)

        # TODO: double check that all target objects are where they should be

    def make_space(self, counter_name, exclude_types):

        while True:
            counter = self.get_obj_by_name(counter_name)
            available_space = self.available_space(counter)

            if available_space >= c.MIN_COUNTER_SPACE:
                return

            # Get all objects on the counter, that aren't exluded
            objects = self.get_container_contents(counter_name, exclude_types)
            if objects is None or len(objects) == 0:
                Utils.print_color(
                    c.Color.YELLOW, f"Counter {counter_name} has no objects to remove."
                )
                return

            # Remove objects that are not pickupable
            objects = [obj for obj in objects if self.is_pickupable(obj["name"])]

            # Nothing I can remove
            if len(objects) == 0:
                return

            # Sort objects by size, largest first
            objects.sort(key=lambda obj: self.size_xz(obj), reverse=True)

            # Remove the largest object
            self.controller.step(
                action="DisableObject",
                objectId=objects[0]["objectId"],
            )

            counter = self.get_obj_by_name(counter_name)
            new_available_space = self.available_space(counter)
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"{counter_name}: Removed object[{objects[0]['name']}] to make space. [{available_space:.4f} -> {new_available_space:.4f}]",
            )

    # Unused Util
    """
    def print_object_sizes(self):
        # Print the sizes of all objects in the scene
        objects = self.all_objects()
        for obj in objects:
            size = self.size_max(obj)
            Utils.print_color(c.Color.YELLOW, f"{obj['name']:<20}  Size: {size:.4f}")
    """

    def run_setup_actions(self, plan: RawPlan):
        if not plan.setup_actions or len(plan.setup_actions) == 0:
            return

        for sa in plan.setup_actions:
            match sa.action:
                case c.Action.DIRTY:
                    self.agent_dirty(sa.object_name, admin_task=True)
                case c.Action.FILL:
                    liquid_type = c.LiquidType(sa.argument)
                    self.agent_fill(sa.object_name, liquid_type, admin_task=True)
                case c.Action.EMPTY_LIQUID:
                    if self.is_filled(sa.object_name):
                        self.agent_empty(sa.object_name, admin_task=True)
                case _:
                    raise Exception(
                        f"Unknown setup action {sa.action} for object {sa.object_name}"
                    )

    def should_inject_error(self, injection_type: InjectionType) -> bool:
        """
        Return true if I should inject an error for this action type.
        """
        if self.injection_state != c.InjectionState.PRE_INJECTION:
            return False

        if self.raw_plan.randomization.error_injection is not None:
            if (
                self.raw_plan.randomization.error_injection.injection_type
                == injection_type
            ):
                action_type = INJECTIONTYPE_TO_ACTION[injection_type]

                # If being injected at the very beginning of the plan
                if action_type.value not in self.action_counts:
                    self.action_counts[action_type.value] = 0

                # Otherwise check if the number of time action has been taken to determine whether to inject
                if (
                    self.action_counts[action_type.value]
                    == self.raw_plan.randomization.error_injection.action_index
                ):
                    self.injection_state = c.InjectionState.INJECTING
                    return True

        return False

    def filter_out_observed_objects(self, objects: List[str]) -> List[str]:
        """
        Filter out objects that have been observed in the current scene.
        """
        observed_types = Specifier.get_observed_types(self.all_objects())

        # Filter out objects that are relevant to the current plan, to avoid unsensible actions
        unobserved_objects = [
            obj for obj in objects if obj["objectType"] not in observed_types
        ]

        return unobserved_objects

    def random_holdable_names(
        self,
    ):
        """
        Return list of unobserved holdable object names
        """
        objects = self.all_objects()

        # Filter out observed object type
        objects = self.filter_out_observed_objects(objects)

        # Filter out non-pickupable objects
        pickupable_objects = [obj for obj in objects if self.is_pickupable(obj["name"])]

        if len(pickupable_objects) == 0:
            raise AgentFatal("No pickupable objects found.")

        # Filter out avoided object types
        avoid_types = ["Footstool", "Laptop"]
        avoid_types.extend(self.raw_plan.randomization.error_injection.avoid_types)
        objects = []
        for obj in pickupable_objects:
            if obj["objectType"] not in avoid_types:
                objects.append(obj)

        # Convert objects to names
        holdable_names = [obj["name"] for obj in objects]

        # Put in random order
        np.random.shuffle(holdable_names)
        return holdable_names

    def random_openable_names(
        self,
        exclude_observed_types: bool = True,
    ):
        """
        Return list of openable unobserved object names
        """
        objects = self.all_objects()

        # Filter out observed object types
        objects = self.filter_out_observed_objects(objects)

        # Filter out non-pickupable objects
        openable_objects = [obj for obj in objects if self.is_openable(obj["name"])]

        # Filter out exclude types (things I don't want to open)
        exclute_types = ["Kettle", "Blinds"]
        openable_objects = [
            obj for obj in openable_objects if obj["objectType"] not in exclute_types
        ]

        if len(openable_objects) == 0:
            raise AgentFatal("No openable objects found.")

        # Convert objects to names
        openable_names = [obj["name"] for obj in openable_objects]

        return openable_names

    def update_compatible_receptacles(self):
        """
        Limit to object that are actually in the scene
        """
        all_object_types = self.all_object_types()
        compatible_recepticles = {}
        for obj_type, rep_types in c.COMPATIBLE_RECEPTACLES.items():
            if obj_type not in all_object_types:
                continue
            found_rep_types = []
            for rep_type in rep_types:
                if rep_type in all_object_types:
                    found_rep_types.append(rep_type)
            compatible_recepticles[obj_type] = found_rep_types

        # Add objects that might be created
        if "Egg" in compatible_recepticles:
            compatible_recepticles["EggCracked"] = c.COMPATIBLE_RECEPTACLES[
                "EggCracked"
            ]
        if "Bread" in compatible_recepticles:
            compatible_recepticles["BreadSliced"] = c.COMPATIBLE_RECEPTACLES[
                "BreadSliced"
            ]
        if "Tomato" in compatible_recepticles:
            compatible_recepticles["TomatoSliced"] = c.COMPATIBLE_RECEPTACLES[
                "TomatoSliced"
            ]
        if "Lettuce" in compatible_recepticles:
            compatible_recepticles["LettuceSliced"] = c.COMPATIBLE_RECEPTACLES[
                "LettuceSliced"
            ]
        if "Apple" in compatible_recepticles:
            compatible_recepticles["AppleSliced"] = c.COMPATIBLE_RECEPTACLES[
                "AppleSliced"
            ]
        if "Potato" in compatible_recepticles:
            compatible_recepticles["PotatoSliced"] = c.COMPATIBLE_RECEPTACLES[
                "PotatoSliced"
            ]

        return compatible_recepticles

    def all_object_types(self, filters: List[c.FilterType] = None) -> List[str]:
        """
        Get all object types in the current scene
        """
        objects = self.all_objects()
        object_types = [obj["objectType"] for obj in objects]

        # Remove duplicates
        object_types = list(set(object_types))

        # Filter out fridge objects if requested
        if filters is not None:
            object_types = [
                obj_type for obj_type in object_types if obj_type not in filters
            ]

        return object_types

    def is_action_object(self, object_name: str) -> bool:
        """
        Is object one of my target object for the current scenario?
        """
        object_type = self.get_obj_by_name(object_name)
        target_types = self.raw_plan.object_setup.get_target_types(self.all_objects())
        if object_type in target_types:
            return True
        return False

    def filter_objects(self, objects, filters: List[c.FilterType]):
        """
        Filter the list of objects based on the given filter
        """
        if objects is None:
            return None

        if len(filters) == 0:
            return objects

        if c.FilterType.NOT_IN_FRIDGE in filters:
            filtered_objects = []
            for obj in objects:
                recepticles = obj["parentReceptacles"]
                if recepticles is None or "Fridge" not in recepticles[0]:
                    filtered_objects.append(obj)

        if c.FilterType.NOT_USED_UP in filters:
            filtered_objects = []
            for obj in objects:
                if not self.is_used_up(obj["name"]):
                    filtered_objects.append(obj)

        # Filter objects that already satisfy current goal
        if c.FilterType.GOAL_CONFLICTS in filters:
            filtered_objects = self.filter_goal_conflicts(objects)

        if c.FilterType.DEFECTIVE_CONTAINERS in filters:
            filtered_objects = self.filter_defective_containers(objects)

        return filtered_objects

    def filter_goal_conflicts(self, objects):
        """
        Filter out objects that already satisfy the current goal.
        """
        if objects is None:
            return None

        filtered_objects = []
        for obj in objects:
            # If object is not currently satisfying the goal, keep it
            if not self.is_used_for_goal(obj):
                filtered_objects.append(obj)
            else:
                Utils.print_color(
                    c.Color.LIGHT_BLUE,
                    f"      Filtered out {obj['name']} because it already satisfies the current goal.",
                )
        return filtered_objects

    def filter_defective_containers(self, objects):
        """
        Filter out defective containers for the held object.
        """
        if objects is None:
            return None

        holding_name = self.holding_obj_name()

        if (holding_name is None) or (holding_name == ""):
            return objects

        # TODO: move to filter_objects
        orig_num_objs = len(objects)

        filtered_objecta = []
        for obj in objects:
            obj_short = Utils.short_name(obj["name"])
            holding_short = Utils.short_name(holding_name)
            if not DefectiveContainers.is_defective(
                self.raw_plan.scene, obj_short, holding_short
            ):
                filtered_objecta.append(obj)

        if orig_num_objs > len(filtered_objecta):
            Utils.print_color(
                c.Color.YELLOW,
                f"Filtered out {orig_num_objs - len(filtered_objecta)} defective containers for {holding_name} in scene {self.raw_plan.scene}",
            )
        return filtered_objecta

    def filter_object_names(self, object_names: List[str], filters: List[c.FilterType]):
        # Get objects from names
        objects = [
            self.get_obj_by_name(name)
            for name in object_names
            if self.get_obj_by_name(name) is not None
        ]
        filtered_objects = self.filter_objects(objects, filters)
        # Convert objects back to names
        filtered_names = [obj["name"] for obj in filtered_objects]
        return filtered_names

    def all_objects(self):
        return self.controller.last_event.metadata["objects"]

    def get_existing_types_from_classes(
        self, object_classes: List[str], filters: List[c.FilterType] = None
    ):
        """
        Given list of object_classes, return list of object
        types that exist in this scene
        """
        object_types = []
        for obj_class in object_classes:
            if obj_class in c.CLASS_TO_TYPES:
                for obj_type in c.CLASS_TO_TYPES[obj_class]:
                    new_objects = self.get_objs_by_types(
                        [obj_type], filters=filters, must_exist=False
                    )
                    if new_objects is not None and len(new_objects) > 0:
                        object_types.append(obj_type)
        return object_types

    def get_existing_types(
        self, object_types: List[str], filters: List[c.FilterType] = None
    ):
        """
        Given list of object_classes, return list of object
        types that exist in this scene
        """
        found_types = []
        for obj_type in object_types:
            new_objects = self.get_objs_by_types(
                [obj_type], filters=filters, must_exist=False
            )
            if new_objects is not None and len(new_objects) > 0:
                found_types.append(obj_type)
        return found_types

    def is_compatible(self, object_name, receptacle_name):
        """
        Check if the object can be placed in the receptacle
        """
        obj = self.get_obj_by_name(object_name)
        receptacle = self.get_obj_by_name(receptacle_name)

        if obj is None or receptacle is None:
            return False

        # Check if the object type is compatible with the receptacle type
        if receptacle["objectType"] in self.compatible_recepticles.get(
            obj["objectType"], []
        ):
            return True

        return False

    def is_fillable(self, container_name):
        not_fillable = [
            "CounterTop",
            "CoffeeTable",
            "DiningTable",
            "Fridge",
            "Floor",
            "SideTable",
            "Shelf",
            "Desk",
            "Sofa",
        ]
        for name in not_fillable:
            if name in container_name:
                return False
        return True

    def is_used_up(self, object_name):
        """
        Check if the object is used up
        """
        obj = self.get_obj_by_name(object_name)
        if obj["isUsedUp"] is True:
            return True

        # AI2Thor doesn't always set isUsedUp correctly, so check assetId name
        if "Used_Up" in obj["assetId"]:
            return True

        return False

    def is_empty(self, container_name):
        """
        Check if the container is empty
        """
        obj = self.get_obj_by_name(container_name)
        if obj["receptacleObjectIds"] is None or len(obj["receptacleObjectIds"]) == 0:
            return True

        return False

    def is_dirty(self, container_name):
        obj = self.get_obj_by_name(container_name)
        if obj["isDirty"] is True:
            return True
        return False

    def is_filled(self, container_name):
        obj = self.get_obj_by_name(container_name)
        if obj["isFilledWithLiquid"] is True:
            return True
        return False

    def get_object_screen_center(self, obj):
        try:
            ulx, uly, lrx, lry = self.controller.last_event.instance_detections2D[
                obj["objectId"]
            ]
            xcenter = (ulx + lrx) / 2
            ycenter = (uly + lry) / 2
            return (xcenter / c.IMAGE_WIDTH, ycenter / c.IMAGE_WIDTH)
        except KeyError:
            # TODO: sometimes blocking object isn't on the screen.
            return 0, 0

    def are_slices(self, first_obj, second_obj):
        """
        Check if the two objects are slices of each other
        """
        if first_obj["name"].startswith("Bread") and second_obj["name"].startswith(
            "Bread"
        ):
            return True
        if first_obj["name"].startswith("Tomato") and second_obj["name"].startswith(
            "Tomato"
        ):
            return True
        if first_obj["name"].startswith("Lettuce") and second_obj["name"].startswith(
            "Lettuce"
        ):
            return True
        if first_obj["name"].startswith("Apple") and second_obj["name"].startswith(
            "Apple"
        ):
            return True
        if first_obj["name"].startswith("Potato") and second_obj["name"].startswith(
            "Potato"
        ):
            return True
        return False

    def is_item_blocking(self, object_name, holding_name):
        """
        Check if the object is blocking the container
        """
        obj = self.get_obj_by_name(object_name)
        # Get x/y position of the object on the screen
        x, y = self.get_object_screen_center(obj)

        query = self.controller.step(
            action="GetObjectInFrame", x=x, y=y, checkVisible=False
        )
        object_id = query.metadata["actionReturn"]
        if object_id is None:
            return None

        if object_id != obj["objectId"]:
            block_obj = self.get_obj_by_object_id(object_id)

            # Can't be blocked by item I'm holding
            if block_obj["name"] == holding_name:
                return None

            if not self.is_pickupable(block_obj["name"]):
                # If the blocking object is not pickupable, then it's not blocking
                return None

            # If part of same sliced object, don't block
            if self.are_slices(obj, block_obj):
                return None

            # If blocking object is inside the obj, it's ok (i.e. egg in pan)
            if (
                block_obj["parentReceptacles"] is not None
                and obj["objectId"] in block_obj["parentReceptacles"]
            ):
                return None

            # Ai2Thor sometimes loses containment (egg in pan) so ignore objet that I'm interacting with
            if block_obj["name"] in Specifier.observed_names:
                return None

            return block_obj
        return None

    def liquid_contents(self, container_name):
        if not self.is_filled(container_name):
            return None
        obj = self.get_obj_by_name(container_name)
        if obj["fillLiquid"] is not None:
            return obj["fillLiquid"]

    def is_pickupable(self, object_name):
        obj = self.get_obj_by_name(object_name)
        pickupable = obj["pickupable"]
        return pickupable

    def need_to_empty(self, container_name, ignore_object=None):
        # If container is full, need to empty it first
        if self.is_fillable(container_name):
            objs = self.get_container_contents(container_name, ignore_object)
            if objs is not None and len(objs) > 0:
                return True
        return False

    def calculate_standing(self, object_name):
        """
        Check if the object is too low to reach without crouching
        """
        obj = self.get_obj_by_name(object_name)

        # Always stand for items in Bathtubs as rim is too high
        if self.is_in_container(object_name):
            container_obj = self.get_non_surface_container(object_name)
            if "Bathtub" in container_obj["objectId"]:
                return True
            if "Drawer" in container_obj["objectId"]:
                return True

        # Always stand when looking in bathtub
        if obj["objectType"] == "BathtubBasin":
            return True

        object_height = obj["axisAlignedBoundingBox"]["size"]["y"]
        # Use boudnigbox rather than obj["position"]["y"], as it can be wrong
        object_pos = obj["axisAlignedBoundingBox"]["center"]["y"]

        if obj["objectType"] in c.OBJECT_STANDING_THRESHOLD:
            standing_threshold = c.OBJECT_STANDING_THRESHOLD[obj["objectType"]]
        else:
            standing_threshold = c.DEFAULT_STANDING_THRESHOLD

        # Only if the object is small and low, need to crouch
        if object_height < 0.5 and object_pos < standing_threshold:
            return False
        return True

    def is_crouched(self):
        """
        Check if the agent is crouched
        """
        agent = self.get_agent()
        return not agent["isStanding"]

    def is_sliced(self, object_type):
        obj = self.get_obj_by_name(object_type)
        if obj is None:
            raise ValueError("Cannot find sliced object of type:", object_type)

        return obj["isSliced"]

    def is_sliceable(self, object_type):
        obj = self.get_obj_by_name(object_type)
        if obj is None:
            raise ValueError("Cannot find sliced object of type:", object_type)

        return obj["sliceable"]

    def is_closed(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["openable"] and not obj["isOpen"]:
            return True
        return False

    def is_open(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["openable"] and obj["isOpen"]:
            return True
        return False

    def is_in_container(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return False
        if self.is_on_countertop(object_name):
            return False
        if self.is_on_stove(object_name):
            return False
        if self.is_on_floor(object_name):
            return False
        if self.is_on_toilet(object_name):
            return False
        if Utils.short_name(object_name) in [
            "Fridge",
            "Cabinet",
            "Drawer",
            "Microwave",
            "CounterTop",
            "SinkBasin",
        ]:
            return False
        return True

    def get_closed_drawer_or_cabinet(self, object_name):
        """Return the closed drawer or cabinet containing this object, or None."""
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return None

        for parent_id in obj["parentReceptacles"]:
            # Parent receptacles use pipe format like "Drawer|-01.56|+00.66|-00.20"
            parent_type = parent_id.split("|")[0]
            if parent_type in ["Drawer", "Cabinet"]:
                # Find parent object by objectId
                parent_obj = None
                for o in self.all_objects():
                    if o["objectId"] == parent_id:
                        parent_obj = o
                        break
                if parent_obj and parent_obj["openable"] and not parent_obj["isOpen"]:
                    return parent_obj["name"]

        return None

    def is_on_floor(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return True

        # If just floor
        if len(obj["parentReceptacles"]) == 1:
            parent = obj["parentReceptacles"][0]
            if "Floor" in parent:
                return True

        return False

    def is_on_toilet(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return True

        # If just toilet
        if len(obj["parentReceptacles"]) == 1:
            parent = obj["parentReceptacles"][0]
            if "Toilet" in parent:
                return True

        return False

    def is_on_stove(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return False

        # If just stove
        if len(obj["parentReceptacles"]) == 1:
            parent = obj["parentReceptacles"][0]
            if "Stove" in parent:
                return True

        return False

    def get_burner_under_pan(self) -> str | None:
        """
        Find the pan object and return the name of the StoveBurner containing it.
        Returns None if no pan is found or pan is not on a burner.
        If multiple burners overlap, uses geometry to find the closest one.
        """
        for obj in self.all_objects():
            if obj["objectType"] == "Pan":
                pan_name = obj["name"]
                parent_receptacles = obj.get("parentReceptacles")
                if parent_receptacles:
                    burner_names = []
                    for parent_id in parent_receptacles:
                        parent_obj = self.get_obj_by_object_id(parent_id)
                        if parent_obj and parent_obj["objectType"] == "StoveBurner":
                            burner_names.append(parent_obj["name"])
                    if len(burner_names) == 1:
                        return burner_names[0]
                    elif len(burner_names) > 1:
                        # Pan overlaps multiple burners, pick the closest one
                        burner_names = self.sort_objects_by_distance(
                            burner_names, pan_name
                        )
                        return burner_names[0]
        return None

    def in_type_name(self, object_name, type_name):
        """
        Check if the object is in a container of the given type name.
        Returns the container name if it is of the given type, otherwise None."""
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return None

        parent_names = []
        if len(obj["parentReceptacles"]) > 0:
            for objId in obj["parentReceptacles"]:
                parent_obj = self.get_obj_by_object_id(objId)
                if parent_obj["objectType"] == type_name:
                    parent_names.append(parent_obj["name"])

        if len(parent_names) == 0:
            return None

        # Pan can be on two burners, so sort by distance to objects so get the
        # one that it is most on top of
        if len(parent_names) > 0:
            parent_names = self.sort_objects_by_distance(parent_names, object_name)

        return parent_names[0]

    def is_on_countertop(self, object_name):
        obj = self.get_obj_by_name(object_name)
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return False

        # IF only suface container, then on countertop
        objects = self.filter_surface_containers(obj["parentReceptacles"], ["Shelf"])
        if len(objects) == 0:
            return True

        return False

    def is_object_home(self, object_name):
        """
        Returns true if the object already in a home location
        """

        # If object is not in a container, it is not home
        if not self.is_in_container(object_name):
            return False

        obj = self.get_obj_by_name(object_name)
        if obj["objectType"] in c.PREFERED_OBJECT_HOMES:
            container = self.get_non_surface_container(object_name)
            if container["objectType"] in c.PREFERED_OBJECT_HOMES[obj["objectType"]]:
                # If the object is in a home location, return true
                return True

        # TODO: add secondary positions?
        return False

    def smallest_containing_object(self, obj):
        """
        Get the smallest containing object for the given object name.
        If the object is not in a container, return None.
        Will give me frying pan ahead of table
        """
        if obj["parentReceptacles"] is None or len(obj["parentReceptacles"]) == 0:
            return None

        # Get all containing objects
        containing_objects = []
        for container_id in obj["parentReceptacles"]:
            container_obj = self.get_obj_by_object_id(container_id)
            if container_obj is not None:
                containing_objects.append(container_obj)

        # Sort by size
        containing_objects.sort(key=self.size_xz)

        return containing_objects[0] if containing_objects else None

    def object_type_exists(self, object_types):
        objs = self.get_objs_by_types(object_types, must_exist=False)
        if objs is None or len(objs) == 0:
            return False
        return True

    @classmethod
    def nonclosing_container_types(cls):
        container_types = c.ACTIONABLE_OBJECTS["Receptacle"]
        openable_types = c.ACTIONABLE_OBJECTS["Openable"]

        # Remove it doesn't have a receptacle type or is closable receptacle type
        container_types = [t for t in container_types if t not in openable_types]
        return container_types

    def get_empty_hand_specifier(self, obj, blocked_obj) -> Specifier:

        obj_type = obj["objectType"]

        # If I'm emptying hand, only put cooked egg back in Pan even if can go other place
        if obj_type == "EggCracked":
            return Specifier(types=["Pan"])

        home_types: List[str] = c.COMPATIBLE_RECEPTACLES[obj_type]

        # Remove items that are closable, so I keep it visible
        openable_types = c.ACTIONABLE_OBJECTS["Openable"]
        home_types = [t for t in home_types if t not in openable_types]

        # If there is a SinkBasin, remove the Sink
        if "SinkBasin" in home_types and "Sink" in home_types:
            home_types.remove("Sink")

        type_specifier = Specifier(types=home_types)

        all_objects = self.all_objects()
        objects = type_specifier.get_objs_from_specified_types(all_objects)

        # Only keep obects that have space and aren't inside another container
        available_names = []
        for test_obj in objects:
            test_name = test_obj["name"]
            if (
                self.need_to_empty(test_name) is False
                and self.is_in_container(test_name) is False
                and test_name != blocked_obj
            ):
                available_names.append(test_obj["name"])

        # Filter out names in self.empty_hand_containers to prevent loops
        available_names = [
            name for name in available_names if name not in self.empty_hand_containers
        ]

        if len(available_names) == 0:
            raise AgentFailure(
                f"No empty hand receptacles found for {obj['name']} of type {obj['objectType']}"
            )

        # Shuffle the order of the available names
        np.random.shuffle(available_names)

        return Specifier(names=available_names)

    def get_object_homes(self, obj) -> Specifier:
        """
        Get the preferred home for an object.
        """

        # Get list of containers that the object has failed to be placed in in the past
        failed_containers = PlacementCache.get_failure_names(
            self.raw_plan.scene, obj["name"]
        )

        preferred_types = None

        # If I'm using an item for my task, prefer non-closing containers rather than the fridge
        if obj["objectType"] in self.raw_plan.object_setup.employed_types:
            preferred_types = c.DEFAULT_STARTING_PLACES

        # If I have a perferred home for the object and it exists, use that
        elif obj["objectType"] in c.PREFERED_OBJECT_HOMES:
            preferred_types = c.PREFERED_OBJECT_HOMES[obj["objectType"]]
            # If none of that type exists in world, remove
            if not self.object_type_exists(preferred_types):
                # If the object is in a home location, return true
                preferred_types = None

        # Convert to names removing any that have failed placements
        preferred_names = []
        if preferred_types is not None:
            container_objects = self.get_objs_by_types(
                preferred_types, must_exist=False
            )
            if container_objects is not None:
                for container_obj in container_objects:
                    if container_obj["name"] not in failed_containers:
                        preferred_names.append(container_obj["name"])

        secondary_types = None
        if obj["objectType"] in c.SECONDARY_OBJECT_HOMES:
            secondary_types = c.SECONDARY_OBJECT_HOMES[obj["objectType"]]
            # If none of that type exists in world, remove
            if not self.object_type_exists(secondary_types):
                # If the object is in a home location, return true
                secondary_types = None

        # Convert to names removing any that have failed placements
        secondary_names = []
        if secondary_types is not None:
            container_objects = self.get_objs_by_types(
                secondary_types, must_exist=False
            )
            if container_objects is not None:
                for container_object in container_objects:
                    if container_object["name"] not in failed_containers:
                        secondary_names.append(container_object["name"])

        # If I have a preferred home
        if len(preferred_names) > 0:
            return Specifier(names=preferred_names, secondary_names=secondary_names)

        # Otherwise use the secondary as preferred if it exists
        if len(secondary_names) > 0:
            return Specifier(names=secondary_names)

        # Otherwise If I know where the object came from orginally, use that
        if obj["name"] in self.object_metadata:
            # TODO: get this working to return to home location
            if self.object_metadata[obj["name"]].start_location is not None:
                return Specifier(
                    names=[self.object_metadata[obj["name"]].start_location]
                )

        # Otherwise, use the default TODO: check that at least one exists
        return Specifier(types=c.COMPATIBLE_RECEPTACLES[obj["objectType"]])

    # LARS TODO should be 2d distance
    def distance_to_object(self, object_name):
        agent, obj = self.get_agent_and_object(object_name)
        agent_pos = self.get_nppos(agent)
        obj_pos = self.get_nppos(obj)
        distance = np.linalg.norm(agent_pos - obj_pos)
        return distance

    def distance_between_objects(self, object_name1, object_name2):
        obj1 = self.get_obj_by_name(object_name1)
        obj2 = self.get_obj_by_name(object_name2)
        if obj1 is None or obj2 is None:
            raise Exception(
                f"One of the objects {object_name1} or {object_name2} does not exist in the scene."
            )
        obj1_pos = self.get_nppos(obj1)
        obj2_pos = self.get_nppos(obj2)
        distance = np.linalg.norm(obj1_pos - obj2_pos)
        return distance

    def sort_objects_by_distance(self, object_names: List[str], reference_object: str):
        distances = {}
        for obj_name in object_names:
            distances[obj_name] = self.distance_between_objects(
                obj_name, reference_object
            )
        sorted_objects = sorted(distances.items(), key=lambda x: x[1])
        return [obj[0] for obj in sorted_objects]

    def get_container_contents(self, container_name, exclude_types=None):
        container_obj = self.get_obj_by_name(container_name)
        objectIds = container_obj["receptacleObjectIds"]
        if objectIds is None or len(objectIds) == 0:
            return None
        else:
            objects = []
            for objId in objectIds:
                obj = self.get_obj_by_object_id(objId)

                # Some contained object (i.e. faucet) don't need to be cleared
                if (
                    container_obj["objectType"] == "SinkBasin"
                    or container_obj["objectType"] == "Sink"
                    or container_obj["objectType"] == "Faucet"
                ):
                    if obj["objectType"] == "Faucet":
                        continue

                # Account for bugs in AITHOR where items that shouldn't be in others sometimes are
                ok_objs = [
                    "CounterTop",
                    "Toilet",
                    "Microwave",
                    "Sink",
                    "SinkBasin",
                    "Cabinet",
                    "Faucet",
                ]
                if obj["objectType"] in ok_objs:
                    continue

                # Account for bugs in AITHOR where pan can think it's on a plate
                if (
                    container_obj["objectType"] == "Plate"
                    and obj["objectType"] == "Pan"
                ):
                    continue

                objects.append(obj)

            # If there is a ignored object, remove it from the list
            if objects is not None and exclude_types is not None:
                objects = [
                    obj for obj in objects if obj["objectType"] not in exclude_types
                ]

            return objects

    def clarify_pose_error(
        self, error_msg: str, action_type: c.Action, object_name: str
    ):
        """
        Check if the pose error can be clarified for the given action and object.
        """
        if c.POSES_ERROR not in error_msg:
            return None
        if action_type != c.Action.PUT:
            return None

        fillables = ["SinkBasin", "Bowl", "Plates", "Cup", "Mug", "Pot", "Pan"]
        # If object name is one of the fillable containers, clarify the error
        if any(fillable in object_name for fillable in fillables):
            object_contents = self.get_container_contents(object_name)
            object_type = self.get_obj_by_name(object_name)["objectType"]
            if object_contents is not None and len(object_contents) > 0:
                return f"The {object_type} can't fit more items."

        return False

    def avg_interaction_distance(self, object_name):
        max_distance = self.max_interaction_distance(object_name)
        min_distance = self.min_interaction_distance(object_name)
        return (max_distance + min_distance) / 2.0

    # TODO remove duplice wiht utils
    def max_interaction_distance(self, object_name):
        obj = self.get_obj_by_name(object_name)
        obj_type = obj["objectType"]

        if obj_type in c.OBJECT_MAX_INTERACTION_DIST:
            return c.OBJECT_MAX_INTERACTION_DIST[obj_type]
        else:
            return c.DEFAULT_MAX_INTERACTION_DIST

    # TODO duplicate with utils
    def min_interaction_distance(self, object_name):
        obj = self.get_obj_by_name(object_name)
        obj_type = obj["objectType"]

        if obj_type in c.OBJECT_MIN_INTERACTION_DIST:
            return c.OBJECT_MIN_INTERACTION_DIST[obj_type]
        else:
            return c.DEFAULT_MIN_INTERACTION_DIST

    def within_interaction_range(self, object_name, distance, margin=0.0):

        if distance < (self.min_interaction_distance(object_name) - margin):
            return False

        if distance > (self.max_interaction_distance(object_name) + margin):
            return False

        return True

    def holding_obj(self):
        heldItems = self.controller.last_event.metadata["inventoryObjects"]
        if len(heldItems) > 0:
            held_object = self.get_obj_by_object_id(
                self.controller.last_event.metadata["inventoryObjects"][0]["objectId"]
            )

            # Theres a bug in AI2Thor where sometimes an object shows as held even after put in container
            """if held_object['parentReceptacles'] is not None and len(held_object['parentReceptacles']) > 0:
                self.controller.step(
                    action="DropHandObject",
                    forceAction=False,
                )
                return None"""
            return held_object
        return None

    def holding_obj_name(self):
        holding_obj = self.holding_obj()
        if holding_obj is not None:
            return holding_obj["name"]
        return None

    def get_controller_name(self, object_name):
        """
        Get the controller for the object, which may the object itself
        """
        obj = self.get_obj_by_name(object_name)
        object_id = obj["objectId"]
        for controller in self.all_objects():
            if controller["controlledObjects"] is not None:
                if object_id in controller["controlledObjects"]:
                    return controller["name"]

        # If a faucet, controlledObjects is not set, so if there is more than
        # one faucet, return the closest one
        if obj["objectType"] == "SinkBasin":
            faucets = self.get_objs_by_types(["Faucet"], must_exist=False)
            if len(faucets) > 1:
                # Sort by distance to the object
                faucets = self.sort_objects_by_distance(
                    [f for f in faucets], object_name
                )
                return faucets[0]["name"]
            else:
                return faucets[0]["name"]

        return obj["name"]  # If no controller found, return the object itself

    def filter_surface_containers(
        self, container_ids: List[str], extra_filters: List[str] = None
    ):
        objects = []
        filters = c.CLASS_TO_TYPES["Surface"].copy()
        if extra_filters is not None:
            filters.extend(extra_filters)
        for container_id in container_ids:
            container = self.get_obj_by_object_id(container_id)
            obj_type = container["objectType"]
            if obj_type not in filters:
                objects.append(container)

        return objects

    def get_surface_containers(self, object_name):
        """
        Get the surface containers for the object, i.e. the least nested container that is a surface.
        """
        surfaces = []
        containerIds = self.get_obj_by_name(object_name)["parentReceptacles"]
        if (containerIds is None) or (len(containerIds) == 0):
            return surfaces

        for containerId in containerIds:
            container = self.get_obj_by_object_id(containerId)
            if container["objectType"] in c.CLASS_TO_TYPES["Surface"]:
                # If the object is on a surface, return it
                surfaces.append(container)
        return surfaces

    def get_non_surface_container(
        self,
        object_name,
        must_exist=True,
    ):
        if not self.is_in_container(object_name):
            if must_exist:
                raise Exception(f"Object {object_name} is not in a container")
            else:
                return None

        obj = self.get_obj_by_name(object_name)
        # Want the least nested container, so get the last one
        objects = self.filter_surface_containers(obj["parentReceptacles"], ["Floor"])
        if len(objects) == 0:
            # TEMP
            print("Error")
        if (len(objects) > 1) and objects[0]["objectType"] != "StoveBurner":
            # If more than one (i.e. item on plate which is in sink) pick the smallest
            objects.sort(key=self.size_xyz)
        return objects[0]

    def get_sliced_object_name(self, object_name, middle=False):

        unsliced_obj = self.get_obj_by_name(object_name)

        # If an egg, return the cracked egg name
        if unsliced_obj["objectType"] == "Egg":
            return self.get_obj_by_type("EggCracked")["name"]

        sliced_names = self.get_sliced_object_names(unsliced_obj)

        # Some objects just change state, so same object
        if sliced_names is None or len(sliced_names) == 0:
            return object_name
        elif middle is True:
            # Return in the middle of the list
            return sliced_names[len(sliced_names) // 2]
        else:
            # Return the first sliced name
            return sliced_names[0]

    def get_sliced_object_names(self, unsliced_obj) -> Optional[List[str]]:

        sliced_names = []
        unsliced_asset_id = f"{unsliced_obj['assetId']}_"

        # Iterate through all objects in the scene
        sliced_objs = []
        for obj in self.all_objects():
            if obj["name"] == unsliced_asset_id:
                continue
            if unsliced_obj["assetId"] not in obj["name"]:
                continue
            if "Slice" not in obj["objectId"]:
                continue
            sliced_objs.append(obj)

        if len(sliced_objs) == 0:
            # If no sliced objects found, return None
            return None

        # Sort the object by size, small to large
        # Want smallest pieces as they will fit the best
        sliced_objs.sort(key=self.size_xyz)

        # Get the name of the first 4 sliced objects
        sliced_names = []
        for i in range(3):
            sliced_names.append(sliced_objs[i]["name"])

        return sliced_names

    def get_obj_by_object_id(self, object_id):
        # Iterate through all objects in the scene
        for obj in self.all_objects():
            if obj["objectId"] == object_id:
                return obj

        raise AgentFailure(f"Cannot find object of object_id: {object_id}")

    def size_max(self, obj) -> float:
        """
        Get the maximum size of the object, which is the area of the largest face.
        This is used to determine max space that this object can take
        """
        x = obj["axisAlignedBoundingBox"]["size"]["x"]
        y = obj["axisAlignedBoundingBox"]["size"]["z"]
        max_dim = max(x, y)
        return max_dim * max_dim

    def size_xz(self, obj) -> float:
        # Get the size of the object
        size = (
            obj["axisAlignedBoundingBox"]["size"]["x"]
            * obj["axisAlignedBoundingBox"]["size"]["z"]
        )
        return size

    def size_xyz(self, obj) -> float:
        # Get the size of the object
        size = (
            obj["axisAlignedBoundingBox"]["size"]["x"]
            * obj["axisAlignedBoundingBox"]["size"]["y"]
            * obj["axisAlignedBoundingBox"]["size"]["z"]
        )
        return size

    def get_object_dimensions(self, obj):
        x_size = obj["axisAlignedBoundingBox"]["size"]["x"]
        z_size = obj["axisAlignedBoundingBox"]["size"]["z"]

        # Length is the larger dimension, width is the smaller
        length = max(x_size, z_size)
        width = min(x_size, z_size)

        return length, width

    def not_occupied(self, obj):
        """
        Some object can only contain one item
        """
        if obj["objectType"] == "StoveBurner":
            objectIds = obj["receptacleObjectIds"]
            if objectIds is not None and len(objectIds) > 0:
                return False
        return True

    def will_fit_in_container(self, receptacle, obj):
        """
        Check if the object will fit in the receptacle
        """
        r_length, r_width = self.get_object_dimensions(receptacle)
        o_lenght, o_width = self.get_object_dimensions(obj)

        if r_length > o_lenght and r_width > o_width:
            return True
        if r_length > o_width and r_width > o_lenght:
            return True
        return False

    def available_space(self, obj) -> float:
        """
        Calculate the available space in a receptacle by subtracting the size of the receptacle from the size of the object.
        """
        size = self.size_xz(obj)
        if obj["receptacleObjectIds"] is not None:
            for receptacleId in obj["receptacleObjectIds"]:
                receptacle = self.get_obj_by_object_id(receptacleId)
                receptacle_size = self.size_max(receptacle)
                size -= receptacle_size

        sink = self.contains_sink(obj)
        if sink is not None:
            size -= self.size_xz(sink)
            print(f"Object {obj['name']} contains sink {sink['name']}")
        return size

    def contains_sink(self, countertop):
        """
        Check to see if any sinks are inside the object by comparing axisAlignedBoundingBox
        for the object and the sink.
        """
        sinks = self.get_objs_by_types(["Sink"], must_exist=False)
        if sinks is None or len(sinks) == 0:
            return None
        for sink in sinks:
            if self.check_aabb_overlap(
                countertop["axisAlignedBoundingBox"], sink["axisAlignedBoundingBox"]
            ):
                return sink
        return None

    def check_aabb_overlap(self, aabb1, aabb2):
        """
        Check if two axis-aligned bounding boxes (AABBs) overlap.
        Each AABB is represented by its center and size.
        """
        aabb1_min_x = aabb1["center"]["x"] - aabb1["size"]["x"] / 2
        aabb1_max_x = aabb1["center"]["x"] + aabb1["size"]["x"] / 2
        aabb1_min_z = aabb1["center"]["z"] - aabb1["size"]["z"] / 2
        aabb1_max_z = aabb1["center"]["z"] + aabb1["size"]["z"] / 2

        aabb2_min_x = aabb2["center"]["x"] - aabb2["size"]["x"] / 2
        aabb2_max_x = aabb2["center"]["x"] + aabb2["size"]["x"] / 2
        aabb2_min_z = aabb2["center"]["z"] - aabb2["size"]["z"] / 2
        aabb2_max_z = aabb2["center"]["z"] + aabb2["size"]["z"] / 2

        return not (
            aabb1_max_x < aabb2_min_x
            or aabb1_min_x > aabb2_max_x
            or aabb1_max_z < aabb2_min_z
            or aabb1_min_z > aabb2_max_z
        )

    def get_specifier_types(self, specifier: Specifier):
        """
        Get a list of objects that match the given specifier.
        """
        if specifier.names is not None:
            return self.get_objs_by_names(specifier.names)
        elif specifier.types is not None:
            return specifier.types
        elif specifier.classes is not None:
            return self.get_objs_by_classes(specifier.classes)
        else:
            return None

    def get_specifier_objects(self, specifier: Specifier):
        """
        Get a list of objects that match the given specifier.
        """
        if specifier.names is not None:
            return self.get_objs_by_names(specifier.names)
        elif specifier.types is not None:
            return self.get_objs_by_types(specifier.types)
        elif specifier.classes is not None:
            return self.get_objs_by_classes(specifier.classes)
        else:
            return None

    def get_specifier_secondary_objects(self, specifier: Specifier):
        """
        Get secondary objects (used in case primary objects fail)
        """
        if specifier.secondary_names is not None:
            return self.get_objs_by_names(specifier.secondary_names)
        elif specifier.secondary_types is not None:
            return self.get_objs_by_types(specifier.secondary_types)
        return None

    def get_specifier_alias_objects(self, specifier: Specifier):

        # To the Model a CounterTop, DiningTable and SideTable can
        # all look the same, so if one is specified, return the others as alias objects
        # that are in the agent's view
        if (
            specifier.names is None
            and specifier.types is not None
            and len(specifier.types) == 1
        ):
            alias_objects = ["CounterTop", "DiningTable", "SideTable"]
            if specifier.types[0] in alias_objects:
                # Get remaining alias objects
                alias_types = [t for t in alias_objects if t != specifier.types[0]]
                objs = self.get_objs_by_types(alias_types, must_exist=False)
                if objs:
                    return [obj for obj in objs if obj["visible"]]
                return None

    def get_objs_by_classes(self, object_classes: List[str]):
        objects = []
        for obj_class in object_classes:
            if obj_class in c.CLASS_TO_TYPES:
                for obj_type in c.CLASS_TO_TYPES[obj_class]:
                    new_objects = self.get_objs_by_types([obj_type], must_exist=False)
                    if new_objects is not None:
                        objects.extend(new_objects)

        # TODO - this should be done by the caller
        objects = self.sort_by_location(objects)
        return objects

    def get_objs_by_types(
        self,
        object_types: List[str],
        filters: List[c.FilterType] = None,
        must_exist=True,
    ):
        # Return list of objects that match the given type
        objects = []
        for obj in self.all_objects():
            if obj["objectType"] in object_types:
                objects.append(obj)
        if len(objects) == 0:
            if must_exist:
                Utils.print_color(
                    c.Color.YELLOW, f"Cannot find object of types: {object_types}"
                )
            return None

        # If there are filters, apply them
        if filters is not None:
            objects = self.filter_objects(objects, filters)

        # TODO - this should be done by the caller
        objects = self.sort_by_location(objects)
        return objects

    def get_obj_by_type(self, object_type: str, must_exist=True):
        """
        Get a single object by type. If multiple objects of the same type exist, return the first one.
        """
        for obj in self.all_objects():
            if obj["objectType"] == object_type:
                return obj

        if must_exist:
            raise AgentFailure(f"Cannot find object of type: {object_type}")

        return None

    def sort_by_location(self, objects):

        # If multiple choices
        if len(objects) > 1:
            # First sort by distance
            objects = sorted(objects, key=lambda x: self.distance_to_object(x["name"]))

            # Then sort by angle
            objects = sorted(objects, key=lambda x: self.calculate_turn_to_object(x))

            # Then sort by which ones are visible
            objects = sorted(
                objects, key=lambda x: self.is_visible(x["name"]), reverse=True
            )

        return objects

    def get_objs_by_names(self, object_names: List[str]):
        # Iterate through all objects in the scene
        objects = []
        for name in object_names:
            obj = self.get_obj_by_name(name)
            if obj is not None:
                objects.append(obj)
        return objects

    def get_obj_by_name(self, object_name: str, must_exist=True) -> Any:

        if object_name is None:
            raise AgentFailure("Object name cannot be None")

        # Iterate through all objects in the scene
        for obj in self.all_objects():
            if obj["name"] == object_name:
                return obj

        if must_exist:
            raise AgentFailure(f"Cannot find object of name: {object_name}")

        return None

    def get_agent(self):
        return self.controller.last_event.metadata["agent"]

    def get_agent_and_object(self, object_name):
        agent = self.get_agent()
        object = self.get_obj_by_name(object_name)

        assert (
            object is not None and agent is not None
        ), f"Cannot find agent or object {object_name}"

        return agent, object

    def get_nppos(self, obj):
        position = obj["position"]
        return Utils.array_to_np(position)

    def get_rot(self, obj):
        rotation = obj["rotation"]
        return rotation["y"] % 360  # Normalize to [0, 360)

    def double_spaces(self, n: int) -> str:
        return " " * (n * 2)

    def add_substep(self, parent_step: RawStep):
        """
        Add a substep to the current step if the current step is not already a substep
        """
        if self.raw_plan.task_failed:
            return

        last_substep = self.cur_step.substeps[-1]
        if last_substep == parent_step:
            self.cur_step = last_substep

    def step_add(
        self,
        action: c.Action,
        obj: str,
        action_desc,
        reasoning,
        include_in_substep=True,
    ) -> RawStep:
        self.do_step = RawStep(
            action_desc=action_desc,
            action=action,
            obj=obj,
            parent=self.cur_step,
            reasoning=reasoning,
            include_in_substep=include_in_substep,
            is_injection=self.injection_state != c.InjectionState.POST_INJECTION,
        )
        self.cur_step.substeps.append(self.do_step)

        self.print_cur_step()

        return self.do_step

    def step_set_data(self, step: RawStep, image_filename: str = None):
        step.set_image_filename(image_filename)
        if self.cur_find_observations is not None:
            step.observations = self.cur_find_observations
            self.cur_find_observations = None

            step.object_bounding_boxes = self.cur_find_bounding_boxes
            self.cur_find_bounding_boxes = None

            step.pose = self.cur_find_pose
            self.cur_find_pose = None
        else:
            step.observations = self.cur_observations
            self.cur_observations = None

            step.object_bounding_boxes = self.cur_bounding_boxes
            self.cur_bounding_boxes = None

            step.pose = self.cur_pose
            self.cur_pose = None

        step.memory = copy.deepcopy(self.memory)

    def step_done(self, step: RawStep):

        # If task failed, don't do anything
        if self.raw_plan.task_failed:
            return

        # If I've popped back up a level and parent task should be part of the substep, add it
        if step.level != self.cur_step.substeps[-1].level and step.include_in_substep:
            new_step = RawStep(
                action_desc=step.action_desc,
                action=step.action,
                obj=step.object,
                parent=self.cur_step,
                is_injection=self.injection_state != c.InjectionState.POST_INJECTION,
            )
            self.cur_step.substeps.append(new_step)

            if self.cur_image is not None:
                image_filename = self.save_image(step.action_desc)
                self.step_set_data(new_step, image_filename=image_filename)

        elif len(step.substeps) == 0:
            image_filename = self.save_image(step.action_desc)
            self.step_set_data(step, image_filename=image_filename)

        # Save the iamge
        self.previous_image = self.cur_image

    def step_complete(self, step: RawStep):
        """
        Complete the current step and move back to the parent step
        """
        while self.cur_step.level != (step.level - 1):
            self.cur_step = self.cur_step.parent
            if self.cur_step.level == 0:
                return

    def print_cur_step(self, file=None, include_images=True):
        if DEBUG_SHOW_PLAN:
            self.print_plan(self.raw_plan.first_step, file, include_images)
        else:
            last_step = (
                self.cur_step.substeps[-1] if self.cur_step.substeps else self.cur_step
            )
            spaces = self.double_spaces(last_step.level)
            step_count = self.raw_plan.step_count()
            task = (
                f"[{step_count}]{spaces}{last_step.action_desc} - {last_step.reasoning}"
            )

            if self.injection_state == c.InjectionState.PRE_INJECTION:
                color = c.Color.GREY_BLUE
            elif self.injection_state == c.InjectionState.INJECTING:
                color = c.Color.YELLOW
            else:
                color = c.Color.GREY
            Utils.print_color(color, task)

    def print_plan(self, cur_step: RawStep, file=None, include_images=True):
        spaces = self.double_spaces(cur_step.level)
        task = f"{spaces}{cur_step.action_desc}"

        if file is None and (cur_step == self.cur_step):
            task = Utils.add_color(c.Color.GREEN, task)

        if include_images:
            task = f"{task:<35}  {cur_step.image_filename()}"

        if file is not None:
            file.write(task + "\r\n")
        else:
            if self.raw_plan.task_failed:
                task = Utils.add_color(c.Color.ORANGE, task)
            print(task)
        for step in cur_step.substeps:
            self.print_plan(step, file, include_images)

    def complete_manual(self):
        # This will stop Unity and reset the controller
        self.controller.stop()
        PlacementCache.save()

    def complete(self):

        # Evaluate goals
        success = self.raw_plan.goal.evaluate_goals(self)
        if not success:
            Utils.print_color(c.Color.RED, "Plan goals not met, plan failed")
            if len(self.plan_errors) == 0:
                self.raw_plan.task_failed = True
                self.plan_errors.append("Plan goals not met, plan failed")

        # This will stop Unity and reset the controller
        self.controller.stop()
        PlacementCache.save()

        # Save plan
        filename = f"{self.save_dir}/steps_w_images.txt"
        with open(filename, "w") as file:
            self.print_plan(self.raw_plan.first_step, file, include_images=True)

        # Save plan steps only
        filename = f"{self.save_dir}/steps.txt"
        with open(filename, "w") as file:
            self.print_plan(self.raw_plan.first_step, file, include_images=False)

        # Save raw json version of the plan
        filename = f"{self.save_dir}/raw_plan.json"
        with open(filename, "w") as file:
            json.dump(self.raw_plan.to_dict(), file)

        action_plan = self.raw_plan.plan_from_raw_plan(self.save_dir)

        if len(action_plan.steps) == 0:
            # This can happen in injection step occurs after plan has been completed
            # Delete the save_dir
            os.system(f"rm -rf '{self.save_dir}'")
            Utils.print_color(
                c.Color.RED, "No actions in action plan, deleting save directory"
            )
            return

        # Save json version of action plan
        filename = f"{self.save_dir}/plan.json"
        with open(filename, "w") as file:
            json.dump(action_plan.to_dict(), file)

        # Save log
        filename = f"{self.save_dir}/log.txt"
        with open(filename, "w") as file:
            for line in self.log:
                file.write(f"{line}\n")

        self.create_video()

        # Add number of steps to the plan folder name
        num_steps = len(action_plan.steps)

        # For GENERATED plan any plan_error is considered a failure
        if len(self.plan_errors) > 0:
            # Rename director to indicate a failure
            fail_dir = self.save_dir.replace(
                self.raw_plan.name, f"_{self.raw_plan.name}"
            )
            os.rename(self.save_dir, fail_dir)
            save_dir = fail_dir
        else:
            save_dir = self.save_dir

        new_dir = f"{save_dir} [{num_steps}]"
        if os.path.exists(new_dir):
            shutil.rmtree(new_dir)

        os.rename(save_dir, new_dir)
        save_dir = new_dir

        if len(self.plan_errors) == 0 and c.DATASET_DIR in save_dir:
            # Otherwise also copy directory to "best of" folder
            copy_dir = save_dir.replace(
                f"{self.data_folder}/", f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}/"
            )

            # If the directory exists delete it
            if os.path.exists(copy_dir):
                os.system(f"rm -rf '{copy_dir}'")

            # Copy the directory
            os.system(f"cp -rf '{save_dir}' '{copy_dir}'")

            # If failures were recorded make a failure recovery example
            action_plan.make_failed_plans(3, copy_dir)

        if len(self.plan_errors) > 0:
            color = c.Color.RED
        else:
            color = c.Color.GREEN

        Utils.print_color(
            color, f"{self.raw_plan.name} : {self.raw_plan.task_description}"
        )
        print("-----------------------------------------------")

    def try_bring_into_view(self, object_name: str):
        """
        Try to bring an object into view
        """
        if not self.is_visible(object_name):
            # Save observation in case this fails, I can go back
            image = self.cur_image
            observations = self.cur_observations
            bounding_boxes = self.cur_bounding_boxes
            pose = self.cur_pose
            try:
                self.execute_action(object_name, self.agent_find)
            except:
                self.controller.step(
                    action="Teleport",
                    position=pose.position,
                    rotation=pose.rotation,
                    horizon=pose.horizon,
                    standing=pose.isStanding,
                )

                # If this fails, restore the previous state
                self.cur_image = image
                self.cur_observations = observations
                self.cur_bounding_boxes = bounding_boxes
                self.cur_pose = pose

    def capture_video_frame(self):
        """Capture the current frame for video recording."""
        if self.record_video:
            image = Image.fromarray(self.controller.last_event.frame)
            self.video_frames.append(image)

    NEED_ROTATION_TYPES = ["Knife", "ButterKnife"]

    def rotate_held_object(self):
        # Some objects (knives) can't be seen well without rotation
        holding_name = self.holding_obj_name()
        if holding_name is not None:
            holding_obj = self.get_obj_by_name(holding_name)
            if holding_obj["objectType"] in ["Knife", "ButterKnife"]:
                self.controller.step(
                    action="RotateHeldObject", rotation=dict(x=0, y=0, z=90)
                )
                # Give the image a second to update
                self.controller.step(action="Pass")
                self.controller.step(action="Pass")

    def try_back_up(self, move_action):
        """
        Hand held item can get in way of rotation
        If it does, back up and try again
        """
        error_msg = self.controller.last_event.metadata["errorMessage"]
        if error_msg != "":
            if move_action == "RotateLeft" or move_action == "RotateRight":
                if "agent rotates" in error_msg:
                    self.controller.step(action="MoveBack")
                    self.controller.step(action="Pass")

    def agent_move_action(self, move_action):
        self.controller.step(move_action)

        if self.controller.last_event.metadata["errorMessage"]:
            self.try_back_up(move_action)

        success = self.controller.last_event.metadata["lastActionSuccess"]

        self.controller.step(action="Pass")

        return success

    def visible_relevant_objects(self):
        relevant_objects = []
        named_types = []
        for name in Specifier.observed_names:
            obj = self.get_obj_by_name(name)
            if obj not in relevant_objects:
                relevant_objects.append(obj)
                named_types.append(obj["objectType"])

        for type in Specifier.observed_types:
            # If I have a concrete names item of type, don't add the type
            if type not in named_types:
                objs = self.get_objs_by_types([type], must_exist=False)
                if objs is not None:
                    for obj in objs:
                        if obj not in relevant_objects:
                            relevant_objects.append(obj)
                            named_types.append(obj["objectType"])

        for klass in Specifier.observed_classes:
            objs = self.get_objs_by_classes([klass])
            if objs is not None:
                for obj in objs:
                    type = obj["objectType"]
                    # If I have a concrete names item of type, don't add the type
                    if type not in named_types:
                        if obj not in relevant_objects:
                            relevant_objects.append(obj)

        # Filter to only object that can be seen
        visible_objects = [obj for obj in relevant_objects if obj["visible"]]
        return visible_objects

    def get_object_bounding_boxes(self):
        visible_objects = self.visible_relevant_objects()
        bounding_boxes = {}
        for obj in visible_objects:
            name = obj["name"]
            if name in bounding_boxes:
                print(f"Warning: {name} already has a bounding box, skipping")
            if obj["objectId"] in self.controller.last_event.instance_detections2D:
                bounding_boxes[name] = self.controller.last_event.instance_detections2D[
                    obj["objectId"]
                ]
            else:
                Utils.print_color(
                    c.Color.YELLOW,
                    f"Warning: {name} does not have a bounding box in the scene.",
                )

        holding_name = self.holding_obj_name()
        if holding_name is not None:
            holding_obj = self.get_obj_by_name(holding_name)
            name = Utils.short_name(holding_name)
            if (
                holding_obj["objectId"]
                in self.controller.last_event.instance_detections2D
            ):
                bounding_boxes[name] = self.controller.last_event.instance_detections2D[
                    holding_obj["objectId"]
                ]
            else:
                Utils.print_color(
                    c.Color.YELLOW,
                    f"Warning: {holding_name} does not have a bounding box in the scene.",
                )

        return bounding_boxes

    def get_memory(self, object_name) -> Memory:
        if object_name not in self.memory:
            memory = Memory(object_name)
            self.memory[object_name] = memory
        return self.memory[object_name]

    def clear_holding_object(self, object_name):
        obj = self.get_obj_by_name(object_name)
        for holding_name, holding_contents in self.holding_contents.items():
            if obj["objectId"] in holding_contents:
                holding_contents.remove(obj["objectId"])
                if len(holding_contents) == 0:
                    del self.holding_contents[holding_name]
                break

    def get_agent_pose(self) -> Pose:
        agent = self.get_agent()
        pose = Pose(
            position=agent["position"],
            rotation=agent["rotation"]["y"],
            isStanding=agent["isStanding"],
            horizon=agent["cameraHorizon"],
        )
        return pose

    def get_observations(self):
        my_observations = []
        holding_name = self.holding_obj_name()
        if holding_name is not None:
            holding_name = Utils.short_name(holding_name)
            my_observations.append(f"I'm holding {holding_name}")
        else:
            my_observations.append("I'm not holding anything")

        visible_objects = self.visible_relevant_objects()

        # Are any of the objects of type Microwave?  I can see through the door
        microwave = None
        for object in visible_objects:
            if "Microwave" in object["objectType"]:
                microwave = object
                break

        if microwave is not None:
            inside_microwave = []
            for object in visible_objects:
                if (
                    object["parentReceptacles"] is not None
                    and microwave["objectId"] in object["parentReceptacles"]
                ):
                    if object not in visible_objects:
                        inside_microwave.append(object)
            visible_objects.extend(inside_microwave)

        # Convert to name counts
        if len(visible_objects) == 0:
            my_observations.append("I see nothing relevant to my task.")
            return my_observations

        # Convert all seen object names to short names
        seen_names = [Utils.short_name(obj["name"]) for obj in visible_objects]

        name_counts = Counter(seen_names)
        formatted_items = []
        for name, count in name_counts.items():
            if holding_name == name:
                formatted_items.append(f"{name} (held)")
                if count > 2:
                    formatted_items.append(f"{name} (multiple unheld)")
                elif count > 1:
                    formatted_items.append(f"{name} (unheld)")
            elif count > 1:
                formatted_items.append(f"{name} (multiple)")
            else:
                formatted_items.append(f"{name}")

        seen = Utils.join_with_and(formatted_items)
        my_observations.append(f"I see: {seen}.")

        # Add containment
        for object in visible_objects:
            name = Utils.short_name(object["name"])
            if self.is_in_container(object["name"]):
                container_obj = self.smallest_containing_object(object)
                if container_obj is not None and container_obj in visible_objects:
                    parent_name = Utils.short_name(container_obj["name"])
                    my_observations.append(f"{name} is in {parent_name}.")
                    memory = self.get_memory(name)
                    memory.container = parent_name

            elif name in self.memory:
                # If the object is not in a container, remove the container memory
                self.memory[name].container = None

        # Now handle that AI2Thor loses containment when item is inside another
        # object that has been picked up.  Saved this when I picked up the container
        for container_name, holding_contents in self.holding_contents.items():
            container_short_name = Utils.short_name(container_name)
            for holding_id in holding_contents:
                held_obj = self.get_obj_by_object_id(holding_id)
                name = Utils.short_name(held_obj["name"])
                my_observations.append(
                    f"{name} is in {Utils.short_name(container_short_name)}."
                )
                memory = self.get_memory(name)
                memory.container = container_short_name

        # Group by object type (array[type][objects])
        type_groups = {}
        for object in visible_objects:
            obj_type = object["objectType"]
            if obj_type not in type_groups:
                type_groups[obj_type] = []
            type_groups[obj_type].append(object)

        # Now add object states
        for type_group, objects in type_groups.items():
            objects = type_groups[type_group]

            for object in objects:
                name = Utils.short_name(object["name"])
                if object["isFilledWithLiquid"]:
                    my_observations.append(
                        f"{name} is filled with {object['fillLiquid']}."
                    )
                    memory = self.get_memory(name)
                    memory.filled_with = object["fillLiquid"]
                elif name in self.memory:
                    self.memory[name].filled_with = None

            if any(o["openable"] for o in objects):
                name = Utils.short_name(objects[0]["name"])
                memory = self.get_memory(name)
                # If any of the object in the group are open, say so
                if any(o["isOpen"] for o in objects):
                    my_observations.append(f"{name} is open.")
                    memory.is_open = True
                else:
                    memory.is_open = False
                    if len(objects) > 1:
                        my_observations.append(f"{name}s are closed.")
                    else:
                        my_observations.append(f"{name} is closed.")

            if any(o["cookable"] for o in objects):
                name = Utils.short_name(objects[0]["name"])
                memory = self.get_memory(name)
                if objects[0]["isCooked"]:
                    my_observations.append(f"{name} is cooked.")
                    memory.is_cooked = True
                else:
                    my_observations.append(f"{name} is uncooked.")
                    memory.is_cooked = False

            if any(o["dirtyable"] for o in objects):
                name = Utils.short_name(objects[0]["name"])
                if objects[0]["isDirty"]:
                    memory = self.get_memory(name)
                    memory.is_dirty = True
                    my_observations.append(f"{name} is dirty.")
                elif name in self.memory:
                    self.memory[name].is_dirty = None

            if any(o["toggleable"] for o in objects):
                name = Utils.short_name(objects[0]["name"])
                memory = self.get_memory(name)
                if objects[0]["isToggled"]:
                    my_observations.append(f"{name} is on.")
                    memory.is_toggled = True
                else:
                    my_observations.append(f"{name} is off.")
                    memory.is_toggled = False

        # Now check for goal satisfaction
        for obj in visible_objects:
            memory = self.get_memory(obj["name"])

            # If object is not currently satisfying the goal, note it
            if self.is_used_for_goal(obj):
                container = self.get_non_surface_container(
                    obj["name"], must_exist=False
                )
                satisfy_text = f"Objective met {Utils.short_name(obj['name'])} inside {Utils.short_name(container['name'])}."
                my_observations.append(satisfy_text)
                memory.satisfies_goal = satisfy_text
            elif memory is not None and memory.satisfies_goal is not None:
                # If the object was previously satisfying the goal, but no longer does, clear the memory
                memory.satisfies_goal = None

        for action, outcome in self.raw_plan.goal.action_goals.items():
            memory = self.get_memory(action)
            if outcome is True:
                memory.satisfies_goal = f"Action {action} completed successfully."
            else:
                memory.satisfies_goal = None

        # Remove any memories that are empty
        for name in list(self.memory.keys()):
            if self.memory[name].is_empty():
                del self.memory[name]

        return my_observations

    def compose_hand_overlay(self, base_image: Image.Image) -> Image.Image:
        """
        Compose the hand image on top of the base image with the specified transparency.
        Uses calibrated x/y shifts from handshifts.json based on the held object's assetId
        and current camera angle.
        Returns the composited image if hand_transparency > 0 and agent is holding something,
        otherwise returns the base image.
        """
        if self.hand_transparency <= 0:
            return base_image

        # Only overlay hand if holding something
        held_obj = self.holding_obj()
        if held_obj is None:
            return base_image

        # Load and cache the hand image
        if self._hand_image is None:
            hand_path = os.path.join(os.path.dirname(__file__), "Data", "hand.png")
            self._hand_image = Image.open(hand_path).convert("RGBA")

        # Load and cache the handshifts data
        if self._handshifts is None:
            handshifts_path = os.path.join(
                os.path.dirname(__file__), "Data", "handshifts.json"
            )
            if os.path.exists(handshifts_path):
                with open(handshifts_path, "r", encoding="utf-8") as f:
                    self._handshifts = json.load(f)
            else:
                self._handshifts = {}

        # Determine the asset key for looking up handshifts
        # Sliced objects have "Slice" in their objectId and use {assetId}_Sliced as key
        asset_id = held_obj.get("assetId", "")

        if "Slice" in held_obj.get("objectId", ""):
            # For sliced objects, we need to find the original assetId from the name
            # The name format includes the original object type
            if asset_id is None or asset_id == "":
                # Extract from the name Potato_20_Slice_6 => Potato_20
                name_parts = held_obj["name"].split("_")
                asset_id = f"{name_parts[0]}_{name_parts[1]}"

            asset_key = f"{asset_id}_Sliced"
        else:
            asset_key = asset_id if asset_id else None

        # Get the shift for the current camera angle
        shift_x = 0
        shift_y = 0
        if asset_key and asset_key in self._handshifts:
            camera_horizon = self.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]
            # Round to nearest calibrated angle
            calibrated_angles = [-30, -20, -10, 0, 10, 20, 30, 40]
            nearest_angle = min(
                calibrated_angles, key=lambda a: abs(a - camera_horizon)
            )
            angle_key = str(int(nearest_angle))
            if angle_key in self._handshifts[asset_key]:
                shift_x = self._handshifts[asset_key][angle_key].get("x", 0)
                shift_y = self._handshifts[asset_key][angle_key].get("y", 0)
            else:
                Utils.print_color(
                    c.Color.RED,
                    f"Warning: No handshift data for angle {angle_key} of asset {asset_key}.",
                )

        # Resize hand image to match base image size if needed
        hand = self._hand_image
        if hand.size != base_image.size:
            hand = hand.resize(base_image.size, Image.Resampling.LANCZOS)

        # Convert base image to RGBA if not already
        base_rgba = base_image.convert("RGBA")

        # Create a new transparent image to hold the shifted hand
        shifted_hand = Image.new("RGBA", base_image.size, (0, 0, 0, 0))

        # Paste the hand onto the shifted position
        shifted_hand.paste(hand, (shift_x, shift_y))

        # Apply transparency to the hand image's alpha channel
        alpha = shifted_hand.split()[3]
        alpha = alpha.point(lambda p: int(p * (self.hand_transparency / 100)))
        shifted_hand.putalpha(alpha)

        # Composite the hand on top of the base image
        composited = Image.alpha_composite(base_rgba, shifted_hand)

        # Convert back to RGB for consistency
        return composited.convert("RGB")

    def capture_world_state(self):
        self.rotate_held_object()
        self.cur_image = Image.fromarray(self.controller.last_event.frame)
        self.cur_image = self.compose_hand_overlay(self.cur_image)
        self.cur_observations = self.get_observations()
        self.cur_bounding_boxes = self.get_object_bounding_boxes()
        self.cur_pose = self.get_agent_pose()

        if DEBUG_IMAGES:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(self.cur_image)
            plt.axis("off")
            plt.title("CAPTURE")
            plt.show(block=True)

    def capture_world_state_find(self):
        self.rotate_held_object()
        self.cur_find_image = Image.fromarray(self.controller.last_event.frame)
        self.cur_find_image = self.compose_hand_overlay(self.cur_find_image)
        self.cur_find_observations = self.get_observations()
        self.cur_find_bounding_boxes = self.get_object_bounding_boxes()
        self.cur_find_pose = self.get_agent_pose()

        if DEBUG_IMAGES:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.imshow(self.cur_image)
            plt.axis("off")
            plt.title("CAPTURE")
            plt.show(block=True)

    def save_image(self, name):

        # Don't save image if I'm not done with any injections
        if self.injection_state == c.InjectionState.PRE_INJECTION:
            return "PRE-INJECTION"

        if DEBUG_IMAGES:
            import matplotlib.pyplot as plt

            if self.cur_find_image is not None:
                image = self.cur_find_image
            else:
                image = self.cur_image

            plt.figure(figsize=(10, 8))
            plt.imshow(image)
            plt.axis("off")
            plt.title(name)
            plt.show(block=True)

        # Pass step
        self.controller.step(action="Pass")
        filename = f"_{self.image_index}_{name}.png"
        fullname = f"{self.save_dir}/{filename}"

        # Find needs to extract image and save it separately
        if self.cur_find_image is not None:
            # Save the find image
            self.cur_find_image.save(fullname)
            self.cur_find_image = None
        else:
            self.cur_image.save(fullname)
            self.cur_image = None

        self.image_index += 1
        return filename

    def get_obj_from_put_cache(self, objects):
        if objects is None or len(objects) == 0:
            return None

        holding_name = self.holding_obj_name()

        for obj in objects:
            pose = PutCache.get_put_pose(self.hash, holding_name, obj["name"])
            if pose is not None:
                return self.get_obj_by_name(obj["name"])
        return None

    def calculate_turn_to_object(self, obj):
        # Calculate angle to face the object from new position
        agent = self.get_agent()
        agent_rot = self.get_rot(agent)

        angle = self.calculate_rotation(obj["name"])
        turn_distance = self.norm_angle(angle - agent_rot)
        return turn_distance

    def calculate_rotation(self, object_name: str, from_nppos=None) -> float:
        """
        Compute the angle in degrees that object A (position a) needs to turn on the horizontal (x, z)
        plane to face object B (position b).
        """
        obj = self.get_obj_by_name(object_name)
        obj_nppos = self.get_nppos(obj)

        # If from_nppos is not provided, use the agent's position
        if from_nppos is None:
            agent = self.get_agent()
            from_nppos = self.get_nppos(agent)

        # Calculate the differences along x and z axes (ignore y)
        delta = obj_nppos - from_nppos
        delta_x = delta[0]
        delta_z = delta[2]

        # Compute the angle in radians using arctan2
        angle_rad = np.arctan2(delta_x, delta_z)

        # Convert the angle to degrees
        angle_deg = np.degrees(angle_rad)
        return float(angle_deg % 360)  # Normalize to [0, 360)

    def norm_angle(self, yaw):
        return yaw % 360

    def shortest_turn_direction(self, current_angle, desired_angle, threshold=0):
        """
        Determine whether to turn left or right to reach the desired angle
        using the shortest path.

        Args:
            current_angle (int): The current angle (can be any integer, positive or negative).
            desired_angle (int): The desired angle (can be any integer, positive or negative).

        Returns:
            str: "left" if the shortest turn is to the left, "right" if to the right.
        """
        # Normalize angles to 0-360
        current_normalized = current_angle % 360
        desired_normalized = desired_angle % 360

        abs_diff = abs(desired_normalized - current_normalized)
        if abs_diff < threshold:
            return c.TurnDirection.NONE

        # Calculate the turn difference
        diff = (desired_normalized - current_normalized) % 360

        # Determine the shortest direction
        if diff == 0:
            return c.TurnDirection.NONE  # Already facing the desired angle
        elif diff <= 180:
            return c.TurnDirection.RIGHT
        else:
            return c.TurnDirection.LEFT

    def is_visible(self, object_name):
        _, obj = self.get_agent_and_object(object_name)
        if obj["visible"]:
            return True

        # Sometimes contained objects, block view even though they are visible
        # Check to see if object_name contains any visible objects
        contained_objects = self.get_container_contents(object_name)
        if contained_objects is not None:
            for contained_obj in contained_objects:
                if contained_obj["visible"]:
                    return True

        # Sometimes objects I'm holding show up as not visible
        holding_name = self.holding_obj_name()
        if holding_name is not None and holding_name == object_name:
            # If I'm holding the object, it's considered visible
            return True
        return False

    def is_type_visible(self, object_type):
        """
        Check if any object of the given type is visible.
        """
        for obj in self.all_objects():
            if obj["objectType"] == object_type and obj["visible"]:
                return True
        return False

    def is_used_for_goal(self, object):
        """
        Check if the object is used for the goal.
        """
        container = self.get_non_surface_container(object["name"], must_exist=False)
        if self.raw_plan.goal.employed_for_goal(object, container):
            return True
        return False

    def is_openable(self, object_name):
        _, obj = self.get_agent_and_object(object_name)
        if obj["openable"] is True:
            return True
        return False

    # NOTE: Without this pause it seems the things fall behind
    # this makes sure that things are properly updated first
    def pause(self):
        self.controller.step(action="Done")
        self.controller.step(action="Done")
        self.controller.step(action="Done")

    def close_on_failure(self, action_type, object_name):
        if action_type != c.Action.PUT:
            return
        obj = self.get_obj_by_name(object_name)
        if not obj["openable"] or not obj["isOpen"]:
            return

        self.agent_close(object_name, admin_task=True)
        # Clear last opened container tracking since we're closing it
        obj_type = obj["objectType"]
        if (
            obj_type in self.last_opened_container
            and self.last_opened_container[obj_type] == object_name
        ):
            del self.last_opened_container[obj_type]
            print(
                f"[DEBUG] close_on_failure: Cleared last opened {obj_type}: {object_name}"
            )

    def try_graduate_candidates(self, action_type, candiates, secondary_candidates):

        # Only used cache positions for PUT
        if action_type != c.Action.PUT:
            return

        # Is the a secondary put cached
        cached_put = self.get_obj_from_put_cache(secondary_candidates)
        if cached_put is not None:
            # If so, graduate to primary candidates
            # Remove cached_put from candates
            secondary_candidates.remove(cached_put)
            candiates.insert(0, cached_put)

    def prioritize_cached(self, action_type, objects):
        """
        If doing a PUT action, prioritize the cached container that works so don't have to search for it
        """
        # Only used cache positions for PUT (or OPEN/CLOSE openable containers)
        if not self.should_check_put_cache(action_type, objects):
            return objects

        # First check if we have a last opened container of the same type in the candidate list
        if len(self.last_opened_container) > 0:
            for index, obj in enumerate(objects):
                obj_type = obj["objectType"]
                if obj_type in self.last_opened_container:
                    last_opened_name = self.last_opened_container[obj_type]
                    if obj["name"] == last_opened_name:
                        objects.insert(0, objects.pop(index))
                        print(
                            f"[DEBUG] prioritize_cached: Prioritizing last opened {obj_type}: {last_opened_name}"
                        )
                        return objects

        holding_object = self.holding_obj_name()

        # First check memoery based put_cache (handle object put on counters)
        cached_put = self.get_obj_from_put_cache(objects)
        if cached_put is not None:
            # If the cached object is in the list, move it to the front
            for index, obj in enumerate(objects):
                if obj["name"] == cached_put["name"]:
                    objects.insert(0, objects.pop(index))
                    return objects

        # Check if the object is in the disk cache (for permanant container fit)
        cached_containers = PlacementCache.get_container_names(
            self.raw_plan.scene, holding_object
        )
        if cached_containers is None:
            return objects

        # Put obj_name at the top of the list
        for index, obj in enumerate(objects):
            if obj["name"] in cached_containers:
                # Move it to the front of the list
                objects.insert(0, objects.pop(index))

        # Finally if one is visible and not in failed cache, put it at the front
        cached_failures = PlacementCache.get_failure_names(
            self.raw_plan.scene, holding_object
        )
        for obj in objects:
            if obj["name"] not in cached_failures and obj["visible"]:
                # Move it to the front of the list
                objects.insert(0, objects.pop(objects.index(obj)))
                return objects

        # May have not been a candidate
        return objects

    # Openable container types that can have items PUT in them
    OPENABLE_CONTAINER_TYPES = [
        "Cabinet",
        "Drawer",
        "Fridge",
        "Microwave",
        "Safe",
        "Box",
    ]

    def should_check_put_cache(self, action_type: c.Action, objects) -> bool:
        """
        Determines if the put cache should be checked for the given action type and specifier.
        """
        if action_type == c.Action.PUT:
            return True

        # If opening or closing an openable container, prioritize the last opened one
        if action_type in [c.Action.OPEN, c.Action.CLOSE]:
            obj = self.get_obj_by_name(objects[0]["name"])
            if obj["objectType"] in self.OPENABLE_CONTAINER_TYPES:
                return True

        return False

    #####################################
    #
    # DO Functions
    #
    ####################################
    def do(
        self,
        action_type: c.Action,
        object_specifier: Specifier,
        reasoning: str,
        flags: List[c.Flags] = None,
    ) -> str:
        """
        Main loop for doing an action.  All exceptions are caught and handled here
        """
        try:
            # Clear out any previous step errors
            self.step_error = None
            self.last_retry_error = "--"

            if flags is None:
                flags = []

            if self.raw_plan.step_count() > 0.9 * MAX_PLAN_STEPS:
                Utils.print_color(
                    c.Color.YELLOW, f"Plan step count: {self.raw_plan.step_count()}"
                )

            if self.raw_plan.step_count() > MAX_PLAN_STEPS:
                raise AgentFatal(
                    f"Plan execeeded {MAX_PLAN_STEPS} steps, stopping to avoid likely loop"
                )

            # If a generated plan and done, return None
            if (
                self.raw_plan.task_failed
                and self.raw_plan.plan_type == PlanType.GENERATED
            ):
                return None

            # Note: it's ok if there are no candidates, not all actions require them
            all_candidates = self.get_specifier_objects(object_specifier)
            secondary_candidates = self.get_specifier_secondary_objects(
                object_specifier
            )

            # CounterTop, SideTable and DiningTable are aliased as model
            # doesn't have clear way to distinguish them
            alias_candidates = self.get_specifier_alias_objects(object_specifier)
            if alias_candidates is not None:
                if secondary_candidates is None:
                    secondary_candidates = []
                secondary_candidates.extend(alias_candidates)

            # Filter out any object that are already fullfilling the goal for the curret plan
            candidates = self.filter_objects(
                all_candidates, [c.FilterType.GOAL_CONFLICTS]
            )
            secondary_candidates = self.filter_objects(
                secondary_candidates, [c.FilterType.GOAL_CONFLICTS]
            )

            if (
                all_candidates is not None
                and len(all_candidates) > 0
                and len(candidates) == 0
            ):
                raise AgentFailure(
                    f"All {object_specifier.to_string()} are already being used to fullfill a goal"
                )

            if c.Flags.NOT_USED_UP in flags:
                candidates = self.filter_objects(candidates, [c.FilterType.NOT_USED_UP])
                secondary_candidates = self.filter_objects(
                    secondary_candidates, [c.FilterType.NOT_USED_UP]
                )

            # If a Put action, filter out objects that we know are bad containers
            if action_type == c.Action.PUT:
                candidates = self.filter_objects(
                    candidates, [c.FilterType.DEFECTIVE_CONTAINERS]
                )
                secondary_candidates = self.filter_objects(
                    secondary_candidates, [c.FilterType.DEFECTIVE_CONTAINERS]
                )

                # If we have cached put in secondary candidates, move to primary candidates
                self.try_graduate_candidates(
                    action_type, candidates, secondary_candidates
                )

            # If sliced types, pick smallest object first as more likely to fit in container
            if (
                object_specifier.types
                and object_specifier.types[0] in c.SLICED_TYPES.values()
            ):
                flags.append(c.Flags.SMALLEST_FIRST)

            # If no object type is given, just do the action
            # This is used for actions like empty hand
            if candidates is None or len(candidates) == 0:
                no_candidate_acions = [
                    c.Action.EMPTY_HAND,
                    c.Action.TURN_LEFT,
                    c.Action.TURN_RIGHT,
                    c.Action.TURN_UP,
                    c.Action.TURN_DOWN,
                    c.Action.MOVE_BACKWARD,
                    c.Action.MOVE_FORWARD,
                    c.Action.DIRTY,
                    c.Action.INVALID_ACTION,
                    c.Action.INVALID_OBJECT,
                    c.Action.INVALID_RESPONSE,
                ]
                if action_type in no_candidate_acions:
                    return self._do(action_type, None, reasoning, flags)
                else:
                    raise AgentFailure(
                        f"No candidate objects for action {action_type} with specifier {object_specifier.to_string()}"
                    )

            result, errors = self.do_candidates(
                candidates,
                object_specifier.preferred_name,
                action_type,
                reasoning,
                flags,
                object_specifier.all,
            )
            if result is None and secondary_candidates is not None:
                result, errors = self.do_candidates(
                    secondary_candidates,
                    object_specifier.preferred_name,
                    action_type,
                    reasoning,
                    flags,
                    object_specifier.all,
                )

            if result is not None:
                self.count_action(action_type)

                return result

            if len(errors) > 0:
                self.add_step_errors(errors)

            return None

        except Exception as e:
            print(e)
            traceback.print_exc()
            error = self.cleaned_error(None, str(e))
            self.record_step_error(
                action_name=action_type.value,
                object_name=object_specifier.to_simple_name(),
                error_msg=error,
                error_type=StepErrorType.UNDOABLE,
            )

            # Disable this to debug
            # raise e
            return None

    def do_candidates(
        self,
        candidate_objs: List[str],
        preferred_name: str,
        action_type,
        reasoning: str,
        flags: List[c.Flags],
        do_all,
    ) -> Tuple[str, List[StepError]]:

        all_step_errors: List[StepError] = []
        trying_again = False
        while len(candidate_objs) > 0:
            step_errors: List[StepError] = []
            cur_step = self.cur_step

            # Need to re-sort at agent may have moved
            if c.Flags.DONT_SORT not in flags:
                candidate_objs = self.sort_by_location(candidate_objs)
            candidate_objs = self.prioritize_cached(action_type, candidate_objs)

            if c.Flags.SMALLEST_FIRST in flags:
                candidate_objs.sort(key=self.size_xyz)

            # If preferred instance, put at top
            obj_names = [obj["name"] for obj in candidate_objs]
            if preferred_name is not None and preferred_name in obj_names:
                preferred_index = obj_names.index(preferred_name)
                preferred_obj = candidate_objs.pop(preferred_index)

                if "StoveBurner" in preferred_name:
                    # If given preferred stove burner, it's the only valid one
                    candidate_objs = [preferred_obj]
                else:
                    # Put preferred at top of list
                    candidate_objs.insert(0, preferred_obj)

            obj = candidate_objs.pop(0)

            object_name = obj["name"]

            # If not generating a plan and I'm trying to put in a different place
            # I need to open the put object as retrying another location is automatic
            if trying_again and self.raw_plan.plan_type != PlanType.GENERATED:
                if action_type == c.Action.PUT:
                    self._do(c.Action.FIND, object_name, reasoning, flags)

                    if self.is_openable(object_name) and not self.is_open(object_name):
                        self._do(c.Action.OPEN, object_name, reasoning, flags)

            try:
                object_name = self._do(action_type, object_name, reasoning, flags)

                if self.was_impossible_action():
                    step_error = StepError(
                        action_name=action_type.value,
                        object_name=object_name,
                        error_msg=self.last_retry_error,
                        error_type=StepErrorType.UNDOABLE,
                    )
                    return None, [step_error]

                if not do_all:
                    # If not performing on all objects, I'm done
                    return object_name, step_errors

            # If a fatal error occurs, bail out
            except AgentFatal as e:
                error_msg = self.cleaned_error(object_name, str(e))

                step_error = StepError(
                    action_name=action_type.value,
                    object_name=object_name,
                    error_msg=error_msg,
                    error_type=StepErrorType.UNDOABLE,
                )
                step_errors.append(step_error)
                return None, step_errors

            # If a non-fatal error occurs, try again with another object
            except (AgentFailure, AgentCantDo) as e:
                # Get error message and type based on exception type
                if isinstance(e, AgentCantDo):
                    error_msg = str(e)
                    err_type = (
                        e.error_type
                        if e.error_type is not None
                        else StepErrorType.UNDOABLE
                    )
                else:
                    error_msg = self.cleaned_error(object_name, str(e))
                    err_type = StepErrorType.UNDOABLE

                    # Can we give more detailed message than just pose error?
                    clarification = self.clarify_pose_error(
                        error_msg, action_type, object_name
                    )
                    if clarification:
                        error_msg = clarification
                    # If there is a last retry error, use that instead
                    elif (
                        self.last_retry_error is not None
                        and self.last_retry_error != ""
                        and "T"
                    ):
                        error_msg = self.last_retry_error

                step_error = StepError(
                    action_name=action_type.value,
                    object_name=object_name,
                    error_msg=error_msg,
                    error_type=err_type,
                )
                step_errors.append(step_error)

                # Delete any saved image for this step
                if self.cur_step._image_filename is not None:
                    os.remove(self.cur_step._image_filename)
                    self.image_index -= 1

                # If the action failed, undo the step
                self.cur_step: RawStep = cur_step

                # Remove the last failed substep
                self.cur_step.substeps = self.cur_step.substeps[:-1]

                # If I opened something close it again
                self.close_on_failure(action_type, object_name)

                # If I'm failed the task, stop trying
                if self.raw_plan.task_failed:
                    return None, step_errors

                # Add to list of all errors for all objects
                all_step_errors.extend(step_errors)

                trying_again = True

        # If none of the objects worked, return all the errors
        return None, all_step_errors

    def _do(self, action_type, object_name, reasoning, flags):

        if self.raw_plan.plan_type == PlanType.GENERATED:
            return self._do_auto(action_type, object_name, reasoning, flags)
        else:
            return self._do_manual_action(action_type, object_name, reasoning, flags)

    def _do_auto(self, action_type, object_name, reasoning, flags):

        match (action_type):
            case c.Action.PUT:
                self.do_put(object_name, reasoning, flags)
                return object_name
            case c.Action.FIND:
                self.do_find(object_name, reasoning, flags)
                return object_name
            case c.Action.FACE:
                self.do_face(object_name, reasoning, flags)
                return object_name
            case c.Action.TOGGLE_ON:
                self.do_toggle_on(object_name, reasoning, flags)
                return object_name
            case c.Action.TOGGLE_OFF:
                self.do_toggle_off(object_name, reasoning, flags)
                return object_name
            case c.Action.CLOSE:
                self.do_close(object_name, reasoning, flags)
                return object_name
            case c.Action.PICKUP:
                self.do_pickup(object_name, reasoning, flags)
                return object_name
            case c.Action.OPEN:
                self.do_open(object_name, reasoning, flags)
                return object_name
            case c.Action.CLEAR_CONTAINER:
                self.do_clear_container(object_name, reasoning, flags)
                return object_name
            case c.Action.MOVE_OUT_OF_WAY:
                self.do_move_out_of_way(object_name, reasoning, flags)
                return object_name
            case c.Action.MOVE_TO_COUNTER_TOP:
                self.do_move_to_countertop(object_name, reasoning, flags)
                return object_name
            case c.Action.PUT_AWAY:
                self.do_put_away(object_name, reasoning, flags)
                return object_name
            case c.Action.SINK_WASH:
                self.do_sink_wash(object_name, reasoning, flags)
                return object_name
            case c.Action.WASH:
                self.do_wash(object_name, reasoning, flags)
                return object_name
            case c.Action.EMPTY_HAND:
                holding_object = self.holding_obj_name()
                self.do_empty_hand(object_name, reasoning, flags)
                return holding_object
            case c.Action.EMPTY_LIQUID:
                self.do_empty_liquid(object_name, reasoning, flags)
                return object_name
            case c.Action.DRINK:
                # Same as EMPTY_LIQUID but different label for reasoning
                self.do_drink(object_name, reasoning, flags)
                return object_name
            case c.Action.BREW_AND_DRINK:
                self.do_brew_and_drink(object_name, reasoning, flags)
                return object_name
            case c.Action.BREW:
                self.do_brew(object_name, reasoning, flags)
                return object_name
            case c.Action.SPRAY:
                self.do_spray(object_name, reasoning, flags)
                return object_name
            case c.Action.COOK:
                return self.do_cook(object_name, reasoning, flags)
            case c.Action.SERVE:
                self.do_serve(object_name, reasoning, flags)
                return object_name
            case c.Action.SLICE:
                self.do_slice(object_name, reasoning, flags)
                return object_name
            case c.Action.CLEAN:
                self.do_clean(object_name, reasoning, flags)
                return object_name

        raise Exception(f"Cannot do action {action_type} on {object_name}")

    def empty_water_from_containers(self):
        """
        After turning off a water source, empty any containers that still have water in them
        During testing I need to remove water from containers after turning off the sink
        """

        # Containers that are washed may have water in them
        containers = ["Bowl", "Cup", "Mug", "Pot", "Pan", "Plate"]

        # Now look for objects of container type
        objects = [obj for obj in self.all_objects() if obj["objectType"] in containers]
        for obj in objects:
            if obj["isFilledWithLiquid"] and obj["fillLiquid"] == "water":
                self.agent_empty(obj["name"])

    def restore_crouch_and_vert(self, was_crouched, camera_horizon):
        # Restore previous crouch state
        if was_crouched and not self.is_crouched():
            self.controller.step(action="Crouch")
        elif not was_crouched and self.is_crouched():
            self.controller.step(action="Stand")

        # Restore camera horizon
        self.look_vert(camera_horizon)

    def _do_manual_action(self, action_type, object_name, reasoning, flags):

        # Special case for FIND as it handles facing on its own
        if action_type == c.Action.FIND:

            try:
                # If object is already sliced, it doesn't exist in scene
                if self.is_sliced(object_name):
                    raise AgentCantDo(
                        "Object is already sliced", StepErrorType.INVALID_OBJECT
                    )

                # If item is in a container, face it instead
                if self.is_in_container(object_name):
                    container_obj = self.get_non_surface_container(object_name)
                    container_name = container_obj["name"]
                    self.execute_action(container_name, self.agent_find)

                    # We need to automate drawers and cabinets as the agent can't know which one to open
                    # so if the object is in a closed drawer or cabinet, open it
                    closed_container = self.get_closed_drawer_or_cabinet(object_name)
                    if closed_container:
                        self.execute_action(closed_container, self.agent_open)

                else:
                    self.execute_action(object_name, self.agent_find)
            except AgentFailure as e:
                # Allow failed find.  Not the fault of the agent
                Utils.print_color(
                    c.Color.YELLOW,
                    f"FIND action failed for {object_name}: {str(e)}",
                )

            return object_name

        # Set crouch and Face object vertically before testing visibility in can_do
        # Skip if has already been sliced as object no longer in the scene
        if object_name is not None and not self.is_sliced(object_name):

            # Save current pose to restore later, if still not visible
            is_crouched = self.is_crouched()
            camera_horizon = self.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]

            # Now crouch and look vertically
            self.agent_set_crouch(object_name)
            self.look_vert_at_object(object_name)

            # Restore previous crouch state if still not visible and action is not FIND
            if not self.is_visible(object_name):
                self.restore_crouch_and_vert(is_crouched, camera_horizon)

        error, error_type = self.can_do_manual_action(
            action_type, object_name, reasoning, flags
        )
        if error is not None:
            # Get closer to the object if possible so agent can see why failed (i.e. sink is full)
            if self.is_visible(object_name):
                self.approach_object_after_manual_failure(object_name)

            # Raise AgentCantDo so do_candidates can try other objects
            raise AgentCantDo(error, error_type)

        # Get in right position for the action
        if object_name is not None:
            # Face the object first
            self.agent_face_object(object_name)

            # Set crouch if needed
            self.agent_set_crouch(object_name)

            # Then face vertically again after crouching
            self.look_vert_at_object(object_name)

        match (action_type):
            case c.Action.PUT:
                self.execute_action(object_name, self.agent_put)
                return object_name
            case c.Action.FACE:
                self.do_face(object_name, reasoning, flags)
                return object_name
            case c.Action.TOGGLE_ON:
                # Object may have separate controller
                controller_name = self.get_controller_name(object_name)

                # Make sure controller is visible, burner, faucet could be but not the knob
                self.agent_face_object(controller_name)
                self.execute_action(controller_name, self.agent_toggle_on)
                return object_name
            case c.Action.TOGGLE_OFF:
                # Object may have separate controller
                controller_name = self.get_controller_name(object_name)

                # Make sure controller is visible, burner, faucet could be but not the knob
                self.agent_face_object(controller_name)

                self.execute_action(controller_name, self.agent_toggle_off)

                # Sometimes washed objects have liquid still in them
                # which should be emptied (not part of agent task)
                self.empty_water_from_containers()

                return object_name
            case c.Action.CLOSE:
                self.execute_action(object_name, self.agent_close)
                # Clear last opened container tracking if closing the tracked container
                obj = self.get_obj_by_name(object_name)
                obj_type = obj["objectType"]
                if (
                    obj_type in self.last_opened_container
                    and self.last_opened_container[obj_type] == object_name
                ):
                    del self.last_opened_container[obj_type]
                    print(
                        f"[DEBUG] _do_manual CLOSE: Cleared last opened {obj_type}: {object_name}"
                    )
                return object_name
            case c.Action.PICKUP:
                self.execute_action(object_name, self.agent_pickup)
                return object_name
            case c.Action.OPEN:
                self.execute_action(object_name, self.agent_open)

                # Track the last opened container for prioritization
                obj = self.get_obj_by_name(object_name)
                obj_type = obj["objectType"]
                if obj_type in self.OPENABLE_CONTAINER_TYPES:
                    self.last_opened_container[obj_type] = object_name
                    print(
                        f"[DEBUG] _do_manual OPEN: Tracking last opened {obj_type}: {object_name}"
                    )

                # After an open, move agent to PUT position so they have a
                # better view of the opened object
                pose = PlacementCache.get_interaction_pose(
                    self.raw_plan.scene, object_name, c.Action.PUT
                )
                if pose is not None:
                    self.agent_teleport(pose, object_name)

                return object_name
            case c.Action.EMPTY_LIQUID:
                self.agent_empty(object_name)
                return object_name
            case c.Action.DRINK:
                self.reached_new_subgoal = True
                # Same as EMPTY_LIQUID but different label for reasoning
                self.raw_plan.goal.set_action_goal(ActionGoal.DRINK_COFFEE, True)
                self.agent_empty(object_name)
                return object_name
            case c.Action.BREW_AND_DRINK:
                self.reached_new_subgoal = True
                self.agent_empty(object_name)
                return object_name
            case c.Action.SPRAY:
                self.reached_new_subgoal = True
                self.agent_face_object(object_name)
                self.raw_plan.goal.set_action_goal(ActionGoal.SPRAY_MIRROR, True)
                return object_name
            case c.Action.SLICE:
                self.reached_new_subgoal = True
                self.execute_action(object_name, self.agent_slice)
                return object_name
            case c.Action.CLEAN:
                self.reached_new_subgoal = True
                self.execute_action(object_name, self.agent_clean)
                return object_name
            case c.Action.TURN_LEFT:
                self.agent_turn_left("", admin_task=True)
                return object_name
            case c.Action.TURN_RIGHT:
                self.agent_turn_right("", admin_task=True)
                return object_name
            case c.Action.TURN_UP:
                self.agent_turn_up("", admin_task=True)
                return object_name
            case c.Action.TURN_DOWN:
                self.agent_turn_down("", admin_task=True)
                return object_name
            case c.Action.MOVE_FORWARD:
                self.controller.step(action="MoveAhead")
                return object_name
            case c.Action.MOVE_BACKWARD:
                self.controller.step(action="MoveBack")
                return object_name
            case c.Action.DIRTY:
                self.agent_dirty(object_name, admin_task=True)
                return object_name

        raise Exception(f"Cannot do action {action_type} on {object_name}")

    def can_do_manual_action(
        self, action_type, object_name, reasoning, flags
    ) -> Optional[Tuple[str, StepErrorType]]:

        if action_type == c.Action.INVALID_OBJECT:
            return "Object is not a valid object.", StepErrorType.INVALID_OBJECT
        elif action_type == c.Action.INVALID_ACTION:
            return "Action is not a valid action.", StepErrorType.INVALID_ACTION
        elif action_type == c.Action.INVALID_RESPONSE:
            return (
                "Response from model is in invalid format.",
                StepErrorType.INVALID_RESPONSE,
            )

        if object_name is None:
            return "No object name provided.", StepErrorType.INVALID_OBJECT

        obj = self.get_obj_by_name(object_name, must_exist=False)
        if obj is None:
            return (
                f"Object {object_name} does not exist in this scene.",
                StepErrorType.INVALID_OBJECT,
            )

        match (action_type):
            case c.Action.PUT:
                held_object_name = self.holding_obj_name()
                if held_object_name is None:
                    return ("No object held.", StepErrorType.UNDOABLE)

                if not self.is_visible(object_name):
                    return (
                        f"Cannot PUT in {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if self.is_dirty(object_name):
                    return (
                        f"Cannot PUT in {object_name} as it is dirty.",
                        StepErrorType.UNDOABLE,
                    )
                if self.is_openable(object_name) and self.is_closed(object_name):
                    return (
                        f"Cannot PUT in {object_name} as it is closed. Open it first.",
                        StepErrorType.UNDOABLE,
                    )

                compatible = self.is_compatible(held_object_name, object_name)
                if not compatible:
                    return (
                        f"Cannot PUT {held_object_name} in {object_name} as they are not compatible.",
                        StepErrorType.UNDOABLE,
                    )
                return None, None
            case c.Action.FIND:
                # Failure cases handled automatically
                return None, None
            case c.Action.FACE:
                # Failure cases handled automatically
                return None, None
            case c.Action.TOGGLE_ON:
                obj = self.get_obj_by_name(object_name)
                if obj["isToggled"]:
                    return (
                        f"Cannot toggle {object_name} as it is already on.",
                        StepErrorType.UNDOABLE,
                    )

                if not self.is_visible(object_name):
                    return (
                        f"Cannot toggle {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if "Television" in object_name:
                    holding_name = self.holding_obj_name()
                    if holding_name is not None and "RemoteControl" in holding_name:
                        return None, None
                    else:
                        return (
                            f"Cannot toggle {object_name} without holding a remote control.",
                            StepErrorType.UNDOABLE,
                        )

                if "CoffeeMachine" in object_name:
                    sink_contents = self.get_container_contents(object_name)
                    if not sink_contents:
                        return (
                            f"Cannot toggle {object_name} as it contains no objects.",
                            StepErrorType.UNDOABLE,
                        )

                    container = sink_contents[0]
                    if container["objectType"] != "Mug":
                        return (
                            f"Cannot toggle {object_name} as it contains a {container['objectType']}.",
                            StepErrorType.UNDOABLE,
                        )
                    if container["isFilledWithLiquid"] is True:
                        return (
                            f"Cannot toggle {object_name} as the mug is not empty.",
                            StepErrorType.UNDOABLE,
                        )
                    if self.is_dirty(container["name"]):
                        return (
                            f"Cannot toggle {object_name} as the mug is dirty.",
                            StepErrorType.UNDOABLE,
                        )

                if "Faucet" in object_name:
                    sink_obj = self.get_obj_by_type("SinkBasin")
                    sink_contents = self.get_container_contents(sink_obj["name"])
                    if sink_contents is not None:
                        for contained_obj in sink_contents:
                            contained_type = contained_obj["objectType"]
                            if (
                                contained_type not in c.CLASS_TO_TYPES["Dishes"]
                                and contained_type not in c.CLASS_TO_TYPES["Cookware"]
                                and contained_type != "DishSponge"
                            ):
                                return (
                                    f"Cannot toggle {object_name} as SinkBasin contains a {contained_type}.",
                                    StepErrorType.UNDOABLE,
                                )

                if "Microwave" in object_name:
                    if self.is_open(object_name):
                        return (
                            f"Cannot toggle {object_name} as it is open.",
                            StepErrorType.UNDOABLE,
                        )

                return None, None

            case c.Action.TOGGLE_OFF:
                obj = self.get_obj_by_name(object_name)
                if not obj["isToggled"]:
                    return (
                        f"Cannot toggle {object_name} as it is already off.",
                        StepErrorType.UNDOABLE,
                    )

                if not self.is_visible(object_name):
                    return (
                        f"Cannot toggle {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if "Television" in object_name:
                    holding_name = self.holding_obj_name()
                    if holding_name is not None and "RemoteControl" in holding_name:
                        return None, None
                    else:
                        return (
                            f"Cannot toggle {object_name} without holding a remote control.",
                            StepErrorType.UNDOABLE,
                        )

                # TODO: what about faucets?
                return None, None

            case c.Action.CLOSE:
                if not self.is_visible(object_name):
                    return (
                        f"Cannot close {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if not self.is_openable(object_name):
                    return f"{object_name} is not openable.", StepErrorType.UNDOABLE
                if not self.is_open(object_name):
                    return f"{object_name} is already closed.", StepErrorType.UNDOABLE
                return None, None

            case c.Action.PICKUP:
                holding_name = self.holding_obj_name()
                if holding_name is not None:
                    return (
                        f"Cannot pick up {object_name} while holding {holding_name}.",
                        StepErrorType.UNDOABLE,
                    )
                if not self.is_visible(object_name):
                    return (
                        f"Cannot pick up {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if self.is_in_container(object_name):
                    container_obj = self.get_non_surface_container(object_name)
                    if container_obj is not None and self.is_closed(
                        container_obj["name"]
                    ):
                        return (
                            f"Cannot pick up {object_name} as it is in a closed container.",
                            StepErrorType.UNDOABLE,
                        )
                if not self.is_pickupable(object_name):
                    return (
                        f"You cannot pick up object of type {object_name} as it is not pickupable.",
                        StepErrorType.UNDOABLE,
                    )
                return None, None

            case c.Action.OPEN:
                if not self.is_visible(object_name):
                    return (
                        f"Cannot open {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                if not self.is_openable(object_name):
                    return f"{object_name} is not openable.", StepErrorType.UNDOABLE
                if self.is_open(object_name):
                    return f"{object_name} is already open.", StepErrorType.UNDOABLE
                return None, None

            case c.Action.EMPTY_LIQUID:
                if not self.is_visible(object_name):
                    return (
                        f"Cannot empty {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )

                if not self.is_filled(object_name):
                    return (
                        f"{object_name} is not filled with liquid.",
                        StepErrorType.UNDOABLE,
                    )

                return None, None

            case c.Action.DRINK:
                if not self.is_visible(object_name):
                    return (
                        f"Cannot drink from the {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                holding_name = self.holding_obj_name()
                if holding_name is None:
                    return (
                        f"Cannot drink from the {object_name} without holding it.",
                        StepErrorType.UNDOABLE,
                    )
                if not self.is_filled(object_name):
                    return (
                        f"{object_name} is not filled with liquid.",
                        StepErrorType.UNDOABLE,
                    )
                if "Mug" in holding_name:
                    # Force mug to be dirty
                    self.agent_dirty(object_name)
                return None, None

            case c.Action.SPRAY:
                if not self.is_visible(object_name):
                    return (
                        f"Cannot spray {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )
                holding_name = self.holding_obj_name()
                if holding_name is None or "SprayBottle" not in holding_name:
                    return (
                        f"Cannot spray {object_name} without holding a spray bottle.",
                        StepErrorType.UNDOABLE,
                    )
                return None, None

            case c.Action.SLICE:
                if not self.is_visible(object_name):
                    obj = self.get_obj_by_name(object_name)

                    # Handle removed objects (i.e. "Lettuce") when already sliced
                    if obj["isSliced"]:
                        return (
                            f"{object_name} is already sliced.",
                            StepErrorType.UNDOABLE,
                        )

                    # Handle trying to slice an object that is already a slice
                    if "Slice" in object_name:
                        return (
                            f"{object_name} is already sliced.",
                            StepErrorType.UNDOABLE,
                        )
                    return (
                        f"Cannot slice {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )

                obj = self.get_obj_by_name(object_name)
                if obj["objectType"] in c.FOOD_SLICE_ON_COUNTER_TOP:
                    if (
                        not self.is_on_countertop(object_name)
                        or obj["position"]["y"] < c.LOWER_SHELF_THRESHOLD
                    ):
                        return (
                            f"Cannot slice {object_name} as it is not on a countertop.",
                            StepErrorType.UNDOABLE,
                        )
                    holding_name = self.holding_obj_name()
                    if holding_name is None or "Knife" not in holding_name:
                        return (
                            f"Cannot slice {object_name} without holding a knife.",
                            StepErrorType.UNDOABLE,
                        )
                return None, None

            case c.Action.CLEAN:
                # Note: Ok for object not to be dirty, some get cleaned automatically by water being turned on
                if not self.is_visible(object_name):
                    return (
                        f"Cannot clean {object_name} as it is not visible.",
                        StepErrorType.UNDOABLE,
                    )

                holding_name = self.holding_obj_name()
                obj = self.get_obj_by_name(object_name)
                if "Mirror" in obj["objectType"]:
                    if not self.raw_plan.goal.did_action_goal(ActionGoal.SPRAY_MIRROR):
                        return (
                            f"Cannot clean {object_name} without having sprayed it first.",
                            StepErrorType.UNDOABLE,
                        )
                    if holding_name is None or "Cloth" not in holding_name:
                        return (
                            f"Cannot clean {object_name} without holding a cloth.",
                            StepErrorType.UNDOABLE,
                        )
                else:
                    faucet_obj = self.get_obj_by_type("Faucet")
                    if not faucet_obj or not faucet_obj["isToggled"]:
                        return (
                            f"Cannot clean {object_name} as the faucet is not turned on.",
                            StepErrorType.UNDOABLE,
                        )
                    container_obj = self.get_non_surface_container(
                        object_name, must_exist=False
                    )
                    if (
                        container_obj is None
                        or container_obj["objectType"] != "SinkBasin"
                    ):
                        return (
                            f"{object_name} must be in the SinkBasin to clean",
                            StepErrorType.UNDOABLE,
                        )
                    if holding_name is None or "Sponge" not in holding_name:
                        return (
                            f"Cannot clean {object_name} without holding a sponge.",
                            StepErrorType.UNDOABLE,
                        )
                return None, None

        raise Exception(f"{action_type} missing")

    def try_execute_action(
        self, object_name, agent_action, interaction_margin: float = 0
    ):
        obj = self.get_obj_by_name(object_name)
        holding_name = self.holding_obj_name()

        distance = self.distance_to_object(object_name)
        visible = self.is_visible(object_name)
        action_name = Utils.action_name_from_action(agent_action)

        within_interaction_range = self.within_interaction_range(
            object_name, distance, interaction_margin
        )

        # If I'm within reach distance, do the action
        if visible and within_interaction_range:
            # Capture image before executing the action, unless it's a find action
            if action_name != c.Action.FIND:
                self.capture_world_state()

            success = agent_action(object_name)
            if success:
                # Cache this working location
                agent = self.get_agent()
                PlacementCache.add_interaction_pose(
                    self.raw_plan.scene, obj, agent, agent_action
                )
                PlacementCache.add_location_pose(
                    self.raw_plan.scene, obj["position"], agent
                )

                # Note that put works with this object, and save to put_cache
                if action_name == "put":
                    DefectiveContainers.save(
                        self.raw_plan.scene, object_name, holding_name, True, self.hash
                    )
                    PutCache.add_put_pose(self.hash, holding_name, object_name, agent)
                return True

        return False

    def execute_action(self, object_name, agent_action):
        if self.raw_plan.task_failed:
            return

        action_name = Utils.action_name_from_action(agent_action)
        if action_name != c.Action.FIND:
            # Try to execute the action from where I am (unless find as want to approach)
            try:
                # Face the object first
                self.agent_face_object(object_name)

                # Set crouch if needed
                self.agent_set_crouch(object_name)

                success = self.try_execute_action(object_name, agent_action)
                if success:
                    return

                if self.was_impossible_action():
                    return

            except AgentFailure:
                # First failure is ok, try to appoach the object next
                pass

        self.num_tries = 0
        self.candidate_poses = None
        action_name = Utils.action_name_from_action(agent_action)
        while True:
            # First try to approach the object
            success = self.approach_and_act(object_name, agent_action)
            if success:
                break

            if self.was_impossible_action():
                break

            # If that fails try to teleport
            success = self.teleport_and_act(object_name, agent_action)
            if success:
                break

            if self.was_impossible_action():
                break

            self.num_tries += 1
            if self.num_tries > 5:
                raise AgentFailure(f"Max Retries {action_name} {object_name}")

            # If elapsed time is too long, give up
            elapsed_time = (datetime.now() - self.start_time).total_seconds()
            if elapsed_time > MAX_TIME:
                raise AgentFailure(f"Max time exceeded {action_name} {object_name}")

    def was_impossible_action(self) -> bool:
        """
        Examines the last retry error to determine if the action was impossible to perform due to the state of the environment.
        Returns True if the action was impossible, to indicate it's not worth retrying, otherwise returns False.
        """
        impossible_errors = [
            "NOT a receptacle",
            "openable Receptacle is CLOSED",
            "cannot be sliced",
        ]
        if any(error in self.last_retry_error for error in impossible_errors):
            return True
        return False

    def do_put(self, container_name: str, reasoning: str, flags) -> bool:
        held_object_name = self.holding_obj_name()
        if held_object_name is None:
            raise AgentFatal("PUT: No object held")

        compatible = self.is_compatible(held_object_name, container_name)
        if not compatible:
            raise AgentFailure(
                f"{Utils.short_name(held_object_name)} can't go inside {Utils.short_name(container_name)}"
            )

        # Remember what I was holding
        holding_name = self.holding_obj_name()

        # If emptying my hand, remember the container I used to prevent loops
        if c.Flags.HAND_EMPTY_PUT in flags:
            self.empty_hand_containers.append(container_name)

        step = self.step_add(
            c.Action.PUT,
            container_name,
            f"Put {Utils.short_name(held_object_name)} in {Utils.short_name(container_name)}",
            reasoning,
        )

        container_specifier = Specifier(names=[container_name])
        container_obj = self.get_obj_by_name(container_name)

        if container_obj is None:
            raise AgentFatal(f"Object names {container_name} not found in scene")

        # If container is on a shelf, make sure its on the top or move it up
        moved_to_countertop = False
        while True:
            if self.raw_plan.task_failed:
                return

            if not self.is_visible(container_name):
                # If the container is in another container that can be opened
                inside_obj = self.get_non_surface_container(
                    container_name, must_exist=False
                )
                if (
                    inside_obj is not None
                    and self.is_openable(inside_obj["name"])
                    and self.is_closed(inside_obj["name"])
                ):
                    # First face the container
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(container_name)} is not visible, so I need to find it first"
                    self.do(c.Action.FIND, container_specifier, reasoning)

                    # Then open the parent container
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(container_name)} is inside {inside_obj['name']} and it is closed, so I need to open it first"
                    self.do(
                        c.Action.OPEN, Specifier(names=[inside_obj["name"]]), reasoning
                    )
                    continue

            # If object is not visible, need to find it first
            if not self.is_visible(container_name):
                self.add_substep(step)
                reasoning = f"{Utils.short_name(container_name)} is not visible, so I need to find it first"
                self.do(c.Action.FIND, container_specifier, reasoning)
                continue

            # If the container is dirty clean it
            if self.is_dirty(container_name):
                self.add_substep(step)
                reasoning = f"If {Utils.short_name(container_name)} is dirty, I need to clean it first"
                self.do(
                    c.Action.SINK_WASH,
                    Specifier(names=[container_name]),
                    reasoning,
                    [c.Flags.OK_ALREADY_CLEAN, c.Flags.DONT_PUT_AWAY],
                )
                continue

            # If slicing on pan or putting in dishes, don't do on lower shelf
            is_dish = container_obj["objectType"] in c.CLASS_TO_TYPES["Dishes"]
            if (
                not moved_to_countertop
                and (c.Flags.SLICE_ON_PAN in flags or is_dish)
                and container_obj["position"]["y"] < c.LOWER_SHELF_THRESHOLD
            ):

                # Move container to the counter top
                self.add_substep(step)
                reasoning = f"{Utils.short_name(container_name)} is on a lower shelf and needs to be moved it to the counter top first"
                self.do(c.Action.MOVE_TO_COUNTER_TOP, container_specifier, reasoning)
                moved_to_countertop = True
                continue

            # If the container is inside another container, move it to the countertop
            if self.is_pickupable(container_name) and self.is_in_container(
                container_name
            ):

                inside_obj = self.get_non_surface_container(container_name)

                # Move container to the counter top
                self.add_substep(step)
                reasoning = f"{Utils.short_name(container_name)} is inside {Utils.short_name(inside_obj['name'])}, so I need to move it to the counter top first"
                self.do(c.Action.MOVE_TO_COUNTER_TOP, container_specifier, reasoning)
                continue

            # Am I supposed to clear the container first?
            if c.Flags.PUT_DONT_CLEAR not in flags:
                # If container is full, need to empty it first
                if self.need_to_empty(container_name, held_object_name):

                    # Make sure looking at container to see it is full
                    if not self.is_visible(container_name):
                        self.add_substep(step)
                        reasoning = f"{Utils.short_name(container_name)} is not visible, so I need to find it first"
                        self.do(c.Action.FIND, container_specifier, reasoning)

                    # Convert from objects_to_clear to a list of comma separated names
                    objects_to_clear = self.get_container_contents(container_name)
                    objects_to_clear_names = [obj["name"] for obj in objects_to_clear]
                    Specifier.observed_names.update(objects_to_clear_names)

                    self.add_substep(step)
                    reasoning = f"Need to clear the {Utils.short_name(container_name)} to put the {Utils.short_name(held_object_name)}"
                    self.do(c.Action.CLEAR_CONTAINER, container_specifier, reasoning)
                    continue

            # If the container is openable I need to open it
            if self.is_openable(container_name) and self.is_closed(container_name):
                self.add_substep(step)
                reasoning = f"{Utils.short_name(container_name)} is closed, so I need to open it first"
                self.do(c.Action.OPEN, container_specifier, reasoning)
                continue

            # May need to pick back up the held object
            now_holding_name = self.holding_obj_name()
            if now_holding_name != holding_name:
                self.add_substep(step)
                reasoning = f"I need to pick back up {Utils.short_name(holding_name)} to put it in {Utils.short_name(container_name)}"
                self.do(c.Action.PICKUP, Specifier(names=[holding_name]), reasoning)
                continue
            break

        # Now put the object in the container
        try:
            self.execute_action(container_name, self.agent_put)

            # Trying to bring the object into view after placing it
            self.try_bring_into_view(holding_name)

        except AgentFailure as e:
            # Remember failed put so can inject training errors and recovers
            filename = f"_{self.image_index}_{step.action_desc}_{c.FAIL}.png"
            fullname = f"{self.save_dir}/{filename}"

            self.cur_image.save(fullname)

            self.raw_plan.failed_steps.append(
                Step(
                    action_desc=step.action_desc,
                    action=step.action,
                    obj=step.object,
                    task_description=step.task_description,
                    observations=step.observations,
                    updated_memory={},
                    object_bounding_boxes=step.object_bounding_boxes,
                    pose=step.pose,
                    reasoning=[step.reasoning],
                    filename=filename,
                    history=[],
                )
            )
            raise e
        self.step_done(step)

        # If I opened the container, I need to close it after putting the item in
        if self.is_openable(container_name) and self.is_open(container_name):

            # Make sure I I'm facing the container
            self.do_face_object(holding_name)
            self.capture_world_state()

            self.add_substep(step)
            reasoning = f"{Utils.short_name(container_name)} is open, so I need to close it after putting the item in"
            self.do(c.Action.CLOSE, container_specifier, reasoning)

        # Cache the result
        PlacementCache.add_container(
            self.raw_plan.scene, held_object_name, container_name
        )
        self.step_complete(step)

        # Stand-up (in case I crouched to place item)
        if self.is_crouched():
            self.controller.step(action="Stand")

        return container_name

    def do_clear_container(self, container_name, reasoning, flags) -> None:
        step = self.step_add(
            c.Action.CLEAR_CONTAINER,
            container_name,
            f"Make space in {Utils.short_name(container_name)}",
            reasoning,
            include_in_substep=False,
        )

        # If object is not visible, need to find it first
        if not self.is_visible(container_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(container_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[container_name]), reasoning)

        while True:
            if self.raw_plan.task_failed:
                return None

            # Recalculate each loop as the object may have moved if contained in another
            # i.e. salt on plate with move when plate is moved
            objs = self.get_container_contents(container_name)

            if objs is None or len(objs) == 0:
                break

            # Remove objs that can't be picked up
            objs = [obj for obj in objs if self.is_pickupable(obj["name"])]

            # Remove objects that are in excluded_objects list
            objs = [
                obj
                for obj in objs
                if obj["objectType"] not in Specifier.excluded_objects
            ]

            if objs is None or len(objs) == 0:
                break

            self.add_substep(step)
            obj_name = objs[0]["name"]
            reasoning = f"Need to clear {Utils.short_name(obj_name)} from {Utils.short_name(container_name)}"
            Specifier.excluded_containers.append(container_name)
            self.do(c.Action.MOVE_OUT_OF_WAY, Specifier(names=[obj_name]), reasoning)

        self.step_done(step)
        self.step_complete(step)
        return None

    def do_move_out_of_way(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.MOVE_OUT_OF_WAY,
            object_name,
            f"Move {Utils.short_name(object_name)} out of way",
            reasoning,
            include_in_substep=False,
        )

        self.add_substep(step)
        reasoning = f"To move {Utils.short_name(object_name)} out of the way, I need to pick it up first"
        self.do(c.Action.PICKUP, Specifier(names=[object_name]), reasoning)

        # If I'm moving out of the way, I don't want to put it back in it's original location
        if object_name in self.object_metadata:
            self.object_metadata[object_name].start_location = None

        # If an item that the plan is addressing put in reachable space
        if self.is_action_object(object_name):
            container_types = Scenario.nonclosing_container_types()
            container_specifier = Specifier(types=container_types)
        # Otherwise put it away
        else:
            obj = self.get_obj_by_name(object_name)
            container_specifier = self.get_object_homes(obj)

        # If specifier not given as names, convert to names so we can remove excluded containers
        if container_specifier.names is None:
            all_objects = self.all_objects()
            container_specifier.convert_to_names(all_objects)

        # Now remove containers that aren't in Specifier.excluded_containers
        for exclude_name in Specifier.excluded_containers:
            if exclude_name in container_specifier.names:
                container_specifier.names.remove(exclude_name)
        Specifier.excluded_containers = []

        # If I'm moving something out of the way, don't move that's there to make space
        reasoning = f"To put {Utils.short_name(object_name)} away, I need to put it somewhere else"
        self.do(c.Action.PUT, container_specifier, reasoning, [c.Flags.PUT_DONT_CLEAR])

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_put_away(self, object_name, reasoning, flags):
        # If object is already in a container, skip
        if self.is_object_home(object_name):
            print(f"Object {object_name} is already in a container")
            return object_name

        step = self.step_add(
            c.Action.PUT_AWAY,
            object_name,
            f"Put {Utils.short_name(object_name)} away",
            reasoning,
            include_in_substep=False,
        )

        holding = self.holding_obj_name()
        if holding is None or holding != object_name:
            self.add_substep(step)
            reasoning = f"To put {Utils.short_name(object_name)} away, I need to pick it up first"
            self.do(c.Action.PICKUP, Specifier(names=[object_name]), reasoning)

        if c.Flags.CLEAN_WHEN_PUTTING_AWAY in flags:
            # If object is dirty, need to clean it first
            if self.is_dirty(object_name):
                self.add_substep(step)
                reasoning = f"If {Utils.short_name(object_name)} is dirty, I need to clean it first"
                self.do(c.Action.SINK_WASH, Specifier(names=[object_name]), reasoning)

            # Object should have been put away
            if self.is_object_home(object_name):
                return object_name

        # If I'm moving out of the way, I don't want to put it back in it's original location
        if object_name in self.object_metadata:
            self.object_metadata[object_name].start_location = None

        obj = self.get_obj_by_name(object_name)
        container_specifier = self.get_object_homes(obj)

        # Now remove containers that are full or too small
        candidate_containers = self.get_specifier_objects(container_specifier)
        container_names = []
        for container in candidate_containers:
            if self.not_occupied(container):
                container_names.append(container["name"])

        # Now remove containers that are full or too small
        secondary_containers = self.get_specifier_secondary_objects(container_specifier)
        secondary_names = []
        if secondary_containers is not None:
            for container in secondary_containers:
                if self.not_occupied(container):
                    secondary_names.append(container["name"])

        # Put objects away, but don't take other things out to make space
        self.add_substep(step)
        reasoning = f"To put {Utils.short_name(object_name)} away, I need to put it in a container"
        self.do(
            c.Action.PUT,
            Specifier(names=container_names, secondary_names=secondary_names),
            reasoning,
            [c.Flags.PUT_DONT_CLEAR],
        )

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_toggle_on(self, object_name, reasoning, flags) -> bool:
        step = self.step_add(
            c.Action.TOGGLE_ON,
            object_name,
            f"Turn on {Utils.short_name(object_name)}",
            reasoning,
        )

        # If tv need a remove
        if "Television" in object_name:
            self.add_substep(step)
            reasoning = f"Need RemoteControl to turn on {Utils.short_name(object_name)}"
            self.do(c.Action.PICKUP, Specifier(types=["RemoteControl"]), reasoning)

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        # Face the object first
        self.agent_face_object(object_name)
        self.capture_world_state()

        # I may be controlled by somthing else
        controller_name = self.get_controller_name(object_name)
        self.execute_action(controller_name, self.agent_toggle_on)
        self.step_done(step)
        self.step_complete(step)
        return controller_name

    def do_toggle_off(self, object_name, reasoning, flags) -> bool:
        step = self.step_add(
            c.Action.TOGGLE_OFF,
            object_name,
            f"Turn off {Utils.short_name(object_name)}",
            reasoning,
        )

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        # I may be controlled by somthing else
        controller_name = self.get_controller_name(object_name)
        self.execute_action(controller_name, self.agent_toggle_off)
        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_slice(self, object_name, reasoning, flags) -> bool:
        step = self.step_add(
            c.Action.SLICE,
            object_name,
            f"Slice {Utils.short_name(object_name)}",
            reasoning,
        )

        # If object is not on a countertop, need to put it there first
        obj = self.get_obj_by_name(object_name)
        if obj["objectType"] in c.FOOD_SLICE_ON_COUNTER_TOP:
            if (
                not self.is_on_countertop(object_name)
                or obj["position"]["y"] < c.LOWER_SHELF_THRESHOLD
            ):
                self.add_substep(step)
                reasoning = f"{Utils.short_name(object_name)} is not on a counter top, so I need to put it there first"
                self.do(
                    c.Action.MOVE_TO_COUNTER_TOP,
                    Specifier(names=[object_name]),
                    reasoning,
                )

            # If I'm not holding a knife, need to pick it up first
            holding_object = self.holding_obj_name()
            if holding_object is None or holding_object["objectType"] != "Knife":
                self.add_substep(step)
                reasoning = f"To slice {Utils.short_name(object_name)}, I need to pick up a knife first"
                knife_specifier = Specifier(types=["Knife"])
                self.do(c.Action.PICKUP, knife_specifier, reasoning)

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        self.execute_action(object_name, self.agent_slice)
        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_clean(self, object_name, reasoning, flags) -> bool:
        step = self.step_add(
            c.Action.CLEAN,
            object_name,
            f"Clean {Utils.short_name(object_name)}",
            reasoning,
        )

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        try:
            self.execute_action(object_name, self.agent_clean)
        except Exception as e:
            # Eat error if ok for object o be already clean
            if (not c.Flags.OK_ALREADY_CLEAN in flags) or (
                "already Clean" not in self.last_retry_error
            ):
                raise e
            else:
                # Reuse the last image
                image = self.previous_image

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_spray(self, surface_name, reasoning, flags) -> bool:
        step = self.step_add(
            c.Action.SPRAY,
            surface_name,
            f"Spray {Utils.short_name(surface_name)}",
            reasoning,
        )

        spray_type = "SprayBottle"

        self.add_substep(step)
        reasoning = f"To spray {Utils.short_name(surface_name)}, I need to pickup a spray bottle first"
        spray_name = self.do(c.Action.PICKUP, Specifier(types=[spray_type]), reasoning)

        # If object is not visible, need to find it first
        if not self.is_visible(surface_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(surface_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[surface_name]), reasoning)

        # Face the object first
        self.agent_face_object(surface_name)

        # Not using Exectute Action so, capture image here
        self.capture_world_state()

        # TODO: Note always assuming mirror for now as only thing sprayed
        self.raw_plan.goal.set_action_goal(ActionGoal.SPRAY_MIRROR, True)

        self.step_done(step)
        self.step_complete(step)
        return surface_name

    def do_close(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.CLOSE,
            object_name,
            f"Close {Utils.short_name(object_name)}",
            reasoning,
        )

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        self.execute_action(object_name, self.agent_close)

        # Clear last opened container tracking if closing the tracked container
        obj = self.get_obj_by_name(object_name)
        obj_type = obj["objectType"]
        if (
            obj_type in self.last_opened_container
            and self.last_opened_container[obj_type] == object_name
        ):
            del self.last_opened_container[obj_type]
            print(f"[DEBUG] do_close: Cleared last opened {obj_type}: {object_name}")

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_empty_hand(self, blocked_obj, reasoning, flags):
        step = self.step_add(
            c.Action.EMPTY_HAND, "", f"Empty Hand", reasoning, include_in_substep=False
        )

        object_name = self.holding_obj_name()
        obj = self.get_obj_by_name(object_name)
        self.add_substep(step)
        reasoning = (
            f"To empty my hand, I need to put {Utils.short_name(object_name)} down"
        )

        empty_hand_specifier = self.get_empty_hand_specifier(obj, blocked_obj)

        # Set flag to remember the container I'm tryin to prevent loops
        self.do(c.Action.PUT, empty_hand_specifier, reasoning, [c.Flags.HAND_EMPTY_PUT])
        self.empty_hand_containers.pop()  # Remove last empty hand container

        self.step_done(step)
        self.step_complete(step)

        return object_name

    def do_pickup(self, pickup_name: str, reasoning, flags):
        step = self.step_add(
            c.Action.PICKUP,
            pickup_name,
            f"Pickup {Utils.short_name(pickup_name)}",
            reasoning,
        )

        pickup_specifier = Specifier(names=[pickup_name])
        pickup_obj = self.get_obj_by_name(pickup_name)

        # Potentially Inject an error
        if self.should_inject_error(InjectionType.PICKUP_RANDOM):
            # Find a random object to pickup
            holdable_names = self.random_holdable_names()
            holdable_specifier = Specifier(names=holdable_names, observe=False)
            self.add_substep(step)
            # Note: Don't sort items, use in random order from above
            holdable_name = self.do(
                c.Action.PICKUP,
                holdable_specifier,
                f"{InjectionType.PICKUP_RANDOM.value} INJECTION",
                [c.Flags.DONT_SORT],
            )
            self.injection_state = c.InjectionState.POST_INJECTION
            self.raw_plan.randomization.error_injection.object_name = holdable_name

            # Now recover from that error by emptying my hand
            self.add_substep(step)
            reasoning = f"To pick up {Utils.short_name(pickup_name)}, I need to empty my hand first"
            self.do(c.Action.EMPTY_HAND, pickup_specifier, reasoning)

        find_count = 0
        while True:
            if self.raw_plan.task_failed:
                return

            # If something is in my hand, put it down first
            holding_name = self.holding_obj_name()
            if holding_name is not None and holding_name != pickup_name:
                self.add_substep(step)
                reasoning = f"To pick up {Utils.short_name(pickup_name)}, I need to empty my hand first"
                self.do(c.Action.EMPTY_HAND, pickup_specifier, reasoning)
                continue

            # If item is in a container, need to open it first
            container_name = None
            container_obj = self.get_non_surface_container(
                pickup_name, must_exist=False
            )
            if container_obj is not None:
                # Save original location of object, so we can put it back later
                if pickup_name not in self.object_metadata:
                    metadata = ObjectMetadata(container_obj["name"])
                    self.object_metadata[pickup_name] = metadata

                container_name = container_obj["name"]
                if self.is_closed(container_name):
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(pickup_name)} is in the {Utils.short_name(container_name)} and it is closed, so I need to open it first"
                    self.do(c.Action.OPEN, Specifier(names=[container_name]), reasoning)
                    continue

            # Potentially Inject an error
            if self.should_inject_error(InjectionType.PICKUP_NOT_VISIBLE):
                # If object is visible, I need to look away
                while self.is_visible(pickup_name):
                    self.agent_move_action("RotateLeft")

                self.capture_world_state()
                self.step_done(step)
                self.step_complete(step)
                self.add_substep(step)
                step = self.step_add(
                    c.Action.PICKUP,
                    pickup_name,
                    f"Pickup {Utils.short_name(pickup_name)}",
                    reasoning,
                    include_in_substep=False,
                )

                self.injection_state = c.InjectionState.POST_INJECTION
                self.raw_plan.randomization.error_injection.object_name = pickup_name

                # Now find the object
                self.add_substep(step)
                reasoning = f"I failed to pick up {Utils.short_name(pickup_name)} because it isn't visible. I need to find it first."
                self.do(c.Action.FIND, pickup_specifier, reasoning)

            # If object is not visible, need to find it first
            if not self.is_visible(pickup_name) and find_count < 10:
                find_count += 1

                # If type is visible but not object, assume being used for a goal (i.e. a slice)
                obj_type = self.get_obj_by_name(pickup_name)["objectType"]
                if self.is_type_visible(obj_type):
                    reasoning = f"Visible {Utils.short_name(pickup_name)} is being used to meet an objective, so I need to find a different one"
                else:
                    reasoning = f"{Utils.short_name(pickup_name)} is not visible, so I need to find it first"

                self.add_substep(step)
                self.do(c.Action.FIND, pickup_specifier, reasoning)

                # If in sink basin may never be visible so only try finding it once
                if container_obj is None or container_obj["objectType"] != "SinkBasin":
                    continue

            # Am I supposed to clear the container first?
            if c.Flags.PICKUP_CLEAR in flags:
                # If object is full, need to empty it first
                if self.need_to_empty(pickup_name):
                    # Make sure looking at container to see it is full
                    if not self.is_visible(pickup_name):
                        self.add_substep(step)
                        reasoning = f"{Utils.short_name(pickup_name)} is not visible, so I need to find it first"
                        self.do(c.Action.FIND, pickup_specifier, reasoning)

                    # Convert from objects_to_clear to a list of comma separated names
                    objects_to_clear = self.get_container_contents(pickup_name)
                    objects_to_clear_names = [obj["name"] for obj in objects_to_clear]
                    Specifier.observed_names.update(objects_to_clear_names)

                    self.add_substep(step)
                    reasoning = f"Need to clear the {Utils.short_name(pickup_name)} before picking up"
                    self.do(c.Action.CLEAR_CONTAINER, pickup_specifier, reasoning)
                    continue

            blocking_obj = self.is_item_blocking(pickup_name, holding_name)
            if blocking_obj is not None and blocking_obj["pickupable"] is True:
                self.add_substep(step)
                reasoning = f"Item {Utils.short_name(blocking_obj['name'])} is in the way, so I need to move it out of the way first"
                self.do(
                    c.Action.MOVE_OUT_OF_WAY,
                    Specifier(names=[blocking_obj["name"]]),
                    reasoning,
                )
                continue

            break

        self.execute_action(pickup_name, self.agent_pickup)
        self.step_done(step)

        if (
            container_name is not None
            and self.is_openable(container_name)
            and self.is_open(container_name)
        ):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(container_name)} is open, so I need to close it after picking up {Utils.short_name(pickup_name)}"
            self.do(c.Action.CLOSE, Specifier(names=[container_name]), reasoning)

        self.step_complete(step)
        return pickup_name

    def do_open(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.OPEN,
            object_name,
            f"Open {Utils.short_name(object_name)}",
            reasoning,
        )

        # Potentially Inject an error
        if self.should_inject_error(InjectionType.OPEN_RANDOM):
            # Find a random object to pickup
            openable_names = self.random_openable_names()
            openable_specifier = Specifier(names=openable_names, observe=False)
            self.add_substep(step)
            openable_name = self.do(
                c.Action.OPEN,
                openable_specifier,
                f"{InjectionType.OPEN_RANDOM.value} INJECTION",
            )
            self.injection_state = c.InjectionState.POST_INJECTION
            self.raw_plan.randomization.error_injection.object_name = openable_name

            # Now recover from that error by emptying my hand
            self.add_substep(step)
            reasoning = f"{Utils.short_name(openable_name)} has no reason to be open, so I should close it first"
            self.do(c.Action.CLOSE, Specifier(names=[openable_name]), reasoning)

        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        self.execute_action(object_name, self.agent_open)

        # Track the last opened container for prioritization
        obj = self.get_obj_by_name(object_name)
        obj_type = obj["objectType"]
        if obj_type in self.OPENABLE_CONTAINER_TYPES:
            self.last_opened_container[obj_type] = object_name
            print(f"[DEBUG] do_open: Tracking last opened {obj_type}: {object_name}")

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def replay_caputure_action(self, action_type, action_desc, object_name, reasoning):
        observations = self.get_observations()
        object_bounding_boxes = self.get_object_bounding_boxes()
        pose = self.get_agent_pose()

        self.cur_image = Image.fromarray(self.controller.last_event.frame)
        image_filename = self.save_image(action_desc)
        filename = image_filename
        history = "todo"

        step = Step(
            action_desc=action_desc,
            action=action_type,
            obj=object_name,
            task_description="TODO",
            observations=observations,
            updated_memory=[],  # TODO
            object_bounding_boxes=object_bounding_boxes,
            pose=pose,
            reasoning=reasoning,
            filename=filename,
            history=history,
        )
        return step

    def replay_find_by_turning(self, object_name, reasoning) -> List[Step]:

        # TODO look neutral first
        new_steps: List[Step] = []
        reasoning = (
            f"I can't see {Utils.short_name(object_name)}, so I ned to look for it"
        )
        search_sequence = ["down", "turn", "up", "up", "turn", "down"]
        self.rotation_speed_horiz = 30
        self.rotation_speed_vert = 15
        while not self.is_visible(object_name):

            # Get first item off search sequence, add to end
            search_direction = search_sequence.pop(0)
            search_sequence.append(search_direction)

            if search_direction == "down":
                action = c.Action.TURN_DOWN
                action_desc = "Look down"
            elif search_direction == "up":
                action = c.Action.TURN_UP
                action_desc = "Look up"
            elif search_direction == "turn":
                turn_dir = self.agent_turn_direction(object_name, threshold=5)
                if turn_dir == c.TurnDirection.RIGHT:
                    action = c.Action.TURN_RIGHT
                    action_desc = "Turn right"
                elif turn_dir == c.TurnDirection.LEFT:
                    action = c.Action.TURN_LEFT
                    action_desc = "Turn left"
                else:
                    # TODO: something in the way need to take a diff action
                    break

            step = self.replay_caputure_action(
                action, action_desc, object_name, reasoning
            )
            new_steps.append(step)
            self._do_manual_action(action, None, reasoning, [])

        return new_steps

    def do_find(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.FIND,
            object_name,
            f"Find {Utils.short_name(object_name)}",
            reasoning,
        )

        # Potentially Inject an error
        if self.should_inject_error(InjectionType.PICKUP_RANDOM):
            # Find a random object to pickup
            holdable_names = self.random_holdable_names()
            holdable_specifier = Specifier(names=holdable_names, observe=False)
            self.add_substep(step)
            # Note: Don't sort items, use in random order from above
            holdable_name = self.do(
                c.Action.PICKUP,
                holdable_specifier,
                f"{InjectionType.PICKUP_RANDOM.value} PICKUP_RANDOM",
                [c.Flags.DONT_SORT],
            )
            self.injection_state = c.InjectionState.POST_INJECTION
            self.raw_plan.randomization.error_injection.object_name = holdable_name

            # Now recover from that error by emptying my hand
            self.add_substep(step)
            reasoning = f"To pick up {Utils.short_name(object_name)}, I need to empty my hand first"
            self.do(c.Action.EMPTY_HAND, Specifier(names=[object_name]), reasoning)

        # Capture world state before facing object
        self.capture_world_state()
        self.do_face_object(object_name)

        # If item is in a container, face it instead
        if self.is_in_container(object_name):
            container_obj = self.get_non_surface_container(object_name)
            object_name = container_obj["name"]

        self.execute_action(object_name, self.agent_find)
        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_empty_liquid(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.EMPTY_LIQUID,
            object_name,
            f"Empty {Utils.short_name(object_name)}",
            reasoning,
        )

        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = (
                f"To empty the {Utils.short_name(object_name)}, I need to find it first"
            )
            object_name = self.do(
                c.Action.FIND, Specifier(names=[object_name]), reasoning
            )

        contents = self.liquid_contents(object_name)

        # Not using Exectute Action so, capture image here
        self.capture_world_state()
        success = self.agent_empty(object_name)
        if not success:
            raise AgentFailure(f"Cannot empty the {object_name}")
        elif contents == c.LiquidType.COFFEE:
            self.raw_plan.goal.set_action_goal(ActionGoal.DRINK_COFFEE, True)

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_drink(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.DRINK,
            object_name,
            f"Drink {Utils.short_name(object_name)}",
            reasoning,
        )

        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"To drink from the {Utils.short_name(object_name)}, I need to find it first"
            object_name = self.do(
                c.Action.FIND, Specifier(names=[object_name]), reasoning
            )

        contents = self.liquid_contents(object_name)

        # Not using Exectute Action so, capture image here
        self.capture_world_state()
        success = self.agent_empty(object_name)
        if not success:
            raise AgentFailure(f"Cannot drink from the {object_name}")
        elif contents == c.LiquidType.COFFEE:
            self.raw_plan.goal.set_action_goal(ActionGoal.DRINK_COFFEE, True)

        self.step_done(step)
        self.step_complete(step)
        return object_name

    def do_face(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.FACE,
            object_name,
            f"Face {Utils.short_name(object_name)}",
            reasoning,
        )

        self.agent_face_object(object_name)
        self.capture_world_state()

        self.step_done(step)
        self.step_complete(step)
        return True

    def do_move_to_countertop(self, object_name, reasoning, flags):

        step = self.step_add(
            c.Action.MOVE_TO_COUNTER_TOP,
            object_name,
            f"Move {Utils.short_name(object_name)} to counter top",
            reasoning,
            include_in_substep=False,
        )

        self.add_substep(step)
        reasoning = f"To move {Utils.short_name(object_name)} to the counter top, I need to pick it up first"
        self.do(
            c.Action.PICKUP,
            Specifier(names=[object_name]),
            reasoning,
            [c.Flags.PICKUP_CLEAR],
        )

        self.add_substep(step)
        reasoning = f"To put {Utils.short_name(object_name)} on the counter top, I need to find a counter top first"
        counter_top_name = self.do(
            c.Action.PUT,
            Specifier(types=["CounterTop", "DiningTable", "SideTable"]),
            reasoning,
        )

        self.step_done(step)
        self.step_complete(step)

    def do_serve(self, food_name, reasoning, flags):

        step = self.step_add(
            c.Action.SERVE,
            food_name,
            f"Serve {Utils.short_name(food_name)}",
            reasoning,
            include_in_substep=False,
        )
        food_specifier = Specifier(names=[food_name])
        food_obj = self.get_obj_by_name(food_name)

        # What does this type of food get served in?
        serve_type = c.SERVE_CONTAINER[food_obj["objectType"]]
        serve_name = self.get_obj_by_type(serve_type)["name"]
        serve_specifier = Specifier(names=[serve_name])

        while True:
            if self.raw_plan.task_failed:
                return

            # If not visible and inside something
            if not self.is_visible(serve_name):
                # If the container is in another container that can be opened
                inside_obj = self.get_non_surface_container(
                    serve_name, must_exist=False
                )
                if (
                    inside_obj is not None
                    and self.is_openable(inside_obj["name"])
                    and self.is_closed(inside_obj["name"])
                    and self.is_closed(inside_obj["name"])
                ):
                    # First face the container
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(serve_name)} is not visible, so I need to find it first"
                    self.do(c.Action.FIND, serve_specifier, reasoning)

                    # Then open the parent container
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(serve_name)} is inside {inside_obj['name']} and it is closed, so I need to open it first"
                    self.do(
                        c.Action.OPEN, Specifier(names=[inside_obj["name"]]), reasoning
                    )
                    continue

            # If no longer visible, need to find it first
            if not self.is_visible(serve_name):
                self.add_substep(step)
                reasoning = f"{Utils.short_name(serve_name)} is not visible, so I need to find it first"
                self.do(c.Action.FIND, serve_specifier, reasoning)
                continue

            # Make sure the plate doesn't already contain anything
            if self.need_to_empty(serve_name):
                # Convert from objects_to_clear to a list of comma separated names
                objects_to_clear = self.get_container_contents(serve_name)
                objects_to_clear_names = [obj["name"] for obj in objects_to_clear]
                Specifier.observed_names.update(objects_to_clear_names)
                self.add_substep(step)
                reasoning = f"Need to clear the {Utils.short_name(serve_name)} to serve {Utils.short_name(food_name)}"
                self.do(c.Action.CLEAR_CONTAINER, serve_specifier, reasoning)
                continue

            # Make sure plate is on the counter top
            if not self.is_on_countertop(serve_name):
                self.add_substep(step)
                reasoning = f"{Utils.short_name(serve_name)} is not on the counter top, so I need to put it there first"
                self.do(c.Action.MOVE_TO_COUNTER_TOP, serve_specifier, reasoning)
                continue

            # If I'm not holding the food item pick it up
            holding = self.holding_obj_name()
            if holding is None or holding != food_name:
                self.add_substep(step)
                reasoning = f"To put {Utils.short_name(food_name)} on {Utils.short_name(serve_name)}, I need to pick it up first"
                self.do(c.Action.PICKUP, food_specifier, reasoning)
                break

        # Put the food on the plate
        self.add_substep(step)
        reasoning = f"To serve {Utils.short_name(food_name)} I need to put it on {Utils.short_name(serve_name)}"
        self.do(c.Action.PUT, serve_specifier, reasoning)

        # If scene does not have a dining table, terminate the plan early (still valid)
        have_dining_table = self.object_type_exists(["DiningTable"])
        if not have_dining_table:
            self.raw_plan.early_termiation = True
            return

        if not self.is_on_countertop(serve_name):
            # Move the food to the dining table
            self.add_substep(step)
            reasoning = f"{Utils.short_name(serve_name)} is not on the counter top, so I need to put it there first"
            food_name = self.do(c.Action.PICKUP, serve_specifier, reasoning)

            self.add_substep(step)
            reasoning = f"To serve {Utils.short_name(food_name)} I need to put on the dining table"
            serve_name = self.do(
                c.Action.PUT, Specifier(types=["DiningTable"]), reasoning
            )

    def do_wash(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.WASH,
            object_name,
            f"Wash {Utils.short_name(object_name)}",
            reasoning,
            include_in_substep=False,
        )

        # TODO: select thse based on the object type
        sponge_type = "Cloth"

        # If object is not visible, need to find it first
        if not self.is_visible(object_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
            self.do(c.Action.FIND, Specifier(names=[object_name]), reasoning)

        self.add_substep(step)
        reasoning = f"To wash {Utils.short_name(object_name)}, I need to spray it with a cleaning solution"
        cloth_name = self.do(c.Action.SPRAY, Specifier(names=[object_name]), reasoning)

        self.add_substep(step)
        reasoning = (
            f"To wash {Utils.short_name(object_name)}, I need a cloth to clean it"
        )
        cloth_name = self.do(c.Action.PICKUP, Specifier(types=[sponge_type]), reasoning)

        # Just putting in some sinks can clean objects automatically.  This is ok
        self.add_substep(step)
        reasoning = (
            f"To wash {Utils.short_name(object_name)}, I need to clean it with a cloth"
        )
        object_name = self.do(
            c.Action.CLEAN,
            Specifier(names=[object_name]),
            reasoning,
            [c.Flags.OK_ALREADY_CLEAN],
        )

        self.step_done(step)
        self.step_complete(step)

    def do_sink_wash(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.WASH,
            object_name,
            f"Wash {Utils.short_name(object_name)}",
            reasoning,
            include_in_substep=False,
        )

        sink_types = ["SinkBasin"]
        sponge_type = "DishSponge"
        object_specifier = Specifier(names=[object_name])

        while True:
            if self.raw_plan.task_failed:
                return

            # If object is full, need to empty it first
            if self.need_to_empty(object_name):

                # Make sure looking at object to see it is full
                if not self.is_visible(object_name):
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(object_name)} is not visible, so I need to find it first"
                    self.do(c.Action.FIND, object_specifier, reasoning)
                    continue

                # Convert from objects_to_clear to a list of comma separated names
                objects_to_clear = self.get_container_contents(object_name)
                Specifier.observed_names.update(
                    [obj["name"] for obj in objects_to_clear]
                )

                self.add_substep(step)
                reasoning = f"Before washing, I need to clear the {Utils.short_name(object_name)}"
                self.do(c.Action.CLEAR_CONTAINER, object_specifier, reasoning)
                continue

            # Check that item is the sink
            container_obj = self.get_non_surface_container(
                object_name, must_exist=False
            )
            sink_name = container_obj["name"] if container_obj is not None else None
            if sink_name is None or "SinkBasin" not in sink_name:

                # If I'm not holding the object, pick it up
                holding = self.holding_obj_name()
                if holding is None or holding != object_name:
                    self.add_substep(step)
                    reasoning = f"To wash {Utils.short_name(object_name)}, I need to pick it up first"
                    self.do(c.Action.PICKUP, object_specifier, reasoning)

                container_specifier = Specifier(types=sink_types)

                # Put it in the sink
                self.add_substep(step)
                reasoning = f"To wash {Utils.short_name(object_name)}, I need to put it in a sink first"
                sink_name = self.do(c.Action.PUT, container_specifier, reasoning)
                continue

            # Make sure I'm holding the sponge could be dropeed because of an error injectsion
            holding = self.holding_obj_name()
            if holding is None or "DishSponge" not in holding:
                self.add_substep(step)
                reasoning = f"To wash {Utils.short_name(object_name)}, I need a sponge to clean it"
                sponge_name = self.do(
                    c.Action.PICKUP, Specifier(types=[sponge_type]), reasoning
                )
                continue

            # Try to bring the object into view, might be inside the sink and not visible
            self.try_bring_into_view(object_name)

            break

        # Make sure faucet is on
        faucet_name = self.get_controller_name(sink_name)
        faucet = self.get_obj_by_name(faucet_name)
        faucet_specifier = Specifier(names=[faucet_name])
        if faucet["isToggled"] == False:
            # Just putting in some sinks can clean objects automatically.  This is ok
            self.add_substep(step)
            reasoning = (
                f"To wash {Utils.short_name(object_name)}, I need to turn on the faucet"
            )
            basin_name = self.do(c.Action.TOGGLE_ON, faucet_specifier, reasoning)

            # Just putting in some sinks can clean objects automatically.  Other need manual cleaning
            self.add_substep(step)
            reasoning = f"To wash {Utils.short_name(object_name)}, I need to clean it with a sponge"
            object_name = self.do(
                c.Action.CLEAN,
                Specifier(names=[object_name]),
                reasoning,
                [c.Flags.OK_ALREADY_CLEAN],
            )

        self.add_substep(step)
        reasoning = f"After washing {Utils.short_name(object_name)}, I need to turn off the faucet"
        basin_name = self.do(c.Action.TOGGLE_OFF, faucet_specifier, reasoning)

        while True:
            if self.raw_plan.task_failed:
                return

            # Drop the sponge if I'm holding it
            holding_name = self.holding_obj_name()
            if holding_name is not None and holding_name != object_name:
                self.add_substep(step)
                reasoning = f"To pick up {Utils.short_name(object_name)}, I need to empty my hand first"
                self.do(c.Action.EMPTY_HAND, Specifier(names=[object_name]), reasoning)
                continue

            if not self.is_visible(object_name):
                self.add_substep(step)
                reasoning = f"To pick up the {Utils.short_name(object_name)}, I need to find it first"
                object_name = self.do(
                    c.Action.FIND, Specifier(names=[object_name]), reasoning
                )
                continue

            # Sometime washed object can be full of water when done
            if self.is_filled(object_name):
                self.add_substep(step)
                reasoning = f"There is water still in  {Utils.short_name(object_name)}, so I need to empty it"
                object_name = self.do(
                    c.Action.EMPTY_LIQUID, Specifier(names=[object_name]), reasoning
                )
                continue

            if c.Flags.DONT_PUT_AWAY not in flags:
                self.add_substep(step)
                reasoning = f"After washing {Utils.short_name(object_name)}, I need to put it away"
                object_name = self.do(
                    c.Action.PUT_AWAY, Specifier(names=[object_name]), reasoning
                )

            break

        self.step_done(step)
        self.step_complete(step)

    def do_brew_and_drink(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.BREW_AND_DRINK,
            object_name,
            f"Brew and Drink Coffee",
            reasoning,
            include_in_substep=False,
        )

        # TODO: for now always assume is coffee

        # If something is in my hand, put it down first
        holding_name = self.holding_obj_name()
        holding_object = self.get_obj_by_name(holding_name) if holding_name else None
        if holding_object is None or holding_object["objectType"] != "Mug":
            self.add_substep(step)
            reasoning = f"Coffee is drunk out of a mug {Utils.short_name(object_name)}"
            object_name = self.do(
                c.Action.PICKUP, Specifier(names=[object_name]), reasoning
            )

        # If mug is empty, I need to brew
        contents = self.liquid_contents(object_name)
        if contents is None or contents != c.LiquidType.COFFEE:
            self.add_substep(step)
            reasoning = f"If {Utils.short_name(object_name)} does not contain coffee, I need to brew some coffee first"
            object_name = self.do(
                c.Action.BREW, Specifier(names=[object_name]), reasoning
            )

        # Face the object first
        self.agent_face_object(object_name)
        self.capture_world_state()

        self.add_substep(step)
        reasoning = (
            f"The {Utils.short_name(object_name)} contains coffee so, I can drink it"
        )
        object_name = self.do(c.Action.DRINK, Specifier(names=[object_name]), reasoning)

        # Force mug to be dirty
        self.agent_dirty(object_name)

        self.step_done(step)
        self.step_complete(step)

    def do_brew(self, object_name, reasoning, flags):
        step = self.step_add(
            c.Action.BREW,
            object_name,
            f"Brew some Coffee",
            reasoning,
            include_in_substep=False,
        )

        machine_type = "CoffeeMachine"

        # If something is in my hand, put it down first
        holding_name = self.holding_obj_name()
        if holding_name is not None:
            holding_obj = self.get_obj_by_name(holding_name)
            if holding_obj["objectType"] != "Mug":
                self.add_substep(step)
                reasoning = f"To brew, I need to first pick up a {Utils.short_name(object_name)}"
                object_name = self.do(
                    c.Action.PICKUP, Specifier(names=[object_name]), reasoning
                )

        # If the mug is dirty clean if
        if self.is_dirty(object_name):
            self.add_substep(step)
            reasoning = (
                f"If {Utils.short_name(object_name)} is dirty, I need to clean it first"
            )
            object_name = self.do(
                c.Action.SINK_WASH,
                Specifier(names=[object_name]),
                reasoning,
                [c.Flags.OK_ALREADY_CLEAN, c.Flags.DONT_PUT_AWAY],
            )

            self.add_substep(step)
            reasoning = f"I need to pickup the clean {Utils.short_name(object_name)}"
            object_name = self.do(
                c.Action.PICKUP, Specifier(names=[object_name]), reasoning
            )

        self.add_substep(step)
        reasoning = f"To brew coffee, I need to put a {Utils.short_name(object_name)} in a coffee machine"
        machine_name = self.do(c.Action.PUT, Specifier(types=[machine_type]), reasoning)

        self.add_substep(step)
        reasoning = (
            f"To brew coffee, I need to turn on the {Utils.short_name(machine_name)}"
        )
        machine_name = self.do(
            c.Action.TOGGLE_ON, Specifier(names=[machine_name]), reasoning
        )

        # Some coffee machines don't work so fill manually
        if self.is_filled(object_name) is False:
            self.agent_fill(object_name, c.LiquidType.COFFEE, admin_task=True)

        self.add_substep(step)
        reasoning = f"After brewing coffee, I need to turn off the {Utils.short_name(machine_name)}"
        machine_name = self.do(
            c.Action.TOGGLE_OFF, Specifier(names=[machine_name]), reasoning
        )

        self.add_substep(step)
        reasoning = f"After brewing coffee, I need to pick up the {Utils.short_name(object_name)} containing coffee"
        object_name = self.do(
            c.Action.PICKUP, Specifier(names=[object_name]), reasoning
        )

        self.step_done(step)
        self.step_complete(step)

        return object_name

    def do_cook(self, food_name, reasoning, flags):
        if "Egg" in food_name:
            return self.do_cook_egg(food_name, reasoning, flags)

        step = self.step_add(
            c.Action.COOK,
            food_name,
            f"Cook {Utils.short_name(food_name)}",
            reasoning,
            include_in_substep=False,
        )

        food_specifier = Specifier(names=[food_name])
        pan_name = None
        find_count = 0
        while True:
            if self.raw_plan.task_failed:
                return

            # Make sure food is visible
            if not self.is_visible(food_name) and find_count < 10:
                find_count += 1
                self.add_substep(step)
                reasoning = f"To cook {Utils.short_name(food_name)}, I need to find {Utils.short_name(food_name)} first"
                food_name = self.do(c.Action.FIND, food_specifier, reasoning)
                continue

            # Slice if needed
            if c.Flags.DONT_SLICE not in flags and c.Flags.SLICE_ON_PAN not in flags:
                if self.is_sliceable(food_name) and not self.is_sliced(food_name):
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(food_name)} is sliceable, so I need to slice it first"
                    unsliced_obj = self.do(c.Action.SLICE, food_specifier, reasoning)
                    sliced_name = self.get_sliced_object_name(unsliced_obj, middle=True)
                    food_specifier = Specifier(names=[sliced_name])
                    continue

            # If I'm not holding the food item pick it up
            holding = self.holding_obj_name()
            if holding is None or holding != food_name:
                self.add_substep(step)
                reasoning = (
                    f"To cook {Utils.short_name(food_name)}, I need to pick it up first"
                )
                food_name = self.do(c.Action.PICKUP, food_specifier, reasoning)
                continue

            # Pick a stove
            food_obj = self.get_obj_by_name(food_name)
            food_type = food_obj["objectType"]

            # If cooked on a pan, need to put in on the pan
            if food_type in c.FOODS_COOKED_IN_PAN:
                pan_name = self.in_type_name(food_name, "Pan")
                if not pan_name:
                    self.add_substep(step)
                    reasoning = (
                        f"To cook the {food_name}, I need to put it in a Pan first"
                    )
                    pan_name = self.do(
                        c.Action.PUT, Specifier(types=["Pan"]), reasoning
                    )
                else:
                    Utils.print_color(
                        c.Color.YELLOW, f"{food_name} already in a Pan, skipping put"
                    )

            stove_types = c.FOOD_COOK_TYPES[food_type]
            stove_specifier = Specifier(types=stove_types)
            stove_objects = self.get_specifier_objects(stove_specifier)
            stove_name = stove_objects[0]["name"]

            # If openable stove, I need to pick one to open
            if self.is_openable(stove_name) and self.is_closed(stove_name):
                stove_specifier = Specifier(names=[stove_name])
                if self.is_closed(stove_name):
                    self.add_substep(step)
                    reasoning = f"{Utils.short_name(stove_name)} is closed, so I need to open it first"
                    stove_name = self.do(c.Action.OPEN, stove_specifier, reasoning)
                    continue

            break

        if pan_name is not None:
            # Pan needs to be on a stove burner
            burner_name = self.in_type_name(pan_name, "StoveBurner")
            if not burner_name:
                # Pickup pan if I'm not holding it
                holding_obj = self.holding_obj_name()
                if holding_obj is None or holding_obj["objectType"] != "Pan":
                    self.add_substep(step)
                    reasoning = "To put Pan on StoveBurner, I need to pick up the Pan"
                    pan_name = self.do(
                        c.Action.PICKUP, Specifier(names=[pan_name]), reasoning
                    )

                # Put pan on StoveBurner
                burner_specifier = Specifier(types=["StoveBurner"])
                self.add_substep(step)
                reasoning = (
                    f"To cook the {food_name}, I need to put the Pan on a StoveBurner"
                )
                stove_name = self.do(c.Action.PUT, burner_specifier, reasoning)
                stove_specifier = Specifier(names=[stove_name])
            else:
                stove_specifier = Specifier(names=[burner_name])

        else:
            # Now put the object, will pick particular stove if not already specified
            container_obj = self.get_non_surface_container(
                food_obj["name"], must_exist=False
            )
            if container_obj is None or container_obj["type"] not in stove_types:
                self.add_substep(step)
                reasoning = f"To cook {Utils.short_name(food_name)}, I need to put it on a {Utils.short_name(stove_types[0])} first"
                stove_name = self.do(c.Action.PUT, stove_specifier, reasoning)
                stove_specifier = Specifier(names=[stove_name])

        # Does it need to be closed
        if self.is_openable(stove_name) and self.is_open(stove_name):
            self.add_substep(step)
            reasoning = f"{Utils.short_name(stove_name)} is open, so I need to close it before cooking"
            stove_name = self.do(c.Action.CLOSE, stove_specifier, reasoning)

        # Does it need to be turned on
        stove_obj = self.get_obj_by_name(stove_name)
        if stove_obj["isToggled"] == False:
            self.add_substep(step)
            reasoning = f"To cook {Utils.short_name(food_name)}, I need to turn on the {Utils.short_name(stove_name)}"
            stove_name = self.do(c.Action.TOGGLE_ON, stove_specifier, reasoning)

        # Turn off
        self.add_substep(step)
        reasoning = f"After cooking {Utils.short_name(food_name)}, I need to turn off the {Utils.short_name(stove_name)}"
        stove_name = self.do(c.Action.TOGGLE_OFF, stove_specifier, reasoning)

        if self.is_openable(stove_name) and self.is_open(stove_name):
            # Pick back up food while still open
            self.add_substep(step)
            reasoning = f"I need to take the {Utils.short_name(food_name)} out of the {Utils.short_name(stove_name)}"
            self.do(c.Action.PICKUP, food_specifier, reasoning)

            self.add_substep(step)
            reasoning = f"{Utils.short_name(stove_name)} is open, so I need to close it after cooking"
            stove_name = self.do(c.Action.OPEN, stove_specifier, reasoning)

        self.step_done(step)
        self.step_complete(step)

        return food_name

    def do_cook_egg(self, food_name, reasoning, flags):
        step = self.step_add(
            c.Action.COOK,
            food_name,
            f"Cook {Utils.short_name(food_name)}",
            reasoning,
            include_in_substep=False,
        )

        food_specifier = Specifier(names=[food_name])

        # (1) Need to be holding the egg
        holding_obj = self.holding_obj_name()
        if holding_obj is None or holding_obj["objectType"] != "Egg":

            # If it's not visible, need to find it first
            if not self.is_visible(food_name):
                self.add_substep(step)
                reasoning = f"To cook an Egg, I need to find it first"
                food_name = self.do(c.Action.FIND, food_specifier, reasoning)

            self.add_substep(step)
            reasoning = "To cook an Egg, I need to pick it up first"
            food_name = self.do(c.Action.PICKUP, food_specifier, reasoning)
        else:
            Utils.print_color(c.Color.YELLOW, "Already holding Egg, skipping pickup")

        # (2) Egg needs to be in a pan
        pan_name = self.in_type_name(food_name, "Pan")
        if not pan_name:
            self.add_substep(step)
            reasoning = "To cook the Egg, I need to put it in a Pan first"
            pan_name = self.do(
                c.Action.PUT,
                Specifier(types=["Pan"]),
                reasoning,
                [c.Flags.SLICE_ON_PAN],
            )
        else:
            Utils.print_color(c.Color.YELLOW, "Egg already in a Pan, skipping put")

        # (3) Need to crack the egg
        self.add_substep(step)
        reasoning = "I need to crack the Egg before cooking it"
        unsliced_name = self.do(c.Action.SLICE, food_specifier, reasoning)
        if self.raw_plan.task_failed:
            return None

        sliced_name = self.get_sliced_object_name(unsliced_name)
        food_specifier = Specifier(names=[unsliced_name])

        # (4) Pan needs to be on a stove burner
        burner_name = self.in_type_name(pan_name, "StoveBurner")
        if not burner_name:
            # Pickup pan if I'm not holding it
            holding_obj = self.holding_obj_name()
            if holding_obj is None or holding_obj["objectType"] != "Pan":
                self.add_substep(step)
                reasoning = "To put Pan on StoveBurner, I need to pick up the Pan"
                pan_name = self.do(
                    c.Action.PICKUP, Specifier(names=[pan_name]), reasoning
                )

            # Put pan on StoveBurner
            burner_specifier = Specifier(types=["StoveBurner"])
            self.add_substep(step)
            reasoning = "To cook the Egg, I need to put the Pan on a StoveBurner"
            burner_name = self.do(c.Action.PUT, burner_specifier, reasoning)

        else:
            Utils.print_color(
                c.Color.YELLOW, "Pan already on StoveBurner, skipping pickup"
            )

        # (5) Burner needs to be turned on
        burner_specifier = Specifier(names=[burner_name])
        self.add_substep(step)
        reasoning = "To cook the Egg, I need to turn on the StoveBurner"
        stove_name = self.do(c.Action.TOGGLE_ON, burner_specifier, reasoning)

        # (6) Burner needs to be turned off
        self.add_substep(step)
        reasoning = "After cooking the Egg, I need to turn off the StoveBurner"
        stove_name = self.do(c.Action.TOGGLE_OFF, burner_specifier, reasoning)

        self.step_done(step)
        self.step_complete(step)

        return sliced_name

    #######################################
    def should_print(self, output: str) -> bool:
        no_print = [
            "MoveAhead",
            "RotateLeft",
            "RotateRight",
            "LookUp",
            "LookDown",
            "Pass",
            "Jumping Location",
            "TeleportFull",
            "Teleport",
        ]
        # If output contains one of the actions in no_print, return False
        for action in no_print:
            if action in output:
                return False
        return True

    def cleaned_error(self, obj_name: str = None, message: str = None) -> str:
        if self.controller.last_event.metadata["errorMessage"] == "":
            error = message
        else:
            error = self.controller.last_event.metadata["errorMessage"]

        # Remove full trace
        error = error.split("trace")[0]

        # Substitute "specified by objectId" with actual object name
        if obj_name is not None and obj_name != "":
            # replace "specified objectId" with actual object naem
            error = error.replace("specified by objectId", obj_name)

        return error

    def add_step_errors(self, step_errors: List[StepError]):
        for step_error in step_errors:
            self.record_step_error(
                action_name=step_error.action_name,
                object_name=step_error.object_name,
                error_msg=step_error.error_msg,
                error_type=step_error.error_type,
            )

    def record_step_error(
        self,
        action_name: str,
        object_name: str,
        error_msg: str,
        error_type: StepErrorType,
    ):

        # Any error is fatal if a GENERATED plan
        if self.raw_plan.plan_type == PlanType.GENERATED:
            self.raw_plan.task_failed = True

        # Note error for this step
        self.step_error = StepError(action_name, object_name, error_msg, error_type)

        # Add to list of all errors for the plan
        self.plan_errors.append(self.step_error)

        Utils.print_color(c.Color.RED, error_msg)

    single_try_errors = ["No valid positions to place object found"]

    def print_result(
        self,
        error_message: str,
        last_action: str,
        obj_name: str = None,
        message: str = None,
    ):
        if error_message is None:
            raise ValueError("Error message is None")

        output = f"({self.num_tries}) "

        # Show what I'm doing
        output += f"{last_action:<23} "

        # Show object I'm acting on
        if obj_name is None:
            obj_name = ""
        output += f"{obj_name:<22} "

        # Show what I'm holding
        holding_object = self.holding_obj_name()
        if holding_object is None:
            holding_object = ""
        holding_object = f"({holding_object})"
        output += f"{holding_object:<22}"

        if obj_name is not None and obj_name != "":
            distance = self.distance_to_object(obj_name)
            output += f" [{distance:.2f}m] "

        # If there is an error, make output message
        error_output = None
        if error_message != "":
            # Remove full trace
            error_output = error_message.split("trace")[0]
            if obj_name is not None and obj_name != "":
                error_output = f"     [{obj_name}] {error_output}"

        # Show any extra message
        if message is not None:
            output += f" ({message})"

        # Filter out messages that are not needed (debug setting)
        if self.should_print(last_action):

            if self.injection_state == c.InjectionState.PRE_INJECTION:
                color = c.Color.GREY_BLUE
            elif self.injection_state == c.InjectionState.INJECTING:
                color = c.Color.LIGHT_BLUE
            else:
                color = c.Color.NONE

            if self.last_repeat:
                Utils.print_color(color, f"\n{output}")
                self.last_repeat = False
            else:
                Utils.print_color(color, output)

                if error_output is not None:
                    if error_message in self.single_try_errors:
                        # Warning
                        Utils.print_color(c.Color.YELLOW, f"{error_output:<22}")
                    else:
                        # Error
                        Utils.print_color(c.Color.ORANGE, f"{error_output:<22}")

                    if "cannot be placed" in error_output:
                        raise AgentFatal(f"Cannot place object: {error_output}")

                # Print to log
                self.log.append(output)
                if error_output is not None:
                    self.log.append(error_output)

    def check_result(
        self, obj_name: str = None, message: str = None, admin_task: bool = False
    ) -> bool:

        last_action = self.controller.last_event.metadata["lastAction"]
        error_message = self.controller.last_event.metadata["errorMessage"]

        if "another object" in error_message:
            print("opportunity to move blocking item")  # TODO

        # If admins-task, not recording so handle differently
        if admin_task or self.injection_state != c.InjectionState.POST_INJECTION:
            success = self.controller.last_event.metadata["lastActionSuccess"]
            if not success:
                error = self.cleaned_error(obj_name)
                Utils.print_color(c.Color.PURPLE, error)
            self.controller.step(action="Pass")
            self.print_result(error_message, last_action, obj_name, message)
            return success

        elif self.controller.last_event.metadata["lastActionSuccess"] is False:
            self.print_result(error_message, last_action, obj_name, message)

            self.last_retry_error = self.cleaned_error(obj_name)
            self.controller.step(action="Pass")  # TODO: needed?
            return False
        else:
            self.last_retry_error = "--"
            self.controller.step(action="Pass")  # TODO: needed?
            self.controller.step(action="Pass")
            self.capture_video_frame()  # Capture frame after each action
            self.print_result(error_message, last_action, obj_name, message)
            return True

    #####################################
    #
    # AGENT Functions
    #
    ####################################
    def agent_find(self, object_name):

        # Unlike other agent actions
        self.do_face_object(object_name)
        is_visible = self.is_visible(object_name)
        return is_visible

    def agent_pickup(self, object_name):
        obj = self.get_obj_by_name(object_name)

        if obj["pickupable"] is False:
            raise AgentFatal(f"Cannot pickup {object_name}, not pickupable")

        holding_object = self.holding_obj_name()
        if holding_object is not None:
            raise AgentFatal(
                f"Cannot pickup {object_name}, already holding {holding_object}"
            )

        self.controller.step(
            action="PickupObject",
            objectId=obj["objectId"],
            forceAction=False,
            manualInteract=False,
        )

        # If I picked up an object, also remember it what it contained, as AITHOR
        # loses containment once an object has been picked up
        if (
            self.controller.last_event.metadata["lastActionSuccess"]
            and obj["receptacleObjectIds"] is not None
            and obj["receptacleObjectIds"] != []
        ):
            self.holding_contents[obj["name"]] = obj["receptacleObjectIds"]

        # If I picked up an object, remove it from the holding contents
        self.clear_holding_object(object_name)

        return self.check_result(object_name)

    def agent_open(self, object_name):
        _, obj = self.get_agent_and_object(object_name)
        self.controller.step(
            action="OpenObject", objectId=obj["objectId"], openness=1, forceAction=True
        )
        return self.check_result(object_name)

    def agent_close(self, object_name, admin_task=False):
        _, obj = self.get_agent_and_object(object_name)
        self.controller.step(
            action="CloseObject", objectId=obj["objectId"], forceAction=False
        )
        return self.check_result(object_name, admin_task=admin_task)

    def agent_put(self, dest_name):
        obj = self.get_obj_by_name(dest_name)

        held_object_name = self.holding_obj_name()
        if held_object_name is None:
            raise AgentFatal("PUT: No object held")

        self.controller.step(
            action="PutObject",
            objectId=obj["objectId"],
            forceAction=False,
            placeStationary=True,
        )

        return self.check_result(dest_name)

    def agent_toggle_on(self, dest_name):
        _, obj = self.get_agent_and_object(dest_name)
        self.controller.step(
            action="ToggleObjectOn",
            objectId=obj["objectId"],
            forceAction=False,
        )
        return self.check_result(dest_name)

    def agent_toggle_off(self, dest_name):
        _, obj = self.get_agent_and_object(dest_name)
        self.controller.step(
            action="ToggleObjectOff",
            objectId=obj["objectId"],
            forceAction=False,
        )
        return self.check_result(dest_name)

    def agent_slice(self, dest_name):
        _, obj = self.get_agent_and_object(dest_name)
        object_type = obj["objectType"]
        self.controller.step(
            action="SliceObject",
            objectId=obj["objectId"],
            forceAction=False,
        )
        result = self.check_result(dest_name)

        # Move slices apart so more visible (except for eggs)
        if result and object_type != "Egg":
            self._adjust_slices(obj)

        return result

    def _adjust_slices(self, unsliced_obj):
        """
        Push slices apart after slicing to make them more visible.
        Falls back to removing the largest slice if push fails.
        """
        sliced_objs = []
        unsliced_asset_id = f"{unsliced_obj['assetId']}_"

        # Find all sliced objects from this original object
        for obj in self.all_objects():
            if obj["name"] == unsliced_asset_id:
                continue
            if unsliced_obj["assetId"] not in obj["name"]:
                continue
            if "Slice" not in obj["objectId"]:
                continue
            sliced_objs.append(obj)

        if len(sliced_objs) == 0:
            return

        # Try to push slices apart using DirectionalPush
        push_succeeded = self._push_slices_apart_nudge(sliced_objs)

        if push_succeeded:
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"Pushed {len(sliced_objs)} slices apart",
            )
            return

        # Fallback: Sort by size (largest first) and remove the largest one
        sliced_objs.sort(key=self.size_xyz, reverse=True)
        largest_slice = sliced_objs[0]

        try:
            self.controller.step(
                action="DisableObject",
                objectId=largest_slice["objectId"],
            )
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"Removed largest slice {largest_slice['name']} from scene",
            )
        except Exception as e:
            Utils.print_color(
                c.Color.RED,
                f"Failed to remove largest slice {largest_slice['name']}: {e}",
            )

    def _push_slices_apart_nudge(self, sliced_objs) -> bool:
        """
        Push slices apart using DirectionalPush with small nudges.
        Algorithm:
        1. Save original position and rotation (x, y, z) for each slice
        2. Loop until stopping criteria met or max attempts reached:
           - For each slice:
             * Nudge it once in a different direction
             * Wait for physics to settle
             * Check ALL slices for stopping criteria
             * If any meets criteria, return success immediately
           - If no criteria met after one full pass, loop again

        Stops if ANY slice:
        - Moves more than 0.10m from initial position
        - Rotates more than 20° on X or Z axis (tips)

        Returns True if stopping criteria met or at least one push succeeded.
        """
        import random

        any_success = False
        max_attempts = 10

        # Store slice names for later lookup (objects change after each step)
        slice_names = [obj["name"] for obj in sliced_objs]

        # Record ORIGINAL positions and rotations (x, y, z) for all slices
        initial_positions = {}
        initial_rotations = {}
        for slice_obj in sliced_objs:
            initial_positions[slice_obj["name"]] = self.get_nppos(slice_obj)
            rot = slice_obj["rotation"]
            initial_rotations[slice_obj["name"]] = (rot["x"], rot["y"], rot["z"])

        # Track slices that have failed to push
        failed_slices = set()

        # Try multiple passes of nudging all slices
        for attempt in range(max_attempts):
            attempt_success = False

            # Increase nudge magnitude slightly with each attempt
            push_magnitude = 4.0 + (attempt * 0.5)  # 4.0, 4.5, 5.0, 5.5, etc.

            # For each slice: nudge, wait for physics, then check ALL slices for stopping criteria
            for i, slice_name in enumerate(slice_names):
                # Skip slices that have already failed
                if slice_name in failed_slices:
                    continue

                try:
                    # Get fresh object data
                    slice_obj = self.get_obj_by_name(slice_name, must_exist=False)
                    if slice_obj is None:
                        # Slice no longer exists (might have fallen/disappeared)
                        failed_slices.add(slice_name)
                        continue

                    # Determine push direction based on slice's longest horizontal dimension
                    # For bread/food slices, push along their longest axis for best separation
                    bbox = slice_obj.get("boundingBox", {})
                    if bbox:
                        # Get dimensions from bounding box
                        size_x = abs(
                            bbox.get("max", {}).get("x", 0)
                            - bbox.get("min", {}).get("x", 0)
                        )
                        size_z = abs(
                            bbox.get("max", {}).get("z", 0)
                            - bbox.get("min", {}).get("z", 0)
                        )
                    else:
                        # Fallback: use default distribution if no bounding box
                        size_x = 1.0
                        size_z = 1.0

                    # If X is longer, push along Z axis (0° or 180°)
                    # If Z is longer, push along X axis (90° or 270°)
                    if size_x > size_z:
                        # Loaf is oriented along X, push perpendicular (along Z)
                        push_angle = 90.0 if i % 2 == 0 else 270.0
                    else:
                        # Loaf is oriented along Z, push perpendicular (along X)
                        push_angle = 0.0 if i % 2 == 0 else 180.0

                    # Add randomness to the base angle (±25° for variety in push directions)
                    # This prevents overly predictable patterns while maintaining general direction
                    push_angle += random.uniform(-25, 25)

                    self.controller.step(
                        action="DirectionalPush",
                        objectId=slice_obj["objectId"],
                        moveMagnitude=push_magnitude,  # Increases each attempt
                        pushAngle=push_angle,
                    )

                    if self.controller.last_event.metadata["lastActionSuccess"]:
                        any_success = True
                        attempt_success = True
                    else:
                        error = self.controller.last_event.metadata.get(
                            "errorMessage", ""
                        )
                        Utils.print_color(
                            c.Color.YELLOW,
                            f"DirectionalPush failed for {slice_name}: {error}",
                        )
                        failed_slices.add(slice_name)
                        continue

                    # Wait for physics to settle AFTER THIS NUDGE
                    self.pause()

                    # Check ALL slices for stopping criteria AFTER EACH NUDGE
                    for check_name in slice_names:
                        check_obj = self.get_obj_by_name(check_name, must_exist=False)
                        if check_obj is None:
                            Utils.print_color(
                                c.Color.YELLOW,
                                f"Slice {check_name} no longer exists, stopping pushes",
                            )
                            return any_success

                        # Check if slice moved too far (absolute threshold of 0.10m = 10cm)
                        current_pos = self.get_nppos(check_obj)
                        initial_pos = initial_positions[check_name]
                        distance_moved = float(
                            np.linalg.norm(current_pos - initial_pos)
                        )
                        movement_threshold = 0.10  # 10cm max movement

                        if distance_moved > movement_threshold:
                            Utils.print_color(
                                c.Color.YELLOW,
                                f"STOPPING: Slice {check_name} moved {distance_moved:.4f}m (exceeded {movement_threshold}m threshold)",
                            )
                            return any_success

                        # Check if slice tipped over (X or Z rotation)
                        rot = check_obj["rotation"]
                        current_rotation = (rot["x"], rot["y"], rot["z"])
                        initial_rotation = initial_rotations[check_name]

                        x_diff = abs(current_rotation[0] - initial_rotation[0])
                        z_diff = abs(current_rotation[2] - initial_rotation[2])

                        # Handle wrap-around
                        if x_diff > 180:
                            x_diff = 360 - x_diff
                        if z_diff > 180:
                            z_diff = 360 - z_diff

                        if x_diff >= 20 or z_diff >= 20:
                            Utils.print_color(
                                c.Color.YELLOW,
                                f"STOPPING: Slice {check_name} tipped (X: {x_diff:.1f}°, Z: {z_diff:.1f}°, threshold: 20°)",
                            )
                            return any_success

                except Exception as e:
                    Utils.print_color(
                        c.Color.YELLOW,
                        f"Push failed for {slice_name}: {e}",
                    )

            # If no stopping criteria met and no successful pushes, try again
            if not attempt_success and attempt < max_attempts - 1:
                Utils.print_color(
                    c.Color.LIGHT_BLUE,
                    f"Attempt {attempt + 1} complete, retrying (max {max_attempts} attempts)",
                )

        # Finished all attempts without hitting stopping criteria
        if any_success:
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"Completed {max_attempts} nudge attempts (all slices within thresholds)",
            )

        return any_success

    def _push_slices_apart(self, sliced_objs) -> bool:
        """
        Original version: Push slices apart using DirectionalPush with random angles.
        Uses aggressive pushes without extensive stopping criteria.
        """
        import random

        any_success = False

        # Push each slice in a different direction
        for i, obj in enumerate(sliced_objs):
            try:
                push_angle = (360.0 / len(sliced_objs)) * i + random.uniform(-20, 20)

                self.controller.step(
                    action="DirectionalPush",
                    objectId=obj["objectId"],
                    moveMagnitude=15.0,
                    pushAngle=push_angle,
                )

                if self.controller.last_event.metadata["lastActionSuccess"]:
                    any_success = True
                else:
                    error = self.controller.last_event.metadata.get("errorMessage", "")
                    Utils.print_color(
                        c.Color.YELLOW,
                        f"DirectionalPush failed for {obj['name']}: {error}",
                    )

                self.pause()

            except Exception as e:
                Utils.print_color(
                    c.Color.YELLOW,
                    f"Push failed for {obj['name']}: {e}",
                )

        return any_success

    def agent_clean(self, dest_name, admin_task=False):
        _, obj = self.get_agent_and_object(dest_name)

        # Ok if already dirty, mugs sometimes clean automatically under water
        if not obj["isDirty"]:
            return True
        self.controller.step(
            action="CleanObject",
            objectId=obj["objectId"],
            forceAction=False,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_dirty(self, dest_name, admin_task=False):
        _, obj = self.get_agent_and_object(dest_name)
        self.controller.step(
            action="DirtyObject",
            objectId=obj["objectId"],
            forceAction=admin_task,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_fill(self, dest_name: str, liquid_type: c.LiquidType, admin_task=False):
        _, obj = self.get_agent_and_object(dest_name)
        self.controller.step(
            action="FillObjectWithLiquid",
            objectId=obj["objectId"],
            fillLiquid=liquid_type.value,
            forceAction=admin_task,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_empty(self, obj_name: str, admin_task=False):
        obj = self.get_obj_by_name(obj_name)
        self.controller.step(
            action="EmptyLiquidFromObject",
            objectId=obj["objectId"],
            forceAction=admin_task,
        )
        return self.check_result(obj_name, admin_task=admin_task)

    def agent_teleport(self, pose, object_name: str = None) -> bool:
        # Clip horizon to valid range
        if pose["horizon"] > 60:
            pose["horizon"] = 60
        if pose["horizon"] < -30:
            pose["horizon"] = -30

        try:
            self.controller.step(
                action="Teleport",
                position=pose["position"],
                rotation=pose["rotation"],
                horizon=pose["horizon"],
                standing=pose["standing"],
            )
        except Exception as e:
            Utils.print_color(c.Color.RED, f"Teleport failed: {e}")
            return False

        return self.check_result(object_name, admin_task=True)

    def agent_turn_left(self, dest_name, admin_task=False):
        self.controller.step(
            action="RotateLeft",
            degrees=self.rotation_speed_horiz,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_turn_right(self, dest_name, admin_task=False):
        self.controller.step(
            action="RotateRight",
            degrees=self.rotation_speed_horiz,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_turn_up(self, dest_name, admin_task=False):
        self.controller.step(
            action="LookUp",
            degrees=self.rotation_speed_vert,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    def agent_turn_down(self, dest_name, admin_task=False):
        self.controller.step(
            action="LookDown",
            degrees=self.rotation_speed_vert,
        )
        return self.check_result(dest_name, admin_task=admin_task)

    # NOTE: Note currently used
    def camera_offset(self) -> np.ndarray:
        """
        Get the camera offset from the agent's position.
        This is used to calculate the camera's position in the world.
        """
        agent_pos = Utils.array_to_np(
            self.controller.last_event.metadata["agent"]["position"]
        )
        camera_pos = Utils.array_to_np(
            self.controller.last_event.metadata["cameraPosition"]
        )
        camera_offset = camera_pos - agent_pos

        return camera_offset

    def calculate_camera_position(self, from_nppos: np.ndarray = None) -> np.ndarray:
        """
        Calculate the camera position based on the agent's position and the camera offset.
        """
        if from_nppos is None:
            # If agent_nppos is not provided, use the last event's camera position
            camera_nppos = Utils.array_to_np(
                self.controller.last_event.metadata["cameraPosition"]
            )

        else:
            # Otherwise use it to calculate the camera position
            camera_nppos = from_nppos.copy()
            if not self.is_crouched():
                # If the agent isn't crouched, adjust the camera position
                camera_nppos[1] += c.CAMERA_Y_OFFSET

        return camera_nppos

    def calculate_horizon(self, object_name: str, from_nppos=None) -> float:
        obj = self.get_obj_by_name(object_name)
        obj_nppos = self.get_nppos(obj)

        camera_nppos = self.calculate_camera_position(from_nppos)

        # Calculate the vertical difference between the object and the camera
        vertical_diff = camera_nppos[1] - obj_nppos[1]

        # Calculate the horizontal distance between the object and the camera
        horizontal_distance = Utils.horizontal_distance(obj_nppos, camera_nppos)

        # Calculate the angle to the object in the vertical plane (in degrees)
        horizon = np.degrees(np.arctan2(vertical_diff, horizontal_distance))

        return float(horizon)

    def agent_turn_vert(self, object_name: str = None) -> bool:

        camera_horizon = self.controller.last_event.metadata["agent"]["cameraHorizon"]
        target_horizon = self.calculate_horizon(object_name)

        # Compare the target angle with the current camera horizon
        if (
            target_horizon > camera_horizon + c.VERTICAL_LOOK_THRESHOLD
        ):  # Add a small threshold for tolerance
            look_success = self.agent_move_action("LookDown")
        elif (
            target_horizon < camera_horizon - c.VERTICAL_LOOK_THRESHOLD
        ):  # Subtract a small threshold for tolerance
            look_success = self.agent_move_action("LookUp")
        else:
            look_success = True

        return look_success

    # RENAME
    def do_face_object(self, object_name):
        self.agent_face_object(object_name)

    def agent_set_crouch(self, object_name: str = None):
        if self.calculate_standing(object_name):
            if self.is_crouched():
                self.controller.step(action="Stand")
                return self.check_result(admin_task=True)
        else:
            if not self.is_crouched():
                self.controller.step(action="Crouch")
                return self.check_result(admin_task=True)

        return True

    def agent_turn_direction(self, object_name: str, threshold=0) -> str:
        agent = self.get_agent()
        agent_rot = self.get_rot(agent)

        # Calculate the angle to the target object
        angle = self.calculate_rotation(object_name)

        return self.shortest_turn_direction(agent_rot, angle, threshold)

    def agent_face_object(self, object_name: str = None):
        """
        Turn the agent to face the object, and look at it vertically
        """

        # First try forcing agent direction
        angle = self.calculate_rotation(object_name)
        position = self.controller.last_event.metadata["agent"]["position"]
        agent_nppos = Utils.array_to_np(position)
        rotation = self.calculate_rotation(object_name, agent_nppos)
        horizon = self.calculate_horizon(object_name, agent_nppos)
        standing = self.calculate_standing(object_name)
        pose = {
            "position": position,
            "rotation": rotation,
            "standing": standing,
            "horizon": horizon,
        }
        teleport_success = self.agent_teleport(pose, object_name)
        if teleport_success:
            self.look_vert_at_object(object_name)
            return

        num_tries = 0
        while True:
            agent = self.get_agent()
            agent_rot = self.get_rot(agent)

            # Calculate the angle to the target object
            angle = self.calculate_rotation(object_name)

            if angle is None or abs(angle - agent_rot) < 5:
                break

            turn_direction = self.shortest_turn_direction(agent_rot, angle)
            if turn_direction == c.TurnDirection.LEFT:
                turn_success = self.agent_move_action("RotateLeft")
            elif turn_direction == c.TurnDirection.RIGHT:
                turn_success = self.agent_move_action("RotateRight")
            else:
                # Already facing
                turn_success = True

            # Set crouch if needed
            self.agent_set_crouch(object_name)

            # Agent should facing the object vertially
            look_success = self.agent_turn_vert(object_name)

            # If I can't turn any further, I'm done
            if not look_success and (angle < 5 or turn_success is False):
                break

            is_looking_at = self.is_looking_vert_at_object(object_name)
            is_facing = angle is None or abs(angle - agent_rot) < 5
            if is_looking_at and is_facing:
                break

            num_tries += 1
            if num_tries > 100:
                raise AgentFailure(
                    f"Cannot agent_face_object {object_name} after 100 tries"
                )

    def look_vert(self, target_horizon: float):
        """
        Look at target horizon vertically
        """
        num_tries = 0
        while True:

            camera_horizon = self.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]

            # Compare the target angle with the current camera horizon
            if (
                target_horizon > camera_horizon + c.VERTICAL_LOOK_THRESHOLD
            ):  # Add a small threshold for tolerance
                look_success = self.agent_move_action("LookDown")
            elif (
                target_horizon < camera_horizon - c.VERTICAL_LOOK_THRESHOLD
            ):  # Subtract a small threshold for tolerance
                look_success = self.agent_move_action("LookUp")
            else:
                look_success = True

            # If I can't turn any further, I'm done
            if not look_success:
                break

            camera_horizon = self.controller.last_event.metadata["agent"][
                "cameraHorizon"
            ]
            if abs(target_horizon - camera_horizon) <= 10.0:
                break

            num_tries += 1
            if num_tries > 100:
                raise AgentFailure(f"Cannot look_vert {target_horizon} after 100 tries")

    def look_vert_at_object(self, object_name: str):
        """
        Look at object vertically
        """
        num_tries = 0
        while True:

            # Agent should facing the object vertially
            look_success = self.agent_turn_vert(object_name)

            # If I can't turn any further, I'm done
            if not look_success:
                break

            is_looking_at = self.is_looking_vert_at_object(object_name)
            if is_looking_at:
                break

            num_tries += 1
            if num_tries > 100:
                raise AgentFailure(
                    f"Cannot look_vert_at_object {object_name} after 100 tries"
                )

    def is_looking_vert_at_object(
        self, object_name, threshold: float = c.VERTICAL_LOOK_THRESHOLD
    ) -> bool:
        """
        Is the agent looking at the object verically
        """
        camera_horizon = self.controller.last_event.metadata["agent"]["cameraHorizon"]
        target_horizon = self.calculate_horizon(object_name)
        return abs(target_horizon - camera_horizon) <= threshold

    def teleport_and_act(self, object_name: str, agent_action) -> bool:
        """
        Teleport to the object and then try to execute the action.
        Returns True if the action was successful, False otherwise.
        """
        # If put action, try to find a cached pose first
        action_name = Utils.action_name_from_action(agent_action)
        if action_name == c.Action.PUT:
            holding_name = self.holding_obj_name()
            pose = PutCache.get_put_pose(self.hash, holding_name, object_name)
            if pose is not None:
                teleport_success = self.agent_teleport(pose, object_name)
                if not teleport_success:
                    PutCache.delete_put_pose(self.hash, holding_name, object_name)
                else:
                    action_success = self.try_execute_action(object_name, agent_action)
                    if not action_success:
                        PutCache.delete_put_pose(self.hash, holding_name, object_name)
                    else:
                        return action_success

        # Check for a cached teleport position
        pose = PlacementCache.get_interaction_pose(
            self.raw_plan.scene, object_name, agent_action
        )
        if pose is None:
            obj = self.get_obj_by_name(object_name)
            pose = PlacementCache.get_location_pose(
                self.raw_plan.scene, obj["position"]
            )
        if pose is not None:
            teleport_success = self.agent_teleport(pose, object_name)
            if not teleport_success:
                PlacementCache.delete_interaction_pose(
                    self.raw_plan.scene, object_name, agent_action
                )
            else:
                action_success = self.try_execute_action(object_name, agent_action)
                if not action_success:
                    PlacementCache.delete_interaction_pose(
                        self.raw_plan.scene, object_name, agent_action
                    )
                else:
                    return action_success

        # Then try to jump to the object
        teleport_success = self.jump_to_object(object_name, agent_action)
        return teleport_success

    def approach_object_after_manual_failure(self, object_name: str):
        """
        When testing if an action is possible becuase of rules, approach an object to get in interaction distance
        That way the model can better see why (i.e. there were items in the sink)
        """
        last_distance = 0
        num_tries = 0
        obj = self.get_obj_by_name(object_name)
        while True:
            if num_tries > 20:
                return

            # Face the object first
            self.agent_face_object(object_name)

            # Set crouch if needed
            self.agent_set_crouch(object_name)

            # Then move towards it
            move_success = self.agent_move_action("MoveAhead")
            if not move_success:
                return False

            # Get distance and last move amount
            distance = self.distance_to_object(object_name)
            move_distance = abs(distance - last_distance)
            last_distance = distance

            in_interaction_distance = self.within_interaction_range(
                object_name, distance
            )
            # If I can't move further or less than max distance
            if move_distance < 0.01 or in_interaction_distance:
                return

            num_tries += 1

    def approach_and_act(self, object_name: str, agent_action):

        last_distance = 0
        num_tries = 0
        obj = self.get_obj_by_name(object_name)
        while True:
            if num_tries > 20:
                raise AgentFailure(
                    f"Cannot approach object {object_name} after 20 tries"
                )

            # Face the object first
            self.agent_face_object(object_name)

            # Set crouch if needed
            self.agent_set_crouch(object_name)

            # If a FIND action, approach or back up until visible or move fails
            action_name = Utils.action_name_from_action(agent_action)
            if action_name == c.Action.FIND and num_tries < 15:
                if not self.is_visible(object_name):
                    # Only keep approaching if on a lower shelf, as getting close won't help
                    if obj["position"]["y"] > c.LOWER_SHELF_THRESHOLD:
                        move_success = self.agent_move_action("MoveAhead")
                        if move_success:
                            continue
                    else:
                        move_success = self.agent_move_action("MoveBack")
                        if move_success:
                            continue
            else:
                # Then move towards it
                move_success = self.agent_move_action("MoveAhead")
                if not move_success:
                    return False

            # If I'm within reach distance, do the action
            distance = self.distance_to_object(object_name)
            move_distance = abs(distance - last_distance)
            last_distance = distance

            in_interaction_distance = self.within_interaction_range(
                object_name, distance
            )
            # If I can't move further and less that max distance
            if move_distance < 0.01 or in_interaction_distance:
                return self.try_execute_action(object_name, agent_action)

            num_tries += 1

    def get_positions_around_object(self, obj, distance: float = 1.5):
        """
        Calculate 8 positions and rotations equally spaced around an object at the given distance.
        """
        # Extract the object's position
        obj_pos = obj["position"]
        agent_y = self.controller.last_event.metadata["agent"]["position"][
            "y"
        ]  # Keep height consistent

        # Generate 8 equally spaced angles (in degrees)
        angles = np.linspace(0, 360, 9)[:8]  # 0, 45, 90, 135, 180, 225, 270, 315

        positions = []
        for angle_deg in angles:
            # Convert angle to radians
            angle_rad = np.radians(angle_deg)

            # Calculate position at this angle
            new_x = float(obj_pos["x"] + distance * np.sin(angle_rad))
            new_z = float(obj_pos["z"] + distance * np.cos(angle_rad))

            # Calculate rotation to face the object
            facing_rotation = {
                "x": 0,
                "y": (angle_deg + 180) % 360,  # Face toward the object
                "z": 0,
            }

            positions.append(
                {
                    "position": {"x": new_x, "y": agent_y, "z": new_z},
                    "rotation": facing_rotation,
                    "angle": angle_deg,  # Include the angle for reference
                }
            )

        return positions

    def get_pose_in_front(self, obj, n: float):
        """
        Calculate a position in front of an object at a distance `n` and a rotation facing the object.
        """
        # Extract the object's position and rotation
        obj_pos = obj["position"]
        obj_rot = obj["rotation"]

        # Convert the rotation (yaw) to radians
        yaw_rad = np.radians(obj_rot["y"])

        # Calculate the new position in front of the object
        target_x = float(obj_pos["x"] + n * np.sin(yaw_rad))
        target_z = float(obj_pos["z"] + n * np.cos(yaw_rad))
        target_y = self.controller.last_event.metadata["agent"]["position"][
            "y"
        ]  # Keep the same height as the agent
        target_position = {"x": target_x, "y": target_y, "z": target_z}

        # The new rotation should face the object
        target_nppos = Utils.array_to_np(target_position)

        target_rotation = self.calculate_rotation(obj["name"], target_nppos)
        target_horizon = self.calculate_horizon(obj["name"], target_nppos)
        target_standing = self.calculate_standing(obj["name"])

        return {
            "position": target_position,
            "rotation": target_rotation,
            "standing": target_standing,
            "horizon": target_horizon,
        }

    def closest_point(self, target: dict, poses: list[dict]) -> dict:
        """
        Find the point in the list that is closest to the target point.
        """
        target_np = np.array([target["x"], target["y"], target["z"]])
        best_pose = None
        min_distance = float("inf")

        for pose in poses:
            pos = pose["position"]
            point_np = np.array([pos["x"], pos["y"], pos["z"]])

            horizontal_distance = Utils.horizontal_distance(target_np, point_np)
            if horizontal_distance < min_distance:
                min_distance = horizontal_distance
                best_pose = pose

        return best_pose

    def sort_poses_by_distance(self, target: dict, poses: list[dict]) -> list[dict]:
        """
        Given a target position (dict with x, y, z) and a list of poses (dicts with position and standing),
        return a new list sorted by distance to the target (closest first).
        """

        def distance(pose):
            pos = pose["position"]
            return (
                (pos["x"] - target["x"]) ** 2
                + (pos["y"] - target["y"]) ** 2
                + (pos["z"] - target["z"]) ** 2
            ) ** 0.5

        def standing(pose):
            standing = pose["standing"]
            if standing:
                return 0
            else:
                return 1

        # First sort by distance
        sorted_poses = sorted(poses, key=distance)

        # Then sort by 'standing' with True first
        sorted_poses = sorted(poses, key=standing)
        return sorted_poses

    def filter_poses_by_distance(
        self, object_name: str, poses: list[dict], extra_distance: float = 0
    ) -> list[dict]:

        min_dist = self.min_interaction_distance(object_name) - extra_distance
        max_dist = self.max_interaction_distance(object_name) + extra_distance
        filtered_poses = []
        for pose in poses:
            pos = pose["position"]
            pos_nppos = Utils.array_to_np(pos)
            object_nppos = self.get_nppos(self.get_obj_by_name(object_name))
            horizontal_distance = Utils.horizontal_distance(object_nppos, pos_nppos)
            if horizontal_distance < max_dist and horizontal_distance > min_dist:
                filtered_poses.append(pose)

        if len(filtered_poses) == 0:
            raise AgentFatal(f"filter_poses_by_distance: No poses found")

        return filtered_poses

    def calculate_jump_poses(self, object_name: str, extra_distance: float = 0.0):
        obj = self.get_obj_by_name(object_name)

        jump_candidates = JumpCandidates(obj)

        # Get interactible poses for the object
        self.controller.step(
            action="GetInteractablePoses",
            objectId=obj["objectId"],
            horizons=[0, 15, 30],  # np.linspace(-30, 60, 30),
            standings=[True, False],
        )
        interactable_poses = self.controller.last_event.metadata["actionReturn"]

        if len(interactable_poses) > 0:
            # Get unique poses and recalculate rotation and standing
            jump_candidates.interactable_poses = self.unique_poses(
                interactable_poses, obj
            )
            jump_candidates.initialize()
            return jump_candidates.interactable_poses

        print(f"No poses found for object {object_name}, picking random")

        positions = self.controller.step(action="GetReachablePositions").metadata[
            "actionReturn"
        ]

        # Now convert to poses, setting standing to True
        jump_candidates.reachable_poses = [
            {
                "x": pos["x"],
                "y": pos["y"],
                "z": pos["z"],
                "standing": self.calculate_standing(obj["name"]),
                "horizon": self.calculate_horizon(obj["name"], Utils.array_to_np(pos)),
                "rotation": self.calculate_rotation(
                    obj["name"], Utils.array_to_np(pos)
                ),
            }
            for pos in positions
        ]

        jump_candidates.reachable_poses = self.unique_poses(
            jump_candidates.reachable_poses, obj
        )
        jump_candidates.initialize()
        return jump_candidates.reachable_poses

    def count_action(self, action_type: c.Action):
        action_name = action_type.value
        if action_name not in self.action_counts:
            self.action_counts[action_name] = 0
        self.action_counts[action_name] += 1

    def unique_poses(self, poses, obj):
        """
        Given a list of pose dicts, return a list of unique position+standing objects.
        """
        seen = {}
        for pose in poses:
            key = f"{round(pose['x'], 5)}_{round(pose['y'], 5)}_{round(pose['z'], 5)}"
            if key not in seen:
                seen[key] = pose

        # Convert to pose format
        # Calculate our own standing, rotation and horizon as we do a better job
        obj_name = obj["name"]
        standing = self.calculate_standing(obj_name)
        unique_poses = []
        for unique_pos in seen.values():
            pos = {"x": unique_pos["x"], "y": unique_pos["y"], "z": unique_pos["z"]}
            np_pos = Utils.array_to_np(pos)
            rotation = self.calculate_rotation(obj_name, np_pos)
            horizon = self.calculate_horizon(obj_name, np_pos)
            # TODO: compare calculated version to existing valus for rot, horz and standing
            # TODO: potentially flag to try either?
            unique_poses.append(
                {
                    "position": pos,
                    "rotation": rotation,
                    "standing": standing,
                    "horizon": horizon,
                }
            )

        return unique_poses

    def jump_to_object(self, object_name, agent_action):
        action_name = Utils.action_name_from_action(agent_action)
        holding_name = self.holding_obj_name()

        # If I don't yet have a list of candidate poses, calculate them
        if self.candidate_poses is None:
            self.candidate_poses = self.calculate_jump_poses(object_name)

        min_distance = float("inf")
        max_distance = 0
        try_count = 0
        num_poses = len(self.candidate_poses)
        Utils.print_start_overwrite()
        while True:

            if len(self.candidate_poses) == 0:
                Utils.print_end_overwrite()
                action_name = Utils.action_name_from_action(agent_action)

                # If it was a put action, note the failure in the cache
                if (
                    (action_name == c.Action.PUT)
                    and holding_name is not None
                    and not self.is_closed(object_name)
                ):
                    # Clear last opened container tracking if PUT failed on it
                    obj = self.get_obj_by_name(object_name)
                    obj_type = obj["objectType"]
                    if (
                        obj_type in self.last_opened_container
                        and self.last_opened_container[obj_type] == object_name
                    ):
                        del self.last_opened_container[obj_type]
                        print(
                            f"[DEBUG] PUT failed: Cleared last opened {obj_type}: {object_name}"
                        )

                    # Only if the container is empty
                    blocking_objects = self.get_container_contents(object_name)
                    if blocking_objects is None or len(blocking_objects) == 0:
                        PlacementCache.add_failure(
                            self.raw_plan.scene, holding_name, object_name
                        )
                        # Keep track of failed containers, so I know which scenes are problematic with what object
                        DefectiveContainers.save(
                            self.raw_plan.scene,
                            object_name,
                            holding_name,
                            False,
                            self.hash,
                        )

                Utils.print_color(
                    c.Color.ORANGE,
                    f"    {c.POSES_ERROR} for {object_name} / {holding_name}",
                )

                raise AgentFailure(
                    f"{c.POSES_ERROR} for {object_name} / {holding_name}"
                )

            # Get remove the first position from the list
            pose = self.candidate_poses.pop(0)

            # Try to teleport there
            teleport_success = self.agent_teleport(pose, object_name)
            if not teleport_success:
                continue

            # Track distancd
            distance = self.distance_to_object(object_name)
            if distance < min_distance:
                min_distance = distance
            if distance > max_distance:
                max_distance = distance

            # Try to perform the action
            try_count += 1

            if self.is_crouched():
                standing = "(crouched)"
            else:
                standing = "(standing)"
            Utils.print_overwrite(
                c.Color.LIGHT_BLUE,
                f"    Try [{try_count}/{num_poses}] {action_name} {object_name} at {distance:.2f}m Margin: {pose['extra_range']:.2f} [{min_distance:.2f}-{max_distance:.2f}m] {standing}",
            )
            action_success = self.try_execute_action(
                object_name, agent_action, pose["extra_range"]
            )
            if not action_success:
                # These errors will always fail on retry
                if self.was_impossible_action():
                    Utils.print_end_overwrite()
                    return action_success
                continue

            Utils.print_end_overwrite()
            return action_success

    def create_video(self):
        """Create a web-compatible5 MP4 video that will play in Streamlit."""
        if not self.record_video or not self.video_frames:
            return

        try:
            import imageio

            # Use imageio instead of OpenCV for browser-compatible video
            movie_path = f"{self.save_dir}/video.mp4"

            # Convert frames to numpy arrays
            frames = []
            for frame in self.video_frames:
                np_frame = np.array(frame)
                frames.append(np_frame)

            # Write video with imageio using h264 codec
            imageio.mimsave(
                movie_path, frames, fps=self.frame_rate, quality=7, codec="libx264"
            )

        except ImportError:
            print("Missing libraries. Install with: pip install imageio imageio-ffmpeg")
        except Exception as e:
            print(f"Error creating video: {str(e)}")
