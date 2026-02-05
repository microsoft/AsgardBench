from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import os
import random
from datetime import datetime
from typing import List

import numpy as np

import AsgardBench.constants as c
import AsgardBench.utils as Utils
from AsgardBench.cache.item_cache import ItemCache
from AsgardBench.defective_containers import DefectiveContainers
from AsgardBench.goal import ActionGoal, Goal
from AsgardBench.plan import PlanType
from AsgardBench.randomization import Randomization
from AsgardBench.scenario import Scenario
from AsgardBench.scenes import Scenes
from AsgardBench.setup import ObjectSetup, SetupAction
from AsgardBench.specifier import Specifier
from AsgardBench.Utils.count_plans import PlanStats, count_plans


class PlanGenerator:
    def __init__(
        self,
        skip_completed: bool = False,
        # Stop creating more examples after this many have been created
        max_examples: int = 3,
        # Number of examples to create
        num_to_create: int = 3,
        # For each example how many to create with random injected states
        num_injections: int = 1,
        # Optional list of plan name patterns to filter by (e.g., ["cook__Egg", "coffee__"])
        # Only plans containing one of these substrings will be generated
        plan_filters: List[str] = None,
    ):
        # If True, skip items where the task is already completed
        self.skip_completed = skip_completed

        self.max_examples = max_examples
        self.num_to_create = num_to_create
        self.num_injections = num_injections
        self.plan_filters = plan_filters

        # Clear failures from previous run and load existing plans from disk
        from AsgardBench.Utils.count_plans import clear_failures

        clear_failures()

        self.plan_stats: PlanStats = count_plans()
        self.plan_stats.print_stats(max_samples=max_examples)

        self.failure_reasons = {}
        self.starttime = datetime.now()
        self.data_folder = f"{c.DATASET_DIR}"

        self.all_types = set()

        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

    def add_results(self, scenario: Scenario, is_injection: bool):

        scenario.complete()

        plan_name = scenario.raw_plan.name
        scene = scenario.raw_plan.scene
        steps = scenario.raw_plan.step_count()

        if scenario.raw_plan.task_failed is True:
            self.plan_stats.add_failure(plan_name, scene, steps)
            self.failure_reasons[scenario.raw_plan.name] = scenario.plan_errors[-1]
        elif is_injection:
            self.plan_stats.add_success_injection(plan_name, scene, steps)
        else:
            self.plan_stats.add_success(plan_name, scene, steps)

        self.print_results()
        self.save_results()

    def has_enough_examples(self, plan_name: str) -> bool:
        if self.plan_stats.num_plans(plan_name) >= self.max_examples:
            Utils.print_color(
                c.Color.PURPLE,
                f"{plan_name:<60}: already have {self.max_examples} examples",
            )
            return True
        return False

    def matches_plan_filter(self, plan_name: str) -> bool:
        """Check if plan name matches any of the configured filters.

        Returns True if no filters are set, or if plan_name starts with any filter pattern.
        Examples:
            - "cook__Egg" matches "cook__Egg_clean_FloorPlan1_V1"
            - "cook__Egg_Pan(d)" matches "cook__Egg_Pan(d)_FloorPlan1_V1" but not "cook__Egg_clean_FloorPlan1_V1"
            - "cook__Egg" does NOT match "cook__PotatoSliced_Pan(d)_Plate(d)"
        """
        if self.plan_filters is None or len(self.plan_filters) == 0:
            return True  # No filters = match everything
        for filter_pattern in self.plan_filters:
            if plan_name.startswith(filter_pattern):
                return True
        return False

    def should_skip_plan(self, scene: str, plan_name: str) -> bool:
        """Check if plan should be skipped (doesn't match filter, already exists, has enough examples, or family limit reached)."""
        # If filters are set, skip plans that don't match any filter
        if not self.matches_plan_filter(plan_name):
            return True
        if self.plan_exists(scene, plan_name):
            return True
        if self.has_enough_examples(plan_name):
            return True
        if self.has_enough_family_examples(plan_name):
            return True
        return False

    def has_enough_family_examples(self, plan_name: str) -> bool:
        """Check if the family has reached its limit (3x max_examples)."""
        family_limit = 3 * self.max_examples
        num_family = self.plan_stats.num_family_plans(plan_name)
        if num_family >= family_limit:
            Utils.print_color(
                c.Color.PURPLE,
                f"{plan_name:<60}: family already has {num_family} examples (limit: {family_limit})",
            )
            return True
        return False

    def plan_exists(self, scene: str, plan_name: str) -> bool:

        if self.skip_completed is False:
            return False

        if self.plan_stats.plan_exists(plan_name):
            Utils.print_color(
                c.Color.PURPLE,
                f"{plan_name:<60} already exists in {c.DATASET_DIR}/{c.NEW_PLANS_DIR}",
            )
            return True
        return False

    def print_results(self, file=None):

        Utils.print_color(
            c.Color.LIGHT_BLUE,
            f"{c.DATASET_DIR}/{c.NEW_PLANS_DIR}",
        )

        totals = self.plan_stats.print_stats(file, max_samples=self.max_examples)

        if (totals["success"] + totals["failures"]) == 0:
            if file is not None:
                file.write("No tasks completed.\r\n")
            else:
                print("No tasks completed.")
            return

        passed_time = datetime.now() - self.starttime
        per_task = passed_time / (totals["success"] + totals["failures"])
        time_str = f"Run time: {passed_time.total_seconds() / 60:.1f} minutes, Per Task: {per_task.seconds} seconds"
        if file is not None:
            file.write(f"{time_str}\r\n")
        else:
            print(time_str)
        print("-------------------------------")

    def save_results(self):
        with open(f"{self.data_folder}/results.txt", "w") as f:
            self.print_results(f)

        with open(f"{self.data_folder}/errors.txt", "w") as f:
            for key, value in self.failure_reasons.items():
                f.write(f"{key}: {value}\r\n")

    def end_run(self):
        self.save_results()
        self.print_results()

    def can_fit(self, scene: str, container_names: list, object_type: str) -> bool:
        """
        Check if an object can fit in one of the containers
        """
        for container_name in container_names:
            if not DefectiveContainers.is_defective(scene, container_name, object_type):
                return True

        Utils.print_color(
            c.Color.PURPLE,
            f"Skip: {object_type} as won't fit in {Utils.join_with_or(container_names)} in {scene}.",
        )
        return False

    def get_dirty_types(self, starting_condition: str) -> list[str]:
        """
        Extract dirty object types from a starting condition.

        Examples:
            "Plate(d)" -> ["Plate"]
            "Pan(d)_Plate(d)" -> ["Pan", "Plate"]
            "clean" -> []
            "empty" -> []
        """
        if "(d)" not in starting_condition:
            return []

        dirty_types = []
        parts = starting_condition.split("_")
        for part in parts:
            if part.endswith("(d)"):
                dirty_types.append(part[:-3])  # Remove "(d)" suffix
        return dirty_types

    def add_clean_goal(
        self, obj_type: str, scene: str, goal: Goal, setup_actions: List[SetupAction]
    ):
        goal.add_state_goal(obj_type, "isDirty", False)
        items = ItemCache.get_names_by_type(scene, obj_type)
        item = items[0]
        setup_actions.append(SetupAction(c.Action.DIRTY, item))

    def plan_cook(
        self,
        scene,
        object_type,
        starting_condition: str,
        flags=[],
        num_to_create=None,
        num_injections=None,
        task_description="",
    ):

        if scene == "FloorPlan7" and object_type == "Egg":
            print("Skipping FloorPlan7 due to known issues with unusable pan.")
            return

        server_type = c.SERVE_CONTAINER[object_type]
        bad_container = DefectiveContainers.is_defective(
            scene, server_type, object_type
        )
        if bad_container:
            Utils.print_color(
                c.Color.RED,
                f"Skipping {object_type} in {scene} due to defective containers.",
            )
            return

        names = ItemCache.get_scene_names(scene)

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        # Get list of basin names from names array
        basin_names = [name for name in names if "SinkBasin" in name]

        # Check if starting condition is valid for this scene
        dirty_types = self.get_dirty_types(starting_condition)
        for dirty_type in dirty_types:
            if not self.can_fit(scene, basin_names, dirty_type):
                return

        for i in range(num_to_create):
            # Start each variation with same seed so only diff is cup state
            seed = random.randint(0, 10000)

            cook_name = object_type
            if (object_type == "Potato") and (c.Flags.DONT_SLICE not in flags):
                cook_name = "PotatoSliced"

            # Build condition suffix for naming
            if starting_condition == "" or starting_condition == "clean":
                condition_suffix = ""
            else:
                condition_suffix = f"_{starting_condition}"

            base_name = f"cook__{cook_name}{condition_suffix}_{scene}_V{i+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            # Randomize starting conditions
            randomization = Randomization(seed)

            # Make sure things I need are visible
            visible_types = [
                object_type,
                "DishSponge",
                "Pan",
                "Plate",
                "Bowl",
                "Knife",
            ]

            object_setup = ObjectSetup()
            object_setup.employed_types = visible_types

            visible_specifier = Specifier(types=visible_types, observe=False)
            object_setup.add_target(visible_specifier)

            # Randomize distractors
            distractor_classes = ["Kitchen", "Silverware", "Food"]
            distractor_specifier = Specifier(classes=distractor_classes, observe=False)
            object_setup.add_distractors(distractor_specifier)

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=[]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = base_name

                goal = Goal(room="Kitchen")

                # Any cooking items need to be turned off and closed
                # and should be on serving container
                if object_type == "Bread":
                    goal.add_location_goal("BreadSliced", "Plate")
                    goal.add_state_goal(object_type, "isCooked", True)
                elif object_type == "Egg":
                    goal.add_location_goal("EggCracked", "Plate")
                    goal.add_state_goal(object_type, "isCooked", True)
                elif object_type == "Potato":
                    if c.Flags.DONT_SLICE in flags:
                        goal.add_location_goal("Potato", "Bowl")
                        goal.add_state_goal("Potato", "isCooked", True)
                    else:
                        goal.add_location_goal("PotatoSliced", "Plate")
                        goal.add_state_goal("PotatoSliced", "isCooked", True)

                setup_actions: List[SetupAction] = []
                for dirty_type in dirty_types:
                    self.add_clean_goal(dirty_type, scene, goal, setup_actions)

                scenario = Scenario(
                    task=task_description,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    setup_actions=setup_actions,
                    object_setup=object_setup,
                    randomization=randomization,
                    goal=goal,
                )

                food_name = scenario.do(
                    c.Action.COOK, Specifier(types=[object_type]), "", flags
                )

                # TODO: goal check that item is on serving dish if it exists
                scenario.do(c.Action.SERVE, Specifier(names=[food_name]), "", flags)

                self.add_results(scenario, r > 0)

                # Save number of steps on on-injected do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break

    def plan_set_table(
        self,
        scene: str,
        starting_condition: str,
        num_to_create=None,
        num_injections=None,
    ):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        types = ItemCache.get_scene_types(scene)

        # Must have a dining table
        if "DiningTable" not in types:
            return

        # Skip floor plan with broken coffee machine
        if scene == "FloorPlan27":
            return

        mug_name = ItemCache.get_names_by_type(scene, "Mug")[0]

        for v in range(num_to_create):
            # Start each variation with same seed so only diff is cup state
            seed = random.randint(0, 10000)

            # Build condition suffix for naming - just use the condition directly
            base_name = f"settable__{starting_condition}_{scene}_V{v+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            # Randomize starting conditions
            randomization = Randomization(seed=seed)

            visible_types = ["DishSponge", "Mug", "Plate", "Bread", "Knife"]

            object_setup = ObjectSetup()
            object_setup.employed_types = visible_types

            # Randomize distractors
            distractor_classes = ["Kitchen", "Silverware", "Food"]
            distractor_specifier = Specifier(classes=distractor_classes, observe=False)
            object_setup.add_distractors(distractor_specifier)
            task = "Clear the dining table and set it with a mug of coffee and slice of toast"

            # Make sure things I need are visible
            visible_types = ["DishSponge", "Mug", "Plate", "Bread", "Knife"]
            visible_specifier = Specifier(types=visible_types, observe=False)
            object_setup.add_target(visible_specifier)

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=[]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = f"{base_name}"

                # TODO
                goal = Goal(room="Kitchen")
                goal.add_state_goal("Mug", "isFilledWithLiquid", True)
                goal.add_contents_goal("DiningTable", ["Mug", "Plate", "BreadSliced"])
                goal.add_location_goal("BreadSliced", "Plate")
                setup_actions: List[SetupAction] = []

                if starting_condition == "empty":
                    setup_actions.append(SetupAction(c.Action.EMPTY_LIQUID, mug_name))
                elif starting_condition == "Mug(d)":
                    self.add_clean_goal("Mug", scene, goal, setup_actions)
                elif starting_condition == "has_water":
                    setup_actions.append(
                        SetupAction(c.Action.FILL, mug_name, c.LiquidType.WATER)
                    )
                elif starting_condition == "has_coffee":
                    setup_actions.append(
                        SetupAction(c.Action.FILL, mug_name, c.LiquidType.COFFEE)
                    )
                else:
                    raise ValueError(
                        f"Unknown starting condition: {starting_condition}"
                    )

                scenario = Scenario(
                    task=task,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    setup_actions=setup_actions,
                    object_setup=object_setup,
                    randomization=randomization,
                    goal=goal,
                )

                # If more than one dining table, skip as ambiguous
                dining_tables = scenario.get_objs_by_types(["DiningTable"])
                if len(dining_tables) > 1:
                    Utils.print_color(
                        c.Color.RED, "Skipping: Ambiguous dining table found"
                    )
                    continue

                mugs = scenario.get_objs_by_types(["Mug"])
                mug = mugs[0]

                mug_specifier = Specifier(names=[mug["name"]])
                table_specifier = Specifier(types=["DiningTable"])
                plate_specifier = Specifier(types=["Plate"])

                # TODO: excluded should be part of specifier
                Specifier.excluded_objects = ["Plate", "Mug"]
                Specifier.excluded_containers = ["DiningTable", "Plate", "Mug"]
                scenario.do(
                    c.Action.CLEAR_CONTAINER,
                    table_specifier,
                    "First I need to empty the dining table",
                )
                Specifier.excluded_objects = []
                Specifier.excluded_containers = []

                scenario.do(
                    c.Action.PICKUP,
                    mug_specifier,
                    "Next, I need to pick up the mug to make coffee",
                )
                if starting_condition != "has coffee":
                    scenario.do(
                        c.Action.BREW,
                        mug_specifier,
                        "Then, I need to brew the coffee",
                    )
                scenario.do(
                    c.Action.PUT,
                    table_specifier,
                    "I need to put the mug on the table",
                )
                scenario.do(
                    c.Action.PICKUP,
                    plate_specifier,
                    "Next, I need to pickup the Plate to put in on the table",
                )
                scenario.do(
                    c.Action.PUT,
                    table_specifier,
                    "I need to put the plate on the table",
                )
                food_name = scenario.do(
                    c.Action.COOK,
                    Specifier(types=["Bread"]),
                    "Now I need to make toast",
                )
                scenario.do(
                    c.Action.SERVE,
                    Specifier(names=[food_name]),
                    "Now put the toast on the plate",
                )

                self.add_results(scenario, r > 0)

                # Save number of steps do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break

    def plan_slice_and_distribute(
        self,
        scene: str,
        slicable_type,
        starting_condition: str,
        serving_types: List[str],
        num_to_create=None,
        num_injections=None,
    ):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        sliced_type = c.SLICED_TYPES[slicable_type]
        names = ItemCache.get_scene_names(scene)
        types = ItemCache.get_scene_types(scene)

        # Validate that all requested serving_types exist in this scene
        for serving_type in serving_types:
            if serving_type not in types:
                Utils.print_color(
                    c.Color.PURPLE,
                    f"Skip: {serving_type} not in {scene}.",
                )
                return

        # Check that serving types can fit the sliced type
        for serving_type in serving_types:
            serving_names = Utils.get_names_by_type(serving_type, names)
            if not self.can_fit(scene, serving_names, sliced_type):
                return

        # Get list of basin names from names array
        basin_names = [name for name in names if "SinkBasin" in name]

        # Check if starting condition is valid for this scene
        dirty_types = self.get_dirty_types(starting_condition)
        for dirty_type in dirty_types:
            if not self.can_fit(scene, basin_names, dirty_type):
                return

        # Check that all serving types work - if any is defective, return
        # (the subset combinations are already in plan_configs)
        for serving_type in serving_types:
            if DefectiveContainers.is_defective(scene, serving_type, sliced_type):
                Utils.print_color(
                    c.Color.YELLOW,
                    f"Skipping {serving_type} in {scene} due to defective containers for {sliced_type}.",
                )
                return

        # Build object_names with (d) suffix for dirty items
        # e.g., "Plate(d)_Bowl" for Plate(d) condition with Plate and Bowl
        name_parts = []
        for st in serving_types:
            if st in dirty_types:
                name_parts.append(f"{st}(d)")
            else:
                name_parts.append(st)
        object_names = "_".join(name_parts)

        for v in range(num_to_create):
            # Start each variation with same seed
            seed = random.randint(0, 10000)

            base_name = f"distribute__{slicable_type}_{object_names}_{scene}_V{v+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            randomization = Randomization(seed)

            # Make sure things I need are visible
            visible_types = [
                slicable_type,
                "DishSponge",
                "Knife",
                "Pan",
                "Plate",
                "Bowl",
                "Pot",
            ]

            object_setup = ObjectSetup()
            object_setup.employed_types = visible_types

            visible_specifier = Specifier(types=visible_types, observe=False)
            object_setup.add_target(visible_specifier)

            # Randomize distractors and put some in sink
            distractor_classes = ["Kitchen", "Silverware"]
            distractor_specifier = Specifier(classes=distractor_classes, observe=False)
            object_setup.add_distractors(distractor_specifier)

            dishes = Utils.join_with_and(serving_types)
            task = f"Slice the {slicable_type} and put a piece in the {dishes}"

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=[]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = base_name

                # Create goal
                goal = Goal(room="Kitchen")
                for container_type in serving_types:
                    goal.add_location_goal(
                        object_type=sliced_type, destination_type=container_type
                    )

                setup_actions: List[SetupAction] = []
                for dirty_type in dirty_types:
                    self.add_clean_goal(dirty_type, scene, goal, setup_actions)

                scenario = Scenario(
                    task=task,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    setup_actions=setup_actions,
                    object_setup=object_setup,
                    randomization=randomization,
                    goal=goal,
                )

                unsliced_name = scenario.do(
                    c.Action.SLICE,
                    Specifier(types=[slicable_type]),
                    f"I need to slice the {slicable_type}",
                )

                unsliced_obj = scenario.get_obj_by_name(unsliced_name)
                sliced_names = scenario.get_sliced_object_names(unsliced_obj)

                if sliced_names is None or len(sliced_names) == 0:
                    Utils.print_color(
                        c.Color.RED,
                        f"No sliced {slicable_type} found in {scene}.",
                    )
                    break

                food_specifier = Specifier(names=sliced_names, all=False)

                dishes_or = Utils.join_with_or(serving_types)

                def get_unused_slices_and_containers() -> str:
                    unused_slices = []
                    unused_containers = serving_types.copy()

                    for name in sliced_names:
                        container_obj = scenario.get_non_surface_container(
                            name, must_exist=False
                        )
                        if container_obj is None:
                            unused_slices.append(name)
                        elif container_obj["objectType"] in serving_types:
                            if container_obj["objectType"] in unused_containers:
                                unused_containers.remove(container_obj["objectType"])
                        else:
                            unused_slices.append(name)

                    print(
                        f"Unused slices: {unused_slices}, Unused containers: {unused_containers}"
                    )
                    return unused_slices, unused_containers

                # Loop for length of serving_types
                for i in range(len(serving_types)):
                    unused_slices, unused_containers = (
                        get_unused_slices_and_containers()
                    )
                    if len(unused_slices) == 0:
                        Utils.print_color(
                            c.Color.RED,
                            f"No unused slices found for {slicable_type} in {scene}.",
                        )
                        return

                    food_specifier = Specifier(names=unused_slices)
                    container_specifier = Specifier(types=unused_containers)

                    scenario.do(
                        c.Action.PICKUP,
                        food_specifier,
                        f"I need to pick up the sliced {slicable_type} that isn't already in a {dishes_or}",
                    )

                    scenario.do(c.Action.PUT, container_specifier, "")

                self.add_results(scenario, r > 0)

                # Save number of steps on non-injected do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break

    def plan_cleanup(
        self,
        scene,
        object_classes,
        starting_condition: str,
        num_to_create=None,
        num_injections=None,
    ):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        # Get list of object types that are in the scene
        names = ItemCache.get_scene_names(scene)
        types = ItemCache.get_scene_types(scene)

        # Filter to object_classes
        obj_types = []
        for obj_type in types:
            for object_class in object_classes:
                if obj_type in c.CLASS_TO_TYPES[object_class]:
                    if obj_type in types:
                        obj_types.append(obj_type)

        type_combinations = Utils.create_all_combinations(obj_types)

        # TODO: handle not in fridge but after randomization
        """# Get random combinations of objects
        types_combinations = Scenario.scene_get_class_combinations(
            scene, object_classes, filters=[c.FilterType.NOT_IN_FRIDGE]
        )"""

        # Limits the number of combinations to create to num_to_create
        type_combinations = type_combinations[:num_to_create]

        # Get list of basin names from names array
        basin_names = [name for name in names if "SinkBasin" in name]

        # Check if starting condition is valid for this scene

        scene_types = ItemCache.get_scene_types(scene)

        dirty_types = self.get_dirty_types(starting_condition)
        for dirty_type in dirty_types:
            if not self.can_fit(scene, basin_names, dirty_type):
                Utils.print_color(
                    c.Color.PURPLE,
                    f"Skip: {dirty_type} cannot fit in sink in {scene}.",
                )
                return
            if not dirty_type in scene_types:
                Utils.print_color(
                    c.Color.PURPLE,
                    f"Skip: {dirty_type} not in {scene}.",
                )
                return

        for v in range(num_to_create):

            for combination in type_combinations:

                # Start each variation with same seed so only diff is cup state
                seed = random.randint(0, 10000)

                # Make base name
                object_names = " ".join(combination)
                class_string = " ".join(object_classes)

                # Build condition suffix for naming
                if starting_condition == "" or starting_condition == "clean":
                    condition_suffix = ""
                else:
                    condition_suffix = f"_{starting_condition}"

                base_name = f"putaway__{class_string}{condition_suffix}_{scene}_V{v+1}"

                # Skip if already exists or have enough examples
                if self.should_skip_plan(scene, base_name):
                    continue

                # Randomize starting conditions for distractors
                randomization = Randomization(seed)

                # Make sure things I need are visible
                visible_types = ["DishSponge"]

                object_setup = ObjectSetup()
                object_setup.employed_types = ["DishSponge"]

                for object_class in object_classes:
                    visible_types.extend(c.CLASS_TO_TYPES[object_class])
                object_specifier = Specifier(types=visible_types, observe=False)
                object_setup.add_target(object_specifier)

                # Randomize distractors
                distractor_classes = ["Kitchen"]
                for object_class in ["Silverware", "Dishes", "Food"]:
                    if object_class not in object_classes:
                        distractor_classes.extend(c.CLASS_TO_TYPES[object_class])
                distractor_specifier = Specifier(
                    classes=distractor_classes, observe=False
                )
                object_setup.add_distractors(distractor_specifier)

                task = f"Put away the {Utils.join_with_and(combination)}"

                # Run scenarios
                action_counts = {}
                for r in range(num_injections + 1):
                    if r > 0:
                        injection_type = randomization.add_error_injection(
                            action_counts, avoid_types=[]
                        )
                        if injection_type is None:
                            continue
                        name = f"{base_name}R{r}_{injection_type.value}-{object_names}"
                    else:
                        name = f"{base_name}-{object_names}"

                    goal = Goal(room="Kitchen")
                    setup_actions = []
                    for dirty_type in dirty_types:
                        self.add_clean_goal(dirty_type, scene, goal, setup_actions)

                    for object_type in combination:
                        goal.add_put_away_goal(object_type=object_type)
                    scenario = Scenario(
                        task=task,
                        scene=scene,
                        name=name,
                        plan_type=PlanType.GENERATED,
                        data_folder=self.data_folder,
                        setup_actions=setup_actions,
                        object_setup=object_setup,
                        randomization=randomization,
                        goal=goal,
                    )

                    scenario.do(
                        c.Action.PUT_AWAY,
                        Specifier(types=combination, all=True),
                        "",
                        flags=[c.Flags.CLEAN_WHEN_PUTTING_AWAY],
                    )

                    self.add_results(scenario, r > 0)

                    # Save number of steps on non-injected do know range I can inject into
                    if r == 0:
                        action_counts = scenario.action_counts
                        # If base plan failed, we don't want to inject errors
                        if scenario.raw_plan.task_failed:
                            break

    def plan_make_coffee(
        self, scene, starting_condition: str, num_to_create=None, num_injections=None
    ):

        # Skip floor plan with broken coffee machine
        if scene == "FloorPlan27":
            return

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        mug_name = ItemCache.get_names_by_type(scene, "Mug")[0]
        starting_conditions = [starting_condition]

        for v in range(num_to_create):

            # Start each variation with same seed so only diff is cup state
            seed = random.randint(0, 10000)
            for starting_condition in starting_conditions:

                # Make base name - just use condition directly
                base_name = f"coffee__{starting_condition}_{scene}_V{v+1}"

                # Skip if already exists or have enough examples
                if self.should_skip_plan(scene, base_name):
                    continue

                # Randomize starting conditions
                randomization = Randomization(seed=seed)

                # Make sure the mug is visible
                object_setup = ObjectSetup()
                object_setup.employed_types = ["DishSponge"]

                visible_types = ["Mug", "DishSponge"]
                visible_specifier = Specifier(types=visible_types, observe=False)
                object_setup.add_target(visible_specifier)

                # Randomize distractors
                distractor_classes = ["Kitchen", "Silverware", "Food"]
                distractor_specifier = Specifier(
                    classes=distractor_classes, observe=False
                )
                object_setup.add_distractors(distractor_specifier)
                task = "Consume coffee from a mug, then wash and store the mug"

                # Run scenarios
                action_counts = {}
                for r in range(num_injections + 1):
                    if r > 0:
                        injection_type = randomization.add_error_injection(
                            action_counts, avoid_types=[]
                        )
                        if injection_type is None:
                            continue
                        name = f"{base_name}R{r}_{injection_type.value}"
                    else:
                        name = f"{base_name}"

                    goal = Goal(room="Kitchen")
                    goal.add_action_goal(ActionGoal.DRINK_COFFEE)
                    goal.add_state_goal("Mug", "isDirty", False)
                    goal.add_state_goal("Mug", "isFilledWithLiquid", False)
                    goal.add_put_away_goal(object_type="Mug")
                    setup_actions: List[SetupAction] = []

                    if starting_condition == "empty":
                        setup_actions.append(
                            SetupAction(c.Action.EMPTY_LIQUID, mug_name)
                        )
                    elif starting_condition == "Mug(d)":
                        self.add_clean_goal("Mug", scene, goal, setup_actions)
                    elif starting_condition == "has_water":
                        setup_actions.append(
                            SetupAction(c.Action.FILL, mug_name, c.LiquidType.WATER)
                        )
                    elif starting_condition == "has_coffee":
                        setup_actions.append(
                            SetupAction(c.Action.FILL, mug_name, c.LiquidType.COFFEE)
                        )
                    else:
                        raise ValueError(
                            f"Unknown starting condition: {starting_condition}"
                        )

                    scenario = Scenario(
                        task=task,
                        scene=scene,
                        name=name,
                        plan_type=PlanType.GENERATED,
                        data_folder=self.data_folder,
                        setup_actions=setup_actions,
                        object_setup=object_setup,
                        randomization=randomization,
                        goal=goal,
                    )

                    mugs = scenario.get_objs_by_types(["Mug"])
                    mug = mugs[0]

                    mug_specifier = Specifier(names=[mug["name"]])

                    scenario.do(c.Action.BREW_AND_DRINK, mug_specifier, "")

                    container_name = scenario.do(c.Action.SINK_WASH, mug_specifier, "")
                    self.add_results(scenario, r > 0)

                    # Save number of steps do know range I can inject into
                    if r == 0:
                        action_counts = scenario.action_counts
                        # If base plan failed, we don't want to inject errors
                        if scenario.raw_plan.task_failed:
                            break

    def plan_put_toilet_paper(self, scene, num_to_create=None, num_injections=None):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        for v in range(num_to_create):

            # Make base name
            base_name = f"put_toilet_paper_{scene}_V{v+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            randomization = Randomization(seed=random.randint(0, 10000))

            # Make sure things I need are visible
            object_setup = ObjectSetup()
            visible_types = ["ToiletPaper"]
            visible_specifier = Specifier(types=visible_types, observe=False)
            object_setup.add_target(visible_specifier)

            # Randomize other objects
            distractor_specifier = Specifier(
                types=c.CLASS_TO_TYPES["Bathroom"], observe=False
            )
            cont_specifier = Specifier(
                types=["CounterTop", "Floor", "Shelf"], observe=False
            )
            object_setup.add_distractors(distractor_specifier, cont_specifier)

            task = "Put toilet paper on the holder"

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    # HandTowel has no valid home, so avoid it
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=["HandTowel"]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = base_name

                goal = Goal(room="Bathroom")
                goal.add_location_goal(
                    object_type="ToiletPaper", destination_type="ToiletPaperHanger"
                )
                scenario = Scenario(
                    task=task,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    randomization=randomization,
                    object_setup=object_setup,
                    goal=goal,
                )

                scenario.do(
                    c.Action.PICKUP,
                    Specifier(types=["ToiletPaper"]),
                    "",
                    flags=[c.Flags.NOT_USED_UP],
                )
                scenario.do(c.Action.PUT, Specifier(types=["ToiletPaperHanger"]), "")

                self.add_results(scenario, r > 0)

                # Save number of steps do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break

    def plan_clean_mirror(self, scene, num_to_create=None, num_injections=None):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        for v in range(num_to_create):

            # Make base name
            base_name = f"clean_mirror__{scene}_V{v+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            # Randomize starting conditions
            randomization = Randomization(seed=random.randint(0, 10000))

            # Make sure things I need are visible
            visible_types = ["Cloth", "SprayBottle"]

            object_setup = ObjectSetup()
            object_setup.employed_types = visible_types
            visible_specifier = Specifier(types=visible_types, observe=False)
            object_setup.add_target(visible_specifier)

            # Randomize other objects
            distractor_specifier = Specifier(
                types=c.CLASS_TO_TYPES["Bathroom"], observe=False
            )
            cont_specifier = Specifier(
                types=["CounterTop", "Floor", "Shelf"], observe=False
            )
            object_setup.add_distractors(distractor_specifier, cont_specifier)

            task = "Clean the mirror"

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    # HandTowel has no valid home, so avoid it
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=["HandTowel"]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = base_name

                goal = Goal(room="Bathroom")
                setup_actions: List[SetupAction] = []
                goal.add_action_goal(ActionGoal.SPRAY_MIRROR)
                self.add_clean_goal("Mirror", scene, goal, setup_actions)

                # TODO prevent clean action w/o rag
                scenario = Scenario(
                    task=task,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    setup_actions=setup_actions,
                    object_setup=object_setup,
                    randomization=randomization,
                    goal=goal,
                )

                mirrors = scenario.get_objs_by_types(["Mirror"])
                mirror_name = mirrors[0]["name"]
                scenario.do(
                    c.Action.WASH,
                    Specifier(names=[mirror_name]),
                    "",
                    flags=[c.Flags.NOT_USED_UP],
                )

                self.add_results(scenario, r > 0)

                # Save number of steps do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break

    def plan_turn_on_tv(self, scene, num_to_create=None, num_injections=None):

        if num_to_create is None:
            num_to_create = self.num_to_create
        if num_injections is None:
            num_injections = self.num_injections

        for v in range(num_to_create):

            # Make base name
            base_name = f"turn_on_tv__{scene}_V{v+1}"

            # Skip if already exists or have enough examples
            if self.should_skip_plan(scene, base_name):
                continue

            # Randomize starting conditions
            randomization = Randomization(seed=random.randint(0, 10000))

            # Make sure things I need are visible
            object_setup = ObjectSetup()
            visible_types = ["RemoteControl"]
            visible_specifier = Specifier(types=visible_types, observe=False)
            container_types = [
                "CoffeeTable",
                "CounterTop",
                "Floor",
                "Shelf",
                "SideTable",
            ]
            container_specifier = Specifier(types=container_types, observe=False)
            object_setup.add_target(visible_specifier, container_specifier)

            # Randomize other objects
            objects = [
                "Boots",
                "Bowl",
                "Box",
                "Candle",
                "HousePlant",
                "Statue",
                "TissueBox",
                "WateringCan",
                "Vase",
            ]
            containers = ["CounterTop", "Floor", "Shelf", "SideTable", "ShelvingUnit"]
            object_specifier = Specifier(types=objects, observe=False)
            container_specifier = Specifier(types=containers, observe=False)
            object_setup.add_distractors(object_specifier, container_specifier)

            objects = [
                "Book",
                "CellPhone",
                "CreditCard",
                "KeyChain",
                "Laptop",
                "Newspaper",
                "Pen",
                "Pillow",
                "Pencil",
                "TissueBox",
                "Watch",
            ]
            containers = ["CounterTop", "Shelf", "Sofa", "SideTable", "ShelvingUnit"]
            object_specifier = Specifier(types=objects, observe=False)
            container_specifier = Specifier(types=containers, observe=False)
            object_setup.add_distractors(object_specifier, container_specifier)

            task = "Turn on the television"

            # Run scenarios
            action_counts = {}
            for r in range(num_injections + 1):
                if r > 0:
                    # HandTowel has no valid home, so avoid it
                    injection_type = randomization.add_error_injection(
                        action_counts, avoid_types=["HandTowel"]
                    )
                    if injection_type is None:
                        continue
                    name = f"{base_name}R{r}_{injection_type.value}"
                else:
                    name = base_name

                goal = Goal(room="LivingRoom")
                goal.add_state_goal("Television", "isToggled", True)
                scenario = Scenario(
                    task=task,
                    scene=scene,
                    name=name,
                    plan_type=PlanType.GENERATED,
                    data_folder=self.data_folder,
                    object_setup=object_setup,
                    randomization=randomization,
                    goal=goal,
                )

                televisions = scenario.get_objs_by_types(["Television"])
                tv_name = televisions[0]["name"]
                scenario.do(
                    c.Action.TOGGLE_ON,
                    Specifier(names=[tv_name]),
                    "",
                    flags=[c.Flags.NOT_USED_UP],
                )

                self.add_results(scenario, r > 0)

                # Save number of steps do know range I can inject into
                if r == 0:
                    action_counts = scenario.action_counts
                    # If base plan failed, we don't want to inject errors
                    if scenario.raw_plan.task_failed:
                        break


if __name__ == "__main__":

    # If plan_filters is set, will only create plans of this type
    plan_filters = [
        "cook__Egg",
        "cook__Egg_Pan(d)",
        "cook__PotatoSliced_Plate(d)",
        "coffee__has_coffee",
        "cook__PotatoSliced",
    ]
    plan_generator = PlanGenerator(
        skip_completed=True,
        num_to_create=1,
        num_injections=0,
        plan_filters=plan_filters,
    )

    """
    Living Room
    """
    max_plans = 0
    livingrooms = Scenes.get_living_rooms()
    np.random.shuffle(livingrooms)
    count = 0
    for livingroom in livingrooms:
        if count >= max_plans:
            break
        plan_generator.plan_turn_on_tv(livingroom)
        count += 1

    """
    Bathroom
    """
    bathrooms = Scenes.get_bathrooms()
    max_plans = 0
    np.random.shuffle(bathrooms)
    count = 0
    for bathroom in bathrooms:
        if count >= max_plans:
            break
        plan_generator.plan_clean_mirror(bathroom)
        # runner.plan_put_toilet_paper(bathroom)
        count += 1

    """
    Kitchen - Diagonal interleaving so early termination gives variety in both plans and kitchens
    """
    kitchens = Scenes.get_kitchens()
    np.random.shuffle(kitchens)
    """kitchens = [
        "FloorPlan3",
    ]"""

    # Build plan_configs using loops for cleaner organization
    plan_configs = []

    # Coffee plans
    for cond in ["empty", "Mug(d)", "has_coffee"]:
        plan_configs.append(
            (plan_generator.plan_make_coffee, {"starting_condition": cond})
        )

    # Slice and distribute plans - generate all combinations of serving types
    from itertools import combinations

    slicable_types = ["Lettuce", "Apple", "Tomato"]
    all_serving_types = ["Plate", "Bowl", "Pan", "Pot"]
    # Generate all non-empty subsets of serving types
    all_serving_combos = []
    for r in range(1, len(all_serving_types) + 1):
        all_serving_combos.extend(combinations(all_serving_types, r))
    all_serving_combos = [list(combo) for combo in all_serving_combos]
    random.shuffle(all_serving_combos)

    for slicable in slicable_types:
        for serving_types in all_serving_combos:
            # Always include clean condition
            conditions = ["clean"]
            # Add dirty conditions for each serving type in this combo
            for st in serving_types:
                conditions.append(f"{st}(d)")
            for cond in conditions:
                plan_configs.append(
                    (
                        plan_generator.plan_slice_and_distribute,
                        {
                            "slicable_type": slicable,
                            "serving_types": serving_types,
                            "starting_condition": cond,
                        },
                    )
                )

    # Cleanup plans - with starting conditions
    cleanup_dirty_conditions = {
        "Dishes": ["clean", "Plate(d)", "Bowl(d)", "Mug(d)", "Cup(d)"],
        "Silverware": ["clean"],
        "Food": ["clean"],
    }
    for obj_class in ["Dishes", "Silverware", "Food"]:
        for cond in cleanup_dirty_conditions[obj_class]:
            plan_configs.append(
                (
                    plan_generator.plan_cleanup,
                    {"object_classes": [obj_class], "starting_condition": cond},
                )
            )

    # Cooking plans - defined as (object_type, conditions, flags, task_description)
    cook_configs = [
        (
            "Egg",
            ["clean", "Pan(d)", "Plate(d)", "Pan(d)_Plate(d)"],
            [c.Flags.SLICE_ON_PAN],
            "Cook an egg in a pan and serve it on a plate.",
        ),
        (
            "Potato",
            ["clean", "Bowl(d)"],
            [c.Flags.DONT_SLICE],
            "Microwave a potato and serve it in a bowl",
        ),
        (
            "Bread",
            ["clean", "Plate(d)"],
            [],
            "Make a slice of toast and serve it on a plate",
        ),
        # Test Set - Fried potato
        (
            "Potato",
            ["clean", "Plate(d)", "Pan(d)", "Pan(d)_Plate(d)"],
            [],
            "Fry a potato slice and serve it on a plate",
        ),
    ]
    for obj_type, conditions, flags, task_desc in cook_configs:
        for cond in conditions:
            config = {
                "object_type": obj_type,
                "starting_condition": cond,
                "task_description": task_desc,
            }
            if flags:
                config["flags"] = flags
            plan_configs.append((plan_generator.plan_cook, config))

    # Set table plans
    for cond in ["empty", "Mug(d)", "has_coffee"]:
        plan_configs.append(
            (plan_generator.plan_set_table, {"starting_condition": cond})
        )

    # Diagonal interleaving: p1_k1, p2_k2, p3_k3, ..., p1_k2, p2_k3, p3_k4, ...
    # Each step advances both plan index and kitchen index (with offset)
    num_plans = len(plan_configs)
    num_kitchens = len(kitchens)
    total_combinations = num_plans * num_kitchens

    for i in range(total_combinations):
        plan_idx = i % num_plans
        kitchen_idx = i % num_kitchens
        plan_func, kwargs = plan_configs[plan_idx]
        kitchen = kitchens[kitchen_idx]
        plan_func(scene=kitchen, **kwargs)

    plan_generator.end_run()
