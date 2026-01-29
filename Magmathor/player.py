import copy
import json

import Magmathor.utils as Utils
from Magmathor import constants as c
from Magmathor.plan import Plan, PlanType
from Magmathor.scenario import Scenario
from Magmathor.specifier import Specifier
from Magmathor.step import Step
from Magmathor.Utils.config_utils import EvaluationConfig


class Player:
    """
    Allows manual control of the environment
    """

    def __init__(
        self,
        test_plan: Plan,
        plan_type: PlanType,
        config: EvaluationConfig,
        save_directory: str = None,
    ):
        self.scenario = Scenario(
            task=test_plan.task_description,
            scene=test_plan.scene,
            name=test_plan.name,
            plan_type=plan_type,
            data_folder=save_directory,
            hand_transparency=config.hand_transparency,
            object_setup=test_plan.object_setup,
            goal=test_plan.goal,
            setup_actions=test_plan.setup_actions,
            randomization=test_plan.randomization,
        )
        self.plan = copy.deepcopy(test_plan)

        # Clear out existing steps
        self.plan.steps = []
        self.step_number = 0

        # Unsed to pick the same object instance
        self.last_object_name_dict = {}
        self.last_action = None

    def start(self):
        self.scenario.capture_world_state()
        return self.scenario.cur_image

    def get_last_object_name(self, action: c.Action, object_type: str):

        # If toggling on a stove burner, pick the burner under the pan
        if action == c.Action.TOGGLE_ON and object_type == "StoveBurner":
            return self.scenario.get_burner_under_pan()

        last_object_name = None
        # If last action was a FIND or OPEN always resolve object type to last found instance
        if self.last_action == c.Action.FIND or self.last_action == c.Action.OPEN:
            if object_type in self.last_object_name_dict:
                last_object_name = self.last_object_name_dict[object_type]

        # If instance has been chosen for object type, pick same instance
        # but exclude PUT actions as may try to put in different locations
        elif (
            object_type in self.last_object_name_dict
            and self.last_object_name_dict[object_type] is not None
            and action != c.Action.PUT
        ):
            last_object_name = self.last_object_name_dict[object_type]

            # Don't select last object if it has been employed for a goal
            obj = self.scenario.get_obj_by_name(last_object_name)
            if self.scenario.is_used_for_goal(obj):
                last_object_name = None  # Reset if object is used for goal

        return last_object_name

    def step(self, action, object_type):

        if object_type is None:
            action_desc = f"{action}"
        else:
            action_desc = f"{action} {object_type}"

        # Save current image
        image_filename = f"{len(self.plan.steps)}_{action_desc}.png"
        self.scenario.cur_image.save(f"{self.scenario.save_dir}/{image_filename}")

        new_step = Step(
            action_desc=action_desc,
            action=action,
            obj=object_type,
            task_description=self.plan.task_description,
            observations=self.scenario.cur_observations,
            updated_memory=None,
            object_bounding_boxes=self.scenario.cur_bounding_boxes,
            pose=self.scenario.cur_pose,
            reasoning=[],
            filename=image_filename,
            history=[],
        )
        self.plan.steps.append(new_step)

        last_object_name = self.get_last_object_name(action, object_type)

        specifier = Specifier(types=[object_type], preferred_name=last_object_name)

        if last_object_name is not None:
            Utils.print_color(
                c.Color.LIGHT_BLUE,
                f"Executing action: {action_desc} -> {last_object_name}",
            )
        else:
            Utils.print_color(c.Color.LIGHT_BLUE, f"Executing action: {action_desc}")

        # Process to handle sliced object types
        object_type = Utils.short_name(object_type)

        if action == c.Action.FIND:
            self.last_object_name_dict[object_type] = self.scenario.do(
                c.Action.FIND, specifier, "", []
            )
        else:
            return_name = self.scenario.do(action, specifier, "", [])
            if return_name is not None:
                self.last_object_name_dict[object_type] = return_name

        # Set action_success based on whether there was an error
        new_step.action_success = self.scenario.step_error is None

        self.last_action = action

        plan_filename = f"{self.scenario.save_dir}/plan.json"
        with open(plan_filename, "w") as file:
            json.dump(self.plan.to_dict(), file)

        self.scenario.capture_world_state()

        return self.scenario.cur_image

    def get_last_action_success(self):
        return self.plan.steps[-1].action_success if self.plan.steps else None

    def get_plan(self) -> Plan:
        """
        Returns the plan with all steps taken so far.
        """
        return self.plan

    def complete(self) -> Plan:
        # Copy goals from raw plan to plan
        self.plan.goal = self.scenario.raw_plan.goal

        # Evaluate them
        success = self.plan.goal.evaluate_goals(self.scenario)
        if not success:
            self.plan.task_failed = True
            Utils.print_color(c.Color.RED, "Plan goals not met, plan failed")

        self.scenario.complete_manual()
        return self.plan
