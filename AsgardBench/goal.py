from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

from enum import Enum
from typing import TYPE_CHECKING, List

import AsgardBench.constants as c
import AsgardBench.utils as Utils

# Only import Scenario for type checking, not at runtime
if TYPE_CHECKING:
    from AsgardBench.scenario import Scenario


class GoalOutcome(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ActionGoal(str, Enum):
    DRINK_COFFEE = "Drink Coffee"
    SPRAY_MIRROR = "Spray Mirror"


class PutAwayGoal:
    def __init__(self, object_type: str, outcome: GoalOutcome = None):
        self.object_type = object_type
        self.outcome = outcome

    def to_dict(self):
        return {
            "object_type": self.object_type,
            "outcome": self.outcome.value if self.outcome else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PutAwayGoal:
        return cls(
            object_type=data["object_type"],
            outcome=GoalOutcome(data["outcome"]) if data.get("outcome") else None,
        )


class LocationGoal:
    def __init__(
        self, object_type: str, destination_type: str, outcome: GoalOutcome = None
    ):
        self.object_type = object_type
        self.destination_type = destination_type
        self.outcome = outcome

    def to_dict(self):
        return {
            "object_type": self.object_type,
            "destination_type": self.destination_type,
            "outcome": self.outcome.value if self.outcome else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LocationGoal:
        return cls(
            object_type=data["object_type"],
            destination_type=data["destination_type"],
            outcome=GoalOutcome(data["outcome"]) if data.get("outcome") else None,
        )


class ContentsGoal:
    def __init__(
        self, container_type: str, contents: List[str], outcome: GoalOutcome = None
    ):
        self.container_type = container_type
        self.contents = contents
        self.outcome = outcome

    def to_dict(self):
        return {
            "container_type": self.container_type,
            "contents": self.contents,
            "outcome": self.outcome.value if self.outcome else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContentsGoal:
        return cls(
            container_type=data["container_type"],
            contents=data["contents"],
            outcome=GoalOutcome(data["outcome"]) if data.get("outcome") else None,
        )


class StateGoal:
    def __init__(
        self, object_type: str, state: str, value: bool, outcome: GoalOutcome = None
    ):
        self.object_type = object_type
        self.state = state
        self.value = value
        self.outcome = outcome

    def to_dict(self):
        return {
            "object_type": self.object_type,
            "state": self.state,
            "value": self.value,
            "outcome": self.outcome.value if self.outcome else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> StateGoal:
        return cls(
            object_type=data["object_type"],
            state=data["state"],
            value=data["value"],
            outcome=GoalOutcome(data["outcome"]) if data.get("outcome") else None,
        )


class Goal:
    def __init__(
        self,
        location_goals: List[LocationGoal] = None,
        state_goals: List[StateGoal] = None,
        put_away_goals: List[PutAwayGoal] = None,
        action_goals: dict[ActionGoal, bool] = None,
        contents_goals: List[ContentsGoal] = None,
        room=None,
    ):
        self.location_goals = location_goals if location_goals is not None else []
        self.state_goals = state_goals if state_goals is not None else []
        self.put_away_goals = put_away_goals if put_away_goals is not None else []
        self.action_goals = action_goals if action_goals is not None else {}
        self.contents_goals = contents_goals if contents_goals is not None else []

        if room == "Kitchen":
            self.add_state_goal("CoffeeMachine", "isToggled", False)
            self.add_state_goal("Toaster", "isToggled", False)
            self.add_state_goal("StoveKnob", "isToggled", False)
            self.add_state_goal("Microwave", "isOpen", False)
            self.add_state_goal("Fridge", "isOpen", False)
            self.add_state_goal("Faucet", "isToggled", False)

    def to_dict(self):
        return {
            "location_goals": [lg.to_dict() for lg in self.location_goals],
            "state_goals": [sg.to_dict() for sg in self.state_goals],
            "put_away_goals": [pw.to_dict() for pw in self.put_away_goals],
            "action_goals": {k: v for k, v in self.action_goals.items()},
            "contents_goals": [cg.to_dict() for cg in self.contents_goals],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Goal:
        location_goals = [
            LocationGoal.from_dict(lg) for lg in data.get("location_goals", [])
        ]
        state_goals = [StateGoal.from_dict(sg) for sg in data.get("state_goals", [])]
        put_away_goals = [
            PutAwayGoal.from_dict(pw) for pw in data.get("put_away_goals", [])
        ]
        # Fix: Create ActionGoal from the key (which is the enum value), keep boolean as completion status
        action_goals = {
            ActionGoal(k): v for k, v in data.get("action_goals", {}).items()
        }
        contents_goals = [
            ContentsGoal.from_dict(cg) for cg in data.get("contents_goals", [])
        ]
        return cls(
            location_goals=location_goals,
            state_goals=state_goals,
            put_away_goals=put_away_goals,
            action_goals=action_goals,
            contents_goals=contents_goals,
        )

    def add_location_goal(self, object_type: str, destination_type: str):
        """
        Add a location goal to the goal.
        """
        self.location_goals.append(LocationGoal(object_type, destination_type))

    def add_state_goal(self, object_type: str, state: str, value: bool):
        """
        Add a state goal to the goal, but only if it doesn't already exist.
        """
        # Check if a state goal with the same object_type and state already exists
        for existing_goal in self.state_goals:
            if (
                existing_goal.object_type == object_type
                and existing_goal.state == state
            ):
                # Goal already exists, don't add it again
                return

        # Goal doesn't exist, add it
        self.state_goals.append(StateGoal(object_type, state, value))

    def add_put_away_goal(self, object_type: str):
        """
        Add a put away goal to the goal.
        """
        self.put_away_goals.append(PutAwayGoal(object_type))

    def add_action_goal(self, action: ActionGoal):
        """
        Add an action goal to the goal.
        """
        if action not in self.action_goals:
            self.action_goals[action] = False

    def set_action_goal(self, action: ActionGoal, value: bool):
        """
        Set the value of an action goal.
        """
        if action in self.action_goals:
            self.action_goals[action] = value
        else:
            Utils.print_color(
                c.Color.DARK_RED, f"Action goal {action} not found in the goal."
            )

    def add_contents_goal(self, container_type: str, contents: List[str]):
        """
        Add a contents goal to the goal.
        """
        self.contents_goals.append(
            ContentsGoal(container_type=container_type, contents=contents)
        )

    def employed_for_goal(self, obj, container) -> bool:
        """
        Return true if the object satisfies any of the goals
        """
        if not container:
            return False

        for location_goal in self.location_goals:
            if location_goal.object_type == obj["objectType"]:
                # Check if the object is in the correct destination type
                if container["objectType"] == location_goal.destination_type:
                    return True
        return False

    def percent_goals_reached(self) -> float:
        """
        Return the percent of goals that have been reached.
        """
        total_goals = (
            len(self.location_goals)
            + len(self.state_goals)
            + len(self.put_away_goals)
            + len(self.action_goals)
            + len(self.contents_goals)
        )
        if total_goals == 0:
            return 1.0

        reached_goals = 0
        for location_goal in self.location_goals:
            if location_goal.outcome == GoalOutcome.SUCCESS:
                reached_goals += 1
        for state_goal in self.state_goals:
            if state_goal.outcome == GoalOutcome.SUCCESS:
                reached_goals += 1
        for put_away_goal in self.put_away_goals:
            if put_away_goal.outcome == GoalOutcome.SUCCESS:
                reached_goals += 1
        for action_goal in self.action_goals.values():
            if action_goal:
                reached_goals += 1
        for contents_goal in self.contents_goals:
            if contents_goal.outcome == GoalOutcome.SUCCESS:
                reached_goals += 1

        return reached_goals / total_goals

    def all_goals_reached(self, scenario: Scenario) -> bool:
        """
        Check if any goal has failed.
        """
        self.evaluate_goals(scenario)
        for location_goal in self.location_goals:
            if location_goal.outcome == GoalOutcome.FAILURE:
                return False
        for state_goal in self.state_goals:
            if state_goal.outcome == GoalOutcome.FAILURE or state_goal.outcome is None:
                return False
        for put_away_goal in self.put_away_goals:
            if (
                put_away_goal.outcome == GoalOutcome.FAILURE
                or put_away_goal.outcome is None
            ):
                return False
        for action_goal in self.action_goals:
            if action_goal is False:
                return False
        for contents_goal in self.contents_goals:
            if (
                contents_goal.outcome == GoalOutcome.FAILURE
                or contents_goal.outcome is None
            ):
                return False
        return True

    def reset_goals(self):
        """
        Reset all goals to their initial state.
        """
        for location_goal in self.location_goals:
            location_goal.outcome = None
        for state_goal in self.state_goals:
            state_goal.outcome = None
        for put_away_goal in self.put_away_goals:
            put_away_goal.outcome = None
        for key in self.action_goals:
            self.action_goals[key] = False
        for key in self.contents_goals:
            key.outcome = None

    def evaluate_goals(self, scenario: Scenario):
        print("Evaluating goals...")
        failed_location = self.evaluate_location_goals(scenario)
        failed_state = self.evaluate_state_goals(scenario)
        failsed_put = self.evaluate_put_away_goals(scenario)
        failed_action = self.evaluate_action_goals()
        failed_contents = self.evaluate_contents_goals(scenario)
        if (
            failed_location is not None
            or failed_state is not None
            or failsed_put is not None
            or failed_action is not None
            or failed_contents is not None
        ):
            return False
        return True

    def evaluate_location_goals(self, scenario: Scenario) -> None:
        """
        Return a list of location goals that are not satisfied by the object and container.
        """
        failed_goals: List[LocationGoal] = []
        for location_goal in self.location_goals:
            objs = scenario.get_objs_by_types(
                [location_goal.object_type], must_exist=False
            )
            if objs is None:
                location_goal.outcome = GoalOutcome.FAILURE
                failed_goals.append(location_goal)
                Utils.print_color(
                    c.Color.DARK_RED,
                    f"  Failed location goals: No object of type {location_goal.object_type}",
                )
                continue

            # Only one needs to satisfy the condition
            for obj in objs:
                container = scenario.get_non_surface_container(
                    obj["name"], must_exist=False
                )
                if (
                    container
                    and container["objectType"] == location_goal.destination_type
                ):
                    location_goal.outcome = GoalOutcome.SUCCESS
                    break

                # AI2Thor can fail to correctly calculate containment, so check manually
                container_objs = scenario.get_objs_by_types(
                    [location_goal.destination_type]
                )
                for container_obj in container_objs:
                    inside = Utils.is_object_inside(obj, container_obj)
                    if inside:
                        location_goal.outcome = GoalOutcome.SUCCESS
                        break

            if location_goal.outcome is None:
                location_goal.outcome = GoalOutcome.FAILURE

            if location_goal.outcome == GoalOutcome.FAILURE:
                failed_goals.append(location_goal)
                Utils.print_color(
                    c.Color.DARK_RED,
                    f"  Failed location goals: {location_goal.object_type} not in {location_goal.destination_type}",
                )
                failed_goals.append(location_goal)

            else:
                Utils.print_color(
                    c.Color.DARK_GREEN,
                    f"  Pass location goals: {location_goal.object_type} in {location_goal.destination_type}",
                )

        if len(failed_goals) > 0:
            return failed_goals
        return None

    def evaluate_state_goals(self, scenario: Scenario) -> None:
        """
        Return a list of state goals that are not satisfied by the object.
        """
        failed_goals: List[StateGoal] = []
        for state_goal in self.state_goals:
            # Default to None
            state_goal.outcome = None

            obj = scenario.get_obj_by_type(state_goal.object_type, must_exist=False)
            if obj is None:
                state_goal.outcome = GoalOutcome.FAILURE
                Utils.print_color(
                    c.Color.DARK_RED,
                    f"  Failed state goals: {state_goal.object_type} {state_goal.state} does not exist.",
                )

            # If object is egg that must be cooked, I need to check if the cracked egg exists
            elif obj["objectType"] == "Egg" and state_goal.state == "isCooked":
                cracked_egg = scenario.get_obj_by_type("EggCracked", must_exist=False)
                if cracked_egg is None:
                    state_goal.outcome = GoalOutcome.FAILURE
                elif cracked_egg["isCooked"] != state_goal.value:
                    state_goal.outcome = GoalOutcome.FAILURE
                else:
                    state_goal.outcome = GoalOutcome.SUCCESS

            # If object is bread that must be cooked, I need to check if one one slices is cooked
            elif obj["objectType"] == "Bread" and state_goal.state == "isCooked":
                state_goal.outcome = GoalOutcome.FAILURE
                breads = scenario.get_objs_by_types(["BreadSliced"], must_exist=False)
                if breads is not None:
                    for bread in breads:
                        if bread["isCooked"] == state_goal.value:
                            state_goal.outcome = GoalOutcome.SUCCESS
                            break

            # If object is potato that must be cooked, I need to check if potato one slices is cooked
            elif obj["objectType"] == "PotatoSliced" and state_goal.state == "isCooked":
                state_goal.outcome = GoalOutcome.FAILURE
                potatoes = scenario.get_objs_by_types(
                    ["PotatoSliced"], must_exist=False
                )
                if potatoes is not None:
                    for potato in potatoes:
                        if potato["isCooked"] == state_goal.value:
                            state_goal.outcome = GoalOutcome.SUCCESS
                            break

            # Check if the object's state matches the goal
            elif obj[state_goal.state] != state_goal.value:
                state_goal.outcome = GoalOutcome.FAILURE
                Utils.print_color(
                    c.Color.DARK_RED,
                    f"  Failed state goals: {obj['name']} {state_goal.state} != {state_goal.value}",
                )
            elif obj[state_goal.state] == state_goal.value:
                state_goal.outcome = GoalOutcome.SUCCESS

            assert state_goal.outcome is not None

            if state_goal.outcome == GoalOutcome.SUCCESS:
                Utils.print_color(
                    c.Color.DARK_GREEN,
                    f"  Pass state goals: {obj['name']} {state_goal.state} == {state_goal.value}",
                )
            else:
                failed_goals.append(state_goal)

        if len(failed_goals) > 0:
            return failed_goals
        return None

    def evaluate_put_away_goals(self, scenario: Scenario) -> None:
        """
        Return a list of put away goals that are not satisfied by the object.
        """
        failed_goals: List[PutAwayGoal] = []
        for put_away_goal in self.put_away_goals:
            # Default to failure, set to true if the goal is satisfied
            put_away_goal.outcome = GoalOutcome.FAILURE

            obj = scenario.get_obj_by_type(put_away_goal.object_type)

            # Check if the object is not put away
            container = scenario.get_non_surface_container(
                obj["name"], must_exist=False
            )
            surfaces = scenario.get_surface_containers(obj["name"])

            container_types = []
            if container is not None:
                container_types.append(container["objectType"])

            for surface in surfaces:
                container_types.append(surface["objectType"])

            matched_type = None

            for container_type in container_types:
                if container_type in c.PREFERED_OBJECT_HOMES[obj["objectType"]]:
                    put_away_goal.outcome = GoalOutcome.SUCCESS
                    matched_type = container_type
                    break
                elif (
                    obj["objectType"] in c.SECONDARY_OBJECT_HOMES
                    and container_type in c.SECONDARY_OBJECT_HOMES[obj["objectType"]]
                ):
                    put_away_goal.outcome = GoalOutcome.SUCCESS
                    matched_type = container_type
                    break

            if put_away_goal.outcome == GoalOutcome.FAILURE:
                if container is None:
                    Utils.print_color(
                        c.Color.DARK_RED, f"  {obj['name']} is not in a container"
                    )
                else:
                    Utils.print_color(
                        c.Color.DARK_RED,
                        f"  {obj['name']} in {container['objectType']} (Wrong)",
                    )

                failed_goals.append(put_away_goal)

            else:
                # Set container from surface for printing debug
                if (container is None) and (len(surfaces) > 0):
                    container = surfaces[0]

                Utils.print_color(
                    c.Color.DARK_GREEN,
                    f"  {obj['name']} in {container['objectType']}",
                )

        if len(failed_goals) > 0:
            return failed_goals

        return None

    def evaluate_action_goals(self) -> None:
        """
        Evaluate action goals and set their outcomes.
        """
        failed_actions: List[ActionGoal] = []
        for action, outcome in self.action_goals.items():
            if not outcome:
                failed_actions.append(action)
                Utils.print_color(c.Color.DARK_RED, f"  Failed action goal: {action}")
            else:
                Utils.print_color(c.Color.DARK_GREEN, f"  Pass action goal: {action}")

        if len(failed_actions) > 0:
            return failed_actions

        return None

    def evaluate_contents_goals(self, scenario: Scenario) -> None:
        failed_goals = []
        for contents_goal in self.contents_goals:
            contents_goal.outcome = GoalOutcome.SUCCESS
            container_obj = scenario.get_obj_by_type(contents_goal.container_type)
            objs = scenario.get_container_contents(container_obj["name"])
            not_found = contents_goal.contents.copy()

            if objs is not None:
                for obj in objs:
                    if obj["objectType"] not in contents_goal.contents:
                        if scenario.is_pickupable(obj["name"]):
                            Utils.print_color(
                                c.Color.DARK_RED,
                                f"  Failed contents goal: {obj['objectType']} shouldn't be in {contents_goal.container_type}",
                            )
                            contents_goal.outcome = GoalOutcome.FAILURE
                    elif obj["objectType"] in not_found:
                        not_found.remove(obj["objectType"])

            if len(not_found) > 0:
                Utils.print_color(
                    c.Color.DARK_RED,
                    f"  Failed contents goal: {not_found} NOT in {contents_goal.container_type}",
                )
                contents_goal.outcome = GoalOutcome.FAILURE

            if contents_goal.outcome == GoalOutcome.FAILURE:
                failed_goals.append(contents_goal)
            else:
                Utils.print_color(
                    c.Color.DARK_GREEN,
                    f"  Pass contents goal: Only {contents_goal.contents} in {contents_goal.container_type}",
                )

        if len(failed_goals) > 0:
            return failed_goals

        return None

    def did_action_goal(self, action_goal: ActionGoal) -> bool:
        """
        Check if the action goal was achieved.
        """
        return self.action_goals.get(action_goal, False)
