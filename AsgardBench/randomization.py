import random
from enum import Enum
from typing import List, Optional

from AsgardBench.constants import Action, Color
from AsgardBench.specifier import Specifier
from AsgardBench.utils import print_color


class InjectionType(Enum):
    PICKUP_RANDOM = "Pickup Random"
    PICKUP_NOT_VISIBLE = "Pickup Not Visible"
    OPEN_RANDOM = "Open Random"


INJECTIONTYPE_TO_ACTION = {
    InjectionType.PICKUP_RANDOM: Action.PICKUP,
    InjectionType.PICKUP_NOT_VISIBLE: Action.PICKUP,
    InjectionType.OPEN_RANDOM: Action.OPEN,
}


class ObjectRandomization:
    def __init__(self, obj_specifier: Specifier, cont_specifier: Specifier):
        self.obj_specifier = obj_specifier
        self.cont_specifier = cont_specifier

    def to_dict(self):
        return {
            "obj_specifier": self.obj_specifier.to_dict(),
            "cont_specifier": self.cont_specifier.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectRandomization":
        obj_specifier = Specifier.from_dict(data["obj_specifier"])
        cont_specifier = Specifier.from_dict(data["cont_specifier"])
        return cls(obj_specifier, cont_specifier)


class ErrorInjection:
    def __init__(
        self,
        injection_type: InjectionType,
        action_index: int,
        object_name: str = None,
        avoid_types: List[str] = None,
    ):

        # What type of injection is this
        self.injection_type = injection_type

        # Which attempt to the take action should this occur
        self.action_index = action_index

        # What object does it relate to
        self.object_name: Optional[str] = object_name

        # What object should be avoided
        self.avoid_types = avoid_types if avoid_types is not None else []

    def to_dict(self):
        return {
            "injection_type": self.injection_type.value,
            "action_index": self.action_index,
            "object_name": self.object_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ErrorInjection":
        return cls(
            injection_type=InjectionType(data["injection_type"]),
            action_index=data["action_index"],
            object_name=data.get("object_name", None),
        )


class Randomization:
    def __init__(self, seed: int):
        self.seed = seed
        self.error_injection: Optional[ErrorInjection] = None
        self.fail_action: Optional[str] = None

    def to_dict(self):
        return {
            "seed": self.seed,
            "error_injection": (
                self.error_injection.to_dict() if self.error_injection else None
            ),
            "fail_action": self.fail_action,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Randomization":
        randomization = cls(data["seed"])
        randomization.error_injection = (
            ErrorInjection.from_dict(data["error_injection"])
            if data.get("error_injection")
            else None
        )
        randomization.fail_action = data.get("fail_action", None)
        return randomization

    def add_error_injection(
        self, action_counts, avoid_types: List[str] = []
    ) -> Optional[InjectionType]:
        """
        Generate a random error injection based on the action counts.
        action_counts of action counts in non-error version
        """

        # Pick a random injection type
        injection_types = list(InjectionType)
        random.shuffle(injection_types)

        # Now loop to find an existing action that can error
        while len(injection_types) > 0:
            injection_type = injection_types.pop(0)
            action_type = INJECTIONTYPE_TO_ACTION[injection_type].value
            if action_type in action_counts:
                injection_step = random.randint(0, action_counts[action_type] - 1)
                self.error_injection = ErrorInjection(
                    injection_type, injection_step, avoid_types=avoid_types
                )

                print_color(
                    Color.YELLOW,
                    f"Will inject {injection_type.name} error on step {self.error_injection.action_index} of {injection_type.value}",
                )
                return injection_type

        print_color(Color.RED, "No actions available for error injection")
        return None
