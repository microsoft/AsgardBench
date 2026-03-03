from __future__ import annotations

from typing import Optional

from AsgardBench.specifier import Specifier


class DistractorSetup:
    def __init__(
        self,
        object_specifier: Specifier,
        container_specifier: Optional[Specifier] = None,
    ):
        """
        DistractorSetup defines the containers that an object can be placed in.
        :param object_specifier: Specifier for the types of objects that can be placed in the containers.
        :param container_specifier: Specifier for the types of containers that the objects can be placed in (if None, use c.DEFAULT_STARTING_PLACES).
        """
        self.object_specifier = object_specifier
        self.container_specifier = (
            container_specifier if container_specifier is not None else None
        )

    def to_dict(self):
        return {
            "object_specifier": self.object_specifier.to_dict(),
            "container_specifier": (
                self.container_specifier.to_dict() if self.container_specifier else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DistractorSetup:
        object_specifier = Specifier.from_dict(data["object_specifier"])
        container_data = data.get("container_specifier", None)
        if container_data is None:
            container_specifier = None
        else:
            container_specifier = Specifier.from_dict(container_data)

        return cls(
            object_specifier=object_specifier,
            container_specifier=container_specifier,
        )


class TargetSetup:
    def __init__(
        self,
        object_specifier: Specifier,
        container_specifier: Optional[Specifier] = None,
    ):
        """
        TargetSetup defines how target items should be place
        :param object_specifier: Specifier for target objects
        :param container_specifier: Specifier for the types of containers that the objects can be placed in (if None, must be visible).
        """
        self.object_specifier = object_specifier
        self.container_specifier = (
            container_specifier if container_specifier is not None else None
        )

    def to_dict(self):
        return {
            "object_specifier": self.object_specifier.to_dict(),
            "container_specifier": (
                self.container_specifier.to_dict() if self.container_specifier else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TargetSetup:
        object_specifier = Specifier.from_dict(data["object_specifier"])
        container_data = data.get("container_specifier", None)
        if container_data is None:
            container_specifier = None
        else:
            container_specifier = Specifier.from_dict(container_data)

        return cls(
            object_specifier=object_specifier, container_specifier=container_specifier
        )


class ObjectSetup:
    def __init__(
        self,
        target_setup: TargetSetup = None,
        distractor_setups: list[DistractorSetup] = None,
        employed_types: list[str] = None,
    ):
        """
        ObjectSetup defines the initial state of objects in the scenario.
        :param target_setup: Object setup for target items.
        :param distractor_setups: Object setups for distractor items.
        :param employed_types: List of types used in the scenario, so I don't put in fridge
        """
        self.target_setup = target_setup if target_setup is not None else None
        self.distractor_setups = (
            distractor_setups if distractor_setups is not None else []
        )
        self.employed_types = employed_types if employed_types is not None else []

    def to_dict(self):
        return {
            "target_setup": self.target_setup.to_dict() if self.target_setup else None,
            "distractor_setups": [obj.to_dict() for obj in self.distractor_setups],
            "employed_types": self.employed_types,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ObjectSetup:
        return cls(
            target_setup=(
                TargetSetup.from_dict(data["target_setup"])
                if data.get("target_setup")
                else None
            ),
            distractor_setups=[
                DistractorSetup.from_dict(obj)
                for obj in data.get("distractor_setups", [])
            ],
            employed_types=data.get("employed_types", []),
        )

    def add_distractors(
        self, object_specifier: Specifier, container_specifier: Specifier = None
    ):
        """
        Add distractor objects to the setup.
        :param object_specifier: Specifier for the types of objects that can be placed in the containers.
        :param container_specifier: Specifier for the types of containers that the objects can be placed in (if None, use c.DEFAULT_STARTING_PLACES).
        """
        self.distractor_setups.append(
            DistractorSetup(object_specifier, container_specifier)
        )

    def add_target(
        self, object_specifier: Specifier, container_specifier: Specifier = None
    ):
        """
        Add a target object to the setup.
        :param object_specifier: Specifier for the target object.
        :param container_specifier: Specifier for the types of containers that the target object can be placed in (if None, must be visible).
        """
        self.target_setup = TargetSetup(object_specifier, container_specifier)

    def get_target_types(self, all_objects) -> list[str]:

        if self.target_setup is None:
            return []
        types = self.target_setup.object_specifier.get_specified_types(all_objects)
        return types


class SetupAction:
    def __init__(self, action: str, object_name: str, argument=None):
        self.action = action
        self.object_name = object_name
        self.argument = argument

    def to_dict(self):
        return {
            "action": self.action,
            "object_name": self.object_name,
            "argument": self.argument,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SetupAction:
        return cls(
            action=data["action"],
            object_name=data["object_name"],
            argument=data.get("argument", None),
        )
