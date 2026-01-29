from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import json
import os
from enum import Enum
from pathlib import Path

import Magmathor.constants as c
import Magmathor.utils as Utils


class CacheType(str, Enum):
    # Containers that works with the object
    CONTAINERS = "containers"

    # Containers that didn't with the object
    FAILURES = "failures"

    # Pose for container when interacting with it
    INTERACTION = "interaction"

    # Pose for placing the object or from a the container
    PLACEMENT = "placement"


class PlacementCache:

    cache = {}

    base = Path(__file__).parent.parent
    FILE_PATH = os.path.join(base, "Data", "placement_cache.json")

    session_hits = 0
    session_misses = 0

    def __init__(self, value: any, hits: int = 0):
        self.value = value
        self.hits = hits

    def to_dict(self):
        return {"value": self.value, "hits": self.hits}

    @classmethod
    def from_dict(cls, data):
        return cls(value=data["value"], hits=data["hits"])

    @classmethod
    def _get(cls, key) -> PlacementCache:
        if cls.cache == {}:
            cls._load()
        if key in cls.cache:
            cls.cache[key].hits += 1
            cls.session_hits += 1
            return cls.cache[key]
        else:
            cls.session_misses += 1
            return None

    @classmethod
    def _key_interaction(cls, scene, object_name, agent_action):

        def action_type(action_type):
            # Some actions should be stored separetely as they
            # require different poses, e.g. opening and closing a fridge, vs putting an object in or taking it out
            action_names = ["open", "close", "toggle", "slice", "find"]
            action_string = str(action_type)
            for action_name in action_names:
                if action_name in action_string:
                    return action_name
            return None

        action_name = action_type(agent_action)
        key = f"{CacheType.INTERACTION}_{scene}_{object_name}{f'_{action_name}' if action_name else ''}"
        return key

    @classmethod
    def _key_container(cls, scene, obj_name):
        key = f"{CacheType.CONTAINERS.value}_{scene}_{obj_name}"
        return key

    @classmethod
    def _key_failure(cls, scene, obj_name):
        key = f"{CacheType.FAILURES.value}_{scene}_{obj_name}"
        return key

    @classmethod
    def _key_location(cls, scene, target_location):
        # Round target location to nearest 0.1 to keep cache size down
        rounded_location = {
            "x": round(target_location["x"], 1),
            "y": round(target_location["y"], 1),
            "z": round(target_location["z"], 1),
        }
        key = f"{CacheType.PLACEMENT.value}_{scene}_{rounded_location['x']}_{rounded_location['y']}_{rounded_location['z']}"
        return key

    @classmethod
    def get_container_names(cls, scene, obj_name):
        key = cls._key_container(scene, obj_name)
        value = cls._get(key)
        if value is not None:
            return value.value
        else:
            return []

    @classmethod
    def get_failure_names(cls, scene, obj_name):
        key = cls._key_failure(scene, obj_name)
        value = cls._get(key)
        if value is not None:
            return value.value
        else:
            return []

    @classmethod
    def get_interaction_pose(cls, scene, obj_name, agent_action):
        key = cls._key_interaction(scene, obj_name, agent_action)
        value = cls._get(key)
        if value is not None:
            return value.value
        else:
            return None

    @classmethod
    def get_location_pose(cls, scene, location):
        key = cls._key_location(scene, location)
        value = cls._get(key)
        if value is not None:
            return value.value
        else:
            return None

    @classmethod
    def delete_interaction_pose(cls, scene: str, obj_name: str, agent_action):
        key = cls._key_interaction(scene, obj_name, agent_action)
        if key in cls.cache:
            del cls.cache[key]

    @classmethod
    def add_container(cls, scene, obj_name, container):
        key = cls._key_container(scene, obj_name)

        containers = cls.get_container_names(scene, obj_name)
        if containers is None:
            containers = []

        if container not in containers:
            containers.append(container)
            cls._add(key, containers)

        # Remove from any previous failures if exists
        failure_key = cls._key_failure(scene, obj_name)
        if failure_key in cls.cache:
            failure_containers = cls.get_failure_names(scene, obj_name)
            if container in failure_containers:
                failure_containers.remove(container)
                cls.cache[failure_key].value = failure_containers
                if not failure_containers:
                    del cls.cache[failure_key]

    @classmethod
    def add_failure(cls, scene, obj_name, container):
        key = cls._key_failure(scene, obj_name)

        containers = cls.get_failure_names(scene, obj_name)
        if containers is None:
            containers = []

        if container not in containers:
            containers.append(container)
            cls._add(key, containers)

    cache_types = [
        "Fridge",
        "Cabinet",
        "Drawer",
        "Oven",
        "StoveBurner",
        "StoveKnob",
        "Sink",
        "Dishwasher",
        "Microwave",
        "Countertop",
        "Toaster",
        "CoffeeMachine",
    ]

    @classmethod
    def add_interaction_pose(cls, scene, obj, agent, agent_action):
        if obj["objectType"] in PlacementCache.cache_types:
            pose = {
                "position": agent["position"],
                "rotation": agent["rotation"]["y"],
                "standing": agent["isStanding"],
                "horizon": agent["cameraHorizon"],
            }
            key = cls._key_interaction(scene, obj["name"], agent_action)
            cls._add(key, pose)

    @classmethod
    def add_location_pose(cls, scene, target_loction, agent):
        pose = {
            "position": agent["position"],
            "rotation": agent["rotation"]["y"],
            "standing": agent["isStanding"],
            "horizon": agent["cameraHorizon"],
        }
        key = cls._key_location(scene, target_loction)
        cls._add(key, pose)

    @classmethod
    def _add(cls, key, value):
        if cls.cache == {}:
            cls._load()

        placement = PlacementCache(value)
        cls.cache[key] = placement

    @classmethod
    def save(cls):
        if cls.cache == {}:
            return

        cache_dict = {}
        with open(cls.FILE_PATH, "w") as f:
            # Convert cache to a dictionay
            for key, placement in cls.cache.items():
                cache_dict[key] = placement.to_dict()

            # Write the dictionary to the file
            json.dump(cache_dict, f)

        Utils.print_color(
            c.Color.PURPLE,
            f"Session hits: {cls.session_hits}, session misses: {cls.session_misses}",
        )

    # Save incase need to use again in future
    """@classmethod
    def util_perge_field(cls, field: str):
        # Remove all entries in the cache that match the given field.
        keys_to_remove = [key for key in cls.cache if key.startswith(field)]
        for key in keys_to_remove:
            del cls.cache[key]

        Utils.print_color(c.Color.PURPLE, f"Removed {len(keys_to_remove)} entries with field '{field}' from cache.")
        cls.save()"""

    @classmethod
    def _load(cls):
        # If file doesn't exist use empty cache
        if not os.path.exists(cls.FILE_PATH):
            cls.cache = {}
            return

        try:
            with open(cls.FILE_PATH, "r") as f:
                # Read the JSON data from the file
                cache_dict = json.load(f)

            # Convert the dictionary back to PlacementCache objects
            cls.cache = {}
            for key, placement_data in cache_dict.items():
                placement = PlacementCache.from_dict(placement_data)
                cls.cache[key] = placement

            cls.session_hits = 0
            cls.session_misses = 0

        except Exception as e:
            print(f"Could not load placement cache: {e}")
            cls.cache = {}
