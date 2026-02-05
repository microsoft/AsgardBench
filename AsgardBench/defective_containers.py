import json
import os

from AsgardBench import constants as c
from AsgardBench import utils as Utils

# Once a container has failed more than this many times, it is considered defective
# If fail only once or twise, could be due to randoming positioning in the scene
FAIL_LIMIT = 5

# Indicates that the container worked successfully at least once, so not defective
SUCCESS = "success"


class DefectiveContainers:
    """
    Some containers in particular scenes are defective, meaning they cannot be used with objects.
    This class keeps track of those containers and their failure counts so runner
    knows which scenes to skip for particular scenarios
    """

    _defective_containers = None
    FILE_PATH = os.path.join(
        os.path.dirname(__file__), "Data", "defective_containers.json"
    )

    @classmethod
    def defective_containers(cls) -> dict:
        """
        Load the defective containers from the JSON file.
        Returns an empty dictionary if the file does not exist.
        """
        if cls._defective_containers is None:
            if not os.path.exists(cls.FILE_PATH):
                return {}

            with open(cls.FILE_PATH, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                # Convert lists back to sets
                cls._defective_containers = {k: set(v) for k, v in loaded_data.items()}

        return cls._defective_containers

    @classmethod
    def key(cls, scene: str, container_name: str, object_name: str) -> str:
        """
        Generate a unique key for a container based on scene, container name, and object name.
        """

        # Use full name for containers as they differ (i.e. multiple drawers)
        s_container_name = container_name

        # Use short name for objects (like sliced tomoto) as they are the same
        s_object_name = Utils.short_name(object_name)
        return f"{scene}_{s_container_name}_{s_object_name}"

    @classmethod
    def save(
        cls, scene: str, container_name: str, object_name: str, success: bool, hash: str
    ):

        # Ingnore surface that have mutiple places to put things
        container_short = Utils.short_name(container_name)
        if container_short in [
            "CounterTop",
            "DiningTable",
            "Shelf",
            "ShelvingUnit",
            "SideTable",
            "Floor",
        ]:
            return

        def_containers = cls.defective_containers()

        key = cls.key(scene, container_name, object_name)
        if key not in def_containers:
            def_containers[key] = set()

        # If the action was successful, note that
        if success:
            def_containers[key] = {SUCCESS}

        # Count failure unless I ever succeeded
        elif (
            SUCCESS not in def_containers[key] and len(def_containers[key]) < FAIL_LIMIT
        ):
            def_containers[key].add(hash)

        # Sort alphabetically by key, and within each value set, put SUCCESS last
        def_containers = dict(sorted(def_containers.items()))
        def_containers = {
            k: sorted(list(v), key=lambda x: (x != SUCCESS, x))
            for k, v in def_containers.items()
        }

        with open(cls.FILE_PATH, "w", encoding="utf-8") as s:
            json.dump(def_containers, s)
            s.flush()

    @classmethod
    def is_defective(cls, scene: str, container_name: str, object_name: str) -> bool:
        """
        Check if a container has defective in the given scene.
        Returns True if it has defective, False otherwise.
        """
        key = cls.key(scene, container_name, object_name)
        # Check if the key exists and if it has defective more than 5 times
        if (
            key in cls.defective_containers()
            and len(cls.defective_containers()[key]) >= FAIL_LIMIT
        ):
            return True
        return False
