import json
import os
from pathlib import Path


class ItemCache:
    """
    Cache for scene item names and types.
    This cache is loaded from a JSON file and provides methods to access item names and types for specific scenes.
    """

    cache = {}
    base = Path(__file__).parent.parent
    FILE_PATH = os.path.join(base, "Data", "item_cache.json")

    @classmethod
    def get_scene_names(cls, scene):
        if cls.cache == {}:
            cls._load()
        return cls.cache["names"][scene]

    @classmethod
    def get_scene_types(cls, scene):
        if cls.cache == {}:
            cls._load()
        return cls.cache["types"][scene]

    @classmethod
    def get_names_by_type(cls, scene, type):
        names = cls.get_scene_names(scene)
        return [name for name in names if name.startswith(f"{type}_")]

    @classmethod
    def _load(cls):
        # If file doesn't exist use empty cache
        if not os.path.exists(cls.FILE_PATH):
            raise FileNotFoundError(
                f"Cache file {cls.FILE_PATH} does not exist. Please run the save_scene_lists script to create it."
            )

        try:
            with open(cls.FILE_PATH, "r") as f:
                cls.cache = json.load(f)

        except FileNotFoundError as e:
            print(f"Could not load item cache: {e}")
            cls.cache = {}
