import json
import os
from pathlib import Path

from Magmathor.scenario import Scenario
from Magmathor.scenes import Scenes


class MakeItemCache:
    """
    Script to create a cache of item names and types for each scene.
    This cache is saved to a JSON file for quick access during scene processing."""

    @staticmethod
    def run():

        name_dict = {}
        type_dict = {}
        for scene in Scenes.all:
            print(f"Processing scene: {scene}")
            names, types = Scenario.scene_get_names_and_types(scene)
            name_dict[scene] = names
            type_dict[scene] = types

        data = {
            "names": name_dict,
            "types": type_dict,
        }

        base = Path(__file__).parent.parent
        file_path = os.path.join(base, "Data", "item_cache.json")
        with open(file_path, "w", encoding="utf-8") as s:
            json.dump(data, s)
            s.flush()


if __name__ == "__main__":
    MakeItemCache.run()
