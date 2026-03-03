from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)


class PutCache:
    """
    This is an short in-memory cache for placement (put) for a spectific scene randomization setting.
    Is speeds this up when doing mulitpe itterations of the same counter top configurations
    as the code doesn't have to search for space on the counter
    """

    cache = {}
    configuration_hash = ""

    def __init__(self, value: any, hits: int = 0):
        self.value = value
        self.hits = hits

    @classmethod
    def _get(cls, key) -> PutCache:
        if key in cls.cache:
            return cls.cache[key]
        else:
            return None

    @classmethod
    def _key(cls, holding_name, destination_name):

        key = f"{holding_name}_{destination_name}"
        return key

    @classmethod
    def get_put_pose(cls, hash, holding_name: str, destination_name: str):
        # Has hash changed?
        if cls.configuration_hash != hash:
            cls.configuration_hash = hash
            cls.cache = {}
            return None

        key = cls._key(holding_name, destination_name)
        value = cls._get(key)
        if value is not None:
            return value.value
        else:
            return None

    @classmethod
    def get_put_pose_by_type(cls, hash, holding_name: str, destination_type: str):
        """Look for items of same type that can hold the object"""
        # Has hash changed?
        if cls.configuration_hash != hash:
            cls.configuration_hash = hash
            cls.cache = {}
            return None

        for key in cls.cache.keys():
            if holding_name in key and destination_type in key:
                value = cls._get(key)
                if value is not None:
                    return value.value

            return None

    @classmethod
    def delete_put_pose(cls, hash: str, holding_name: str, destination_name: str):
        # Has hash changed?
        if cls.configuration_hash != hash:
            cls.configuration_hash = hash
            cls.cache = {}
            return

        key = cls._key(holding_name, destination_name)
        if key in cls.cache:
            del cls.cache[key]

    @classmethod
    def add_put_pose(cls, hash: str, holding_name: str, destination_name: str, agent):
        # Has hash changed?
        if cls.configuration_hash != hash:
            cls.configuration_hash = hash
            cls.cache = {}

        pose = {
            "position": agent["position"],
            "rotation": agent["rotation"]["y"],
            "standing": agent["isStanding"],
            "horizon": agent["cameraHorizon"],
        }
        key = cls._key(holding_name, destination_name)
        cls._add(key, pose)

    @classmethod
    def _add(cls, key, value):
        put_cache = PutCache(value)
        cls.cache[key] = put_cache
