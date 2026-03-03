from typing import List

import AsgardBench.constants as c
import AsgardBench.utils as Utils


class Specifier:
    observed_names = set()
    observed_types = set()
    observed_classes = set()
    get_obj_by_name: callable

    # Hacky way to exclude containers from picked for objects
    # Putting in specifier would require a lot of changes
    excluded_containers: List[str] = []
    excluded_objects: List[str] = []

    """
    Allow specification of objects are mulitple granularities
    """

    def __init__(
        self,
        names: List[str] = None,
        types: List[str] = None,
        classes: List[str] = None,
        secondary_names: List[str] = None,
        secondary_types: List[str] = None,
        preferred_name: str = None,
        all=False,
        observe: bool = True,
    ):
        # Assert to make sure that at least one of names, types, classes is provided
        # and they they are lists
        assert (
            names is not None or types is not None or classes is not None
        ), "At least one of names, types, classes must be provided"
        assert names is None or isinstance(
            names, list
        ), "Names must be a list of strings"
        assert types is None or isinstance(
            types, list
        ), "Types must be a list of strings"
        assert classes is None or isinstance(
            classes, list
        ), "Classes must be a list of strings"

        self.names = names if names is not None else None
        self.types = types if types is not None else None
        self.classes = classes if classes is not None else None
        self.secondary_names = secondary_names if secondary_names is not None else None
        self.secondary_types = secondary_types if secondary_types is not None else None

        # If set will use this instance before others
        self.preferred_name = preferred_name if preferred_name is not None else None

        # If true action will be taken on all objects, rather than the first one
        self.all = all

        # If I'm tracking for observation, then update the global sets
        if observe:
            Specifier.update_observed(self)

    @classmethod
    def update_observed(cls, specifier):
        """
        Update the observed sets with the names, types, and classes from the specifier
        """
        if specifier.names:
            remove_names = set()
            for name in specifier.names:
                if (name is None) or (name == ""):
                    Utils.print_color(
                        c.Color.RED, "Specifier name is None or empty, skipping"
                    )
                    continue
                obj = cls.get_obj_by_name(name)
                obs_type = obj["objectType"]

                # For certain containers, I only care about one at time, namely the
                # one I'm trying to use, for other's like slices, more than one is needed
                only_one_types = [
                    "CounterTop",
                    "DiningTable",
                    "Shelf",
                    "ShelvingUnit",
                    "SideTable",
                ]
                if obs_type in only_one_types:
                    # If type already observed, replace remove it
                    for ext_name in cls.observed_names:
                        if ext_name != name:
                            ext_obj = cls.get_obj_by_name(ext_name)
                            ext_type = ext_obj["objectType"]
                            if ext_type == obs_type:
                                remove_names.add(ext_name)

                cls.observed_names.add(name)

            # Now remove any names that are redundant
            cls.observed_names.difference_update(remove_names)

        if specifier.types:
            cls.observed_types.update(specifier.types if specifier else [])
        if specifier.classes:
            cls.observed_classes.update(specifier.classes if specifier else [])

        # Now reduce redundancy

    def to_dict(self):
        return {
            "names": self.names,
            "types": self.types,
            "classes": self.classes,
            "secondary_names": self.secondary_names,
            "secondary_types": self.secondary_types,
            "preferred_name": self.preferred_name,
            "all": self.all,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Specifier":
        return cls(
            names=data.get("names"),
            types=data.get("types"),
            classes=data.get("classes"),
            secondary_names=data.get("secondary_names"),
            secondary_types=data.get("secondary_types"),
            preferred_name=data.get("preferred_name"),
            all=data.get("all", False),
        )

    @classmethod
    def clear_observations(self):
        self.observed_names = set()
        self.observed_types = set()
        self.observed_classes = set()

    #####################################
    def unspecified_objects(self, all_objects):
        """
        Get the objects in all_objects that are not in the specification
        """
        included_objects = self.get_specified_objects(all_objects)
        excluded_objects = [obj for obj in all_objects if obj not in included_objects]
        return excluded_objects

    def unspecified_types(self, all_objects):
        """
        Get all the types in all objects that are not in the specification
        """
        included_types = self.get_specified_types(all_objects)
        exclude_types = set()
        all_types = set()
        for obj in all_objects:
            if obj["objectType"] not in included_types:
                exclude_types.add(obj["objectType"])
            all_types.add(obj["objectType"])

        return list(exclude_types)

    def convert_to_names(self, all_objects):
        """
        Convert specifer to use just names
        """
        objs = self.get_specified_objects(all_objects)

        names = []
        for obj in objs:
            names.append(obj["name"])

        self.names = names
        self.types = None
        self.classes = None

    def get_specified_objects(self, all_objects):
        """
        Get the objects that are specified by the names, types, classes
        """
        objects = self.get_objs_from_specified_names(all_objects)
        objects_from_types = self.get_objs_from_specified_types(all_objects)
        objects_from_classes = self.get_objs_from_specified_classes(all_objects)

        objects.extend(objects_from_types)
        objects.extend(objects_from_classes)

        if len(objects) == 0:
            raise ValueError(
                f"No objects found for specifier: {self.to_string()}. "
                "Please check the names, types, or classes specified."
            )
        return objects

    def get_specified_types(self, all_objects):
        """
        Get the types in all objects that are specified
        """
        types = set(self.types) if self.types is not None else set()
        types_from_classes = self.get_types_from_specified_classes(all_objects)
        types_from_objects = self.get_types_from_specified_names(all_objects)
        return list(types | types_from_classes | types_from_objects)

    def get_types_from_specified_classes(self, all_objects):
        """
        Get the types tht exist in all_objects in the specified classes
        """
        specified_types = set()

        if self.classes is None:
            return specified_types

        for obj_class in self.classes:
            if obj_class in c.CLASS_TO_TYPES:
                specified_types.update(c.CLASS_TO_TYPES[obj_class])

        # Now filter by those that exist
        existing_types = set()
        new_objects = self.get_objs_by_types(specified_types, all_objects)
        if new_objects is not None:
            for obj in new_objects:
                existing_types.add(obj["objectType"])

        return existing_types

    @classmethod
    def get_observed_types(cls, all_objects):
        """
        Get the types in all objects that have been observed
        """
        types = set(cls.observed_types) if cls.observed_types is not None else set()
        types_from_classes = cls.get_types_from_observed_classes(all_objects)
        types_from_objects = cls.get_types_from_observed_names(all_objects)
        return list(types | types_from_classes | types_from_objects)

    @classmethod
    def get_types_from_observed_classes(cls, all_objects):
        """
        Get the types tht exist in all_objects that have been observed
        """
        specified_types = set()

        if cls.observed_classes is None:
            return specified_types

        for obj_class in cls.observed_classes:
            if obj_class in c.CLASS_TO_TYPES:
                specified_types.update(c.CLASS_TO_TYPES[obj_class])

        # Now filter by those that exist
        existing_types = set()
        new_objects = cls.get_objs_by_types(specified_types, all_objects)
        if new_objects is not None:
            for obj in new_objects:
                existing_types.add(obj["objectType"])

        return existing_types

    @classmethod
    def get_types_from_observed_names(cls, all_objects):
        """
        Get the types that exist in all_objects that have been observed
        """
        types = set()

        if cls.observed_names is None:
            return types

        for obj in all_objects:
            if obj["name"] in cls.observed_names:
                types.add(obj["objectType"])
        return types

    def get_types_from_specified_names(self, all_objects):
        """
        Get the types that exist in all_objects in the specified names
        """
        types = set()

        if self.names is None:
            return types

        for obj in all_objects:
            if obj["name"] in self.names:
                types.add(obj["objectType"])
        return types

    @classmethod
    def get_objs_by_types(cls, types, all_objects):
        """
        Get the objects that exist in all_objects that are of the specified types
        """
        objects = []
        for obj in all_objects:
            if obj["objectType"] in types:
                objects.append(obj)
        return objects

    def get_objs_from_specified_classes(self, all_objects):
        """
        Get the objects that exist in all_objects that are of the specified classes
        """
        objects = []

        if self.classes is None:
            return objects

        for obj_class in self.classes:
            if obj_class in c.CLASS_TO_TYPES:
                types = c.CLASS_TO_TYPES[obj_class]
                new_objects = self.get_objs_by_types(types, all_objects)
                objects.extend(new_objects)
        return objects

    def get_objs_from_specified_types(self, all_objects):
        """
        Get the objects that exist in all_objects that are of the specified types
        """
        objects = []

        if self.types is None:
            return objects

        for obj in all_objects:
            if obj["objectType"] in self.types:
                objects.append(obj)
        return objects

    def get_objs_from_specified_names(self, all_objects):
        """
        Get the objects that exist in all_objects that are of the specified names
        """
        objects = []

        if self.names is None:
            return objects

        for obj in all_objects:
            if obj["name"] in self.names:
                objects.append(obj)
        return objects

    def to_string(self):
        """
        Return a string representation of the specifier
        """

        try:
            parts = []
            if self.names:
                parts.append(f"Names: {', '.join(self.names)}")
            if self.types:
                parts.append(f"Types: {', '.join(self.types)}")
            if self.classes:
                parts.append(f"Classes: {', '.join(self.classes)}")
            if self.secondary_names:
                parts.append(f"Secondary Names: {', '.join(self.secondary_names)}")
            if self.secondary_types:
                parts.append(f"Secondary Types: {', '.join(self.secondary_types)}")
            return " | ".join(parts) if parts else "No specification"
        except:
            return "Invalid Specification"

    def to_simple_name(self) -> str:
        """
        Return a simple name for error logging (without prefixes like 'Types: ').
        Returns the first name, type, or class found.
        """
        if self.names:
            return self.names[0]
        if self.types:
            return self.types[0] if len(self.types) == 1 else ", ".join(self.types)
        if self.classes:
            return self.classes[0]
        return "Unknown"
