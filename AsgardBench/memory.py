from typing import List


class Memory:

    def __init__(self, name: str):
        self.name: str = name
        self.is_open: bool | None = None
        self.is_dirty: bool | None = None
        self.is_cooked: bool | None = None
        self.is_toggled: bool | None = None
        self.satisfies_goal: str | None = None
        self.filled_with: str | None = None
        self.container: str | None = None

    def to_dict(self) -> dict:
        # Only include non-None properties
        result = {"name": self.name}

        if self.is_open is not None:
            result["is_open"] = self.is_open
        if self.is_dirty is not None:
            result["is_dirty"] = self.is_dirty
        if self.is_cooked is not None:
            result["is_cooked"] = self.is_cooked
        if self.is_toggled is not None:
            result["is_toggled"] = self.is_toggled
        if self.satisfies_goal is not None:
            result["satisfies_goal"] = self.satisfies_goal
        if self.filled_with is not None:
            result["filled_with"] = self.filled_with
        if self.container is not None:
            result["container"] = self.container

        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        # Create memory object with required name field
        memory = cls(name=data["name"])

        # Safely get optional fields with explicit None default
        memory.is_open = data.get("is_open", None)
        memory.is_dirty = data.get("is_dirty", None)
        memory.is_cooked = data.get("is_cooked", None)
        memory.is_toggled = data.get("is_toggled", None)
        memory.satisfies_goal = data.get("satisfies_goal", None)
        memory.filled_with = data.get("filled_with", None)
        memory.container = data.get("container", None)

        return memory

    def is_empty(self) -> bool:
        """
        Returns True if all attributes except name are None.
        """
        return (
            self.is_open is None
            and self.is_dirty is None
            and self.is_cooked is None
            and self.is_toggled is None
            and self.satisfies_goal is None
            and self.filled_with is None
            and self.container is None
        )

    def to_string(self) -> List[str]:
        """
        Returns a string representation of the memory.
        """
        parts = []
        if self.container is not None:
            parts.append(f"{self.name} in {self.container}")

        if self.is_open is not None:
            if self.is_open:
                parts.append(f"{self.name} is open")
            else:
                parts.append(f"{self.name} is closed")

        if self.filled_with is not None:
            parts.append(f"{self.name} filled with {self.filled_with}")

        if self.is_cooked is not None:
            if self.is_cooked:
                parts.append(f"{self.name} is cooked")
            else:
                parts.append(f"{self.name} is not cooked")

        if self.is_toggled is not None:
            if self.is_toggled:
                parts.append(f"{self.name} is on")
            else:
                parts.append(f"{self.name} is off")

        if self.is_dirty is not None:
            if self.is_dirty:
                parts.append(f"{self.name} is dirty")

        if self.satisfies_goal is not None:
            parts.append(f"{self.satisfies_goal}")

        return parts
