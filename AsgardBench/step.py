from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)

import uuid
from typing import List, Optional

from AsgardBench.constants import Action
from AsgardBench.memory import Memory
from AsgardBench.objects import Pose
from AsgardBench.prompt_data import PromptData


class Step:
    def __init__(
        self,
        action_desc: str,
        action: Action,
        obj: str,
        task_description: str,
        observations: List[str],
        updated_memory: dict,
        object_bounding_boxes: dict,
        pose: Optional[Pose],
        reasoning: List[str],
        filename: Optional[str],
        history: List[str],
        prompt_data: PromptData = None,
        action_success: Optional[bool] = None,
        model_response: Optional[str] = None,
        log: Optional[str] = None,
    ):
        self.memory = []
        self.action_desc = action_desc
        self.action = action
        self.object = obj
        self.task_description = task_description
        self.observations = observations
        self.updated_memory = updated_memory
        self.object_bounding_boxes = object_bounding_boxes
        self.pose = pose
        self.reasoning = reasoning
        self.image_filename = filename
        self.history = history
        self.prompt_data = prompt_data
        self.action_success = action_success
        self.model_response = model_response
        self.partial_prompt: Optional[str] = (
            None  # Extracted scene state and image info from prompt
        )
        self.log = log or ""

    def to_dict(self):
        return {
            "task_description": self.task_description,
            "memory": self.memory,
            "observations": self.observations,
            "updated_memory": self.updated_memory,
            "object_bounding_boxes": self.object_bounding_boxes,
            "pose": self.pose.to_dict() if self.pose else None,
            "reasoning": self.reasoning,
            "action_desc": self.action_desc,
            "action": self.action,
            "object": self.object,
            "image_filename": self.image_filename,
            "history": self.history,
            "formatted": self.prompt_data.to_dict() if self.prompt_data else None,
            "action_success": self.action_success,
            "model_response": self.model_response,
            "partial_prompt": self.partial_prompt,
            "log": self.log,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Step:
        # Add backward compatibility to handle both old and new format
        memory_data = data.get("updated_memory", data.get("memory", {}))

        step = cls(
            action_desc=data["action_desc"],
            action=Action(data["action"]),
            obj=data["object"],
            task_description=data["task_description"],
            observations=data["observations"],
            updated_memory=memory_data,  # Updated parameter name
            object_bounding_boxes=data["object_bounding_boxes"],
            pose=Pose.from_dict(data["pose"]) if data.get("pose") else None,
            reasoning=data["reasoning"],
            filename=data["image_filename"],
            history=data["history"],
            prompt_data=(
                PromptData.from_dict(data["formatted"])
                if data.get("formatted")
                else None
            ),
            action_success=data.get("action_success"),
            model_response=data.get("model_response"),
            log=data.get("log", ""),
        )
        step.partial_prompt = data.get("partial_prompt")
        return step


class RawStep:

    def __init__(
        self,
        action_desc: str,
        action: Action,
        obj: str,
        parent: Optional[RawStep] = None,
        reasoning: str = "",
        include_in_substep: bool = True,
        is_injection: bool = False,
    ):
        self.id = str(uuid.uuid4())[:6]
        self.action_desc = action_desc
        self.action = action
        self.object = obj
        self.parent: Optional[RawStep] = parent
        self.parent_id = parent.id if parent else None
        self.include_in_substep = include_in_substep
        self.is_injection = is_injection
        self.level = parent.level + 1 if parent else 0
        self.task_description = ""
        self.observations: List[str] = []
        self.pose: Optional[Pose] = None  # Placeholder for pose, can be set later
        self.memory: dict = {}
        self.object_bounding_boxes = {}
        self.reasoning: str = reasoning
        self.substeps: List[Step] = []
        self._image_filename: Optional[str] = None

    def to_dict(self):
        return {
            "id": self.id,
            "action_desc": self.action_desc,
            "action": self.action,
            "object": self.object,
            "parent_id": self.parent_id,
            "include_in_substep": self.include_in_substep,
            "is_injection": self.is_injection,
            "level": self.level,
            "reasoning": self.reasoning,
            "task_description": self.task_description,
            "observations": self.observations,
            "pose": self.pose.to_dict() if self.pose else None,
            # Convert dictionary of memory objects to a dictionary
            "memory": {name: mem.to_dict() for name, mem in self.memory.items()},
            "object_bounding_boxes": self.object_bounding_boxes,
            "image_filename": self.image_filename(),
            "substeps": [s.to_dict() for s in self.substeps],
        }

    @classmethod
    def from_dict(cls, data: dict) -> RawStep:
        action_data = data.get("action", None)
        step = cls(
            action_desc=data["action_desc"],
            action=Action(action_data) if action_data is not None else None,
            obj=data["object"],
            parent=None,  # Parent will be set later
            reasoning=data["reasoning"],
            include_in_substep=data["include_in_substep"],
            is_injection=data["is_injection"],
        )
        step.id = data["id"]
        step.parent_id = data.get("parent_id", None)
        step.level = data["level"]
        step.task_description = data["task_description"]
        step.observations = data["observations"]

        pose_data = data.get("pose", None)
        step.pose = Pose.from_dict(pose_data) if pose_data is not None else None
        # Convert back into dictionary of Memory objects
        step.memory = {
            name: Memory.from_dict(mem) for name, mem in data["memory"].items()
        }
        step.object_bounding_boxes = data["object_bounding_boxes"]
        step._image_filename = data.get("image_filename", None)

        for substep_data in data.get("substeps", []):
            substep = cls.from_dict(substep_data)
            substep.parent = step
            step.substeps.append(substep)
        return step

    def set_image_filename(self, filename: str):
        if self._image_filename is None:
            self._image_filename = filename
        else:
            raise Exception("Image filename already set for this step.")

    def image_filename(self):
        if self._image_filename is None:
            return None
        return self._image_filename

    def step_count(self):
        count = 0
        if self.action_desc:
            count += 1
            for substep in self.substeps:
                count += substep.step_count()
        return count
