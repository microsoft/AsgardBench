from enum import Enum
from typing import Dict, List


class GenerationResults:
    def __init__(self):
        self.success: List[str] = []
        self.success_inj: List[str] = []
        self.failures: List[str] = []
        self.existing: List[str] = []
        self.existing_inj: List[str] = []
        self.min_steps: int = 0
        self.max_steps: int = 0
        self.avg_steps: float = 0.0


class Pose:
    def __init__(
        self,
        position: Dict[int, float],
        rotation: float,
        isStanding: bool,
        horizon: float,
    ):
        self.position = position
        self.rotation = rotation
        self.isStanding = isStanding
        self.horizon = horizon

    def to_dict(self) -> Dict[str, any]:
        return {
            "position": self.position,
            "rotation": self.rotation,
            "isStanding": self.isStanding,
            "horizon": self.horizon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "Pose":
        return cls(
            position=data["position"],
            rotation=data["rotation"],
            isStanding=data["isStanding"],
            horizon=data["horizon"],
        )


class StepErrorType(str, Enum):
    # Provided action name is not a valid action type
    INVALID_ACTION = "Invalid Action"

    # Provided object name is not a valid object name
    INVALID_OBJECT = "Invalid Object"

    # Action isn't doable in current state
    UNDOABLE = "Undoable"

    # Unable to parse response message
    INVALID_RESPONSE = "Unparsable"


class StepError:
    def __init__(
        self,
        action_name: str,
        object_name: str,
        error_msg: str,
        error_type: StepErrorType,
    ):
        self.action_name = action_name
        self.object_name = object_name
        self.error_msg = error_msg
        self.error_type = error_type

    def to_dict(self) -> Dict[str, any]:
        return {
            "action_name": self.action_name,
            "object_name": self.object_name,
            "error_msg": self.error_msg,
            "error_type": self.error_type.value if self.error_type else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "StepError":
        return cls(
            action_name=data["action_name"],
            object_name=data["object_name"],
            error_msg=data["error_msg"],
            error_type=(
                StepErrorType(data["error_type"]) if data.get("error_type") else None
            ),
        )


class ObjectMetadata:
    def __init__(self, start_location=None, last_location=None):
        self.start_location = start_location
        self.last_location = last_location


class AgentFailure(Exception):
    """Custom exception with a message."""

    def __init__(self, message=""):
        super().__init__(message)


class AgentCantDo(Exception):
    """Exception raised when an action cannot be performed on a specific object.
    Carries both the error message and error type for proper error reporting."""

    def __init__(self, message="", error_type=None):
        super().__init__(message)
        self.error_type = error_type


class AgentFatal(Exception):
    """Custom exception with a message."""

    def __init__(self, message=""):
        super().__init__(message)


class AgentPolicyError(Exception):
    """Agent detected a responsible AI policy violation."""

    def __init__(self, message=""):
        super().__init__(message)


class ModelEmptyResponseError(Exception):
    """Raised when the model returns no output (empty choices array).

    This indicates the model itself failed to produce a response,
    not a transient API error. Should be caught separately from
    generic API failures.
    """

    pass
