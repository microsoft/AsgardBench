from __future__ import (  # Add this import at the top of the file for forward references
    annotations,
)


class PromptData:
    def __init__(
        self, user_part: str, agent_part: str, natural_language_agent_part: str
    ):
        self.user = user_part
        self.agent = agent_part
        self.thinking = natural_language_agent_part

    def to_dict(self):
        return {
            "user_part": self.user,
            "agent_part": self.agent,
            "natural_language_agent_part": self.thinking,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PromptData:
        return cls(
            user_part=data.get("user_part", ""),
            agent_part=data.get("agent_part", ""),
            natural_language_agent_part=data.get("natural_language_agent_part", ""),
        )
