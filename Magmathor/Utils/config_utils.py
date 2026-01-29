import argparse
from dataclasses import Field, asdict, dataclass, fields
from enum import Enum
from typing import Any, ClassVar, get_type_hints

from Magmathor.constants import FeedbackType, PreviousImageType, PromptVersion


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation parameters.

    Field metadata can include 'help' key for argparse help text.
    Boolean fields automatically become action='store_true' flags.
    """

    text_only: bool = False
    """Whether to use text-only mode (no images sent to model)"""

    feedback_type: FeedbackType = FeedbackType.NONE
    """Type of feedback to include in prompts"""

    hand_transparency: int = 0
    """Transparency level for hand rendering in images (0.0-1.0)"""

    include_common_sense: bool = True
    """Whether to include common sense rules in prompts"""

    prompt_version: PromptVersion = PromptVersion.V2
    """Prompt template version (v1=original monolithic, v2=new modular)"""

    previous_image: PreviousImageType = PreviousImageType.NONE
    """Type of previous image to include (none, color, grayscale)"""

    use_memory: bool = True
    """Whether to use memory for context persistence"""

    full_steps: bool = False
    """Whether to generate full action sequence instead of single next action"""

    temperature: float = 0.6
    """Temperature for response generation"""

    max_completion_tokens: int = 4096
    """Maximum tokens for model completion"""

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return asdict(self)

    # Obsolete fields that should be ignored when loading old config files
    _OBSOLETE_FIELDS: ClassVar[set[str]] = {
        "include_simulation",
        "flavor",
        "image_count",  # Converted to previous_image
        "include_previous_image",  # Converted to previous_image
    }

    @classmethod
    def normalize_config_dict(cls, data: dict) -> dict:
        """Normalize a config dictionary, handling backwards compatibility.

        Converts obsolete fields to their new equivalents:
        - image_count (1 or 2) -> previous_image (none or color)
        - include_previous_image (bool) -> previous_image (none or color)

        Removes obsolete fields that have no equivalent:
        - include_simulation
        - flavor

        Args:
            data: Raw config dictionary (may contain obsolete fields)

        Returns:
            Normalized config dictionary with only current field names
        """
        normalized = dict(data)  # Make a copy

        # Convert image_count to previous_image
        if "image_count" in normalized:
            image_count = normalized.pop("image_count")
            # Only set if previous_image isn't already present
            if "previous_image" not in normalized:
                # image_count=1 -> none, image_count=2 -> color
                normalized["previous_image"] = "color" if image_count >= 2 else "none"

        # Convert include_previous_image (bool) to previous_image (enum)
        if "include_previous_image" in normalized:
            include_prev = normalized.pop("include_previous_image")
            # Only set if previous_image isn't already present
            if "previous_image" not in normalized:
                normalized["previous_image"] = "color" if include_prev else "none"

        # Remove other obsolete fields
        for obsolete_field in cls._OBSOLETE_FIELDS:
            normalized.pop(obsolete_field, None)

        return normalized

    @classmethod
    def load_from_file(cls, filepath: str) -> tuple["EvaluationConfig", dict]:
        """Load config from a JSON file with backwards compatibility handling.

        Args:
            filepath: Path to the config.json file

        Returns:
            Tuple of (EvaluationConfig instance, normalized config dict)
        """
        import json

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        normalized = cls.normalize_config_dict(raw_data)

        # Extract only fields that are part of EvaluationConfig
        valid_field_names = {field.name for field in fields(cls)}
        config_kwargs = {k: v for k, v in normalized.items() if k in valid_field_names}

        # Handle enum conversions
        if "feedback_type" in config_kwargs:
            from Magmathor.constants import FeedbackType

            val = config_kwargs["feedback_type"]
            if isinstance(val, str):
                config_kwargs["feedback_type"] = FeedbackType(val)

        if "prompt_version" in config_kwargs:
            from Magmathor.constants import PromptVersion

            val = config_kwargs["prompt_version"]
            if isinstance(val, str):
                config_kwargs["prompt_version"] = PromptVersion(val)

        if "previous_image" in config_kwargs:
            from Magmathor.constants import PreviousImageType

            val = config_kwargs["previous_image"]
            if isinstance(val, str):
                config_kwargs["previous_image"] = PreviousImageType(val)

        config = cls(**config_kwargs)
        return config, normalized

    @classmethod
    def from_dict(cls, data: dict) -> "EvaluationConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def _get_field_help(cls, field: Field) -> str:
        """Extract help text from field's docstring or metadata.

        Args:
            field: Dataclass field

        Returns:
            Help text for the field
        """
        # Try to get help from field metadata first
        if hasattr(field, "metadata") and "help" in field.metadata:
            return field.metadata["help"]

        # Otherwise use the docstring comment that follows the field
        # This is extracted from the class's __doc__ string
        # We'll get it from the source if available
        try:
            import inspect

            source = inspect.getsource(cls)
            lines = source.split("\n")

            # Find the line with this field name
            for i, line in enumerate(lines):
                if f"{field.name}:" in line and "=" in line:
                    # Check if next line is a docstring
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('"""') and next_line.endswith('"""'):
                            return next_line.strip('"""').strip()

        except Exception:  # pylint: disable=broad-except
            pass

        # Fallback: generate from field name
        return f"{field.name.replace('_', ' ').title()}"

    @classmethod
    def add_argparse_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add configuration arguments to an argparse parser using reflection.

        This method automatically introspects the dataclass fields and adds
        appropriate arguments to the parser. Boolean fields become flags,
        other fields become value arguments.

        Args:
            parser: ArgumentParser to add arguments to
        """
        # Get type hints to properly resolve types
        type_hints = get_type_hints(cls)

        for field in fields(cls):
            # (snake_case)
            arg_name = f"--{field.name}"

            # Get help text
            help_text = cls._get_field_help(field)

            # Get the actual type from type hints
            field_type = type_hints.get(field.name, field.type)

            # Boolean fields become store_true actions
            if field_type == bool:
                parser.add_argument(arg_name, action="store_true", help=help_text)
            # Enum fields use choices and convert string to enum
            elif isinstance(field_type, type) and issubclass(field_type, Enum):
                choices = [e.value for e in field_type]
                parser.add_argument(
                    arg_name,
                    type=str,
                    choices=choices,
                    default=(
                        field.default.value
                        if isinstance(field.default, Enum)
                        else field.default
                    ),
                    help=f"{help_text} (choices: {', '.join(choices)})",
                )
            else:
                # Other fields are value arguments
                parser.add_argument(
                    arg_name, type=field_type, default=field.default, help=help_text
                )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "EvaluationConfig":
        """Create config from argparse arguments using reflection.

        This method automatically extracts values for all dataclass fields
        from the parsed arguments.

        Args:
            args: Parsed arguments from argparse

        Returns:
            EvaluationConfig instance with values from args
        """
        # Get type hints to properly resolve types
        type_hints = get_type_hints(cls)

        # Build kwargs dict from args using field names
        kwargs = {}
        for field in fields(cls):
            value = getattr(args, field.name)
            field_type = type_hints.get(field.name, field.type)

            # Convert string back to enum if needed
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                if isinstance(value, str):
                    value = field_type(value)

            kwargs[field.name] = value

        hydrated_config = cls(**kwargs)

        assert (
            hydrated_config.hand_transparency >= 0
            and hydrated_config.hand_transparency <= 100
        ), "hand_transparency must be between 0 and 100"

        assert (
            hydrated_config.temperature >= 0.0 and hydrated_config.temperature <= 1.0
        ), "temperature must be between 0.0 and 1.0"

        if hydrated_config.text_only:
            assert (
                hydrated_config.previous_image == PreviousImageType.NONE
            ), "text_only mode cannot have previous_image set"
            assert (
                hydrated_config.hand_transparency == 0
            ), "text_only mode must have hand_transparency of 0"

        return hydrated_config

    def _format_value(self, value: Any) -> str:
        """Format a config value for output suffix.

        Args:
            value: The value to format

        Returns:
            Formatted string representation
        """
        if isinstance(value, Enum):
            # Use just the enum value name (e.g., "none" from FeedbackType.NONE)
            return value.value
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, float):
            # Remove trailing zeros: 0.60 -> 0.6
            return f"{value:g}"
        return str(value)

    # Short names for parameters to keep output folder names concise
    # Maps field_name -> short_prefix
    # Format: T{text}_F{feedback}_H{hand}_C{common}_P{prompt}_I{images}_R{remember}_E{temp}_M{max}
    _SHORT_NAMES: ClassVar[dict[str, str]] = {
        "text_only": "T",
        "feedback_type": "F",
        "hand_transparency": "H",
        "include_common_sense": "C",
        "prompt_version": "P",
        "previous_image": "I",
        "use_memory": "R",
        "full_steps": "S",
        "temperature": "E",
        "max_completion_tokens": "M",
    }

    # Mapping for FeedbackType to single-letter codes
    _FEEDBACK_CODES: ClassVar[dict[str, str]] = {
        "none": "n",
        "simple": "s",
        "detailed": "d",
    }

    # Mapping for PreviousImageType to numeric codes
    _PREVIOUS_IMAGE_CODES: ClassVar[dict[str, str]] = {
        "none": "0",
        "color": "1",
        "grayscale": "2",
    }

    def _format_value_for_suffix(self, field_name: str, value: Any) -> str:
        """Format a config value for the new naming convention suffix.

        Args:
            field_name: Name of the field being formatted
            value: The value to format

        Returns:
            Formatted string representation for the suffix
        """

        if isinstance(value, Enum):
            value = value.value
        else:
            value = value

        # Special handling for feedback_type -> single letter code
        if field_name == "feedback_type":
            return self._FEEDBACK_CODES.get(value, value) or value

        # Special handling for previous_image -> numeric code
        if field_name == "previous_image":
            return self._PREVIOUS_IMAGE_CODES.get(value, "0")
        # Special handling for prompt_version -> extract numeric part
        if field_name == "prompt_version":
            # "v1" -> "1", "v2" -> "2"
            return value.lstrip("v")

        # Special handling for hand_transparency -> 2-digit padded
        if field_name == "hand_transparency":
            return f"{value:02d}"

        # Special handling for temperature -> int * 100 (0.6 -> 60)
        if field_name == "temperature":
            return str(int(value * 100))

        # Boolean -> 0/1
        if isinstance(value, bool):
            return "1" if value else "0"

        return str(value)

    def get_output_suffix(self) -> str:
        """Generate suffix for output directory based on config.

        Uses the new naming convention with fixed order:
        T{text}_F{feedback}_H{hand}_C{common}_P{prompt}_I{images}_R{remember}_S{full_steps}_E{temp}_M{max}

        Examples:
            - T1_Fn_H00_C0_P1_I1_R0_S0_E60_M4096 (text only, no feedback, temp 0.6)
            - T0_Fs_H50_C1_P2_I2_R1_S1_E80_M8192 (with images, simple feedback, full_steps temp 0.8)

        Returns:
            Suffix string in the new naming convention format
        """
        parts = []

        for field in fields(self):
            value = getattr(self, field.name)
            short_name = self._SHORT_NAMES.get(field.name, field.name)
            formatted_value = self._format_value_for_suffix(field.name, value)
            parts.append(f"{short_name}{formatted_value}")

        return "_".join(parts)
