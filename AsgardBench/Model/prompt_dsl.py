"""
Prompt DSL Parser

A simple domain-specific language for conditional prompt templates,
replacing Jinja2 with a more readable format.

Format:
    ## Comment lines start with ##

    Conditional groups use [-------- and --------] brackets:
    [--------
        T0   | Text for temperature 0
        T1   | Text for temperature 1
        *    | Default text (fallback)
    --------]
    Within a group, only the FIRST matching line is used (* is fallback).

    Standalone lines (outside groups) are ALL included if they match:
    T0   | This appears for T0
    *    | This appears for ALL configs (including T0)

    CONDITION    Text content for this condition
                 Indented continuation lines are appended

    Multiple conditions can be specified with space separation:
        I1 R1    This matches when I=1 AND R=1

    Wildcards:
        *        Matches any value (default/fallback in groups, always matches standalone)

Condition Codes (from EvaluationConfig._SHORT_NAMES):
    T{0,1}       text_only (0=images, 1=text only)
    F{n,s,d}     feedback_type (n=none, s=simple, d=detailed)
    H{0,1}       hand_transparency (0=invisible, 1=visible/any non-zero)
    C{0,1}       include_common_sense
    I{0,1,2}     previous_image (0=none, 1=color, 2=grayscale)
    R{0,1}       use_memory (remember)
    S{0,1}       full_steps
    A{0,1}       first_action (extra condition, not in EvaluationConfig)

Example:
    # Hand visibility rules (mutually exclusive)
    [--------
        H1   | When you see your robotic hand holding an object, this means
               you are currently holding that item.
        H0   | Objects appear to float when you are holding them.
    --------]
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Optional

from AsgardBench.Utils.config_utils import EvaluationConfig

# Fields from EvaluationConfig._SHORT_NAMES that aren't used in DSL conditions
_EXCLUDED_PREFIXES = {"temperature", "max_completion_tokens", "prompt_version"}


@dataclass
class ConditionCode:
    """A parsed condition code like H60+ or Fn."""

    prefix: str  # The short name prefix (H, F, T, etc.)
    value: str  # The value part (60, n, 1, etc.)
    operator: str  # Comparison: "==" (default), ">=", "<="

    @classmethod
    def parse(cls, code: str) -> ConditionCode:
        """Parse a condition code string like 'H1' or 'Fn'."""
        if not code or code == "*":
            raise ValueError("Cannot parse wildcard as condition code")

        # Valid prefixes are single letters: T, F, H, C, I, R, S, A
        if len(code) < 2:
            raise ValueError(f"Condition code too short: {code}")

        prefix = code[0]
        value = code[1:]

        if prefix not in {"T", "F", "H", "C", "I", "R", "S", "A"}:
            raise ValueError(f"Invalid condition prefix: {prefix}")

        # All comparisons are equality
        return cls(prefix=prefix, value=value, operator="==")


@dataclass
class ConditionalLine:
    """A line with conditions and text content."""

    conditions: list[ConditionCode]  # Empty list means wildcard (*)
    is_wildcard: bool
    text: str

    def matches(
        self,
        config: EvaluationConfig,
        extra_conditions: Optional[dict[str, str]] = None,
    ) -> bool:
        """Check if this line's conditions match the given config.

        Args:
            config: EvaluationConfig to evaluate conditions against
            extra_conditions: Optional dict of {prefix: value} for additional
                             conditions not in EvaluationConfig (e.g., A for first_action)
        """
        if self.is_wildcard:
            return True

        # All conditions must match (AND logic)
        for cond in self.conditions:
            if not self._check_condition(cond, config, extra_conditions):
                return False
        return True

    # Supported condition prefixes - inverted from EvaluationConfig._SHORT_NAMES
    # Uses module-level _EXCLUDED_PREFIXES
    _SUPPORTED_PREFIXES = {
        prefix: field_name
        for field_name, prefix in EvaluationConfig._SHORT_NAMES.items()
        if field_name not in _EXCLUDED_PREFIXES
    }

    def _check_condition(
        self,
        cond: ConditionCode,
        config: EvaluationConfig,
        extra_conditions: Optional[dict[str, str]] = None,
    ) -> bool:
        """Check if a single condition matches the config.

        Supports union values where multiple characters mean OR:
        - Fnd matches Fn OR Fd (feedback_type is none or detailed)
        - I12 matches I1 OR I2 (previous_image is color or grayscale)
        """
        # Check extra conditions first (like A for first_action)
        if extra_conditions and cond.prefix in extra_conditions:
            # For extra conditions, support union matching too
            actual_value = extra_conditions[cond.prefix]
            return actual_value in cond.value

        field_name = self._SUPPORTED_PREFIXES.get(cond.prefix)

        if field_name is None:
            raise ValueError(f"Unknown condition prefix: {cond.prefix}")

        config_value = getattr(config, field_name)
        actual = self._normalize_config_value(config_value, field_name)

        # Check if ANY of the union values match
        # For single-char values, this is just one check
        # For multi-char values like "nd", check each character
        for single_value in cond.value:
            target = self._convert_value(single_value, field_name, config_value)
            if actual == target:
                return True

        return False

    # Inverse mappings derived from EvaluationConfig
    _FEEDBACK_CODE_TO_VALUE = {
        code: value for value, code in EvaluationConfig._FEEDBACK_CODES.items()
    }
    _PREVIOUS_IMAGE_CODE_TO_VALUE = {
        code: value for value, code in EvaluationConfig._PREVIOUS_IMAGE_CODES.items()
    }

    def _convert_value(self, value_str: str, field_name: str, config_value: Any) -> Any:
        """Convert a condition value string to the appropriate type for comparison."""
        # Handle feedback_type - n/s/d codes map to none/simple/detailed
        # This field can be a string or enum, so check by field_name not type
        if field_name == "feedback_type":
            return self._FEEDBACK_CODE_TO_VALUE.get(value_str, value_str)

        # Handle previous_image - 0/1/2 codes map to none/color/grayscale
        if field_name == "previous_image":
            return self._PREVIOUS_IMAGE_CODE_TO_VALUE.get(value_str, value_str)

        # Handle hand_transparency as boolean (0=false, any other=true)
        if field_name == "hand_transparency":
            return value_str == "1"

        # Handle booleans
        if isinstance(config_value, bool):
            return value_str == "1"

        return value_str

    def _normalize_config_value(self, value: Any, field_name: str) -> Any:
        """Normalize a config value for comparison."""
        # For feedback_type and previous_image, return the string value
        # (whether from enum.value or already a string)
        if field_name in ("feedback_type", "previous_image"):
            if isinstance(value, Enum):
                return value.value
            return value

        if isinstance(value, Enum):
            return value.value

        # Treat hand_transparency as boolean (0=false, any other=true)
        if field_name == "hand_transparency":
            return value != 0

        if isinstance(value, bool):
            return value

        return value


@dataclass
class ConditionalGroup:
    """A group of conditional lines delimited by [-------- and --------].

    Within a group, only the FIRST matching line is used.
    The wildcard (*) acts as a fallback if no specific condition matches.
    """

    lines: list[ConditionalLine]

    def render(
        self,
        config: EvaluationConfig,
        extra_conditions: Optional[dict[str, str]] = None,
    ) -> str:
        """Render this group for the given config.

        Returns the text from the FIRST matching line only.
        Wildcard lines are checked last as fallback.
        """
        # First pass: check non-wildcard lines
        for line in self.lines:
            if not line.is_wildcard and line.matches(config, extra_conditions):
                return line.text

        # Second pass: check wildcard lines (fallback)
        for line in self.lines:
            if line.is_wildcard:
                return line.text

        return ""


@dataclass
class Section:
    """A section of conditional lines (legacy, kept for compatibility)."""

    lines: list[ConditionalLine]

    def render(
        self,
        config: EvaluationConfig,
        extra_conditions: Optional[dict[str, str]] = None,
    ) -> str:
        """Render this section for the given config.

        Returns the text from ALL matching lines, joined by newlines.
        """
        matching_lines = []
        for line in self.lines:
            if line.matches(config, extra_conditions):
                matching_lines.append(line.text)
        return "\n".join(matching_lines)


# Type alias for template elements (either a group or an independent line)
TemplateElement = ConditionalGroup | ConditionalLine


class PromptTemplate:
    """A parsed prompt template with conditional groups and independent lines."""

    def __init__(self, elements: list[TemplateElement]):
        """
        Args:
            elements: List of ConditionalGroup or ConditionalLine objects
        """
        self.elements = elements
        # Legacy compatibility
        self.sections: list[Section] = []
        self.static_text: list[str] = []

    def render(
        self,
        config: EvaluationConfig,
        variables: Optional[dict[str, str]] = None,
        extra_conditions: Optional[dict[str, str]] = None,
    ) -> str:
        """Render the complete template for the given config.

        Args:
            config: EvaluationConfig to evaluate conditions against
            variables: Optional dict of {name: value} for variable substitution.
                       Variables in the template are written as {name} and will
                       be replaced with the corresponding value.
            extra_conditions: Optional dict of {prefix: value} for additional
                             conditions not in EvaluationConfig (e.g., {"A": "1"}
                             for first_action=True)

        Returns:
            Rendered template string
        """
        result_parts = []

        for element in self.elements:
            if isinstance(element, ConditionalGroup):
                # Group: render returns FIRST matching line
                output = element.render(config, extra_conditions)
                if output:
                    result_parts.append(output)
            elif isinstance(element, ConditionalLine):
                # Independent line: include if it matches
                if element.matches(config, extra_conditions):
                    result_parts.append(element.text)

        result = "\n".join(result_parts)

        # Apply variable substitutions if provided
        if variables:
            for name, value in variables.items():
                result = result.replace(f"{{{name}}}", value)

        return result


def parse_prompt_file(filepath: str | Path) -> PromptTemplate:
    """Parse a .prompt file and return a PromptTemplate."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return parse_prompt_string(content)


def parse_prompt_string(content: str) -> PromptTemplate:
    """Parse a prompt template string.

    Format:
    - [ starts a conditional group (only first match is used)
    - ] ends a conditional group
    - ## starts a comment line
    - Lines starting with whitespace continue the previous line
    - Standalone lines (outside groups) are ALL included if they match

    Within a group [ ... ], only the FIRST matching line is output.
    The wildcard (*) acts as a fallback if no specific condition matches.

    Outside groups, ALL matching lines are included in output.
    """
    lines = content.split("\n")
    elements: list[TemplateElement] = []
    current_conditional: Optional[ConditionalLine] = None
    in_group = False
    group_lines: list[ConditionalLine] = []
    group_start_line = 0

    for line_num, line in enumerate(lines, 1):
        stripped = line.rstrip()

        # Skip empty lines
        if not stripped:
            continue

        # Skip comment lines (starts with ##)
        if stripped.startswith("##"):
            continue

        # Reject legacy --- separators - must use [-------- and --------] now
        if stripped == "---":
            raise ValueError(
                f"Line {line_num}: Legacy '---' separator is not supported. "
                "Use '[--------' and '--------]' to define conditional groups."
            )

        # Check for group start
        if stripped == "[--------":
            # Save any pending conditional before starting group
            if current_conditional:
                elements.append(current_conditional)
                current_conditional = None
            in_group = True
            group_lines = []
            group_start_line = line_num
            continue

        # Check for group end
        if stripped == "--------]":
            if not in_group:
                raise ValueError(
                    f"Line {line_num}: Unexpected --------] without matching [--------"
                )
            # Save any pending conditional in the group
            if current_conditional:
                group_lines.append(current_conditional)
                current_conditional = None
            # Create the group and add to elements
            if group_lines:
                elements.append(ConditionalGroup(lines=group_lines))
            in_group = False
            group_lines = []
            continue

        # Check if this is a continuation line (starts with whitespace)
        if line and line[0].isspace():
            if current_conditional:
                # Parse continuation - look for | separator
                if "|" in stripped:
                    text = stripped.split("|", 1)[1]
                    if text.startswith(" "):
                        text = text[1:]
                    text = text.replace("\t", "    ")
                    current_conditional.text += "\n" + text
                else:
                    current_conditional.text += "\n" + stripped
            else:
                raise ValueError(
                    f"Line {line_num}: Indented line without preceding condition"
                )
        else:
            # New condition line - save previous if exists
            if current_conditional:
                if in_group:
                    group_lines.append(current_conditional)
                else:
                    elements.append(current_conditional)

            current_conditional = _parse_condition_line(stripped, line_num)

    # Don't forget the last conditional
    if current_conditional:
        if in_group:
            group_lines.append(current_conditional)
        else:
            elements.append(current_conditional)

    # Check for unclosed group
    if in_group:
        raise ValueError(f"Unclosed group starting at line {group_start_line}")

    return PromptTemplate(elements=elements)


# Valid condition prefixes
_VALID_PREFIXES = {"T", "F", "H", "C", "I", "R", "S", "A"}


def _parse_condition_line(line: str, line_num: int) -> ConditionalLine:
    """Parse a condition line like 'T0 C1   Some text here'.

    Preserves the original spacing/indentation in the text portion.
    """
    # Find where the conditions end and text begins
    # Conditions are single uppercase letters followed by values
    # Only valid prefixes are: T, F, H, C, I, R, S

    stripped = line.lstrip()
    if not stripped:
        raise ValueError(f"Line {line_num}: Empty condition line")

    # Check for wildcard
    if stripped.startswith("*"):
        rest = stripped[1:]
        # Use | as separator if present, otherwise fall back to tab or spaces
        if "|" in rest:
            text = rest.split("|", 1)[1]
            if text.startswith(" "):
                text = text[1:]
            text = text.replace("\t", "    ")
        elif "\t" in rest:
            text = rest.split("\t", 1)[1]
            text = text.replace("\t", "    ")
        else:
            text = rest.lstrip()
        return ConditionalLine(conditions=[], is_wildcard=True, text=text)

    # Check for | separator first - it's the preferred format
    if "|" in stripped:
        parts = stripped.split("|", 1)
        condition_part = parts[0].strip()
        text = parts[1]
        # Strip exactly one leading space after | (the separator space)
        if text.startswith(" "):
            text = text[1:]
        # Convert tabs to 4 spaces for output
        text = text.replace("\t", "    ")

        # Parse conditions from the condition part
        conditions = []
        for token in condition_part.split():
            if token[0] in _VALID_PREFIXES and len(token) >= 2:
                cond = ConditionCode.parse(token)
                conditions.append(cond)

        if not conditions:
            raise ValueError(f"Line {line_num}: No valid conditions found in: {line}")

        return ConditionalLine(conditions=conditions, is_wildcard=False, text=text)

    # Fall back to tab or space-separated parsing
    conditions: list[ConditionCode] = []
    pos = 0  # Position in stripped string

    while pos < len(stripped):
        # Skip spaces between conditions (but NOT tabs - tab is separator to text)
        while pos < len(stripped) and stripped[pos] == " ":
            pos += 1

        # If we hit a tab, that's the separator - stop parsing conditions
        if pos < len(stripped) and stripped[pos] == "\t":
            break

        if pos >= len(stripped):
            break

        # Check if this looks like a condition (single letter from valid set + value)
        if stripped[pos] in _VALID_PREFIXES:
            # Find end of this token
            token_end = pos + 1
            while token_end < len(stripped) and not stripped[token_end].isspace():
                token_end += 1

            token = stripped[pos:token_end]

            # Validate it's a proper condition (prefix + at least one char value, not uppercase)
            if len(token) >= 2 and not token[1].isupper():
                try:
                    cond = ConditionCode.parse(token)
                    conditions.append(cond)
                    pos = token_end
                    continue
                except ValueError:
                    pass

        # Not a condition, text starts here
        break

    if not conditions:
        raise ValueError(f"Line {line_num}: No valid conditions found in: {line}")

    # Text is everything after the conditions
    # If there's a tab, use it as the separator and preserve text after it
    remaining = stripped[pos:]
    if "\t" in remaining:
        text = remaining.split("\t", 1)[1]
    else:
        text = remaining.lstrip()
    return ConditionalLine(conditions=conditions, is_wildcard=False, text=text)


# =============================================================================
# Helper function to format condition value for DSL output
# =============================================================================

# DSL prefixes - derived from EvaluationConfig._SHORT_NAMES using shared exclusion list
_DSL_PREFIXES = {
    field_name: prefix
    for field_name, prefix in EvaluationConfig._SHORT_NAMES.items()
    if field_name not in _EXCLUDED_PREFIXES
}


def format_config_to_dsl_values(config: EvaluationConfig) -> dict[str, str]:
    """Convert an EvaluationConfig to DSL condition values.

    Returns a dict mapping short prefix to formatted value.
    E.g., {"T": "0", "F": "n", "H": "1", ...}
    """
    result = {}
    for field_name, short_name in _DSL_PREFIXES.items():
        value = getattr(config, field_name)

        # Special handling for hand_transparency as boolean
        if field_name == "hand_transparency":
            result[short_name] = "1" if value != 0 else "0"
        else:
            formatted = config._format_value_for_suffix(field_name, value)
            result[short_name] = formatted

    return result
