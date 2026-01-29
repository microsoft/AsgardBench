"""Tests for the prompt DSL parser."""

import pytest

from Magmathor.constants import FeedbackType, PreviousImageType, PromptVersion
from Magmathor.Model.prompt_dsl import (
    ConditionalLine,
    ConditionCode,
    Section,
    format_config_to_dsl_values,
    parse_prompt_string,
)
from Magmathor.Utils.config_utils import EvaluationConfig


class TestConditionCodeParse:
    """Test ConditionCode.parse()"""

    def test_simple_code(self):
        code = ConditionCode.parse("H1")
        assert code.prefix == "H"
        assert code.value == "1"
        assert code.operator == "=="

    def test_single_char_value(self):
        code = ConditionCode.parse("Fn")
        assert code.prefix == "F"
        assert code.value == "n"
        assert code.operator == "=="

    def test_boolean_code(self):
        code = ConditionCode.parse("T1")
        assert code.prefix == "T"
        assert code.value == "1"
        assert code.operator == "=="

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            ConditionCode.parse("invalid")

    def test_wildcard_raises(self):
        with pytest.raises(ValueError):
            ConditionCode.parse("*")


class TestConditionalLineMatches:
    """Test ConditionalLine.matches()"""

    def test_wildcard_always_matches(self):
        line = ConditionalLine(conditions=[], is_wildcard=True, text="default")
        config = EvaluationConfig()
        assert line.matches(config) is True

    def test_text_only_matches(self):
        line = ConditionalLine(
            conditions=[ConditionCode.parse("T1")], is_wildcard=False, text="text mode"
        )
        config_text = EvaluationConfig(text_only=True)
        config_image = EvaluationConfig(text_only=False)

        assert line.matches(config_text) is True
        assert line.matches(config_image) is False

    def test_hand_transparency_visible(self):
        """H1 matches any non-zero hand_transparency"""
        line = ConditionalLine(
            conditions=[ConditionCode.parse("H1")],
            is_wildcard=False,
            text="hand visible",
        )
        config_60 = EvaluationConfig(hand_transparency=60)
        config_50 = EvaluationConfig(hand_transparency=50)
        config_0 = EvaluationConfig(hand_transparency=0)

        assert line.matches(config_60) is True
        assert line.matches(config_50) is True
        assert line.matches(config_0) is False

    def test_hand_transparency_invisible(self):
        """H0 matches only when hand_transparency is 0"""
        line = ConditionalLine(
            conditions=[ConditionCode.parse("H0")],
            is_wildcard=False,
            text="hand invisible",
        )
        config_60 = EvaluationConfig(hand_transparency=60)
        config_0 = EvaluationConfig(hand_transparency=0)

        assert line.matches(config_60) is False
        assert line.matches(config_0) is True

    def test_feedback_type_none(self):
        line = ConditionalLine(
            conditions=[ConditionCode.parse("Fn")],
            is_wildcard=False,
            text="no feedback",
        )
        config_none = EvaluationConfig(feedback_type=FeedbackType.NONE)
        config_simple = EvaluationConfig(feedback_type=FeedbackType.SIMPLE)

        assert line.matches(config_none) is True
        assert line.matches(config_simple) is False

    def test_feedback_type_simple(self):
        line = ConditionalLine(
            conditions=[ConditionCode.parse("Fs")],
            is_wildcard=False,
            text="simple feedback",
        )
        config_simple = EvaluationConfig(feedback_type=FeedbackType.SIMPLE)
        config_detailed = EvaluationConfig(feedback_type=FeedbackType.DETAILED)

        assert line.matches(config_simple) is True
        assert line.matches(config_detailed) is False

    def test_previous_image_color(self):
        line = ConditionalLine(
            conditions=[ConditionCode.parse("I1")],
            is_wildcard=False,
            text="color previous image",
        )
        config_color = EvaluationConfig(previous_image=PreviousImageType.COLOR)
        config_none = EvaluationConfig(previous_image=PreviousImageType.NONE)

        assert line.matches(config_color) is True
        assert line.matches(config_none) is False

    def test_multiple_conditions_and(self):
        line = ConditionalLine(
            conditions=[
                ConditionCode.parse("T1"),
                ConditionCode.parse("R1"),
            ],
            is_wildcard=False,
            text="text mode with memory",
        )
        config_both = EvaluationConfig(text_only=True, use_memory=True)
        config_text_only = EvaluationConfig(text_only=True, use_memory=False)
        config_memory_only = EvaluationConfig(text_only=False, use_memory=True)

        assert line.matches(config_both) is True
        assert line.matches(config_text_only) is False
        assert line.matches(config_memory_only) is False


class TestSectionRender:
    """Test Section.render()"""

    def test_first_match_wins(self):
        section = Section(
            lines=[
                ConditionalLine(
                    conditions=[ConditionCode.parse("H1")],
                    is_wildcard=False,
                    text="hand visible",
                ),
                ConditionalLine(conditions=[], is_wildcard=True, text="hand invisible"),
            ]
        )

        config_visible = EvaluationConfig(hand_transparency=60)
        config_invisible = EvaluationConfig(hand_transparency=0)

        # Both lines match for visible (H1 and wildcard), only wildcard for invisible
        assert section.render(config_visible) == "hand visible\nhand invisible"
        assert section.render(config_invisible) == "hand invisible"

    def test_wildcard_fallback(self):
        section = Section(
            lines=[
                ConditionalLine(
                    conditions=[ConditionCode.parse("T1")],
                    is_wildcard=False,
                    text="text mode",
                ),
                ConditionalLine(conditions=[], is_wildcard=True, text="image mode"),
            ]
        )

        config_text = EvaluationConfig(text_only=True)
        config_image = EvaluationConfig(text_only=False)

        # All matching lines are returned
        assert section.render(config_text) == "text mode\nimage mode"
        assert section.render(config_image) == "image mode"


class TestParsePromptString:
    """Test parse_prompt_string()"""

    def test_simple_section(self):
        content = """## This is a comment
[--------
H1      | Hand is visible
H0      | Hand is invisible
--------]
"""
        template = parse_prompt_string(content)

        config_visible = EvaluationConfig(hand_transparency=60)
        config_invisible = EvaluationConfig(hand_transparency=0)

        assert template.render(config_visible) == "Hand is visible"
        assert template.render(config_invisible) == "Hand is invisible"

    def test_multiline_continuation(self):
        content = """[--------
H1      | This is a long description that
          continues on the next line
          and even a third line.
H0      | Short invisible text.
--------]
"""
        template = parse_prompt_string(content)

        config = EvaluationConfig(hand_transparency=60)
        result = template.render(config)

        assert "This is a long description" in result
        assert "continues on the next line" in result
        assert "and even a third line" in result

    def test_static_text_preserved(self):
        content = """[--------
T1      | Text mode line
T0      | Image mode line
*       | Fallback text
--------]
"""
        template = parse_prompt_string(content)

        config_text = EvaluationConfig(text_only=True)
        config_image = EvaluationConfig(text_only=False)

        # In a group, first match wins - specific conditions match before fallback
        assert template.render(config_text) == "Text mode line"
        assert template.render(config_image) == "Image mode line"

    def test_multiple_sections(self):
        content = """[--------
T1      | Text only
T0      | With images
--------]
[--------
R1      | Using memory
R0      | No memory
--------]
"""
        template = parse_prompt_string(content)

        config = EvaluationConfig(text_only=True, use_memory=False)
        result = template.render(config)

        assert "Text only" in result
        assert "No memory" in result

    def test_compound_conditions(self):
        content = """[--------
T1 R1   | Text mode with memory
T1 R0   | Text mode without memory
T0      | Image mode
--------]
"""
        template = parse_prompt_string(content)

        config_text_mem = EvaluationConfig(text_only=True, use_memory=True)
        config_text_nomem = EvaluationConfig(text_only=True, use_memory=False)
        config_image = EvaluationConfig(text_only=False)

        assert template.render(config_text_mem) == "Text mode with memory"
        assert template.render(config_text_nomem) == "Text mode without memory"
        assert template.render(config_image) == "Image mode"


class TestFormatConfigToDslValues:
    """Test format_config_to_dsl_values()"""

    def test_default_config(self):
        config = EvaluationConfig()
        values = format_config_to_dsl_values(config)

        assert values["T"] == "0"  # text_only=False
        assert values["F"] == "n"  # feedback_type=NONE
        assert values["H"] == "0"  # hand_transparency=0 -> "0"
        assert values["C"] == "1"  # include_common_sense=True
        assert values["I"] == "0"  # previous_image=NONE
        assert values["R"] == "1"  # use_memory=True
        assert values["S"] == "0"  # full_steps=False
        # E, M, and P are not included in DSL
        assert "E" not in values
        assert "M" not in values
        assert "P" not in values

    def test_custom_config(self):
        config = EvaluationConfig(
            text_only=True,
            hand_transparency=60,
            feedback_type=FeedbackType.SIMPLE,
            previous_image=PreviousImageType.COLOR,
            use_memory=False,
        )
        values = format_config_to_dsl_values(config)

        assert values["T"] == "1"
        assert values["H"] == "1"  # Any non-zero -> "1"
        assert values["F"] == "s"
        assert values["I"] == "1"
        assert values["R"] == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
