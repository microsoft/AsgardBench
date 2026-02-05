"""
Step logging module.

Provides a log_print function that both prints to the terminal and appends
the message to the current step's log field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from AsgardBench.step import Step

# Module-level reference to the current step being processed
_current_step: Optional[Step] = None

# Buffer for log messages before a step is created
_log_buffer: str = ""


def set_current_step(step: Optional[Step]) -> None:
    """Set the current step for logging and flush any buffered messages."""
    global _current_step, _log_buffer
    _current_step = step

    # Flush buffered messages to the step
    if _current_step is not None and _log_buffer:
        _current_step.log += _log_buffer
        _log_buffer = ""


def get_current_step() -> Optional[Step]:
    """Get the current step for logging."""
    return _current_step


def clear_log_buffer() -> None:
    """Clear the log buffer without flushing to a step."""
    global _log_buffer
    _log_buffer = ""


def log_print(*args, **kwargs) -> None:
    """
    Print to the terminal and append the message to the current step's log.

    Works like the built-in print function, but also appends the output
    to the current step's log field if a step is set. If no step is set,
    the message is buffered and will be flushed when a step is set.
    """
    global _log_buffer

    # Build the message string the same way print would
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    message = sep.join(str(arg) for arg in args) + end

    # Print to terminal (without the end since we already added it to message)
    print(*args, **kwargs)

    # Append to current step's log if available, otherwise buffer
    if _current_step is not None:
        _current_step.log += message
    else:
        _log_buffer += message
