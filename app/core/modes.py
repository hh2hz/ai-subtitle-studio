"""Processing modes."""

from enum import Enum


class Mode(str, Enum):
    """Processing mode selected by the user. Values are stable storage keys."""

    FAST = "fast"
    BALANCED = "balanced"
    MAXIMUM_ACCURACY = "maximum_accuracy"


DEFAULT_MODE = Mode.BALANCED
