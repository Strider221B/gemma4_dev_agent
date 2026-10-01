"""Complexity tier classification for SWE-bench tasks."""

from enum import Enum


class ComplexityTier(Enum):
    """Task complexity classification based on patch scope."""

    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"
