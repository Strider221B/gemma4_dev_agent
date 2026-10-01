"""Overfitting signal classification for evaluation tracking."""

from enum import Enum


class OverfittingSignal(Enum):
    """Categorical signal indicating public/private evaluation discrepancy."""

    NONE = "NONE"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
