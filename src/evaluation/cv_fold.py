"""Data model representing a cross-validation fold."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CVFold:
    """Represents train and validation task splits for a cross-validation fold."""

    fold_idx: int
    train_ids: list[str]
    val_ids: list[str]
    val_repo: str
    complexity_distribution: dict[str, int]
