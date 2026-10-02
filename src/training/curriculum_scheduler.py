"""Curriculum learning scheduler for phased training complexity filtering."""

from __future__ import annotations

from typing import Any


class CurriculumScheduler:
    """Filters training datasets based on task complexity across training epochs."""

    _EARLY_STAGE_CUTOFF: float = 0.4
    _MIDDLE_STAGE_CUTOFF: float = 0.8
    _TIER_SIMPLE: str = "SIMPLE"
    _TIER_MODERATE: str = "MODERATE"
    _COMPLEXITY_FIELD: str = "complexity"

    def __init__(self) -> None:
        """Initialize CurriculumScheduler."""
        pass

    def get_epoch_data(self, dataset: object, epoch: int, total_epochs: int) -> object:
        """Filter dataset by complexity tier matching the current curriculum stage."""
        if total_epochs <= 0:
            raise ValueError("total_epochs must be greater than zero")
        progress = self._compute_progress(epoch, total_epochs)
        if progress < self._EARLY_STAGE_CUTOFF:
            return self._filter_dataset(dataset, (self._TIER_SIMPLE,))
        if progress < self._MIDDLE_STAGE_CUTOFF:
            return self._filter_dataset(
                dataset, (self._TIER_SIMPLE, self._TIER_MODERATE)
            )
        return dataset

    def _compute_progress(self, epoch: int, total_epochs: int) -> float:
        """Compute training progress ratio given current epoch and total epochs."""
        return float(epoch) / float(total_epochs)

    def _filter_dataset(
        self, dataset: object, allowed_tiers: tuple[str, ...]
    ) -> object:
        """Filter dataset records against an allowed tuple of complexity tiers."""
        if hasattr(dataset, "filter"):
            return getattr(dataset, "filter")(
                lambda item: self._is_allowed_tier(item, allowed_tiers)
            )
        if isinstance(dataset, list):
            return [
                item for item in dataset if self._is_allowed_tier(item, allowed_tiers)
            ]
        return dataset

    def _is_allowed_tier(
        self, item: Any, allowed_tiers: tuple[str, ...]
    ) -> bool:
        """Check if an item matches one of the allowed complexity tiers."""
        if isinstance(item, dict):
            return item.get(self._COMPLEXITY_FIELD) in allowed_tiers
        return getattr(item, self._COMPLEXITY_FIELD, None) in allowed_tiers
