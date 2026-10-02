"""Deterministic Group K-Fold cross-validation splitter for SWE-bench tasks."""

from __future__ import annotations

from src.data.complexity_tier import ComplexityTier
from src.data.task import Task
from src.evaluation.cv_fold import CVFold


class CVSplitter:
    """Deterministic Group K-Fold splitter grouped by repository."""

    _DEFAULT_NUM_FOLDS: int = 4
    _DEFAULT_SEED: int = 42
    _COMPLEXITY_SIMPLE_MAX_LINES: int = 20
    _COMPLEXITY_SIMPLE_MAX_FILES: int = 1
    _COMPLEXITY_MODERATE_MAX_LINES: int = 100
    _COMPLEXITY_MODERATE_MAX_FILES: int = 3
    _DIFF_HEADER: str = "diff --git"
    _DIFF_OLD_PREFIX: str = "--- "
    _DEV_NULL: str = "/dev/null"
    _DEV_NULL_REL: str = "dev/null"
    _ADD_PREFIX: str = "+"
    _REMOVE_PREFIX: str = "-"
    _ADD_HEADER_PREFIX: str = "+++"
    _REMOVE_HEADER_PREFIX: str = "---"
    _SLASH: str = "/"
    _MIN_GROUPS: int = 2
    _ERR_INVALID_FOLD_IDX: str = "Fold index out of bounds: "
    _ERR_INSUFFICIENT_GROUPS: str = (
        "At least 2 distinct groups are required for cross-validation"
    )

    def __init__(
        self,
        num_folds: int = _DEFAULT_NUM_FOLDS,
        random_seed: int = _DEFAULT_SEED,
    ) -> None:
        """Initialize CVSplitter with fold count and random seed."""
        self._num_folds: int = num_folds
        self._random_seed: int = random_seed

    def create_splits(self, tasks: list[Task]) -> list[CVFold]:
        """Create Group K-Fold splits grouped by repository."""
        if not tasks:
            return []
        groups = [self._extract_repo_name(task.repo) for task in tasks]
        unique_groups = sorted(set(groups))
        if len(unique_groups) < self._MIN_GROUPS:
            raise ValueError(self._ERR_INSUFFICIENT_GROUPS)
        n_splits = min(self._num_folds, len(unique_groups))
        split_indices = self._compute_fold_indices(tasks, groups, n_splits)
        folds: list[CVFold] = []
        for fold_idx, (train_idx, val_idx) in enumerate(split_indices):
            folds.append(self._build_cv_fold(tasks, fold_idx, train_idx, val_idx))
        return folds

    def get_fold(self, tasks: list[Task], fold_idx: int) -> CVFold:
        """Get a specific cross-validation fold by index."""
        splits = self.create_splits(tasks)
        if fold_idx < 0 or fold_idx >= len(splits):
            raise IndexError(f"{self._ERR_INVALID_FOLD_IDX}{fold_idx}")
        return splits[fold_idx]

    def _extract_repo_name(self, repo: str) -> str:
        """Extract short repo name from repository identifier."""
        stripped = repo.strip()
        if self._SLASH in stripped:
            return stripped.split(self._SLASH)[-1]
        return stripped

    def _classify_complexity(self, task: Task) -> ComplexityTier:
        """Classify task complexity based on patch files and lines touched."""
        num_files = self._count_patch_files(task.patch)
        num_lines = self._count_patch_lines(task.patch)
        if (
            num_files <= self._COMPLEXITY_SIMPLE_MAX_FILES
            and num_lines < self._COMPLEXITY_SIMPLE_MAX_LINES
        ):
            return ComplexityTier.SIMPLE
        if (
            num_files <= self._COMPLEXITY_MODERATE_MAX_FILES
            and num_lines <= self._COMPLEXITY_MODERATE_MAX_LINES
        ):
            return ComplexityTier.MODERATE
        return ComplexityTier.COMPLEX

    def _compute_complexity_distribution(
        self, tasks: list[Task], task_ids: list[str]
    ) -> dict[str, int]:
        """Count tasks per complexity tier for a subset of task IDs."""
        id_set = set(task_ids)
        dist = {
            ComplexityTier.SIMPLE.value: 0,
            ComplexityTier.MODERATE.value: 0,
            ComplexityTier.COMPLEX.value: 0,
        }
        for task in tasks:
            if task.instance_id in id_set:
                tier = self._classify_complexity(task)
                dist[tier.value] += 1
        return dist

    def _build_cv_fold(
        self,
        tasks: list[Task],
        fold_idx: int,
        train_idx: list[int],
        val_idx: list[int],
    ) -> CVFold:
        """Construct a CVFold instance with metadata and complexity distribution."""
        train_ids = [tasks[i].instance_id for i in train_idx]
        val_ids = [tasks[i].instance_id for i in val_idx]
        val_repo = self._extract_repo_name(tasks[val_idx[0]].repo)
        dist = self._compute_complexity_distribution(tasks, val_ids)
        return CVFold(
            fold_idx=fold_idx,
            train_ids=train_ids,
            val_ids=val_ids,
            val_repo=val_repo,
            complexity_distribution=dist,
        )

    def _compute_fold_indices(
        self, tasks: list[Task], groups: list[str], n_splits: int
    ) -> list[tuple[list[int], list[int]]]:
        """Compute train and validation index pairs using GroupKFold or fallback."""
        try:
            from sklearn.model_selection import GroupKFold

            gkf = GroupKFold(n_splits=n_splits)
            return [
                (list(train_idx), list(val_idx))
                for train_idx, val_idx in gkf.split(tasks, groups=groups)
            ]
        except (ImportError, Exception):
            return self._fallback_group_kfold(groups, n_splits)

    def _fallback_group_kfold(
        self, groups: list[str], n_splits: int
    ) -> list[tuple[list[int], list[int]]]:
        """Deterministic fallback Group K-Fold partition without scikit-learn."""
        unique_groups = sorted(set(groups))
        group_to_fold = {grp: idx % n_splits for idx, grp in enumerate(unique_groups)}
        folds: list[tuple[list[int], list[int]]] = []
        for fold_idx in range(n_splits):
            train_idx = [
                i for i, grp in enumerate(groups) if group_to_fold[grp] != fold_idx
            ]
            val_idx = [
                i for i, grp in enumerate(groups) if group_to_fold[grp] == fold_idx
            ]
            folds.append((train_idx, val_idx))
        return folds

    def _count_patch_files(self, patch: str) -> int:
        """Count distinct files modified in patch."""
        git_count = 0
        old_count = 0
        for line in patch.splitlines():
            if line.startswith(self._DIFF_HEADER):
                git_count += 1
            elif line.startswith(self._DIFF_OLD_PREFIX):
                target = line[len(self._DIFF_OLD_PREFIX) :].strip()
                if target not in (self._DEV_NULL, self._DEV_NULL_REL):
                    old_count += 1
        return git_count if git_count > 0 else old_count

    def _count_patch_lines(self, patch: str) -> int:
        """Count added and removed lines excluding diff headers."""
        count = 0
        for line in patch.splitlines():
            if line.startswith(self._ADD_PREFIX) and not line.startswith(
                self._ADD_HEADER_PREFIX
            ):
                count += 1
            elif line.startswith(self._REMOVE_PREFIX) and not line.startswith(
                self._REMOVE_HEADER_PREFIX
            ):
                count += 1
        return count
