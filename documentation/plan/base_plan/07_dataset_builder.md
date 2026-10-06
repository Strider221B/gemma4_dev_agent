# Epic 6 — Dataset Builder

> **Runs on:** Local machine
> **Depends on:** Epic 5 (Chat Formatting)
> **Estimated effort:** ~2 hours
> **Goal:** Implement `DatasetBuilder` (orchestrates full data pipeline: ingest → synthesise → augment → format → split → HuggingFace Dataset) and `CVSplitter` (Group K-Fold splitting by repository).

---

## Pre-Requisites

- Epic 5 is complete
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Design Reference

- `detailed/01_data_preprocessing.md` § 1 (DatasetBuilder class diagram)
- `high_level/03_data_flow_pipeline.md` § 5 — Dataset Assembly
- `high_level/05_evaluation_framework.md` § 2.1 — CVSplitter

---

## Task 6.1: Create `src/evaluation/cv_splitter.py`

**Class:** `CVSplitter`

**Design reference:** `high_level/05_evaluation_framework.md` § 2.1

**Constructor:**
```python
def __init__(self, num_folds: int = 4, random_seed: int = 42) -> None:
```

**Class-level constants:**
- `_DEFAULT_NUM_FOLDS: int = 4`
- `_DEFAULT_SEED: int = 42`

**Public methods:**
- `create_splits(tasks: list[Task]) -> list[CVFold]` — create Group K-Fold splits grouped by repository
- `get_fold(tasks: list[Task], fold_idx: int) -> CVFold` — get a specific fold

**Private methods:**
- `_extract_repo_name(repo: str) -> str` — extract short repo name (e.g., `"fastapi"` from `"tiangolo/fastapi"`)
- `_classify_complexity(task: Task) -> ComplexityTier` — classify task complexity
- `_compute_complexity_distribution(tasks: list[Task], task_ids: list[str]) -> dict[str, int]` — count tasks per complexity tier

**Implementation notes:**
- Use `sklearn.model_selection.GroupKFold` for splitting
- Groups are repository names
- Each fold holds out one entire repository (4 repos → 4 natural folds)
- Record complexity distribution per fold for reporting

---

## Task 6.2: Create `src/data/dataset_builder.py`

**Class:** `DatasetBuilder`

**Constructor:**
```python
def __init__(
    self,
    ingestor: DataIngestor,
    synthesiser: TrajectorySynthesiser,
    augmentor: TrajectoryAugmentor,
    formatter: ChatFormatter,
    validator: TrajectoryValidator,
    splitter: CVSplitter,
    mock_mode: bool = False,
) -> None:
```

**Public methods:**
- `build() -> object` — full pipeline: load → synthesise → augment → validate → format → split → return HuggingFace Dataset. Return type is `object` (HuggingFace `Dataset` is optional import).
- `build_mock() -> object` — mock pipeline using `MockDataFactory`, bypassing real data loading

**Private methods:**
- `_load_tasks() -> list[Task]` — delegate to ingestor
- `_synthesise_trajectories(tasks: list[Task]) -> list[Trajectory]` — run synthesis
- `_augment_trajectories(trajectories: list[Trajectory]) -> list[Trajectory]` — run augmentation
- `_validate_trajectories(trajectories: list[Trajectory]) -> list[Trajectory]` — filter invalid
- `_format_trajectories(trajectories: list[Trajectory]) -> list[dict[str, object]]` — format to chat text + metadata
- `_assign_folds(records: list[dict[str, object]], tasks: list[Task]) -> list[dict[str, object]]` — add fold assignments
- `_build_hf_dataset(records: list[dict[str, object]]) -> object` — create HuggingFace Dataset
- `_filter_by_token_budget(trajectories: list[Trajectory]) -> list[Trajectory]` — remove over-budget

**Dataset schema (columns of the HF Dataset):**
- `instance_id: str`
- `repo: str`
- `complexity: str` (SIMPLE/MODERATE/COMPLEX)
- `messages: list[dict]` (HF conversation format)
- `formatted_text: str` (raw chat template text)
- `token_count: int`
- `num_tool_calls: int`
- `num_files_changed: int`
- `fold: int`

---

## Task 6.3: Write Tests

### `tests/unit/evaluation/test_cv_splitter.py`
- `test_create_splits_returns_correct_fold_count` — 4 repos → 4 folds
- `test_create_splits_each_fold_holds_out_one_repo` — each fold's val_repo is unique
- `test_create_splits_no_train_val_overlap` — train_ids ∩ val_ids = ∅ for each fold
- `test_create_splits_all_tasks_covered` — union of all val_ids = all instance_ids
- `test_get_fold_by_index`
- `test_create_splits_complexity_distribution_populated`

### `tests/unit/data/test_dataset_builder.py`
- `test_build_mock_returns_dataset` — verify mock pipeline produces a dataset
- `test_build_mock_has_required_columns` — `instance_id`, `messages`, `formatted_text`, `fold`
- `test_build_mock_formatted_text_contains_special_tokens`
- `test_build_filters_invalid_trajectories`
- `test_build_assigns_folds`

**Integration test:**

### `tests/integration/test_data_pipeline.py`
- `test_full_mock_pipeline` — end-to-end: `MockDataFactory` → `DatasetBuilder.build_mock()` → verify dataset structure

---

## Task 6.4: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `CVSplitter` produces 4 folds with one repo held out per fold
- [ ] `DatasetBuilder.build_mock()` produces a valid HuggingFace Dataset
- [ ] Dataset has all required columns
- [ ] `formatted_text` contains Gemma 4 special tokens
- [ ] Invalid trajectories are filtered out
- [ ] Integration test passes end-to-end
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/evaluation/cv_splitter.py
src/data/dataset_builder.py
tests/unit/evaluation/test_cv_splitter.py
tests/unit/data/test_dataset_builder.py
tests/integration/test_data_pipeline.py
```
