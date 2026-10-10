# Epic 4 — Trajectory Synthesis

> **Runs on:** Local machine
> **Depends on:** Epics 2 (Data Models), 3 (Data Ingestion)
> **Estimated effort:** ~3 hours
> **Goal:** Implement `TrajectorySynthesiser` (convert `(problem, patch)` pairs into multi-turn agent trajectories with tool calls) and `TrajectoryAugmentor` (diversify trajectories via augmentation strategies).

---

## Pre-Requisites

- Epics 2 and 3 are complete
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Design Reference

- `detailed/01_data_preprocessing.md` § 3 — Trajectory Synthesis Algorithm
- `high_level/03_data_flow_pipeline.md` § 3 — Synthesis + Augmentation

---

## Task 4.1: Create `src/data/trajectory_synthesiser.py`

**Class:** `TrajectorySynthesiser`

**Design reference:** `detailed/01_data_preprocessing.md` § 3

**Constructor:**
```python
def __init__(
    self,
    ingestor: DataIngestor,
    patch_parser: PatchParser,
    token_counter: TokenCounter,
    mock_mode: bool = False,
) -> None:
```

**Class-level constants:**
- `_MAX_TRAJECTORY_TOKENS: int = 28672`
- `_READ_CONTEXT_LINES: int = 30`
- `_MAX_OLD_STRING_LINES: int = 15`

**Public methods:**
- `synthesise(task: Task) -> Trajectory` — convert a single task into a gold trajectory
- `synthesise_all(tasks: list[Task]) -> list[Trajectory]` — batch synthesis with filtering

**Private methods (each ≤ 20 lines):**
- `_build_initial_prompt(task: Task) -> Turn` — create the first user turn with problem statement, budget info, and environment rules (mirroring the harness `build_agent_prompt`)
- `_synthesise_navigation(task: Task, graph: object, symbols: list[str]) -> list[Turn]` — create turns for `search_similar_code` + `get_code_neighbors` calls
- `_synthesise_diagnosis(task: Task, file_changes: list[FileChange]) -> list[Turn]` — create turns for `read_file` calls and model thinking about root cause
- `_synthesise_patches(task: Task, file_changes: list[FileChange]) -> list[Turn]` — create `edit_file` tool call turns for each hunk
- `_synthesise_verification(task: Task) -> list[Turn]` — create `run_command` turn for pytest
- `_synthesise_submission(task: Task) -> list[Turn]` — create `submit_patch` call
- `_extract_symbols(file_changes: list[FileChange], graph: object) -> list[str]` — find function/class names from the changed code
- `_extract_keywords(problem_statement: str) -> str` — extract top keywords
- `_simulate_search(query: str, graph: object) -> dict[str, object]` — simulate `search_similar_code` tool result JSON
- `_simulate_neighbors(node: str, graph: object) -> dict[str, object]` — simulate `get_code_neighbors` tool result JSON
- `_simulate_read_file(filepath: str, start_line: int, end_line: int) -> dict[str, object]` — simulate `read_file` tool result
- `_simulate_edit_result(filepath: str) -> dict[str, object]` — simulate `edit_file` success result
- `_simulate_pytest_result(task: Task) -> dict[str, object]` — simulate pytest pass result
- `_split_hunk(hunk: Hunk) -> tuple[str, str]` — split a large hunk into smaller old_string/new_string
- `_classify(task: Task) -> ComplexityTier` — delegate to ingestor
- `_count_tokens(turns: list[Turn]) -> int` — delegate to token_counter

**Key implementation notes:**
- For mock mode (`mock_mode=True`), skip graph/embedding loading and use simplified simulations
- Tool results must be valid JSON matching the harness tool output schemas
- The `initial_prompt` Turn must match the format in `high_level/03_data_flow_pipeline.md` § 4.1

---

## Task 4.2: Create `src/data/trajectory_augmentor.py`

**Class:** `TrajectoryAugmentor`

**Design reference:** `detailed/01_data_preprocessing.md` § 3.2 (design doc `high_level/03_data_flow_pipeline.md` § 3.2)

**Constructor:**
```python
def __init__(self, seed: int = 42) -> None:
```

**Class-level constants:**
- `_MAX_AUGMENTATIONS_PER_TRAJECTORY: int = 3`

**Public methods:**
- `augment(trajectory: Trajectory) -> list[Trajectory]` — produce augmented variants of a single trajectory

**Private methods (each implements one augmentation strategy, ≤ 20 lines):**
- `_permute_tool_order(trajectory: Trajectory) -> Trajectory` — swap order of navigation tool calls (e.g., `search_similar_code` before/after `read_file`)
- `_inject_error_recovery(trajectory: Trajectory) -> Trajectory` — insert a failed `edit_file` call (wrong `old_string`) followed by a correction
- `_vary_read_windows(trajectory: Trajectory) -> Trajectory` — change `start_line`/`end_line` in `read_file` calls
- `_toggle_graph_vs_grep(trajectory: Trajectory) -> Trajectory` — replace `search_similar_code` with `run_command("grep ...")` or vice versa
- `_add_scratchpad_discipline(trajectory: Trajectory) -> Trajectory` — insert a `write_file("/tmp/repro.py", ...)` + `run_command` + cleanup turn
- `_add_budget_checks(trajectory: Trajectory) -> Trajectory` — insert `get_status()` calls at 1/3 and 2/3 points
- `_clone_trajectory(trajectory: Trajectory) -> Trajectory` — deep copy helper

---

## Task 4.3: Create `src/data/trajectory_validator.py`

**Class:** `TrajectoryValidator`

**Design reference:** `detailed/01_data_preprocessing.md` § 5

**Constructor:**
```python
def __init__(self, token_counter: TokenCounter) -> None:
```

**Class-level constants:**
- `_MAX_TOKENS: int = 28672`
- `_MAX_TOOL_CALLS: int = 80`

**Public methods:**
- `validate(trajectory: Trajectory) -> "ValidationResult"` — run all validation checks

**Private methods:**
- `_check_token_budget(trajectory: Trajectory) -> list[str]` — errors if over budget
- `_check_tool_call_budget(trajectory: Trajectory) -> list[str]` — warnings if over recommended
- `_check_edit_file_validity(trajectory: Trajectory) -> list[str]` — verify `old_string` args exist
- `_check_no_test_modifications(trajectory: Trajectory) -> list[str]` — verify no test file edits
- `_check_scratch_file_paths(trajectory: Trajectory) -> list[str]` — verify `/tmp/` usage
- `_check_submit_patch_is_last(trajectory: Trajectory) -> list[str]` — verify final tool call
- `_is_test_file(filepath: str) -> bool` — check if path matches test patterns

### `src/data/validation_result.py`
**Class:** `ValidationResult` (frozen dataclass)
**Fields:**
- `valid: bool`
- `errors: list[str]`
- `warnings: list[str]`

---

## Task 4.4: Write Tests

### `tests/unit/data/test_trajectory_synthesiser.py`
- `test_synthesise_mock_task_returns_trajectory` — use MockDataFactory, verify trajectory structure
- `test_synthesise_produces_correct_turn_order` — user → model → user → model → ...
- `test_synthesise_initial_prompt_contains_problem_statement`
- `test_synthesise_includes_navigation_tools` — verify `search_similar_code` or `get_code_neighbors` calls
- `test_synthesise_includes_edit_file_calls` — verify at least one `edit_file` call
- `test_synthesise_ends_with_submit_patch`
- `test_synthesise_all_filters_over_budget` — inject an oversized task, verify it's excluded
- `test_synthesise_mock_mode_works_without_graph_files`

### `tests/unit/data/test_trajectory_augmentor.py`
- `test_augment_returns_list_of_trajectories`
- `test_augment_preserves_instance_id`
- `test_permute_tool_order_changes_tool_sequence`
- `test_inject_error_recovery_adds_failed_edit`
- `test_add_budget_checks_inserts_get_status`
- `test_augment_respects_max_augmentations`

### `tests/unit/data/test_trajectory_validator.py`
- `test_validate_valid_trajectory_returns_valid`
- `test_validate_over_token_budget_returns_invalid`
- `test_validate_test_file_modification_returns_invalid`
- `test_validate_scratch_in_workspace_returns_invalid`
- `test_validate_no_submit_patch_warns`

---

## Task 4.5: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `TrajectorySynthesiser.synthesise()` produces valid multi-turn trajectories from mock tasks
- [ ] `TrajectoryAugmentor` produces augmented variants
- [ ] `TrajectoryValidator` catches invalid trajectories
- [ ] Mock mode works without GPU or real data files
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/data/trajectory_synthesiser.py
src/data/trajectory_augmentor.py
src/data/trajectory_validator.py
src/data/validation_result.py
tests/unit/data/test_trajectory_synthesiser.py
tests/unit/data/test_trajectory_augmentor.py
tests/unit/data/test_trajectory_validator.py
```
