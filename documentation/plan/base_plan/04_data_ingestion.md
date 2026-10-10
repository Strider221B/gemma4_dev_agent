# Epic 3 — Data Ingestion

> **Runs on:** Local machine
> **Depends on:** Epic 2 (Data Models)
> **Estimated effort:** ~2 hours
> **Goal:** Implement `DataIngestor` (parse `tasks.jsonl`, load code graphs, load embeddings), `PatchParser` (parse unified diffs), and `MockDataFactory` (generate test data).

---

## Pre-Requisites

- Epic 2 is complete (all data models exist)
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Design Reference

- `detailed/01_data_preprocessing.md` § 2–4, § 6
- `high_level/03_data_flow_pipeline.md` § 2

---

## Task 3.1: Create `src/data/patch_parser.py`

**Class:** `PatchParser`

**Design reference:** `detailed/01_data_preprocessing.md` § 4

**Class-level constants:**
- `_DIFF_HEADER: str = "diff --git"`
- `_HUNK_HEADER_PATTERN: str = r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@"`
- `_FILE_HEADER_PATTERN: str = r"^diff --git a/(.*?) b/(.*)$"`

**Constructor:** `__init__(self) -> None`

**Public methods:**
- `parse(patch_text: str) -> list[FileChange]` — parse full unified diff into FileChange objects

**Private methods:**
- `_extract_filepath(line: str) -> str` — extract file path from `diff --git a/... b/...`
- `_parse_hunk_header(line: str) -> Hunk` — parse `@@ -N,M +N,M @@` into Hunk
- `_accumulate_hunk_lines(hunk: Hunk, line: str) -> None` — add `+`/`-`/` ` lines to hunk
- `_count_patch_lines(file_changes: list[FileChange]) -> int` — total lines changed

**Edge cases to handle:**
- New files (`--- /dev/null`)
- Deleted files (`+++ /dev/null`)
- Binary files (skip)
- Multi-hunk single file
- No newline at end of file markers

---

## Task 3.2: Create `src/data/ingestor.py`

**Class:** `DataIngestor`

**Design reference:** `detailed/01_data_preprocessing.md` § 1, `high_level/03_data_flow_pipeline.md` § 2

**Constructor:** `__init__(self, data_paths: DataPathsConfig) -> None`

**Injected dependencies:**
- `DataPathsConfig` (from config schemas)

**Class-level constants:**
- `_COMPLEXITY_SIMPLE_MAX_LINES: int = 20`
- `_COMPLEXITY_SIMPLE_MAX_FILES: int = 1`
- `_COMPLEXITY_MODERATE_MAX_LINES: int = 100`
- `_COMPLEXITY_MODERATE_MAX_FILES: int = 3`

**Public methods:**
- `load_tasks() -> list[Task]` — read `tasks.jsonl`, parse each line into a `Task`
- `load_graph(instance_id: str) -> object` — load NetworkX `MultiDiGraph` from `graphs/<instance_id>.json`. Return type is `object` (NetworkX is optional import). If the file doesn't exist, return None.
- `load_embeddings(instance_id: str) -> dict[str, object]` — load `.npz` file. Return dict mapping node_id to numpy array. If file doesn't exist, return empty dict.
- `classify_complexity(task: Task) -> ComplexityTier` — classify a task based on its patch

**Private methods:**
- `_parse_task_line(line: str) -> Task` — parse single JSONL line
- `_count_changed_files(patch: str) -> int` — count files in patch
- `_count_changed_lines(patch: str) -> int` — count `+`/`-` lines in patch
- `_load_json_file(path: str) -> dict[str, object]` — generic JSON loader

---

## Task 3.3: Create `src/data/mock_data_factory.py`

**Class:** `MockDataFactory`

**Design reference:** `detailed/01_data_preprocessing.md` § 6

**Class-level constants (from design doc):**
- `_MOCK_SOURCE: str` — the buggy `utils.py` content
- `_MOCK_FIXED: str` — the fixed `utils.py` content
- `_MOCK_INSTANCE_ID: str = "mock_utils_001"`
- `_MOCK_REPO: str = "mock/utils-lib"`
- `_MOCK_COMMIT: str = "0" * 40`

**Constructor:** `__init__(self) -> None`

**Public methods:**
- `create_mock_task() -> Task` — return a single mock task (off-by-one bug fix)
- `create_mock_workspace() -> str` — create a temp directory with `utils.py` + `__init__.py` + minimal git repo. Returns the path as string.
- `create_mock_graph() -> dict[str, object]` — return a minimal graph structure
- `create_mock_embeddings() -> dict[str, object]` — return a minimal embedding dict

**Private methods:**
- `_generate_patch() -> str` — create the unified diff for the fix
- `_generate_test_patch() -> str` — create the test file diff
- `_init_git_repo(workspace_path: str) -> None` — `git init` + `git add .` + `git commit`

---

## Task 3.4: Write Tests

### `tests/unit/data/test_patch_parser.py`
- `test_parse_single_file_single_hunk` — basic diff
- `test_parse_single_file_multi_hunk` — one file, 2 hunks
- `test_parse_multi_file` — 3 files changed
- `test_parse_new_file` — `--- /dev/null`
- `test_parse_deleted_file` — `+++ /dev/null`
- `test_parse_empty_patch_returns_empty` — empty string input
- `test_parse_hunk_header_extracts_line_numbers`
- `test_parse_preserves_old_and_new_lines`

### `tests/unit/data/test_ingestor.py`
- `test_load_tasks_reads_jsonl` — create temp JSONL, load, verify
- `test_load_tasks_empty_file_returns_empty_list`
- `test_classify_complexity_simple` — 1 file, <20 lines
- `test_classify_complexity_moderate` — 2 files, 50 lines
- `test_classify_complexity_complex` — 4 files
- `test_load_graph_missing_file_returns_none`
- `test_load_embeddings_missing_file_returns_empty`

**Use `tmp_path` pytest fixture for temp files. Mock file I/O where appropriate.**

### `tests/unit/data/test_mock_data_factory.py`
- `test_create_mock_task_returns_task` — verify all fields populated
- `test_create_mock_workspace_creates_files` — verify `utils.py` and `__init__.py` exist
- `test_create_mock_workspace_has_git_repo` — verify `.git/` exists
- `test_create_mock_graph_returns_dict`
- `test_create_mock_embeddings_returns_dict`
- `test_mock_patch_is_valid_diff` — parse with PatchParser, verify it works

---

## Task 3.5: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `PatchParser` correctly parses single-file, multi-file, new-file, and deleted-file diffs
- [ ] `DataIngestor` loads `tasks.jsonl` and classifies complexity
- [ ] `MockDataFactory` creates mock task, workspace, graph, and embeddings
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean
- [ ] `mypy` clean

---

## Files Created in This Epic

```
src/data/patch_parser.py
src/data/ingestor.py
src/data/mock_data_factory.py
tests/unit/data/test_patch_parser.py
tests/unit/data/test_ingestor.py
tests/unit/data/test_mock_data_factory.py
```
