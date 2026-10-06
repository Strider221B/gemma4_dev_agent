# Epic 2 — Data Models

> **Runs on:** Local machine
> **Depends on:** Epic 1 (Config & Utils)
> **Estimated effort:** ~1.5 hours
> **Goal:** Implement all core data model classes used throughout the pipeline: `Task`, `FileChange`, `Hunk`, `ToolCall`, `Turn`, `Trajectory`, `ComplexityTier`, and all result/report dataclasses.

---

## Pre-Requisites

- Epic 1 is complete
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Design Reference

- `detailed/01_data_preprocessing.md` § 2 — Core Data Types
- `detailed/04_cv_evaluator.md` § 1 — CV Result types
- `detailed/05_peer_solution_protocol.md` § 3.1 — Peer Solution types

---

## Task 2.1: Create `src/data/complexity_tier.py`

**Class:** `ComplexityTier` (Enum)

```python
"""Complexity tier classification for SWE-bench tasks."""
from enum import Enum

class ComplexityTier(Enum):
    """Task complexity classification based on patch scope."""
    SIMPLE = "SIMPLE"       # 1 file, <20 lines changed
    MODERATE = "MODERATE"   # 1-3 files, 20-100 lines
    COMPLEX = "COMPLEX"     # >3 files or >100 lines
```

---

## Task 2.2: Create `src/data/task.py`

**Class:** `Task` (frozen dataclass)

**Fields (from design doc):**
- `instance_id: str`
- `repo: str`
- `base_commit: str`
- `problem_statement: str`
- `hints_text: str`
- `patch: str`
- `test_patch: str`
- `created_at: str`

---

## Task 2.3: Create `src/data/hunk.py`

**Class:** `Hunk` (frozen dataclass)

**Fields:**
- `old_start: int`
- `old_count: int`
- `new_start: int`
- `new_count: int`
- `old_lines: list[str]`
- `new_lines: list[str]`

---

## Task 2.4: Create `src/data/file_change.py`

**Class:** `FileChange` (frozen dataclass)

**Fields:**
- `filepath: str`
- `hunks: list[Hunk]`

---

## Task 2.5: Create `src/data/tool_call.py`

**Class:** `ToolCall` (frozen dataclass)

**Fields:**
- `tool_name: str`
- `args: dict[str, object]`

---

## Task 2.6: Create `src/data/turn.py`

**Class:** `Turn` (frozen dataclass)

**Fields:**
- `role: str` — `"user"` or `"model"`
- `thought: str | None` — model thinking (within `<|thought|>` tags)
- `text: str | None` — model text response
- `tool_calls: list[ToolCall]` — tool calls (within `<|tool_call|>` tags)
- `tool_result: str | None` — tool result JSON (for user turns)

---

## Task 2.7: Create `src/data/trajectory.py`

**Class:** `Trajectory` (dataclass, NOT frozen — token_count is computed)

**Fields:**
- `instance_id: str`
- `repo: str`
- `complexity: ComplexityTier`
- `turns: list[Turn]`
- `token_count: int`
- `num_tool_calls: int`
- `num_files_changed: int`
- `patch: str | None = None` — extracted agent patch (for evaluation)
- `had_truncation: bool = False`
- `elapsed_seconds: float = 0.0`

---

## Task 2.8: Create CV/Evaluation Result Models

### `src/evaluation/cv_fold.py`
**Class:** `CVFold` (frozen dataclass)
**Fields:**
- `fold_idx: int`
- `train_ids: list[str]`
- `val_ids: list[str]`
- `val_repo: str`
- `complexity_distribution: dict[str, int]`

### `src/evaluation/task_result.py`
**Class:** `TaskResult` (frozen dataclass)
**Fields:**
- `instance_id: str`
- `repo: str`
- `complexity: ComplexityTier | None = None`
- `resolved: bool = False`
- `patch_size: int = 0`
- `tool_calls_used: int = 0`
- `tokens_used: int = 0`
- `had_truncation: bool = False`
- `time_seconds: float = 0.0`
- `error: str | None = None`

### `src/evaluation/cv_result.py`
**Class:** `CVResult` (frozen dataclass)
**Fields:**
- `fold_idx: int`
- `val_repo: str`
- `resolution_rate: float`
- `resolved_count: int`
- `total_count: int`
- `per_task_results: list[TaskResult]`
- `avg_tool_calls: float = 0.0`
- `avg_tokens: float = 0.0`
- `truncation_rate: float = 0.0`
- `per_complexity: dict[str, float] | None = None`

### `src/evaluation/cv_report.py`
**Class:** `CVReport` (frozen dataclass)
**Fields:**
- `version: str`
- `timestamp: str`
- `fold_results: list[CVResult]`
- `aggregate_resolution_rate: float`
- `per_repo_rates: dict[str, float]`
- `per_complexity_rates: dict[str, float]`
- `overfitting_signal: str`
- `recommendations: list[str]`
- `total_resolved: int`
- `total_tasks: int`

### `src/evaluation/verification_result.py`
**Class:** `VerificationResult` (frozen dataclass)
**Fields:**
- `resolved: bool`
- `error: str | None = None`
- `passed_tests: int = 0`
- `failed_tests: int = 0`
- `total_tests: int = 0`

### `src/evaluation/execution_result.py`
**Class:** `ExecutionResult` (frozen dataclass)
**Fields:**
- `instance_id: str`
- `patch: str | None = None`
- `num_tool_calls: int = 0`
- `total_tokens: int = 0`
- `had_truncation: bool = False`
- `elapsed_seconds: float = 0.0`

---

## Task 2.9: Create Peer Analysis Models

### `src/evaluation/peer_solution.py`
**Class:** `PeerSolution` (dataclass)
**Fields (from design doc `05_peer_solution_protocol.md` § 3.1):**
- `notebook_name: str`
- `kaggle_url: str`
- `author: str`
- `lb_score: float`
- `capture_date: str`
- `local_path: str`
- `analysis_status: str` — `"pending" | "inspected" | "tested" | "adopted" | "rejected"`
- `prompt_strategy: str | None = None`
- `agent_architecture: str | None = None`
- `adapter_details: str | None = None`
- `key_techniques: list[str]` — default empty list
- `cv_score: float | None = None`
- `cv_lb_gap: float | None = None`
- `leak_detected: bool | None = None`

### `src/evaluation/inspection_check.py`
**Class:** `InspectionCheck` (frozen dataclass)
**Fields:**
- `name: str`
- `passed: bool`
- `findings: list[str]`
- `severity: str = "info"`

### `src/evaluation/inspection_report.py`
**Class:** `InspectionReport` (frozen dataclass)
**Fields:**
- `solution: PeerSolution`
- `checks: list[InspectionCheck]`
- `architecture: str | None`
- `prompts: list[str]`
- `training: str | None`
- `novel_techniques: list[str]`
- `risk_level: str`

### `src/evaluation/adversarial_result.py`
**Class:** `AdversarialResult` (frozen dataclass)
**Fields:**
- `peer_cv_score: float`
- `baseline_cv_score: float`
- `improvement: float`
- `cv_lb_gap: float`
- `is_significant: bool`
- `per_fold_comparison: dict[str, dict[str, float]]`

### `src/evaluation/decision.py`
**Class:** `Decision` (frozen dataclass)
**Fields:**
- `action: str` — `"ADOPT" | "PARTIAL_ADOPT" | "REJECT"`
- `reason: str`
- `confidence: str`
- `components_to_adopt: list[str] | None = None`

---

## Task 2.10: Create Deployment Models

### `src/deployment/validation_check.py`
**Class:** `ValidationCheck` (frozen dataclass)
**Fields:**
- `name: str`
- `passed: bool`
- `detail: str`

### `src/deployment/validation_report.py`
**Class:** `ValidationReport` (frozen dataclass)
**Fields:**
- `checks: list[ValidationCheck]`
- `all_passed: bool`

---

## Task 2.11: Create Overfitting Signal Enum

### `src/evaluation/overfitting_signal.py`
**Class:** `OverfittingSignal` (Enum)
```python
class OverfittingSignal(Enum):
    NONE = "NONE"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
```

---

## Task 2.12: Write Tests

### `tests/unit/data/test_task.py`
- `test_task_creation_with_all_fields`
- `test_task_is_frozen` — assigning a field raises error
- `test_task_equality`

### `tests/unit/data/test_trajectory.py`
- `test_trajectory_creation`
- `test_trajectory_default_values`
- `test_trajectory_mutable` — can set `token_count`

### `tests/unit/data/test_tool_call.py`
- `test_tool_call_creation`
- `test_tool_call_is_frozen`

### `tests/unit/data/test_turn.py`
- `test_model_turn_with_tool_calls`
- `test_user_turn_with_tool_result`

### `tests/unit/data/test_file_change.py`
- `test_file_change_with_hunks`
- `test_hunk_fields`

### `tests/unit/evaluation/test_cv_result.py`
- `test_cv_result_creation`
- `test_cv_report_creation`
- `test_verification_result_defaults`

---

## Task 2.13: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] All data model files exist in `src/data/`, `src/evaluation/`, `src/deployment/`
- [ ] All dataclasses are frozen where appropriate
- [ ] All fields have type hints
- [ ] All classes have docstrings
- [ ] Tests pass with ≥90% coverage
- [ ] `ruff check` clean
- [ ] `mypy` clean (strict)

---

## Files Created in This Epic

```
src/data/complexity_tier.py
src/data/task.py
src/data/hunk.py
src/data/file_change.py
src/data/tool_call.py
src/data/turn.py
src/data/trajectory.py
src/evaluation/cv_fold.py
src/evaluation/task_result.py
src/evaluation/cv_result.py
src/evaluation/cv_report.py
src/evaluation/verification_result.py
src/evaluation/execution_result.py
src/evaluation/peer_solution.py
src/evaluation/inspection_check.py
src/evaluation/inspection_report.py
src/evaluation/adversarial_result.py
src/evaluation/decision.py
src/evaluation/overfitting_signal.py
src/deployment/validation_check.py
src/deployment/validation_report.py
tests/unit/data/test_task.py
tests/unit/data/test_trajectory.py
tests/unit/data/test_tool_call.py
tests/unit/data/test_turn.py
tests/unit/data/test_file_change.py
tests/unit/evaluation/test_cv_result.py
```
