# Epic 7 — SFT Training Pipeline

> **Runs on:** Server (Kaggle 4×L4 GPUs) — **code is written locally but actual training runs on Kaggle**
> **Depends on:** Epic 6 (Dataset Builder)
> **Estimated effort:** ~3 hours (code writing) + GPU time on Kaggle
> **Goal:** Implement `SFTTrainerPipeline`, `CheckpointManager`, `TelemetryCallback`, and `CurriculumScheduler`.

---

## Pre-Requisites

- Epic 6 is complete
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`
- Training dependencies (Unsloth, TRL, PEFT) only available on Kaggle — local tests must mock these

---

## Design Reference

- `detailed/02_sft_training.md` — Full SFT pipeline design
- `high_level/04_training_strategy.md` § 2 — SFT configuration

---

## IMPORTANT: Local vs Server

All code is written and tested locally with mocks. The classes must:
- Import `unsloth`, `trl`, `peft` lazily (inside methods, not at module top-level)
- All external library calls must be mockable in tests
- Local tests verify class logic, not actual training

---

## Task 7.1: Create `src/training/checkpoint_manager.py`

**Class:** `CheckpointManager`

**Constructor:**
```python
def __init__(self, max_size_bytes: int = 1_500_000_000) -> None:
```

**Class-level constants:**
- `_ADAPTER_CONFIG_FILE: str = "adapter_config.json"`
- `_ADAPTER_WEIGHTS_FILE: str = "adapter_model.safetensors"`
- `_FORBIDDEN_PATTERNS: list[str] = ["*.bin", "*.pt", "*.pth"]`

**Public methods:**
- `save_adapter(model: object, path: str) -> str` — save adapter with safetensors, return path
- `load_adapter(base_model: object, path: str) -> object` — load adapter from path
- `validate_size(path: str) -> bool` — check total size < limit
- `list_checkpoints(directory: str) -> list[str]` — list checkpoint directories

**Private methods:**
- `_verify_safetensors(path: str) -> bool` — confirm `.safetensors` exists, no `.bin`/`.pt`
- `_compute_total_size(path: str) -> int` — recursive file size sum
- `_check_forbidden_files(path: str) -> list[str]` — return list of forbidden files found

---

## Task 7.2: Create `src/training/callback_handler.py`

**Class:** `TelemetryCallback`

**Design reference:** `detailed/02_sft_training.md` § 4.1

**Constructor:**
```python
def __init__(self, logger: TelemetryLogger) -> None:
```

**Public methods (TrainerCallback interface):**
- `on_log(args: object, state: object, control: object, logs: dict[str, object] | None = None, **kwargs: object) -> None`
- `on_evaluate(args: object, state: object, control: object, metrics: dict[str, object] | None = None, **kwargs: object) -> None`
- `on_save(args: object, state: object, control: object, **kwargs: object) -> None`

---

## Task 7.3: Create `src/training/curriculum_scheduler.py`

**Class:** `CurriculumScheduler`

**Design reference:** `detailed/02_sft_training.md` § 3.1

**Constructor:** `__init__(self) -> None`

**Class-level constants:**
- `_EARLY_STAGE_CUTOFF: float = 0.4`
- `_MIDDLE_STAGE_CUTOFF: float = 0.8`

**Public methods:**
- `get_epoch_data(dataset: object, epoch: int, total_epochs: int) -> object` — filter dataset by complexity tier based on curriculum stage

**Private methods:**
- `_compute_progress(epoch: int, total_epochs: int) -> float` — epoch / total_epochs

---

## Task 7.4: Create `src/training/training_config.py`

**Class:** `TrainingConfigBuilder`

**Purpose:** Build TRL `SFTConfig`/`TrainingArguments` from our Pydantic `SFTConfig`.

**Constructor:**
```python
def __init__(self, config: SFTConfig) -> None:
```

**Public methods:**
- `build_training_args() -> dict[str, object]` — return a dict of training arguments suitable for `trl.SFTConfig()`

---

## Task 7.5: Create `src/training/sft_trainer.py`

**Class:** `SFTTrainerPipeline`

**Design reference:** `detailed/02_sft_training.md` § 2

**Constructor:**
```python
def __init__(
    self,
    checkpoint_mgr: CheckpointManager,
    telemetry: TelemetryLogger,
    curriculum: CurriculumScheduler,
) -> None:
```

**Class-level constants:**
- `_ADAPTER_SIZE_LIMIT: int = 1_500_000_000`
- `_MIN_EVAL_LOSS_IMPROVEMENT: float = 0.001`
- `_REQUIRED_SPECIAL_TOKENS: list[str] = ["<start_of_turn>", "<end_of_turn>", "<|tool_call|>", "<|/tool_call|>", "<|thought|>", "<|/thought|>"]`

**Public methods:**
- `run(config: SFTConfig, dataset: object) -> str` — execute complete SFT training pipeline, return adapter path

**Private methods:**
- `_load_model(config: ModelConfig) -> tuple[object, object]` — load model + tokenizer via Unsloth (lazy import)
- `_apply_lora(model: object, config: LoRAConfig) -> object` — apply LoRA adapters
- `_prepare_splits(dataset: object, eval_fold: int) -> tuple[object, object]` — split dataset by fold
- `_apply_curriculum(dataset: object, config: TrainingConfig) -> object` — apply curriculum if enabled
- `_create_trainer(model: object, tokenizer: object, train_ds: object, val_ds: object, config: SFTConfig) -> object` — build TRL SFTTrainer (lazy import)
- `_save_adapter(model: object, path: str) -> str` — delegate to CheckpointManager
- `_verify_special_tokens(tokenizer: object) -> None` — assert all required tokens exist
- `_compute_size(path: str) -> int` — delegate to CheckpointManager

---

## Task 7.6: Write Tests

### `tests/unit/training/test_checkpoint_manager.py`
- `test_validate_size_under_limit_returns_true` — create small temp dir
- `test_validate_size_over_limit_returns_false`
- `test_verify_safetensors_with_valid_files` — create mock `.safetensors` file
- `test_verify_safetensors_with_forbidden_files` — create `.bin` file, should fail
- `test_list_checkpoints_returns_dirs`
- `test_compute_total_size_correct`

### `tests/unit/training/test_callback_handler.py`
- `test_on_log_logs_metrics` — mock TelemetryLogger, verify `log_metrics` called
- `test_on_evaluate_logs_eval_metrics`
- `test_on_save_logs_info`

### `tests/unit/training/test_curriculum_scheduler.py`
- `test_early_stage_returns_simple_only` — epoch 1 of 5
- `test_middle_stage_returns_simple_and_moderate` — epoch 3 of 5
- `test_late_stage_returns_all` — epoch 5 of 5

### `tests/unit/training/test_sft_trainer.py`
- `test_run_calls_load_model` — mock all external calls, verify pipeline sequence
- `test_run_calls_apply_lora`
- `test_run_calls_save_adapter`
- `test_run_logs_start_and_end`
- `test_prepare_splits_filters_by_fold`

**All external imports (`unsloth`, `trl`, `peft`, `torch`) must be mocked in tests.**

---

## Task 7.7: Run CI

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `SFTTrainerPipeline` orchestrates the full training flow (mocked)
- [ ] `CheckpointManager` validates adapter format and size
- [ ] `TelemetryCallback` logs training metrics
- [ ] `CurriculumScheduler` filters by complexity tier per epoch
- [ ] All external library calls are lazy-imported and mockable
- [ ] Tests pass with ≥90% coverage (using mocks)
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/training/checkpoint_manager.py
src/training/callback_handler.py
src/training/curriculum_scheduler.py
src/training/training_config.py
src/training/sft_trainer.py
tests/unit/training/test_checkpoint_manager.py
tests/unit/training/test_callback_handler.py
tests/unit/training/test_curriculum_scheduler.py
tests/unit/training/test_sft_trainer.py
```
