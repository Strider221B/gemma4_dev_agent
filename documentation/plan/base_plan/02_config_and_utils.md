# Epic 1 — Config & Utils

> **Runs on:** Local machine
> **Depends on:** Epic 0 (Project Scaffolding)
> **Estimated effort:** ~2 hours
> **Goal:** Implement the centralised configuration system (`ConfigManager`, Pydantic schemas) and utility classes (`TelemetryLogger`, `GitUtils`, `TokenCounter`).

---

## Pre-Requisites

- Epic 0 is complete (all directories exist, `pyproject.toml` installed)
- Activate environment: `source ~/python_envs/p312_kaggle/bin/activate`

---

## Task 1.1: Create `src/config/schema.py` — Pydantic Config Schemas

**File:** `src/config/schema.py`
**Class:** `ConfigSchema`

This file contains the Pydantic models that define all configuration shapes. Despite the one-class-per-file rule, this is a schema/types file — so define ONE top-level class `ConfigSchema` as a namespace that contains nested model classes via composition.

**However**, to strictly follow one-class-per-file, split into separate files:

### Task 1.1a: `src/config/model_config.py`
**Class:** `ModelConfig`
```python
"""Pydantic schema for model configuration."""
from pydantic import BaseModel
```

**Fields:**
- `name: str` — model identifier (e.g., `"google/gemma-4-31b-it-qat-w4a16-ct"`)
- `load_in_4bit: bool = True`
- `max_seq_length: int = 32768`
- `dtype: str = "bfloat16"`

### Task 1.1b: `src/config/lora_config.py`
**Class:** `LoRAConfig`

**Fields:**
- `r: int = 32`
- `lora_alpha: int = 64`
- `lora_dropout: float = 0.05`
- `target_modules: list[str]` — default: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- `bias: str = "none"`
- `task_type: str = "CAUSAL_LM"`

### Task 1.1c: `src/config/training_config.py`
**Class:** `TrainingConfig`

**Fields (all from design doc `02_sft_training.md` § 5):**
- `num_epochs: int = 5`
- `per_device_train_batch_size: int = 1`
- `gradient_accumulation_steps: int = 8`
- `learning_rate: float = 2e-4`
- `lr_scheduler_type: str = "cosine"`
- `warmup_ratio: float = 0.05`
- `weight_decay: float = 0.01`
- `max_grad_norm: float = 1.0`
- `bf16: bool = True`
- `gradient_checkpointing: bool = True`
- `max_seq_length: int = 28672`
- `packing: bool = True`
- `eval_strategy: str = "steps"`
- `eval_steps: int = 50`
- `save_strategy: str = "steps"`
- `save_steps: int = 50`
- `load_best_model_at_end: bool = True`
- `metric_for_best_model: str = "eval_loss"`
- `save_total_limit: int = 3`
- `curriculum_enabled: bool = True`

### Task 1.1d: `src/config/sft_config.py`
**Class:** `SFTConfig`

**Fields:**
- `model: ModelConfig`
- `lora: LoRAConfig`
- `training: TrainingConfig`
- `eval_fold: int = 0`
- `output_dir: str = "/kaggle/working/checkpoints"`
- `log_path: str = "/kaggle/working/logs/run_sft.log"`

### Task 1.1e: `src/config/grpo_config.py`
**Class:** `GRPOHyperConfig`

**Fields (from design doc `03_rl_training.md` § 5):**
- `num_generations: int = 4`
- `max_new_tokens: int = 16384`
- `temperature: float = 0.7`
- `top_p: float = 0.95`
- `lora_r: int = 32`
- `beta: float = 0.1`
- `num_epochs: int = 2`
- `per_device_train_batch_size: int = 1`
- `gradient_accumulation_steps: int = 4`
- `learning_rate: float = 5e-5`
- `lr_scheduler_type: str = "cosine"`
- `warmup_ratio: float = 0.1`
- `max_grad_norm: float = 0.5`
- `max_trajectory_tokens: int = 28672`
- `discard_truncated: bool = True`

### Task 1.1f: `src/config/dpo_config.py`
**Class:** `DPOHyperConfig`

**Fields:**
- `beta: float = 0.1`
- `loss_type: str = "sigmoid"`
- `max_length: int = 28672`
- `max_prompt_length: int = 4096`
- `num_epochs: int = 2`
- `per_device_train_batch_size: int = 1`
- `gradient_accumulation_steps: int = 4`
- `learning_rate: float = 5e-6`

### Task 1.1g: `src/config/rl_config.py`
**Class:** `RLConfig`

**Fields:**
- `model: ModelConfig`
- `adapter_path: str = "/kaggle/working/checkpoints/sft_lora"`
- `mode: str = "grpo"` (Literal["grpo", "dpo"])
- `grpo: GRPOHyperConfig`
- `dpo: DPOHyperConfig`
- `output_dir: str = "/kaggle/working/checkpoints"`
- `log_path: str = "/kaggle/working/logs/run_rl.log"`

### Task 1.1h: `src/config/eval_config.py`
**Class:** `EvalConfig`

**Fields (from design doc `04_cv_evaluator.md` § 5):**
- `num_folds: int = 4`
- `random_seed: int = 42`
- `max_tool_calls: int = 100`
- `max_time_minutes: float = 60.0`
- `max_turns: int = 500`
- `command_timeout_seconds: int = 300`
- `mode: str = "full"` (Literal["full", "cached", "dry_run"])
- `gap_threshold: float = 0.15`
- `trend_window: int = 5`
- `output_dir: str = "/kaggle/working/cv_results"`

### Task 1.1i: `src/config/data_paths_config.py`
**Class:** `DataPathsConfig`

**Fields:**
- `tasks_path: str`
- `graphs_dir: str`
- `embeddings_dir: str`
- `snapshots_dir: str`

### Task 1.1j: `src/config/deploy_config.py`
**Class:** `DeployConfig`

**Fields:**
- `templates_dir: str`
- `output_dir: str`
- `zip_path: str`
- `adapters: list[dict[str, str]]` — each dict has `name` and `checkpoint_path`
- `min_cv_threshold: float = 0.30`
- `bump_type: str = "minor"`
- `message: str = ""`
- `staging_dir: str = "kaggle_staging"`

---

## Task 1.2: Create `src/config/config_manager.py`

**Class:** `ConfigManager`

**Responsibilities:**
- Load YAML config files and parse into Pydantic schema objects
- Merge environment variable overrides (prefix: `SWEGEMMA_`)
- Provide typed access to sub-configs

**Public methods:**
- `load(config_path: str) -> "ConfigManager"` — class method or static; reads YAML, returns ConfigManager instance
- `get_sft_config() -> SFTConfig`
- `get_rl_config() -> RLConfig`
- `get_eval_config() -> EvalConfig`
- `get_deploy_config() -> DeployConfig`
- `get_data_paths() -> DataPathsConfig`

**Private methods:**
- `_load_yaml(path: str) -> dict[str, object]` — read YAML file
- `_apply_env_overrides(config: dict[str, object]) -> dict[str, object]` — scan `os.environ` for `SWEGEMMA_` prefix

**Constructor:** `__init__(self, raw_config: dict[str, object]) -> None`

**Dependencies:** `pyyaml`, `pydantic`

---

## Task 1.3: Create `src/utils/telemetry_logger.py`

**Class:** `TelemetryLogger`

**Design reference:** `high_level/04_training_strategy.md` § 5, `high_level/08_risk_mitigation.md` § 4.1

**Class-level constants:**
- `_LOG_FORMAT: str = "json"`
- `_DEFAULT_LOG_DIR: str = "logs"`

**Constructor:** `__init__(self, log_dir: str, run_version: str) -> None`

**Public methods:**
- `log_start(phase: str, config: object) -> None` — log training start with config snapshot
- `log_end(phase: str, metrics: dict[str, object]) -> None` — log completion
- `log_metrics(metrics: dict[str, object]) -> None` — append metric entry to log file
- `log_info(message: str) -> None` — log informational message
- `log_reward(instance_id: str, components: dict[str, float], total: float) -> None` — log reward breakdown

**Private methods:**
- `_write_entry(entry: dict[str, object]) -> None` — write JSON line to log file
- `_get_timestamp() -> str` — ISO 8601 UTC timestamp

**Output format:** JSON Lines (one JSON object per line) as specified in training strategy doc.

---

## Task 1.4: Create `src/utils/git_utils.py`

**Class:** `GitUtils`

**Public methods:**
- `parse_unified_diff(diff_text: str) -> list[dict[str, object]]` — parse git diff into structured data
- `apply_patch(workspace_path: str, patch_text: str) -> bool` — apply a unified diff
- `list_changed_files(workspace_path: str) -> list[str]` — `git diff --name-only`
- `checkout_file(workspace_path: str, filepath: str) -> bool` — `git checkout HEAD -- filepath`

**Private methods:**
- `_run_git_command(workspace_path: str, args: list[str]) -> str` — subprocess wrapper

---

## Task 1.5: Create `src/utils/token_counter.py`

**Class:** `TokenCounter`

**Purpose:** Estimate token counts for context budget analysis. On local machine (no GPU), use a simple heuristic or tiktoken-compatible tokenizer.

**Constructor:** `__init__(self, tokenizer_name: str | None = None) -> None`

**Public methods:**
- `count_tokens(text: str) -> int` — count tokens using tokenizer or heuristic
- `estimate_from_chars(char_count: int) -> int` — rough estimate (~4 chars per token)
- `fits_budget(text: str, budget: int) -> bool` — check if text fits within budget

**Private methods:**
- `_load_tokenizer(name: str) -> object` — lazy-load tokenizer (optional import)

---

## Task 1.6: Create `src/utils/constants.py`

**This is a constants file, not a class file — it's the one exception noted in coding standards § 10.**

```python
"""Shared utility constants used across multiple modules."""

# Model identifier
MODEL_NAME: str = "google/gemma-4-31b-it-qat-w4a16-ct"

# Submission constraints
MAX_SUBMISSION_SIZE_BYTES: int = 3_221_225_472  # 3 GiB
MAX_ADAPTERS: int = 8
MAX_LORA_RANK: int = 128
MAX_CONTEXT_WINDOW: int = 32_768

# Budget defaults
DEFAULT_MAX_TOOL_CALLS: int = 100
DEFAULT_MAX_TIME_MINUTES: float = 60.0
DEFAULT_COMMAND_TIMEOUT_SECONDS: int = 300
DEFAULT_MAX_STDOUT_CHARS: int = 5_000
DEFAULT_MAX_FILE_LINES: int = 150
DEFAULT_MAX_FILE_CHARS: int = 10_000

# Token budgets
MAX_TRAJECTORY_TOKENS: int = 28_672
THINKING_BUDGET_TOKENS: int = 4_096
COMPACTION_THRESHOLD_TOKENS: int = 14_336

# Adapter constraints
ADAPTER_SIZE_LIMIT_BYTES: int = 1_500_000_000  # 1.5 GiB safety
ALLOWED_ADAPTER_EXTENSIONS: frozenset[str] = frozenset({".safetensors", ".json"})
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle"})
```

---

## Task 1.7: Create Constants Files for Each Layer

Create these constants files (each with layer-specific constants):

### `src/data/constants.py`
```python
"""Data layer constants."""
MAX_TRAJECTORY_TOKENS: int = 28_672
READ_CONTEXT_LINES: int = 30
MAX_OLD_STRING_LINES: int = 15
MAX_TOOL_CALLS_PER_TRAJECTORY: int = 80
```

### `src/training/constants.py`
```python
"""Training layer constants."""
ADAPTER_SIZE_LIMIT_BYTES: int = 1_500_000_000
MIN_EVAL_LOSS_IMPROVEMENT: float = 0.001
DEFAULT_SEED: int = 42
```

### `src/evaluation/constants.py`
```python
"""Evaluation layer constants."""
MIN_RESOLUTION_RATE: float = 0.10
GENERALIZATION_GAP_THRESHOLD: float = 0.15
MAX_NUDGES: int = 3
```

### `src/deployment/constants.py`
```python
"""Deployment layer constants."""
MAX_TOTAL_SIZE_BYTES: int = 3_221_225_472
REQUIRED_FILES: list[str] = ["agent.yaml"]
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({
    ".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors",
})
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle"})
VERSION_FILE: str = "VERSION"
```

---

## Task 1.8: Write Tests

### `tests/unit/config/test_config_manager.py`
Test:
- `test_load_valid_yaml_returns_config_manager` — load a valid YAML, check it returns a ConfigManager
- `test_load_nonexistent_file_raises_error` — FileNotFoundError
- `test_get_sft_config_returns_sft_config` — verify returned type
- `test_get_rl_config_returns_rl_config`
- `test_env_override_replaces_value` — set `SWEGEMMA_` env var, verify override
- `test_default_values_applied` — load minimal YAML, check defaults

**Create `tests/unit/config/__init__.py`** (empty)

### `tests/unit/utils/test_telemetry_logger.py`
Test:
- `test_log_metrics_writes_json_line` — write metric, read file, parse JSON
- `test_log_info_writes_message` — verify "info" type entry
- `test_log_start_includes_config` — verify config snapshot
- `test_log_end_includes_metrics` — verify completion metrics
- `test_log_reward_includes_components` — verify reward breakdown

### `tests/unit/utils/test_git_utils.py`
Test:
- `test_parse_unified_diff_single_file` — parse simple diff
- `test_parse_unified_diff_multi_file` — parse diff with 2+ files
- `test_parse_empty_diff_returns_empty_list`

### `tests/unit/utils/test_token_counter.py`
Test:
- `test_count_tokens_returns_positive_int`
- `test_estimate_from_chars_approximation`
- `test_fits_budget_within_budget_returns_true`
- `test_fits_budget_exceeds_budget_returns_false`

### `tests/unit/config/test_model_config.py` (and similar for each schema)
Test:
- `test_model_config_defaults`
- `test_model_config_custom_values`
- `test_model_config_validation_error`

---

## Task 1.9: Create Default Config YAML

Create `/home/somesh/git_repos/gemma4_dev_agent/configs/sft_config.yaml` with content exactly matching `detailed/02_sft_training.md` § 5.

Create `/home/somesh/git_repos/gemma4_dev_agent/configs/rl_config.yaml` with content matching `detailed/03_rl_training.md` § 5.

Create `/home/somesh/git_repos/gemma4_dev_agent/configs/eval_config.yaml` with content matching `detailed/04_cv_evaluator.md` § 5.

---

## Task 1.10: Run CI Checks

```bash
source ~/python_envs/p312_kaggle/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

All three must pass with zero errors and ≥90% coverage for the modules created.

---

## Completion Criteria

- [ ] All config schema files exist and are valid Pydantic models
- [ ] `ConfigManager` loads YAML and produces typed configs
- [ ] `TelemetryLogger` writes JSON Lines
- [ ] `GitUtils` parses diffs
- [ ] `TokenCounter` estimates token counts
- [ ] All constants files created
- [ ] All tests pass
- [ ] `ruff check` clean
- [ ] `mypy` clean
- [ ] ≥90% coverage

---

## Files Created in This Epic

```
src/config/model_config.py
src/config/lora_config.py
src/config/training_config.py
src/config/sft_config.py
src/config/grpo_config.py
src/config/dpo_config.py
src/config/rl_config.py
src/config/eval_config.py
src/config/data_paths_config.py
src/config/deploy_config.py
src/config/config_manager.py
src/utils/telemetry_logger.py
src/utils/git_utils.py
src/utils/token_counter.py
src/utils/constants.py
src/data/constants.py
src/training/constants.py
src/evaluation/constants.py
src/deployment/constants.py
configs/sft_config.yaml
configs/rl_config.yaml
configs/eval_config.yaml
tests/unit/config/__init__.py
tests/unit/config/test_config_manager.py
tests/unit/config/test_model_config.py
tests/unit/utils/test_telemetry_logger.py
tests/unit/utils/test_git_utils.py
tests/unit/utils/test_token_counter.py
```
