# Implementation Plan: Local Kaggle Execution Environment

---

## 1. Objective

Enable local end-to-end execution of the SweGemma-Agent training and packaging pipeline within the cloned Kaggle Python 3.12 virtual environment (`~/python_envs/p312_kaggle`), eliminating remote trial-and-error debugging on Kaggle GPU clusters. The plan introduces an environment detector, a lightweight surrogate model factory that exercises the real Hugging Face Transformers, PEFT, and TRL stack with zero mocks, a unified local pipeline runner, and harmonized notebook execution.

---

## 2. Tasks & Implementation Steps

### Phase 1: Create `EnvironmentDetector`

- **File**: `src/utils/environment_detector.py`
- **Class**: `EnvironmentDetector`
- **Implementation Requirements**:
  1. Adhere to **Strict OOP**: Single class in file matching file name (`environment_detector.py` -> `EnvironmentDetector`).
  2. Class-level private constants:
     - `_KAGGLE_DIR: str = "/kaggle"`
     - `_KAGGLE_WORKING_DIR: str = "/kaggle/working"`
     - `_KAGGLE_INPUT_DIR: str = "/kaggle/input"`
     - `_LOCAL_WORKING_DIR: str = "./kaggle_working"`
     - `_LOCAL_INPUT_DIR: str = "./kaggle_staging"`
     - `_ENV_KAGGLE_KERNEL: str = "KAGGLE_KERNEL_RUN_TYPE"`
     - `_ENV_WORKING_DIR_OVERRIDE: str = "SWEGEMMA_WORKING_DIR"`
     - `_SUBMISSION_ZIP_NAME: str = "submission.zip"`
     - `_SUBMISSION_DIR_NAME: str = "submission"`
     - `_LOGS_DIR_NAME: str = "logs"`
     - `_CHECKPOINTS_DIR_NAME: str = "checkpoints"`
  3. Public methods (all ≤ 20 lines):
     - `is_kaggle(self) -> bool`: Checks if `/kaggle/working` exists or `KAGGLE_KERNEL_RUN_TYPE` is in `os.environ`.
     - `get_working_dir(self) -> str`: Returns working directory path with env override support.
     - `get_input_dir(self) -> str`: Returns input dataset directory path.
     - `get_logs_dir(self) -> str`: Returns path to logs directory under working directory.
     - `get_checkpoints_dir(self) -> str`: Returns path to checkpoints directory under working directory.
     - `get_submission_dir(self) -> str`: Returns path to submission directory under working directory.
     - `get_submission_zip_path(self) -> str`: Returns path to `submission.zip`.
  4. Protected/private helper methods:
     - `_resolve_base_working_dir(self) -> str`
     - `_resolve_base_input_dir(self) -> str`

---

### Phase 2: Create `SurrogateModelFactory`

- **File**: `src/training/surrogate_model_factory.py`
- **Class**: `SurrogateModelFactory`
- **Implementation Requirements**:
  1. Adhere to **Strict OOP**: Single class in file matching file name.
  2. Class-level private constants:
     - `_VOCAB_SIZE: int = 1000`
     - `_HIDDEN_SIZE: int = 64`
     - `_INTERMEDIATE_SIZE: int = 128`
     - `_NUM_HIDDEN_LAYERS: int = 2`
     - `_NUM_ATTENTION_HEADS: int = 2`
     - `_MAX_POSITION_EMBEDDINGS: int = 512`
     - `_TARGET_MODULES: tuple[str, ...] = ("q_proj", "v_proj")`
     - `_GEMMA4_SPECIAL_TOKENS: tuple[str, ...] = (...)` (all 14 tokens matching `sft_trainer.py`)
  3. Public methods (all ≤ 20 lines):
     - `create_surrogate_model(self, output_dir: str) -> str`: Generates and saves a minimal causal LM and tokenizer to disk with Gemma 4 special tokens, returning the directory path.
     - `get_target_modules(self) -> list[str]`: Returns supported target projection layer names.
  4. Protected/private helper methods:
     - `_build_model_config(self) -> object`: Builds a small `LlamaConfig` with tiny dimensions.
     - `_save_model(self, config: object, output_dir: str) -> None`: Instantiates and saves `LlamaForCausalLM`.
     - `_save_tokenizer(self, output_dir: str) -> None`: Builds and saves a tokenizer containing the special tokens.

---

### Phase 3: Create `LocalPipelineRunner`

- **File**: `src/training/local_pipeline_runner.py`
- **Class**: `LocalPipelineRunner`
- **Implementation Requirements**:
  1. Adhere to **Strict OOP**: Single class in file matching file name.
  2. Class-level private constants:
     - `_MODE_SMOKE: str = "smoke"`
     - `_MODE_MOCK: str = "mock"`
     - `_MODE_FULL: str = "full"`
     - `_RUN_VERSION: str = "v0.1.0-local"`
     - `_ADAPTER_NAME: str = "coder_lora"`
     - `_DEFAULT_CONFIG_PATH: str = "configs/sft_config.yaml"`
     - `_TEMPLATES_DIR: str = "kaggle_staging/submission_templates"`
  3. Constructor injection:
     - `__init__(self, env_detector: EnvironmentDetector | None = None, work_dir: str | None = None) -> None`
  4. Public methods (all ≤ 20 lines):
     - `run(self, mode: str = "smoke") -> dict[str, object]`: Dispatches execution based on mode and returns summary report.
     - `run_smoke(self) -> dict[str, object]`: Executes real end-to-end dataset build, real SFT training with surrogate model, real checkpoint validation, and submission packaging.
     - `run_mock(self) -> dict[str, object]`: Executes lightweight mock pipeline without ML dependencies.
  5. Protected/private helper methods:
     - `_init_workspace(self) -> None`
     - `_build_smoke_config(self, surrogate_path: str) -> SFTConfig`
     - `_execute_sft(self, sft_config: SFTConfig, dataset: object) -> str`
     - `_package_submission(self, adapter_path: str) -> str`
     - `_build_summary(self, zip_path: str, adapter_path: str, elapsed: float) -> dict[str, object]`
  6. Entry point wrapper:
     - `if __name__ == "__main__":` invoking a runner class method with CLI argument parsing.

---

### Phase 4: Harmonize `notebooks/train_notebook.ipynb`

- Update initial environment configuration cell in `train_notebook.ipynb`:
  ```python
  from src.utils.environment_detector import EnvironmentDetector

  env = EnvironmentDetector()
  working_dir = env.get_working_dir()
  logs_dir = env.get_logs_dir()
  DATASET_DIR = env.get_input_dir() if env.is_kaggle() else "."
  ```
- Ensure paths for `sft_config.yaml`, `logs`, checkpoints, and submission artifacts derive dynamically from `EnvironmentDetector`.

---

### Phase 5: Testing Strategy & Coverage Requirements

- **Unit Tests**:
  - `tests/unit/utils/test_environment_detector.py`: Tests Kaggle detection, local fallback, environment variable overrides, directory resolution.
  - `tests/unit/training/test_surrogate_model_factory.py`: Tests creation of surrogate model configuration, tokenizer special token population, directory persistence.
  - `tests/unit/training/test_local_pipeline_runner.py`: Tests mode dispatching, mock execution, error handling, summary generation.
- **Integration Tests**:
  - `tests/integration/test_local_pipeline_integration.py`: End-to-end smoke run verifying real `SFTTrainerPipeline` execution, LoRA adapter attachment, safetensors serialization, and `submission.zip` creation without mocks.
- **Coverage Requirement**: Maintain 90%+ line and branch coverage across all newly created and modified modules.

---

### Phase 6: Local CI Verification

Run all mandatory CI gates inside `~/python_envs/p312_kaggle`:
1. **Linting**:
   ```bash
   ruff check ./src ./tests
   ```
   *Requirement*: 0 errors.
2. **Type Checking**:
   ```bash
   mypy ./src
   ```
   *Requirement*: 0 errors in strict mode.
3. **Tests with Coverage**:
   ```bash
   pytest -o addopts=""
   ```
   *Requirement*: 100% test pass rate with coverage ≥ 90%.

---

## 3. Verification Checklist

| Rule | Requirement | Validation Method |
|---|---|---|
| SOLID | Single responsibility per class, dependency injection | Code review |
| Strict OOP | Single class per file, no freestanding functions | Static inspection |
| Member Ordering | Constants → `__init__` → Public → Protected → Private | Static inspection |
| Method Length | Body ≤ 20 logical lines (hard limit) | Line count audit |
| File Length | Total lines ≤ 500 lines | `wc -l` audit |
| No Magic Values | Constants defined with `_` prefix | Code review |
| CI Verification | Ruff + Mypy + Pytest all pass cleanly | Command execution |
