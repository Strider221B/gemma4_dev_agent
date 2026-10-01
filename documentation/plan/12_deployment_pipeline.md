# Epic 11 — Deployment Pipeline

> **Runs on:** Local machine
> **Depends on:** Epics 7 (SFT Training), 8 (RL Training)
> **Estimated effort:** ~2.5 hours
> **Goal:** Implement `SubmissionPackager` (assembles `submission.zip` on Kaggle), `ConstraintValidator` (pre-flight validation), `VersionManager` (semantic versioning + Kaggle push), and `PromptCompiler`.

---

## Pre-Requisites

- Epics 7 and 8 are complete
- Activate environment: `source ~/python_envs/p313_llm/bin/activate`

---

## Design Reference

- `high_level/07_deployment_strategy.md` — Full deployment pipeline
- `high_level/02_architecture_diagram.md` § 3.4 — Deployment layer responsibilities

---

## Task 11.1: Create `src/deployment/constraint_validator.py`

**Class:** `ConstraintValidator`

**Design reference:** `high_level/07_deployment_strategy.md` § 2.2

**Constructor:** `__init__(self) -> None`

**Class-level constants:**
- `_MAX_TOTAL_SIZE: int = 3_221_225_472` (3 GiB)
- `_REQUIRED_FILES: list[str] = ["agent.yaml"]`
- `_ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"})`
- `_FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset({".bin", ".pt", ".pth", ".pkl", ".pickle"})`
- `_MAX_ADAPTERS: int = 8`
- `_MAX_LORA_RANK: int = 128`
- `_EXPECTED_MODEL_NAME: str = "gemma-4-31b-it-qat-w4a16-ct"`

**Public methods:**
- `validate(staging_dir: str) -> ValidationReport` — run all pre-flight checks

**Private methods (one per check, ≤ 20 lines each):**
- `_check_root_config(staging_dir: str) -> ValidationCheck` — `agent.yaml` exists
- `_check_total_size(staging_dir: str) -> ValidationCheck` — total < 3 GiB
- `_check_extensions(staging_dir: str) -> ValidationCheck` — no forbidden extensions
- `_check_adapters(staging_dir: str) -> ValidationCheck` — safetensors + adapter_config.json
- `_check_yaml_schema(staging_dir: str) -> ValidationCheck` — all YAML parseable
- `_check_single_model(staging_dir: str) -> ValidationCheck` — one model declared
- `_check_no_symlinks(staging_dir: str) -> ValidationCheck` — no symlinks
- `_check_adapter_count(staging_dir: str) -> ValidationCheck` — ≤ 8 adapters
- `_check_lora_rank(staging_dir: str) -> ValidationCheck` — rank ≤ 128
- `_check_token_limits(staging_dir: str) -> ValidationCheck` — reasonable token limits
- `_compute_total_size(staging_dir: str) -> int` — recursive size
- `_find_adapters(staging_dir: str) -> list[str]` — find adapter directories
- `_parse_adapter_config(path: str) -> dict[str, object]` — read adapter_config.json

---

## Task 11.2: Create `src/deployment/submission_packager.py`

**Class:** `SubmissionPackager`

**Design reference:** `high_level/07_deployment_strategy.md` § 2.1

**Constructor:**
```python
def __init__(self, validator: ConstraintValidator) -> None:
```

**Public methods:**
- `package(config: DeployConfig) -> str` — assemble submission directory and zip it, return zip path

**Private methods:**
- `_clean_output(output_dir: str) -> None` — remove existing output
- `_copy_templates(templates_dir: str, output_dir: str) -> None` — copy agent.yaml, prompts/, etc.
- `_copy_adapter(checkpoint_path: str, target_path: str) -> None` — copy adapter files
- `_validate(output_dir: str) -> None` — run ConstraintValidator
- `_create_zip(output_dir: str, zip_path: str) -> str` — zip the output directory

---

## Task 11.3: Create `src/deployment/version_manager.py`

**Class:** `VersionManager`

**Design reference:** `high_level/07_deployment_strategy.md` § 3

**Constructor:**
```python
def __init__(self, staging_dir: str) -> None:
```

**Class-level constants:**
- `_VERSION_FILE: str = "VERSION"`
- `_CHANGELOG_FILE: str = "CHANGELOG.md"`
- `_VALID_BUMP_TYPES: frozenset[str] = frozenset({"major", "minor", "patch"})`

**Public methods:**
- `bump(bump_type: str, message: str) -> str` — increment version, return new version string
- `push(staging_dir: str, version: str, message: str) -> None` — push dataset via Kaggle CLI
- `get_current_version() -> str` — read current version

**Private methods:**
- `_read_version() -> str` — read VERSION file
- `_write_version(version: str) -> None` — write VERSION file
- `_update_changelog(version: str, message: str) -> None` — append to CHANGELOG.md
- `_parse_version(version: str) -> tuple[int, int, int]` — parse "X.Y.Z"
- `_log_deployment(version: str, message: str) -> None`

---

## Task 11.4: Create `src/deployment/prompt_compiler.py`

**Class:** `PromptCompiler`

**Purpose:** Resolve `!include` directives in agent YAML and compile prompt template files.

**Constructor:**
```python
def __init__(self, templates_dir: str) -> None:
```

**Public methods:**
- `compile_prompt(template_path: str, variables: dict[str, str]) -> str` — read template, substitute variables
- `resolve_includes(yaml_content: str) -> str` — replace `!include path` with file contents

**Private methods:**
- `_read_template(path: str) -> str` — read template file
- `_substitute_variables(template: str, variables: dict[str, str]) -> str` — `{variable}` replacement

---

## Task 11.5: Write Tests

### `tests/unit/deployment/test_constraint_validator.py`
- `test_validate_valid_staging_dir_passes` — create valid mock staging dir
- `test_validate_missing_agent_yaml_fails`
- `test_validate_oversized_submission_fails`
- `test_validate_forbidden_extension_fails` — add `.bin` file
- `test_validate_symlink_detected`
- `test_validate_too_many_adapters_fails` — 9 adapter dirs
- `test_validate_lora_rank_exceeded_fails`
- `test_validate_single_model_passes`
- `test_validate_multiple_models_fails`

### `tests/unit/deployment/test_submission_packager.py`
- `test_package_creates_zip` — mock templates + adapter, verify zip created
- `test_package_copies_templates` — verify agent.yaml in output
- `test_package_copies_adapters` — verify adapter files in output
- `test_package_validates_before_zip` — verify validator called
- `test_package_cleans_output_first`

### `tests/unit/deployment/test_version_manager.py`
- `test_bump_patch_increments` — "0.1.0" → "0.1.1"
- `test_bump_minor_increments` — "0.1.0" → "0.2.0"
- `test_bump_major_increments` — "0.1.0" → "1.0.0"
- `test_bump_invalid_type_raises_error`
- `test_get_current_version_reads_file`
- `test_push_calls_kaggle_cli` — mock subprocess, verify command

### `tests/unit/deployment/test_prompt_compiler.py`
- `test_compile_prompt_substitutes_variables`
- `test_resolve_includes_replaces_directives`

### Integration test:

### `tests/integration/test_packaging_roundtrip.py`
- `test_full_packaging_roundtrip` — create mock templates + mock adapter → package → unzip → validate

---

## Task 11.6: Create `kaggle_staging/VERSION`

```
0.1.0
```

---

## Task 11.7: Run CI

```bash
source ~/python_envs/p313_llm/bin/activate
cd /home/somesh/git_repos/gemma4_dev_agent

ruff check src/ tests/
mypy src/
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Completion Criteria

- [ ] `ConstraintValidator` checks all 10 constraints
- [ ] `SubmissionPackager` assembles and zips submission
- [ ] `VersionManager` bumps versions and calls Kaggle CLI
- [ ] `PromptCompiler` resolves includes and substitutes variables
- [ ] Integration test passes end-to-end
- [ ] All tests pass with ≥90% coverage
- [ ] `ruff check` clean, `mypy` clean

---

## Files Created in This Epic

```
src/deployment/constraint_validator.py
src/deployment/submission_packager.py
src/deployment/version_manager.py
src/deployment/prompt_compiler.py
kaggle_staging/VERSION
tests/unit/deployment/test_constraint_validator.py
tests/unit/deployment/test_submission_packager.py
tests/unit/deployment/test_version_manager.py
tests/unit/deployment/test_prompt_compiler.py
tests/integration/test_packaging_roundtrip.py
```
